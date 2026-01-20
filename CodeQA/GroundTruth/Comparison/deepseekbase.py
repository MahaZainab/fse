#!/usr/bin/env python3
"""
coreQA_CodeQA_hf_same_main.py

This is your original "Teacher Judge" pipeline logic from coreQA_CodeQA.ipynb,
kept the same structure (async batching, save_every, CSV export, score plots),
but with the ONLY functional change being:
  - Replace Ollama (ChatOllama) with Hugging Face Transformers inference
    using: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

How to run (cluster example):
  srun --partition=general --gres=gpu:1 --cpus-per-task=32 --pty bash -l
  pip install -U torch transformers accelerate tqdm pandas matplotlib

  python coreQA_CodeQA_hf_same_main.py

Notes:
- By default it reads: stratified_balanced_500_combined.json
  (change INPUT_JSON below if your filename differs)
- Outputs:
  deepseek_coder_teacher_Judge.json
  deepseek_coder_teacher_Judge.csv
  score_plot.png
"""

import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# KEEP: same concurrency pattern
# -----------------------------
executor = ThreadPoolExecutor(max_workers=5)  # tune concurrency


# -----------------------------
# NEW: HF-backed "llm.invoke(messages)" drop-in replacement
# (so the rest of your code stays the same)
# -----------------------------
# -----------------------------
# MODEL UNDER TEST
# -----------------------------
_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
_tokenizer = None
_model = None

def _get_hf_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)

        # Ensure pad token is defined to avoid warnings during open-ended generation
        if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _tokenizer.pad_token = _tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # Disable KV-cache to avoid DynamicCache compatibility issues on some clusters/
        # transformers versions (e.g., 'DynamicCache' has no attribute 'seen_tokens').
        try:
            _model.config.use_cache = False
        except Exception:
            pass
        try:
            _model.generation_config.use_cache = False
        except Exception:
            pass

        _model.eval()
    return _tokenizer, _model

class _HFResponse:
    def __init__(self, content: str):
        self.content = content

class HFChat:
    """
    Minimal adapter that mimics the bits you used from ChatOllama:
      response = llm.invoke(messages)
      return response.content
    """
    def __init__(self, temperature: float = 0.0, max_new_tokens: int = 256):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def invoke(self, messages):
        # messages are LangChain-like: objects with .content (SystemMessage/HumanMessage)
        # We'll convert to [{role, content}, ...] and use the model's chat template.
        tokenizer, model = _get_hf_model()

        chat = []
        # Preserve ordering: system then user
        for m in messages:
            # If it looks like a system message, use role=system; otherwise user
            role = "system" if m.__class__.__name__ == "SystemMessage" else "user"
            chat.append({"role": role, "content": m.content})

        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature and self.temperature > 0),
        )
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(self.temperature)

        # Force-disable cache at generation time as well.
        gen_kwargs["use_cache"] = False
        # Explicit pad/eos IDs to keep generation stable
        if tokenizer.eos_token_id is not None:
            gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens (avoid fragile string slicing).
        # NOTE: `decoded` with skip_special_tokens=True may not match the raw prompt string length,
        # so we slice by token count instead of character count.
        gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
        generated = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return _HFResponse(generated)


# -----------------------------
# KEEP: your original helpers
# -----------------------------
def save_json_data_append(path, new_data):
    """
    Load existing data from path, append new_data (list of dict),
    then save combined data back to path.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Combine and deduplicate by "id" if present, else just append
    existing_ids = {item.get("id") for item in existing_data if "id" in item}
    filtered_new = [item for item in new_data if item.get("id") not in existing_ids]
    combined = existing_data + filtered_new

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)


async def call_llm_LangChain_correct_v2_async(prompt, temperature=0.0):
    # ONLY CHANGE vs your notebook:
    # llm = ChatOllama(model="llama3.1:8b", temperature=temperature)
    # -> HFChat(...) using DeepSeek-Coder-V2-Lite-Instruct
    llm = HFChat(temperature=temperature, max_new_tokens=256)

    system_prompt = """
You are an expert system to assess the quality of code comprehension by an LLM. 
You will receive:
- A Python code snippet
- A programming question about that code
- A reference (correct) answer
- A model-generated answer (prediction)

Your task is to evaluate the prediction against the reference answer using four metrics:
accuracy, completeness, relevance, and clarity. For each, provide:
- An integer score from 1 to 3

### Accuracy
Compare the prediction with the reference to assess factual ...correctness and understanding of the code’s behavior and intent.
You must judge whether the prediction reflects accurate behavior and matches core facts from the reference. 
You need to consider semantic meaning of code comprehension:...anding the structure, functionality, and intent behind the code.

Score meanings:
- 1: Completely incorrect or irrelevant; does not address the reference answer.
- 2: Partially correct; some key facts are accurate, but major details are wrong or missing.
- 3: Fully correct; matches the reference answer in meaning and factual content.

### Completeness
Check if the prediction covers all important parts of the reference answer, including key concepts or conditions.

Score meanings:
- 1: Omits most key information or contains only a tiny fragment of relevant content.
- 2: Covers some elements but misses important parts.
- 3: Fully covers all essential information from the reference.

### Relevance
Assess whether the prediction directly addresses the question and stays on-topic.

Score meanings:
- 1: Completely irrelevant or mostly unrelated.
- 2: Partially related but misses the main point.
- 3: Fully focused and directly answers the question.

### Clarity
Evaluate how clearly and logically the prediction is expressed, ensuring it is easy to understand.

Score meanings:
- 1: Confusing, vague, or incoherent.
- 2: Understandable but awkwardly phrased or slightly unclear.
- 3: Clear, concise, and easy to follow.

Example:

{
  "Code": "def aggregate metadata get by host context host k...ne return IMPL aggregate metadata get by host context host key",
  "Question": "What does the code get ?",
  "Reference Answer": "metadata for all aggregates that host belongs to",
  "Model Answer": "Hosts",
  "Evaluation": {
    "accuracy": { "score": 1 },
    "completeness": { "score": 1 },
    "relevance": { "score": 2 },
    "clarity": { "score": 3 }
  }
}

Final Instructions:
Base your evaluation strictly on the content provided. Do not hallucinate missing information. Be consistent and objective.
Provide evaluation only in integer score only for each dimension.
Respond only with a JSON object in this exact format:
{
"accuracy": { "score": 1-3 },
"completeness": { "score": 1-3 },
"relevance": { "score": 1-3 },
"clarity": { "score": 1-3 }
}
"""

    # Keep your LangChain message pattern, but use tiny stubs below
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(executor, lambda: llm.invoke(messages))
    return response.content


# Simple stubs to keep your existing message objects usage without LangChain install.
class SystemMessage:
    def __init__(self, content: str):
        self.content = content

class HumanMessage:
    def __init__(self, content: str):
        self.content = content


def make_llm_eval_prompt(code, question, reference, prediction):
    return f"""
Code:
{code}

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction}
"""


def extract_all_scores(response_text):
    """
    Parse JSON like:
    {
      "accuracy": {"score": 1-3},
      "completeness": {"score": 1-3},
      "relevance": {"score": 1-3},
      "clarity": {"score": 1-3}
    }
    """
    # If model prints extra text, try to grab the first JSON object
    response_text = response_text.strip()
    if not response_text.startswith("{"):
        m = re.search(r"\{.*\}", response_text, re.DOTALL)
        if m:
            response_text = m.group(0)

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return {}

    results = {}
    for metric, details in parsed.items():
        if isinstance(details, dict):
            score = details.get("score")
            # Accept integer-like scores even if the model returns them as strings (e.g., "3") or floats (e.g., 2.0).
            if isinstance(score, str) and score.strip().isdigit():
                score = int(score.strip())
            elif isinstance(score, float) and score.is_integer():
                score = int(score)

            if isinstance(score, int):
                results[metric] = {"score": score}
    return results


async def evaluate_batch_async(data, save_every=25, output_path="CodeLlama_llm_as_judge.json"):
    # `results` are kept in the *wide* format you asked for:
    # code, question, answer, prediction, accuracy, completeness, relevance, clarity
    results = []
    # `flat_records` are still kept for plotting (one row per metric).
    flat_records = []

    for i, item in enumerate(tqdm_asyncio(data, desc="Evaluating")):
        code = item["code"]
        question = item["question"]
        answer = item["answer"]
        prediction = item["prediction"]
        q_id = item.get("id", f"q{i+1}")

        # IMPORTANT: do not change prompts (kept exactly as-is)
        prompt = make_llm_eval_prompt(code, question, answer, prediction)
        try:
            response = await call_llm_LangChain_correct_v2_async(prompt)
            metric_results = extract_all_scores(response)
        except Exception as e:
            print(f"Error evaluating entry {i+1}: {e}")
            metric_results = {}

        # Wide row (matches your CSV screenshot format)
        sample_result = {
            "id": q_id,
            "code": code,
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "accuracy": None,
            "completeness": None,
            "relevance": None,
            "clarity": None,
        }

        for metric in ["accuracy", "completeness", "relevance", "clarity"]:
            score = metric_results.get(metric, {}).get("score", None)
            sample_result[metric] = score
            flat_records.append({
                "id": q_id,
                "code": code,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "metric": metric,
                "score": score
            })

        results.append(sample_result)

        # Periodic checkpoint
        if (i + 1) % save_every == 0:
            save_json_data_append(output_path, results)
            results = []  # clear buffer

    return results, flat_records


def export_csv(flat_records, path="llm_judge_scores.csv"):
    """Export CSV in the wide format:
    code, question, answer, prediction, accuracy, completeness, relevance, clarity
    """
    df = pd.DataFrame(flat_records)
    # If the caller accidentally passes long-form records, pivot them.
    if "metric" in df.columns and "score" in df.columns:
        wide = (
            df.pivot_table(
                index=["id", "code", "question", "answer", "prediction"],
                columns="metric",
                values="score",
                aggfunc="first",
            )
            .reset_index()
        )
        # Ensure column order
        for col in ["accuracy", "completeness", "relevance", "clarity"]:
            if col not in wide.columns:
                wide[col] = None
        wide = wide[[
            "code", "question", "answer", "prediction",
            "accuracy", "completeness", "relevance", "clarity",
        ]]
        wide.to_csv(path, index=False)
    else:
        # Already wide
        cols = [
            "code", "question", "answer", "prediction",
            "accuracy", "completeness", "relevance", "clarity",
        ]
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(path, index=False)
    print(f"Saved CSV to {path}")


def visualize_scores(flat_records):
    df = pd.DataFrame(flat_records)
    df = df.dropna(subset=["score"])
    if df.empty:
        print("No scores to visualize.")
        return

    avg_df = df.groupby("metric")["score"].mean().reset_index()

    plt.figure(figsize=(6, 4))
    plt.bar(avg_df["metric"], avg_df["score"])
    plt.ylim(0, 3)
    plt.title("Average Scores by Metric")
    plt.ylabel("Average Score (1-3)")
    plt.tight_layout()
    plt.savefig("score_plot.png", dpi=150)
    print("Saved plot to score_plot.png")


# -----------------------------
# MAIN (kept the same pattern)
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # INPUT / OUTPUT FILES (hard-coded)
    # -----------------------------
    INPUT_JSON = "model.json"

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    output_json_path = "qwen3_coder.json"
    output_csv_path = "qwen3_coder.csv"

    results, flat_records = asyncio.run(
        evaluate_batch_async(dataset, save_every=25, output_path=output_json_path)
    )

    # Save any leftover results after loop
    if results:
        save_json_data_append(output_json_path, results)

    # Export CSV in the wide format you requested.
    export_csv(flat_records, path=output_csv_path)
    visualize_scores(flat_records)
