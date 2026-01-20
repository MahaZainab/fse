#!/usr/bin/env python3
"""
Single teacher intervention -> student rescore once
Cluster-ready + robust JSON parsing + better logging.

INPUT_JSON expects list[dict] with keys: code, question, answer, prediction (id optional)
"""

import json
import random
import re
import os
from typing import Any, Dict, Optional, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =============================
# Configuration
# =============================
INPUT_JSON = "mini.json"
OUTPUT_JSON = "cl_cot_sixty.json"
OUTPUT_CSV = "cl_cot_sixty.csv"

STUDENT_JUDGE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

TEACHER_INTERVENTION_RATE = 0.60
RANDOM_SEED = 42

# HuggingFace cache directory (cluster-friendly)
HF_CACHE_DIR = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", "/tmp/huggingface_cache"))

# Prompt truncation length (avoid runaway prompt sizes)
MAX_PROMPT_TOKENS = 4096


# -----------------------------
# Model wrapper (HF)
# -----------------------------
class HFChat:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        cache_dir: str = HF_CACHE_DIR,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        print(f"\nLoading model: {model_name}")
        print(f"HF cache: {cache_dir}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()

        # Ensure pad token exists (important for generation w/ padding)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Some sharded models may not have a single device; keep for logging
        try:
            print(f"Model primary device: {self.model.device}")
        except Exception:
            print("Model device: (sharded / device_map=auto)")

    def invoke(self, chat_messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )

        # Move to model device if applicable (for sharded models, HF handles it)
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass

        gen_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": self.temperature > 0,
            # HF expects temperature to exist even if do_sample=False on some setups
            "temperature": float(self.temperature) if self.temperature > 0 else 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# -----------------------------
# Anchored rubric
# -----------------------------
ANCHORED_RUBRIC = r"""
Scoring uses an ANCHORED 3-point rubric:

GLOBAL ANCHORS:
- Score 3: Essentially correct and expert-level for this task.
  * Matches the reference answer in meaning (semantic equivalence).
  * Correctly reflects the code’s behavior and intent.
  * Contains no major errors or misleading statements.

- Score 2: Partially correct but flawed.
  * Some core facts are correct.
  * However, important details/conditions are missing, incorrect, ambiguous, or misweighted.
  * A student relying on it could be misled or left with an incomplete understanding.

- Score 1: Fails to answer the question.
  * Mostly incorrect, irrelevant, or contradicts the code/reference.
  * Does not demonstrate real understanding of the problem.

DIMENSIONS:
Accuracy: correctness vs reference/code behavior.
Completeness: coverage of all essential points from the reference.
Relevance: directly answers the question, stays on-topic.
Clarity: easy to follow, unambiguous, well-structured.
""".strip()


# -----------------------------
# Student prompt (judge): scores only (strict JSON)
# -----------------------------
JUDGE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing the code comprehension capabilities of an LLM.

You will receive:
- A Python code snippet
- A student question about that code
- A reference (correct) answer
- A LLM-generated answer (called the prediction)

Evaluate the LLM prediction on four dimensions (Accuracy, Completeness, Relevance, Clarity).

{ANCHORED_RUBRIC}

Respond ONLY with valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations or extra keys.
""".strip()


def build_judge_user_prompt(code: str, question: str, reference: str, prediction: str) -> str:
    return f"""
Code:
```python
{code}
```

Question:
{question}

Reference Answer:
{reference}

Prediction:
{prediction}
""".strip()


# -----------------------------
# Teacher prompt: intervene once, then student will rescore once.
# FIX: Removed the contradictory "do not share your own score directly" requirement.
# -----------------------------
TEACHER_SYSTEM_PROMPT = f"""
You are a 30B teacher LLM reviewing a student LLM-as-judge's evaluation of an LLM's answer.

You will receive:
- code, question, reference answer, LLM prediction
- student judge scores (1-3) for accuracy/completeness/relevance/clarity

Use the SAME anchored rubric:

{ANCHORED_RUBRIC}

INTERNAL INSTRUCTION:
- Solve the evaluation using step-by-step reasoning internally (private).
- Compare TA prediction vs reference and code behavior; identify key mismatches, missing conditions, and misleading claims.

OUTPUT RULES:
- Do NOT output any private chain-of-thought.
- Output only concise, high-signal justifications: short "evidence" bullets per dimension (1–3 bullets each).

Return ONLY valid JSON in EXACTLY this format:

{{
  "teacher_scores": {{
    "accuracy": 1-3,
    "completeness": 1-3,
    "relevance": 1-3,
    "clarity": 1-3
  }},
  "comparison": {{
    "accuracy": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "completeness": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "relevance": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "clarity": {{"student": 1-3, "teacher": 1-3, "match": true/false}}
  }},
  "evidence": {{
    "accuracy": ["bullet", "bullet"],
    "completeness": ["bullet", "bullet"],
    "relevance": ["bullet", "bullet"],
    "clarity": ["bullet", "bullet"]
  }},
  "guidance": {{
    "accuracy": "2-4 sentences if mismatch else empty string",
    "completeness": "2-4 sentences if mismatch else empty string",
    "relevance": "2-4 sentences if mismatch else empty string",
    "clarity": "2-4 sentences if mismatch else empty string"
  }},
  "checklist_next_time": ["short actionable item", "short actionable item"]
}}

Strict JSON only. No markdown.
""".strip()


def build_teacher_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    student_scores: Dict[str, int],
) -> str:
    return f"""
Code:
```python
{code}
```

Question:
{question}

Reference Answer:
{reference}

LLM Prediction:
{prediction}

Student Judge Scores:
{json.dumps(student_scores, indent=2)}
""".strip()


# -----------------------------
# Student rescore prompt
# -----------------------------
RESCORE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing an LLM answer.

You will receive:
- code, question, reference answer, LLM prediction
- your previous scores
- teacher guidance JSON

IMPORTANT:
- Do not use teacher scores directly.
- Re-evaluate the prediction vs reference/code using the rubric.

{ANCHORED_RUBRIC}

Return ONLY valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations or extra keys.
""".strip()


def build_rescore_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    prev_scores: Dict[str, int],
    teacher_json: Dict[str, Any],
) -> str:
    return f"""
Code:
```python
{code}
```

Question:
{question}

Reference Answer:
{reference}

Prediction:
{prediction}

Your Previous Scores:
{json.dumps(prev_scores, indent=2)}

Teacher Guidance JSON:
{json.dumps(teacher_json, indent=2, ensure_ascii=False)}

Now update your scores and return ONLY the strict score JSON.
""".strip()


# -----------------------------
# Parsing helpers (FIXED)
# -----------------------------
# FIX: non-greedy to avoid swallowing multiple JSON objects
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*?\}", re.M)


def _extract_first_json_obj(text: str) -> Optional[str]:
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    js_text = _extract_first_json_obj(text)
    if not js_text:
        return None
    try:
        obj = json.loads(js_text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_score_json(raw_text: str) -> Optional[Dict[str, int]]:
    obj = _safe_json(raw_text)
    if not obj:
        return None

    out: Dict[str, int] = {}
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = obj.get(k, {})
        score = v.get("score") if isinstance(v, dict) else None
        if isinstance(score, str) and score.isdigit():
            score = int(score)
        if isinstance(score, float):
            score = int(round(score))
        if not isinstance(score, int) or score < 1 or score > 3:
            return None
        out[k] = score
    return out


def parse_teacher_json(raw_text: str) -> Optional[Dict[str, Any]]:
    obj = _safe_json(raw_text)
    if not obj:
        return None
    required = {"teacher_scores", "comparison", "evidence", "guidance", "checklist_next_time"}
    if not required.issubset(obj.keys()):
        return None

    ts = obj.get("teacher_scores", {})
    if not isinstance(ts, dict):
        return None
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = ts.get(k)
        if not isinstance(v, int) or v < 1 or v > 3:
            return None
    return obj


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("=" * 60)
    print("Single teacher intervention -> student rescore once")
    print("=" * 60)
    print(f"Input:  {INPUT_JSON}")
    print(f"Output: {OUTPUT_JSON} / {OUTPUT_CSV}")
    print(f"HF cache dir: {HF_CACHE_DIR}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print("=" * 60)

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    student = HFChat(STUDENT_JUDGE_MODEL, temperature=0.0, max_new_tokens=256, cache_dir=HF_CACHE_DIR)
    # FIX: raise tokens to reduce truncation risk for large JSON
    teacher = HFChat(TEACHER_MODEL, temperature=0.0, max_new_tokens=1400, cache_dir=HF_CACHE_DIR)

    n = len(data)
    k = int(round(TEACHER_INTERVENTION_RATE * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()

    print(f"\nTeacher will intervene on {len(intervene_ids)}/{n} examples ({TEACHER_INTERVENTION_RATE*100:.0f}%).")

    records = []

    for idx, ex in enumerate(tqdm(data, desc="Processing")):
        code = ex.get("code", "")
        question = ex.get("question", "")
        reference = ex.get("answer", "")
        prediction = ex.get("prediction", "")

        # Student initial scoring
        raw_student = student.invoke([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(code, question, reference, prediction)},
        ])
        student_scores = parse_score_json(raw_student)

        # LOG if parsing failed (previously silent)
        if student_scores is None:
            print(f"\n[WARN] Student JSON parse failed at idx={idx}. Raw output:\n{raw_student}\n")

        teacher_intervened = idx in intervene_ids
        raw_teacher = None
        teacher_out = None
        raw_rescore = None
        rescored_scores = None

        if teacher_intervened and student_scores is not None:
            raw_teacher = teacher.invoke([
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_user_prompt(code, question, reference, prediction, student_scores)},
            ])
            teacher_out = parse_teacher_json(raw_teacher)

            if teacher_out is None:
                print(f"\n[WARN] Teacher JSON parse failed at idx={idx}. Raw output:\n{raw_teacher}\n")

            if teacher_out is not None:
                raw_rescore = student.invoke([
                    {"role": "system", "content": RESCORE_SYSTEM_PROMPT},
                    {"role": "user", "content": build_rescore_user_prompt(code, question, reference, prediction, student_scores, teacher_out)},
                ])
                rescored_scores = parse_score_json(raw_rescore)

                if rescored_scores is None:
                    print(f"\n[WARN] Rescore JSON parse failed at idx={idx}. Raw output:\n{raw_rescore}\n")

        final_scores = rescored_scores if rescored_scores is not None else (student_scores or {})

        record = {
            "id": ex.get("id", idx),
            "code": code,
            "question": question,
            "answer": reference,
            "prediction": prediction,

            "raw_student_judge_output": raw_student,
            "initial_student_scores": student_scores,

            "teacher_intervened": teacher_intervened,
            "raw_teacher_output": raw_teacher,
            "teacher_guidance_json": teacher_out,

            "raw_student_rescore_output": raw_rescore,
            "rescored_student_scores": rescored_scores,

            "accuracy": final_scores.get("accuracy"),
            "completeness": final_scores.get("completeness"),
            "relevance": final_scores.get("relevance"),
            "clarity": final_scores.get("clarity"),

            "accuracy_initial": student_scores.get("accuracy") if student_scores else None,
            "completeness_initial": student_scores.get("completeness") if student_scores else None,
            "relevance_initial": student_scores.get("relevance") if student_scores else None,
            "clarity_initial": student_scores.get("clarity") if student_scores else None,

            "accuracy_rescored": rescored_scores.get("accuracy") if rescored_scores else None,
            "completeness_rescored": rescored_scores.get("completeness") if rescored_scores else None,
            "relevance_rescored": rescored_scores.get("relevance") if rescored_scores else None,
            "clarity_rescored": rescored_scores.get("clarity") if rescored_scores else None,
        }
        records.append(record)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(records)
    df_out = df[
        [
            "id", "code", "question", "answer", "prediction",
            "accuracy", "completeness", "relevance", "clarity",
            "accuracy_initial", "completeness_initial", "relevance_initial", "clarity_initial",
            "accuracy_rescored", "completeness_rescored", "relevance_rescored", "clarity_rescored",
            "teacher_intervened",
        ]
    ]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\nDone")
    print("JSON:", OUTPUT_JSON)
    print("CSV :", OUTPUT_CSV)
    print(f"Teacher intervened on {len(intervene_ids)}/{n} examples ({TEACHER_INTERVENTION_RATE*100:.0f}%).")


if __name__ == "__main__":
    main()
