import json
import random
import re
from typing import Any, Dict, Optional, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================
# Experiment: Single teacher intervention -> student updates scores once
# Teacher is instructed to "use chain-of-thought" INTERNALLY (private),
# but outputs only concise justifications + guidance (no hidden reasoning text).
# =============================

INPUT_JSON = "Codellama_predictions.json"                         # expects list of dicts with keys: code, question, answer, prediction
OUTPUT_JSON = "cl_cot_thirty.json"
OUTPUT_CSV = "cl_cot_thirty.csv"

STUDENT_JUDGE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

TEACHER_INTERVENTION_RATE = .30
RANDOM_SEED = 42


# -----------------------------
# Model wrapper (HF)
# -----------------------------
class HFChat:
    def __init__(self, model_name: str, temperature: float = 0.0, max_new_tokens: int = 256):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()

    def invoke(self, chat_messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": float(self.temperature),
        }

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# -----------------------------
# Anchored rubric (consistent meaning for 1/2/3)
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

Do NOT include explanations.
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
# Teacher is asked to "solve" the evaluation using step-by-step reasoning INTERNALLY.
# NOTE: teacher must NOT reveal private chain-of-thought; instead gives concise evidence-based justification.
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
- Decide your teacher_scores using the anchored rubric.

OUTPUT RULES:
- Do NOT output any private chain-of-thought.
- Output only concise, high-signal justifications: short "evidence" bullets per dimension (1–3 bullets each).
- Do not share your own score directly with the student judge.

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
# Student rescore prompt (after teacher guidance): strict JSON only
# -----------------------------
RESCORE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing an LLM answer.

You will receive:
- code, question, reference answer, LLM prediction
- your previous scores
- teacher guidance 
do not use teacher scores directly.

Use the SAME anchored rubric:

{ANCHORED_RUBRIC}

Update your scores accordingly.

Return ONLY valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations.
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
# Parsing helpers
# -----------------------------
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


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
    if "teacher_scores" not in obj or "comparison" not in obj or "guidance" not in obj:
        return None
    # light validate teacher_scores
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
    random.seed(RANDOM_SEED)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    student = HFChat(STUDENT_JUDGE_MODEL, temperature=0.0, max_new_tokens=256)
    teacher = HFChat(TEACHER_MODEL, temperature=0.0, max_new_tokens=900)

    n = len(data)
    k = int(round(TEACHER_INTERVENTION_RATE * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()

    records = []

    for idx, ex in enumerate(data):
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

        teacher_intervened = idx in intervene_ids
        raw_teacher = None
        teacher_out = None

        raw_rescore = None
        rescored_scores = None

        if teacher_intervened and student_scores is not None:
            # Teacher intervention (single turn)
            raw_teacher = teacher.invoke([
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_user_prompt(code, question, reference, prediction, student_scores)},
            ])
            teacher_out = parse_teacher_json(raw_teacher)

            # Student updates scores based on teacher guidance (single rescore turn)
            if teacher_out is not None:
                raw_rescore = student.invoke([
                    {"role": "system", "content": RESCORE_SYSTEM_PROMPT},
                    {"role": "user", "content": build_rescore_user_prompt(code, question, reference, prediction, student_scores, teacher_out)},
                ])
                rescored_scores = parse_score_json(raw_rescore)

        # Final scores: use rescored if available else original
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

            # flattened final
            "accuracy": final_scores.get("accuracy"),
            "completeness": final_scores.get("completeness"),
            "relevance": final_scores.get("relevance"),
            "clarity": final_scores.get("clarity"),

            # initial vs rescored
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
    df_out = df[[
        "code", "question", "answer", "prediction",
        "accuracy", "completeness", "relevance", "clarity",
        "accuracy_initial", "completeness_initial", "relevance_initial", "clarity_initial",
        "accuracy_rescored", "completeness_rescored", "relevance_rescored", "clarity_rescored",
        "teacher_intervened"
    ]]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("Done")
    print("JSON:", OUTPUT_JSON)
    print("CSV :", OUTPUT_CSV)
    print(f"Teacher intervened on {len(intervene_ids)}/{n} examples ({TEACHER_INTERVENTION_RATE*100:.0f}%).")


if __name__ == "__main__":
    main()
