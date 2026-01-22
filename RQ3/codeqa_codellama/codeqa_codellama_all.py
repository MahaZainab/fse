#!/usr/bin/env python3
"""
Single-turn Theory-of-Mind Supervision (ToMS) Evaluation
Adapted for remote cluster execution with GPU support
WITH TIME TRACKING AND VISUALIZATION

Experiment: Single teacher intervention (Theory-of-Mind guidance) -> student updates scores once
- Teacher intervention is ONE message (no teacher-student dialogue, no multi-turn mentoring loop)
- Teacher applies Theory of Mind to diagnose the student's evaluation and give targeted guidance
- Student re-scores once using the teacher's ToM guidance
"""

import json
import random
import re
import os
import time
from typing import Any, Dict, Optional, List
from datetime import datetime

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =============================
# Configuration
# =============================
INPUT_JSON = "mini.json"
OUTPUT_JSON = "miniofsixty.json"
OUTPUT_CSV = "miniofsixty.csv"
OUTPUT_TIMING_JSON = "timing_data.json"
OUTPUT_TIMING_CSV = "timing_data.csv"

STUDENT_JUDGE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TEACHER_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

TEACHER_INTERVENTION_RATE = 1.00
RANDOM_SEED = 42

# HuggingFace cache directory
HF_CACHE_DIR = os.getenv("HF_HOME", "/aiau010_scratch/maz0032/.cache/huggingface")

# =============================
# Time Tracking Class
# =============================
class TimeTracker:
    """Track and record execution times for different operations."""
    
    def __init__(self):
        self.timings = {
            'model_loading': {},
            'per_example': [],
            'phase_totals': {
                'student_initial': 0.0,
                'teacher_intervention': 0.0,
                'student_rescore': 0.0
            }
        }
        self.start_time = time.time()
    
    def record_model_load(self, model_name: str, duration: float):
        """Record model loading time."""
        self.timings['model_loading'][model_name] = duration
    
    def record_example(self, example_timing: Dict[str, Any]):
        """Record timing for a single example."""
        self.timings['per_example'].append(example_timing)
        
        # Update phase totals
        self.timings['phase_totals']['student_initial'] += example_timing.get('student_initial_time', 0)
        self.timings['phase_totals']['teacher_intervention'] += example_timing.get('teacher_time', 0)
        self.timings['phase_totals']['student_rescore'] += example_timing.get('rescore_time', 0)
    
    def get_total_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time
    
    def save(self, json_path: str, csv_path: str):
        """Save timing data to files."""
        # Save complete timing data as JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.timings, f, indent=2)
        
        # Save per-example timings as CSV
        if self.timings['per_example']:
            df = pd.DataFrame(self.timings['per_example'])
            df.to_csv(csv_path, index=False)

# =============================
# Model wrapper (HF)
# =============================
class HFChat:
    """HuggingFace chat model wrapper with GPU support."""
    
    def __init__(
        self, 
        model_name: str, 
        temperature: float = 0.0, 
        max_new_tokens: int = 256,
        cache_dir: str = HF_CACHE_DIR
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        print(f"\nLoading model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        load_start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.load_time = time.time() - load_start
        print(f"Model loaded on device: {self.model.device}")
        print(f"Load time: {self.load_time:.2f}s")

    def invoke(self, chat_messages: List[Dict[str, str]]) -> tuple[str, float]:
        """Generate response from chat messages. Returns (response, inference_time)."""
        start_time = time.time()
        
        prompt = self.tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": float(self.temperature) if self.temperature > 0 else 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = out[0, inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        inference_time = time.time() - start_time
        return response, inference_time


# =============================
# Prompts and Rubric
# =============================
ANCHORED_RUBRIC = r"""
Scoring uses an ANCHORED 3-point rubric:

GLOBAL ANCHORS:
- Score 3: Essentially correct and expert-level for this task.
  * Matches the reference answer in meaning (semantic equivalence).
  * Correctly reflects the code's behavior and intent.
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

JUDGE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing the quality of code comprehension by an LLM

You will receive:
- A Python code snippet
- A student question about that code
- A reference (correct) answer
- A TA LLM-generated answer (called the prediction)

Your task is to evaluate the prediction against the reference answer using four metrics:
accuracy, completeness, relevance, and clarity. For each, provide:
{ANCHORED_RUBRIC}

Return ONLY valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations.
""".strip()

TEACHER_TOM_SYSTEM_PROMPT = f"""
You are a 30B teacher LLM supervising a student LLM-as-judge.

You will receive:
- code, question, reference answer, TA prediction
- student judge scores (1-3) for accuracy/completeness/relevance/clarity

Use the SAME anchored rubric:

{ANCHORED_RUBRIC}

Your job is a SINGLE intervention (one response) that uses THEORY OF MIND:
1) Independently form your own scores based on code/question/reference/prediction.
2) Compare your scores to the student's scores.
3) Infer the student's likely mental model from the mismatch patterns (e.g., what it over-weighted, overlooked, assumed).
4) Provide targeted guidance that updates the student's evaluation strategy.

- Keep guidance concrete and evidence-based (point to specific missing/wrong aspects in the TA prediction vs reference/code).

Return ONLY valid JSON in EXACTLY this format:

{{
  "teacher_scores": {{
    "accuracy": 1-3,
    "completeness": 1-3,
    "relevance": 1-3,
    "clarity": 1-3
  }},
  "score_comparison": {{
    "accuracy": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "completeness": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "relevance": {{"student": 1-3, "teacher": 1-3, "match": true/false}},
    "clarity": {{"student": 1-3, "teacher": 1-3, "match": true/false}}
  }},
  "inferred_student_mental_model": {{
    "beliefs_or_criteria": "what the student judge seems to optimize for",
    "likely_blind_spots": ["...","..."],
    "likely_assumptions": ["...","..."]
  }},
  "tom_guidance": {{
    "accuracy": "2-4 sentences if mismatch else empty string",
    "completeness": "2-4 sentences if mismatch else empty string",
    "relevance": "2-4 sentences if mismatch else empty string",
    "clarity": "2-4 sentences if mismatch else empty string"
  }},
  "checklist_next_time": ["short actionable item", "short actionable item"]
}}

Rules:
- In tom_guidance, leave empty string for dimensions where match=true.
- Keep the mental model concise and plausible based on the student's scores.
""".strip()

RESCORE_SYSTEM_PROMPT = f"""
You are a large language model acting as a judge for assessing a the code comprehension by an LLM.

You will receive:
- code, question, reference answer, TA prediction
- your previous scores
- You will now see Theory-of-Mind guidance from a teacher describing:
- how you may have been thinking,
- what conceptual gaps or misconceptions you likely had,
- and how to improve.
IMPORTANT:
• Do not use teacher scores.
• You must NOT try to infer or guess any teacher scores.
• You must rescore ONLY by re-evaluating the work itself using the rubric.

Your job is to reflect, correct your understanding, and then rescore.
Use the SAME anchored rubric:
{ANCHORED_RUBRIC}

Return ONLY valid JSON in EXACTLY this format:
{{
  "accuracy": {{ "score": 1-3 }},
  "completeness": {{ "score": 1-3 }},
  "relevance": {{ "score": 1-3 }},
  "clarity": {{ "score": 1-3 }}
}}

Do NOT include explanations.
""".strip()


# =============================
# Prompt builders
# =============================
def build_judge_user_prompt(code: str, question: str, reference: str, prediction: str) -> str:
    """Build initial judge prompt."""
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


def build_teacher_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    student_scores: Dict[str, int],
) -> str:
    """Build teacher intervention prompt."""
    return f"""
Code:
```python
{code}
```

Question:
{question}

Reference Answer:
{reference}

TA Prediction:
{prediction}

Student Judge Scores:
{json.dumps(student_scores, indent=2)}
""".strip()


def build_rescore_user_prompt(
    code: str,
    question: str,
    reference: str,
    prediction: str,
    prev_scores: Dict[str, int],
    teacher_json: Dict[str, Any],
) -> str:
    """Build student rescore prompt after teacher guidance."""
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


# =============================
# Parsing helpers
# =============================
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def _extract_first_json_obj(text: str) -> Optional[str]:
    """Extract first JSON object from text."""
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else None


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON from text."""
    js_text = _extract_first_json_obj(text)
    if not js_text:
        return None
    try:
        obj = json.loads(js_text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_score_json(raw_text: str) -> Optional[Dict[str, int]]:
    """Parse score JSON with validation."""
    obj = _safe_json(raw_text)
    if not obj:
        return None
    
    out: Dict[str, int] = {}
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = obj.get(k, {})
        score = v.get("score") if isinstance(v, dict) else None
        
        # Handle string numbers
        if isinstance(score, str) and score.isdigit():
            score = int(score)
        # Handle floats
        if isinstance(score, float):
            score = int(round(score))
        
        # Validate range
        if not isinstance(score, int) or score < 1 or score > 3:
            return None
        
        out[k] = score
    
    return out


def parse_teacher_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse teacher JSON with validation."""
    obj = _safe_json(raw_text)
    if not obj:
        return None
    
    required = {
        "teacher_scores", 
        "score_comparison", 
        "inferred_student_mental_model", 
        "tom_guidance", 
        "checklist_next_time"
    }
    
    if not required.issubset(set(obj.keys())):
        return None
    
    # Validate teacher scores
    ts = obj.get("teacher_scores", {})
    if not isinstance(ts, dict):
        return None
    
    for k in ["accuracy", "completeness", "relevance", "clarity"]:
        v = ts.get(k)
        if not isinstance(v, int) or v < 1 or v > 3:
            return None
    
    return obj


# =============================
# Visualization Functions
# =============================
def create_timing_visualizations(time_tracker: TimeTracker, output_dir: str = "."):
    """Create comprehensive timing visualizations."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Model Loading Times (if available)
    if time_tracker.timings['model_loading']:
        ax1 = plt.subplot(2, 3, 1)
        models = list(time_tracker.timings['model_loading'].keys())
        times = list(time_tracker.timings['model_loading'].values())
        bars = ax1.bar(range(len(models)), times, color=['#3498db', '#e74c3c'])
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.split('/')[-1] for m in models], rotation=15, ha='right')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Model Loading Times', fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 2. Phase Total Times
    ax2 = plt.subplot(2, 3, 2)
    phases = list(time_tracker.timings['phase_totals'].keys())
    phase_times = list(time_tracker.timings['phase_totals'].values())
    colors = ['#2ecc71', '#f39c12', '#9b59b6']
    bars = ax2.bar(range(len(phases)), phase_times, color=colors)
    ax2.set_xticks(range(len(phases)))
    ax2.set_xticklabels([p.replace('_', ' ').title() for p in phases], rotation=15, ha='right')
    ax2.set_ylabel('Total Time (seconds)')
    ax2.set_title('Total Time by Phase', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 3. Per-Example Time Distribution
    if time_tracker.timings['per_example']:
        ax3 = plt.subplot(2, 3, 3)
        total_times = [ex.get('total_time', 0) for ex in time_tracker.timings['per_example']]
        ax3.hist(total_times, bins=20, color='#16a085', edgecolor='black', alpha=0.7)
        ax3.axvline(sum(total_times)/len(total_times), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {sum(total_times)/len(total_times):.2f}s')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Per-Example Processing Time', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Time Series: Processing Time per Example
    if time_tracker.timings['per_example']:
        ax4 = plt.subplot(2, 3, 4)
        example_ids = [ex.get('example_id', i) for i, ex in enumerate(time_tracker.timings['per_example'])]
        total_times = [ex.get('total_time', 0) for ex in time_tracker.timings['per_example']]
        ax4.plot(example_ids, total_times, marker='o', linestyle='-', color='#3498db', linewidth=2, markersize=4)
        ax4.set_xlabel('Example ID')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Processing Time per Example (Time Series)', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3)
    
    # 5. Stacked Bar: Time Breakdown by Phase for Each Example
    if time_tracker.timings['per_example']:
        ax5 = plt.subplot(2, 3, 5)
        example_ids = [ex.get('example_id', i) for i, ex in enumerate(time_tracker.timings['per_example'])]
        student_times = [ex.get('student_initial_time', 0) for ex in time_tracker.timings['per_example']]
        teacher_times = [ex.get('teacher_time', 0) for ex in time_tracker.timings['per_example']]
        rescore_times = [ex.get('rescore_time', 0) for ex in time_tracker.timings['per_example']]
        
        # Sample if too many examples
        if len(example_ids) > 50:
            indices = list(range(0, len(example_ids), len(example_ids)//50))
            example_ids = [example_ids[i] for i in indices]
            student_times = [student_times[i] for i in indices]
            teacher_times = [teacher_times[i] for i in indices]
            rescore_times = [rescore_times[i] for i in indices]
        
        ax5.bar(range(len(example_ids)), student_times, label='Student Initial', color='#2ecc71')
        ax5.bar(range(len(example_ids)), teacher_times, bottom=student_times, 
               label='Teacher Intervention', color='#f39c12')
        bottom = [s+t for s, t in zip(student_times, teacher_times)]
        ax5.bar(range(len(example_ids)), rescore_times, bottom=bottom, 
               label='Student Rescore', color='#9b59b6')
        
        ax5.set_xlabel('Example Index')
        ax5.set_ylabel('Time (seconds)')
        ax5.set_title('Time Breakdown by Phase (Stacked)', fontweight='bold', fontsize=12)
        ax5.legend(loc='upper right')
        ax5.grid(axis='y', alpha=0.3)
    
    # 6. Pie Chart: Phase Distribution
    ax6 = plt.subplot(2, 3, 6)
    phase_labels = [p.replace('_', ' ').title() for p in phases]
    ax6.pie(phase_times, labels=phase_labels, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax6.set_title('Time Distribution Across Phases', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'timing_analysis_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTiming visualization saved: {output_path}")
    
    # Create additional detailed plot for intervention vs non-intervention
    if time_tracker.timings['per_example']:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        intervened = [ex for ex in time_tracker.timings['per_example'] if ex.get('teacher_intervened', False)]
        not_intervened = [ex for ex in time_tracker.timings['per_example'] if not ex.get('teacher_intervened', False)]
        
        if intervened and not_intervened:
            # Box plot comparison
            data_to_plot = [
                [ex.get('total_time', 0) for ex in not_intervened],
                [ex.get('total_time', 0) for ex in intervened]
            ]
            bp = ax1.boxplot(data_to_plot, labels=['No Intervention', 'With Intervention'],
                           patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Processing Time: Intervention vs No Intervention', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Average times bar chart
            avg_no_interv = sum(ex.get('total_time', 0) for ex in not_intervened) / len(not_intervened)
            avg_interv = sum(ex.get('total_time', 0) for ex in intervened) / len(intervened)
            
            bars = ax2.bar(['No Intervention', 'With Intervention'], 
                          [avg_no_interv, avg_interv],
                          color=['#3498db', '#e74c3c'], alpha=0.7)
            ax2.set_ylabel('Average Time (seconds)')
            ax2.set_title('Average Processing Time by Group', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path2 = os.path.join(output_dir, f'intervention_comparison_{timestamp}.png')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        print(f"Intervention comparison saved: {output_path2}")
    
    plt.close('all')


# =============================
# Main execution
# =============================
def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("Single-Turn Theory-of-Mind Supervision (ToMS)")
    print("WITH TIME TRACKING AND VISUALIZATION")
    print("=" * 60)
    
    # Initialize time tracker
    time_tracker = TimeTracker()
    
    # Set random seed
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load data
    print(f"\nLoading data from: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    # Initialize models
    print("\n" + "=" * 60)
    print("Initializing models...")
    print("=" * 60)
    
    student = HFChat(
        STUDENT_JUDGE_MODEL, 
        temperature=0.0, 
        max_new_tokens=256,
        cache_dir=HF_CACHE_DIR
    )
    time_tracker.record_model_load(STUDENT_JUDGE_MODEL, student.load_time)
    
    teacher = HFChat(
        TEACHER_MODEL, 
        temperature=0.0, 
        max_new_tokens=900,
        cache_dir=HF_CACHE_DIR
    )
    time_tracker.record_model_load(TEACHER_MODEL, teacher.load_time)

    # Determine intervention examples
    n = len(data)
    k = int(round(TEACHER_INTERVENTION_RATE * n))
    intervene_ids = set(random.sample(range(n), k)) if n > 0 and k > 0 else set()
    
    print(f"\n" + "=" * 60)
    print(f"Will intervene on {k}/{n} examples ({TEACHER_INTERVENTION_RATE*100:.0f}%)")
    print("=" * 60 + "\n")

    records = []

    # Process each example
    for idx, ex in enumerate(tqdm(data, desc="Processing examples")):
        example_start_time = time.time()
        
        code = ex.get("code", "")
        question = ex.get("question", "")
        reference = ex.get("answer", "")
        prediction = ex.get("prediction", "")

        # Step 1: Student initial scoring
        raw_student, student_time = student.invoke([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(code, question, reference, prediction)},
        ])
        student_scores = parse_score_json(raw_student)

        teacher_intervened = idx in intervene_ids
        raw_teacher = None
        teacher_out = None
        raw_rescore = None
        rescored_scores = None
        teacher_time = 0.0
        rescore_time = 0.0

        # Step 2: Teacher intervention (if selected)
        if teacher_intervened and student_scores is not None:
            raw_teacher, teacher_time = teacher.invoke([
                {"role": "system", "content": TEACHER_TOM_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_user_prompt(
                    code, question, reference, prediction, student_scores
                )},
            ])
            teacher_out = parse_teacher_json(raw_teacher)

            # Step 3: Student rescores based on teacher guidance
            if teacher_out is not None:
                raw_rescore, rescore_time = student.invoke([
                    {"role": "system", "content": RESCORE_SYSTEM_PROMPT},
                    {"role": "user", "content": build_rescore_user_prompt(
                        code, question, reference, prediction, student_scores, teacher_out
                    )},
                ])
                rescored_scores = parse_score_json(raw_rescore)

        # Determine final scores
        final_scores = rescored_scores if rescored_scores is not None else (student_scores or {})
        
        # Record timing for this example
        example_total_time = time.time() - example_start_time
        example_timing = {
            'example_id': ex.get("id", idx),
            'student_initial_time': student_time,
            'teacher_time': teacher_time,
            'rescore_time': rescore_time,
            'total_time': example_total_time,
            'teacher_intervened': teacher_intervened
        }
        time_tracker.record_example(example_timing)

        # Build record
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
            "teacher_tom_json": teacher_out,

            "raw_student_rescore_output": raw_rescore,
            "rescored_student_scores": rescored_scores,

            # Final scores (flattened)
            "accuracy": final_scores.get("accuracy"),
            "completeness": final_scores.get("completeness"),
            "relevance": final_scores.get("relevance"),
            "clarity": final_scores.get("clarity"),

            # Initial scores
            "accuracy_initial": student_scores.get("accuracy") if student_scores else None,
            "completeness_initial": student_scores.get("completeness") if student_scores else None,
            "relevance_initial": student_scores.get("relevance") if student_scores else None,
            "clarity_initial": student_scores.get("clarity") if student_scores else None,

            # Rescored scores
            "accuracy_rescored": rescored_scores.get("accuracy") if rescored_scores else None,
            "completeness_rescored": rescored_scores.get("completeness") if rescored_scores else None,
            "relevance_rescored": rescored_scores.get("relevance") if rescored_scores else None,
            "clarity_rescored": rescored_scores.get("clarity") if rescored_scores else None,
            
            # Timing data
            "student_initial_time": student_time,
            "teacher_time": teacher_time,
            "rescore_time": rescore_time,
            "total_example_time": example_total_time,
        }
        records.append(record)

    # Calculate total execution time
    total_execution_time = time_tracker.get_total_time()

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {OUTPUT_JSON}")

    # Save CSV
    df = pd.DataFrame(records)
    df_out = df[[
        "id", "code", "question", "answer", "prediction",
        "accuracy", "completeness", "relevance", "clarity",
        "accuracy_initial", "completeness_initial", "relevance_initial", "clarity_initial",
        "accuracy_rescored", "completeness_rescored", "relevance_rescored", "clarity_rescored",
        "teacher_intervened",
        "student_initial_time", "teacher_time", "rescore_time", "total_example_time"
    ]]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"CSV saved: {OUTPUT_CSV}")
    
    # Save timing data
    time_tracker.save(OUTPUT_TIMING_JSON, OUTPUT_TIMING_CSV)
    print(f"Timing JSON saved: {OUTPUT_TIMING_JSON}")
    print(f"Timing CSV saved: {OUTPUT_TIMING_CSV}")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating timing visualizations...")
    print("=" * 60)
    create_timing_visualizations(time_tracker)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total examples: {n}")
    print(f"Teacher interventions: {len(intervene_ids)} ({TEACHER_INTERVENTION_RATE*100:.0f}%)")
    print(f"\nTiming Summary:")
    print(f"  Total execution time: {total_execution_time:.2f}s ({total_execution_time/60:.2f} min)")
    print(f"  Average time per example: {total_execution_time/n:.2f}s")
    
    if time_tracker.timings['per_example']:
        avg_student = sum(ex['student_initial_time'] for ex in time_tracker.timings['per_example']) / len(time_tracker.timings['per_example'])
        print(f"  Average student initial time: {avg_student:.2f}s")
        
        intervened_examples = [ex for ex in time_tracker.timings['per_example'] if ex['teacher_intervened']]
        if intervened_examples:
            avg_teacher = sum(ex['teacher_time'] for ex in intervened_examples) / len(intervened_examples)
            avg_rescore = sum(ex['rescore_time'] for ex in intervened_examples) / len(intervened_examples)
            print(f"  Average teacher time (intervened only): {avg_teacher:.2f}s")
            print(f"  Average rescore time (intervened only): {avg_rescore:.2f}s")
    
    print(f"\nPhase Totals:")
    for phase, time_val in time_tracker.timings['phase_totals'].items():
        print(f"  {phase.replace('_', ' ').title()}: {time_val:.2f}s")
    
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_TIMING_JSON}")
    print(f"  - {OUTPUT_TIMING_CSV}")
    print(f"  - timing_analysis_*.png")
    print(f"  - intervention_comparison_*.png")
    print("\nDone!")


if __name__ == "__main__":
    main()