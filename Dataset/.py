"""
Equal-per-category sampler for CodeQA and CS1QA datasets.

CodeQA  — no questionType field; type is DERIVED from question text
          Categories (7, dropping "Other"):
            What, How, Where, When, Why, For what purpose, Yes/No

CS1QA   — questionType field already present
          Categories (9):
            Student (6): code_understanding, logical_error, error,
                         usage (Function/Syntax Usage), algorithm, task
            TA     (3): reasoning, code_explain (Explanation), variable (Meaning)

Usage:
    python sample_equal_categories.py \
        --codeqa  CodeQA_Part1.json \
        --cs1qa   CS1QA.json \
        --n       50 \
        --seed    42 \
        --out_codeqa  CodeQA_sampled.json \
        --out_cs1qa   CS1QA_sampled.json
"""

import json
import argparse
import random
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────
# CodeQA: derive question type from question text
# ──────────────────────────────────────────────

CODEQA_CATEGORIES = [
    "What",
    "How",
    "Where",
    "When",
    "Why",
    "For what purpose",
    "Yes/No",
    # "Other" is intentionally excluded
]

def classify_codeqa(question: str) -> str:
    """
    Classify a CodeQA question into one of 7 types based on the
    question's opening words, exactly matching the paper's Table 6.
    Returns None if the question falls into 'Other' (excluded).
    """
    q = question.strip().lower()

    # Strip leading punctuation / quotes that sometimes appear
    q = q.lstrip("""\"''"" ''()[]{}\t\n\r")

    if q.startswith("for what purpose"):
        return "For what purpose"
    if q.startswith("what"):
        return "What"
    if q.startswith("how"):
        return "How"
    if q.startswith("where"):
        return "Where"
    if q.startswith("when"):
        return "When"
    if q.startswith("why"):
        return "Why"
    # Yes/No questions start with auxiliary verbs
    yn_starters = (
        "does", "do", "is", "are", "can", "should",
        "could", "would", "will", "did", "was", "were",
        "has", "have", "had"
    )
    if q.startswith(yn_starters):
        return "Yes/No"

    return None   # "Other" → excluded


# ──────────────────────────────────────────────
# CS1QA: map raw questionType strings → clean labels
# ──────────────────────────────────────────────

# The 9 included categories from the paper (Table 3 / Image 1).
# Comparison and Guiding are excluded (no code annotations in the paper).
CS1QA_TYPE_MAP = {
    # Student types
    "code_understanding": "Code Understanding",
    "logical_error":      "Logical Error",
    "logical":            "Logical Error",       # alternate prefix seen in data
    "error":              "Error",
    "usage":              "Function/Syntax Usage",
    "algorithm":          "Algorithm",
    "task":               "Task",
    # TA types
    "reasoning":          "Reasoning",
    "code_explain":       "Explanation",
    "variable":           "Meaning",
}

CS1QA_CATEGORIES = list(dict.fromkeys(CS1QA_TYPE_MAP.values()))  # ordered, unique

def classify_cs1qa(question_type: str) -> str | None:
    """
    Map a CS1QA questionType string to a clean category label.
    Returns None for excluded types (comparison, guiding, unknown).
    """
    if not isinstance(question_type, str):
        return None
    qt = question_type.strip().lower()
    # Match by prefix to handle values like "code_understanding_1"
    for key, label in CS1QA_TYPE_MAP.items():
        if qt == key or qt.startswith(key + "_"):
            return label
    return None   # excluded / unknown


# ──────────────────────────────────────────────
# Generic sampler
# ──────────────────────────────────────────────

def sample_equal(records: list[dict], get_category, n: int, seed: int) -> list[dict]:
    """
    Group records by category, sample exactly n per category.
    Categories with fewer than n records are kept in full with a warning.
    Records whose category is None (excluded) are silently dropped.
    """
    rng = random.Random(seed)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        cat = get_category(rec)
        if cat is not None:
            rec = dict(rec, _category=cat)   # attach category for reference
            buckets[cat].append(rec)

    sampled = []
    for cat in sorted(buckets):
        pool = buckets[cat]
        if len(pool) < n:
            print(f"  ⚠  '{cat}': only {len(pool)} records available "
                  f"(requested {n}) — keeping all.")
            chosen = pool
        else:
            chosen = rng.sample(pool, n)
        print(f"  ✓  '{cat}': sampled {len(chosen)}")
        sampled.extend(chosen)

    return sampled


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")

def save_json(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  → Saved {len(records)} records to '{path}'")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample equal number of questions per category "
                    "from CodeQA and CS1QA datasets."
    )
    parser.add_argument("--codeqa",     required=True,  help="Path to CodeQA JSON file")
    parser.add_argument("--cs1qa",      required=True,  help="Path to CS1QA JSON file")
    parser.add_argument("--n",          type=int, default=60,
                        help="Number of samples per category (default: 60)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out_codeqa", default="CodeQA_sampled.json",
                        help="Output path for sampled CodeQA")
    parser.add_argument("--out_cs1qa",  default="CS1QA_sampled.json",
                        help="Output path for sampled CS1QA")
    args = parser.parse_args()

    # ── CodeQA ──
    print(f"\n{'='*50}")
    print(f"CodeQA  →  {args.codeqa}")
    print(f"  Categories (7, 'Other' excluded): {CODEQA_CATEGORIES}")
    print(f"  Sampling {args.n} per category  |  seed={args.seed}")
    print(f"{'='*50}")

    codeqa_data = load_json(args.codeqa)
    print(f"  Loaded {len(codeqa_data)} records")

    codeqa_sampled = sample_equal(
        records=codeqa_data,
        get_category=lambda rec: classify_codeqa(rec.get("question", "")),
        n=args.n,
        seed=args.seed,
    )
    save_json(codeqa_sampled, args.out_codeqa)

    # ── CS1QA ──
    print(f"\n{'='*50}")
    print(f"CS1QA   →  {args.cs1qa}")
    print(f"  Categories (9, Comparison/Guiding excluded): {CS1QA_CATEGORIES}")
    print(f"  Sampling {args.n} per category  |  seed={args.seed}")
    print(f"{'='*50}")

    cs1qa_data = load_json(args.cs1qa)
    print(f"  Loaded {len(cs1qa_data)} records")

    cs1qa_sampled = sample_equal(
        records=cs1qa_data,
        get_category=lambda rec: classify_cs1qa(rec.get("questionType", "")),
        n=args.n,
        seed=args.seed,
    )
    save_json(cs1qa_sampled, args.out_cs1qa)

    # ── Summary ──
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  CodeQA  sampled: {len(codeqa_sampled):>5}  "
          f"({len(CODEQA_CATEGORIES)} categories × up to {args.n})")
    print(f"  CS1QA   sampled: {len(cs1qa_sampled):>5}  "
          f"({len(CS1QA_CATEGORIES)} categories × up to {args.n})")
    print(f"  Total          : {len(codeqa_sampled) + len(cs1qa_sampled):>5}")
    print()


if __name__ == "__main__":
    main()
