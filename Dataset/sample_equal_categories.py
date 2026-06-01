import json
import argparse
import random
from collections import defaultdict
from typing import Optional, List, Dict


CODEQA_CATEGORIES = [
    "What",
    "How",
    "Where",
    "When",
    "Why",
    "For what purpose",
    "Yes/No",
]

CS1QA_TYPE_MAP = {
    "code_understanding": "Code Understanding",
    "logical_error":      "Logical Error",
    "logical":            "Logical Error",
    "error":              "Error",
    "usage":              "Function/Syntax Usage",
    "algorithm":          "Algorithm",
    "task":               "Task",
    "reasoning":          "Reasoning",
    "code_explain":       "Explanation",
    "variable":           "Meaning",
}

CS1QA_CATEGORIES = list(dict.fromkeys(CS1QA_TYPE_MAP.values()))


def classify_codeqa(question: str) -> Optional[str]:
    q = question.strip().lower()
    q = q.lstrip('\"\'\u201c\u201d\u2018\u2019()[]{} \t\n\r')
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
    yn_starters = (
        "does", "do", "is", "are", "can", "should",
        "could", "would", "will", "did", "was", "were",
        "has", "have", "had"
    )
    if q.startswith(yn_starters):
        return "Yes/No"
    return None


def classify_cs1qa(question_type: str) -> Optional[str]:
    if not isinstance(question_type, str):
        return None
    qt = question_type.strip().lower()
    for key, label in CS1QA_TYPE_MAP.items():
        if qt == key or qt.startswith(key + "_"):
            return label
    return None


def sample_equal(records: List[dict], get_category, n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        cat = get_category(rec)
        if cat is not None:
            buckets[cat].append(dict(rec, _category=cat))

    sampled = []
    for cat in sorted(buckets):
        pool = buckets[cat]
        if len(pool) < n:
            print(f"  WARNING  '{cat}': only {len(pool)} records available (requested {n}) — keeping all.")
            chosen = pool
        else:
            chosen = rng.sample(pool, n)
        print(f"  OK  '{cat}': sampled {len(chosen)}")
        sampled.extend(chosen)
    return sampled


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")


def save_json(records: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(records)} records to '{path}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codeqa",     default="CodeQA_dataset.json")
    parser.add_argument("--cs1qa",      default="CS1QA_dataset.json")
    parser.add_argument("--n",          type=int, default=60)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out_codeqa", default="CodeQA_sampled.json")
    parser.add_argument("--out_cs1qa",  default="CS1QA_sampled.json")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"CodeQA  ->  {args.codeqa}")
    print(f"Categories (7): {CODEQA_CATEGORIES}")
    print(f"Sampling {args.n} per category  |  seed={args.seed}")
    print(f"{'='*50}")
    codeqa_data = load_json(args.codeqa)
    print(f"Loaded {len(codeqa_data)} records")
    codeqa_sampled = sample_equal(
        records=codeqa_data,
        get_category=lambda rec: classify_codeqa(rec.get("question", "")),
        n=args.n,
        seed=args.seed,
    )
    save_json(codeqa_sampled, args.out_codeqa)

    print(f"\n{'='*50}")
    print(f"CS1QA   ->  {args.cs1qa}")
    print(f"Categories (9): {CS1QA_CATEGORIES}")
    print(f"Sampling {args.n} per category  |  seed={args.seed}")
    print(f"{'='*50}")
    cs1qa_data = load_json(args.cs1qa)
    print(f"Loaded {len(cs1qa_data)} records")
    cs1qa_sampled = sample_equal(
        records=cs1qa_data,
        get_category=lambda rec: classify_cs1qa(rec.get("questionType", "")),
        n=args.n,
        seed=args.seed,
    )
    save_json(cs1qa_sampled, args.out_cs1qa)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  CodeQA  sampled: {len(codeqa_sampled):>5}  (7 categories x {args.n})")
    print(f"  CS1QA   sampled: {len(cs1qa_sampled):>5}  (9 categories x {args.n})")
    print(f"  Total          : {len(codeqa_sampled) + len(cs1qa_sampled):>5}")
    print()


if __name__ == "__main__":
    main()
