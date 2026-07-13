"""Generate SFT Q&A dataset using Azure OpenAI on Modal.

Reads cleaned corpus from the Modal volume, generates grounded Q&A pairs,
filters and deduplicates, outputs chat JSONL.

Usage:
    modal run sft/datagen_azure.py::generate
    modal run sft/datagen_azure.py::filter_dataset
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm125m-sft-data")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

SFT_DIR = f"{config.DATA_ROOT}/sft"
RAW_QA_DIR = f"{SFT_DIR}/raw"
FINAL_QA_PATH = f"{SFT_DIR}/sft_train.jsonl"
REPORT_PATH = f"{SFT_DIR}/generation_report.json"

TARGET_PAIRS = 12_000
OVERGENERATE_FACTOR = 2.0
RAW_TARGET = int(TARGET_PAIRS * OVERGENERATE_FACTOR)

MAX_PASSAGE_CHARS = 3000
MIN_PASSAGE_CHARS = 200
PAIRS_PER_PASSAGE = 3
PASSAGES_NEEDED = RAW_TARGET // PAIRS_PER_PASSAGE

TASK_DISTRIBUTION = {
    "grounded_qa": 0.50,
    "extraction": 0.20,
    "summarization": 0.20,
    "refusal": 0.10,
}

PROMPTS = {
    "grounded_qa": """You are generating training data for a legal/financial language model.
Given the passage below, generate {n} diverse question-answer pairs.

Rules:
- Every answer MUST be supported ONLY by information in the passage
- Do NOT use external knowledge
- Questions should vary in difficulty (simple lookup, inference, multi-step)
- Keep answers concise (1-3 sentences)

Passage:
---
{passage}
---

Return a JSON array of objects with "question" and "answer" fields. Nothing else.""",

    "extraction": """You are generating training data for a legal/financial language model.
Given the passage below, generate {n} extraction/classification tasks.

Rules:
- Ask the model to extract structured information (parties, dates, amounts, rulings, filings)
- Answers must be supported ONLY by the passage
- Format answers as brief structured text

Passage:
---
{passage}
---

Return a JSON array of objects with "question" and "answer" fields. Nothing else.""",

    "summarization": """You are generating training data for a legal/financial language model.
Given the passage below, generate {n} summarization tasks.

Rules:
- Ask for summaries of different aspects/lengths
- Summaries must only contain information from the passage
- No invented facts

Passage:
---
{passage}
---

Return a JSON array of objects with "question" and "answer" fields. Nothing else.""",

    "refusal": """You are generating training data for a legal/financial language model.
Given the passage below, generate {n} questions whose answers are NOT in the passage.

Rules:
- Ask plausible legal/financial questions that the passage does NOT answer
- The correct answer for each should be a polite refusal like "The passage does not contain information about..."
- Make questions sound natural

Passage:
---
{passage}
---

Return a JSON array of objects with "question" and "answer" fields. The answer should be a refusal stating the information is not in the passage. Nothing else.""",
}

sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "openai==1.82.0",
        "numpy",
        "scikit-learn==1.6.1",
    )
    .add_local_python_source("config")
)


def _load_passages(max_passages: int) -> list[dict]:
    """Load passages from corpus, chunked to MAX_PASSAGE_CHARS."""
    import glob
    import random

    passages = []
    for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
        source = path.split("/")[-2]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                text = line.rstrip("\n")
                if len(text) < MIN_PASSAGE_CHARS:
                    continue
                if len(text) > MAX_PASSAGE_CHARS:
                    text = text[:MAX_PASSAGE_CHARS]
                passages.append({"text": text, "source": source})
                if len(passages) >= max_passages * 3:
                    break
        if len(passages) >= max_passages * 3:
            break

    random.seed(42)
    random.shuffle(passages)
    return passages[:max_passages]


def _assign_tasks(passages: list[dict]) -> list[dict]:
    """Assign task types to passages based on distribution."""
    import random

    random.seed(42)
    tasks = []
    for task_type, fraction in TASK_DISTRIBUTION.items():
        count = int(len(passages) * fraction)
        tasks.extend([task_type] * count)
    while len(tasks) < len(passages):
        tasks.append("grounded_qa")
    random.shuffle(tasks)

    for p, t in zip(passages, tasks):
        p["task_type"] = t
    return passages


@app.function(
    image=sft_image,
    volumes=VOLUMES,
    timeout=60 * 10,
    secrets=[modal.Secret.from_dotenv(filename=".env.local")],
    max_containers=10,
)
def generate_batch(batch: list[dict]) -> list[dict]:
    """Generate Q&A pairs for a batch of passages using Azure OpenAI."""
    import json
    import os
    import time

    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_BASE_URL"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    model = os.environ["AZURE_MODEL"].removeprefix("azure/")
    results = []

    for item in batch:
        prompt = PROMPTS[item["task_type"]].format(
            n=PAIRS_PER_PASSAGE, passage=item["text"]
        )
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a training data generator. Always respond with valid JSON arrays."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_completion_tokens=2048,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = json.loads(text)
                pairs = parsed if isinstance(parsed, list) else parsed.get("pairs", parsed.get("questions", parsed.get("data", [parsed])))
                if not isinstance(pairs, list):
                    pairs = [pairs]
                for pair in pairs:
                    if "question" in pair and "answer" in pair:
                        results.append({
                            "question": pair["question"].strip(),
                            "answer": pair["answer"].strip(),
                            "source": item["source"],
                            "task_type": item["task_type"],
                            "passage": item["text"],
                        })
                break
            except Exception as e:
                if "429" in str(e) or "RateLimitError" in type(e).__name__:
                    wait = min(60, 2 ** (attempt + 2))
                    print(f"Rate limited, waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                elif attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after 5 attempts: {e}")
        time.sleep(0.5)

    return results


@app.function(
    image=sft_image,
    volumes=VOLUMES,
    timeout=60 * 60,
    cpu=4.0,
    memory=8_192,
)
def run_filters(raw_path: str) -> dict:
    """Apply the 4-stage filter pipeline to raw Q&A pairs."""
    import json

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    with open(raw_path, encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh]

    initial = len(raw)
    stats = {"initial": initial}

    # Filter 1: Format — drop empty, truncated, or malformed
    valid = []
    for item in raw:
        q, a = item.get("question", ""), item.get("answer", "")
        if len(q) < 10 or len(a) < 10:
            continue
        if len(q) > 500 or len(a) > 800:
            continue
        valid.append(item)
    stats["after_format"] = len(valid)

    # Filter 2: Deduplication by TF-IDF cosine similarity
    if len(valid) > 100:
        questions = [item["question"] for item in valid]
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(questions)

        keep_mask = [True] * len(valid)
        for i in range(len(valid)):
            if not keep_mask[i]:
                continue
            sims = cosine_similarity(tfidf[i:i+1], tfidf[i+1:])[0]
            for j, sim in enumerate(sims, start=i+1):
                if keep_mask[j] and sim > 0.85:
                    keep_mask[j] = False

        valid = [item for item, keep in zip(valid, keep_mask) if keep]
    stats["after_dedup"] = len(valid)

    # Filter 3: Grounding check — answer words should overlap with passage
    grounded = []
    for item in valid:
        answer_words = set(item["answer"].lower().split())
        passage_words = set(item.get("passage", "").lower().split())
        if item["task_type"] == "refusal":
            grounded.append(item)
            continue
        overlap = len(answer_words & passage_words) / max(len(answer_words), 1)
        if overlap >= 0.3:
            grounded.append(item)
    valid = grounded
    stats["after_grounding"] = len(valid)

    # Filter 4: Task balance — cap over-represented types
    target_per_type = {}
    for task_type, frac in TASK_DISTRIBUTION.items():
        target_per_type[task_type] = int(TARGET_PAIRS * frac * 1.1)

    balanced = []
    type_counts: dict[str, int] = {}
    for item in valid:
        tt = item["task_type"]
        if type_counts.get(tt, 0) < target_per_type.get(tt, TARGET_PAIRS):
            balanced.append(item)
            type_counts[tt] = type_counts.get(tt, 0) + 1
    valid = balanced
    stats["after_balance"] = len(valid)

    # Trim to target
    valid = valid[:TARGET_PAIRS]
    stats["final"] = len(valid)

    # Write final dataset as chat JSONL
    with open(FINAL_QA_PATH, "w", encoding="utf-8") as fh:
        for item in valid:
            chat = {
                "messages": [
                    {"role": "system", "content": "You are a helpful legal and financial assistant. Answer based only on the provided context."},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ]
            }
            fh.write(json.dumps(chat) + "\n")
    volume.commit()

    print(f"FILTER REPORT: {initial} -> {stats['final']}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


@app.function(
    image=sft_image,
    volumes=VOLUMES,
    timeout=60 * 90,
    cpu=2.0,
    memory=4_096,
    secrets=[modal.Secret.from_dotenv(filename=".env.local")],
)
def orchestrate_generation() -> dict:
    """Main orchestrator: load passages, fan out generation, collect results."""
    import json
    import os
    import random

    random.seed(42)
    os.makedirs(RAW_QA_DIR, exist_ok=True)

    print(f"Loading passages (need {PASSAGES_NEEDED})...")
    passages = _load_passages(PASSAGES_NEEDED)
    passages = _assign_tasks(passages)
    print(f"Loaded {len(passages)} passages")

    # Split into batches for parallel generation
    batch_size = 10
    batches = [passages[i:i + batch_size] for i in range(0, len(passages), batch_size)]
    print(f"Dispatching {len(batches)} batches ({batch_size} passages each)...")

    all_results = []
    for result_batch in generate_batch.map(batches):
        all_results.extend(result_batch)
        if len(all_results) % 500 < batch_size * PAIRS_PER_PASSAGE:
            print(f"  Progress: {len(all_results)} pairs generated...")

    raw_path = f"{RAW_QA_DIR}/raw_all.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for item in all_results:
            fh.write(json.dumps(item) + "\n")
    volume.commit()

    print(f"Generation complete: {len(all_results)} raw pairs")
    print("Running filters...")
    stats = run_filters.remote(raw_path)

    report = {"raw_pairs": len(all_results), "filter_stats": stats, "passages_used": len(passages)}
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()

    return report


@app.local_entrypoint()
def generate():
    """Generate the full SFT dataset."""
    report = orchestrate_generation.remote()
    print(f"\nDONE. Final dataset: {report['filter_stats']['final']} pairs at {FINAL_QA_PATH}")
    print(f"Report: {report}")


@app.local_entrypoint()
def filter_dataset():
    """Re-run filters on existing raw data (for iteration)."""
    raw_path = f"{RAW_QA_DIR}/raw_all.jsonl"
    stats = run_filters.remote(raw_path)
    print(f"Filter complete: {stats}")
