"""Generate SFT Q&A dataset using Gemini 3.5 Flash on Modal.

Supplements existing Azure-generated pairs with hard negatives and
higher diversity. Combines with existing raw data, then re-filters.

Usage:
    modal run sft/datagen_gemini.py::generate
    modal run sft/datagen_gemini.py::filter_only
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm125m-sft-gemini")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

SFT_DIR = f"{config.DATA_ROOT}/sft"
RAW_QA_DIR = f"{SFT_DIR}/raw"
FINAL_QA_PATH = f"{SFT_DIR}/sft_train.jsonl"
REPORT_PATH = f"{SFT_DIR}/generation_report.json"

TARGET_PAIRS = 12_000
RAW_TARGET = 18_000
MAX_PASSAGE_CHARS = 3000
MIN_PASSAGE_CHARS = 200
PAIRS_PER_PASSAGE = 3
PASSAGES_NEEDED = RAW_TARGET // PAIRS_PER_PASSAGE

TASK_DISTRIBUTION = {
    "grounded_qa": 0.60,
    "hard_negative": 0.15,
    "extraction": 0.15,
    "summarization": 0.10,
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

Return a JSON array of objects with "question" and "answer" fields.""",

    "hard_negative": """You are generating training data for a legal/financial language model.
Given the passage below, generate {n} questions that APPEAR relevant to this passage's topic but whose answers are NOT actually contained in the passage.

Rules:
- Questions must be about the same domain/topic as the passage (legal, financial, regulatory)
- Questions must sound natural and specific (not vague)
- The passage must NOT contain enough information to answer them
- The correct response is a polite refusal explaining the passage doesn't contain that information
- Make the questions tricky — they should seem like the passage might answer them

Passage:
---
{passage}
---

Return a JSON array of objects with "question" and "answer" fields. Each answer should be a specific refusal like "Based on the provided context, I cannot determine [specific thing asked]. The passage discusses [what it actually covers] but does not address [what was asked]." """,

    "extraction": """You are generating training data for a legal/financial language model.
Given the passage below, generate {n} extraction/classification tasks.

Rules:
- Ask the model to extract structured information (parties, dates, amounts, rulings, filings, jurisdictions)
- Answers must be supported ONLY by the passage
- Format answers as brief structured text
- Vary the extraction targets across examples

Passage:
---
{passage}
---

Return a JSON array of objects with "question" and "answer" fields.""",

    "summarization": """You are generating training data for a legal/financial language model.
Given the passage below, generate {n} summarization tasks.

Rules:
- Ask for summaries of different aspects/lengths (one-sentence, key findings, procedural history)
- Summaries must only contain information from the passage
- No invented facts
- Vary between "summarize the main holding", "what are the key facts", "describe the procedural history"

Passage:
---
{passage}
---

Return a JSON array of objects with "question" and "answer" fields.""",
}

sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "google-genai==1.14.0",
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

    random.seed(99)
    random.shuffle(passages)
    return passages[:max_passages]


def _assign_tasks(passages: list[dict]) -> list[dict]:
    """Assign task types to passages based on distribution."""
    import random

    random.seed(99)
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
    timeout=60 * 30,
    secrets=[modal.Secret.from_dotenv(filename=".env.local")],
    max_containers=3,
)
def generate_batch(batch: list[dict]) -> list[dict]:
    """Generate Q&A pairs for a batch of passages using Gemini 3.5 Flash."""
    import json
    import os
    import time

    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results = []

    for item in batch:
        prompt = PROMPTS[item["task_type"]].format(
            n=PAIRS_PER_PASSAGE, passage=item["text"]
        )
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.7,
                    },
                )
                text = response.text.strip()
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
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = min(60, 2 ** (attempt + 2))
                    print(f"Rate limited, waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                elif attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Failed after 5 attempts: {e}")
        time.sleep(3.0)

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

    # Filter 1: Format
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
        batch_size = 500
        for start in range(0, len(valid), batch_size):
            end = min(start + batch_size, len(valid))
            sims = cosine_similarity(tfidf[start:end], tfidf)
            for i in range(end - start):
                if not keep_mask[start + i]:
                    continue
                for j in range(start + i + 1, len(valid)):
                    if keep_mask[j] and sims[i, j] > 0.85:
                        keep_mask[j] = False

        valid = [item for item, keep in zip(valid, keep_mask) if keep]
    stats["after_dedup"] = len(valid)

    # Filter 3: Grounding check (skip for hard negatives — refusals are expected)
    grounded = []
    for item in valid:
        if item["task_type"] in ("hard_negative", "refusal"):
            grounded.append(item)
            continue
        answer_words = set(item["answer"].lower().split())
        passage_words = set(item.get("passage", "").lower().split())
        overlap = len(answer_words & passage_words) / max(len(answer_words), 1)
        if overlap >= 0.3:
            grounded.append(item)
    valid = grounded
    stats["after_grounding"] = len(valid)

    # Filter 4: Task balance
    target_per_type = {}
    for task_type, frac in TASK_DISTRIBUTION.items():
        target_per_type[task_type] = int(TARGET_PAIRS * frac * 1.2)

    balanced = []
    type_counts: dict[str, int] = {}
    for item in valid:
        tt = item["task_type"]
        if type_counts.get(tt, 0) < target_per_type.get(tt, TARGET_PAIRS):
            balanced.append(item)
            type_counts[tt] = type_counts.get(tt, 0) + 1
    valid = balanced
    stats["after_balance"] = len(valid)
    stats["type_distribution"] = dict(type_counts)

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
    timeout=60 * 120,
    cpu=2.0,
    memory=4_096,
    secrets=[modal.Secret.from_dotenv(filename=".env.local")],
)
def orchestrate_generation() -> dict:
    """Main orchestrator: load passages, fan out generation, persist incrementally."""
    import json
    import os
    import random

    random.seed(99)
    os.makedirs(RAW_QA_DIR, exist_ok=True)

    gemini_raw_path = f"{RAW_QA_DIR}/raw_gemini.jsonl"

    # Resume: count existing pairs from prior partial runs
    existing_count = 0
    if os.path.exists(gemini_raw_path):
        with open(gemini_raw_path, encoding="utf-8") as fh:
            existing_count = sum(1 for _ in fh)
        print(f"Resuming: found {existing_count} existing Gemini pairs")

    print(f"Loading passages (need {PASSAGES_NEEDED})...")
    passages = _load_passages(PASSAGES_NEEDED)
    passages = _assign_tasks(passages)
    print(f"Loaded {len(passages)} passages")

    task_counts = {}
    for p in passages:
        task_counts[p["task_type"]] = task_counts.get(p["task_type"], 0) + 1
    print(f"Task distribution: {task_counts}")

    # Skip batches already completed in prior runs (conservative: round down)
    batch_size = 8
    batches = [passages[i:i + batch_size] for i in range(0, len(passages), batch_size)]
    if existing_count > 0:
        batches_to_skip = existing_count // (batch_size * PAIRS_PER_PASSAGE)
        if batches_to_skip > 0:
            batches = batches[batches_to_skip:]
            print(f"Skipping {batches_to_skip} already-completed batches, {len(batches)} remaining")

    print(f"Dispatching {len(batches)} batches ({batch_size} passages each)...")

    total_pairs = existing_count
    commit_interval = 50  # commit every 50 batches
    batch_count = 0

    # Open in append mode to persist incrementally
    with open(gemini_raw_path, "a", encoding="utf-8") as fh:
        for result_batch in generate_batch.map(batches):
            for item in result_batch:
                fh.write(json.dumps(item) + "\n")
            total_pairs += len(result_batch)
            batch_count += 1

            if batch_count % commit_interval == 0:
                fh.flush()
                volume.commit()
                print(f"  Progress: {total_pairs} pairs (committed to volume)")

    # Final commit
    volume.commit()
    print(f"Gemini generation complete: {total_pairs} total pairs ({total_pairs - existing_count} new)")

    # Combine with existing Azure raw data
    existing_raw_path = f"{RAW_QA_DIR}/raw_all.jsonl"
    combined_path = f"{RAW_QA_DIR}/raw_combined.jsonl"

    with open(combined_path, "w", encoding="utf-8") as out_fh:
        # Write Gemini pairs
        with open(gemini_raw_path, encoding="utf-8") as fh:
            for line in fh:
                out_fh.write(line)
        # Append Azure pairs if they exist
        azure_count = 0
        if os.path.exists(existing_raw_path):
            with open(existing_raw_path, encoding="utf-8") as fh:
                for line in fh:
                    out_fh.write(line)
                    azure_count += 1

    volume.commit()
    print(f"Combined: {total_pairs} Gemini + {azure_count} Azure = {total_pairs + azure_count} total")

    print("Running filters on combined dataset...")
    stats = run_filters.remote(combined_path)

    report = {
        "gemini_raw_pairs": total_pairs,
        "azure_raw_pairs": azure_count,
        "combined_raw": total_pairs + azure_count,
        "filter_stats": stats,
        "passages_used": len(passages),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()

    return report


@app.local_entrypoint()
def generate():
    """Generate supplemental SFT dataset with Gemini and combine with existing."""
    report = orchestrate_generation.remote()
    print(f"\nDONE. Final dataset: {report['filter_stats']['final']} pairs at {FINAL_QA_PATH}")
    print(f"Report: {report}")


@app.local_entrypoint()
def filter_only():
    """Re-run filters on combined raw data."""
    combined_path = f"{RAW_QA_DIR}/raw_combined.jsonl"
    stats = run_filters.remote(combined_path)
    print(f"Filter complete: {stats}")
