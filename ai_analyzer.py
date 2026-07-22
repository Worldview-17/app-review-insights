"""
AI Analysis Engine — DeepSeek V4 Pro (OpenAI-compatible)
==========================================================
Three-stage pipeline over cleaned App Store reviews:

1. ``analyze_topics``      → dynamic topic clustering
2. ``generate_insights``   → evidence-backed issue report per topic
3. ``generate_prd_and_tests`` → traceable PRD + test cases

All LLM calls go through the OpenAI-compatible DeepSeek API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Always load .env from the project root (next to this file), regardless of CWD.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek client (OpenAI-compatible)
# ---------------------------------------------------------------------------

_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
_DEEPSEEK_BASE_URL = (
    os.getenv("DEEPSEEK_BASE_URL", "")
    or os.getenv("OPENAI_BASE_URL", "")
    or "https://api.deepseek.com/v1"
)
_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy-init the OpenAI client pointed at DeepSeek.

    Raises:
        RuntimeError: If ``DEEPSEEK_API_KEY`` is missing or still the
            default placeholder.
        openai.AuthenticationError: If the key is rejected by DeepSeek
            (401).  In that case the log will show the key prefix so you
            can verify it matches what you expect.
    """
    global _client
    if _client is None:
        if not _DEEPSEEK_API_KEY or _DEEPSEEK_API_KEY == "你的密钥":
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未设置。请在 .env 中填入有效的 API Key。\n"
                f".env 路径: {_ENV_PATH}"
            )

        # Log masked key for debugging without leaking the full secret
        masked = _DEEPSEEK_API_KEY[:8] + "…" + _DEEPSEEK_API_KEY[-4:] if len(_DEEPSEEK_API_KEY) > 12 else "***"
        logger.info(
            "DeepSeek client initialising — model=%s  base=%s  key=%s (loaded from %s)",
            _DEEPSEEK_MODEL, _DEEPSEEK_BASE_URL, masked, _ENV_PATH,
        )

        _client = OpenAI(
            api_key=_DEEPSEEK_API_KEY,
            base_url=_DEEPSEEK_BASE_URL,
        )
    return _client


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

_SYSTEM_HEADER: str = (
    "You are a senior product analyst. "
    "Always respond in valid, parseable JSON. "
    "Do NOT include Markdown fences, code blocks, or extra commentary — "
    "output raw JSON only."
)

_RETRY_MAX: int = 3
_RETRY_BACKOFF: float = 2.0  # seconds


def _call_llm(
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Send a prompt to DeepSeek and return the raw completion text.

    Retries on transient errors with exponential backoff.
    **Does not retry** on authentication errors (401/403).
    """
    client = _get_client()
    last_err: Optional[Exception] = None

    for attempt in range(1, _RETRY_MAX + 1):
        try:
            resp = client.chat.completions.create(
                model=_DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_HEADER},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            logger.info("LLM raw response (attempt %d) length=%d chars.", attempt, len(text))
            # Always print the raw response for debugging — truncated to 2000 chars
            logger.info("LLM raw response preview:\n%s", text[:2000])

            # Guard: empty response
            if not text.strip():
                raise RuntimeError(
                    f"LLM returned an empty response (attempt {attempt}). "
                    "This usually means the model's output was blocked or the prompt was too short."
                )

            return text
        except Exception as exc:
            last_err = exc
            # Auth errors are fatal — don't retry
            err_type = type(exc).__name__
            if "Auth" in err_type or "auth" in str(exc).lower():
                raise RuntimeError(
                    f"DeepSeek 认证失败 — 请检查 .env 中的 DEEPSEEK_API_KEY 是否有效。\n"
                    f"当前使用的 Key 前缀: {_DEEPSEEK_API_KEY[:8]}…\n"
                    f"API 返回: {exc}"
                ) from exc
            logger.warning("LLM call failed (attempt %d/%d): %s",
                           attempt, _RETRY_MAX, exc)
            if attempt < _RETRY_MAX:
                time.sleep(_RETRY_BACKOFF ** attempt)

    raise RuntimeError(f"LLM call failed after {_RETRY_MAX} attempts: {last_err}")


def _parse_json_response(raw: str, label: str = "LLM") -> Any:
    """Robust JSON extraction from an LLM response.

    Handles the case where the model wraps JSON in fences despite instructions.
    """
    text = raw.strip()
    # Strip ```json … ``` fences if present
    if text.startswith("```"):
        # find the first newline and last ```
        first_nl = text.find("\n")
        last_fence = text.rfind("```")
        if first_nl != -1 and last_fence != -1:
            text = text[first_nl + 1 : last_fence].strip()
        else:
            text = text.strip("`").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("%s: direct parse failed; attempting repair.", label)
        # Try to find the outermost { ... } or [ ... ]
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"{label}: 无法解析 LLM 返回的 JSON。原始内容前 500 字符:\n{raw[:500]}")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 60  # reviews per LLM call — safe for token limits


def _batches(items: List[dict], size: int = BATCH_SIZE) -> List[List[dict]]:
    """Split *items* into equal-sized batches."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def _review_snapshot(r: dict) -> str:
    """One-line review representation for the LLM prompt."""
    return (
        f'[id={r.get("id","?")}] '
        f'★{r.get("rating","?")} | '
        f'{r.get("title","")} — '
        f'{(r.get("content","") or "")[:240]}'
    )


# ====================================================================
# Stage 1 — Topic clustering
# ====================================================================

_TOPIC_PROMPT = """Analyse the following App Store user reviews (each tagged with its review_id).

**Task:**
1. Read ALL reviews below carefully.
2. Dynamically discover 3-8 distinct topic/themes that emerge FROM the data.
   Do NOT use a pre-defined keyword list — let the data speak.
3. Assign each review to exactly ONE topic.
4. Give each topic a concise, descriptive label (2-5 words in English).

**CRITICAL — Response format:**
Your response MUST be pure, parseable JSON and NOTHING else.
Do NOT wrap the JSON in Markdown code fences (```json ... ```).
Do NOT add any introductory or concluding text — just the JSON object.

Return a JSON object with this EXACT shape (every field is required):
{{
  "topics": [
    {{
      "label": "string (topic name, 2-5 words)",
      "description": "string (one sentence explaining this topic)",
      "review_ids": ["id1", "id2", "..."]
    }}
  ],
  "unassigned_review_ids": ["idX", "..."]
}}

Reviews:
---
{reviews_text}
---"""


def analyze_topics(cleaned_file_path: str) -> pd.DataFrame:
    """Assign a dynamic topic label to every review using LLM clustering.

    Args:
        cleaned_file_path: Path to ``data/cleaned_reviews.json``.

    Returns:
        DataFrame with a ``topic`` column added.  Also writes
        ``data/reviews_with_topics.json``.
    """
    # ---- Load cleaned data ---------------------------------------------------
    cleaned_path = Path(cleaned_file_path).resolve()
    if not cleaned_path.is_file():
        raise FileNotFoundError(f"清洗后数据不存在: {cleaned_path}")

    with cleaned_path.open("r", encoding="utf-8") as fh:
        cleaned = json.load(fh)

    reviews: List[dict] = cleaned.get("reviews", [])
    if not reviews:
        logger.warning("No reviews to analyse — returning empty DataFrame.")
        return pd.DataFrame()

    logger.info("Stage 1 · Topic clustering: %d reviews in %d batch(es).",
                 len(reviews), len(_batches(reviews)))

    # ---- LLM: assign topics per batch ----------------------------------------
    all_topics: List[Dict[str, Any]] = []

    for i, batch in enumerate(_batches(reviews), 1):
        reviews_text = "\n".join(_review_snapshot(r) for r in batch)
        prompt = _TOPIC_PROMPT.format(reviews_text=reviews_text)

        logger.info("  Batch %d/%d → LLM topic discovery …", i, len(_batches(reviews)))
        raw = _call_llm(prompt, temperature=0.4, max_tokens=4096)
        result = _parse_json_response(raw, label=f"Topics batch {i}")

        batch_topics = result.get("topics", [])
        all_topics.extend(batch_topics)
        logger.info("  Batch %d/%d: %d topic(s) found.", i, len(_batches(reviews)), len(batch_topics))

    # ---- Merge topics across batches (same-label → merge) --------------------
    merged: Dict[str, List[str]] = {}       # label → [review_ids]
    topic_descriptions: Dict[str, str] = {}  # label → description

    for t in all_topics:
        label = t.get("label", "Other").strip()
        desc = t.get("description", "")
        ids = t.get("review_ids", [])
        if label not in merged:
            merged[label] = []
            topic_descriptions[label] = desc
        merged[label].extend(ids)

    # ---- Build review_id → topic lookup --------------------------------------
    id_to_topic: Dict[str, str] = {}
    for label, ids in merged.items():
        for rid in ids:
            if rid not in id_to_topic:
                id_to_topic[rid] = label

    # ---- Attach topic to each review -----------------------------------------
    df = pd.DataFrame(reviews)
    df["topic"] = df["id"].map(id_to_topic).fillna("Unclassified")

    # Summarise
    topic_counts = df["topic"].value_counts().to_dict()
    logger.info("Topic distribution: %s", topic_counts)

    # ---- Persist -------------------------------------------------------------
    output_path = cleaned_path.parent / "reviews_with_topics.json"
    output_path.write_text(
        json.dumps({
            "app_id": cleaned.get("app_id", ""),
            "total_reviews": len(df),
            "topics": [
                {
                    "label": label,
                    "description": topic_descriptions.get(label, ""),
                    "count": topic_counts.get(label, 0),
                }
                for label in topic_counts
            ],
            "reviews": df.to_dict(orient="records"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("→ reviews_with_topics.json saved (%d reviews, %d topics).",
                 len(df), len(topic_counts))

    return df


# ====================================================================
# Stage 2 — Evidence-backed insights per topic
# ====================================================================

_INSIGHT_PROMPT = """You are a senior product analyst.  Below are App Store reviews ALL belonging
to the topic **"{topic_label}"**.

**Task:**
1. Identify the 1-3 core pain-points / issues within this topic.
2. For EACH issue, provide:
   - ``issue``: a concise title (1 sentence).
   - ``severity``: "high" / "medium" / "low" based on frequency & rating.
   - ``summary``: a 2-3 sentence synthesis of the problem.
   - ``evidence``: a list of concrete ``review_id`` values that DIRECTLY support this issue.
   - ``sample_count``: number of reviews supporting this issue.
   - ``avg_rating``: average star rating of the supporting reviews.

**Return ONLY a JSON object**:
{{
  "topic": "{topic_label}",
  "issues": [
    {{
      "issue": "string",
      "severity": "high|medium|low",
      "summary": "string",
      "evidence": ["review_id_1", "review_id_2", ...],
      "sample_count": 42,
      "avg_rating": 2.3
    }}
  ]
}}

Reviews:
---
{reviews_text}
---"""


def generate_insights(reviews_with_topics_path: str) -> Dict[str, Any]:
    """Generate evidence-backed issue insights for each topic.

    Args:
        reviews_with_topics_path: Path to ``data/reviews_with_topics.json``.

    Returns:
        The full insights report dict.  Also writes ``data/insights.json``.
    """
    path = Path(reviews_with_topics_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"带主题数据不存在: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    reviews: List[dict] = data.get("reviews", [])
    topics: List[str] = sorted({r.get("topic", "Unclassified") for r in reviews})

    logger.info("Stage 2 · Insight generation: %d topics to analyse.", len(topics))

    all_topic_insights: List[Dict[str, Any]] = []

    for topic in topics:
        topic_reviews = [r for r in reviews if r.get("topic") == topic]
        logger.info("  Topic \"%s\": %d review(s) → LLM …", topic, len(topic_reviews))

        reviews_text = "\n".join(_review_snapshot(r) for r in topic_reviews)
        prompt = _INSIGHT_PROMPT.format(topic_label=topic, reviews_text=reviews_text)

        raw = _call_llm(prompt, temperature=0.3, max_tokens=4096)
        result = _parse_json_response(raw, label=f"Insight [{topic}]")

        all_topic_insights.append(result)
        logger.info("  Topic \"%s\": %d issue(s) found.", topic, len(result.get("issues", [])))

    # ---- Assemble report -----------------------------------------------------
    report: Dict[str, Any] = {
        "app_id": data.get("app_id", ""),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_topics": len(all_topic_insights),
        "topics": all_topic_insights,
    }

    output_path = path.parent / "insights.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("→ insights.json saved (%d topic reports).", len(all_topic_insights))

    return report


# ====================================================================
# Stage 3 — PRD + Test Cases with full traceability
# ====================================================================

_PRD_PROMPT = """You are a senior Product Manager AND QA Lead.

Below is an **issue insights report** extracted from real App Store user reviews.
Every issue includes ``evidence`` (review_ids) that prove it exists.

**Task — produce TWO structured outputs in one JSON:**

1. **PRD (Product Requirement Document):**
   - For each evidence-backed issue, draft 1-2 actionable product requirements.
   - Each requirement MUST include:
     - ``req_id``: "PRD-001", "PRD-002", ...
     - ``title``: short requirement name.
     - ``description``: what to build / change, informed by user feedback.
     - ``priority``: "P0" (blocker) / "P1" (high) / "P2" (nice-to-have).
     - ``source_topic``: the topic label this requirement originated from.
     - ``trace_review_ids``: the specific review_ids that justify this requirement
       (MUST come from the insight evidence lists below).

2. **Test Cases:**
   - For each PRD requirement, write 1-2 test cases to validate the fix.
   - Each test case MUST include:
     - ``tc_id``: "TC-001", "TC-002", ...
     - ``title``: what this test verifies.
     - ``steps``: numbered list of verification steps.
     - ``expected_result``: what should happen when the fix is correct.
     - ``linked_req_id``: the PRD requirement this test covers.
     - ``trace_review_ids``: the review_ids that inspired this test case.

**Return ONLY a JSON object:**
{{
  "prd": [
    {{
      "req_id": "PRD-001",
      "title": "string",
      "description": "string",
      "priority": "P0|P1|P2",
      "source_topic": "string",
      "trace_review_ids": ["id1", "id2"]
    }}
  ],
  "test_cases": [
    {{
      "tc_id": "TC-001",
      "title": "string",
      "steps": ["step 1", "step 2"],
      "expected_result": "string",
      "linked_req_id": "PRD-001",
      "trace_review_ids": ["id1"]
    }}
  ]
}}

Insights report:
---
{insights_json}
---"""


def generate_prd_and_tests(insights_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate traceable PRD and test cases from the insights report.

    Args:
        insights_path: Path to ``data/insights.json``.

    Returns:
        Tuple of ``(prd_data, test_cases_data)``.  Also writes
        ``data/prd.json`` and ``data/test_cases.json``.
    """
    path = Path(insights_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"洞察报告不存在: {path}")

    with path.open("r", encoding="utf-8") as fh:
        insights = json.load(fh)

    topics = insights.get("topics", [])
    if not topics:
        logger.warning("No insights to process — returning empty PRD.")
        return {}, {}

    total_issues = sum(len(t.get("issues", [])) for t in topics)
    logger.info("Stage 3 · PRD & Tests: %d topic(s), %d issue(s) → LLM …",
                 len(topics), total_issues)

    # ---- LLM: generate PRD + tests in one shot --------------------------------
    insights_json = json.dumps(insights, ensure_ascii=False, indent=2)
    prompt = _PRD_PROMPT.format(insights_json=insights_json)

    raw = _call_llm(prompt, temperature=0.3, max_tokens=8192)
    result = _parse_json_response(raw, label="PRD & Tests")

    prd = result.get("prd", [])
    test_cases = result.get("test_cases", [])

    logger.info("  PRD: %d requirement(s) generated.", len(prd))
    logger.info("  Test Cases: %d case(s) generated.", len(test_cases))

    # ---- Persist -------------------------------------------------------------
    parent = path.parent
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()

    prd_doc = {
        "app_id": insights.get("app_id", ""),
        "generated_at": now_iso,
        "total_requirements": len(prd),
        "requirements": prd,
    }
    (parent / "prd.json").write_text(
        json.dumps(prd_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("→ prd.json saved.")

    tc_doc = {
        "app_id": insights.get("app_id", ""),
        "generated_at": now_iso,
        "total_test_cases": len(test_cases),
        "test_cases": test_cases,
    }
    (parent / "test_cases.json").write_text(
        json.dumps(tc_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("→ test_cases.json saved.")

    return prd_doc, tc_doc


# ====================================================================
# Full pipeline runner
# ====================================================================

def run_full_pipeline(cleaned_file_path: str) -> Dict[str, Any]:
    """Convenience: run all three stages sequentially.

    Returns a summary dict with paths and counts.
    """
    logger.info("=" * 60)
    logger.info("AI Pipeline START")
    logger.info("=" * 60)

    # Stage 1
    df_topics = analyze_topics(cleaned_file_path)
    topics_path = str(Path(cleaned_file_path).resolve().parent / "reviews_with_topics.json")

    # Stage 2
    insights = generate_insights(topics_path)
    insights_path = str(Path(cleaned_file_path).resolve().parent / "insights.json")

    # Stage 3
    prd, tests = generate_prd_and_tests(insights_path)

    logger.info("=" * 60)
    logger.info("AI Pipeline DONE")
    logger.info("=" * 60)

    return {
        "reviews_with_topics": topics_path,
        "topics_count": df_topics["topic"].nunique() if len(df_topics) > 0 else 0,
        "insights": insights_path,
        "issues_found": sum(len(t.get("issues", [])) for t in insights.get("topics", [])),
        "prd": str(Path(cleaned_file_path).resolve().parent / "prd.json"),
        "prd_requirements": prd.get("total_requirements", 0),
        "test_cases": str(Path(cleaned_file_path).resolve().parent / "test_cases.json"),
        "test_cases_count": tests.get("total_test_cases", 0),
    }
