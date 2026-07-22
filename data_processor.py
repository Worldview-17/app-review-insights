"""
Data Cleaning & Preprocessing Module
=====================================
Handles deduplication, structuring, and cleaning of raw App Store review data.

Usage:
    from data_processor import clean_reviews
    df = clean_reviews("data/raw_reviews.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum meaningful content length (characters, after stripping whitespace)
MIN_CONTENT_LENGTH: int = 3

# Source file name and output file name
RAW_FILE_NAME: str = "raw_reviews.json"
CLEANED_FILE_NAME: str = "cleaned_reviews.json"

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

# Broad emoji / symbol ranges — covers the most common emoji blocks
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # Miscellaneous Symbols, Emoticons, Supplemental
    "\U00002600-\U000027BF"   # Miscellaneous Symbols
    "\U0001FA00-\U0001FAFF"   # Chess Symbols, Symbols Extended-A
    "\U0001F600-\U0001F64F"   # Emoticons
    "]"
)


def _content_hash(content: str) -> str:
    """MD5 hex digest of the stripped content — used for deduplication."""
    return hashlib.md5(content.strip().encode("utf-8")).hexdigest()


def _is_junk(content: str) -> bool:
    """Return ``True`` if *content* is too short, emoji-only, or meaningless.

    A review is considered junk when, after stripping emoji, whitespace, and
    punctuation, fewer than *MIN_CONTENT_LENGTH* word characters remain.
    """
    text = content.strip()
    if len(text) < MIN_CONTENT_LENGTH:
        return True

    # Strip emoji, then strip everything that isn't a letter/digit
    no_emoji = _EMOJI_RE.sub("", text)
    alphanum_only = re.sub(r"[^A-Za-z0-9一-鿿]", "", no_emoji)

    return len(alphanum_only) < MIN_CONTENT_LENGTH


# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------


def clean_reviews(raw_file_path: str) -> pd.DataFrame:
    """Load raw reviews, deduplicate, structure, clean, and save the result.

    Args:
        raw_file_path: Absolute or relative path to ``raw_reviews.json``.

    Returns:
        A pandas DataFrame with the cleaned reviews (columns: ``id``,
        ``user_name``, ``rating``, ``title``, ``content``, ``date``,
        ``version``).

    Side effects:
        Writes ``data/cleaned_reviews.json`` next to the raw file.

    Raises:
        FileNotFoundError: If the raw file does not exist.
        ValueError: If the raw file is not valid JSON or missing ``reviews``.
    """
    # ---- 1. Load ----------------------------------------------------------------
    raw_path = Path(raw_file_path).resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"原始数据文件不存在: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    raw_reviews: List[dict] = raw.get("reviews", [])
    logger.info("加载 %d 条原始评论 ← %s", len(raw_reviews), raw_path)

    if not raw_reviews:
        logger.warning("原始数据中无评论记录，返回空 DataFrame。")
        return pd.DataFrame()

    # ---- 2. Deduplicate ---------------------------------------------------------
    seen_ids: set = set()
    seen_hashes: set = set()
    unique: List[dict] = []

    for r in raw_reviews:
        rid = r.get("id", "")
        chash = _content_hash(r.get("content", ""))

        if rid and rid in seen_ids:
            continue
        if chash in seen_hashes:
            continue

        if rid:
            seen_ids.add(rid)
        seen_hashes.add(chash)
        unique.append(r)

    dupes = len(raw_reviews) - len(unique)
    logger.info("去重: 移除 %d 条重复，剩余 %d 条。", dupes, len(unique))

    # ---- 3. Structure -----------------------------------------------------------
    records: List[Dict[str, Any]] = []
    for r in unique:
        records.append({
            "id": r.get("id", ""),
            "user_name": r.get("author", "Anonymous"),
            "rating":   r.get("rating", 0),
            "title":    r.get("title", ""),
            "content":  r.get("content", ""),
            "date":     r.get("updated", ""),
            "version":  r.get("version", ""),
        })

    df = pd.DataFrame(records)

    # ---- 4. Filter junk content -------------------------------------------------
    before = len(df)
    if "content" in df.columns:
        df = df[~df["content"].apply(_is_junk)]
    junk_removed = before - len(df)
    logger.info("内容筛选: 移除 %d 条无意义评论。", junk_removed)

    # ---- 5. Handle missing / invalid values -------------------------------------
    df["user_name"] = df["user_name"].fillna("Anonymous").replace("", "Anonymous")
    df["title"]     = df["title"].fillna("")
    df["content"]   = df["content"].fillna("")
    df["date"]      = df["date"].fillna("")
    df["id"] = df["id"].fillna("")

    # Force rating to int 1-5; coerce junk to 3 (neutral) as a safe default
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int).clip(1, 5)

    logger.info("缺失值处理完成。最终清洗后评论数: %d。", len(df))

    # ---- 6. Persist -------------------------------------------------------------
    output_path = raw_path.parent / CLEANED_FILE_NAME

    payload: Dict[str, Any] = {
        "app_id":       raw.get("app_id", ""),
        "country":      raw.get("country", ""),
        "total_reviews": len(df),
        "fetched_at":   raw.get("fetched_at", ""),
        "cleaned_at":   pd.Timestamp.now(tz="UTC").isoformat(),
        "dedup": {
            "original_count":    len(raw_reviews),
            "duplicates_removed": dupes,
            "junk_removed":      junk_removed,
        },
        "reviews": df.to_dict(orient="records"),
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("清洗后数据已保存 → %s", output_path)

    return df


# ---------------------------------------------------------------------------
# Convenience: auto-locate raw file inside the project's data/ directory
# ---------------------------------------------------------------------------

def clean_reviews_auto(data_dir: str | Path = "data") -> pd.DataFrame:
    """Same as :func:`clean_reviews` but infers the path to
    ``data/raw_reviews.json`` from the project root (where *this* module
    lives)."""
    project_root = Path(__file__).resolve().parent
    raw_path = project_root / data_dir / RAW_FILE_NAME
    return clean_reviews(str(raw_path))
