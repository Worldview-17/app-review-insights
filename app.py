"""
App Review Insights — FastAPI Backend
=====================================
Fetches App Store user reviews, persists them locally, and exposes
endpoints for collection and analysis.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from ai_analyzer import run_full_pipeline
from data_processor import clean_reviews
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Path as FPath

# ---------------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------------

# Always load .env from the project root — independent of CWD.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

DATA_DIR: Path = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_REVIEWS_PATH: Path = DATA_DIR / "raw_reviews.json"

# iTunes RSS feed template:
#   {country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/page={page}/json
ITUNES_RSS_BASE: str = "https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/page={page}/json"

REQUEST_TIMEOUT: int = 30  # seconds
MAX_PAGES: int = 10        # RSS feed max (Apple serves up to 10 pages)
PAGE_SIZE: int = 50        # typical entries per RSS page

USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("mcp_services.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("AppReviewInsights")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="App Review Insights",
    description="Fetch & analyse App Store user reviews to generate product insights.",
    version="0.1.0",
)

# Allow the frontend (served from same origin or localhost) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers — URL parsing
# ---------------------------------------------------------------------------

# Matches both modern (apps.apple.com) and legacy (itunes.apple.com) URLs.
_APP_ID_RE = re.compile(r"/id(\d{6,12})")
_COUNTRY_RE = re.compile(r"apple\.com/([a-z]{2})/")


def extract_app_id(url: str) -> str:
    """Extract the numeric App Store ID from an Apple app URL.

    Args:
        url: e.g. ``https://apps.apple.com/us/app/.../id839285684``

    Returns:
        The app ID string, e.g. ``"839285684"``.

    Raises:
        ValueError: If no valid app ID is found in the URL.
    """
    match = _APP_ID_RE.search(url)
    if not match:
        raise ValueError(
            f"无法从 URL 中提取 App ID。"
            f" 期望格式: https://apps.apple.com/xx/app/.../idXXXXXXXXXX\n"
            f" 收到的 URL: {url}"
        )
    return match.group(1)


def extract_country(url: str) -> str:
    """Extract the two-letter country code from an Apple app URL.

    Falls back to ``"us"`` if extraction fails.
    """
    match = _COUNTRY_RE.search(url)
    return match.group(1) if match else "us"


# ---------------------------------------------------------------------------
# Review fetcher — iTunes RSS Feed (official, no scraping)
# ---------------------------------------------------------------------------


def _parse_rss_entry(entry: dict) -> dict:
    """Normalise a single RSS feed entry into a flat review record."""
    return {
        "id": entry.get("id", {}).get("label", ""),
        "author": entry.get("author", {}).get("name", {}).get("label", "Anonymous"),
        "title": entry.get("title", {}).get("label", ""),
        "content": entry.get("content", {}).get("label", ""),
        "rating": int(entry.get("im:rating", {}).get("label", 0)),
        "version": entry.get("im:version", {}).get("label", ""),
        "vote_count": int(entry.get("im:voteCount", {}).get("label", 0)),
        "vote_sum": int(entry.get("im:voteSum", {}).get("label", 0)),
        "updated": entry.get("updated", {}).get("label", ""),
    }


def fetch_reviews_from_rss(
    app_id: str,
    country: str = "us",
    max_pages: int = MAX_PAGES,
) -> Dict[str, Any]:
    """Fetch App Store reviews via Apple's official RSS feed.

    This is the **public, documented** iTunes RSS endpoint — it is not
    scraping.  Each request returns one page of reviews in JSON format.

    Args:
        app_id:   Numeric App Store ID.
        country:  Two-letter ISO country code (default ``"us"``).
        max_pages: How many pages to fetch (Apple caps at 10).

    Returns:
        A dict with keys:
        - ``app_id``
        - ``country``
        - ``total_reviews``
        - ``fetched_at`` (ISO-8601 UTC)
        - ``reviews`` (list of normalised review dicts)
        - ``errors`` (list of per-page error messages, if any)

    Notes:
        **Limitations of the RSS feed approach:**

        - Each page returns at most 50 reviews; Apple exposes at most 10
          pages → **~500 reviews max** per fetch.
        - Only the **most recent** reviews are returned; there is no
          pagination into historical data.
        - The feed does not include metadata like developer response,
          country-specific store region of the reviewer, or device info.
        - Rate-limiting: Apple may throttle or return 403 if you fetch
          too aggressively.  We add a 1 s delay between pages as a courtesy.

        For a production-grade solution, consider ``app-store-scraper``
        (PyPI) which wraps the Node.js library and can fetch deeper
        history, or use a third-party service like AppFollow / Appbot.
    """
    all_reviews: List[dict] = []
    errors: List[str] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for page in range(1, max_pages + 1):
        url = ITUNES_RSS_BASE.format(country=country, app_id=app_id, page=page)
        logger.info("Fetching reviews page %d/%d: %s", page, max_pages, url)

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            msg = f"Page {page}: request timed out after {REQUEST_TIMEOUT}s"
            logger.warning(msg)
            errors.append(msg)
            continue
        except requests.exceptions.HTTPError as exc:
            # 400 = no more pages / invalid page; stop fetching.
            if resp.status_code == 400:
                logger.info("Page %d returned 400 — no more pages available.", page)
                break
            msg = f"Page {page}: HTTP {resp.status_code} — {exc}"
            logger.warning(msg)
            errors.append(msg)
            continue
        except requests.exceptions.RequestException as exc:
            msg = f"Page {page}: request failed — {exc}"
            logger.warning(msg)
            errors.append(msg)
            continue
        except json.JSONDecodeError as exc:
            msg = f"Page {page}: invalid JSON response — {exc}"
            logger.warning(msg)
            errors.append(msg)
            continue

        # The RSS feed wraps entries inside a "feed" → "entry" list.
        feed = data.get("feed", {})
        entries = feed.get("entry", [])

        if not entries:
            logger.info("Page %d returned 0 entries — stopping.", page)
            break

        for entry in entries:
            all_reviews.append(_parse_rss_entry(entry))

        logger.info("Page %d: collected %d review(s).", page, len(entries))

        # Be a good citizen — don't hammer Apple's servers.
        time.sleep(1.0)

    session.close()

    result: Dict[str, Any] = {
        "app_id": app_id,
        "country": country,
        "total_reviews": len(all_reviews),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "reviews": all_reviews,
    }
    if errors:
        result["errors"] = errors

    logger.info(
        "Finished fetching: %d review(s) across %d page(s) for app %s (%s).",
        len(all_reviews), max_pages, app_id, country,
    )
    return result


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/collect_reviews")
async def collect_reviews(request: Request):
    """Fetch App Store reviews and persist them to ``data/raw_reviews.json``.

    **Request body** (JSON):

    .. code-block:: json

        {
            "app_store_url": "https://apps.apple.com/us/app/.../id839285684"
        }

    **Returns**:

    .. code-block:: json

        {
            "success": true,
            "message": "成功获取 287 条评论，已保存至 data/raw_reviews.json",
            "file_path": "data/raw_reviews.json",
            "review_count": 287
        }
    """
    # 1. Parse the incoming JSON body -----------------------------------------
    try:
        body: dict = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="请求体不是有效的 JSON。")

    app_store_url: Optional[str] = body.get("app_store_url", "").strip() if body else ""
    if not app_store_url:
        raise HTTPException(status_code=400, detail="缺少必填字段: app_store_url")

    logger.info("Collecting reviews for URL: %s", app_store_url)

    # 2. Parse URL ------------------------------------------------------------
    try:
        app_id = extract_app_id(app_store_url)
        country = extract_country(app_store_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info("Parsed: app_id=%s  country=%s", app_id, country)

    # 3. Fetch reviews --------------------------------------------------------
    try:
        data = fetch_reviews_from_rss(app_id=app_id, country=country)
    except Exception as exc:
        logger.exception("Unexpected error while fetching reviews.")
        raise HTTPException(status_code=500, detail=f"获取评论时发生异常: {exc}")

    # 4. Persist to disk ------------------------------------------------------
    try:
        RAW_REVIEWS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Reviews saved to %s", RAW_REVIEWS_PATH)
    except OSError as exc:
        logger.exception("Failed to write reviews file.")
        raise HTTPException(status_code=500, detail=f"保存文件失败: {exc}")

    # 4.5  Clean & preprocess ---------------------------------------------------
    cleaned_count: int = 0
    dupes_removed: int = 0
    junk_removed: int = 0
    try:
        df = clean_reviews(str(RAW_REVIEWS_PATH))
        cleaned_count = len(df)
        # Read back the cleaning metadata we just wrote
        cleaned_path = DATA_DIR / "cleaned_reviews.json"
        if cleaned_path.exists():
            with cleaned_path.open("r", encoding="utf-8") as fh:
                cleaned_data = json.load(fh)
                dd = cleaned_data.get("dedup", {})
                dupes_removed = dd.get("duplicates_removed", 0)
                junk_removed = dd.get("junk_removed", 0)
        logger.info(
            "Cleaning complete: %d reviews, %d dupes removed, %d junk removed.",
            cleaned_count, dupes_removed, junk_removed,
        )
    except Exception as exc:
        logger.warning("Cleaning step failed (non-fatal): %s", exc)

    # 4.6  AI analysis pipeline --------------------------------------------------
    ai_summary: Dict[str, Any] = {}
    if cleaned_count > 0:
        try:
            ai_summary = run_full_pipeline(str(DATA_DIR / "cleaned_reviews.json"))
            logger.info("AI pipeline complete: %s", ai_summary)
        except Exception as exc:
            logger.warning("AI pipeline failed (non-fatal): %s", exc)
    else:
        logger.info("Skipping AI pipeline — no reviews after cleaning.")

    # 5. Build response -------------------------------------------------------
    review_count: int = data.get("total_reviews", 0)
    errors: list = data.get("errors", [])

    message = f"成功获取 {review_count} 条评论，已保存至 {RAW_REVIEWS_PATH.name}"
    if errors:
        message += f" （{len(errors)} 个非致命错误）"

    return {
        "success": True,
        "message": message,
        "file_path": str(RAW_REVIEWS_PATH.relative_to(Path.cwd())),
        "review_count": review_count,
        "cleaned_count": cleaned_count,
        "duplicates_removed": dupes_removed,
        "junk_removed": junk_removed,
        "ai_pipeline": ai_summary if ai_summary else None,
        "errors": errors if errors else None,
    }


@app.post("/api/analyze")
async def analyze(request: Request):
    """Alias for ``/api/collect_reviews`` — preferred frontend endpoint."""
    return await collect_reviews(request)


@app.get("/api/data/{filename}")
async def get_data_file(filename: str):
    """Serve a JSON file from the ``data/`` directory.

    Allowed files: ``insights``, ``reviews_with_topics``, ``cleaned_reviews``,
    ``prd``, ``test_cases`` (without the ``.json`` extension, or with it).
    """
    # Sanitize — only allow known data files (prevent path traversal)
    ALLOWED = {
        "insights", "reviews_with_topics", "cleaned_reviews",
        "prd", "test_cases", "raw_reviews",
    }
    stem = filename.removesuffix(".json")
    if stem not in ALLOWED:
        raise HTTPException(status_code=404, detail=f"Unknown data file: {stem}")

    file_path = DATA_DIR / f"{stem}.json"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {stem}.json")

    try:
        content = json.loads(file_path.read_text(encoding="utf-8"))
        return content
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Data file is corrupted.")


@app.get("/api/summary")
async def get_summary():
    """Return a lightweight summary for the dashboard overview.

    Reads whatever data files exist and returns aggregate stats.
    """
    summary: Dict[str, Any] = {"reviews": None, "insights": None, "prd": None}

    cleaned_path = DATA_DIR / "cleaned_reviews.json"
    if cleaned_path.is_file():
        try:
            data = json.loads(cleaned_path.read_text(encoding="utf-8"))
            reviews = data.get("reviews", [])
            ratings = [r.get("rating", 0) for r in reviews if r.get("rating")]
            summary["reviews"] = {
                "total": len(reviews),
                "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0,
                "positive": sum(1 for r in ratings if r >= 4),
                "negative": sum(1 for r in ratings if r <= 2),
                "neutral": sum(1 for r in ratings if r == 3),
            }
        except Exception:
            pass

    insights_path = DATA_DIR / "insights.json"
    if insights_path.is_file():
        try:
            data = json.loads(insights_path.read_text(encoding="utf-8"))
            topics = data.get("topics", [])
            all_issues = []
            for t in topics:
                all_issues.extend(t.get("issues", []))
            summary["insights"] = {
                "total_topics": len(topics),
                "total_issues": len(all_issues),
                "high_severity": sum(1 for i in all_issues if i.get("severity") == "high"),
                "topics": [t.get("topic", t.get("label", "?")) for t in topics],
            }
        except Exception:
            pass

    prd_path = DATA_DIR / "prd.json"
    if prd_path.is_file():
        try:
            data = json.loads(prd_path.read_text(encoding="utf-8"))
            summary["prd"] = {
                "total_requirements": data.get("total_requirements", 0),
                "p0_count": sum(
                    1 for r in data.get("requirements", []) if r.get("priority") == "P0"
                ),
            }
        except Exception:
            pass

    return summary


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).resolve().parent / "ui"
app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting App Review Insights server...")
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
