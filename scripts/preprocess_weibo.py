#!/usr/bin/env python3
"""Clean and reshape the scraped Weibo dataset."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_POSTS = PROJECT_ROOT / "data" / "raw" / "weibo_quanyunhui_multi_http.csv"
RAW_META = PROJECT_ROOT / "data" / "raw" / "weibo_quanyunhui_multi_http.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MONTH_DAY_RE = re.compile(
    r"(?P<month>\d{1,2})月(?P<day>\d{1,2})日(?:\s*(?P<time>\d{1,2}:\d{2}))?"
)
RELATIVE_MIN_RE = re.compile(r"^(?P<minutes>\d+)分钟前")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", type=Path, default=RAW_POSTS, help="Raw CSV path")
    parser.add_argument("--meta", type=Path, default=RAW_META, help="Metadata JSON path")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for processed artifacts",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_string(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_clock(clock: str) -> tuple[int, int, int]:
    match = re.match(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?", clock)
    if not match:
        return 0, 0, 0
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second") or 0)
    return hour, minute, second


def combine_date_and_clock(date_source: datetime, clock: str, day_offset: int = 0) -> datetime:
    hour, minute, second = parse_clock(clock)
    base_date = (date_source + timedelta(days=day_offset)).date()
    return datetime.combine(base_date, datetime.min.time()).replace(hour=hour, minute=minute, second=second)


def parse_publish_time(raw_value: Any, base_ts: datetime) -> Optional[datetime]:
    text = sanitize_string(raw_value)
    if not text:
        return None

    if text.startswith("今天"):
        return combine_date_and_clock(base_ts, text.replace("今天", "", 1) or "00:00")
    if text.startswith("昨天"):
        return combine_date_and_clock(base_ts, text.replace("昨天", "", 1) or "00:00", day_offset=-1)

    rel_match = RELATIVE_MIN_RE.match(text.split(" ", 1)[0])
    if rel_match:
        minutes = int(rel_match.group("minutes"))
        return base_ts - timedelta(minutes=minutes)

    month_match = MONTH_DAY_RE.match(text.replace(" ", ""))
    if month_match:
        month = int(month_match.group("month"))
        day = int(month_match.group("day"))
        clock = month_match.group("time") or "00:00"
        year = base_ts.year
        if month > base_ts.month + 1 and base_ts.month <= 2:
            year -= 1
        try:
            hour, minute, second = parse_clock(clock)
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None

    try:
        return pd.to_datetime(text).to_pydatetime()
    except (TypeError, ValueError):
        return None


def normalize_url(raw: Any) -> Optional[str]:
    text = sanitize_string(raw)
    if not text:
        return None
    if text.startswith("//"):
        return f"https:{text}"
    if text.startswith("/u/"):
        return f"https://weibo.com{text}"
    if text.startswith("/profile") or text.startswith("/p/"):
        return f"https://weibo.com{text}"
    return text


def try_load_json(raw: Any) -> List[Dict[str, Any]]:
    text = sanitize_string(raw)
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        return []
    return []


def prepare_posts(df: pd.DataFrame, base_ts: datetime) -> pd.DataFrame:
    df = df.rename(columns=lambda col: sanitize_string(col))
    df["content"] = df["content"].fillna("").apply(lambda x: sanitize_string(x).replace("\u200b", ""))
    df["topic"] = df["topic"].astype(str).str.strip()
    df["detail_url"] = df["detail_url"].apply(normalize_url)
    df["author_url"] = df["author_url"].apply(normalize_url)
    df["publish_time_dt"] = df["publish_time"].apply(lambda val: parse_publish_time(val, base_ts))
    df["publish_time"] = pd.to_datetime(df["publish_time_dt"])
    df["publish_date"] = df["publish_time"]
    df["publish_date"] = df["publish_date"].dt.date
    df["text_length"] = df["content"].str.len()
    numeric_cols = ["retweet_count", "review_count", "like_count", "fetched_review_total"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["publish_time", "weibo_id"], ascending=[False, False]).drop_duplicates("weibo_id")
    return df


def flatten_comments(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in df.itertuples():
        comments = try_load_json(getattr(row, "comments", ""))
        if not comments:
            continue
        for idx, comment in enumerate(comments):
            review_time = sanitize_string(comment.get("review_time"))
            try:
                parsed_time = pd.to_datetime(review_time) if review_time else None
            except (TypeError, ValueError):
                parsed_time = None
            records.append(
                {
                    "weibo_id": row.weibo_id,
                    "topic": row.topic,
                    "comment_index": idx,
                    "review_id": sanitize_string(comment.get("review_id")) or None,
                    "review_time_raw": review_time or None,
                    "review_time": parsed_time,
                    "review_like": int(comment.get("review_like", 0) or 0),
                    "review_loc": sanitize_string(comment.get("review_loc")) or None,
                    "review_content": sanitize_string(comment.get("review_content")),
                    "review_length": len(sanitize_string(comment.get("review_content"))),
                }
            )
    return pd.DataFrame.from_records(records)


def flatten_reviewers(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in df.itertuples():
        reviewers = try_load_json(getattr(row, "reviewers", ""))
        if not reviewers:
            continue
        for reviewer in reviewers:
            records.append(
                {
                    "weibo_id": row.weibo_id,
                    "topic": row.topic,
                    "reviewer_id": reviewer.get("reviewer_id"),
                    "reviewer_name": sanitize_string(reviewer.get("reviewer_name")) or None,
                    "reviewer_url": normalize_url(reviewer.get("reviewer_url")),
                    "reviewer_loc": sanitize_string(reviewer.get("reviewer_loc")) or None,
                    "reviewer_followers": int(reviewer.get("reviewer_followers", 0) or 0),
                    "reviewer_following": int(reviewer.get("reviewer_following", 0) or 0),
                    "reviewer_weibo_count": int(reviewer.get("reviewer_weibo_count", 0) or 0),
                    "reviewer_description": sanitize_string(reviewer.get("reviewer_description")) or None,
                    "reviewer_gender": sanitize_string(reviewer.get("reviewer_gender")) or None,
                }
            )
    df_out = pd.DataFrame.from_records(records)
    if df_out.empty:
        return df_out
    df_out = df_out.drop_duplicates(["weibo_id", "reviewer_id", "reviewer_name"])
    return df_out


def write_outputs(posts: pd.DataFrame, comments: pd.DataFrame, reviewers: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    posts_path = out_dir / "weibo_posts_clean.csv"
    comments_path = out_dir / "weibo_comments_flat.csv"
    reviewers_path = out_dir / "weibo_reviewers_flat.csv"

    posts.to_csv(posts_path, index=False)
    if not comments.empty:
        comments.to_csv(comments_path, index=False)
    if not reviewers.empty:
        reviewers.to_csv(reviewers_path, index=False)

    publish_times = posts["publish_time"].dropna()
    summary = {
        "posts_rows": int(len(posts)),
        "comments_rows": int(len(comments)),
        "reviewers_rows": int(len(reviewers)),
        "posts_with_comments": int(comments["weibo_id"].nunique()) if not comments.empty else 0,
        "topics": sorted(posts["topic"].unique().tolist()),
        "time_range": {
            "min": publish_times.min().isoformat() if not publish_times.empty else None,
            "max": publish_times.max().isoformat() if not publish_times.empty else None,
        },
        "paths": {
            "posts": str(posts_path),
            "comments": str(comments_path) if not comments.empty else None,
            "reviewers": str(reviewers_path) if not reviewers.empty else None,
        },
    }
    with (out_dir / "preprocess_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.meta)
    base_timestamp = datetime.fromisoformat(metadata["generated_at"])
    raw_posts = pd.read_csv(args.posts)
    posts = prepare_posts(raw_posts, base_timestamp)
    comments = flatten_comments(posts)
    reviewers = flatten_reviewers(posts)
    summary = write_outputs(posts, comments, reviewers, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
