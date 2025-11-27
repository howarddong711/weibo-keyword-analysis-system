import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

try:
    import jieba  # type: ignore
except ImportError:  # Segmentation is optional; fall back to whitespace split
    jieba = None  # pragma: no cover

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from matplotlib import font_manager

FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "NotoSerifCJKsc-Black.otf"
FONT_PROP = None
if FONT_PATH.exists():
    font_manager.fontManager.addfont(FONT_PATH)
    FONT_PROP = font_manager.FontProperties(fname=str(FONT_PATH))
    FONT_NAME = FONT_PROP.get_name()
    WORDCLOUD_FONT_PATH = str(FONT_PATH)
else:
    FONT_NAME = "SimHei"
    WORDCLOUD_FONT_PATH = None

matplotlib.rcParams["font.family"] = FONT_NAME
matplotlib.rcParams["font.sans-serif"] = [FONT_NAME, "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid")
STOPWORDS = {
    "全运会",
    "真的",
    "就是",
    "我们",
    "他们",
    "一个",
    "这个",
    "那个",
    "还是",
    "可以",
    "自己",
    "已经",
    "以及",
    "因为",
    "所以",
    "如果",
    "觉得",
    "时候",
}


def apply_font(ax: plt.Axes) -> None:
    if FONT_PROP is None:
        return
    ax.title.set_fontproperties(FONT_PROP)
    ax.xaxis.label.set_fontproperties(FONT_PROP)
    ax.yaxis.label.set_fontproperties(FONT_PROP)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontproperties(FONT_PROP)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_title().set_fontproperties(FONT_PROP)
        for text in legend.get_texts():
            text.set_fontproperties(FONT_PROP)


def load_dataframe(path: Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    parse_dates = parse_dates or []
    df = pd.read_csv(path, parse_dates=list(parse_dates))
    return df


def ensure_label_order(labels: Iterable[str]) -> list:
    order = ["positive", "neutral", "negative", "unknown"]
    existing = [label for label in order if label in labels]
    for label in labels:
        if label not in existing:
            existing.append(label)
    return existing


def plot_sentiment_distribution(df: pd.DataFrame, label_col: str, title: str, out_path: Path) -> Dict[str, int]:
    counts = df[label_col].value_counts(dropna=False).to_dict()
    order = ensure_label_order(counts.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), order=order, palette="Set2", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Sentiment label")
    ax.set_ylabel("Count")
    text_kwargs = {"fontproperties": FONT_PROP} if FONT_PROP else {}
    for idx, label in enumerate(order):
        value = counts.get(label)
        if value:
            ax.text(
                idx,
                value + max(counts.values()) * 0.01,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=10,
                **text_kwargs,
            )
    apply_font(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return counts


def plot_topic_sentiment(df: pd.DataFrame, title: str, out_path: Path) -> Dict[str, Dict[str, int]]:
    grouped = df.groupby(["topic", "sentiment_label"]).size().unstack(fill_value=0)
    order = ensure_label_order(grouped.columns)
    grouped = grouped[order]
    grouped.sort_index(inplace=True)
    ax = grouped.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20c")
    ax.set_title(title)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Posts")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_font(ax)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=200)
    plt.close(ax.figure)
    return grouped.to_dict()


def plot_hourly_trend(df: pd.DataFrame, datetime_col: str, label_col: str, title: str, out_path: Path) -> Dict[str, Dict[str, int]]:
    if datetime_col not in df:
        return {}
    temp = df.copy()
    temp["hour"] = temp[datetime_col].dt.floor("H")
    trend = temp.groupby(["hour", label_col]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=trend, x="hour", y="count", hue=label_col, marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Volume")
    ax.tick_params(axis="x", rotation=45)
    apply_font(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    pivot = trend.pivot(index="hour", columns=label_col, values="count").fillna(0)
    return {str(idx): row.dropna().to_dict() for idx, row in pivot.iterrows()}


def plot_engagement(df: pd.DataFrame, metrics: Iterable[str], label_col: str, title: str, out_path: Path) -> Dict[str, Dict[str, float]]:
    available = [col for col in metrics if col in df]
    if not available:
        return {}
    agg = df.groupby(label_col)[available].mean().round(2)
    melted = agg.reset_index().melt(id_vars=label_col, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=melted, x="metric", y="value", hue=label_col, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Engagement metric")
    ax.set_ylabel("Average value")
    apply_font(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return agg.to_dict()


def tokenize(text: str) -> Iterable[str]:
    text = str(text)
    if jieba:
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    return [token for token in text.split() if token]


def extract_keywords(df: pd.DataFrame, text_col: str, label_col: str, label: str, top_n: int = 20) -> Dict[str, int]:
    subset = df[df[label_col] == label]
    counter: Counter = Counter()
    for text in subset[text_col].dropna():
        tokens = tokenize(text)
        filtered = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
        counter.update(filtered)
    return dict(counter.most_common(top_n))


def plot_keyword_bars(keywords: Dict[str, int], title: str, out_path: Path) -> None:
    if not keywords:
        return
    items = list(keywords.items())
    tokens, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(8, max(4, len(tokens) * 0.3)))
    sns.barplot(x=list(counts), y=list(tokens), orient="h", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Token")
    apply_font(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_wordcloud_from_freq(keywords: Dict[str, int], title: str, out_path: Path) -> bool:
    if not keywords:
        return False
    wc_kwargs = {
        "width": 900,
        "height": 500,
        "background_color": "white",
        "collocations": False,
    }
    if WORDCLOUD_FONT_PATH:
        wc_kwargs["font_path"] = WORDCLOUD_FONT_PATH
    wc = WordCloud(**wc_kwargs).generate_from_frequencies(keywords)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title)
    ax.axis("off")
    if FONT_PROP:
        ax.title.set_fontproperties(FONT_PROP)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sentiment visualizations and summaries.")
    parser.add_argument("--posts", default=DATA_DIR / "weibo_posts_sentiment.csv", type=Path, help="Path to post-level sentiment CSV")
    parser.add_argument("--comments", default=DATA_DIR / "weibo_comments_sentiment.csv", type=Path, help="Path to comment-level sentiment CSV")
    parser.add_argument("--fig-dir", default=FIG_DIR, type=Path, help="Directory to store generated figures")
    parser.add_argument("--summary", default=PROJECT_ROOT / "reports" / "visualization_summary.json", type=Path, help="Path to write visualization summary JSON")
    parser.add_argument("--top-n", type=int, default=15, help="Top-N keywords per sentiment label")
    parser.add_argument("--text-col", default="content", help="Column name containing text for keyword extraction")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    posts = load_dataframe(args.posts, parse_dates=["publish_time_dt"])
    comments = load_dataframe(args.comments)

    summary = {
        "paths": {
            "posts": str(args.posts.relative_to(PROJECT_ROOT)),
            "comments": str(args.comments.relative_to(PROJECT_ROOT)),
            "figures": str(args.fig_dir.relative_to(PROJECT_ROOT)),
        }
    }

    summary["posts_sentiment_counts"] = plot_sentiment_distribution(
        posts,
        "sentiment_label",
        "Posts sentiment distribution",
        args.fig_dir / "posts_sentiment_distribution.png",
    )

    summary["comments_sentiment_counts"] = plot_sentiment_distribution(
        comments,
        "sentiment_label",
        "Comments sentiment distribution",
        args.fig_dir / "comments_sentiment_distribution.png",
    )

    summary["topic_breakdown"] = plot_topic_sentiment(
        posts,
        "Topic vs sentiment",
        args.fig_dir / "topic_sentiment.png",
    )

    summary["hourly_trend"] = plot_hourly_trend(
        posts,
        "publish_time_dt",
        "sentiment_label",
        "Hourly sentiment trend (posts)",
        args.fig_dir / "hourly_sentiment_trend.png",
    )

    summary["engagement_means"] = plot_engagement(
        posts,
        ["like_count", "review_count", "retweet_count"],
        "sentiment_label",
        "Mean engagement per sentiment (posts)",
        args.fig_dir / "engagement_by_sentiment.png",
    )

    keyword_summary = {}
    wordcloud_paths = {}
    for label in ensure_label_order(posts["sentiment_label"].unique()):
        keywords = extract_keywords(posts, args.text_col, "sentiment_label", label, top_n=args.top_n)
        if keywords:
            keyword_summary[label] = keywords
            plot_keyword_bars(
                keywords,
                f"Top tokens ({label})",
                args.fig_dir / f"keywords_{label}.png",
            )
            wc_path = args.fig_dir / f"wordcloud_{label}.png"
            if plot_wordcloud_from_freq(keywords, f"Keyword cloud ({label})", wc_path):
                wordcloud_paths[label] = str(wc_path.relative_to(PROJECT_ROOT))
    summary["keyword_top_terms"] = keyword_summary
    if wordcloud_paths:
        summary["keyword_wordclouds"] = wordcloud_paths

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
