import argparse
import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
    import torch
except ImportError:  # transformers/torch 可能未安装，在 SnowNLP 模式下可不需要
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TextClassificationPipeline = None
    torch = None

try:
    from snownlp import SnowNLP
except ImportError:
    SnowNLP = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class BaseSentimentEngine:
    def score(self, text: str) -> Optional[float]:  # pragma: no cover - interface
        raise NotImplementedError


class SnowSentimentEngine(BaseSentimentEngine):
    def __init__(self):
        if SnowNLP is None:
            raise ImportError("SnowNLP 未安装，请先运行 pip install snownlp")

    def score(self, text: str) -> Optional[float]:
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            return SnowNLP(cleaned).sentiments
        except Exception:
            return None


class HFSentimentEngine(BaseSentimentEngine):
    def __init__(
        self,
        model_name: str,
        device: Optional[int] = None,
        max_length: int = 256,
        batch_size: int = 32,
        use_safetensors: bool = False,
    ):
        if AutoModelForSequenceClassification is None or AutoTokenizer is None:
            raise ImportError("transformers/torch 未安装，请先 pip install transformers torch")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=use_safetensors,
        )
        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = 0
            else:
                device = -1
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            return_all_scores=False,
        )
        self.max_length = max_length
        self.batch_size = batch_size

    def _normalize_label(self, label: str, score: float) -> float:
        label_lower = label.lower()
        if "pos" in label_lower or "1" in label_lower:
            return score
        if "neg" in label_lower or "0" in label_lower:
            return 1 - score
        return score

    def score(self, text: str) -> Optional[float]:
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            result = self.pipeline(
                cleaned,
                truncation=True,
                max_length=self.max_length,
                batch_size=self.batch_size,
            )
        except Exception:
            return None
        if not result:
            return None
        entry = result[0]
        label = entry.get("label", "")
        score = float(entry.get("score", 0.0))
        return self._normalize_label(label, score)


def assign_label(score: Optional[float], pos_th: float, neg_th: float) -> str:
    if score is None:
        return "unknown"
    if score >= pos_th:
        return "positive"
    if score <= neg_th:
        return "negative"
    return "neutral"


def analyze_dataframe(df: pd.DataFrame, text_col: str, pos_th: float, neg_th: float, desc: str, engine: BaseSentimentEngine) -> pd.DataFrame:
    scores = []
    labels = []
    for text in tqdm(df[text_col], desc=desc):
        score = engine.score(text)
        scores.append(score)
        labels.append(assign_label(score, pos_th, neg_th))
    enriched = df.copy()
    enriched["sentiment_score"] = scores
    enriched["sentiment_label"] = labels
    return enriched


def load_dataframe(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df


def build_engine(args) -> BaseSentimentEngine:
    if args.model_type == "hf":
        return HFSentimentEngine(
            model_name=args.model_name,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            use_safetensors=args.use_safetensors,
        )
    return SnowSentimentEngine()


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis on processed Weibo data (HF or SnowNLP).")
    parser.add_argument("--posts", default=DATA_DIR / "weibo_posts_clean.csv", type=Path, help="Path to cleaned posts CSV")
    parser.add_argument("--comments", default=DATA_DIR / "weibo_comments_flat.csv", type=Path, help="Path to flattened comments CSV")
    parser.add_argument("--output-posts", default=DATA_DIR / "weibo_posts_sentiment.csv", type=Path, help="Output CSV for post sentiments")
    parser.add_argument("--output-comments", default=DATA_DIR / "weibo_comments_sentiment.csv", type=Path, help="Output CSV for comment sentiments")
    parser.add_argument("--summary", default=DATA_DIR / "sentiment_summary.json", type=Path, help="Path to write summary JSON")
    parser.add_argument("--pos-threshold", type=float, default=0.6, help="Score threshold for positive label")
    parser.add_argument("--neg-threshold", type=float, default=0.4, help="Score threshold for negative label")
    parser.add_argument("--max-posts", type=int, default=None, help="Optional limit on posts to score")
    parser.add_argument("--max-comments", type=int, default=None, help="Optional limit on comments to score")
    parser.add_argument("--model-type", choices=["hf", "snow"], default="hf", help="HF transformers or SnowNLP sentiment engine")
    parser.add_argument("--model-name", default="uer/roberta-base-finetuned-jd-binary-chinese", help="HF model name when model-type=hf")
    parser.add_argument("--device", type=int, default=None, help="HF pipeline device id (0 for GPU, -1 for CPU, default auto)")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length for HF tokenizer")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for HF pipeline inference")
    parser.add_argument("--use-safetensors", action="store_true", help="Force transformers 使用 safetensors 权重，避免 torch>=2.6 限制")
    args = parser.parse_args()

    posts_df = load_dataframe(args.posts, args.max_posts)
    comments_df = load_dataframe(args.comments, args.max_comments)

    engine = build_engine(args)

    posts_scored = analyze_dataframe(posts_df, "content", args.pos_threshold, args.neg_threshold, "Posts", engine)
    comments_scored = analyze_dataframe(comments_df, "review_content", args.pos_threshold, args.neg_threshold, "Comments", engine)

    posts_scored.to_csv(args.output_posts, index=False)
    comments_scored.to_csv(args.output_comments, index=False)

    summary = {
        "posts_rows": len(posts_scored),
        "comments_rows": len(comments_scored),
        "posts_label_counts": posts_scored["sentiment_label"].value_counts(dropna=False).to_dict(),
        "comments_label_counts": comments_scored["sentiment_label"].value_counts(dropna=False).to_dict(),
        "pos_threshold": args.pos_threshold,
        "neg_threshold": args.neg_threshold,
        "model_type": args.model_type,
        "model_name": args.model_name if args.model_type == "hf" else "SnowNLP",
        "paths": {
            "posts": str(args.output_posts.relative_to(PROJECT_ROOT)),
            "comments": str(args.output_comments.relative_to(PROJECT_ROOT)),
            "summary": str(args.summary.relative_to(PROJECT_ROOT)),
        },
    }

    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
