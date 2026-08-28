"""Answer scoring for LiveSearchBench.

Three metrics are provided, and they are kept deliberately distinct because
they do not measure the same thing:

``exact_match``
    SQuAD-style normalised exact match: case-folded, articles and punctuation
    stripped, whitespace collapsed, then compared for equality. Optionally
    accepts a set of gold aliases.

``token_f1``
    SQuAD-style token-level F1 over the same normalisation.

``contains_match``
    Case-folded substring containment (``gold in prediction``). This is the
    metric the original release computed under the name ``simple_match``. It
    is strictly more permissive than exact match -- a long answer that merely
    mentions the gold string counts as correct -- so it is kept, named
    honestly, and reported alongside the other two rather than in place of
    them.

All three take the same ``(prediction, gold)`` argument order.
"""

from __future__ import annotations

import random
import re
import string
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}


def normalize_answer(text: str) -> str:
    """Lower-case, strip articles and punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


def _gold_variants(gold, aliases: Optional[Iterable[str]] = None) -> List[str]:
    """Collect every acceptable surface form for a gold answer."""
    variants: List[str] = []
    if isinstance(gold, (list, tuple, set)):
        variants.extend(str(g) for g in gold)
    else:
        variants.append(str(gold))
    if aliases:
        variants.extend(str(a) for a in aliases)
    return [v for v in variants if v and v.strip()]


def exact_match(prediction: str, gold, aliases: Optional[Iterable[str]] = None) -> bool:
    """Normalised exact match against the gold answer or any of its aliases."""
    pred = normalize_answer(prediction)
    if not pred:
        return False
    return any(pred == normalize_answer(g) for g in _gold_variants(gold, aliases))


def contains_match(prediction: str, gold, aliases: Optional[Iterable[str]] = None) -> bool:
    """Case-folded substring containment; the legacy ``simple_match`` metric."""
    pred = str(prediction or "").lower().strip()
    if not pred:
        return False
    return any(str(g).lower().strip() in pred for g in _gold_variants(gold, aliases))


def token_f1(prediction: str, gold, aliases: Optional[Iterable[str]] = None) -> float:
    """Best token-level F1 over the gold answer and its aliases."""
    pred_tokens = _tokens(prediction)
    best = 0.0
    for variant in _gold_variants(gold, aliases):
        gold_tokens = _tokens(variant)
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common: Dict[str, int] = {}
        for tok in gold_tokens:
            common[tok] = common.get(tok, 0) + 1
        overlap = 0
        for tok in pred_tokens:
            if common.get(tok, 0) > 0:
                common[tok] -= 1
                overlap += 1
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Returns ``{"mean", "lo", "hi", "n", "resamples", "confidence"}`` with the
    mean and bounds expressed on the same scale as the inputs.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0,
                "resamples": 0, "confidence": confidence}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "n": 1,
                "resamples": 0, "confidence": confidence}

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * resamples) - 1)]
    hi = means[min(resamples - 1, int((1.0 - alpha) * resamples))]
    return {"mean": mean, "lo": lo, "hi": hi, "n": n,
            "resamples": resamples, "confidence": confidence}


def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """Unbiased pass@k estimator of Chen et al. (2021).

    ``n_samples`` generations were drawn, ``n_correct`` of them were correct.
    """
    if k <= 0 or n_samples <= 0:
        return 0.0
    if n_samples - n_correct < k:
        return 1.0
    prob = 1.0
    for i in range(k):
        prob *= (n_samples - n_correct - i) / (n_samples - i)
    return 1.0 - prob


def score_item(prediction: str, gold, aliases: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Compute all three metrics for a single prediction."""
    return {
        "exact_match": float(exact_match(prediction, gold, aliases)),
        "token_f1": token_f1(prediction, gold, aliases),
        "contains_match": float(contains_match(prediction, gold, aliases)),
    }


def aggregate(
    items: Sequence[Dict],
    *,
    prediction_key: str = "model_answer",
    gold_key: str = "expected_answer",
    alias_key: str = "answer_aliases",
    group_key: Optional[str] = "level",
    confidence: float = 0.95,
    resamples: int = 10000,
    seed: int = 0,
) -> Dict:
    """Score a list of per-item result records.

    Returns overall figures plus a per-group breakdown, each with a bootstrap
    confidence interval. Percentages are on a 0-100 scale.
    """
    per_item: List[Dict] = []
    for item in items:
        scores = score_item(
            item.get(prediction_key, ""),
            item.get(gold_key, ""),
            item.get(alias_key),
        )
        record = dict(scores)
        if group_key:
            record["_group"] = item.get(group_key, "all")
        per_item.append(record)

    def _summarise(records: Sequence[Dict]) -> Dict:
        out: Dict = {"n": len(records)}
        for metric in ("exact_match", "token_f1", "contains_match"):
            vals = [100.0 * r[metric] for r in records]
            ci = bootstrap_ci(vals, confidence=confidence, resamples=resamples, seed=seed)
            out[metric] = {
                "value": round(ci["mean"], 2),
                "ci_low": round(ci["lo"], 2),
                "ci_high": round(ci["hi"], 2),
            }
        return out

    result: Dict = {"overall": _summarise(per_item)}
    if group_key:
        groups: Dict = {}
        for record in per_item:
            groups.setdefault(record["_group"], []).append(record)
        result["by_" + group_key] = {
            str(name): _summarise(recs) for name, recs in sorted(groups.items(), key=lambda kv: str(kv[0]))
        }
    return result
