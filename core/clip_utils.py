import torch
from PIL import Image
import cv2
import numpy as np

from config.settings import (
    QUERIES, HUMAN_QUERY_PROTOTYPES, HUMAN_QUERY_THRESHOLD,
    CLIP_NEGATIVE_QUERIES,
)


def build_prompt_ensemble(query: str) -> list[str]:
    """Generate a list of diverse prompts for a given query to improve CLIP robustness."""
    return [
        f"a photo of a {query}",
        f"a road scene containing a {query}",
        f"a cropped object that is a {query}",
        f"a dashcam image of a {query}",
    ]


def encode_text_list(texts: list[str], clip_model, tokenizer, device: str) -> torch.Tensor:
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def encode_text_mean(texts: list[str], clip_model, tokenizer, device: str) -> torch.Tensor:
    """Encode a list of texts and return their normalized mean."""
    feats = encode_text_list(texts, clip_model, tokenizer, device)
    feat = feats.mean(dim=0, keepdim=True)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def build_text_features(clip_model, tokenizer, device: str,
                        queries: list[str] | None = None) -> dict:
    """
    Pre-calcola e mette in cache i feature vector testuali per ogni query.

    Returns:
        dict {query_str: tensor}
    """
    if queries is None:
        queries = QUERIES

    cache = {}
    with torch.no_grad():
        for query in queries:
            prompts = build_prompt_ensemble(query)
            cache[query] = encode_text_mean(prompts, clip_model, tokenizer, device)
    return cache


def build_negative_text_features(clip_model, tokenizer, device: str) -> dict:
    """
    Pre-computes CLIP text features for background/distractor prompts.
    These are used only as rejection classes, never as output labels.
    """
    cache = {}
    with torch.no_grad():
        for query in CLIP_NEGATIVE_QUERIES:
            prompts = build_prompt_ensemble(query)
            cache[query] = encode_text_mean(prompts, clip_model, tokenizer, device)
    return cache


def build_query_type_cache(clip_model, tokenizer, device: str) -> tuple[dict, torch.Tensor]:
    """
    Determina se ogni query è "human-like" confrontandola con i prototipi umani.

    Returns:
        (query_type_cache dict, human_prototype tensor)
    """
    human_proto = encode_text_mean(HUMAN_QUERY_PROTOTYPES, clip_model, tokenizer, device)
    query_type_cache = {}
    for query in QUERIES:
        q_feat = encode_text_mean(build_prompt_ensemble(query), clip_model, tokenizer, device)
        sim = float((q_feat @ human_proto.T).item())
        query_type_cache[query] = {
            "human_like_score": sim,
            "is_human_like": sim >= HUMAN_QUERY_THRESHOLD,
        }
    return query_type_cache, human_proto


def encode_crop(crop, clip_model, clip_preprocess, device: str) -> torch.Tensor:
    """Codifica un'immagine crop (numpy BGR) in un feature vector CLIP normalizzato."""
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image_tensor = clip_preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(image_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def compute_scores(image_feature: torch.Tensor, text_feature_cache: dict) -> dict:
    """
    Calcola la cosine similarity tra il feature visivo e tutti i feature testuali.

    Returns:
        dict {query_str: float_score}
    """
    return {
        query: float((image_feature @ text_feature.T).item())
        for query, text_feature in text_feature_cache.items()
    }


def combine_view_scores(view_scores: dict, view_weights: dict) -> dict:
    """Combines per-view CLIP score dictionaries using normalized view weights."""
    combined = {}
    weights = {}

    for view_name, scores in view_scores.items():
        weight = float(view_weights.get(view_name, 0.0))
        if weight <= 0:
            continue

        for query, score in scores.items():
            combined[query] = combined.get(query, 0.0) + weight * float(score)
            weights[query] = weights.get(query, 0.0) + weight

    return {
        query: combined[query] / weights[query]
        for query in combined
        if weights.get(query, 0.0) > 0
    }


def aggregate_score_window(score_window, method: str = "median") -> dict:
    """Aggregates a short window of score dictionaries query by query."""
    if not score_window:
        return {}

    queries = set()
    for scores in score_window:
        queries.update(scores.keys())

    aggregated = {}
    for query in queries:
        values = [float(scores[query]) for scores in score_window if query in scores]
        if not values:
            continue
        if method == "mean":
            aggregated[query] = float(np.mean(values))
        else:
            aggregated[query] = float(np.median(values))

    return aggregated
