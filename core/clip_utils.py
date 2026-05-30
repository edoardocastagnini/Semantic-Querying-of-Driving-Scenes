#!/usr/bin/env python3
"""
core/clip_utils.py
================
Auxiliary functions for working with CLIP.
"""

import torch
from PIL import Image
import cv2
import numpy as np

from config.settings import (
    QUERIES, HUMAN_QUERY_PROTOTYPES, HUMAN_QUERY_THRESHOLD,
    CLIP_NEGATIVE_QUERIES,
)


def build_prompt_ensemble(query: str) -> list[str]:
    """
    Generates a list of diverse prompts for a given query to improve CLIP robustness.

    Parameters:
        query (str): input query
    
    Returns:
        list: a list of queries
    """

    return [
        f"a photo of a {query}",
        f"a road scene containing a {query}",
        f"a cropped object that is a {query}",
        f"a dashcam image of a {query}",
    ]


def encode_text_list(texts: list[str], clip_model, tokenizer, device: str) -> torch.Tensor:
    """
    Encodes text list into a tensor.

    Parameters:
        texts (list[str]): list of texts to encode
        clip_model: CLIP model
        tokenizer : tokenizes the list into tokens
        device (str): selected device ("cuda" | "mps" | "cpu")
    
    Returns:
        torch.Tensor: tensor of encoded features
    """

    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def encode_text_mean(texts: list[str], clip_model, tokenizer, device: str) -> torch.Tensor:
    """
    Encodes a list of texts and returns their normalized mean.
    
    Parameters:
        texts (list[str]): list of texts to encode
        clip_model: CLIP model
        tokenizer : tokenizes the list into tokens
        device (str): selected device ("cuda" | "mps" | "cpu")
    
    Returns:
        torch.Tensor: tensor of encoded features
    """
    feats = encode_text_list(texts, clip_model, tokenizer, device)
    feat = feats.mean(dim=0, keepdim=True)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def build_text_features(clip_model, tokenizer, device: str,
                        queries: list[str] | None = None) -> dict:
    """
    Pre-computes and caches textual feature vectors for each query.

    Parameters:
        clip_model: CLIP model
        tokenizer : tokenizes the list into tokens
        device (str): selected device ("cuda" | "mps" | "cpu")
        queries (list[str] | None): list of queries

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

    Parameters:
        clip_model: CLIP model
        tokenizer: tokenizes the list into tokens
        device (str): selected device ("cuda" | "mps" | "cpu")

    Returns:
        dict {query: tensor} where tensor is the encoded text after mean
    """

    cache = {}
    with torch.no_grad():
        for query in CLIP_NEGATIVE_QUERIES:
            prompts = build_prompt_ensemble(query)
            cache[query] = encode_text_mean(prompts, clip_model, tokenizer, device)
    return cache


def build_query_type_cache(clip_model, tokenizer, device: str) -> tuple[dict, torch.Tensor]:
    """
    Determines, whether each query is "human-like" by comparing it with human prototypes.

    Params:
        clip_model: CLIP model
        tokenizer: tokenizes the list into tokens
        device (str): selected device ("cuda" | "mps" | "cpu")

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


def encode_crop(crop: np.ndarray, clip_model, clip_preprocess, device: str) -> torch.Tensor:
    """
    Encodes a crop image (numpy BGR) into a normalized CLIP feature vector.

    Parameters:
        crop (np.ndarray): image crop
        clip_model: CLIP model
        clip_preprocess: 
        device (str): selected device ("cuda" | "mps" | "cpu")

    Returns:
        list[float]: normalized feature vector
    """

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image_tensor = clip_preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(image_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def compute_scores(image_feature: torch.Tensor, text_feature_cache: dict) -> dict:
    """
    Calculate the cosine similarity between the visual feature and all textual features.

    Parameters:
        image_feature (torch.Tensor): image (visual) feature 
        text_feature_cache (dict): cached textual features

    Returns:
        dict {query_str: float_score}
    """

    return {
        query: float((image_feature @ text_feature.T).item())
        for query, text_feature in text_feature_cache.items()
    }


def combine_view_scores(view_scores: dict, view_weights: dict) -> dict:
    """
    Combines per-view CLIP score dictionaries using normalized view weights.

    Parameters:
        view_scores (dict): scores of views
        view_weights (dict): weight of each view

    Returns:
        dict: dictionary of per-view CLIP scores
    """

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


def aggregate_score_window(score_window: dict, method: str = "median") -> dict:
    """
    Aggregates a short window of score dictionaries query by query."
    
    Parameters:
        score_window (dict): scores within the window
        method (str): aggregation method ("median" default, otherwise "mean")

    Returns:
        dict: aggregated scores over a short window usign [method]
    """
    
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
