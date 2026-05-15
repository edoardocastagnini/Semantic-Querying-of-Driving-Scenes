#!/usr/bin/env python3
"""
models/loader.py
================
Caricamento dei modelli YOLO e CLIP.
Restituisce oggetti pronti all'uso, incapsulando device e preprocessing.
"""

import torch
import open_clip
from ultralytics import YOLO

from config.settings import YOLO_MODEL, TRAFFIC_SIGN_MODEL, CLIP_MODEL, CLIP_PRETRAINED


def get_device() -> str:
    """Seleziona automaticamente il device migliore disponibile."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def load_yolo(device: str) -> YOLO:
    print("[INFO] Loading YOLO...")
    model = YOLO(YOLO_MODEL)
    return model


def load_traffic_sign_yolo(device: str) -> YOLO | None:
    """Loads the optional traffic-sign detector."""
    if not TRAFFIC_SIGN_MODEL:
        return None
    print("[INFO] Loading traffic-sign YOLO...")
    return YOLO(TRAFFIC_SIGN_MODEL)


def load_clip(device: str):
    """
    Carica il modello CLIP, il preprocessore e il tokenizer.

    Returns:
        clip_model, clip_preprocess, tokenizer
    """
    print("[INFO] Loading CLIP...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
        device=device,
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    print("[INFO] Models loaded.")
    return clip_model, clip_preprocess, tokenizer


def load_all_models():
    """
    Carica tutti i modelli e restituisce un dizionario con tutti gli oggetti.

    Returns:
        dict con chiavi: device, yolo, clip_model, clip_preprocess, tokenizer
    """
    device = get_device()
    print(f"[INFO] Device: {device}")
    yolo = load_yolo(device)
    traffic_sign_yolo = load_traffic_sign_yolo(device)
    clip_model, clip_preprocess, tokenizer = load_clip(device)
    return {
        "device": device,
        "yolo": yolo,
        "traffic_sign_yolo": traffic_sign_yolo,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "tokenizer": tokenizer,
    }
