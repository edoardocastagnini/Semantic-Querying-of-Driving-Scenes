import torch
import clip
from PIL import Image
import cv2

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_text(text_list):
    tokens = clip.tokenize(text_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def encode_crop(frame_bgr, xyxy): 
    x1, y1, x2, y2 = map(int, xyxy)
    crop = frame_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    image_input = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features