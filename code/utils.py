import os
import re
import json
import gc

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import torch
except Exception:
    torch = None

from config import (
    DEFAULT_NEGATIVE_PROMPT,
    COLOR_WORDS,
    LARGE_REPLACEMENT_WORDS,
    TALL_REPLACEMENT_WORDS,
    WIDE_REPLACEMENT_WORDS,
    FLAT_SURFACE_WORDS,
    WALL_OBJECT_WORDS,
    PERSON_WORDS,
    VEHICLE_WORDS,
    DESK_FLAT_OBJECT_WORDS,
    REPLACEMENT_SUGGESTION_DB,
)


def clean_phrase(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = text.strip(" 。.，,;:!！?？")
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def phrase_to_english(text: str) -> str:
    if not text:
        return ""

    return clean_phrase(text)


def _strip_object_prefix(text: str) -> str:
    text = clean_phrase(text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"^(this|that|these|those)\s+", "", text)

    return clean_phrase(text)


def _parse_english_instruction(text: str):
    patterns = [
        r"replace\s+(?P<target>.+?)\s+with\s+(?P<replacement>.+)",
        r"change\s+(?P<target>.+?)\s+to\s+(?P<replacement>.+)",
        r"swap\s+(?P<target>.+?)\s+for\s+(?P<replacement>.+)",
        r"convert\s+(?P<target>.+?)\s+into\s+(?P<replacement>.+)",
        r"turn\s+(?P<target>.+?)\s+into\s+(?P<replacement>.+)",
        r"make\s+(?P<target>.+?)\s+become\s+(?P<replacement>.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            target = _strip_object_prefix(match.group("target"))
            replacement = _strip_object_prefix(match.group("replacement"))
            return target, replacement

    return None, None


def _parse_chinese_instruction(text: str):
    text = text.strip()

    patterns = [
        r"把(?P<target>.+?)替换成(?P<replacement>.+)",
        r"将(?P<target>.+?)替换成(?P<replacement>.+)",
        r"把(?P<target>.+?)换成(?P<replacement>.+)",
        r"将(?P<target>.+?)换成(?P<replacement>.+)",
        r"把(?P<target>.+?)变成(?P<replacement>.+)",
        r"将(?P<target>.+?)变成(?P<replacement>.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            target = clean_phrase(match.group("target"))
            replacement = clean_phrase(match.group("replacement"))
            return target, replacement

    return None, None


def parse_edit_instruction(
    instruction: str,
    manual_target: str = "",
    manual_replacement: str = "",
):
    if not instruction:
        instruction = ""

    raw_instruction = instruction
    instruction = instruction.strip()

    parsed_target, parsed_replacement = _parse_english_instruction(instruction)

    if parsed_target is None:
        parsed_target, parsed_replacement = _parse_chinese_instruction(instruction)

    if manual_target.strip():
        parsed_target = manual_target.strip()

    if manual_replacement.strip():
        parsed_replacement = manual_replacement.strip()

    if not parsed_target or not parsed_replacement:
        return None

    target_en = phrase_to_english(parsed_target)
    replacement_en = phrase_to_english(parsed_replacement)

    if replacement_en in COLOR_WORDS:
        edit_mode = "color_change"
    else:
        edit_mode = "object_swap"

    location_hint = infer_location_hint(
        target=target_en,
        replacement=replacement_en,
    )

    geometry = infer_replacement_geometry(replacement_en)

    return {
        "target": target_en,
        "replacement": replacement_en,
        "edit_mode": edit_mode,
        "location_hint": location_hint,
        "geometry": geometry,
        "raw_instruction": raw_instruction,
    }


def phrase_has_any(text: str, words: set) -> bool:
    text = clean_phrase(text)

    if not text:
        return False

    tokens = set(re.split(r"[\s\-_]+", text))

    for word in words:
        word = clean_phrase(word)

        if word in tokens or word in text:
            return True

    return False


def is_desk_flat_replacement(replacement: str) -> bool:
    replacement = clean_phrase(replacement)

    if phrase_has_any(replacement, DESK_FLAT_OBJECT_WORDS):
        return True

    if "laptop" in replacement or "computer" in replacement:
        return True

    return False


def infer_location_hint(target: str, replacement: str) -> str:
    target = clean_phrase(target)
    replacement = clean_phrase(replacement)

    if phrase_has_any(replacement, WALL_OBJECT_WORDS) or phrase_has_any(target, WALL_OBJECT_WORDS):
        return "mounted on or aligned with the same background surface"

    if is_desk_flat_replacement(replacement):
        return "resting naturally on the same supporting surface"

    if phrase_has_any(replacement, FLAT_SURFACE_WORDS):
        return "resting naturally on the same supporting surface"

    if phrase_has_any(replacement, PERSON_WORDS):
        return "standing naturally in the scene at the target position"

    if phrase_has_any(replacement, VEHICLE_WORDS):
        return "placed naturally on the ground plane"

    return "placed naturally in the same scene location"


def infer_replacement_geometry(replacement: str) -> dict:
    replacement = clean_phrase(replacement)

    geometry = {
        "category": "generic",
        "width_factor": 1.45,
        "height_factor": 1.45,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "anchor": "center",
        "pose": "generic",
    }

    if phrase_has_any(replacement, PERSON_WORDS):
        geometry.update(
            {
                "category": "person",
                "width_factor": 1.20,
                "height_factor": 2.50,
                "x_shift": 0.0,
                "y_shift": -0.45,
                "anchor": "bottom",
                "pose": "standing",
            }
        )
        return geometry

    if phrase_has_any(replacement, VEHICLE_WORDS):
        geometry.update(
            {
                "category": "vehicle",
                "width_factor": 2.10,
                "height_factor": 1.25,
                "x_shift": 0.0,
                "y_shift": -0.05,
                "anchor": "bottom",
                "pose": "grounded",
            }
        )
        return geometry

    if "laptop" in replacement or "computer" in replacement:
        geometry.update(
            {
                "category": "laptop",
                "width_factor": 1.70,
                "height_factor": 1.15,
                "x_shift": 0.02,
                "y_shift": 0.0,
                "anchor": "surface",
                "pose": "desk_open",
            }
        )
        return geometry

    if replacement in {"tablet", "phone", "book", "notebook", "keyboard", "mouse"}:
        geometry.update(
            {
                "category": "flat_object",
                "width_factor": 1.45,
                "height_factor": 0.85,
                "x_shift": 0.0,
                "y_shift": 0.08,
                "anchor": "surface",
                "pose": "flat_on_surface",
            }
        )
        return geometry

    if phrase_has_any(replacement, WALL_OBJECT_WORDS):
        geometry.update(
            {
                "category": "wall_object",
                "width_factor": 1.65,
                "height_factor": 1.35,
                "x_shift": 0.0,
                "y_shift": 0.0,
                "anchor": "center",
                "pose": "wall_aligned",
            }
        )
        return geometry

    if phrase_has_any(replacement, TALL_REPLACEMENT_WORDS):
        geometry.update(
            {
                "category": "tall_object",
                "width_factor": 1.28,
                "height_factor": 1.95,
                "x_shift": 0.0,
                "y_shift": -0.25,
                "anchor": "bottom",
                "pose": "upright",
            }
        )
        return geometry

    if phrase_has_any(replacement, WIDE_REPLACEMENT_WORDS):
        geometry.update(
            {
                "category": "wide_object",
                "width_factor": 1.85,
                "height_factor": 1.20,
                "x_shift": 0.0,
                "y_shift": 0.0,
                "anchor": "center",
                "pose": "wide",
            }
        )
        return geometry

    if phrase_has_any(replacement, LARGE_REPLACEMENT_WORDS):
        geometry.update(
            {
                "category": "large_object",
                "width_factor": 1.70,
                "height_factor": 1.60,
                "x_shift": 0.0,
                "y_shift": -0.10,
                "anchor": "center",
                "pose": "generic_large",
            }
        )
        return geometry

    return geometry


def build_edit_prompt(spec: dict) -> str:
    target = spec["target"]
    replacement = spec["replacement"]
    location_hint = spec.get("location_hint", "placed naturally in the same scene location")
    replacement_l = clean_phrase(replacement)

    if spec.get("edit_mode") == "color_change":
        return (
            f"Change the {target} to {replacement} color. "
            f"Keep the original object shape, perspective, lighting, and background unchanged. "
            f"Photorealistic."
        )

    if "laptop" in replacement_l or "computer" in replacement_l:
        return (
            f"A realistic open laptop replacing the {target}, "
            f"base lying flat on the same desk, screen tilted back, "
            f"visible keyboard and trackpad, aligned with desk perspective, "
            f"correct scale, natural contact shadow, photorealistic, "
            f"background unchanged."
        )

    if replacement_l in {"tablet", "phone", "book", "notebook", "keyboard", "mouse"}:
        return (
            f"A realistic {replacement_l} replacing the {target}, "
            f"lying naturally on the same surface, aligned with perspective, "
            f"correct scale, natural contact shadow, photorealistic, "
            f"background unchanged."
        )

    if phrase_has_any(replacement_l, PERSON_WORDS):
        return (
            f"A realistic {replacement_l} replacing the {target}, "
            f"standing naturally at the target position, correct scale, "
            f"same perspective, natural lighting and shadow, photorealistic, "
            f"background unchanged."
        )

    if phrase_has_any(replacement_l, WALL_OBJECT_WORDS):
        return (
            f"A realistic {replacement_l} replacing the {target}, "
            f"mounted flat on the same wall surface, aligned with wall perspective, "
            f"correct scale, natural lighting, photorealistic, background unchanged."
        )

    return (
        f"A realistic {replacement} replacing the {target}, "
        f"{location_hint}, same perspective, same lighting, natural shadow, "
        f"correct scale, photorealistic, background unchanged."
    )


def build_negative_prompt(spec: dict) -> str:
    target = spec["target"]
    replacement = spec["replacement"]
    replacement_l = clean_phrase(replacement)

    negative = [
        "blurry",
        "low quality",
        "distorted",
        "deformed",
        "bad perspective",
        "wrong scale",
        "floating object",
        "extra object",
        "duplicate object",
        "unnatural shadow",
        "changed background",
        f"remaining {target}",
        f"old {target}",
        f"duplicate {replacement}",
    ]

    if "laptop" in replacement_l or "computer" in replacement_l:
        negative.extend(
            [
                "floating laptop",
                "vertical laptop",
                "upright laptop",
                "screen only",
                "misaligned laptop",
                "oversized laptop",
                "wrong contact shadow",
                "bad keyboard",
                "warped screen",
                "closed vertical slab",
            ]
        )

    if replacement_l in {"tablet", "phone", "book", "notebook", "keyboard", "mouse"}:
        negative.extend(
            [
                "floating object",
                "vertical object",
                "wrong contact shadow",
                "warped surface",
            ]
        )

    if phrase_has_any(replacement_l, PERSON_WORDS):
        negative.extend(
            [
                "bad hands",
                "extra limbs",
                "missing limbs",
                "deformed body",
                "deformed face",
                "cropped person",
            ]
        )

    return ", ".join(negative)


def get_torch_dtype(device: str, prefer_fp16: bool = True):
    if torch is None:
        return None

    if prefer_fp16 and str(device).startswith("cuda"):
        return torch.float16

    return torch.float32


def suggest_replacements_for_object(obj_name: str, max_items: int = 5):
    obj_name = clean_phrase(obj_name)

    if not obj_name:
        return []

    if obj_name in REPLACEMENT_SUGGESTION_DB:
        return REPLACEMENT_SUGGESTION_DB[obj_name][:max_items]

    for key, values in REPLACEMENT_SUGGESTION_DB.items():
        key_l = clean_phrase(key)
        if key_l in obj_name or obj_name in key_l:
            return values[:max_items]

    if "clock" in obj_name:
        return ["framed painting", "poster", "mirror", "whiteboard"][:max_items]

    if "bag" in obj_name or "backpack" in obj_name:
        return ["open laptop", "teddy bear", "camera", "toy robot"][:max_items]

    if "book" in obj_name:
        return ["tablet", "laptop", "small plant", "decorative box"][:max_items]

    if "chair" in obj_name:
        return ["wooden stool", "standing person", "small sofa", "potted plant"][:max_items]

    if "cup" in obj_name or "mug" in obj_name:
        return ["small plant", "candle", "flower vase", "toy figure"][:max_items]

    if "bottle" in obj_name:
        return ["flower vase", "thermos", "lamp", "decorative sculpture"][:max_items]

    if "table" in obj_name or "desk" in obj_name:
        return ["modern desk", "small cabinet", "workbench", "potted plant"][:max_items]

    if "poster" in obj_name or "picture" in obj_name or "photo" in obj_name:
        return ["whiteboard", "framed painting", "mirror", "calendar"][:max_items]

    if "screen" in obj_name or "monitor" in obj_name:
        return ["small TV", "whiteboard", "framed artwork", "large plant"][:max_items]

    return ["laptop", "small plant", "poster", "teddy bear", "flower vase"][:max_items]


def build_replacement_suggestions(detected_objects, max_items_per_object: int = 5):
    suggestions = {}

    for obj in detected_objects:
        obj = clean_phrase(obj)
        if not obj:
            continue

        suggestions[obj] = suggest_replacements_for_object(
            obj,
            max_items=max_items_per_object,
        )

    return suggestions


def format_replacement_suggestions(suggestions: dict) -> str:
    if not suggestions:
        return ""

    lines = []
    for obj, repls in suggestions.items():
        repl_text = ", ".join(repls) if repls else "None"
        lines.append(f"{obj} -> {repl_text}")

    return "\n".join(lines)


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    mask = (mask > 0).astype(np.uint8)

    return mask


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    mask = _as_uint8_mask(mask)
    return Image.fromarray(mask * 255, mode="L")


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 21,
    iterations: int = 1,
) -> np.ndarray:
    mask_img = mask_to_pil(mask)

    kernel_size = int(kernel_size)

    if kernel_size < 3:
        return _as_uint8_mask(mask)

    if kernel_size % 2 == 0:
        kernel_size += 1

    for _ in range(max(1, int(iterations))):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(kernel_size))

    return (np.asarray(mask_img) > 127).astype(np.uint8)


def make_box_mask_from_xyxy(
    box_xyxy,
    image_size,
    expand_ratio: float = 1.35,
) -> np.ndarray:
    w, h = image_size

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    new_w = bw * float(expand_ratio)
    new_h = bh * float(expand_ratio)

    nx1 = int(round(cx - new_w / 2.0))
    ny1 = int(round(cy - new_h / 2.0))
    nx2 = int(round(cx + new_w / 2.0))
    ny2 = int(round(cy + new_h / 2.0))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ny1:ny2, nx1:nx2] = 1

    return mask


def make_adaptive_replacement_mask(
    box_xyxy,
    image_size,
    replacement: str = "",
    target: str = "",
    scale: float = 1.10,
    base_expand: float = 1.35,
):
    w, h = image_size

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    replacement = clean_phrase(replacement)
    geometry = infer_replacement_geometry(replacement)

    width_factor = float(geometry["width_factor"]) * float(scale)
    height_factor = float(geometry["height_factor"]) * float(scale)

    if is_desk_flat_replacement(replacement):
        width_factor = max(width_factor, 1.35)
        height_factor = max(height_factor, 0.90)
    else:
        width_factor = max(width_factor, min(float(base_expand), 1.45))
        height_factor = max(height_factor, min(float(base_expand), 1.45))

    anchor = geometry.get("anchor", "center")

    if anchor == "bottom":
        bottom = y2
        new_w = bw * width_factor
        new_h = bh * height_factor

        ncx = cx + bw * float(geometry.get("x_shift", 0.0))
        nx1 = int(round(ncx - new_w / 2.0))
        nx2 = int(round(ncx + new_w / 2.0))
        ny2 = int(round(bottom + bh * 0.05))
        ny1 = int(round(ny2 - new_h))

    elif anchor == "surface":
        new_w = bw * width_factor
        new_h = max(bh * height_factor, bh * 0.80)

        ncx = cx + bw * float(geometry.get("x_shift", 0.0))

        surface_y = y2 + bh * 0.03

        nx1 = int(round(ncx - new_w / 2.0))
        nx2 = int(round(ncx + new_w / 2.0))
        ny2 = int(round(surface_y))
        ny1 = int(round(ny2 - new_h))

    else:
        new_w = bw * width_factor
        new_h = bh * height_factor

        ncx = cx + bw * float(geometry.get("x_shift", 0.0))
        ncy = cy + bh * float(geometry.get("y_shift", 0.0))

        nx1 = int(round(ncx - new_w / 2.0))
        nx2 = int(round(ncx + new_w / 2.0))
        ny1 = int(round(ncy - new_h / 2.0))
        ny2 = int(round(ncy + new_h / 2.0))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)

    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ny1:ny2, nx1:nx2] = 1

    return mask, (nx1, ny1, nx2, ny2)


def draw_box(
    image: Image.Image,
    box_xyxy,
    label: str = "object",
    score: float = None,
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=(255, 0, 0),
        width=4,
    )

    text = str(label) if label is not None else "object"

    if score is not None:
        text += f" {float(score):.3f}"

    try:
        bbox = draw.textbbox((x1, y1), text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = 160, 18

    y_text = max(0, y1 - th - 6)

    draw.rectangle(
        [x1, y_text, x1 + tw + 8, y_text + th + 6],
        fill=(255, 0, 0),
    )

    draw.text(
        (x1 + 4, y_text + 3),
        text,
        fill=(255, 255, 255),
    )

    return out


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    base = image.copy().convert("RGBA")
    mask = _as_uint8_mask(mask)

    empty = Image.new("RGBA", base.size, (255, 0, 0, 0))
    color = Image.new("RGBA", base.size, (255, 0, 0, 255))

    alpha_mask = Image.fromarray(
        (mask * int(255 * alpha)).astype(np.uint8),
        mode="L",
    )

    overlay = Image.composite(color, empty, alpha_mask)

    return Image.alpha_composite(base, overlay).convert("RGB")


def compute_crop_scale(
    box_xyxy,
    image_size,
    target: str = "",
    replacement: str = "",
) -> float:
    w, h = image_size

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    area_ratio = (bw * bh) / max(1.0, float(w * h))
    replacement = clean_phrase(replacement)

    if area_ratio < 0.005:
        scale = 5.0
    elif area_ratio < 0.015:
        scale = 4.4
    elif area_ratio < 0.040:
        scale = 3.8
    elif area_ratio < 0.100:
        scale = 3.0
    elif area_ratio < 0.200:
        scale = 2.3
    else:
        scale = 1.8

    if phrase_has_any(replacement, PERSON_WORDS):
        scale = max(scale, 4.0)

    if phrase_has_any(replacement, VEHICLE_WORDS):
        scale = max(scale, 3.8)

    if phrase_has_any(replacement, LARGE_REPLACEMENT_WORDS):
        scale = max(scale, 3.4)

    if is_desk_flat_replacement(replacement):
        scale = min(scale, 3.2)

    return float(scale)


def expand_box(
    box_xyxy,
    image_size,
    scale: float = 2.5,
    min_pad: int = 24,
):
    w, h = image_size

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    new_w = max(bw * float(scale), bw + 2 * min_pad)
    new_h = max(bh * float(scale), bh + 2 * min_pad)

    nx1 = max(0, int(round(cx - new_w / 2.0)))
    ny1 = max(0, int(round(cy - new_h / 2.0)))
    nx2 = min(w, int(round(cx + new_w / 2.0)))
    ny2 = min(h, int(round(cy + new_h / 2.0)))

    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)

    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)

    return nx1, ny1, nx2, ny2


def translate_box_to_crop(
    box_xyxy,
    crop_box,
):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx1, cy1, _, _ = crop_box

    return np.array(
        [
            x1 - cx1,
            y1 - cy1,
            x2 - cx1,
            y2 - cy1,
        ],
        dtype=np.float32,
    )


def place_crop_mask_to_full(
    mask_crop: np.ndarray,
    crop_box,
    full_size,
):
    full_w, full_h = full_size

    x1, y1, x2, y2 = [int(v) for v in crop_box]

    mask_crop = _as_uint8_mask(mask_crop)

    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)

    if mask_crop.shape[:2] != (target_h, target_w):
        mask_pil = mask_to_pil(mask_crop).resize(
            (target_w, target_h),
            Image.NEAREST,
        )
        mask_crop = (np.asarray(mask_pil) > 127).astype(np.uint8)

    full = np.zeros((full_h, full_w), dtype=np.uint8)
    full[y1:y2, x1:x2] = mask_crop[: y2 - y1, : x2 - x1]

    return full


def paste_edited_crop_strict(
    original_full: Image.Image,
    edited_crop: Image.Image,
    crop_box,
    mask_crop: np.ndarray,
) -> Image.Image:
    out = original_full.copy().convert("RGB")

    x1, y1, x2, y2 = [int(v) for v in crop_box]
    crop_w = x2 - x1
    crop_h = y2 - y1

    edited_crop = edited_crop.convert("RGB").resize(
        (crop_w, crop_h),
        Image.LANCZOS,
    )

    original_crop = out.crop((x1, y1, x2, y2))

    mask_img = mask_to_pil(mask_crop).resize(
        (crop_w, crop_h),
        Image.NEAREST,
    )

    mask_img = mask_img.filter(
        ImageFilter.GaussianBlur(radius=1.0),
    )

    composed = Image.composite(
        edited_crop,
        original_crop,
        mask_img,
    )

    out.paste(composed, (x1, y1))

    return out


def paste_edited_crop_soft(
    original_full: Image.Image,
    edited_crop: Image.Image,
    crop_box,
    mask_crop: np.ndarray,
    blur_radius: float = 4.5,
) -> Image.Image:
    out = original_full.copy().convert("RGB")

    x1, y1, x2, y2 = [int(v) for v in crop_box]
    crop_w = x2 - x1
    crop_h = y2 - y1

    edited_crop = edited_crop.convert("RGB").resize(
        (crop_w, crop_h),
        Image.LANCZOS,
    )

    original_crop = out.crop((x1, y1, x2, y2))

    mask_img = mask_to_pil(mask_crop).resize(
        (crop_w, crop_h),
        Image.NEAREST,
    )

    mask_img = mask_img.filter(
        ImageFilter.GaussianBlur(radius=float(blur_radius)),
    )

    composed = Image.composite(
        edited_crop,
        original_crop,
        mask_img,
    )

    out.paste(composed, (x1, y1))

    return out


def paste_edited_crop_rect(
    original_full: Image.Image,
    edited_crop: Image.Image,
    crop_box,
    rect_crop,
    feather: float = 4.5,
) -> Image.Image:
    out = original_full.copy().convert("RGB")

    cx1, cy1, cx2, cy2 = [int(v) for v in crop_box]
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    edited_crop = edited_crop.convert("RGB").resize(
        (crop_w, crop_h),
        Image.LANCZOS,
    )

    original_crop = out.crop((cx1, cy1, cx2, cy2))

    rx1, ry1, rx2, ry2 = [int(v) for v in rect_crop]

    rect_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    rx1 = max(0, min(rx1, crop_w))
    rx2 = max(0, min(rx2, crop_w))
    ry1 = max(0, min(ry1, crop_h))
    ry2 = max(0, min(ry2, crop_h))

    rect_mask[ry1:ry2, rx1:rx2] = 255

    mask_img = Image.fromarray(rect_mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather)))

    composed = Image.composite(
        edited_crop,
        original_crop,
        mask_img,
    )

    out.paste(composed, (cx1, cy1))

    return out


def make_comparison_image(
    original: Image.Image,
    edited: Image.Image,
    width: int = 1800,
    left_label: str = "Original",
    right_label: str = "Edited",
    gap: int = 14,
    label_height: int = 42,
) -> Image.Image:
    original = original.convert("RGB")
    edited = edited.convert("RGB")

    if edited.size != original.size:
        edited = edited.resize(original.size, Image.LANCZOS)

    single_w = (int(width) - gap) // 2
    scale = single_w / original.size[0]
    single_h = int(round(original.size[1] * scale))

    original_r = original.resize((single_w, single_h), Image.LANCZOS)
    edited_r = edited.resize((single_w, single_h), Image.LANCZOS)

    canvas = Image.new(
        "RGB",
        (single_w * 2 + gap, single_h + label_height),
        (245, 245, 245),
    )

    canvas.paste(original_r, (0, label_height))
    canvas.paste(edited_r, (single_w + gap, label_height))

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (16, 13),
        left_label,
        fill=(0, 0, 0),
    )

    draw.text(
        (single_w + gap + 16, 13),
        right_label,
        fill=(0, 0, 0),
    )

    return canvas


def save_output(
    output: dict,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    image_keys = [
        "original",
        "detected",
        "mask_overlay",
        "edited",
        "comparison",
    ]

    for key in image_keys:
        if key in output and isinstance(output[key], Image.Image):
            output[key].save(
                os.path.join(out_dir, f"{key}.png"),
            )

    if "full_mask" in output:
        mask_to_pil(output["full_mask"]).save(
            os.path.join(out_dir, "full_mask.png"),
        )

    if "metadata" in output:
        with open(
            os.path.join(out_dir, "metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output["metadata"],
                f,
                ensure_ascii=False,
                indent=2,
            )

    if "parsed_info" in output:
        with open(
            os.path.join(out_dir, "parsed_info.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(output["parsed_info"]))

    if "explanation" in output:
        with open(
            os.path.join(out_dir, "explanation.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(output["explanation"]))

    if "vlm_verification" in output and output["vlm_verification"]:
        with open(
            os.path.join(out_dir, "vlm_verification.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(output["vlm_verification"]))

    if "florence_analysis" in output and output["florence_analysis"]:
        with open(
            os.path.join(out_dir, "florence_analysis.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output["florence_analysis"],
                f,
                ensure_ascii=False,
                indent=2,
            )


def cleanup():
    gc.collect()

    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
