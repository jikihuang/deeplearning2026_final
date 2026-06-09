import torch

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_GDINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-base-plus"
DEFAULT_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

DEFAULT_VLM_VERIFY_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_FLORENCE2_MODEL = "microsoft/Florence-2-base"

DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.15

LOW_CONFIDENCE_THRESHOLD = 0.18

DEFAULT_STEPS = 45
DEFAULT_GUIDANCE = 7.5

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, malformed, "
    "bad anatomy, bad perspective, wrong scale, floating object, duplicated object, "
    "extra object, cut off object, unnatural shadow, inconsistent lighting, "
    "changed background, changed people, changed face, warped scene, artifacts"
)

LARGE_REPLACEMENT_WORDS = {
    "person", "man", "woman", "boy", "girl", "child", "human",
    "chair", "sofa", "couch", "table", "desk", "bed", "cabinet",
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
    "dog", "cat", "horse", "tree", "plant",
}

TALL_REPLACEMENT_WORDS = {
    "person", "man", "woman", "boy", "girl", "child", "human",
    "bottle", "vase", "lamp", "tree", "plant", "statue",
}

WIDE_REPLACEMENT_WORDS = {
    "laptop", "computer", "keyboard", "monitor", "screen",
    "car", "truck", "bus", "bicycle", "motorcycle", "sofa", "couch",
    "bed", "table", "desk", "poster", "painting", "whiteboard", "blackboard",
}

FLAT_SURFACE_WORDS = {
    "laptop", "computer", "keyboard", "book", "notebook", "tablet",
    "phone", "plate", "cup", "bottle", "vase", "bag", "backpack",
}

WALL_OBJECT_WORDS = {
    "poster", "painting", "picture", "photo", "clock", "whiteboard",
    "blackboard", "mirror", "window", "sign", "screen", "monitor",
}

PERSON_WORDS = {
    "person", "man", "woman", "boy", "girl", "child", "human",
}

VEHICLE_WORDS = {
    "car", "truck", "bus", "bicycle", "motorcycle", "scooter",
}

COLOR_WORDS = {
    "red", "blue", "green", "black", "white", "yellow", "purple",
    "pink", "orange", "brown", "gray", "grey", "silver", "gold",
}

DESK_FLAT_OBJECT_WORDS = {
    "laptop", "computer", "tablet", "phone", "book", "notebook",
    "keyboard", "mouse",
}

REPLACEMENT_SUGGESTION_DB = {
    "clock": [
        "framed Van Gogh painting",
        "wall poster",
        "round mirror",
        "whiteboard",
        "landscape photo",
    ],
    "wall clock": [
        "framed Van Gogh painting",
        "modern wall poster",
        "round mirror",
        "calendar",
        "framed landscape photo",
    ],
    "bag": [
        "open laptop",
        "teddy bear",
        "toy robot",
        "camera",
        "flower vase",
    ],
    "backpack": [
        "open laptop",
        "camera",
        "teddy bear",
        "toy robot",
        "small storage box",
    ],
    "book": [
        "tablet",
        "laptop",
        "small plant",
        "notebook",
        "decorative box",
    ],
    "books": [
        "tablet",
        "laptop",
        "small plant",
        "decorative box",
        "stack of magazines",
    ],
    "chair": [
        "wooden stool",
        "standing person",
        "small sofa",
        "potted plant",
        "coat rack",
    ],
    "table": [
        "modern desk",
        "workbench",
        "small cabinet",
        "dining table",
    ],
    "desk": [
        "small cabinet",
        "modern table",
        "workstation",
        "wooden desk",
    ],
    "cup": [
        "small plant",
        "candle",
        "flower vase",
        "toy figure",
    ],
    "mug": [
        "small plant",
        "candle",
        "flower vase",
        "toy figure",
    ],
    "bottle": [
        "flower vase",
        "thermos",
        "decorative sculpture",
        "lamp",
    ],
    "poster": [
        "whiteboard",
        "framed painting",
        "mirror",
        "calendar",
    ],
    "picture": [
        "poster",
        "whiteboard",
        "mirror",
        "calendar",
    ],
    "photo": [
        "poster",
        "whiteboard",
        "mirror",
        "calendar",
    ],
    "monitor": [
        "small TV",
        "whiteboard",
        "framed artwork",
        "large plant",
    ],
    "screen": [
        "small TV",
        "whiteboard",
        "framed artwork",
        "large plant",
    ],
    "keyboard": [
        "closed notebook",
        "tablet",
        "decorative tray",
        "wireless keyboard",
    ],
    "mouse": [
        "wireless earbud case",
        "small toy car",
        "calculator",
        "small figurine",
    ],
    "laptop": [
        "tablet",
        "closed notebook",
        "small plant",
        "book stack",
        "drawing tablet",
    ],
    "sofa": [
        "bed",
        "small bookshelf",
        "large potted plant",
        "armchair",
    ],
    "plant": [
        "flower vase",
        "desk lamp",
        "small sculpture",
        "toy figure",
    ],
    "potted plant": [
        "flower vase",
        "desk lamp",
        "small sculpture",
        "toy figure",
    ],
}
