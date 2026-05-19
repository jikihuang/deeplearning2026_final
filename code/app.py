import argparse
import os

from config import (
    DEFAULT_DEVICE,
    DEFAULT_GDINO_MODEL,
    DEFAULT_SAM2_MODEL,
    DEFAULT_INPAINT_MODEL,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
)
from pipeline import CounterfactualPipeline
from utils import save_output


def main():
    parser = argparse.ArgumentParser(
        description="Generic and natural object replacement pipeline with GroundingDINO + SAM2 + SDXL Inpainting."
    )

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--target_keyword", type=str, default="")
    parser.add_argument("--replacement_keyword", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--gdino_model", type=str, default=DEFAULT_GDINO_MODEL)
    parser.add_argument("--sam2_model", type=str, default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--inpaint_model", type=str, default=DEFAULT_INPAINT_MODEL)

    parser.add_argument("--box_threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text_threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)

    parser.add_argument(
        "--mask_mode",
        type=str,
        default="adaptive",
        choices=["sam", "box", "hybrid", "adaptive", "replace_box"],
        help=(
            "sam: use SAM object mask only; "
            "box: use expanded bbox; "
            "hybrid: SAM + expanded bbox; "
            "adaptive: SAM + replacement-aware region; "
            "replace_box: replacement-aware rectangle only."
        ),
    )

    parser.add_argument(
        "--mask_expand_ratio",
        type=float,
        default=1.35,
        help="Base expansion ratio around the detected target box.",
    )

    parser.add_argument(
        "--replacement_scale",
        type=float,
        default=1.10,
        help="Extra scale for the replacement object region. Increase this for small-to-large replacement.",
    )

    parser.add_argument(
        "--context_scale",
        type=float,
        default=3.0,
        help="Crop context scale. Larger gives the inpainting model more scene context.",
    )

    parser.add_argument(
        "--mask_feather",
        type=float,
        default=4.5,
        help="Boundary feathering radius when pasting the edited crop back.",
    )

    parser.add_argument(
        "--paste_mode",
        type=str,
        default="soft",
        choices=["strict", "soft", "rect"],
        help="strict: paste by mask; soft: feathered mask; rect: feathered adaptive rectangle.",
    )

    parser.add_argument(
        "--inpaint_strength",
        type=float,
        default=0.84,
        help="Lower is more faithful to the original; higher gives more generation freedom.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inpainting denoising steps. If None, use default config.",
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Classifier-free guidance scale. If None, use default config.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible inpainting. If None, use random generation.",
    )

    parser.add_argument(
        "--comparison_width",
        type=int,
        default=1800,
        help="Width of the saved side-by-side comparison image.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    pipeline = CounterfactualPipeline(
        gdino_model=args.gdino_model,
        sam2_model=args.sam2_model,
        inpaint_model=args.inpaint_model,
        device=args.device,
    )

    output = pipeline.run(
        image_path=args.image_path,
        instruction=args.instruction,
        manual_target_keyword=args.target_keyword,
        manual_replacement_keyword=args.replacement_keyword,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        mask_mode=args.mask_mode,
        mask_expand_ratio=args.mask_expand_ratio,
        replacement_scale=args.replacement_scale,
        context_scale=args.context_scale,
        paste_mode=args.paste_mode,
        mask_feather=args.mask_feather,
        inpaint_strength=args.inpaint_strength,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        comparison_width=args.comparison_width,
    )

    save_output(output, args.out_dir)

    print("[DONE]")
    print(f"Saved to: {os.path.abspath(args.out_dir)}")
    print()
    print(output["parsed_info"])
    print()
    print(output["explanation"])


if __name__ == "__main__":
    main()