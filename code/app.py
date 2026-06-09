import argparse
import os
import json

from config import (
    DEFAULT_DEVICE,
    DEFAULT_GDINO_MODEL,
    DEFAULT_SAM2_MODEL,
    DEFAULT_INPAINT_MODEL,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    DEFAULT_VLM_VERIFY_MODEL,
    DEFAULT_FLORENCE2_MODEL,
)
from pipeline import CounterfactualPipeline
from utils import save_output


def main():
    parser = argparse.ArgumentParser(
        description="Generic and natural object replacement pipeline with Florence-2 + GroundingDINO + SAM2 + SDXL Inpainting + Qwen-VL verification."
    )

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--target_keyword", type=str, default="")
    parser.add_argument("--replacement_keyword", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--gdino_model", type=str, default=DEFAULT_GDINO_MODEL)
    parser.add_argument("--sam2_model", type=str, default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--inpaint_model", type=str, default=DEFAULT_INPAINT_MODEL)
    parser.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_VERIFY_MODEL)
    parser.add_argument("--florence2_model", type=str, default=DEFAULT_FLORENCE2_MODEL)

    parser.add_argument("--box_threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text_threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)

    parser.add_argument("--mask_mode", type=str, default="adaptive", choices=["sam", "box", "hybrid", "adaptive", "replace_box"])
    parser.add_argument("--mask_expand_ratio", type=float, default=1.35)
    parser.add_argument("--replacement_scale", type=float, default=1.10)
    parser.add_argument("--context_scale", type=float, default=3.0)
    parser.add_argument("--mask_feather", type=float, default=4.5)
    parser.add_argument("--paste_mode", type=str, default="soft", choices=["strict", "soft", "rect"])
    parser.add_argument("--inpaint_strength", type=float, default=0.84)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--comparison_width", type=int, default=1800)

    parser.add_argument("--use_vlm_verify", action="store_true")
    parser.add_argument("--analyze_image_only", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    pipeline = CounterfactualPipeline(
        gdino_model=args.gdino_model,
        sam2_model=args.sam2_model,
        inpaint_model=args.inpaint_model,
        device=args.device,
        vlm_model=args.vlm_model,
        florence2_model=args.florence2_model,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    if args.analyze_image_only:
        analysis_output = pipeline.analyze_image(args.image_path)

        txt_path = os.path.join(args.out_dir, "florence_analysis.txt")
        json_path = os.path.join(args.out_dir, "florence_analysis.json")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(analysis_output["analysis_text"])

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                analysis_output["analysis"],
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("[DONE]")
        print(f"Florence-2 analysis saved to: {txt_path}")
        print()
        print(analysis_output["analysis_text"])
        return

    if not args.instruction.strip():
        raise ValueError("Editing mode requires --instruction.")

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
        use_vlm_verify=args.use_vlm_verify,
    )

    save_output(output, args.out_dir)

    print("[DONE]")
    print(f"Saved to: {os.path.abspath(args.out_dir)}")
    print()
    print(output["parsed_info"])
    print()
    print(output["explanation"])

    if output.get("vlm_verification"):
        print()
        print(output["vlm_verification"])


if __name__ == "__main__":
    main()
