import time
import traceback
from pathlib import Path

import gradio as gr

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
from utils import parse_edit_instruction, save_output


PROJECT_DIR = Path("/workspace/deeplearning")
TMP_DIR = PROJECT_DIR / "tmp"
OUTPUT_DIR = PROJECT_DIR / "outputs"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE = None


def get_pipeline():
    global PIPELINE

    if PIPELINE is None:
        PIPELINE = CounterfactualPipeline(
            gdino_model=DEFAULT_GDINO_MODEL,
            sam2_model=DEFAULT_SAM2_MODEL,
            inpaint_model=DEFAULT_INPAINT_MODEL,
            device=DEFAULT_DEVICE,
            vlm_model=DEFAULT_VLM_VERIFY_MODEL,
            florence2_model=DEFAULT_FLORENCE2_MODEL,
        )

    return PIPELINE


def parse_instruction_preview(instruction):
    spec = parse_edit_instruction(
        instruction=instruction,
        manual_target="",
        manual_replacement="",
    )

    if spec is None:
        return "", "", "Failed to parse instruction. Please use: Replace the A with B."

    text = (
        "Instruction Parsing Result\n"
        "==========================\n"
        f"Parsed target: {spec['target']}\n"
        f"Parsed replacement: {spec['replacement']}\n"
        f"Edit mode: {spec['edit_mode']}\n"
        f"Location hint: {spec['location_hint']}\n"
        f"Geometry: {spec['geometry']}\n"
    )

    return spec["target"], spec["replacement"], text


def analyze_uploaded_image(image):
    if image is None:
        return "", "", "Please upload an image first."

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_path = TMP_DIR / f"florence_analyze_{timestamp}.png"

    image = image.convert("RGB")
    image.save(input_path)

    try:
        pipeline = get_pipeline()
        output = pipeline.analyze_image(str(input_path))

        analysis = output["analysis"]
        objects = analysis.get("detected_objects", [])
        replacement_text = output.get("replacement_suggestions_text", "")

        object_text = ", ".join(objects) if objects else ""
        analysis_text = output["analysis_text"]

        return object_text, replacement_text, analysis_text

    except Exception:
        return "", "", traceback.format_exc()


def run_editing(
    image,
    instruction,
    target_keyword,
    replacement_keyword,
    mask_mode,
    paste_mode,
    mask_expand_ratio,
    replacement_scale,
    context_scale,
    mask_feather,
    inpaint_strength,
    steps,
    guidance,
    seed,
    box_threshold,
    text_threshold,
    use_vlm_verify,
):
    if image is None:
        return None, None, None, None, "Please upload an image first."

    if not instruction or not instruction.strip():
        return None, None, None, None, "Please enter an editing instruction."

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_path = TMP_DIR / f"input_{timestamp}.png"
    out_dir = OUTPUT_DIR / f"demo_{timestamp}"

    image = image.convert("RGB")
    image.save(input_path)

    try:
        pipeline = get_pipeline()

        seed_value = None
        if seed is not None and str(seed).strip() != "":
            seed_value = int(seed)

        steps_value = None
        if steps is not None and int(steps) > 0:
            steps_value = int(steps)

        guidance_value = None
        if guidance is not None and float(guidance) > 0:
            guidance_value = float(guidance)

        output = pipeline.run(
            image_path=str(input_path),
            instruction=instruction,
            manual_target_keyword=target_keyword or "",
            manual_replacement_keyword=replacement_keyword or "",
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            mask_mode=mask_mode,
            mask_expand_ratio=float(mask_expand_ratio),
            replacement_scale=float(replacement_scale),
            context_scale=float(context_scale),
            paste_mode=paste_mode,
            mask_feather=float(mask_feather),
            inpaint_strength=float(inpaint_strength),
            steps=steps_value,
            guidance=guidance_value,
            seed=seed_value,
            comparison_width=1800,
            use_vlm_verify=bool(use_vlm_verify),
        )

        save_output(output, str(out_dir))

        log_text = (
            "[DONE]\n"
            f"Saved to: {out_dir}\n\n"
            "===== Parsed Info =====\n"
            f"{output['parsed_info']}\n\n"
            "===== Explanation =====\n"
            f"{output['explanation']}\n"
        )

        if output.get("vlm_verification"):
            log_text += "\n\n===== Qwen-VL Verification =====\n"
            log_text += output["vlm_verification"]

        return (
            output["detected"],
            output["mask_overlay"],
            output["edited"],
            output["comparison"],
            log_text,
        )

    except Exception:
        error_text = traceback.format_exc()
        return None, None, None, None, error_text


with gr.Blocks(title="Multi-Foundation-Model Counterfactual Image Editing") as demo:
    gr.Markdown(
        """
        # Multi-Foundation-Model Counterfactual Image Editing Demo

        This demo integrates multiple foundation models:

        **Florence-2** for image analysis and replacement recommendation,  
        **Grounding DINO** for open-vocabulary object detection,  
        **SAM2** for target segmentation,  
        **SDXL Inpainting** for image editing,  
        **Qwen2.5-VL** for result verification.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Input Image",
            )

            analyze_button = gr.Button(
                "Analyze Image with Florence-2",
                variant="secondary",
            )

            suggested_objects_box = gr.Textbox(
                label="Suggested Editable Objects",
                lines=2,
                placeholder="Click Analyze Image to get editable object suggestions.",
            )

            replacement_suggestions_box = gr.Textbox(
                label="Suggested Replacement Contents",
                lines=8,
                placeholder="Florence-2 analysis will recommend possible replacement contents.",
            )

            analysis_box = gr.Textbox(
                label="Florence-2 Analysis",
                lines=10,
            )

            instruction = gr.Textbox(
                label="Instruction",
                value="Replace the clock with a framed Van Gogh painting.",
                lines=2,
            )

            parse_button = gr.Button(
                "Parse Instruction",
                variant="secondary",
            )

            parse_log = gr.Textbox(
                label="Parsed Instruction",
                lines=6,
            )

            target_keyword = gr.Textbox(
                label="Target Keyword",
                value="",
                placeholder="Optional. Leave empty to parse automatically.",
            )

            replacement_keyword = gr.Textbox(
                label="Replacement Keyword",
                value="",
                placeholder="Optional. Leave empty to parse automatically.",
            )

            use_vlm_verify = gr.Checkbox(
                label="Use Qwen2.5-VL Verification",
                value=True,
            )

            run_button = gr.Button(
                "Run Editing",
                variant="primary",
            )

            with gr.Accordion("Advanced Settings", open=False):
                mask_mode = gr.Dropdown(
                    choices=["sam", "box", "hybrid", "adaptive", "replace_box"],
                    value="adaptive",
                    label="Mask Mode",
                )

                paste_mode = gr.Dropdown(
                    choices=["strict", "soft", "rect"],
                    value="soft",
                    label="Paste Mode",
                )

                mask_expand_ratio = gr.Slider(
                    minimum=1.0,
                    maximum=2.5,
                    value=1.20,
                    step=0.05,
                    label="Mask Expand Ratio",
                )

                replacement_scale = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.00,
                    step=0.05,
                    label="Replacement Scale",
                )

                context_scale = gr.Slider(
                    minimum=1.0,
                    maximum=6.0,
                    value=3.0,
                    step=0.1,
                    label="Context Scale",
                )

                mask_feather = gr.Slider(
                    minimum=0.0,
                    maximum=12.0,
                    value=3.5,
                    step=0.5,
                    label="Mask Feather",
                )

                inpaint_strength = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.80,
                    step=0.05,
                    label="Inpaint Strength",
                )

                steps = gr.Number(
                    value=45,
                    precision=0,
                    label="Inpainting Steps",
                )

                guidance = gr.Number(
                    value=7.5,
                    precision=1,
                    label="Guidance Scale",
                )

                seed = gr.Number(
                    value=42,
                    precision=0,
                    label="Seed",
                )

                box_threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.60,
                    value=DEFAULT_BOX_THRESHOLD,
                    step=0.01,
                    label="Grounding DINO Box Threshold",
                )

                text_threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.60,
                    value=DEFAULT_TEXT_THRESHOLD,
                    step=0.01,
                    label="Grounding DINO Text Threshold",
                )

        with gr.Column(scale=1):
            detected_output = gr.Image(
                label="Detection Result",
            )

            mask_output = gr.Image(
                label="Mask Overlay",
            )

            edited_output = gr.Image(
                label="Edited Image",
            )

            comparison_output = gr.Image(
                label="Side-by-Side Comparison",
            )

    log_box = gr.Textbox(
        label="Logs and Qwen-VL Verification",
        lines=20,
    )

    analyze_button.click(
        fn=analyze_uploaded_image,
        inputs=[input_image],
        outputs=[suggested_objects_box, replacement_suggestions_box, analysis_box],
    )

    parse_button.click(
        fn=parse_instruction_preview,
        inputs=[instruction],
        outputs=[target_keyword, replacement_keyword, parse_log],
    )

    run_button.click(
        fn=run_editing,
        inputs=[
            input_image,
            instruction,
            target_keyword,
            replacement_keyword,
            mask_mode,
            paste_mode,
            mask_expand_ratio,
            replacement_scale,
            context_scale,
            mask_feather,
            inpaint_strength,
            steps,
            guidance,
            seed,
            box_threshold,
            text_threshold,
            use_vlm_verify,
        ],
        outputs=[
            detected_output,
            mask_output,
            edited_output,
            comparison_output,
            log_box,
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=8888,
        share=False,
    )
