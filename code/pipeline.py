from PIL import Image
import numpy as np

from models import (
    GroundingDinoDetector,
    Sam2Segmenter,
    Inpainter,
    VLMVerifier,
    Florence2Analyzer,
)

from config import (
    LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
    DEFAULT_VLM_VERIFY_MODEL,
    DEFAULT_FLORENCE2_MODEL,
)

from utils import (
    parse_edit_instruction,
    build_edit_prompt,
    build_negative_prompt,
    phrase_to_english,
    clean_phrase,
    is_desk_flat_replacement,
    draw_box,
    overlay_mask,
    mask_to_pil,
    dilate_mask,
    compute_crop_scale,
    expand_box,
    translate_box_to_crop,
    place_crop_mask_to_full,
    paste_edited_crop_strict,
    paste_edited_crop_soft,
    paste_edited_crop_rect,
    make_box_mask_from_xyxy,
    make_adaptive_replacement_mask,
    make_comparison_image,
    build_replacement_suggestions,
    format_replacement_suggestions,
    cleanup,
)


class CounterfactualPipeline:
    def __init__(
        self,
        gdino_model,
        sam2_model,
        inpaint_model,
        device,
        vlm_model=DEFAULT_VLM_VERIFY_MODEL,
        florence2_model=DEFAULT_FLORENCE2_MODEL,
    ):
        self.device = device
        self.vlm_model = vlm_model
        self.florence2_model = florence2_model

        self.detector = GroundingDinoDetector(gdino_model, device)
        self.segmenter = Sam2Segmenter(sam2_model, device)
        self.inpainter = Inpainter(inpaint_model, device)

        self.verifier = None
        self.analyzer = None

    def _get_verifier(self):
        if self.verifier is None:
            self.verifier = VLMVerifier(
                model_name=self.vlm_model,
                device=self.device,
            )
        return self.verifier

    def _get_analyzer(self):
        if self.analyzer is None:
            self.analyzer = Florence2Analyzer(
                model_name=self.florence2_model,
                device=self.device,
            )
        return self.analyzer

    def analyze_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        analyzer = self._get_analyzer()
        analysis = analyzer.analyze_image(image)

        caption = analysis.get("caption", "")
        detected_objects = analysis.get("detected_objects", [])
        suggested_objects = detected_objects[:10]

        replacement_suggestions = build_replacement_suggestions(
            suggested_objects,
            max_items_per_object=5,
        )
        replacement_text = format_replacement_suggestions(replacement_suggestions)

        analysis["suggested_editable_objects"] = suggested_objects
        analysis["replacement_suggestions"] = replacement_suggestions

        analysis_text = (
            "Florence-2 Image Analysis\n"
            "=========================\n"
            f"Model: {self.florence2_model}\n\n"
            f"Caption:\n{caption}\n\n"
            f"Detected objects:\n"
            f"{', '.join(detected_objects) if detected_objects else 'None'}\n\n"
            f"Suggested editable objects:\n"
            f"{', '.join(suggested_objects) if suggested_objects else 'None'}\n\n"
            f"Suggested replacement contents:\n"
            f"{replacement_text if replacement_text else 'None'}\n"
        )

        return {
            "image": image,
            "analysis": analysis,
            "analysis_text": analysis_text,
            "replacement_suggestions": replacement_suggestions,
            "replacement_suggestions_text": replacement_text,
        }

    def run(
        self,
        image_path: str,
        instruction: str,
        manual_target_keyword: str = "",
        manual_replacement_keyword: str = "",
        box_threshold: float = 0.25,
        text_threshold: float = 0.15,
        mask_mode: str = "adaptive",
        mask_expand_ratio: float = 1.35,
        replacement_scale: float = 1.10,
        context_scale: float = 3.0,
        paste_mode: str = "soft",
        mask_feather: float = 4.5,
        inpaint_strength: float = 0.84,
        steps: int = None,
        guidance: float = None,
        seed: int = None,
        comparison_width: int = 1800,
        use_vlm_verify: bool = False,
    ):
        image = Image.open(image_path).convert("RGB")

        spec = parse_edit_instruction(
            instruction=instruction,
            manual_target=manual_target_keyword,
            manual_replacement=manual_replacement_keyword,
        )

        if spec is None:
            raise ValueError(
                "Failed to parse instruction. "
                "Use examples such as: "
                "'Replace the bag with a laptop.', "
                "'Change the clock to a poster.', "
                "'Swap the books for a laptop.', "
                "or provide --target_keyword and --replacement_keyword."
            )

        parsed_target = spec["target"]
        parsed_replacement = spec["replacement"]
        replacement_l = clean_phrase(parsed_replacement)

        is_flat = is_desk_flat_replacement(replacement_l)

        if is_flat:
            mask_expand_ratio = min(float(mask_expand_ratio), 1.28)
            replacement_scale = min(float(replacement_scale), 1.08)
            context_scale = min(float(context_scale), 3.2)
            inpaint_strength = min(float(inpaint_strength), 0.84)
            mask_feather = min(float(mask_feather), 4.5)

        detection_target = (
            manual_target_keyword.strip()
            if manual_target_keyword.strip()
            else parsed_target
        )
        detection_target = phrase_to_english(detection_target)

        if not detection_target:
            raise ValueError(
                "Target object is empty. Please provide --target_keyword."
            )

        if not parsed_replacement:
            raise ValueError(
                "Replacement object is empty. Please provide --replacement_keyword."
            )

        box, score, label = self._detect_with_retry(
            image=image,
            query=detection_target,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if box is None:
            raise ValueError(
                f"Target not found: {detection_target}. "
                f"Try using a simpler --target_keyword, e.g. 'bag', 'chair', 'person'."
            )

        if score < LOW_CONFIDENCE_THRESHOLD:
            print(
                f"[WARN] Low-confidence detection: {score:.3f}. "
                f"Continuing because generic replacement can still work."
            )

        detected = draw_box(
            image=image,
            box_xyxy=box,
            label=label,
            score=score,
        )

        auto_crop_scale = compute_crop_scale(
            box_xyxy=box,
            image_size=image.size,
            target=detection_target,
            replacement=parsed_replacement,
        )

        if is_flat:
            crop_scale = min(max(float(context_scale), 2.6), 3.2)
        else:
            crop_scale = max(float(context_scale), float(auto_crop_scale))

        crop_box = expand_box(
            box_xyxy=box,
            image_size=image.size,
            scale=crop_scale,
            min_pad=48,
        )

        crop_image = image.crop(crop_box)

        local_box = translate_box_to_crop(
            box_xyxy=box,
            crop_box=crop_box,
        )

        sam_mask = self.segmenter.segment_from_box(
            image=crop_image,
            box_xyxy=local_box,
        )

        sam_kernel = self._auto_sam_dilation_kernel(
            box_xyxy=local_box,
            image_size=crop_image.size,
        )

        sam_mask = dilate_mask(
            mask=sam_mask,
            kernel_size=sam_kernel,
            iterations=1,
        )

        box_mask = make_box_mask_from_xyxy(
            box_xyxy=local_box,
            image_size=crop_image.size,
            expand_ratio=mask_expand_ratio,
        )

        adaptive_mask, adaptive_rect = make_adaptive_replacement_mask(
            box_xyxy=local_box,
            image_size=crop_image.size,
            replacement=parsed_replacement,
            target=parsed_target,
            scale=float(replacement_scale),
            base_expand=float(mask_expand_ratio),
        )

        if mask_mode == "sam":
            local_edit_mask = sam_mask.astype("uint8")
        elif mask_mode == "box":
            local_edit_mask = box_mask.astype("uint8")
        elif mask_mode == "hybrid":
            local_edit_mask = np.maximum(sam_mask, box_mask).astype("uint8")
        elif mask_mode == "adaptive":
            local_edit_mask = np.maximum(sam_mask, adaptive_mask).astype("uint8")
        elif mask_mode == "replace_box":
            local_edit_mask = adaptive_mask.astype("uint8")
        else:
            raise ValueError(f"Unsupported mask_mode: {mask_mode}")

        final_dilate_kernel = 5 if is_flat else 7

        local_edit_mask = dilate_mask(
            mask=local_edit_mask,
            kernel_size=final_dilate_kernel,
            iterations=1,
        )

        prompt = build_edit_prompt(spec)
        negative_prompt = build_negative_prompt(spec)

        mask_pil = mask_to_pil(local_edit_mask)

        edited_crop = self.inpainter.edit(
            image=crop_image,
            mask=mask_pil,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=DEFAULT_STEPS if steps is None else int(steps),
            guidance=DEFAULT_GUIDANCE if guidance is None else float(guidance),
            strength=float(inpaint_strength),
            seed=seed,
        )

        if paste_mode == "rect" or mask_mode == "replace_box":
            edited_full = paste_edited_crop_rect(
                original_full=image,
                edited_crop=edited_crop,
                crop_box=crop_box,
                rect_crop=adaptive_rect,
                feather=float(mask_feather),
            )
        elif paste_mode == "strict":
            edited_full = paste_edited_crop_strict(
                original_full=image,
                edited_crop=edited_crop,
                crop_box=crop_box,
                mask_crop=local_edit_mask,
            )
        else:
            edited_full = paste_edited_crop_soft(
                original_full=image,
                edited_crop=edited_crop,
                crop_box=crop_box,
                mask_crop=local_edit_mask,
                blur_radius=float(mask_feather),
            )

        full_mask = place_crop_mask_to_full(
            mask_crop=local_edit_mask,
            crop_box=crop_box,
            full_size=image.size,
        )

        mask_overlay = overlay_mask(
            image=image,
            mask=full_mask,
        )

        comparison = make_comparison_image(
            original=image,
            edited=edited_full,
            width=comparison_width,
            left_label="Original",
            right_label="Edited",
        )

        vlm_verification = ""

        if use_vlm_verify:
            verifier = self._get_verifier()

            vlm_verification = verifier.verify(
                original=image,
                edited=edited_full,
                instruction=instruction,
                target=parsed_target,
                replacement=parsed_replacement,
            )

        parsed_info = (
            f"Parsed target object: {parsed_target}\n"
            f"Parsed replacement: {parsed_replacement}\n"
            f"Detection keyword used: {detection_target}\n"
            f"Detected label: {label}\n"
            f"Detection score: {score:.4f}\n"
            f"Crop box: {crop_box}\n"
            f"Auto crop scale: {auto_crop_scale:.2f}\n"
            f"Final crop scale: {crop_scale:.2f}\n"
            f"SAM dilation kernel: {sam_kernel}\n"
            f"Final dilation kernel: {final_dilate_kernel}\n"
            f"Mask mode: {mask_mode}\n"
            f"Mask expand ratio: {mask_expand_ratio:.2f}\n"
            f"Replacement scale: {replacement_scale:.2f}\n"
            f"Context scale: {context_scale:.2f}\n"
            f"Paste mode: {paste_mode}\n"
            f"Mask feather: {mask_feather:.2f}\n"
            f"Inpaint strength: {inpaint_strength:.2f}\n"
            f"Adaptive rect in crop: {adaptive_rect}\n"
            f"Flat replacement mode: {is_flat}\n"
            f"Seed: {seed}\n"
            f"VLM verification enabled: {use_vlm_verify}\n\n"
            f"Inpainting prompt:\n{prompt}\n\n"
            f"Negative prompt:\n{negative_prompt}"
        )

        explanation = (
            f"Instruction: {instruction}\n\n"
            f"- Target region: {parsed_target}\n"
            f"- Intended replacement: {parsed_replacement}\n"
            f"- Detection confidence: {score:.3f}\n"
            f"- Mask mode: {mask_mode}\n"
            f"- Adaptive replacement region: enabled\n"
            f"- Flat replacement constraints: {'enabled' if is_flat else 'disabled'}\n"
            f"- New object is not restricted to the old object's exact silhouette.\n"
            f"- Comparison image: enabled\n"
            f"- VLM verification: {'enabled' if use_vlm_verify else 'disabled'}"
        )

        metadata = {
            "image_path": image_path,
            "instruction": instruction,
            "spec": spec,
            "detection_keyword_used": detection_target,
            "detected_label": label,
            "detection_score": float(score),
            "detected_box_xyxy": [float(x) for x in box],
            "crop_box_xyxy": [int(v) for v in crop_box],
            "auto_crop_scale": float(auto_crop_scale),
            "crop_scale": float(crop_scale),
            "sam_dilation_kernel": int(sam_kernel),
            "final_dilation_kernel": int(final_dilate_kernel),
            "mask_mode": mask_mode,
            "mask_expand_ratio": float(mask_expand_ratio),
            "replacement_scale": float(replacement_scale),
            "context_scale": float(context_scale),
            "paste_mode": paste_mode,
            "mask_feather": float(mask_feather),
            "inpaint_strength": float(inpaint_strength),
            "adaptive_rect_crop": [int(v) for v in adaptive_rect],
            "flat_replacement_mode": bool(is_flat),
            "seed": seed,
            "device": self.device,
            "vlm_verification_enabled": bool(use_vlm_verify),
            "vlm_model": self.vlm_model if use_vlm_verify else None,
        }

        cleanup()

        return {
            "original": image,
            "detected": detected,
            "mask_overlay": mask_overlay,
            "edited": edited_full,
            "comparison": comparison,
            "full_mask": full_mask,
            "parsed_info": parsed_info,
            "explanation": explanation,
            "vlm_verification": vlm_verification,
            "metadata": metadata,
        }

    def _detect_with_retry(
        self,
        image,
        query: str,
        box_threshold: float,
        text_threshold: float,
    ):
        retry_settings = [
            (box_threshold, text_threshold),
            (min(box_threshold, 0.20), min(text_threshold, 0.12)),
            (min(box_threshold, 0.15), min(text_threshold, 0.10)),
            (min(box_threshold, 0.10), min(text_threshold, 0.08)),
        ]

        best = (None, None, None)

        for bt, tt in retry_settings:
            box, score, label = self.detector.detect_best_box(
                image=image,
                query=query,
                box_threshold=float(bt),
                text_threshold=float(tt),
            )

            if box is not None:
                if best[0] is None or float(score) > float(best[1]):
                    best = (box, score, label)

                if float(score) >= LOW_CONFIDENCE_THRESHOLD:
                    return box, score, label

        return best

    def _auto_sam_dilation_kernel(self, box_xyxy, image_size):
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        ref = min(bw, bh)
        kernel = int(round(ref * 0.05))

        kernel = max(5, min(kernel, 21))

        if kernel % 2 == 0:
            kernel += 1

        return kernel
