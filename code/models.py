import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForCausalLM,
    Sam2Processor,
    Sam2Model,
    Qwen2_5_VLForConditionalGeneration,
)
from diffusers import AutoPipelineForInpainting

from utils import phrase_to_english, get_torch_dtype


def _move_inputs_to_device(inputs, device: str, dtype=None):
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if dtype is not None and torch.is_floating_point(value):
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device)
    return inputs


class GroundingDinoDetector:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None

    def load(self):
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                use_fast=False,
            )

            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name,
            ).to(self.device)

            self.model.eval()

    @torch.inference_mode()
    def detect_best_box(
        self,
        image: Image.Image,
        query: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.15,
    ):
        self.load()

        query = phrase_to_english(query)
        if not query:
            return None, None, None

        text_prompt = f"a {query}."

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        if len(results["boxes"]) == 0:
            return None, None, None

        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()

        if "text_labels" in results:
            labels = list(results["text_labels"])
        else:
            labels = list(results["labels"])

        best_idx = self._select_best_box(
            image=image,
            boxes=boxes,
            scores=scores,
            query=query,
        )

        return boxes[best_idx], float(scores[best_idx]), str(labels[best_idx])

    def _select_best_box(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        scores: np.ndarray,
        query: str,
    ) -> int:
        if len(boxes) == 1:
            return 0

        w, h = image.size
        query_l = phrase_to_english(query)

        tiny_or_desk_like = {
            "pen", "pencil", "marker", "eraser", "cup", "bottle",
            "phone", "book", "notebook", "bag", "backpack",
            "laptop", "keyboard", "mouse",
        }

        if query_l not in tiny_or_desk_like:
            return int(np.argmax(scores))

        best_idx = 0
        best_value = -1e9
        image_area = max(1.0, float(w * h))

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            area_ratio = (bw * bh) / image_area
            bottomness = y2 / max(1.0, h)

            value = (
                0.72 * float(score)
                + 0.18 * float(np.log(area_ratio + 1e-6) + 14.0) / 14.0
                + 0.10 * float(bottomness)
            )

            if value > best_value:
                best_value = value
                best_idx = i

        return int(best_idx)


class Sam2Segmenter:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None

    def load(self):
        if self.model is None:
            dtype = torch.float32
            self.processor = Sam2Processor.from_pretrained(self.model_name)

            try:
                self.model = Sam2Model.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                ).to(self.device)
            except TypeError:
                self.model = Sam2Model.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                ).to(self.device)

            self.model.eval()

    @torch.inference_mode()
    def segment_from_box(self, image: Image.Image, box_xyxy: np.ndarray) -> np.ndarray:
        self.load()

        input_boxes = [[box_xyxy.tolist()]]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        inputs = _move_inputs_to_device(inputs, self.device, dtype=model_dtype)

        outputs = self.model(
            **inputs,
            multimask_output=False,
        )

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )

        mask = masks[0]

        while hasattr(mask, "ndim") and mask.ndim > 2:
            mask = mask[0]

        if torch.is_tensor(mask):
            mask = mask.numpy()

        return (mask > 0).astype(np.uint8)


class Inpainter:
    def __init__(self, model_name: str, device: str, max_side: int = 1024):
        self.model_name = model_name
        self.device = device
        self.max_side = max_side
        self.pipe = None

    def load(self):
        if self.pipe is None:
            dtype = get_torch_dtype(self.device, prefer_fp16=True)

            try:
                self.pipe = AutoPipelineForInpainting.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                )
            except TypeError:
                self.pipe = AutoPipelineForInpainting.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                )

            self.pipe.enable_attention_slicing()

            try:
                self.pipe.vae.enable_slicing()
            except Exception:
                pass

            if self.device.startswith("cuda"):
                self.pipe = self.pipe.to(self.device)

    def _resize_keep_aspect(self, image: Image.Image, mask: Image.Image):
        w, h = image.size
        scale = min(self.max_side / max(w, h), 1.0)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        new_w = max(8, (new_w // 8) * 8)
        new_h = max(8, (new_h // 8) * 8)

        image_r = image.resize((new_w, new_h), Image.LANCZOS)
        mask_r = mask.resize((new_w, new_h), Image.NEAREST)

        return image_r, mask_r

    @torch.inference_mode()
    def edit(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        steps: int = 45,
        guidance: float = 7.5,
        strength: float = 0.84,
        seed: int = None,
    ) -> Image.Image:
        self.load()

        orig_size = image.size
        image_r, mask_r = self._resize_keep_aspect(image, mask)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_r,
            mask_image=mask_r,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=generator,
        ).images[0]

        return result.resize(orig_size, Image.LANCZOS)


class VLMVerifier:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def load(self):
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False,
            )

            dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32

            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    trust_remote_code=True,
                ).to(self.device)
            except TypeError:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(self.device)

            self.model.eval()

    @torch.inference_mode()
    def answer(self, image: Image.Image, question: str) -> str:
        self.load()

        from qwen_vl_utils import process_vision_info

        image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        inputs = _move_inputs_to_device(inputs, self.device, dtype=model_dtype)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=int(self.max_new_tokens),
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            use_cache=True,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()

    def verify(
        self,
        original: Image.Image,
        edited: Image.Image,
        instruction: str,
        target: str,
        replacement: str,
    ) -> str:
        target = str(target).strip()
        replacement = str(replacement).strip()

        original_question = (
            f"You are verifying an image editing pipeline. "
            f"Does the original image contain the target object '{target}'? "
            f"Answer briefly and mention the visual evidence."
        )

        edited_question = (
            f"The editing instruction was: '{instruction}'. "
            f"The target object is '{target}', and the intended replacement is '{replacement}'. "
            f"Please verify the edited image briefly. "
            f"Answer these four points: "
            f"1) Is the target object removed or changed? "
            f"2) Is the replacement object visible? "
            f"3) Does the edited image look realistic? "
            f"4) Overall, is the edit successful?"
        )

        original_answer = self.answer(original, original_question)
        edited_answer = self.answer(edited, edited_question)

        verification = (
            "Qwen-VL Verification Report\n"
            "===========================\n"
            f"Model: {self.model_name}\n"
            f"Instruction: {instruction}\n"
            f"Target object: {target}\n"
            f"Replacement object: {replacement}\n\n"
            "Original Image Check\n"
            "--------------------\n"
            f"Question:\n{original_question}\n\n"
            f"Answer:\n{original_answer}\n\n"
            "Edited Image Check\n"
            "------------------\n"
            f"Question:\n{edited_question}\n\n"
            f"Answer:\n{edited_answer}\n"
        )

        return verification


class Florence2Analyzer:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def load(self):
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            dtype = torch.float32

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    dtype=dtype,
                ).to(self.device)
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    torch_dtype=dtype,
                ).to(self.device)

            self.model.eval()

    @torch.inference_mode()
    def run_task(self, image: Image.Image, task_prompt: str):
        self.load()

        image = image.convert("RGB")

        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype
        inputs = _move_inputs_to_device(inputs, self.device, dtype=model_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=int(self.max_new_tokens),
            do_sample=False,
            num_beams=1,
            use_cache=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

        return parsed_answer

    def caption(self, image: Image.Image) -> str:
        parsed = self.run_task(image, "<MORE_DETAILED_CAPTION>")

        if isinstance(parsed, dict):
            value = parsed.get("<MORE_DETAILED_CAPTION>", "")
            return str(value).strip()

        return str(parsed).strip()

    def detect_objects(self, image: Image.Image):
        parsed = self.run_task(image, "<OD>")

        if isinstance(parsed, dict):
            result = parsed.get("<OD>", {})
        else:
            result = {}

        labels = result.get("labels", []) if isinstance(result, dict) else []
        bboxes = result.get("bboxes", []) if isinstance(result, dict) else []

        clean_labels = []
        for label in labels:
            label = str(label).strip().lower()
            if label:
                clean_labels.append(label)

        return clean_labels, bboxes

    def analyze_image(self, image: Image.Image):
        caption = self.caption(image)
        labels, bboxes = self.detect_objects(image)

        object_counts = {}
        for label in labels:
            object_counts[label] = object_counts.get(label, 0) + 1

        unique_objects = sorted(object_counts.keys())

        analysis = {
            "caption": caption,
            "detected_objects": unique_objects,
            "object_counts": object_counts,
            "bboxes": bboxes,
        }

        return analysis
