# deeplearning2026_final

This project implements a multimodal foundation-model pipeline for visual understanding and image editing.

The project includes:

- Image analysis and replacement recommendation
- Open-vocabulary object detection
- Point-based object segmentation
- Visual question answering
- Counterfactual object replacement / image editing

## Foundation Models

This project uses the following foundation models:

- **Florence-2** for open-vocabulary detection  
  `microsoft/Florence-2-base`
  
- **Grounding DINO** for image analysis and replacement recommendation  
  `IDEA-Research/grounding-dino-base`

- **SAM2** for object segmentation  
  `facebook/sam2.1-hiera-base-plus`

- **SDXL Inpainting** for counterfactual object replacement  
  `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`

- **Qwen3-VL** for visual question answering  
  `Qwen/Qwen3-VL-4B-Instruct`

---

## Installation Guide

It is recommended to use **Python 3.10 or later**.

Install the required dependencies with:

```bash
pip install -r requirements.txt

1. app.py

Function

Runs the full counterfactual object replacement pipeline.

The pipeline first detects the target object using Grounding DINO, segments the object region with SAM2, builds an adaptive replacement mask, and then uses SDXL Inpainting to generate the replacement object naturally inside the original image.

Input

Image file
Natural language editing instruction
Target object keyword
Replacement object keyword

Output

Original image
Detection visualization
Mask overlay
Edited image
Side-by-side comparison image
Metadata JSON file

python
app.py
--image_path
assets/input_examples/xai506_example_image.jpg
- -instruction
"Replace the bag with a potted plant sitting on the desk."
--target_keyword
bag
--replacement_keyword
"potted plant"
--out_dir
assets/result_examples/counterfactual_bag_laptop
--device
cuda
--mask_mode
adaptive
--mask_expand_ratio
1.25
--replacement_scale
1.05
--context_scale
3.0
--inpaint_strength
0.82
--mask_feather
4.0
--comparison_width
2400
--seed
42

2. models.py

Function

Defines wrappers for the foundation models used in the project.

It includes:

GroundingDinoDetector for open-vocabulary detection
Sam2Segmenter for box-guided segmentation
Inpainter for SDXL-based image inpainting

Input

Model names
Device type, such as cuda or cpu
Image and text prompts

Output

Detected bounding boxes
Segmentation masks
Inpainted image results

3. pipeline.py

Function

Implements the full image editing pipeline.

Main steps:

Parse the editing instruction.
Detect the target object using Grounding DINO.
Segment the target object using SAM2.
Build an adaptive replacement mask.
Generate the edited region using SDXL Inpainting.
Paste the edited crop back into the original image.
Save visual results and metadata.

This file is the main logic controller of the counterfactual editing system.

4. utils.py

Function

Provides utility functions for:

Instruction parsing
Prompt construction
Negative prompt construction
Mask generation
Box expansion
Crop handling
Image blending
Result saving

5. config.py

Function

Stores default model names, thresholds, generation settings, and object-category keyword lists.

Main configurations include:

Default device
Grounding DINO model name
SAM2 model name
SDXL Inpainting model name
Detection thresholds
Inpainting steps and guidance scale
Replacement object categories
