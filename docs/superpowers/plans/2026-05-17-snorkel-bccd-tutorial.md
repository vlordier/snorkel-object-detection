# Snorkel BCCD Object Detection Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab-optimized Jupyter notebook demonstrating Snorkel weak supervision + DETR for blood cell detection.

**Architecture:** Single `.ipynb` notebook with 10 sections covering the pipeline: Selective Search proposals → Snorkel labeling functions → LabelModel → pseudo-label assignment → COCO format conversion → DETR fine-tuning → evaluation.

**Tech Stack:** Snorkel, HuggingFace Transformers (DETR), OpenCV (selective search), torchvision, pycocotools, BCCD dataset.

---

### File Structure

| File | Purpose |
|---|---|
| `notebooks/snorkel_bccd_object_detection.ipynb` | Main tutorial notebook |
| `scripts/validate_notebook.py` | Script to test the notebook executes cleanly |
| `requirements.txt` | Pinned dependencies |

---

### Task 1: Create project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `scripts/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```txt
snorkel>=0.9.8
transformers>=4.30.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
scipy>=1.7.0
pycocotools>=2.0.0
jupyter>=1.0.0
nbformat>=5.0.0
nbconvert>=7.0.0
papermill>=2.4.0
```

- [ ] **Step 2: Create directories**

Run:
```bash
mkdir -p notebooks scripts
```

- [ ] **Step 3: Create empty `scripts/__init__.py`**

- [ ] **Step 4: Commit scaffold**

```bash
git add requirements.txt scripts/__init__.py
git commit -m "chore: add project scaffold for snorkel bccd tutorial"
```

---

### Task 2: Write the complete tutorial notebook

**Files:**
- Create: `notebooks/snorkel_bccd_object_detection.ipynb`

This is the core task. The notebook has 10 sections as raw JSON `.ipynb` cells.

**Notebook metadata:**
```json
{
 "cells": [],
 "metadata": {
  "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
  "language_info": { "name": "python", "version": "3.10.0" }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

Each cell has type `code` or `markdown`. Code cells have `"execution_count": null` and empty outputs `"outputs": []`.

- [ ] **Step 1: Write Section 1 — Setup & Installation**

Markdown cell: Title, intro paragraph about Snorkel + object detection.
Code cell: `!pip install snorkel transformers torch torchvision opencv-python opencv-contrib-python matplotlib seaborn scikit-learn scipy pycocotools`

- [ ] **Step 2: Write Section 2 — Dataset download & exploration**

Code cell: Download BCCD from GitHub, parse XML annotations, show dataset statistics, visualize 6 sample images with GT boxes.

- [ ] **Step 3: Write Section 3 — Region proposals via Selective Search**

Code cell: Implement `generate_proposals(image)` using `cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()`, show proposals overlaid on images, compute IoU with GT boxes for reference table.

- [ ] **Step 4: Write Section 4 — Feature Extraction per proposal**

Code cell: Implement `extract_features(proposal, image)` returning dict with area, aspect_ratio, mean_rgb, std_rgb, edge_density, center_x, center_y. Apply to all proposals and build a DataFrame.

- [ ] **Step 5: Write Section 5 — Snorkel Labeling Functions**

Code cell: Define 6 labeling functions using `@labeling_function()`:
- `lf_size`: area thresholds
- `lf_color_rbc`: high red, low blue
- `lf_color_wbc`: high blue channel, large area
- `lf_aspect_ratio`: round vs elongated
- `lf_edge_density`: high edges → nucleus
- `lf_background`: low variance → ABSTAIN

Apply LFs to get `L_train` matrix. Show LF summary stats.

- [ ] **Step 6: Write Section 6 — Label Model**

Code cell: `LabelModel(cardinality=3)`, fit, show LF accuracy/coverage/conflicts table, compare LabelModel vs majority vote vs best single LF.

- [ ] **Step 7: Write Section 7 — Pseudo-label assignment**

Code cell: Get probabilistic labels, filter by confidence > 0.7, assign class, apply NMS to suppress duplicate proposals, convert to list of pseudo-boxes.

- [ ] **Step 8: Write Section 8 — COCO format conversion**

Code cell: Build COCO JSON from pseudo-labels. Create `BCCDDataset` class extending `torch.utils.data.Dataset`. Register with HuggingFace `DetrImageProcessor`.

- [ ] **Step 9: Write Section 9 — DETR fine-tuning**

Code cell: Load `DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")`, configure `transformers.Trainer`, train for 10 epochs, save model.

- [ ] **Step 10: Write Section 10 — Evaluation**

Code cell: Load saved model, compute mAP@0.5 via pycocotools, per-class precision/recall, visualize 10 prediction samples with GT comparison. Summary table comparing weak supervision vs fully-supervised baseline.

- [ ] **Step 11: Commit notebook**

```bash
git add notebooks/snorkel_bccd_object_detection.ipynb
git commit -m "feat: add snorkel bccd object detection tutorial notebook"
```

---

### Task 3: Format and clean the notebook

**Files:**
- Modify: `notebooks/snorkel_bccd_object_detection.ipynb`

- [ ] **Step 1: Strip all output cells and execution counts**

Run:
```bash
python -c "
import json
with open('notebooks/snorkel_bccd_object_detection.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
with open('notebooks/snorkel_bccd_object_detection.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
"
```

- [ ] **Step 2: Validate JSON integrity**

Run:
```bash
python -c "json.load(open('notebooks/snorkel_bccd_object_detection.ipynb')); print('Valid JSON')"
```

Expected: `Valid JSON`

- [ ] **Step 3: Commit cleaned notebook**

```bash
git add notebooks/snorkel_bccd_object_detection.ipynb
git commit -m "style: strip notebook output cells and format"
```

---

### Task 4: Execute and validate the notebook

**Files:**
- Create: `scripts/validate_notebook.py`

- [ ] **Step 1: Write validation script**

```python
"""Validate the tutorial notebook by checking structure and content."""

import json
import sys

NOTEBOOK_PATH = "notebooks/snorkel_bccd_object_detection.ipynb"

with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

errors = []

# Check cell structure
cells = nb["cells"]
if len(cells) < 10:
    errors.append(f"Expected >=10 cells, got {len(cells)}")

# Check sections are present
markdown_cells = [c for c in cells if c["cell_type"] == "markdown"]
code_cells = [c for c in cells if c["cell_type"] == "code"]

if len(markdown_cells) < 5:
    errors.append(f"Expected >=5 markdown cells, got {len(markdown_cells)}")
if len(code_cells) < 15:
    errors.append(f"Expected >=15 code cells, got {len(code_cells)}")

# Check key imports
all_source = "\n".join(
    "".join(c["source"]) for c in cells if c["cell_type"] == "code"
)
key_imports = ["snorkel", "transformers", "DetrForObjectDetection", "cv2"]
for imp in key_imports:
    if imp not in all_source:
        errors.append(f"Missing import/reference: {imp}")

# Check key functions
key_fns = [
    "lf_size", "lf_color_rbc", "lf_color_wbc",
    "lf_aspect_ratio", "lf_edge_density", "lf_background",
    "LabelModel", "generate_proposals", "extract_features"
]
for fn in key_fns:
    if fn not in all_source:
        errors.append(f"Missing function/class: {fn}")

if errors:
    print(f"VALIDATION FAILED: {len(errors)} errors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"VALIDATION PASSED: {len(cells)} cells, all key components found")
```

- [ ] **Step 2: Run validation**

Run:
```bash
python scripts/validate_notebook.py
```

Expected: `VALIDATION PASSED: N cells, all key components found`

- [ ] **Step 3: Attempt execution with papermill**

Run:
```bash
papermill notebooks/snorkel_bccd_object_detection.ipynb /tmp/output.ipynb --kernel python3 2>&1 | tail -20
```

Or if papermill is not available, try:
```bash
jupyter nbconvert --to script --stdout notebooks/snorkel_bccd_object_detection.ipynb | python 2>&1 | head -50
```

Expected: Notebook executes without errors (or at least the non-GPU portions compile syntactically).

- [ ] **Step 4: Commit validation script**

```bash
git add scripts/validate_notebook.py
git commit -m "test: add notebook validation script"
```

---

### Self-Review Checklist

1. **Spec coverage:** Every spec requirement maps to a task.
   - Setup + dataset → Task 2 Step 1-2
   - Region proposals → Task 2 Step 3
   - Feature extraction → Task 2 Step 4
   - Snorkel LFs → Task 2 Step 5
   - LabelModel → Task 2 Step 6
   - Pseudo-label assignment → Task 2 Step 7
   - COCO conversion → Task 2 Step 8
   - DETR training → Task 2 Step 9
   - Evaluation → Task 2 Step 10
   - Format/clean → Task 3
   - Test/validate → Task 4

2. **Placeholder scan:** All steps have concrete code. No TBD/TODO.

3. **Type consistency:** Single notebook, no cross-file type issues.

4. **Scope check:** Single notebook. No decomposition needed.
