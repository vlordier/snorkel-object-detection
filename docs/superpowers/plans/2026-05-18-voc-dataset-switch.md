# VOC Dataset Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BCCD (blood cells) with Pascal VOC 2012 (4 visually distinct classes) so DETR training converges to non-zero mAP.

**Architecture:** Rewrite `scripts/build_notebook.py` to generate a VOC-based notebook. Same pipeline structure (Selective Search → Snorkel LFs → LabelModel → pseudo-labels → DETR training → evaluation), new dataset, new dataset, new LFs, new class definitions. Output to `notebooks/snorkel_voc_object_detection.ipynb`.

**Tech Stack:** Pascal VOC 2012 (torchvision.datasets.VOCDetection), DETR (facebook/detr-resnet-50), Snorkel, OpenCV, 4 classes

**Why this works:** BCCD has ~290 training images — far too few for DETR (mAP=0.000). VOC 2012 has ~11,530 images. With 500+ objects per selected class, DETR can converge. The Snorkel LFs and LabelModel already work well (87.7%); the bottleneck was dataset size.

## Selected Classes

| ID | Class | Visual Clues | Heuristic LFs |
|----|-------|-------------|---------------|
| 0 | aeroplane | White/gray, upper image half, ~1.0-2.0 aspect ratio, large | `lf_white_upper`, `lf_bright_uniform`, `lf_large_upper` |
| 1 | boat | White hull, lower/center, ~2.5-4.5 aspect ratio, large | `lf_white_boat`, `lf_wide_lower` |
| 2 | bottle | Tall/narrow (~0.3-0.6 AR), small-medium area, glass | `lf_narrow_aspect`, `lf_small_object` |
| 3 | pottedplant | Green leaves, brown pot, lower half, medium area, roundish | `lf_green_dominant`, `lf_lower_medium` |

## File Changes

- Rewrite: `scripts/build_notebook.py` — change all dataset references, parsing, LFs, thresholds, class names
- Create: `notebooks/snorkel_voc_object_detection.ipynb` — built from the updated builder
- No changes to: `scripts/validate_notebook.py`, `requirements.txt`

## Key Design Decisions

1. **VOC downloaded via torchvision**: Use `VOCDetection(root="./data", year="2012", image_set="trainval", download=True)` which auto-downloads. Parse XML annotations from the standard VOC directory structure (`VOCdevkit/VOC2012/`).

2. **4 classes, not 20**: Selecting visually distinct classes ensures LFs have discriminative power. Avoids confusing similar classes (cat/dog, bus/car, etc.).

3. **LFs vote for one class each** (single polarity): Each LF targets exactly one class and abstains otherwise. This keeps the label matrix clean and LabelModel efficient.

4. **Data-driven thresholds**: Same approach as BCCD — compute per-class medians/means from proposals that match GT boxes, use midpoints as thresholds.

---

### Task 1: Rewrite Build Script — Dataset Section

**Files:**
- Rewrite: `scripts/build_notebook.py` (Section 1-2: setup + dataset)

Overall changes in these tasks replace BCCD-specific cells with VOC-specific cells in `build_notebook.py`. The script structure (list of cells → JSON output) stays identical.

- [ ] **Step 1: Update title, description, and class metadata**

In the markdown cell (cell 0), replace:
```python
cells.append(make_md("""# Snorkel for Object Detection: Weak Supervision on BCCD Blood Cell Dataset
...
## Dataset: BCCD (Blood Cell Detection)
- **3 classes:** WBC, RBC, Platelets
- **~364 images** with VOC-format XML annotations
...
"""))
```
with:
```python
cells.append(make_md("""# Snorkel for Object Detection: Weak Supervision on Pascal VOC 2012
...
## Dataset: Pascal VOC 2012
- **4 selected classes:** aeroplane, boat, bottle, pottedplant
- **~11,530 images** with difficult but visually distinct objects
- Each class has distinct visual properties (color, shape, position) suited to heuristic labeling
...
"""))
```

- [ ] **Step 2: Replace dataset download/parsing with VOC loading**

In cell 4 (download BCCD) and cell 5 (parse XML), replace with:

```python
# @title Download Pascal VOC 2012
import torchvision
from torchvision.datasets import VOCDetection

DATA_DIR = "voc_dataset"
VOC_ROOT = os.path.join(DATA_DIR, "VOCdevkit")

if not os.path.exists(VOC_ROOT):
    print("Downloading Pascal VOC 2012 (~2GB)...")
    voc_dataset = VOCDetection(root=DATA_DIR, year="2012", image_set="trainval", download=True)
    print("Download complete!")
else:
    print("VOC dataset already exists!")

# Set image and annotation directories
IMAGES_DIR = os.path.join(VOC_ROOT, "VOC2012", "JPEGImages")
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "VOC2012", "Annotations")
```

And the annotation parsing cell:

```python
# Parse VOC annotations (VOC format XML, same structure as BCCD)
CLASS_MAP = {"aeroplane": 0, "boat": 1, "bottle": 2, "pottedplant": 3}
CLASS_NAMES = {0: "aeroplane", 1: "boat", 2: "bottle", 3: "pottedplant"}

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text + ".jpg"
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAP:
            continue
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        boxes.append({
            "class_id": CLASS_MAP[name],
            "class_name": name,
            "bbox": [xmin, ymin, xmax, ymax],
        })
    return {"filename": filename, "width": width, "height": height, "boxes": boxes}

# Parse all annotations
all_annotations = {}
xml_files = sorted(Path(ANNOTATIONS_DIR).glob("*.xml"))
for xml_path in xml_files:
    ann = parse_voc_xml(xml_path)
    all_annotations[ann["filename"]] = ann

print(f"Total images with selected class annotations: {len(all_annotations)}")

# Class distribution
class_counts = {}
for cls_name in CLASS_MAP:
    class_counts[cls_name] = 0
for ann in all_annotations.values():
    for box in ann["boxes"]:
        class_counts[box["class_name"]] += 1

print(f"\nClass distribution:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count}")

# Image statistics
widths = [ann["width"] for ann in all_annotations.values()]
heights = [ann["height"] for ann in all_annotations.values()]
print(f"\nImage sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
```

Note: `VOCDetection` returns filenames WITHOUT `.jpg` extension (e.g., `2007_000027`). The actual files on disk have `.jpg` — we append it in `parse_voc_xml`. Verify this during implementation.

- [ ] **Step 3: Update visualization cell colors**

In cell 6 (visualize samples), change colors:
```python
colors = {"aeroplane": "cyan", "boat": "yellow", "bottle": "magenta", "pottedplant": "lime"}
```

All other dataset-related cell content (N_IMAGES, MAX_PROPOSALS, proposal generation, feature extraction) stays the same — those are dataset-agnostic.

---

### Task 2: Rewrite Threshold Computation (Cell 13)

**Files:**
- Modify: `scripts/build_notebook.py` (cell 13 — data-driven threshold computation)

Replace the BCCD-specific threshold computation with VOC-appropriate thresholds.

- [ ] **Step 1: Compute per-class stats for VOC classes**

```python
# Compute LF thresholds from data statistics (no magic numbers)
pos_samples = df[df["best_iou"] > 0.5].copy()

if len(pos_samples) > 0:
    print("Per-class statistics for threshold computation:")
    class_stats = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_df = pos_samples[pos_samples["best_class"] == cls_id]
        if len(cls_df) > 0:
            means = cls_df[["mean_r", "mean_g", "mean_b"]].mean()
            total = means.sum() + 1e-10
            class_stats[cls_name] = {
                "n": len(cls_df),
                "brightness": means.sum(),
                "green_ratio": means["mean_g"] / total,
                "area_median": cls_df["area"].median(),
                "area_p25": cls_df["area"].quantile(0.25),
                "area_p75": cls_df["area"].quantile(0.75),
                "std_median": cls_df[["std_r", "std_g", "std_b"]].mean(axis=1).median(),
                "aspect_median": cls_df["aspect_ratio"].median(),
                "aspect_p25": cls_df["aspect_ratio"].quantile(0.25),
                "aspect_p75": cls_df["aspect_ratio"].quantile(0.75),
                "center_y_median": cls_df["center_y"].median(),
            }
            print(f"  {cls_name} (n={class_stats[cls_name]['n']}):")
            print(f"    Brightness={class_stats[cls_name]['brightness']:.0f}, Green ratio={class_stats[cls_name]['green_ratio']:.3f}")
            print(f"    Area: median={class_stats[cls_name]['area_median']:.0f}, p25={class_stats[cls_name]['area_p25']:.0f}, p75={class_stats[cls_name]['area_p75']:.0f}")
            print(f"    Aspect: median={class_stats[cls_name]['aspect_median']:.2f}, p25={class_stats[cls_name]['aspect_p25']:.2f}, p75={class_stats[cls_name]['aspect_p75']:.2f}")
            print(f"    Center Y: median={class_stats[cls_name]['center_y_median']:.3f}")

    # --- Compute data-driven thresholds ---
    aero = class_stats.get("aeroplane")
    boat = class_stats.get("boat")
    bottle = class_stats.get("bottle")
    plant = class_stats.get("pottedplant")

    # Aspect ratio: bottle is narrow, boat is wide
    if bottle and aero:
        NARROW_AR = (bottle["aspect_p75"] + aero["aspect_p25"]) / 2  # midpoint
    else:
        NARROW_AR = 0.6
    if boat and aero:
        WIDE_AR = (boat["aspect_p25"] + aero["aspect_p75"]) / 2  # midpoint
    else:
        WIDE_AR = 2.0

    # Area: bottle small, large for aero/boat
    if bottle and aero:
        SMALL_AREA = int((bottle["area_p75"] + aero["area_p25"]) / 2)
    else:
        SMALL_AREA = 3000

    # Green ratio for pottedplant
    if plant and aero:
        GREEN_RATIO_THRESH = (plant["green_ratio"] + aero["green_ratio"]) / 2
    else:
        GREEN_RATIO_THRESH = 0.36

    # Brightness for white objects
    if aero:
        BRIGHT_THRESH = aero["brightness"] * 0.85
    elif boat:
        BRIGHT_THRESH = boat["brightness"] * 0.85
    else:
        BRIGHT_THRESH = 550

    # Position: upper for aeroplane, lower for boat/pottedplant
    if aero:
        UPPER_Y_THRESH = aero["center_y_median"] * 1.3
    else:
        UPPER_Y_THRESH = 0.4
    if boat and plant:
        LOWER_Y_THRESH = (boat["center_y_median"] + plant["center_y_median"]) / 2 * 0.85
    else:
        LOWER_Y_THRESH = 0.55

    # Color std: white objects are uniform
    UNIFORM_STD = 35  # fixed threshold for white objects

    print(f"\nComputed thresholds:")
    print(f"  Narrow AR < {NARROW_AR:.2f} -> bottle")
    print(f"  Wide AR > {WIDE_AR:.2f} -> boat")
    print(f"  Small area < {SMALL_AREA} -> bottle")
    print(f"  Green ratio > {GREEN_RATIO_THRESH:.3f} -> pottedplant")
    print(f"  Brightness > {BRIGHT_THRESH:.0f} + upper (y<{UPPER_Y_THRESH:.2f}) -> aeroplane")
    print(f"  Brightness > {BRIGHT_THRESH:.0f} + lower (y>{LOWER_Y_THRESH:.2f}) + wide -> boat")
    print(f"  Color std < {UNIFORM_STD} + bright -> white object (aero/boat)")

    # Per-class RGB distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (channel, color) in enumerate([("mean_r", "red"), ("mean_g", "green"), ("mean_b", "blue")]):
        for cls_id, cls_name in CLASS_NAMES.items():
            cls_df = pos_samples[pos_samples["best_class"] == cls_id]
            if len(cls_df) > 0:
                axes[i].hist(cls_df[channel], bins=20, alpha=0.5, label=cls_name)
        axes[i].set_title(f"{color.title()} channel")
        axes[i].set_xlabel("Mean intensity")
        axes[i].legend()
    plt.suptitle("Per-class RGB distributions (thresholds computed from these)")
    plt.tight_layout()
    plt.show()
else:
    # Fallback defaults
    NARROW_AR = 0.6
    WIDE_AR = 2.0
    SMALL_AREA = 3000
    GREEN_RATIO_THRESH = 0.36
    BRIGHT_THRESH = 550
    UPPER_Y_THRESH = 0.4
    LOWER_Y_THRESH = 0.55
    UNIFORM_STD = 35
    print("No GT proposals available. Using default thresholds.")
```

---

### Task 3: Rewrite LFs (Section 5)

**Files:**
- Modify: `scripts/build_notebook.py` (cells 15-16 — LF definitions)

Replace BCCD LFs with 6 VOC-appropriate LFs using the data-driven thresholds.

- [ ] **Step 1: Update LF markdown table**

In the Section 5 markdown cell, replace the LF table:

```python
cells.append(make_md("""## 5. Snorkel Labeling Functions

We write 6 labeling functions that classify each proposal based on visual heuristics.

**4 classes, 6 LFs:**

| LF | Heuristic | Logic |
|---|---|---|
| `lf_aspect_ratio` | Shape | Narrow AR < threshold → bottle, Wide AR > threshold → boat |
| `lf_green_dominant` | Color | G/(R+G+B) > threshold → pottedplant |
| `lf_bright_upper` | Color + Position | High brightness + upper half → aeroplane |
| `lf_white_boat` | Color + Position + Shape | High brightness + lower half + wide → boat |
| `lf_bright_uniform` | Color + Texture | High brightness + low color std → white object (aero/boat) |
| `lf_small_object` | Size | Small area + narrow-ish AR → bottle |
"""))
```

- [ ] **Step 2: Replace LF definitions and label constants**

```python
# Define label constants
AEROPLANE = 0
BOAT = 1
BOTTLE = 2
POTTEDPLANT = 3
ABSTAIN = -1

# Clean feature names for Snorkel (no spaces/dots)
df_lf = df.copy()
df_lf.columns = [c.replace(".", "_").replace(" ", "_") for c in df_lf.columns]

@labeling_function()
def lf_aspect_ratio(row):
    if row.aspect_ratio < NARROW_AR:
        return BOTTLE
    elif row.aspect_ratio > WIDE_AR:
        return BOAT
    return ABSTAIN

@labeling_function()
def lf_green_dominant(row):
    total = row.mean_r + row.mean_g + row.mean_b + 1e-10
    if total > 0 and (row.mean_g / total) > GREEN_RATIO_THRESH:
        return POTTEDPLANT
    return ABSTAIN

@labeling_function()
def lf_bright_upper(row):
    brightness = row.mean_r + row.mean_g + row.mean_b
    if brightness > BRIGHT_THRESH and row.center_y < UPPER_Y_THRESH:
        return AEROPLANE
    return ABSTAIN

@labeling_function()
def lf_white_boat(row):
    brightness = row.mean_r + row.mean_g + row.mean_b
    if brightness > BRIGHT_THRESH and row.center_y > LOWER_Y_THRESH and row.aspect_ratio > WIDE_AR:
        return BOAT
    return ABSTAIN

@labeling_function()
def lf_bright_uniform(row):
    brightness = row.mean_r + row.mean_g + row.mean_b
    mean_std = (row.std_r + row.std_g + row.std_b) / 3
    if brightness > BRIGHT_THRESH and mean_std < UNIFORM_STD:
        return AEROPLANE
    return ABSTAIN

@labeling_function()
def lf_small_object(row):
    if row.area < SMALL_AREA and row.aspect_ratio < 0.9:
        return BOTTLE
    return ABSTAIN

lfs = [lf_aspect_ratio, lf_green_dominant, lf_bright_upper, lf_white_boat, lf_bright_uniform, lf_small_object]
```

- [ ] **Step 3: Update LF visualization cell**

In the LF visualization cell, update the call to `plot_lf_decisions` to point to `lf_green_dominant` (index 1), and update the title message:

```python
print("Showing decisions for lf_green_dominant (index 1):")
plot_lf_decisions(
    os.path.join(IMAGES_DIR, all_filenames[0]),
    df_lf[df_lf["filename"] == all_filenames[0]],
    lf_idx=1
)
```

---

### Task 4: Update LabelModel Section (Section 6)

**Files:**
- Modify: `scripts/build_notebook.py` (cells 18-20 — LabelModel training)

- [ ] **Step 1: Update cardinality to 4**

```python
label_model = LabelModel(cardinality=4, verbose=True)
```

And the confusion matrix display labels:
```python
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Aeroplane", "Boat", "Bottle", "Pottedplant"])
```

---

### Task 5: Update COCO Format and DETR Sections (Sections 7-9)

**Files:**
- Modify: `scripts/build_notebook.py` (cells 22-31 — pseudo-label assignment, COCO conversion, DETR training)

- [ ] **Step 1: Update DETR model configuration**

```python
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=4,  # was 3 for BCCD
    ignore_mismatched_sizes=True,
)
```

- [ ] **Step 2: Update COCO categories in COCO conversion**

In `build_coco_json`:
```python
"categories": [
    {"id": 0, "name": "aeroplane"},
    {"id": 1, "name": "boat"},
    {"id": 2, "name": "bottle"},
    {"id": 3, "name": "pottedplant"},
],
```

Same for `build_gt_coco`.

---

### Task 6: Update Evaluation Section (Section 10)

**Files:**
- Modify: `scripts/build_notebook.py` (cells 33-37 — evaluation and summary)

- [ ] **Step 1: Update per-class colors in visualization**

```python
colors = {0: "cyan", 1: "yellow", 2: "magenta", 3: "lime"}
```

- [ ] **Step 2: Update PR curve colors**

```python
colors_pr = ["cyan", "yellow", "magenta", "lime"]
```

- [ ] **Step 3: Update final summary cell to reference VOC**

Replace the summary text to say "VOC 2012" instead of "BCCD", 4 classes instead of 3.

- [ ] **Step 4: Update conclusion markdown cell**

Replace BCCD-specific next steps with VOC-appropriate ones.

---

### Task 7: Update Validate Script

**Files:**
- Modify: `scripts/validate_notebook.py`

- [ ] **Step 1: Update path and LF names**

Replace the hardcoded path and LF name list:

```python
NOTEBOOK_PATH = "notebooks/snorkel_voc_object_detection.ipynb"
```

And the key functions list:
```python
key_fns = [
    "lf_aspect_ratio", "lf_green_dominant", "lf_bright_upper",
    "lf_white_boat", "lf_bright_uniform", "lf_small_object",
    "LabelModel", "generate_proposals", "extract_features",
    "nms", "build_coco_json",
]
```

- [ ] **Step 2: Verify validate script**

Run: `python scripts/validate_notebook.py`
Expected: `VALIDATION FAILED` (expected — notebook hasn't been rebuilt yet)

---

### Task 8: Rebuild and Validate

**Files:**
- Create: `notebooks/snorkel_voc_object_detection.ipynb`

- [ ] **Step 1: Rebuild notebook**

Run: `python scripts/build_notebook.py`
Expected output: `Notebook written to notebooks/snorkel_voc_object_detection.ipynb`

Verify the output notebook has 4 classes, 6 VOC-appropriate LFs, and all sections.

- [ ] **Step 2: Validate notebook structure**

Run: `python scripts/validate_notebook.py`
Expected: `VALIDATION PASSED`

- [ ] **Step 3: Execute notebook (spot-check)**

Run the notebook headlessly to verify it executes:
```bash
jupyter nbconvert --to notebook --execute notebooks/snorkel_voc_object_detection.ipynb --output /tmp/voc_test.ipynb
```
Expected: All cells execute without errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_notebook.py scripts/validate_notebook.py notebooks/snorkel_voc_object_detection.ipynb docs/superpowers/plans/2026-05-18-voc-dataset-switch.md
git commit -m "feat: switch from BCCD to Pascal VOC 2012, 4-class LFs, 11K images for DETR convergence"
```
Expected: All 39 cells execute without errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_notebook.py notebooks/snorkel_voc_object_detection.ipynb docs/superpowers/plans/2026-05-18-voc-dataset-switch.md
git commit -m "feat: switch from BCCD to Pascal VOC 2012, 4-class LFs, 11K images for DETR convergence"
```