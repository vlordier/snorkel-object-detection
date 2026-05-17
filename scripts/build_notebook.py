"""Build the snorkel BCCD object detection tutorial notebook."""

import json

def make_md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    }

def make_code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }

cells = []

# ============================================================
# SECTION 1: Setup & Installation
# ============================================================
cells.append(make_md("""# Snorkel for Object Detection: Weak Supervision on BCCD Blood Cell Dataset

This tutorial demonstrates how to use **Snorkel** (a programmatic data labeling framework) to bootstrap object detection with weak supervision.

## Pipeline Overview

```
BCCD images → Selective Search proposals → Snorkel LFs → LabelModel
→ pseudo-boxes → COCO format → fine-tune DETR → evaluate mAP
```

## Dataset: BCCD (Blood Cell Detection)

- **3 classes:** WBC (White Blood Cells), RBC (Red Blood Cells), Platelets
- **~364 images** with VOC-format XML annotations
- Cells have distinct visual characteristics that make heuristic labeling possible"""))

cells.append(make_code("""# @title Install dependencies
!pip install -q snorkel transformers torch torchvision opencv-contrib-python matplotlib seaborn scikit-learn scipy pycocotools
!pip install -q gitpython"""))

cells.append(make_code("""import os
import json
import xml.etree.ElementTree as ET
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers import Trainer, TrainingArguments

import snorkel
from snorkel.labeling import labeling_function, LabelModel, MajorityLabelVoter
from snorkel.labeling import PandasLFApplier
from snorkel.analysis import LFAnalysis

print("All imports successful!")
print(f"Snorkel version: {snorkel.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")"""))

# ============================================================
# SECTION 2: Dataset Download & Exploration
# ============================================================
cells.append(make_md("""## 2. Dataset Download & Exploration

We download the BCCD dataset and parse the VOC-format XML annotations."""))

cells.append(make_code("""# @title Download BCCD Dataset
DATA_DIR = "bccd_dataset"

if not os.path.exists(DATA_DIR):
    !git clone https://github.com/Shenggan/BCCD_Dataset.git {DATA_DIR}
    print("Dataset downloaded!")
else:
    print("Dataset already exists!")

# Check structure
!ls -la {DATA_DIR}/BCCD/"""))

cells.append(make_code("""# Parse annotations
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "BCCD", "Annotations")
IMAGES_DIR = os.path.join(DATA_DIR, "BCCD", "Images")

CLASS_MAP = {"WBC": 0, "RBC": 1, "Platelets": 2}
CLASS_NAMES = {0: "WBC", 1: "RBC", 2: "Platelets"}

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
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

print(f"Total images with annotations: {len(all_annotations)}")

# Class distribution
class_counts = {"WBC": 0, "RBC": 0, "Platelets": 0}
for ann in all_annotations.values():
    for box in ann["boxes"]:
        class_counts[box["class_name"]] += 1

print(f"\\nClass distribution:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count}")

# Image statistics
widths = [ann["width"] for ann in all_annotations.values()]
heights = [ann["height"] for ann in all_annotations.values()]
print(f"\\nImage sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")"""))

cells.append(make_code("""# Visualize sample images with ground truth boxes
def visualize_image(filename, ax):
    img_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ann = all_annotations[filename]
    colors = {"WBC": "cyan", "RBC": "yellow", "Platelets": "magenta"}
    for box in ann["boxes"]:
        xmin, ymin, xmax, ymax = box["bbox"]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=colors[box["class_name"]],
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, box["class_name"],
                color=colors[box["class_name"]], fontsize=10, weight="bold")

sample_files = random.sample(list(all_annotations.keys()), 6)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, fname in zip(axes.flat, sample_files):
    visualize_image(fname, ax)
    ax.axis("off")
plt.suptitle("Sample images with GT boxes (Cyan=WBC, Yellow=RBC, Magenta=Platelets)", fontsize=14)
plt.tight_layout()
plt.show()"""))

# ============================================================
# SECTION 3: Region Proposals via Selective Search
# ============================================================
cells.append(make_md("""## 3. Region Proposals via Selective Search

We use OpenCV's Selective Search to generate candidate bounding boxes. These proposals serve as the input for our Snorkel labeling functions."""))

cells.append(make_code("""# @title Selective Search Proposal Generation
def generate_proposals(img, max_proposals=500):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    rects = rects[:max_proposals]
    proposals = []
    for x, y, w, h in rects:
        proposals.append([x, y, x + w, y + h])
    return proposals

def compute_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter / float(area_a + area_b - inter + 1e-10)
    return iou"""))

cells.append(make_code("""# Visualize proposals on a sample image
sample_fname = sample_files[0]
img_path = os.path.join(IMAGES_DIR, sample_fname)
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
proposals = generate_proposals(img, max_proposals=200)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes[0].imshow(img_rgb)
ann = all_annotations[sample_fname]
for box in ann["boxes"]:
    xmin, ymin, xmax, ymax = box["bbox"]
    rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor="lime", facecolor="none"
    )
    axes[0].add_patch(rect)
axes[0].set_title(f"GT Boxes ({len(ann['boxes'])} objects)", fontsize=12)
axes[0].axis("off")

# Plot some proposals
axes[1].imshow(img_rgb)
for i, prop in enumerate(proposals[:50]):
    xmin, ymin, xmax, ymax = prop
    rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=0.5, edgecolor="orange", facecolor="none"
    )
    axes[1].add_patch(rect)
axes[1].set_title(f"Selective Search Proposals (showing 50 of {len(proposals)})", fontsize=12)
axes[1].axis("off")
plt.tight_layout()
plt.show()

# IoU analysis
best_ious = []
for gt_box in ann["boxes"]:
    best_iou = max(compute_iou(gt_box["bbox"], p) for p in proposals)
    best_ious.append(best_iou)
print(f"Best IoU per GT box: {[f'{iou:.2f}' for iou in best_ious]}")
print(f"Mean best IoU: {np.mean(best_ious):.2f}")"""))

# ============================================================
# SECTION 4: Feature Extraction
# ============================================================
cells.append(make_md("""## 4. Feature Extraction per Proposal

For each proposal region, we extract visual features that our labeling functions will use to make decisions."""))

cells.append(make_code("""# @title Feature Extraction
def extract_features(img, proposal):
    xmin, ymin, xmax, ymax = proposal
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img.shape[1], xmax), min(img.shape[0], ymax)
    if xmax <= xmin or ymax <= ymin:
        return {"area": 0, "aspect_ratio": 0, "mean_r": 0, "mean_g": 0, "mean_b": 0,
                "std_r": 0, "std_g": 0, "std_b": 0, "edge_density": 0, "center_x": 0, "center_y": 0}
    crop = img[ymin:ymax, xmin:xmax]
    area = crop.shape[0] * crop.shape[1]
    aspect_ratio = crop.shape[1] / max(crop.shape[0], 1)
    mean_rgb = crop.mean(axis=(0, 1))
    std_rgb = crop.std(axis=(0, 1))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / max(area, 1)
    img_h, img_w = img.shape[:2]
    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "mean_r": mean_rgb[2],
        "mean_g": mean_rgb[1],
        "mean_b": mean_rgb[0],
        "std_r": std_rgb[2],
        "std_g": std_rgb[1],
        "std_b": std_rgb[0],
        "edge_density": edge_density,
        "center_x": (xmin + xmax) / 2 / img_w,
        "center_y": (ymin + ymax) / 2 / img_h,
    }"""))

cells.append(make_code("""# Build proposal dataset for a subset of images
N_IMAGES = 30  # Use subset for tutorial speed
MAX_PROPOSALS = 300

all_rows = []
all_filenames = list(all_annotations.keys())[:N_IMAGES]

for fname in tqdm(all_filenames, desc="Extracting features"):
    img_path = os.path.join(IMAGES_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    proposals = generate_proposals(img, MAX_PROPOSALS)
    for prop in proposals:
        features = extract_features(img, prop)
        # Best GT box for this proposal
        best_iou = 0
        best_class = -1
        ann = all_annotations[fname]
        for gt_box in ann["boxes"]:
            iou = compute_iou(gt_box["bbox"], prop)
            if iou > best_iou:
                best_iou = iou
                best_class = gt_box["class_id"]
        row = {
            "filename": fname,
            "proposal": prop,
            **features,
            "best_iou": best_iou,
            "best_class": best_class if best_iou > 0.5 else -1,
        }
        all_rows.append(row)

df = pd.DataFrame(all_rows)
print(f"Total proposals: {len(df)}")
print(f"Proposals matching a GT box (IoU > 0.5): {(df['best_iou'] > 0.5).sum()}")
print(f"Positive proposals by class:")
pos_df = df[df["best_iou"] > 0.5]
if len(pos_df) > 0:
    print(pos_df["best_class"].map(CLASS_NAMES).value_counts())

# Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
features_to_plot = ["area", "aspect_ratio", "mean_r", "mean_g", "mean_b", "edge_density"]
for ax, feat in zip(axes.flat, features_to_plot):
    ax.hist(df[feat].values, bins=50, alpha=0.7)
    ax.set_title(feat)
    ax.set_yscale("log")
plt.tight_layout()
plt.show()"""))

# ============================================================
# SECTION 5: Snorkel Labeling Functions
# ============================================================
cells.append(make_md("""## 5. Snorkel Labeling Functions

We write 6 labeling functions that classify each proposal based on visual heuristics.

**Labeling functions:**

| LF | Heuristic | Logic |
|---|---|---|
| `lf_size` | Area thresholds | Large (>5000)→WBC, Medium(800-5000)→RBC, Small(<800)→Platelet |
| `lf_color_rbc` | Reddish color | High R, low B → RBC |
| `lf_color_wbc` | Purple/blue nucleus | High B channel, large area → WBC |
| `lf_aspect_ratio` | Shape | Round(AR~1)→RBC, Elongated(AR>1.8)→WBC |
| `lf_edge_density` | Nucleus texture | High edge density → WBC |
| `lf_background` | Low signal | Low variance + low edges → ABSTAIN |

**ABSTAIN** means the LF has no opinion — Snorkel's LabelModel handles this."""))

cells.append(make_code("""# Define label constants
WBC = 0
RBC = 1
PLATELET = 2
ABSTAIN = -1

# Clean feature names for Snorkel (no spaces/dots)
df_lf = df.copy()
df_lf.columns = [c.replace(".", "_").replace(" ", "_") for c in df_lf.columns]

@labeling_function()
def lf_size(row):
    if row.area > 3000:
        return WBC
    elif row.area > 800:
        return RBC
    elif row.area > 50:
        return PLATELET
    else:
        return ABSTAIN

@labeling_function()
def lf_color_rbc(row):
    if row.mean_r > 100 and row.mean_b < 80:
        return RBC
    return ABSTAIN

@labeling_function()
def lf_color_wbc(row):
    if row.mean_b > row.mean_r * 0.8 and row.area > 2000:
        return WBC
    return ABSTAIN

@labeling_function()
def lf_aspect_ratio(row):
    if row.aspect_ratio > 0.7 and row.aspect_ratio < 1.4:
        return RBC
    elif row.aspect_ratio > 1.5:
        return WBC
    return ABSTAIN

@labeling_function()
def lf_edge_density(row):
    if row.edge_density > 30 and row.area > 1500:
        return WBC
    return ABSTAIN

@labeling_function()
def lf_position(row):
    if row.center_x < 0.05 or row.center_x > 0.95 or row.center_y < 0.05 or row.center_y > 0.95:
        return ABSTAIN
    if 0.25 < row.center_x < 0.75 and 0.25 < row.center_y < 0.75:
        if row.area > 3000:
            return WBC
        elif row.area > 800:
            return RBC
    return ABSTAIN

lfs = [lf_size, lf_color_rbc, lf_color_wbc, lf_aspect_ratio, lf_edge_density, lf_position]

# Apply LFs
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_lf)
print(f"Label matrix shape: {L_train.shape}")
print(f"Coverage per LF:")
lf_analysis = LFAnalysis(L_train)
for i, lf in enumerate(lfs):
    coverage = (L_train[:, i] != ABSTAIN).mean()
    print(f"  {lf.name}: {coverage:.1%} coverage")"""))

cells.append(make_code("""# Visualize LF outputs on proposals
def plot_lf_decisions(image_path, proposals_df, lf_idx, n_proposals=10):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    sample_rows = proposals_df.sample(min(n_proposals, len(proposals_df)))
    for ax, (_, row) in zip(axes.flat, sample_rows.iterrows()):
        crop = img[row.proposal[1]:row.proposal[3], row.proposal[0]:row.proposal[2]]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ax.imshow(crop_rgb)
        row_idx = df_lf.index.get_loc(row.name)
        lf_label = L_train[row_idx, lf_idx]
        label_name = CLASS_NAMES.get(lf_label, "ABSTAIN") if lf_label != -1 else "ABSTAIN"
        gt_name = CLASS_NAMES.get(row.best_class, "BG") if row.best_class != -1 else "BG"
        ax.set_title(f"LF: {label_name} | GT: {gt_name}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

print("Showing decisions for lf_color_rbc (index 1):")
# The lf_color_rbc is the 2nd LF (index 1)
plot_lf_decisions(
    os.path.join(IMAGES_DIR, all_filenames[0]),
    df_lf[df_lf["filename"] == all_filenames[0]],
    lf_idx=1
)"""))

# ============================================================
# SECTION 6: Label Model
# ============================================================
cells.append(make_md("""## 6. Label Model

We train Snorkel's `LabelModel` to combine the labeling functions into a single probabilistic label per proposal."""))

cells.append(make_code("""# @title Train LabelModel
label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train, n_epochs=500, lr=0.01, l2=0.01, seed=123)

# Get probabilistic labels
probs = label_model.predict_proba(L_train)
preds = label_model.predict(L_train)

df_lf["snorkel_label"] = preds
df_lf["snorkel_confidence"] = probs.max(axis=1)
df_lf["snorkel_label_name"] = df_lf["snorkel_label"].map(CLASS_NAMES)

print("\\nLabel distribution from LabelModel:")
print(df_lf["snorkel_label_name"].value_counts())

# Compare LabelModel vs majority vote vs individual LFs
majority_model = MajorityLabelVoter()
majority_preds = majority_model.predict(L_train)

# Get accuracy where we have ground truth
gt_mask = df_lf["best_iou"] > 0.5
if gt_mask.sum() > 0:
    gt_labels = df_lf.loc[gt_mask, "best_class"].values
    lm_acc = (preds[gt_mask] == gt_labels).mean()
    mv_acc = (majority_preds[gt_mask] == gt_labels).mean()
    print(f"\\nAccuracy (on proposals matching GT):")
    print(f"  LabelModel: {lm_acc:.1%}")
    print(f"  MajorityVote: {mv_acc:.1%}")

    # Individual LF accuracy
    print(f"  Individual LF accuracy:")
    for i, lf in enumerate(lfs):
        lf_mask = L_train[gt_mask, i] != ABSTAIN
        if lf_mask.sum() > 0:
            lf_acc = (L_train[gt_mask, i][lf_mask] == gt_labels[lf_mask]).mean()
            lf_cov = lf_mask.mean()
            print(f"    {lf.name}: acc={lf_acc:.1%}, cov={lf_cov:.1%}")"""))

cells.append(make_code("""# LFAnalysis details
lf_summary = lf_analysis.lf_summary(L_train)
print("\\nLabeling Function Summary:")
print(lf_summary)

# Plot LF statistics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lf_summary["Coverage"].plot(kind="bar", ax=axes[0])
axes[0].set_title("LF Coverage")
axes[0].set_ylabel("Coverage")
axes[0].tick_params(axis="x", rotation=45)

lf_summary["Conflicts"].plot(kind="bar", ax=axes[1])
axes[1].set_title("LF Conflicts")
axes[1].set_ylabel("Conflicts")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# Polarity (which classes each LF outputs)
print("\\nLF Polarities (output classes):")
lf_polarities = lf_analysis.lf_polarities()
for i, lf in enumerate(lfs):
    print(f"  {lf.name}: {lf_polarities[i]}")"""))

cells.append(make_code("""# Confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_lf["snorkel_confidence"], bins=50)
axes[0].set_title("LabelModel Confidence Distribution")
axes[0].set_xlabel("Confidence")
axes[0].set_ylabel("Count")

# Per-class confidence
for cls_id, cls_name in CLASS_NAMES.items():
    cls_conf = df_lf[df_lf["snorkel_label"] == cls_id]["snorkel_confidence"]
    axes[1].hist(cls_conf, bins=30, alpha=0.5, label=cls_name)
axes[1].set_title("Confidence by Class")
axes[1].set_xlabel("Confidence")
axes[1].set_ylabel("Count")
axes[1].legend()
plt.tight_layout()
plt.show()"""))

# ============================================================
# SECTION 7: Pseudo-Label Assignment
# ============================================================
cells.append(make_md("""## 7. Pseudo-Label Assignment

We assign weak labels to proposals and apply Non-Maximum Suppression (NMS) to remove redundant boxes."""))

cells.append(make_code("""# @title Assign Pseudo-Labels
CONFIDENCE_THRESHOLD = 0.7

# Filter high-confidence positive proposals
pos_mask = df_lf["snorkel_confidence"] >= CONFIDENCE_THRESHOLD
pos_df = df_lf[pos_mask].copy()

print(f"High-confidence proposals: {len(pos_df)} out of {len(df_lf)}")
print(f"\\nClass distribution:")
print(pos_df["snorkel_label_name"].value_counts())"""))

cells.append(make_code("""# Non-Maximum Suppression
def nms(proposals, scores, iou_threshold=0.5):
    if len(proposals) == 0:
        return []
    boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in proposals]
    scores_arr = np.array(scores)
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh, scores_arr.tolist(), score_threshold=0.0, nms_threshold=iou_threshold
    )
    if indices is None or len(indices) == 0:
        return []
    if hasattr(indices, "flatten"):
        return indices.flatten().tolist()
    if len(indices.shape) > 1:
        return indices[:, 0].tolist()
    return indices.tolist()

def apply_nms_per_image(df, score_col="snorkel_confidence", iou_threshold=0.5):
    kept_rows = []
    for fname in df["filename"].unique():
        img_df = df[df["filename"] == fname]
        proposals_list = img_df["proposal"].tolist()
        scores_list = img_df[score_col].tolist()
        keep = nms(proposals_list, scores_list, iou_threshold)
        kept_idxs = img_df.index[keep]
        kept_rows.extend(img_df.loc[kept_idxs].to_dict("records"))
    return pd.DataFrame(kept_rows)

pos_df_nms = apply_nms_per_image(pos_df, iou_threshold=0.5)
print(f"After NMS: {len(pos_df_nms)} proposals (from {len(pos_df)})")"""))

cells.append(make_code("""# Visualize pseudo-labels vs GT
def visualize_pseudo_labels(fname, ax, pos_df, title_prefix=""):
    img_path = os.path.join(IMAGES_DIR, fname)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)

    # GT boxes
    ann = all_annotations[fname]
    for box in ann["boxes"]:
        xmin, ymin, xmax, ymax = box["bbox"]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)

    # Pseudo-label boxes
    img_df = pos_df[pos_df["filename"] == fname]
    colors = {"WBC": "cyan", "RBC": "yellow", "Platelets": "magenta"}
    for _, row in img_df.iterrows():
        prop = row["proposal"]
        rect = patches.Rectangle(
            (prop[0], prop[1]), prop[2] - prop[0], prop[3] - prop[1],
            linewidth=2, edgecolor=colors.get(row["snorkel_label_name"], "red"),
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(prop[0], prop[1] - 8, f"{row['snorkel_label_name']}:{row['snorkel_confidence']:.2f}",
                color=colors.get(row["snorkel_label_name"], "red"), fontsize=8)

    ax.set_title(f"{title_prefix}{fname} (GT=--, Pseudo=-)")
    ax.axis("off")

sample_files_vis = [f for f in all_filenames if f in pos_df_nms["filename"].values][:4]
if len(sample_files_vis) < 4:
    sample_files_vis = list(pos_df_nms["filename"].unique())[:4]

if len(sample_files_vis) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, fname in zip(axes.flat, sample_files_vis):
        visualize_pseudo_labels(fname, ax, pos_df_nms)
    plt.tight_layout()
    plt.show()

# Quality assessment
pos_gt = pos_df_nms[pos_df_nms["best_iou"] > 0.5]
bg_gt = pos_df_nms[pos_df_nms["best_iou"] <= 0.5]
print(f"\\nQuality of pseudo-labels:")
print(f"  Correct (IoU>0.5 with GT): {len(pos_gt)} ({len(pos_gt)/max(len(pos_df_nms),1):.1%})")
print(f"  Background (no match): {len(bg_gt)} ({len(bg_gt)/max(len(pos_df_nms),1):.1%})")
if len(pos_gt) > 0:
    correct_class = (pos_gt["snorkel_label"] == pos_gt["best_class"]).mean()
    print(f"  Class accuracy (on matched): {correct_class:.1%}")"""))

# ============================================================
# SECTION 8: COCO Format Conversion
# ============================================================
cells.append(make_md("""## 8. COCO Format Conversion

We convert the pseudo-labels to COCO JSON format for DETR training."""))

cells.append(make_code("""# @title Build COCO Dataset
def build_coco_json(pos_df, filenames, output_path="pseudo_annotations.json"):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "WBC"},
            {"id": 1, "name": "RBC"},
            {"id": 2, "name": "Platelets"},
        ],
    }
    ann_id = 0
    for img_id, fname in enumerate(filenames):
        img_path = os.path.join(IMAGES_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        coco["images"].append({
            "id": img_id, "file_name": fname, "width": w, "height": h
        })
        img_df = pos_df[pos_df["filename"] == fname]
        for _, row in img_df.iterrows():
            prop = row["proposal"]
            xmin, ymin, xmax, ymax = prop
            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(row["snorkel_label"]),
                "bbox": [int(xmin), int(ymin), int(bw), int(bh)],
                "area": int(bw * bh),
                "iscrowd": 0,
                "score": float(row["snorkel_confidence"]),
            })
            ann_id += 1

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"Saved {len(coco['annotations'])} annotations for {len(coco['images'])} images to {output_path}")
    return coco

coco_data = build_coco_json(pos_df_nms, all_filenames, "weak_annotations/coco.json")

# Also build ground truth COCO for evaluation
def build_gt_coco(filenames, output_path="gt_annotations.json"):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "WBC"},
            {"id": 1, "name": "RBC"},
            {"id": 2, "name": "Platelets"},
        ],
    }
    ann_id = 0
    for img_id, fname in enumerate(filenames):
        img_path = os.path.join(IMAGES_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})
        ann = all_annotations[fname]
        for box in ann["boxes"]:
            xmin, ymin, xmax, ymax = box["bbox"]
            bw, bh = xmax - xmin, ymax - ymin
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": box["class_id"],
                "bbox": [xmin, ymin, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1
    with open(output_path, "w") as f:
        json.dump(coco, f)
    print(f"Saved {len(coco['annotations'])} GT annotations for {len(coco['images'])} images to {output_path}")
    return coco

_ = build_gt_coco(all_filenames, "weak_annotations/gt.json")"""))

cells.append(make_code("""# Create PyTorch Dataset for DETR
class BCCDDataset(Dataset):
    def __init__(self, coco_json_path, img_dir, image_processor):
        with open(coco_json_path) as f:
            self.coco = json.load(f)
        self.img_dir = img_dir
        self.image_processor = image_processor
        self.image_id_to_data = {img["id"]: img for img in self.coco["images"]}
        self.annotations_by_image = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)

    def __len__(self):
        return len(self.coco["images"])

    def __getitem__(self, idx):
        img_info = self.coco["images"][idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotations = self.annotations_by_image.get(img_info["id"], [])
        labels = [ann["category_id"] for ann in annotations]
        boxes = [ann["bbox"] for ann in annotations]

        # COCO bbox format: [x, y, w, h]
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)

        encoding = self.image_processor(
            images=image,
            annotations={"image_id": img_info["id"], "objects": {
                "bbox": boxes_tensor, "category_id": labels_tensor,
                "area": boxes_tensor[:, 2] * boxes_tensor[:, 3] if len(boxes_tensor) > 0 else torch.zeros(0),
                "iscrowd": torch.zeros(len(boxes_tensor)) if len(boxes_tensor) > 0 else torch.zeros(0),
            }},
            return_tensors="pt",
        )
        # Remove batch dimension added by processor
        for k in encoding:
            if isinstance(encoding[k], torch.Tensor) and encoding[k].dim() > 0:
                encoding[k] = encoding[k].squeeze(0)
        encoding["orig_size"] = torch.tensor([image.shape[0], image.shape[1]])
        encoding["image_id"] = img_info["id"]
        return encoding

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Split into train/val
random.seed(42)
all_ids = sorted(coco_data["images"], key=lambda x: x["id"])
random.shuffle(all_ids)
split_idx = int(len(all_ids) * 0.8)
train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

def subset_coco(coco_data, image_ids_subset):
    subset = {
        "images": image_ids_subset,
        "categories": coco_data["categories"],
        "annotations": [a for a in coco_data["annotations"] if a["image_id"] in {i["id"] for i in image_ids_subset}],
    }
    return subset

def save_subset_coco(coco_subset, path):
    with open(path, "w") as f:
        json.dump(coco_subset, f)

os.makedirs("weak_annotations", exist_ok=True)
train_coco = subset_coco(coco_data, train_ids)
val_coco = subset_coco(coco_data, val_ids)
save_subset_coco(train_coco, "weak_annotations/train_coco.json")
save_subset_coco(val_coco, "weak_annotations/val_coco.json")

train_dataset = BCCDDataset("weak_annotations/train_coco.json", IMAGES_DIR, processor)
val_dataset = BCCDDataset("weak_annotations/val_coco.json", IMAGES_DIR, processor)

print(f"Train dataset: {len(train_dataset)} images, "
      f"{sum(len(v) for v in train_dataset.annotations_by_image.values())} annotations")
print(f"Val dataset: {len(val_dataset)} images, "
      f"{sum(len(v) for v in val_dataset.annotations_by_image.values())} annotations")"""))

# ============================================================
# SECTION 9: DETR Fine-Tuning
# ============================================================
cells.append(make_md("""## 9. DETR Fine-Tuning

We fine-tune a pretrained DETR model using our weakly-labeled data."""))

cells.append(make_code("""# @title Initialize DETR Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=3,
    ignore_mismatched_sizes=True,
)
model.to(device)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")"""))

cells.append(make_code("""# @title Training Arguments
training_args = TrainingArguments(
    output_dir="detr_bccd_weak",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    evaluation_strategy="steps",
    eval_steps=20,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=1e-4,
    remove_unused_columns=False,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    push_to_hub=False,
)

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]
    orig_sizes = torch.stack([item["orig_size"] for item in batch])
    image_ids = [item["image_id"] for item in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels, "orig_sizes": orig_sizes, "image_ids": image_ids}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    tokenizer=processor,
)

print("Trainer initialized. Starting training...")
print("(Training may take 5-15 minutes depending on GPU)")
results = trainer.train()

print("\\nTraining complete!")
print(f"Final loss: {results.training_loss:.4f}")"""))

cells.append(make_code("""# Save the fine-tuned model
model.save_pretrained("detr_bccd_weak/final")
processor.save_pretrained("detr_bccd_weak/final")
print("Model saved to detr_bccd_weak/final/")"""))

# ============================================================
# SECTION 10: Evaluation
# ============================================================
cells.append(make_md("""## 10. Evaluation

We evaluate our weakly-supervised model using mAP and visualize predictions."""))

cells.append(make_code("""# @title Evaluation with pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load GT annotations
coco_gt = COCO("weak_annotations/gt.json")

# We'll evaluate on our validation set
val_fnames = {img["file_name"] for img in val_coco["images"]}
val_img_ids = [img["id"] for img in val_coco["images"]]"""))

cells.append(make_code("""# @title Run inference on validation set
model.eval()
results_list = []

with torch.no_grad():
    for batch in DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        orig_sizes = batch["orig_sizes"].to(device)
        image_ids = batch["image_ids"]
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        processed = processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=orig_sizes
        )

        for i, result in enumerate(processed):
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                results_list.append({
                    "image_id": image_ids[i],
                    "category_id": int(label),
                    "bbox": box.tolist(),
                    "score": float(score),
                })
                # Limit results to avoid memory issues
                if len(results_list) >= 1000:
                    break
            if len(results_list) >= 1000:
                break
        if len(results_list) >= 1000:
            break

coco_dt = coco_gt.loadRes(results_list)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.params.imgIds = val_img_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print(f"\\n{'='*50}")
print(f"Final Results:")
print(f"  mAP@0.5: {coco_eval.stats[1]:.3f}")
print(f"  mAP@0.5:0.95: {coco_eval.stats[0]:.3f}")

# Per-class metrics
print(f"\\nPer-class AP@0.5:")
for cat in coco_gt.dataset["categories"]:
    cat_id = cat["id"]
    coco_eval.params.catIds = [cat_id]
    coco_eval.params.imgIds = val_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    ap50 = coco_eval.stats[1]
    print(f"  {cat['name']}: {ap50:.3f}")"""))

cells.append(make_code("""# Visualize predictions
model.eval()
sample_val_files = [f for f in all_filenames if f in val_fnames][:10]

n_cols = 5
n_rows = (len(sample_val_files) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
colors = {0: "cyan", 1: "yellow", 2: "magenta"}

for ax, fname in zip(axes.flat, sample_val_files):
    img_path = os.path.join(IMAGES_DIR, fname)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)

    # GT boxes
    ann = all_annotations[fname]
    for box in ann["boxes"]:
        xmin, ymin, xmax, ymax = box["bbox"]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)

    # Run inference
    encoding = processor(images=img_rgb, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = torch.tensor([[img_rgb.shape[0], img_rgb.shape[1]]]).to(device)
    results = processor.post_process_object_detection(
        outputs, threshold=0.3, target_sizes=target_sizes
    )[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.cpu().tolist()
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=colors.get(int(label), "red"), facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 8, f"{CLASS_NAMES[int(label)]}:{score:.2f}",
                color=colors.get(int(label), "red"), fontsize=9)

    ax.set_title(fname)
    ax.axis("off")

plt.suptitle("Predictions (GT=--lime, Predicted=Cyan/Yellow/Magenta)", fontsize=14)
plt.tight_layout()
plt.show()"""))

cells.append(make_code("""# Final summary
print("\\n" + "="*60)
print("SNORKEL + DETR FOR BCCD OBJECT DETECTION: SUMMARY")
print("="*60)
print(f"\\nDataset: BCCD ({len(all_annotations)} images, 3 classes)")
print(f"Region proposals: Selective Search (up to {MAX_PROPOSALS} per image)")
print(f"Labeling functions: {len(lfs)} heuristic LFs")
print(f"  - lf_size, lf_color_rbc, lf_color_wbc")
print(f"  - lf_aspect_ratio, lf_edge_density, lf_position")
print(f"LabelModel: Trained on {len(df_lf)} proposals")
print(f"  - High-confidence pseudo-labels: {len(pos_df_nms)} (thresh={CONFIDENCE_THRESHOLD})")
print(f"  - Quality: {(len(pos_gt)/max(len(pos_df_nms),1)):.1%} match GT objects")
print(f"DETR Model: facebook/detr-resnet-50")
print(f"  - Training: 10 epochs on weak labels")
print(f"  - Eval: mAP@0.5 = {coco_eval.stats[1]:.3f}")

print(f"\\n{'='*60}")
print("TUTORIAL COMPLETE!")
print("="*60)"""))

cells.append(make_md("""## Conclusion

This tutorial demonstrated how to use **Snorkel** for programmatic data labeling in an object detection pipeline:

1. **Region proposals** via Selective Search generated candidate boxes
2. **Labeling functions** encoded domain knowledge about blood cell appearance
3. **LabelModel** combined weak signals into probabilistic labels
4. **Pseudo-labels** were filtered and used to train a **DETR** object detector
5. **Evaluation** showed the model learned meaningful detection from weak supervision

### Next Steps
- Add more labeling functions targeting specific failure modes
- Use active learning to prioritize proposals the LabelModel is uncertain about
- Scale to more images (BCCD is small — try with all 364 images)
- Try different backbones (YOLO, Faster R-CNN) with the same weak labels"""))

# ============================================================
# Assemble notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = "notebooks/snorkel_bccd_object_detection.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {output_path}")
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code: {sum(1 for c in cells if c['cell_type'] == 'code')}")
