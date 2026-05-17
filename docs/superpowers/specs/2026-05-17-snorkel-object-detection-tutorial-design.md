# Snorkel Weak Supervision for BCCD Object Detection — Tutorial Notebook

## Overview

A Colab-optimized Jupyter notebook demonstrating how to use Snorkel for programmatic data labeling in an object detection pipeline. Uses weak supervision on region proposals to train a DETR model on the BCCD blood cell detection dataset.

## Dataset: BCCD

- 364 images, 3 classes: WBC (white blood cells), RBC (red blood cells), Platelets
- VOC-format XML annotations (bounding boxes + class labels)
- Downloadable from GitHub: https://github.com/Shenggan/BCCD_Dataset
- Known characteristics inform heuristic labeling functions

## Pipeline

```
BCCD images → Selective Search proposals → Feature extraction
→ Snorkel LFs → LabelModel → pseudo-boxes → COCO format
→ fine-tune DETR (HuggingFace) → evaluate mAP
```

## Labeling Functions (6 total)

| LF | Heuristic | Output |
|---|---|---|
| `lf_size` | Area thresholds: >3000px→WBC, 800-3000→RBC, <800→Platelet | WBC/RBC/Platelet/ABSTAIN |
| `lf_color_rbc` | High red mean, low blue mean → RBC | RBC/ABSTAIN |
| `lf_color_wbc` | High blue/purple channel, large area → WBC | WBC/ABSTAIN |
| `lf_aspect_ratio` | AR near 1.0 (round) → RBC, AR > 1.5 → WBC | RBC/WBC/ABSTAIN |
| `lf_edge_density` | High edge density (Canny) suggests nucleus → WBC | WBC/ABSTAIN |
| `lf_background` | Low color variance + low edge density → background | ABSTAIN |

All LFs return {WBC, RBC, Platelet, ABSTAIN}. LabelModel resolves conflicts.

## Notebook Structure (10 sections)

1. **Setup & Installation** — `!pip install snorkel transformers torchvision opencv-python matplotlib seaborn`
2. **Dataset Download & Exploration** — Clone BCCD, parse XML, visualize samples, compute summary stats
3. **Region Proposal Generation** — SelectiveSearch (OpenCV `ximgproc.segmentation`), ~500-2000 proposals/image, compute IoU with GT for reference
4. **Feature Extraction** — Per-proposal: area, aspect ratio, mean RGB, std RGB, Canny edge density, bounding box center
5. **Snorkel Labeling Functions** — Implement 6 LFs, test on sample proposals, apply to full dataset
6. **Label Model** — `LabelModel(cardinality=3)`, train, analyze: coverage, conflicts, empirical accuracy vs GT
7. **Pseudo-Label Assignment** — Filter proposals with confidence > 0.7, assign class from LabelModel, suppress duplicates via NMS
8. **COCO Format Conversion** — Build COCO JSON from pseudo-labels, create `BCCDDataset` for HuggingFace
9. **DETR Fine-Tuning** — `DetrForObjectDetection` from pretrained, `transformers.Trainer`, 10 epochs
10. **Evaluation** — mAP@0.5, per-class precision/recall, sample visualizations, comparison with fully-supervised baseline

## Technical Decisions

- **Selective Search over other RPNs**: No GPU needed for proposal generation, deterministic, well-understood
- **DETR over Faster R-CNN**: Simpler loss (bipartite matching), no anchor tuning, works well with HuggingFace Trainer
- **COCO format**: Native format for DETR, standard evaluation (pycocotools)
- **NMS after pseudo-labels**: Redundant proposals from Selective Search need suppression
- **Colab-optimized**: Self-contained cells, GPU for DETR training, ~30min total runtime

## Success Criteria

- Notebook runs end-to-end in Colab without errors
- Achieves >0.3 mAP@0.5 on BCCD val set (reasonable for weak supervision tutorial)
- All 6 LFs show non-zero coverage
- LabelModel improves over any single LF
- Visualizations present in every section
