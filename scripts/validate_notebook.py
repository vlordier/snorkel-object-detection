"""Validate the tutorial notebook by checking structure and content."""

import json
import sys

NOTEBOOK_PATH = "notebooks/snorkel_bccd_object_detection.ipynb"

with open(NOTEBOOK_PATH) as f:
    nb = json.load(f)

errors = []

cells = nb["cells"]
if len(cells) < 10:
    errors.append(f"Expected >=10 cells, got {len(cells)}")

markdown_cells = [c for c in cells if c["cell_type"] == "markdown"]
code_cells = [c for c in cells if c["cell_type"] == "code"]

if len(markdown_cells) < 5:
    errors.append(f"Expected >=5 markdown cells, got {len(markdown_cells)}")
if len(code_cells) < 15:
    errors.append(f"Expected >=15 code cells, got {len(code_cells)}")

all_source = "\n".join(
    "".join(c["source"]) for c in cells if c["cell_type"] == "code"
)

key_imports = ["snorkel", "transformers", "DetrForObjectDetection", "cv2"]
for imp in key_imports:
    if imp not in all_source:
        errors.append(f"Missing import/reference: {imp}")

key_fns = [
    "lf_size", "lf_red_ratio", "lf_blue_ratio",
    "lf_aspect_ratio", "lf_color_std", "lf_position",
    "LabelModel", "generate_proposals", "extract_features",
    "nms", "build_coco_json",
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
    print(f"  {len(markdown_cells)} markdown + {len(code_cells)} code cells")
