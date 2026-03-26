#!/bin/bash
# Prerequisites:
#   - git-lfs installed (for HuggingFace downloads)
#   - wget or curl
#   - unzip
#   - ~25GB free disk space
#
# Usage:
#   ./setup_datasets.sh
# ============================================================

set -e  # Exit on error

export http_proxy=http://proxy:8888
export https_proxy=$http_proxy

PROJECT_DIR="$(pwd)"
DATA_DIR="$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo "Data directory:    $DATA_DIR"
echo ""

# ---- Create directory structure ----

echo "  Directory tree:"
echo "  data/"
echo "  ├── __init__.py"
echo "  ├── referring.py              <-- your dataset file"
echo "  ├── Helpers/"
echo "  │   ├── __init__.py"
echo "  │   └── grefer.py             <-- gRefCOCO helper"
echo "  ├── grefcoco/"
echo "  │   ├── grefs(unc).json       <-- from HuggingFace"
echo "  │   └── instances.json        <-- from HuggingFace"
echo "  ├── images/"
echo "  │   └── train2014/            <-- COCO train2014 images"
echo "  │       └── COCO_train2014_000000XXXXXX.jpg ..."
echo "  └── ref-youtube-vos/"
echo "      ├── train/"
echo "      │   ├── JPEGImages/"
echo "      │   │   └── <video_id>/"
echo "      │   │       └── XXXXX.jpg ..."
echo "      │   └── Annotations/"
echo "      │       └── <video_id>/"
echo "      │           └── XXXXX.png ..."
echo "      └── meta_expressions/"
echo "          └── train/"
echo "              └── meta_expressions.json"
echo ""

# ---- Download gRefCOCO ----
echo "[1/4] Downloading gRefCOCO from HuggingFace..."
echo "  This requires git-lfs. Install with: sudo apt install git-lfs && git lfs install"
echo ""

if [ ! -f "$DATA_DIR/grefcoco/grefs(unc).json" ] || [ ! -f "$DATA_DIR/grefcoco/instances.json" ]; then
    # Method 1: Direct download via wget (simpler, no git-lfs needed)
    echo "  Downloading grefs(unc).json (~59 MB)..."
    wget -q --show-progress -O "$DATA_DIR/grefcoco/grefs(unc).json" \
        "https://huggingface.co/datasets/FudanCVL/gRefCOCO/resolve/main/grefs(unc).json" \
        2>&1 || echo "  WARNING: wget failed. See manual instructions below."

    echo "  Downloading instances.json (~120 MB)..."
    wget -q --show-progress -O "$DATA_DIR/grefcoco/instances.json" \
        "https://huggingface.co/datasets/FudanCVL/gRefCOCO/resolve/main/instances.json" \
        2>&1 || echo "  WARNING: wget failed. See manual instructions below."
else
    echo "  gRefCOCO files already exist, skipping download."
fi

echo ""

# ---- Download COCO train2014 images ----
echo "[2/4] Downloading COCO train2014 images..."
echo "  This is ~13 GB and will take a while."
echo ""

if [ ! -d "$DATA_DIR/images/train2014" ] || [ -z "$(ls -A "$DATA_DIR/images/train2014" 2>/dev/null)" ]; then
    echo "  Downloading train2014.zip from COCO..."
    wget -q --show-progress -O "$DATA_DIR/images/train2014.zip" \
        "http://images.cocodataset.org/zips/train2014.zip" \
        2>&1 || echo "  WARNING: wget failed. See manual instructions below."

    if [ -f "$DATA_DIR/images/train2014.zip" ]; then
        echo "  Extracting train2014.zip..."
        unzip -q "$DATA_DIR/images/train2014.zip" -d "$DATA_DIR/images/"
        rm "$DATA_DIR/images/train2014.zip"
        echo "  Extracted $(ls "$DATA_DIR/images/train2014/" | wc -l) images."
    fi
else
    echo "  COCO train2014 images already exist, skipping download."
fi

echo ""

# ---- Set up Ref-YouTube-VOS ----
echo "[3/4] Setting up Ref-YouTube-VOS..."
echo ""
echo "  Ref-YouTube-VOS must be downloaded manually from the competition page:"
echo "    https://competitions.codalab.org/competitions/29139"
echo ""
echo "  You need to register and download the training set zip file."
echo ""
echo "  After downloading, extract it so the structure looks like:"
echo "    data/ref-youtube-vos/"
echo "    ├── train/"
echo "    │   ├── JPEGImages/<video_id>/*.jpg"
echo "    │   └── Annotations/<video_id>/*.png"
echo "    └── meta_expressions/"
echo "        └── train/"
echo "            └── meta_expressions.json"
echo ""

# If the user has already downloaded a zip, try to find and extract it
REFYTVOS_ZIP=$(find "$DATA_DIR" "$PROJECT_DIR" -maxdepth 2 -name "train.zip" -o -name "*rvos*.zip" 2>/dev/null | head -1)
if [ -n "$REFYTVOS_ZIP" ]; then
    echo "  Found zip file: $REFYTVOS_ZIP"
    echo "  Extracting to $DATA_DIR/ref-youtube-vos/..."
    unzip -q -o "$REFYTVOS_ZIP" -d "$DATA_DIR/ref-youtube-vos/"
    echo "  Done."
elif [ -d "$DATA_DIR/ref-youtube-vos/train/JPEGImages" ]; then
    echo "  Ref-YouTube-VOS data already exists."
else
    echo "  No zip file found. Please download and extract manually."
    echo "  Then re-run this script or just ensure the directory structure matches above."
fi

echo ""

# ---- Verify setup ----
echo "[4/4] Verifying dataset setup..."
echo ""

ERRORS=0

# Check gRefCOCO
if [ -f "$DATA_DIR/grefcoco/grefs(unc).json" ]; then
    echo "  [OK] grefs(unc).json found ($(du -h "$DATA_DIR/grefcoco/grefs(unc).json" | cut -f1))"
else
    echo "  [MISSING] grefs(unc).json"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$DATA_DIR/grefcoco/instances.json" ]; then
    echo "  [OK] instances.json found ($(du -h "$DATA_DIR/grefcoco/instances.json" | cut -f1))"
else
    echo "  [MISSING] instances.json"
    ERRORS=$((ERRORS + 1))
fi

# Check COCO images
if [ -d "$DATA_DIR/images/train2014" ]; then
    IMG_COUNT=$(ls "$DATA_DIR/images/train2014/" 2>/dev/null | wc -l)
    if [ "$IMG_COUNT" -gt 80000 ]; then
        echo "  [OK] COCO train2014: $IMG_COUNT images (expected ~82,783)"
    else
        echo "  [WARN] COCO train2014: only $IMG_COUNT images (expected ~82,783)"
    fi
else
    echo "  [MISSING] COCO train2014 images directory"
    ERRORS=$((ERRORS + 1))
fi

# Check Ref-YouTube-VOS
if [ -d "$DATA_DIR/ref-youtube-vos/train/JPEGImages" ]; then
    VID_COUNT=$(ls "$DATA_DIR/ref-youtube-vos/train/JPEGImages/" 2>/dev/null | wc -l)
    echo "  [OK] Ref-YouTube-VOS JPEGImages: $VID_COUNT videos"
else
    echo "  [MISSING] Ref-YouTube-VOS train/JPEGImages"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "$DATA_DIR/ref-youtube-vos/train/Annotations" ]; then
    echo "  [OK] Ref-YouTube-VOS Annotations directory found"
else
    echo "  [MISSING] Ref-YouTube-VOS train/Annotations"
    ERRORS=$((ERRORS + 1))
fi

META_FILE="$DATA_DIR/ref-youtube-vos/meta_expressions/train/meta_expressions.json"
if [ -f "$META_FILE" ]; then
    echo "  [OK] meta_expressions.json found ($(du -h "$META_FILE" | cut -f1))"
else
    echo "  [MISSING] meta_expressions.json"
    echo "           Expected at: $META_FILE"
    ERRORS=$((ERRORS + 1))
fi

# Check Python package files
if [ -f "$DATA_DIR/__init__.py" ] && [ -f "$DATA_DIR/Helpers/__init__.py" ]; then
    echo "  [OK] Python __init__.py files present"
else
    echo "  [MISSING] __init__.py files"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "============================================"
    echo "  All datasets ready!"
    echo "============================================"
else
    echo "============================================"
    echo "  Setup incomplete: $ERRORS issue(s) found."
    echo "  Fix the issues above and re-run."
    echo "============================================"
fi