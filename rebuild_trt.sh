#!/bin/bash
# Rebuild TensorRT engines from ONNX models in WSL
# Must be run inside WSL on the target GPU to avoid platform mismatch errors
cd /home/siddarthag/TalkToMe
source venv/bin/activate

ONNX_DIR="checkpoints/ditto_onnx"
TRT_DIR="checkpoints/ditto_trt"
mkdir -p "$TRT_DIR"

echo "=== Rebuilding TensorRT Engines ==="
echo "TensorRT version: $(python3 -c 'import tensorrt; print(tensorrt.__version__)')"
echo ""

# Remove ALL existing .engine files so we force a clean rebuild
echo "Removing old .engine files (platform mismatch protection)..."
rm -f "$TRT_DIR"/*.engine
echo ""

# Models to build
declare -A MODELS=(
    ["appearance_extractor"]="fp16"
    ["motion_extractor"]="fp32"
    ["stitch_network"]="fp16"
    ["decoder"]="fp16"
    ["lmdm_v0.4_hubert"]="fp32"
    ["insightface_det"]="fp16"
    ["landmark106"]="fp16"
    ["landmark203"]="fp16"
    ["blaze_face"]="fp16"
    ["face_mesh"]="fp16"
)

BUILT=0
FAILED=0

for model in "${!MODELS[@]}"; do
    prec="${MODELS[$model]}"
    onnx="$ONNX_DIR/${model}.onnx"
    engine="$TRT_DIR/${model}_${prec}.engine"
    
    if [ ! -f "$onnx" ]; then
        echo "⚠ ONNX not found: $onnx"
        continue
    fi
    
    echo "Building $model ($prec)..."
    
    EXTRA_ARGS=""
    if [ "$prec" = "fp16" ]; then
        EXTRA_ARGS="--fp16"
    fi
    
    if polygraphy convert "$onnx" -o "$engine" --convert-to trt $EXTRA_ARGS 2>&1; then
        echo "✓ Built $model"
        BUILT=$((BUILT + 1))
    else
        echo "✗ Failed $model"
        FAILED=$((FAILED + 1))
    fi
done

# Build hubert with dynamic shapes
HUBERT_ONNX="$ONNX_DIR/hubert.onnx"
HUBERT_ENGINE="$TRT_DIR/hubert_fp32.engine"
if [ -f "$HUBERT_ONNX" ]; then
    echo "Building hubert (fp32, dynamic shapes)..."
    if polygraphy convert "$HUBERT_ONNX" -o "$HUBERT_ENGINE" \
        --convert-to trt \
        --trt-min-shapes input_values:[1,3200] \
        --trt-opt-shapes input_values:[1,16000] \
        --trt-max-shapes input_values:[1,48000] 2>&1; then
        echo "✓ Built hubert"
        BUILT=$((BUILT + 1))
    else
        echo "✗ Failed hubert"
        FAILED=$((FAILED + 1))
    fi
fi

# Copy warp_network.pth (uses PyTorch - GridSample3D not supported in TRT)
WARP_PTH="checkpoints/ditto_pytorch/models/warp_network.pth"
if [ -f "$WARP_PTH" ]; then
    cp "$WARP_PTH" "$TRT_DIR/warp_network.pth"
    echo "✓ Copied warp_network.pth (hybrid mode)"
fi

# Copy hubert.onnx for streaming
if [ -f "$ONNX_DIR/hubert.onnx" ]; then
    cp "$ONNX_DIR/hubert.onnx" "$TRT_DIR/hubert.onnx"
fi

echo ""
echo "=== Build Complete ==="
echo "Built: $BUILT, Failed: $FAILED"
echo ""
ls -sh "$TRT_DIR"/*.engine 2>/dev/null | head -10
