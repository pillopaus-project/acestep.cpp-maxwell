#!/bin/bash
# Roundtrip: audio -> understand -> SFT DiT -> 4 WAV variations
#
# Usage: ./understand-roundtrip.sh input.wav (or input.mp3)
#
# understand:
# input -> ace-understand.json (audio codes + metadata)
#
# dit-vae:
# ace-understand.json -> output0.wav .. output3.wav

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.wav|input.mp3>"
    exit 1
fi

input="$1"

cd ..

SECONDS=0

build-cuda/ace-understand \
    --src-audio "$input" \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --model models/acestep-5Hz-lm-0.6B-Q8_0.gguf \
    -o ace-understand-turbo-0.6-cuda.json

CUSEC=$SECONDS

echo "CUDA done in $CUSEC s"


SECONDS=0

build-cpu/ace-understand \
    --src-audio audio-in/"$input" \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf \
    --model models/acestep-5Hz-lm-0.6B-Q8_0.gguf \
    --vae-chunk 1024 \
    --max-seq 16384 \
    -o ace-understand-turbo-0.6-cpu.json

CPSEC=$SECONDS

echo "CPU done in $CPSEC s"

DIFF=($CPSEC-$CUSEC)

echo "difference = $DIFF"

#build-cuda/dit-vae \
#    --request ace-understand-sft.json \
#    --text-encoder models/Qwen3-Embedding-0.6B-Q8_0.gguf \
#    --dit models/acestep-v15-sft-Q8_0.gguf \
#    --vae models/vae-BF16.gguf \
