#!/bin/bash

SRC="./include/sgemm_global.cuh"
BIN="./build/sgemm_v1"

echo "========================================="
echo "TEST TARGET: $BIN"
echo "CONFIG FILE: $SRC"
echo "========================================="

CASES=(
"128 128 128 perfect_tile"
"128 128 130 k_tail"
"130 130 128 mn_tail"
"130 130 130 full_tail"
"129 129 129 float4_tail"
"130 130 130 float4_tail2"
"131 131 131 float4_tail3"
"255 255 255 warp_edge"
"257 257 257 block_edge"
"1024 8 1024 skinny_k"
"256 4096 256 huge_k"
"4096 16 16 skinny_output"
"512 768 768 transformer_ffn"
"1024 64 1024 attention"
"789 567 678 irregular"
# "2048 2048 2048 large"
"1025 1026 1027 pathological"
)

for CASE in "${CASES[@]}"
do
    read M_DIM K_DIM N_DIM DESC <<< "$CASE"

    sed -i "s/^constexpr int M{.*};/constexpr int M{$M_DIM};/" $SRC
    sed -i "s/^constexpr int K{.*};/constexpr int K{$K_DIM};/" $SRC
    sed -i "s/^constexpr int N{.*};/constexpr int N{$N_DIM};/" $SRC

    cmake --build build --parallel > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "{$M_DIM, $K_DIM, $N_DIM}  $DESC"
        echo "build failed"
        echo ""
        continue
    fi

    OUTPUT=$($BIN)

    MEAN_DIFF=$(echo "$OUTPUT" | awk '/mean_diff/ {print $3}')
    MEAN_REL=$(echo "$OUTPUT" | awk '/mean_rel/ {print $3}')

    echo "{$M_DIM, $K_DIM, $N_DIM}  $DESC"
    echo "mean_diff = $MEAN_DIFF"
    echo "mean_rel  = $MEAN_REL"

done