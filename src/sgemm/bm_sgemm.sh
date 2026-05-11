#!/bin/bash

SRC="./include/sgemm_global.cuh"
BIN="./build/sgemm_v6"

echo "========================================="
echo "TEST TARGET: $BIN"
echo "CONFIG FILE: $SRC"
echo "========================================="

CASES=(
"128 128 128 perfect"
"131 131 131 tail"
"257 257 257 boundary"
"1024 8 1024 skinny_k"
"256 4096 256 huge_k"
"512 768 768 transformer"
"789 567 678 irregular"
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

    RESULT=$(echo "$OUTPUT" | awk '/RESULT/ {print $2}')
    MAX_ABS=$(echo "$OUTPUT" | awk '/MAX_ABS/ {print $2}')
    MAX_REL=$(echo "$OUTPUT" | awk '/MAX_REL/ {print $2}')
    BAD_COUNT=$(echo "$OUTPUT" | awk '/BAD_COUNT/ {print $2}')

    printf "%-25s %-8s abs=%-12s rel=%-12s bad=%s\n" \
    "{$M_DIM,$K_DIM,$N_DIM}" \
    "$RESULT" \
    "$MAX_ABS" \
    "$MAX_REL" \
    "$BAD_COUNT"

done