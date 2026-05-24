#!/bin/bash

SRC="./reduce/reduce_global.cuh"
BIN="./build/reduce_v6"

echo "========================================="
echo "TEST TARGET: $BIN"
echo "CONFIG FILE: $SRC"
echo "========================================="

CASES=(
"1048576  256 small_size_1M"
# "33554432 256 standard_32M"
# "33554432 512 max_threads"
# "33554432 128 min_threads"
# "16777216 256 half_size_16M"
# "67108864 256 double_size_64M"
"33554431 256 non_power_of_two"
# "33554433 256 boundary_plus_one"
)

for CASE in "${CASES[@]}"
do
    read N_DIM TPB_DIM DESC <<< "$CASE"

    sed -i "s/constexpr int N\s*=\s*[^;]*;/constexpr int N = $N_DIM;/" $SRC
    sed -i "s/constexpr int THREAD_PER_BLOCK\s*=\s*[^;]*;/constexpr int THREAD_PER_BLOCK = $TPB_DIM;/" $SRC

    cmake --build build --parallel > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "{N:$N_DIM, TPB:$TPB_DIM}  $DESC"
        echo "build failed"
        echo ""
        continue
    fi

    OUTPUT=$($BIN)

    if echo "$OUTPUT" | grep -qi "right"; then
        RESULT="SUCCESS"
    else
        RESULT="FAILED"
    fi

    printf "%-35s %-12s %s\n" \
    "{N:$N_DIM, TPB:$TPB_DIM}" \
    "[$RESULT]" \
    "$DESC"

done