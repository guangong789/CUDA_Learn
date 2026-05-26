#!/bin/bash

SRC="./reduce/reduce_global.cuh"
BIN="./build/reduce_v4"

echo "========================================="
echo "TEST TARGET: $BIN"
echo "CONFIG FILE: $SRC"
echo "========================================="

FIXED_TPB=256

sed -i \
"s/constexpr int THREAD_PER_BLOCK\s*=\s*[^;]*;/constexpr int THREAD_PER_BLOCK = $FIXED_TPB;/" \
$SRC

CASES=(
"1048576   small_size_1M"
"33554432  standard_32M"
"16777216  half_size_16M"
"33554431  non_power_of_two"
"33554433  boundary_plus_one"
"67108864  double_size_64M"
)

for CASE in "${CASES[@]}"
do
    read N_DIM DESC <<< "$CASE"

    sed -i \
"s/constexpr int N\s*=\s*[^;]*;/constexpr int N = $N_DIM;/" \
$SRC

    cmake --build build --parallel > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        printf "%-35s %-12s %s\n" \
        "{N:$N_DIM}" \
        "[BUILD FAIL]" \
        "$DESC"
        continue
    fi

    OUTPUT=$($BIN)

    if echo "$OUTPUT" | grep -qi "right"; then
        RESULT="SUCCESS"
    else
        RESULT="FAILED"
    fi

    printf "%-35s %-12s %s\n" \
    "{N:$N_DIM}" \
    "[$RESULT]" \
    "$DESC"

done