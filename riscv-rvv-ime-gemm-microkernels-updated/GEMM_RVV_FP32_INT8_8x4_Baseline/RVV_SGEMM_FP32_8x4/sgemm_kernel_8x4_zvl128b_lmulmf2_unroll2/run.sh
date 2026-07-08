#!/usr/bin/env bash

LOGFILE="benchmark_8x8_mf2_unroll2_$(date +%Y%m%d_%H%M%S).log"

echo "=======================================" | tee "$LOGFILE"
echo "RVV SGEMM BENCHMARK (8x8 - LMUL 1/2 - Unroll 2)" | tee -a "$LOGFILE"
echo "Each shape runs 6 times" | tee -a "$LOGFILE"
echo "=======================================" | tee -a "$LOGFILE"

make clean >> "$LOGFILE" 2>&1
make >> "$LOGFILE" 2>&1

if [ ! -f ./bench ]; then
    echo "Compilation failed! Please run 'make' manually to see the error." | tee -a "$LOGFILE"
    exit 1
fi

run_case() {
    NAME=$1
    M=$2
    N=$3
    K=$4

    echo "" | tee -a "$LOGFILE"
    echo "Shape: $NAME (${M}x${N}x${K})" | tee -a "$LOGFILE"

    for i in 1 2 3 4 5 6
    do
        echo "Run $i" | tee -a "$LOGFILE"
        ./bench $M $N $K | tee -a "$LOGFILE"
    done
}

# Strictly running Square shape
run_case Square 4096 4096 4096

echo "Done." | tee -a "$LOGFILE"
