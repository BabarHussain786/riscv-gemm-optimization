#!/bin/bash

# static options, here ...

# number of retry in current experiment
RETRY=3

# qemu launcher command and settings
VLEN=256
QEMU_EXEC="qemu-riscv64 -cpu rv64,v=true,zba=true,vlen=$VLEN,vext_spec=v1.0"

# filter benchmark lines
BENCHMARK_LINE_FILTER="BENCHMARK_RECORD"

#----------------------------------

echo "Testsuite V3"

# Check 3 params passed: exec_name, start_size, end_size
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <exec_name> <start_size> <end_size> [additional_args ...]"
    exit 1
fi

ARCH=$(uname -m)

if [ "$ARCH" = "riscv64" ]; then
    TYPE_EXEC="EXECUTION"
else
    TYPE_EXEC="EMULATION"
fi
echo "Arch: $ARCH ($TYPE_EXEC MODE)"

EXEC_NAME="$1"

START_SIZE="$2"
END_SIZE="$3"

ADDITIONAL_ARGS=("${@:4}")

echo "Target: $EXEC_NAME"
echo "Size: START:$START_SIZE -> END:$END_SIZE, $RETRY-Retry foreach experiment"
echo "Additional Args: $ADDITIONAL_ARGS"
echo " "

# function to run program
exec_prog() {
    local exec_file="$1"
    local size_param="$2"
    shift 2
    local other_params=("$@")

    echo "TEST>: $exec_file   size=$size_param   ${other_params[@]}"

    if [ "$ARCH" = "riscv64" ]; then
        "./$exec_file" "$size_param" "${other_params[@]}"
    else
        qemu-riscv64 -cpu rv64,v=true,zba=true,vlen=$VLEN,vext_spec=v1.0 "$exec_file" "$size_param" "${other_params[@]}"
    fi

}

# Loop from START_SIZE to END_SIZE
i=0
while true; do
    pow2=$(( 2 ** i ))
    if (( pow2 > END_SIZE )); then
        break
    elif (( pow2 >= START_SIZE )); then
        
    echo "> Size: $pow2"

        for ((j = 1; j <= RETRY; j++))
        do
            exec_prog "$EXEC_NAME" "$pow2" "${ADDITIONAL_ARGS[@]}" | grep "$BENCHMARK_LINE_FILTER"
        done

        echo " "

    fi
    ((i++))
done
