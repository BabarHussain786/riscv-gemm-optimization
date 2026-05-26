#!/bin/bash

# builder.sh v1
# usage: <ARCH_TARGET> [all options for compile with gcc ..]
# - this file compiles the target if it is possible
# - if the arch can, try to compile te target else does nothing


# Options

CMD_X86=gcc
CMD_QEMU=/home/sergio/UniSpec/HPC/PROGETTO/riscv64-elf-gcc/bin/riscv64-unknown-elf-gcc
CMD_RISCV=riscv64-linux-gnu-gcc

#----------------------------------------

ARCH_HOST=$(uname -m)
ARCH_TARGET=$1

## riscv_qemu (from x86)
if [ "$ARCH_HOST" == "x86_64" ] && [ "$ARCH_TARGET" == "riscv64_emu" ]; then
    $CMD_QEMU ${@:2}

## x86 (from x86)
elif [ "$ARCH_HOST" == "x86_64" ] && [ "$ARCH_TARGET" == "x86_64" ]; then
    $CMD_X86 ${@:2}

## riscv64 (from riscv64)
elif [ "$ARCH_HOST" == "riscv64" ] && [ "$ARCH_TARGET" == "riscv64" ]; then
    $CMD_RISCV ${@:2}

else
    echo "No (Arch:$ARCH_HOST <-> Target:$ARCH_TARGET)"
    exit 0 # no error, to avoid break in make
fi
