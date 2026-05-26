## BUILD on RISC-V machine
echo "//riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d $1.c -o $1"
riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d $1.c -o $1
