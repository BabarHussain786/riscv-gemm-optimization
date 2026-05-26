#qemu-riscv64 ./$1
VLEN=256

echo "EMU Script v2"
echo "-> VLEN=$VLEN"
echo "-----------------------   START   -----------------------"

qemu-riscv64 -cpu rv64,v=true,zba=true,vlen=$VLEN,vext_spec=v1.0 $1 "${@:2}"
