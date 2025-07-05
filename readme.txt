
See in this order
(scalar)
1)smatmul_baseline.c
2)smatmul_loopinterchange.c
3)smatmul_pfor.c
4)smatmul_tiling.c  ( + version v2, v3, v4 )
5)smatmul_recursive.c

(vectorial)
6)rvv_smatmul_recursive.c  (LMUL=1)
7)rvv2_smatmul_recursive.c (LMUL=2)


------------------

If there was benchmark in execution, 
try to check in the screen name "benchmark"

1. Make a screen:
screen -S benchmark

2. detach
ctrl-a then d

3. Resume the screen:
screen -r benchmark
