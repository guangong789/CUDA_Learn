# `constexpr int M{4096}, K{4096}, N{4096};`
# VERSION REFERENCE: CUBLAS  
![alt text](assets/image.png)  
## SOL
![alt text](assets/image-1.png)  
# VERSION 0: Global Memory  
![alt text](assets/image-3.png)  
仅用 Global Mem 的版本用时为 cublas 的 10 倍
## Overview  
![alt text](assets/image-2.png)  
## SOL  
![alt text](assets/image-4.png)  
查看 NCU Sol 的数据发现，SM Throughput， Memory Throughput 以及 L1 Cache 全部接近 100%，但 L2 Cache 和 DRAM Throughput 却仅有 8%，这可能是因为程序频繁的 global load 请求已经将 L1 堵满了，指令大多 stall 在 L1，向 L2 和 DRAM 发射请求的效率很低  
## Memory WorkLoad Analysis  
![alt text](assets/image-5.png)  
![alt text](assets/image-6.png)  
在 `{4096, 4096, 4096}` 的测试规模下，由于每个线程只处理一个点，`value += A(row, k) * B(k, col)` 最终会有 $4096\times4096\times2\times4096\div32=4294967296$ 条指令，在同一个 warp 中，线程都访问的是 A 的同一个 float，触发广播，只需要一个 sector；对于 B，warp 会访问 32 个不同的 float，而一个 sector 32 字节，8 个 float，这样一条指令需要请求 4 个 sector，$(1+4)\div2=2.5(sector/request)$，与 NCU 显示数据相符合，此时程序的总 sector 就为 $4294967296\times2.5=10737418240$，sector misses to L2 一共有 536870912，$536870912\div10737418240=0.05$，刚好对应 memory analysis 显示的 L1 Hit Rate = 95%  
## Scheduler Statistics  
![alt text](assets/image-7.png)  
![alt text](assets/image-8.png)  
平均每个调度器下挂载的活跃线程束达到了 8 个，但是 Eligible 的线程束却只有 1.15 个，能发射的更是只有 0.23 个，调度器 No Eligible 的时间占据 76%，这也对应了 SOL 中的状态，由于 global memory 的访存延迟太高了，导致绝大多数指令都只能 stall，程序处在一种“虚假繁荣”的状态  
## Summary
L1TEX Global Load Access Pattern：The memory access pattern for global loads from L1TEX might not be optimal. On average, only 26.4 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads

NCU 显示 global load 并非最优，这是因为在 `A(row, k)` load 的时候，线程只需要一个 float 4字节，但是请求必须以 sector 32字节为单位，导致请求的数据没有完全利用。接下来考虑利用高速的片上共享内存来替代过多的全局内存访问，减小访存延迟，避免所有指令都通过 L1 Cache 向 global 请求数据。
![alt text](assets/image-11.png)  
naive SGEMM 性能只有 cublas 的 $\tfrac{1}{8}$，FLOPs 也仅达到 $\tfrac{1}{8}$
# VERSION 1: Shared Memory
![alt text](assets/image-12.png)  
使用 shared mem 后的 kernel 相比于 v0 有 20.7% latency reduction，1.26× speedup  
## Overview  
![alt text](assets/image-14.png)  
## SOL  
![alt text](assets/image-15.png)  
