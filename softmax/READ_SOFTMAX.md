# `constexpr int M = 4096, N = 1024;`
# version 0: Shared Memory  
![alt text](assets/image.png)  
## Overview  
![alt text](assets/image-1.png)  
## SOL
![alt text](assets/image-2.png)  
第一版朴素的代码既不算 compute bound， 也不算 memory bound，由于是直接把整行搬到共享内存，所以 DRAM through 较低。  
## Scheduler Statistics  
![alt text](assets/image-5.png)  
![alt text](assets/image-6.png)  
调度器有超过 $\tfrac{2}{3}$ 的时间都在空转，可以看到活跃的 warps 很多，但是能够就绪和发射的很少。
## Warp Statistics  
![alt text](assets/image-7.png)  
stall 占比最高的就是 stall barrier，主要就是因为循环中有大量的 `__syncthreads()`，还有 shared mem 的频繁读写导致 MIO throttle 和 short scoreboard 也很高。
## Occupancy 
![alt text](assets/image-8.png)  
由于该版本采用了 1024 threads 的 block，导致在当前设备上一个 SM 只能放下 $\left\lfloor \frac{1024}{1536} \right\rfloor=1$ 个 block，占有率为 $\tfrac{2}{3}$。
## Summary  
ncu 提示硬件占有率并非最优，存在线程发散，这个是因为在 reduce 后期时一个 warp 中活跃的线程越来越少，导致了这种现象。
# VERSION 1: Shuffle  
![alt text](assets/image-9.png)  
Duration Reduction ≈ 32.8%，Speedup ≈ 1.49×  
这一版通过对最后一个 warp 进行寄存器洗牌，减少了访存 shared mem 时的同步屏障，也减少了上一版的线程发散。
## Memory Workload Analysis  
![alt text](assets/image-11.png)  
减少了对共享内存的访存，现在整体内存和管线压力也小了一些。  
## Warp Statistics
![alt text](assets/image-12.png)  
按理说减少了总的 `__syncthreads()`，stall barrier 应该减少，但是 ncu 却显示增加，因为 ncu warp stall 统计的是某个时钟周期 warp stall 的原因而不是代码 stall 的总时间，在这一版中大多数 stall 指标相对于上一版都减小了，shuffle 计算也更快，导致 stall 的主力 barrier stall 被放大了。
# VERSION 2: Float4 Vectorized  
![alt text](assets/image-13.png)  
Duration Reduction ≈ 83.0%，Speedup ≈ 5.88× 
使用一个线程算 4 个数据，现在一个 block 只有 256 个线程，理论硬件占有率能达到 100% 了。
## SOL  
![alt text](assets/image-14.png)  
内存吞吐量获得了很大的提升，可能是因为向量化的合并访存，指令数的减少以及 occupancy 的提升。
## Scheduler Statistics  
![alt text](assets/image-17.png)  
![alt text](assets/image-18.png)  
调度器的效率更高了，有 52% 的时间都有就绪的 warp。
## Warp Statistics  
![alt text](assets/image-19.png)  
这一版的 MIO throttle 降低很多，因为使用了更少指令的向量化加载，也因为现在指令发射更顺畅了，所以 stall long scoreboard 增加，指令访存所需的延迟暴露了出来。
# VERSION 3: Warp-level Block Reduction  
![alt text](assets/image-3.png)  
性能与上一版持平，在这一版中，使用 `warpReduce()` 函数直接对一个线程束进行归约，减少了大量之前循环中的 `__syncthreads()`，通过设置 `4 float per thread`，一行只需要 256 threads 即 8 个 warp，最后只需要把这 8 个 wapr 各自的结果存在 `shared mem[8]` 当中，在这里同步一次后再调用 `warpReduce()` 就能得到 row_max，后面的 sum 同理。
## SOL  
![alt text](assets/image-10.png)  
可以看到 shuffle 替代了大量的共享内存操作后，指令数更少了导致 SM throughput 减小，并且 L1 cache throughput 也因为更少的共享内存访存而减小。  
## Memory Workload Analysis
![alt text](assets/image-15.png)  
内存管线繁忙减小了，这是因为 shuffle 通过直接操作寄存器去掉了大量的共享内存指令。
## Scheduler Statistics  
![alt text](assets/image-16.png)  
![alt text](assets/image-21.png)  
调度器 no eligible 增加 22%
## Warp Statistics  
![alt text](assets/image-22.png)  
这一版的 stall reason 由之前的 barrier 变成了 long scoreboard，因为现在 shuffle 指令执行起来更快，从而将 long scoreboard 暴露出来了，也导致了每个周期就绪线程束减少，调度器空转增加。  
# VERSION 4: Warp per Row  
![alt text](assets/image-23.png)  
## Overview  
