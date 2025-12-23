# CUDA

## 初级修饰符和API

### demo1

```cuda
__global__ void hello_cuda(){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("[ %d ] hello cuda\n", idx);
}

int main(){
	hello_cuda<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}
// output
// [ 0 ] hello cuda
```

\_\_gloabl\_\_ : CUDA kernel函数前缀，该函数被CPU调用启动，在GPU上执行

blockDim.x和blockidx.x : block内线程数量和block的id

threadIdx.x : block内，线程的ID

<<<1, 1>>> : 启动CUDA kernel的标志，第一个代表分配block的数量，第二个代表block里的线程数量

cudaDeviceSynchronize() : 让cpu强制等待





### 修饰符

* \_\_gloabl\_\_ : CUDA kernel函数前缀，该函数被CPU调用启动，在GPU上执行
* \_\_host\_\_ : 在cpu端调用并且执行的函数，普通的C++代码。通常与\_\_device\_\_联用，代表代码需要编译成两份，cpu一份，gpu一份
* \_\_devce\_\_ : 在gpu端调用并且执行的函数，nvcc编译器汇决定这个是否inline
* \_\_noinline\_\_ ：强制编译器不要inline函数
* \_\_forceinline\_\_ : 强制编译器inline函数

### API

* cudaDeviceSynchronize() ：等待所有gpu设备上的函数结束



## CUDA程序的线程层次结构

### 硬件方面

Thread - GPU最基础的单位

Warp - GPU基础调度单位，包含32个thread

Block - 用户定义的线程组，硬件层面是多个warp

Grid - 一个或者多个Block

### SMIT

单指令，多线程的机制，是并行编程的基础。

| 特性维度      | SIMD                                                  | SIMT                                                         |
| ------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| 核心思想      | 一条指令同时处理多个数据元素                          | 一条指令同时驱动多个线程执行                                 |
| 编程视角      | 数据集并行；程序员需要关注数据打包，对齐              | 线程级并行；程序员编写标量线程代码，硬件管理并行             |
| 硬件基础      | 宽寄存器（128，256，512位）, 多个ALU单元              | 流式多处理器（SM）、大量CUDA核心、Warp调度器                 |
| 数据/线程组织 | 数据需在内存中**连续且对齐**                          | 线程被分组为**Warp**（通常32个线程为一组）                   |
| 分支处理      | 处理能力弱：所有通道执行相同指令，分支会带来性能损失  | 相对灵活：通过**Mask掩码**机制，允许Warp内线程执行不同分支路径 |
| 灵活性        | 较低，对数据结构和算法规整性要求高                    | 较高，更能适应不规则的数据结构和控制流                       |
| 典型应用      | CPU向量指令扩展（如SSE, AVX），用于图像处理、科学计算 | GPU计算核心架构（如NVIDIA CUDA），用于图形渲染、深度学习     |

#### 如何避免SMIT退化

* GPU如何处理控制流：GPU编译器中的控制流处理与传统CPU编译器有着本质区别，这主要源于其采用的**单指令多线程(SIMT)**。执行架构。在GPU中，线程以**warp**或**wavefront**为单位进行调度执行，每个线程束通常包含32或64个线程，这些线程必须保持同步执行相同的指令。当遇到分支结构时，GPU需要采用特殊机制来处理可能出现的线程执行路径分歧。

  在条件控制流（if-else, switch, loops等）存在的情况下，由于线程间存在逻辑差异，在分支bb中调度器可能只激活部分线程，其余线程 idle，这会导致原本期望的所有 32 个线程**并行**执行同一指令，变为每个分支单独串行执行，分支内部指令并行执行。由于有些线程为idle态，硬件利用率下降，这种现象的根本原因在于SIMT架构下，线程之间只共享指令流，不共享控制流。这也就是GPU在处理控制流的时候遇到的关键挑战。

* 如何处理Warp Divergence：SIMT执行特性：所有线程在同一周期执行相同指令，分支语句导致部分线程执行if块，其他执行else块，硬件必须串行化执行所有分支路径，考虑以下CUDA kernal函数：

  ```cuda
  __global__ void devergenceDemo(int *data){
  	if(data[threadIdx.x]  > 0){
  		// coda path 1
  	} else {
  		// coda path 2
  	}
  }
  ```

  当线程束内部分数据满足条件时，GPU会先执行路径A（禁用不满足条件的线程），再执行路径B（禁用已执行路径A的线程），最终导致实际执行时间翻倍。为了管理Warp Divergence发生的情况下的线程状态，使用Lane Mask/Execution Mask这个关键技术来标识某个线程是否处于激活态。

* Execution Mask 执行掩码：32位掩码对应线程束中的32个线程，每位表示对应线程是否活跃(1=活跃，0=禁用)，示例：0xF0表示前16个线程活跃。编译器生成的控制流指令包括：进入分支前保存当前掩码，进入分支时更新掩码，分支结束后恢复原掩码。GPU会通过硬件或者软件的方式维护一个Lane Mask/Execution Mask栈来实现掩码的动态管理。由于Warp Divergence发生导致硬件利用率低，所以提出控制流优化策略减少idle的线程数。

* 谓词优化 Predication：

  * 原始代码

    ```
    if (x > 0) y = sqrt(x);
    ```

    

  * 完全谓词化，使用场景是简单算术运算，分支体计算量小。

    ```
    float tmep = sqrt(x);
    y = (x > 0) ? temp : y;
    ```

  * 部分谓词化

    ```
    bool cond = x > 0;
    if (cond) { 
    	y = x; //保留分支
    }
    z = cond ? a : b; // 谓词化
    ```

  * 推测执行

    ```
    // 提前计算两个分支
    float tmp1 = sqrt(x);
    float tmp2 = log(x);
    // 最后选择
    y = (x > 0) ? tmp1 : tmp2;
    ```

* 分支重构技术

  * 分支重排序

    ```
    // 线程分布：threadIdx.x ∈ [0, 31]
    // 统计分析显示，约 75% 的线程满足 threadIdx.x >= 24
    // 即多数线程执行 commonPath()
    // 优化前：少数路径在前，主路径在后
    if (threadIdx.x < 24) {
        // Cold path — 执行线程较少
        processColdPath();
    } else {
        // Hot path — 执行线程较多，但放在后面，导致 warp divergence 代价更高
        processHotPath();
    }
    
    // 优化后：主路径（hot path）前置
    if (threadIdx.x >= 24) {
        // Hot path — 占多数线程，优先执行，提升 warp 活跃率
        processHotPath();
    } else {
        // Cold path — 占少数线程，延后执行，减少分歧代价
        processColdPath();
    }
    ```

  * 循环剥离

    ```
    // 处理前5个特殊元素
    for (int i = 0; i < N; i++) {
        if (i < 5) {
            // 特殊处理
        } else {
            // 常规处理
        }
    }
    
    // 优化后
    for (int i = 0; i < 5; i++) {...}
    for (int i = 5; i < N; i++) {...}
    ```

* GPU Lane Mask栈溢出：在一些复杂shader场景下，如果控制流优化不当，可能导致性能劣化。例如当GPU因为**lane mask栈溢出**或其他严重控制流问题（如深度嵌套分支）而**降级到逐线程执行模式**时，其行为会从高效的**SIMT（单指令多线程）**模式退化为类似CPU的**单线程/多线程分派**模式。

  | 特性         | 正常SIMT                               | 逐线程执行模型                   |
  | ------------ | -------------------------------------- | -------------------------------- |
  | 指令发射     | 1条指令同时发给整个warp（32/64线程）   | 每个线程独立发射指令             |
  | **执行效率** | 高（32线程共享1条指令的解码/执行开销） | 极低（每个线程需完整指令流水线） |
  | **分支处理** | 通过lane mask管理分化，硬件自动处理    | 每个线程独立判断分支路径         |
  | **同步开销** | warp内线程隐式同步                     | 需要显式同步（如原子操作/锁）    |
  | **性能表现** | 吞吐量高（理论峰值可达CPU的10-100倍）  | 接近低效CPU多线程（甚至更差）    |

  * 降级后的表现：每个GPU线程像CPU线程一样：1）**独立取指/解码**：失去SIMT的指令共享优势；2）**独立分支预测**：不再有warp级别的掩码控制。

    ```
    // 在降级模式下：
    // 线程0执行if (cond0) {...}
    // 线程1执行if (cond1) {...} 
    // 完全独立，无协同
    ```

    另外资源利用率暴跌，降级到逐个线程执行模式后会导致资源利用率暴跌，具体表现为：

    * **计算单元闲置**：原本可同时处理32线程的ALU，现在可能只服务1个活跃线程；
    * **寄存器压力激增**：每个线程需要完整保存独立上下文（而SIMT模式下可共享部分状态）。

  * 如何避免lane mask栈溢出

    * 控制分支深度

      ```
      // 错误：深层嵌套易触发降级
      if (a) {
          if (b) {
              if (c) { /* 超过硬件栈深度 */ }
          }
      }
      
      // 优化：扁平化分支
      if (a & b & c) { ... }
      ```

    * 使用谓词化替代分支

      ```
      // 原始分支
      if (x > 0) { y = sqrt(x); }
      
      // 优化：无分支
      y = (x > 0) * sqrt(x) + (x <= 0) * y;
      ```

### 软件硬件视觉差异

* 软件视角：软件视角看到的是一个线程组成的block.

* 硬件视角：以32个线程组成Warp为单位调度，执行任务在GPU的CUDA core, 一个warp对应32个CUDA core。block是软件概念，warp是有硬件对应概念。

  * Warp scheduler and dispatch port

    * CUDA 提供零成本的warp和线程切换，不像cpu一样有巨大的成本。线程的创建在一个时钟周期内完成。

    ```mermaid
    flowchart TD
        A[Warp Scheduler<br>每时钟周期选择Warp] --> B{Scoreboard检查}
        B -- "依赖未解除" --> C[Warp状态置为Stalled]
        B -- "依赖已解除<br>Warp标记为Eligible" --> D[Dispatch Port<br>指令解码与操作数收集]
        
        D --> E{执行端口仲裁}
        E -- "端口空闲" --> F[指令发射至执行单元<br>（如ALU, LSU, SFU）]
        E -- "资源冲突/端口繁忙" --> G[指令暂缓<br>Warp Scheduler尝试<br>调度其他Eligible Warp]
        
        F --> H[指令执行<br>（计算/内存访问）]
        H --> I[结果写回寄存器文件]
        I --> J[Scoreboard更新<br>解除相关依赖标记]
        
        J --> K[Warp状态恢复为Eligible]
        C --> K
        G --> A
        K --> A
    ```

    ##### 流程关键环节解读

    这个流程的核心在于通过**极细粒度的线程调度和快速的上下文切换**来隐藏内存访问等长延迟操作，从而最大化硬件利用率和整体吞吐量。

    - **Warp Scheduler（线程束调度器）**：它是SM（流式多处理器）的“指挥中心”。每个时钟周期，它都会检查所有已分配给当前SM的Warp（通常是32个线程为一组）的状态。其首要任务是从一个名为 **Scoreboard（记分板）** 的组件中，查询哪些Warp的所有执行依赖（如数据就绪、同步完成）已被满足。只有满足条件的Warp才会被标记为“Eligible”（就绪），并进入候选队列等待调度。
    - **Dispatch Port（发射端口）**：一旦Warp Scheduler选定了一个就绪的Warp，该Warp的下一条指令就会被送入Dispatch Port。这里主要负责两项工作： **指令解码与分类**：识别指令类型（如整数运算INT、浮点运算FP、内存加载/存储LOAD/STORE等）。 **操作数收集**：通过 **Operand Collector（操作数收集器）**，从庞大的寄存器文件中读取该Warp所有32个线程执行此次指令所需的寄存器值或地址。 完成后，指令会被送往对应的执行端口。如果目标执行端口繁忙或发生资源冲突，指令会被暂时阻挡，调度器则会立刻转向调度其他就绪的Warp。
    - **执行与写回**：指令在相应的执行单元（如CUDA Core、Load/Store Unit等）完成计算或内存访问后，结果会被写回寄存器文件。随后，Scoreboard中对该指令结果的依赖标记会被清除，意味着等待此结果的其他指令可能就绪，该Warp也重新变为“Eligible”，等待下一次调度。