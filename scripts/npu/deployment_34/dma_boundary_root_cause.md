# integrated offload DMA 507899 根因与修复

当前状态：修复已部署并通过最小硬件复现、生产DMA回归和完整256K+1K/MTP3验证。卸载ready后最大同时Decode为11且graph=True，12/12请求全部完成，零重算，无DMA/OOM。

## 根因证据

`capacity256k_offload_graph11_12` 已完成三组MTP相关图捕获，但长序列Prefill仍返回 `Prefill DMA submission failed: 507899`。驱动plog记录：

```text
halMemGetSeg: The va prop is not match.
va=0x3407ffd20000; tmp_va=0x340800000000; size=0x800000
Get segment failed. size=8388608; ret=3
rtMemcpyAsync ... ErrCode=507899
```

MemFabric `acc_offload_shared_dram_entry.cpp` 的 `HOST_MEM_SLICE_SIZE=32ULL*GB`，128GiB共享Host池由四个独立映射构成。它们在虚拟地址上相邻，但驱动不允许该Host DMA一次跨越分片边界。

旧 `native/fsa/prefill_dma.cpp` 为每个连续token run提交一次完整 `aclrtMemcpyAsync`，没有检查映射边界。graph8与graph11均出现同一错误。

## 不加载模型的最小硬件复现

远端release根目录 `dma_probe_red.log`，2026-09-18 10:32:57。SHARED pool=33GiB，在32GiB边界前4MiB处申请16MiB组件；设备tensor仅16MiB。

实际输出：

```text
host/dva = 0x3407ffc00000
query 0x3407ffc00000 -> base 0x340000000000, bytes 34359738368, return 0
query 0x340800000000 -> base 0x340800000000, bytes 1073741824, return 0
query 0x340800400000 -> base 0x340800000000, bytes 1073741824, return 0
BEGIN within
PASS within
BEGIN crossing
Prefill DMA submission failed: 507899
```

这将错误与模型容量、MTP调度、并发数量分离，并证明 `aclrtMemGetAddressRange` 可用于取得真实边界。退出时的double-free发生在首个异步错误之后，不作为容量判断依据。

## 修复方式

本地源码：`mtp_work_20260917/sglang_remote/native/fsa/`，也已同步到工作区主目录 `sglang_remote/native/fsa/`。差异包：`dma_boundary_fix.patch`；修改前后哈希：`dma_boundary_source_hashes.json`。

- `external_host_mapping.cpp/.h`：首次DMA时查询每个已注册allocation覆盖的实际ACL映射范围，将相对终点缓存在注册记录中；注销时一起移除。后续chunk与graph replay不重复查询。
- `dma_segments.h`：使用64位offset/bytes，将连续run拆成不跨映射边界的若干段；检查范围和溢出。
- `prefill_dma.cpp`：H2D/D2H使用相同分段；继续在原current NPU stream上提交 `aclrtMemcpyAsync`，保留recordStream和tensor引用。

没有硬编码32GiB分片大小，没有引入同步Memcpy或NPU synchronize，也没有修改异步Prefill、MTP参数和graph事件依赖。

新库独立构建到：

```text
/home/j00628475/sglang_release_20260917_f524c79a/fsa-binding-dma-segments/libsgl_fsa_extension.so
SHA256: 3eaf2d5a6afe7e1ad492c454688b99cb201f04ba5dadeb1ba57bf7c7e018519e
```

用 `SGLANG_NPU_FSA_EXTENSION` 切换验证，旧库保留。

## 验证标准

已通过C++ CPU分段测试：分片内、跨界、边界对齐、组件非零偏移、多段、超过32位范围、非法范围/溢出，以及10000组随机范围的完整覆盖与不跨界约束。

`dma_boundary_probe.py` 覆盖非均匀数据逐位校验、两侧边界、多run、两个stream事件多代和修改输入后的graph replay。`verify_dma_boundary_fix.sh` 串行运行CPU descriptor、该复现、LOCAL及TP8生产DMA；任何一步失败即停止。

实际回归：CPU descriptor 8 passed；LOCAL 1 passed；TP8各rank 1 passed。日志见 `hardware_logs/dma_boundary_fix/`。TP8首次torchrun使用standalone时，容器主机名解析失败；改为显式127.0.0.1 rendezvous后通过，该问题未涉及DMA或模型容量。

最终卸载batch依据：请求完整256K上下文已建立且Host写回可见后，实际Decode batch及图回放通过，并完整生成1K；同时检查MTP接受、零重算、无DMA/OOM。故障中间态的8/9不构成最大容量。

完整实测 `capacity256k_offload_dmafix_graph11_12`：12/12完成，耗时689.01秒，最大完整上下文Decode11且graph=True，6条batch11图回放日志，第12请求排队；每请求输入262144、输出1024、零重算，MTP接受长度3.303～3.765。逐请求Prefill完成到生成结束的重叠区间也得到同时11。新pool=3117696，而12个完整输入已需3145728 tokens，因此当前脚本预算无法同时容纳12。

对照resident：9/9完成，最大ready并发8且graph=True，额外请求排队，零重算。结果仅针对当前TP8、MTP3、256K+1K及既定内存/请求状态预算。

启动脚本已设offload图档位 `[1,2,4,8,11]`、max11，默认引用修复库；resident保持 `[1,2,4,8]`、max8。完整证据见 `hardware_logs/capacity256k_offload_dmafix_graph11_12/` 和 `capacity256k_progress.md`。

验证服务已停止，确认8/8 NPU无运行进程。源码和脚本本地/远端一致性记录见 `dma_delivery_verified.json`。
