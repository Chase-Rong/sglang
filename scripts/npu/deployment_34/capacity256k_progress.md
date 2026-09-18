# 256K + 1K / MTP3 验证状态（2026-09-18 更新）

用户要求：先定位并修复卸载报错，以卸载 ready 后的完整上下文 Decode 并发为依据，不能用故障中间态决定图 batch。

共同参数：34，TP8，FP8权重，BF16 KV，NEXTN MTP steps=3/topk=1/draft=4；mem_fraction_static=0.85，max_running_requests=64，page=128，chunk=8192，radix disabled。每请求输入262144、输出1024，ignore_eos=true。

| 运行 | 证据 |
| --- | --- |
| capacity256k_resident_8 | 8/8完成，完整上下文Decode8，eager，零重算，491.34秒。已停。 |
| capacity256k_resident_graph8_9 | 9/9完成，最大完整上下文Decode8且graph=True，第9请求排队，零重算，442.34秒。已停。 |
| capacity256k_offload_graph8_8 | 8/8完成，完整上下文Decode8且graph=True，零重算，428.68秒。已停。这是已通过点，不是容量上限。 |
| capacity256k_offload_12 | graph8，pool3117824，Prefill DMA 507899，未完成。已停。 |
| capacity256k_offload_graph11_12 | graph11三组图捕获成功，pool3117440，第10个Prefill附近同一DMA错误，未完成。已停。 |
| capacity256k_offload_graph9_9 | 仅加载服务，未提交benchmark；转入根因诊断前已按进程身份停止。 |
| capacity256k_offload_dmafix_graph11_12 | 修复库，pool3117696，12/12完成，完整上下文Decode11且graph=True，第12请求排队，零重算，无DMA/OOM，689.01秒。 |

MTP已由server_info及请求级spec_accept_length、spec_accepted_drafts证实。graph=False不代表关闭MTP。

当前固定配置下，卸载最大11、不卸载最大8均已实测确认。逐请求Prefill完成到生成结束的时间区间交叉核验，也分别得到11和8。两种模式各有6条最大batch的graph=True日志。

卸载pool=3117696，12个完整输入已需3145728 tokens，超过该pool；不卸载pool=2349312，9个完整输入需2359296 tokens，亦超过pool。超出的请求实际排队，未强制超量分配。结论适用于上述脚本预算，不代表改变内存预算或模型后仍是相同上限。

## DMA修复

详见 `dma_boundary_root_cause.md`。一次8MiB DMA跨过MemFabric Host映射分片边界，驱动halMemGetSeg拒绝。最小硬件复现只用单卡、33GiB Host pool和16MiB设备tensor：分片内拷贝通过，跨片复现507899。

修复动态查询并缓存ACL映射边界，按边界拆分双向异步DMA，保持原流与事件语义。补丁见 `dma_boundary_fix.patch`，新扩展位于release下 `fsa-binding-dma-segments/libsgl_fsa_extension.so`。

已通过C++ CPU边界/溢出和1万组随机测试、8个CPU descriptor测试、最小跨界复现的GREEN测试、多流多代事件、跨界graph replay、生产LOCAL和TP8共享DMA回归。

本轮SSH一度认证失败；用户确认后已恢复，不再阻塞。已确认8个调度进程实际映射新库；容量验证完成后已停止本任务服务，npu-smi确认8/8卡无运行进程，sglang进程已退出。

## 最终证据与配置

- `hardware_logs/capacity256k_offload_dmafix_graph11_12/boundary/ready_capacity_verified.json`：12/12、ready并发11、图并发11、零重算、MTP接受长度3.303～3.765。
- `hardware_logs/capacity256k_resident_graph8_9/boundary/ready_capacity_verified.json`：9/9、ready并发8、图并发8、零重算。
- offload脚本：graph `[1,2,4,8,11]`、max11，默认使用修复库；resident脚本：graph `[1,2,4,8]`、max8。历史文件名graph32/graph4保留。
- 通用角色脚本已同步integrated=11、resident=8，其他PD角色保持32。
