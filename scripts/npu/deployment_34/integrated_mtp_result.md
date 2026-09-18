Integrated offload MTP 修改记录

基线为 SHA256 `f524c79a9579e424b73a10bde8e5c6a080f7c549ef32725264dacea4a0e8627d` 的本地 ZIP，修改副本位于 `D:\Code\sglang-offloading\mtp_work_20260917\sglang_remote`，远端为 `141.61.55.34:/home/j00628475/sglang_release_20260917_f524c79a/sglang_remote`。

已处理的实际阻塞：

| 位置 | 修改 |
| --- | --- |
| `offload_config.py` | 分离 integrated 布局约束与异步 Prefill 拷贝约束，允许受支持的 MTP 链进入 integrated 模式 |
| `model_runner.py` | MTP 和 MTP0 均保留异步 H2D 预取、D2H 写回 |
| `eagle_worker_v2.py` | integrated draft 允许捕获 decode / draft-extend 图；两个直接图回放入口进入 runtime context，等待 Prefill Host visibility 事件 |
| `memfabric_host_kv.py`、`decode_host_kv.py` | target/draft 共用一次初始化的 MemFabric 分配域，统一 Host 总预算，独立组件命名和预算，引用计数控制最终释放 |
| 专用启动脚本 | 改用包含 MemFabric/DMA 算子的 production 扩展及其配套库路径 |

保留 fixed-width、top-k=1 的 MTP 链限制，以及独立 Prefill batch、CP1/DCP1 的布局限制。异步 Prefill 和 Decode MTP 共存：Prefill 结束时在计算流汇合各层 save_done 并完成 TP 发布，再记录 host_visible；后续 target 和 draft 的 eager/graph 入口等待该事件。该依赖使用设备流事件，无需新增 CPU 或设备全局同步。早期同步 Prefill 方案已被本方案替代。

启动配置仍为完整 decode 图，batch `1 2 4 8 16 32`，NEXTN、3 步、top-k 1、draft tokens 4。入口为：

```text
/home/j00628475/sglang_release_20260917_f524c79a/handoff_launch_offload_mtp_graph32.sh
```

新默认扩展：

```text
/home/j00628475/memfabric_a5_fd334486_20260917_v1/fsa-binding-production/libsgl_fsa_extension.so
SHA256 0db9ef1fbe67d3e8dd1a571b9d7fa18c02efd7e7786eed496a8f757c8589ffa7
```

静态检查确认 `memfabric_region`、`prefill_dma_copy`、`register_memfabric_region`、`unregister_memfabric_region` 的注册 schema 存在。配套包与库由同一 MemFabric 根目录提供，避免混用全局旧版本。

验证记录：

- `mtp_red.log`：先复现 3 个行为失败——MTP 被拒绝、MTP 错用异步 Prefill、draft 图被跳过。
- `memfabric_mtp_red.log`：复现共享域、预算及失败清理的阻塞。
- `async_mtp_cpu_final.log`：最终统一 CPU 回归 **187 passed**，涵盖准入、Prefill 流依赖、MemFabric 生命周期、MTP 容量预算、投机尾部同址复用、序列回滚、target/draft selection 隔离，并执行两个真实 MTP 分派方法以验证图回放门控及 select_index 转发。另有 1 条因禁用第三方 pytest 插件导致的 `asyncio_mode` 配置警告。
- 多卡预算错误在局部抛出前完成集体验证；SharedHostGroup 已标记失败时，部分 draft 分配仍能集体清理并归还预算，存活 target 不被销毁。两项均有先失败后通过的回归。
- 独立最终代码复核通过，无未解决的 Critical / Important 问题。
- 测试使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，并通过导入拦截器禁止加载真实 `torch_npu`。张量均使用 CPU，设备流与 MemFabric 外部接口由测试替身提供。
- 已有 missing-binary 测试原先会意外导入系统安装的 SGLang；测试现对扩展加载这一外部边界做隔离，继续验证生产函数返回明确的 missing-binary 错误。

原始远端文件保留 `.before-integrated-mtp` 备份；源码差异见 `integrated_mtp.patch`，最终文件校验见 `integrated_mtp_files.json`。

用户随后授权真实 NPU 验证。2026-09-17 23:38—23:59 在 34 的 8 张 Ascend 950 上完成以下验证（日志时间为服务器北京时间）：

| 验证 | 结果 |
| --- | --- |
| target verify 图捕获 | 8 个 rank 全部完成，batch 1/2/4/8/16/32 |
| draft decode / draft extend 图捕获 | 两类均完成，batch 最大 32 |
| 短输入重复 greedy 请求 | 2/2 成功，各生成 64 token，文本完全一致且正确回答 2+2=4；平均接受长度 3.0476，接受率 68.25% |
| 16K 分块 Prefill | 8192 + 8192 两段完成，随后生成 64 token，平均接受长度 4.0 |
| 32 并发 | 32/32 成功，各输入 256 token、输出 128 token |
| 实际 BS32 图回放 | 23:58:22 日志明确 `#running-req: 32`、`npu graph: True`，当时平均接受长度 3.89、接受率 0.96 |
| 汇总 | 35/35 请求成功，生成 4288 token，num_retractions 合计 0 |

实际 `/server_info` 将 NEXTN 规范化为 EAGLE，steps=3、draft tokens=4、graph batch=[1,2,4,8,16,32]。原始数据位于本目录 `hardware_logs/async_mtp_hardware_v1/`，远端同名运行目录位于 release 根目录内。

首次健康检查曾超时；py-spy 确认 scheduler 正在 KDA Prefill 的 Triton autotune/JIT 编译，detokenizer 等待输出。编译完成后健康检查、metrics、请求均正常，无需绕过健康检查。记录中的初始化与请求时延包含首次编译，不能作为稳态性能对比。上述为功能验证，未进行完整模型精度评测。
