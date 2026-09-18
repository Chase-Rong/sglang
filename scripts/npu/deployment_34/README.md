# 34 环境：integrated offload、MTP 与 DMA 修复快照

本目录记录截至 2026-09-18 已部署、已验证的代码配套脚本和精简证据。
后续容量建模与设计讨论以这个快照为基线；11/8 是当前配置的实测容量，
不能视为 NPU 的理论上限，也不能据此认定当前设计已达到最佳效率。

变更分为四部分：

1. 将 ZIP 中已存在、但原 Git 基线未跟踪的 standalone FSA、A5 attention
   源码和硬件测试辅助代码纳入版本管理。
2. integrated offload 允许异步 Prefill 与 MTP 共存；draft 图回放等待 Host
   visibility 事件，target/draft 共用带总预算及引用计数的 MemFabric 分配域。
3. Prefill DMA 根据实际 ACL 映射边界拆分异步拷贝，修复跨 Host 分片时的
   `507899`；不新增设备全局同步。
4. 保存部署脚本、回归方法及 ready 后容量证据。

## 验证快照

| 配置 | 提交/完成 | ready 后最大并发 | 最大 batch 走图 | 重算 |
| --- | --- | --- | --- | --- |
| 卸载、MTP3、262144 输入 + 1024 输出 | 12/12 | 11 | 是 | 0 |
| 不卸载、MTP3、262144 输入 + 1024 输出 | 9/9 | 8 | 是 | 0 |

除 Decode 日志外，还根据逐请求 Prefill 完成至生成结束的时间区间交叉核验并发。
超出的请求排队，不能把完成数等同于并发数。脚本使用 TP8、FP8 权重、BF16 KV、
128GiB 共享 Host 池（仅卸载）、`mem_fraction_static=0.85`、
`max_running_requests=64`、page128、chunk8192，关闭 radix cache。

`evidence/` 保存 CPU 回归、DMA 失败/通过、TP8 通过、两组 ready 容量和清卡记录。
本次提交前重新执行 191 项 CPU 回归和 C++ 边界测试，均通过；核心修改的15个
源码/测试/角色脚本与34服务器版本逐字节核对一致。入库的文本证据统一为LF换行。
历史文档中的完整日志路径保留原实验位置；大日志、编译库、模型、凭据和缓存不入库。
`integrated_mtp_result.md` 描述较早的 BS32 短上下文功能验证；最终长上下文配置以
`capacity256k_progress.md` 和两个专用启动脚本为准。
文档中提到的原始 `.patch` 文件保留在实验目录；仓库内以独立的 MTP/DMA Git
提交记录前后差异，不重复存放补丁文本。

## 重现方法

无需 NPU 的 C++ 边界测试，在仓库根目录执行：

```bash
c++ -std=c++17 -Wall -Wextra -Werror -I native/fsa \
  native/fsa/tests/dma_segments_test.cpp -o /tmp/dma_segments_test
/tmp/dma_segments_test
```

这些部署脚本保留实际使用的 34 服务器绝对路径。将本目录的 `.sh`、`.py`
文件复制到 `/home/j00628475/sglang_release_20260917_f524c79a/`，源码位于其
`sglang_remote/` 子目录。构建前准备 `native/fsa/README.md` 说明的源代码依赖，
并核对脚本中的 CANN、MemFabric、OPP 和 donor 路径。

```bash
bash build_dma_boundary_fix.sh
# 以下步骤会使用 NPU，应在卡空闲时运行。
bash verify_dma_boundary_fix.sh
python3 boundary_control.py start capacity256k_new_validation --role offload
python3 boundary_control.py status capacity256k_new_validation
# 等到服务健康后提交工作负载。
python3 boundary_control.py bench capacity256k_new_validation --count 12
python3 boundary_control.py status capacity256k_new_validation
# 等完整生成结束后停止本次服务。
python3 boundary_control.py stop capacity256k_new_validation
```

专用 offload 脚本默认引用 `fsa-binding-dma-segments/libsgl_fsa_extension.so`；
原生产库保留。通用 `fsa_pd_role.sh` 需调用方显式提供相同扩展及 MemFabric 环境。
两份专用脚本沿用历史文件名 graph32/graph4，实际图上限分别为 11/8。
