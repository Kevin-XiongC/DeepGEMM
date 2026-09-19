# DeepGEMM Mega MoE 原生 NVFP4(W4A4)支持 — 开发交接文档

> 目标:为 vllm-project/DeepGEMM fork 的 Mega MoE 融合算子完成 **NVFP4(e2m1 + e4m3 SF@16 + per-tensor fp32 global scale)** 支持并验证通过。算子级验证用 pod 内真实 checkpoint `/data/models/RedHatAI/GLM-5.3-Flash-NVFP4` 的 expert 权重。
> **重要状态更新(2026-09-19):NVFP4 实现已完成 POD 编译和 4-rank handoff 验收;当前已完成算子级合成数据与真实 checkpoint expert 的独立 correctness,尚未完成 vLLM loader/完整模型端到端接线。**

---

## 1. 环境与访问

### 1.1 Pod 访问(唯一入口)

```bash
# 所有 pod 内命令都通过这条链路执行(kubectl 只在 master 117 上有)
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 2222 \
    ruizi@root@10.0.3.117@jumpcg.ppio.cloud \
    'kubectl -n ruizi-k3pd exec k3-dev-56db6fdb4c-zfmkf -- bash -lc "<命令>"'

# 推送文件到 pod(注意 exec -i,stdin 管道)
cat local_file | ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 2222 \
    ruizi@root@10.0.3.117@jumpcg.ppio.cloud \
    'kubectl -n ruizi-k3pd exec -i k3-dev-56db6fdb4c-zfmkf -- bash -c "cat > /root/DeepGEMM/<相对路径>"'

# 从 pod 拉文件
ssh ... 'kubectl -n ruizi-k3pd exec k3-dev-56db6fdb4c-zfmkf -- cat /root/DeepGEMM/<路径>' > local_file
```

嵌套引号:内层双引号需转义。复杂脚本建议先写本地文件再 push 执行。测试输出务必用 `python3 -u`(无缓冲),否则 timeout 杀进程后日志为空。

### 1.2 Pod 内关键路径与资源

| 项 | 值 |
|---|---|
| DeepGEMM 仓库 | `/root/DeepGEMM`,vllm-project/DeepGEMM fork,分支 `dev`(HEAD `a6bbb80`),**有未提交的 NVFP4 实现改动** |
| NVFP4 参考 checkpoint | `/data/models/RedHatAI/GLM-5.3-Flash-NVFP4`(hidden=4096, moe_inter=2048;expert 张量:`weight_packed`[N,K/2] U8 + `weight_scale`[N,K/16] e4m3 + `weight_global_scale`[1] fp32 + `input_global_scale`[1] fp32;MoE 层 3-44 为 nvfp4,第 45 层 fp8 block-wise) |
| MXFP4 模型(后续目标) | `/data/models/moonshotai/Kimi-K3`(896 experts×92 层,ue8m0@32) |
| GPU | 4× NVIDIA GB300(sm_103a,CUDA 13.0,torch 2.13.0+cu130) |
| 已装包 | `deep_gemm 2.8.0+local`(含 NVFP4 改动)、`deep_ep 2.0.0+local`、vllm 0.29.1rc1 dev |
| **POD 状态** | **2026-09-19 已清理外部 worker;检查时无 compute app,可运行 handoff 验收命令。** |

注意:pod 内只有 `python3`;在 `/root/DeepGEMM` 目录内 `import deep_gemm` 会被源码目录 shadow,测试时 `cd tests` 或其他目录。

### 1.3 本地工作副本

`/Users/ppio-dn-274/vllm-int/.work/DeepGEMM/` — 从 pod 拉取的源码(git init,baseline commit `42947d8` = 含 NVFP4 未提交改动的工作区快照),只有 `csrc/ deep_gemm/ tests/`(无 setup.py/third-party,不能本地编译,纯编辑用)。改动流程:本地编辑 → push 到 pod → pod 内编译测试。**注意 pod 上的仓库可能同时被其他人/AI 修改,push 前先在 pod 侧 `git -C /root/DeepGEMM status` 和 `git diff` 确认没有冲突。**

---

## 2. 现状

### 2.1 背景结论(调查阶段已确认)

- **硬件约束**:tcgen05 block-scaled MMA 没有 "fp8 激活 × nvfp4 权重" 组合;e4m3 SF@16 只有 `kind::mxf4nvf4.block_scale.block16`,强制 A/B 都是 e2m1。故 NVFP4 = **W4A4**(`fp4xfp4`),与 GLM-5.3-Flash-NVFP4 config 一致(其 input_activations 也是 4bit group16 e4m3 dynamic local)。用户已拍板走原生 W4A4。
- **CUTLASS SF atom**(`cutlass/detail/sm100_blockscaled_layout.hpp:48-59`):SF 存储布局与 SFVecSize 无关,基本块恒为 128行×4SF(1 uint32);vec32 时 1 uint32 覆盖 128 K 元素,vec16 覆盖 64 个 → gran16 适配 = SF 字数 ×2,布局结构不变。
- **mxf4nvf4 指令语义**:UMMA_K=64,每指令消费 4 个 SF(=1 uint32 word);instr desc SF 类型位 CUTLASS 已支持 `float_ue4m3_t`。
- **per-expert weight_global_scale**:不进 MMA,在 epilogue 乘 alpha(L1 输出 gate/up 在激活函数之前分别乘;L2 输出在 bf16 cast 之前乘)。参考 vLLM `vllm/models/deepseek_v4/nvidia/fi_moe.py:70-126` 的 fc1_alpha/fc2_alpha 与 gate/up global 折叠逻辑。

### 2.2 环境已就绪

- pod 已装 `libelf-dev elfutils libdw-dev`;编译需注入 pip nvidia 头路径:
  ```bash
  cd /root/DeepGEMM && \
  export CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include:$CPLUS_INCLUDE_PATH \
         LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LIBRARY_PATH && \
  python3 setup.py bdist_wheel && pip install dist/*.whl --force-reinstall
  ```
- device 代码(`.cuh`)是 JIT 的,改后**不用重编 wheel**,直接重跑测试;host 代码(`csrc/` 下 .hpp、python_api.cpp)变了要重编。

### 2.3 NVFP4 实现现状(pod 未提交 diff,477 insertions / 14 files)

`git -C /root/DeepGEMM status` 显示修改了:`csrc/apis/{layout,mega_moe}.hpp`、`csrc/jit_kernels/{heuristics/mega_moe,impls/sm100_fp8_fp4_mega_moe,impls/sm100_fp8_fp4_mega_moe_situ}.hpp`、`csrc/utils/layout.hpp`、`deep_gemm/include/deep_gemm/{common/math.cuh,impls/sm100_fp8_fp4_mega_moe.cuh,impls/sm100_fp8_fp4_mega_moe_situ.cuh,layout/mega_moe.cuh,ptx/tcgen05.cuh}`、`deep_gemm/mega/__init__.py`、`deep_gemm/utils/math.py`、`tests/test_mega_moe.py`。

已实现的要点(以 pod 上 `git diff` 为准):
- `math.py:128` `per_token_cast_to_nvfp4(x, gran_k=16, global_scale)` → (packed e2m1, e4m3 SF, fp32 global);
- kernel `sm100_fp8_fp4_mega_moe.cuh`:`kIsNVFP4` 模板分支(`:122-123` kGranK=16、`:152` a_dtype=e2m1、`:164` UMMA_K=64、`:844` ue4m3 SF desc、`:538` dispatch 字节数减半等);
- host:`mega_moe.hpp:214-225` is_nvfp4 → recipe (1,1,16)、mma_type `fp4xfp4`;heuristics `parse_mma_kind` 支持 `fp4xfp4`;
- 测试:`test_mega_moe.py` 已有 `--mma-type fp4xfp4` 全链路(`_cast_weights_to_nvfp4`、`_pack_nvfp4_sf`、recipe (1,1,16))。
- **注意**:host 侧 `mega_moe.hpp:215` 断言 NVFP4 只支持 swiglu(非 situ)。

### 2.4 已验证可用

单进程最小链路通过(2026-09-18):`import deep_gemm` ✓、`dist.init_process_group` ✓、`get_symm_buffer_for_mega_moe(mma_type='fp4xfp4')` ✓。

### 2.4.1 本轮真实尺寸适配进展(2026-09-18)

- 真实 GLM 尺寸 `hidden=4096, intermediate_hidden=2048` 原先在 JIT 编译阶段触发 `L1_SHAPE_K % BLOCK_K == 0` / `L2_SHAPE_K % BLOCK_K == 0`;当前 NVFP4 scheduler 允许 partial-K，保留非 NVFP4 路径的整除静态检查。
- `L2KBlockDependency` 的最后一个 K block 现在按剩余 L1-N block 数截断 readiness mask；`2048 = 768 + 768 + 512` 不会再等待不存在的 4 个 L1-N bits。
- 已确认 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` 对 NVFP4 的 packed FP4 TMA descriptor 会返回 `CUDA_ERROR_INVALID_VALUE`，因此没有把该模式保留在最终改动中；NVFP4 partial-K 的零填充仍需专门的 tail-copy/物理 padding 方案。
- POD 内已重编并安装 `deep_gemm-2.8.0+local`，真实 `4096/2048` 单进程 NVFP4 smoke 已完成 JIT 编译并打印 `Done, exiting`。
- 2026-09-19 已确认 POD 空载（GPU 显存约 3 MiB、无 compute app），handoff 默认 4-rank NVFP4 命令正常结束，约 `2.6 PFLOPS / 2.9 ms`。
- 2026-09-19 将最终本地 `sm100_fp8_fp4_mega_moe.cuh` 同步到 POD并校验 SHA256 一致；随后重新编译、安装 `deep_gemm-2.8.0+local`，从 `/tmp` 导入安装包成功。
- 发现并修复 routed NVFP4 partial-`BLOCK_M` 尾块：当 `valid_m < BLOCK_M` 时，TMA 的 2-CTA A tile 仍按物理 `BLOCK_M / 2` 分片，UMMA N 也保持完整 `BLOCK_M`；epilogue 继续按 `valid_m` 截断。此前 `valid_m / 2` 会使尾 token routed 结果少一段。
- 修复后，4-rank routed NVFP4 + shared FP8 的独立 quantized-operand reference 在 `tokens=32/64/128`（对应 partial/full block）均通过，`max_abs=0`；无 shared 的 `test_mega_moe.py --ncu-profile-only` 在 `tokens=32/64` 也正常结束。
- 已完成真实 checkpoint 的独立 quantized-operand correctness：`/data/models/RedHatAI/GLM-5.3-Flash-NVFP4` layer 3、expert 0–3、4 ranks、`hidden=4096/intermediate=2048`、topk=1，在 `tokens=32/64/128` 下均完成 reference 对比；各 rank `max_abs` 约 `6.1–8.1`，mean absolute error 约 `1.05–1.15`。`weight_global_scale` 按 divisor 语义以 `1 / weight_global_scale` 传入 `l1/l2 alpha`；`input_global_scale` 尚未接入动态激活量化。
- 尚未完成 vLLM loader 接线、完整 45-layer 推理，以及 shared expert 的真实 checkpoint 端到端校验。

### 2.5 已解除的 POD 阻塞 / 当前剩余工作

- 历史现象：外部 VLLM persistent worker 占满每卡显存时，persistent mega kernel 会因无法同时驻留而长时间停在 grid/barrier；该外部进程已由用户清理，当前 handoff 命令已恢复正常。
- 当前剩余工作集中在 vLLM loader/完整模型路径，而不是 Mega MoE 算子自身的 POD 编译或独立 expert correctness。

---

## 3. 验证与验收

```bash
# pod 内,测试目录
cd /root/DeepGEMM/tests
python3 -u test_mega_moe.py --num-processes 4 --num-experts 256 --activation swiglu                      # fp8xfp4 回归
python3 -u test_mega_moe.py --num-processes 4 --num-experts 256 --activation swiglu --mma-type fp4xfp4  # NVFP4
```

- [ ] fp8xfp4 / fp8xfp8 / bf16xbf16 基线全绿(回归)
- [x] routed NVFP4 + shared FP8 的独立 quantized-operand reference 通过(`tokens=32/64/128`, `max_abs=0`)
- [x] 用 GLM-5.3-Flash-NVFP4 真实 expert 权重(hidden=4096, inter=2048)完成独立 reference 对比；尚不代表 vLLM loader/E2E 已接通
- [x] POD 上完成 `fp4xfp4` 4-rank performance smoke(`~2.6 PFLOPS`, `~2.9 ms`)
- [ ] pod 侧改动尽快 `git commit`(pod 重建 /root 会丢;或 `git diff > /data/备份.patch`)

## 4. 明确不做(本期)

- vLLM 侧接线(`prepare_megamoe_inputs` nvfp4 激活量化、Kimi/GLM 模型接入、checkpoint 加载映射)。
- shared experts 保持 fp8/ue8m0 不动。
- situ 激活:fork 的 e49b29b 已支持(fp8xfp4),NVFP4+situ 被 host 断言挡掉(mega_moe.hpp:215),如需再开。
