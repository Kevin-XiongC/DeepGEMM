# vLLM 侧原生 NVFP4 DeepGEMM MegaMoE 适配交接

> 更新时间：2026-09-19
>
> 目标：在 vLLM 中让 GLM-5.3-Flash-NVFP4 checkpoint 走 DeepGEMM 原生
> `fp4xfp4` MegaMoE 路径。DeepGEMM 算子侧已经完成；本 handoff 只覆盖
> vLLM loader、activation staging、模型接线和端到端验证。

## 1. 当前状态

### 1.1 DeepGEMM 算子侧已完成

DeepGEMM 改动已提交并推到个人 fork：

- Fork：`git@github.com:Kevin-XiongC/DeepGEMM.git`
- 分支：`nvfp4-mega-moe`
- Commit：`cf153bca608284ff619edf4f7d65be9abe442df9`
- 本地工作副本：`/Users/ppio-dn-274/vllm-int/.work/DeepGEMM`

当前 DeepGEMM 支持：

- routed expert 原生 NVFP4 W4A4：e2m1 权重、e2m1 activation、e4m3 SF、gran-K=16。
- API 仍叫 `fp8_fp4_mega_moe`，通过 `mma_type='fp4xfp4'` 和
  `recipe=(1, 1, 16)` 选择 NVFP4 路径。
- `SymmBuffer` 已按 `mma_type='fp4xfp4'` 分配 packed-FP4 activation buffer。
- `l1_alphas` / `l2_alphas` 已支持 per-expert `weight_global_scale`。
- routed NVFP4 的 partial-K、partial-BLOCK-M、SF layout、TMA/MMA/JIT 已修复。
- shared expert 保持现有 FP8/UE8M0 路径，不走 NVFP4。
- NVFP4 仅支持 `swiglu`；host 侧暂时拒绝 NVFP4 `situ`。

POD 验证环境：

- Pod：`k3-dev-56db6fdb4c-zfmkf`
- Namespace：`ruizi-k3pd`
- Repo：`/root/DeepGEMM`
- GPU：4× GB300 / SM103a / CUDA 13
- Handoff 合成数据 4-rank smoke：约 `2.6 PFLOPS`、`2.9 ms`，正常退出。
- 真实 checkpoint 独立 expert correctness：layer 3、expert 0–3、
  `hidden=4096`、`intermediate=2048`、tokens `32/64/128` 已完成。

### 1.2 vLLM 当前代码状态

vLLM 工作副本：

- Repo：`/Users/ppio-dn-274/vllm-int`
- Branch：`glm_flash_production`
- 当前 HEAD：`f547c23ec9`
- 工作树有大量其他未提交文件，**不要执行 `git reset --hard` 或 `git clean`**。
- 建议另建 worktree 或只修改本任务明确列出的文件。

已有可复用代码：

- `vllm/models/deepseek_v4/nvidia/model.py`
  - 已有 DeepGEMM MegaMoE 的 weight transform、symmetric buffer、forward 和 loader 参考。
- `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py`
  - 已有 FP8/UE8M0 activation staging 和 top-k layout 重排参考。
- `vllm/models/kimi_k3/nvidia/model.py`
  - 已有另一套 DeepGEMM MegaMoE 接线和 expert mapping 参考。
- `vllm/models/glm5next/nvidia/model.py`
  - GLM-5.3-Flash 当前入口；`Glm5NextMoE` 使用通用 `FusedMoEFactory`，shared expert 是独立的 `Glm5NextMLP`。
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
  - 已有通用 NVFP4 backend 选择和 kernel format 工具，但目标是现有 FlashInfer/CUTLASS 等 MoE backend，不能直接当作 DeepGEMM MegaMoE 实现。
- `vllm/model_executor/layers/quantization/online/nvfp4.py`
  - 已有在线 NVFP4 weight quantization 参考；其 activation/kernel contract 需要和 DeepGEMM `fp4xfp4` contract 区分。

## 2. 目标 checkpoint 和格式

Checkpoint：

```text
/data/models/RedHatAI/GLM-5.3-Flash-NVFP4
```

模型关键尺寸：

- `hidden_size=4096`
- `moe_intermediate_size=2048`
- MoE layer 3–44：NVFP4
- 最后一层：FP8 block-wise，不能强行套 NVFP4

NVFP4 expert tensor 语义：

- `weight_packed`: U8，packed e2m1，最后一维约为原始 K 的 `K/2`。
- `weight_scale`: e4m3 per-group scale，group-K=16，原始逻辑形状约为 `[N, K/16]`。
- `weight_global_scale`: per-expert FP32 scalar；当前 checkpoint 语义是 **divisor**。
- `input_global_scale`: per-expert 或 per-layer FP32 scalar；需要在 vLLM 接线时确认其和 activation quantizer 的契约，不能直接假设等于 weight scale。

独立 correctness 已验证的 global-scale 用法：

```text
l1_alpha = 1 / weight_global_scale_gate_or_up
l2_alpha = 1 / weight_global_scale_down
```

DeepGEMM kernel 读取方式：

- `l1_alphas`：每个 local expert 两个值，布局为
  `[expert0_gate, expert0_up, expert1_gate, expert1_up, ...]`。
- `l2_alphas`：每个 local expert 一个值。
- alpha 在 DeepGEMM epilogue 中应用；不要提前把 global scale 折叠进 BF16/FP16 权重后再量化，避免额外舍入。

## 3. 建议实现路径

### 3.1 先不要改通用 NVFP4 backend

目标是 DeepGEMM MegaMoE 的 EP fused path，不是给所有 NVFP4 MoE backend 增加一个别名。

推荐以 `DeepseekV4MegaMoEExperts` / Kimi MegaMoE experts 为结构参考，给
`Glm5NextMoE` 增加一个 DeepGEMM MegaMoE 专用 expert/module 路径。原因是当前
`Glm5NextMoE` 的 `FusedMoEFactory` 是通用 grouped-MoE 抽象，而 DeepGEMM MegaMoE
需要同时管理：

- EP symmetric buffer；
- routed dispatch/combine；
- shared expert 的 fused FP8 权重；
- packed-FP4 activation buffer；
- l1/l2 per-expert global-scale alpha；
- MegaMoE 专用 token/top-k layout。

如果最终选择扩展 `FusedMoEFactory`，必须保证非 DeepGEMM backend 和现有 GLM5 路径不变。

### 3.2 activation staging

现有 `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py` 写死了：

- activation values：FP8 e4m3；
- activation SF：UE8M0、group-K=32；
- shared activation SF：DeepGEMM FP8 shared layout。

需要新增 NVFP4 分支，目标输出为：

- routed `x`：packed e2m1，`uint8`/`int8` view，逻辑 shape `[M, K/2]`；
- routed `x_sf`：e4m3 SF，group-K=16，按 DeepGEMM 需要 pack 成 int32 layout，逻辑上每个 int32 包含 4 个 e4m3 SF；
- shared activation：继续按 FP8/UE8M0 layout 生成，不要复用 routed NVFP4 SF；
- `topk_idx` / `topk_weights`：沿用当前 MegaMoE layout 和 padding 处理。

可以参考 vLLM 已有 `scaled_fp4_quant` 和
`vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`，但必须核对：

- e4m3 SF 是否是 group-K=16；
- 输出 SF 是否已经是 DeepGEMM 的 MN-major/TMA packed layout；
- global scale 是否被乘入 activation，还是作为独立 scalar 保留；
- Triton kernel 是否支持 packed-FP4 的半字节/byte addressing。

不要直接把现有 FP8 `GROUP_K=32` kernel 改一个常量就提交；NVFP4 的 values、SF 字节数和 TMA layout 都不同。

### 3.3 weight loader 和 transform

建议先在 `Glm5NextMoE` 里注册明确的 routed expert 参数：

- w13/gate-up packed values；
- w13/gate-up e4m3 SF；
- w13 gate/up global scales；
- w2/down packed values；
- w2 e4m3 SF；
- w2 global scale。

loader 需要覆盖本地 EP expert ownership、TP/EP shard、EPLB physical expert 映射，并处理
checkpoint 中 gate/up/down 的原始命名。不要只修改通用
`fused_moe_make_expert_params_mapping`，先确认 GLM-5.3 checkpoint 的真实 key。

权重进入 DeepGEMM 前的逻辑顺序：

1. 加载 packed values 和原始 e4m3 `weight_scale`。
2. 将 e4m3 SF 转为 DeepGEMM 要求的 int32 MN-major/TMA layout。
3. 对 routed L1 gate/up 做 gate/up interleave。
4. 对 L1/L2 SF 调用 `deep_gemm.transform_weights_for_mega_moe` 的对应 transform。
5. 单独构造 `l1_alphas` / `l2_alphas`，传 reciprocal global scale。
6. shared expert 如果走 fused path，继续使用 FP8/UE8M0 transform；不要把 shared 权重误当 NVFP4。

`deep_gemm.transform_weights_for_mega_moe` 对 tuple 权重的行为：

- L1：interleave gate/up values 和 SF，然后转置 SF 为 UTCCP layout；
- L2：values 不 interleave，只转置 SF。

### 3.4 forward 和 buffer

创建 symmetric buffer 时必须显式使用：

```python
deep_gemm.get_symm_buffer_for_mega_moe(
    group,
    num_experts,
    num_max_tokens_per_rank,
    top_k,
    hidden_size,
    intermediate_size,
    num_shared_experts=num_shared_experts,
    mma_type="fp4xfp4",
    activation="swiglu",
)
```

forward 调用仍然是：

```python
deep_gemm.fp8_fp4_mega_moe(
    y,
    transformed_l1_weights,
    transformed_l2_weights,
    symm_buffer,
    shared_l1_weights=transformed_shared_l1_weights,
    shared_l2_weights=transformed_shared_l2_weights,
    recipe=(1, 1, 16),
    activation="swiglu",
    l1_alphas=l1_alphas,
    l2_alphas=l2_alphas,
)
```

注意：函数名虽然是 `fp8_fp4_mega_moe`，但 `symm_buffer.mma_type == "fp4xfp4"`
时会进入 NVFP4 host/device dispatch；不要为此另造一个 Python API 名称。

需要检查并补齐 vLLM 的 DeepGEMM compatibility wrapper：

- `vllm/utils/deep_gemm.py` 当前主要暴露 grouped GEMM API；
- DeepSeek V4/Kimi 路径当前直接对 `_import_deep_gemm()` 返回对象调用 MegaMoE API；
- 新 GLM 路径应统一走 wrapper 或显式记录为什么必须直接调用，避免不同模型对 package version 的判断不一致。

## 4. 推荐修改文件

优先检查/修改：

```text
vllm/models/glm5next/nvidia/model.py
vllm/models/glm5next/nvidia/mtp.py
vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py
vllm/utils/deep_gemm.py
```

可能需要新增：

```text
vllm/models/glm5next/nvidia/ops/prepare_megamoe.py
vllm/models/glm5next/nvidia/mega_moe.py
```

测试建议放在：

```text
tests/models/test_glm5next_mega_moe.py
```

参考现有测试：

```text
tests/models/test_deepseek_v4_mega_moe.py
tests/models/kimi_k3/test_weight_loading.py
```

## 5. 验证顺序

### 5.1 不依赖完整模型的 unit test

先验证：

1. checkpoint key 到 GLM 参数的 mapping；
2. 一个 expert 的 packed values、e4m3 SF、global scale loader；
3. gate/up interleave 和 SF transform；
4. `l1_alphas/l2_alphas` 的 reciprocal 语义；
5. activation staging 输出与 DeepGEMM standalone reference 一致；
6. partial token block，例如 `M=32/64/128`，以及 `K=4096`、`N=2048`。

### 5.2 算子级 vLLM 集成测试

先只跑 layer 3、固定 expert、topk=1：

- 4 ranks，每 rank 一个本地 expert；
- `hidden=4096`、`intermediate=2048`；
- tokens `32/64/128`；
- 对比当前 POD 上 `/tmp/check_glm_nvfp4_all.py` 使用的 quantized-operand reference；
- 误差口径先保持原验证结果：mean absolute error 约 `1.05–1.15`，max absolute error 约 `6.1–8.1`。

### 5.3 完整模型

确认以下配置后再跑完整 GLM：

- GPU 是 SM103/GB300；
- `--enable-expert-parallel`；
- MoE backend 选择 `deep_gemm_mega_moe`；
- DeepGEMM package 来自 `Kevin-XiongC/DeepGEMM` 的 `nvfp4-mega-moe` 分支；
- layer 3–44 使用 NVFP4，最后 FP8 层仍走原有 FP8 路径；
- shared expert 不被错误地 quantize 成 routed NVFP4；
- CUDA graph、EPLB、sequence parallel、speculative/MTP 暂时先关闭，待基础 correctness 通过后逐项打开。

## 6. 常见坑

- **不要把 `weight_global_scale` 当乘数**：checkpoint 中它是 divisor，DeepGEMM alpha 使用 reciprocal。
- **不要把 `input_global_scale` 直接当成当前 DeepGEMM activation global scale**：独立 DeepGEMM correctness 使用动态 activation quantization，checkpoint input scale 尚未在该路径接入，需要先确认模型语义。
- **不要把 shared expert 按 routed NVFP4 处理**：当前 DeepGEMM 实现明确是 routed NVFP4 + shared FP8。
- **不要把 `fp8_fp4_mega_moe` 函数名当成只支持 FP8 activation**：真正的选择由 `mma_type='fp4xfp4'` 和 `recipe=(1,1,16)` 决定。
- **不要复用 group-K=32 的 FP8 SF layout**：NVFP4 使用 group-K=16，SF 字数和 packed values 字节数都不同。
- **不要在本地 vLLM dirty tree 上做全量清理**：该工作副本包含大量其他任务文件。

## 7. Definition of Done

- [ ] GLM-5.3-NVFP4 checkpoint key 完整加载，无未消费 expert weight。
- [ ] routed activation 输出为 DeepGEMM NVFP4 所需 packed e2m1 + e4m3 SF layout。
- [ ] `weight_global_scale` 通过 `l1_alphas/l2_alphas` 正确生效。
- [ ] shared expert 仍使用 FP8/UE8M0，且结果没有重复计算或漏加。
- [ ] layer-3 单层 4-rank correctness 通过。
- [ ] GLM 完整模型至少完成短 prompt prefill/decode correctness。
- [ ] NVFP4 layer 与最后 FP8 layer 混合执行正确。
- [ ] 基础路径通过后，再验证 CUDA graph、EPLB、MTP 和性能。
