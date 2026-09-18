# DeepGEMM Mega MoE 原生 NVFP4(W4A4)支持 — 开发交接文档

> 目标:为 vllm-project/DeepGEMM fork 的 Mega MoE 融合算子新增 **NVFP4(e2m1 + e4m3 SF@16 + per-tensor fp32 global scale)** 原生支持,算子级验证使用 pod 内真实 checkpoint `/data/models/RedHatAI/GLM-5.3-Flash-NVFP4` 的 expert 权重。
> 本文档是自包含的交接材料,读者不需要任何前置对话上下文。

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

嵌套引号:内层命令里的双引号需转义为 `\"`(在 bash -lc "..." 内)或 `\"`→`\\\"`(两层以上)。建议复杂脚本先写成本地文件再 push 执行。

### 1.2 Pod 内关键路径与资源

| 项 | 值 |
|---|---|
| DeepGEMM 仓库 | `/root/DeepGEMM`(vllm-project/DeepGEMM fork, baseline HEAD `a6bbb80`) |
| NVFP4 参考 checkpoint | `/data/models/RedHatAI/GLM-5.3-Flash-NVFP4`(hidden=4096, moe_inter=2048, 每 expert: `weight_packed`[N,K/2] U8 + `weight_scale`[N,K/16] e4m3 + `weight_global_scale`[1] fp32 + `input_global_scale`[1] fp32;MoE 层 3-44 为 nvfp4,第 45 层是 fp8 block-wise) |
| MXFP4 模型 | `/data/models/moonshotai/Kimi-K3`(896 experts×92 层, ue8m0@32,后续目标但不在本期) |
| GPU | 4× NVIDIA GB300(sm_103a,CUDA 13.0,torch 2.13.0+cu130) |
| 已装包 | `deep_gemm 2.8.0+a6bbb80`(已 pip 安装,见 §2.2)、`deep_ep 2.0.0+local`、vllm 0.29.1rc1 dev |
| 磁盘 | `/data` 余 7.2TB;RAM 954GB |

注意:pod 内 `python` 不存在,只有 `python3`;在 `/root/DeepGEMM` 目录内 `import deep_gemm` 会被源码目录 shadow,**测试时先 `cd` 到别处**(如 /tmp 或 tests 目录内用已安装包)。

### 1.3 本地工作副本

`/Users/ppio-dn-274/vllm-int/.work/DeepGEMM/` — 从 pod 拉取的源码(git init 过,baseline commit `42947d8`),只含 `csrc/ deep_gemm/ tests/` 三个目录(**没有** setup.py / third-party / CMakeLists,不能本地编译,纯编辑用)。改动流程:本地编辑 → push 到 pod → pod 内编译测试。

参考文件(已从 pod 拉出,只读):`/Users/ppio-dn-274/vllm-int/.work/ref/` 下有
`sm100_blockscaled_layout.hpp`、`mma_sm100_desc.hpp`(CUTLASS)、以及 kernel 各核心文件的副本(可能已被本地副本覆盖更新,以 `.work/DeepGEMM` 为准)。

---

## 2. 当前状态(已完成/进行中)

### 2.1 已完成的调查结论

**硬件约束(最重要)**:PTX tcgen05 block-scaled MMA 不存在 "fp8 激活 × nvfp4 权重" 组合:
- 现有 kernel 用 `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale`,SF **只接受 UE8M0@32**;
- e4m3 SF@16 只有 `kind::mxf4nvf4.block_scale.block16`,**强制 A/B 两侧都是 e2m1**。
- 所以原生 NVFP4 = **W4A4**,激活链路(dispatch 缓冲区、L1 epilogue 量化)也必须 fp4 化。GLM-5.3-Flash-NVFP4 确实是 W4A4(config 里 input_activations 也是 4bit group16 e4m3 dynamic local),语义一致。
- 用户已拍板走**原生 W4A4** 路线(否掉了"加载时转 MXFP4"的捷径)。

**CUTLASS SF atom 关键事实**(`cutlass/detail/sm100_blockscaled_layout.hpp:48-59`):SF 存储 atom 是 `(32,4)mn × (SFVec,4)k`,stride `((16,4),(0,1))` —— **存储布局与 SFVecSize 无关**,每个 128行×4SF(1 uint32)是基本块。vec32 时 1 uint32 覆盖 128 个 K 元素,vec16 时覆盖 64 个。即 gran16 适配 = 所有 `K/128`(=gran_k×4)处变 `K/64`,SF 字数为 2 倍,布局结构不变。

**指令语义**:mxf8f6f4 的 UMMA_K=32(每指令 1 个 SF@32);mxf4nvf4 的 UMMA_K=64(每指令 4 个 SF@16 = 恰好 1 个 uint32 word)。CUTLASS 参考 `cute/arch/mma_sm100_umma.hpp:1381-1398`(`SM100_MMA_MXF4_SS` VS=16 → `mxf4nvf4.block_scale.block16`);instr desc 的 SF dtype 位 `to_ScaleFormat<float_ue4m3_t>` 已存在于 `mma_sm100_desc.hpp:216-234`。

**per-expert global scale 处理**:不进 MMA,在 epilogue 乘 alpha(L1 输出 gate/up 各乘各的 global,**在激活函数之前**;L2 输出乘 down 的 global,在 bf16 cast 之前)。GEMM1 输出列一半是 gate 一半是 up(kernel 已做 gran-8 交织,`TMEM_LOAD_16dp256b1x` 读出的 float2 恰好是 (gate,up) 对),两个分量要分别乘 g_gate、g_up;若 checkpoint 两者不等,要么 host 侧折叠进 e4m3 SF(仅当比值在 e4m3 精确可表,参考 vLLM `fi_moe.py:103-116`),要么传两个 alpha 数组。vLLM  fused w13 要求 w1/w3 global 相等(compressed_tensors_moe_w4a4_nvfp4.py:205-212),可按此假设 + host assert。

### 2.2 已完成的环境工作

1. pod 内 `apt-get install -y libelf-dev elfutils libdw-dev`(deep_jit 需要 libdwfl.h)。
2. 编译需注入 pip nvidia 包头/库路径(系统 CUDA 缺 cusparse 等):
   ```bash
   cd /root/DeepGEMM && \
   export CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include:$CPLUS_INCLUDE_PATH \
          LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LIBRARY_PATH && \
   python3 setup.py bdist_wheel && pip install dist/*.whl --force-reinstall
   ```
   已成功安装 `deep_gemm-2.8.0+a6bbb80`。
3. 基线测试已启动(写本文档时仍在 JIT 编译/运行):
   `cd /root/DeepGEMM/tests && python3 test_mega_moe.py --num-processes 4 --num-experts 256 --activation swiglu`,日志 `/tmp/mega_baseline.log`。**接手后先确认它通过**(tail 日志,成功会有 `Passed` 类似输出和性能数字)。

---

## 3. Kernel 结构地图(sm100_fp8_fp4_mega_moe.cuh,1495 行,已全文精读)

单 kernel 融合 dispatch→GEMM1→激活→量化→GEMM2→combine,5 类 warp 角色:

| Warp 角色 | 行号 | 职责 | NVFP4 改动点 |
|---|---|---|---|
| dispatch warps | 335-681 | topk 计数、跨 rank NVLink pull 激活到 ring buffer | `:530-531` kNumChunks=kHidden/kNumBytesPerPull(fp4 后每 token 字节数减半);`:577` `kNumSFUint32 = kHidden/128` → `kHidden/64`;SF 拷贝 `:587-592` |
| TMA acts warp | 682-747 | A(激活)+SFA 的 TMA load | `:727` `sfa_k_idx = k_block_idx * (BLOCK_K/128)` → `/64` |
| TMA weights warp | 748-808 | B(权重)+SFB 的 TMA load | `:766,778` 同粒度问题;`:798-799` fp4 权重字节数已有 `kIsWeightFP8` 分支可参考 |
| MMA warp | 809-928 | UTCCP + UMMA 发射 | instr desc `:817-824` sf_type→`float_ue4m3_t`;UTCCP `:883-897`(每 umma_k_block 的 SF word 数 ×2);UMMA 内层 `:900-912`(UMMA_K 32→64,`make_runtime_instr_desc_with_sf_id` 步进重推);新增 `SM100_MMA_MXF4NVF4_2x1SM_SS` PTX wrapper |
| epilogue warps | 929-1488 | L1:TMEM 读→SwiGLU/SiTU→**fp8 量化+ue8m0 SF**→TMA 存;L2:TMEM 读→bf16→NVLink 写 combine | L1 量化 `:1118-1176` 改 fp4 e2m1 打包 + e4m3 SF@16(STSM `__nv_fp8x4_e4m3`→fp4x8 打包;SF 写 `:1151-1172` 粒度变);L1 alpha 注入点 `:1088`(gate/up 分量分别乘);L2 alpha 注入点 `:1277-1281`(cast 前乘) |

关键常量:`kGranK=32`(`:135`)、`UMMA_BLOCK_K=128`、`UMMA_K=32`(`:159-160`)、smem SF 尺寸 `:198-199`(`BLOCK_K/128` uint32 → `/64`)、TMEM SF 列 `:223-227`。
另有 situ 变体 `sm100_fp8_fp4_mega_moe_situ.cuh`,同样位置同样改(或抽公共模板参数)。

### Host/JIT 侧

- `csrc/apis/mega_moe.hpp`:`fp8_fp4_mega_moe`(`:156-305`)。recipe assert `:180` 放宽 (1,1,16);SF 检查 `:214-218`(e4m3 packed 分支);buffer size 计算走 `get_symm_buffer_size_for_mega_moe`(`:33-154`,mma_type 决定 `with_sf` 和各 buffer 尺寸——新增 `'fp4xfp4'`,x 每 token hidden/2 字节、x_sf hidden/16 字节);新增 `l1_alphas`/`l2_alphas` tensor 参数透传。
- `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`:`kGranK=32`(`:52`)、8 处 `make_tma_sf_desc`、JIT 模板实例化字符串(`:166-211`,加模板参数)。situ 版同理。
- `csrc/jit_kernels/heuristics/mega_moe.hpp`:`parse_mma_kind`(`:65-78`)加 `fp4xfp4`;`:162-163` smem SF 尺寸、`:204` gran_k 参数化。
- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh:387-394`:`num_mma_elem_bytes` 和 `*/32` 的 SF 尺寸按 mma kind 参数化(给 `MegaMoEBuffer` 加参)。
- `deep_gemm/mega/__init__.py`:Python wrapper 加 `mma_type='fp4xfp4'` 支持 + alpha 张量参数。

### 工具与测试

- `deep_gemm/utils/math.py`:新增 `per_token_cast_to_nvfp4(x, gran_k=16)` → (packed e2m1 int8, e4m3 SF [M, N/16], fp32 global)。参考现有 `per_token_cast_to_fp4`(`:99-125`,只支持 ue8m0)和 `_quantize_to_fp4_e2m1`。e4m3 SF 计算:`sf = amax/6/global`,round-to-nearest 到 e4m3(注意不是 ceil_to_ue8m0;可参考 torch `.to(torch.float8_e4m3fn)`)。
- `csrc/apis/layout.hpp:11-58` `transform_sf_into_required_layout`:新增 (e4m3, gran_mn=1, gran_k=16) 分支 → MN-major packed uint32(4 e4m3/int32),输出形状 `[mn, k/64]`;现有分支只接受 gran_k∈{32,128}。
- `deep_gemm/mega/__init__.py` 的 `_transpose_sf_for_utccp`/`_interleave_weights`:**不用改**(MN 维转置,与 gran_k 无关)。
- `tests/test_mega_moe.py`:新增 `--mma-type fp4xfp4` 用例。参考链不能用 `m_grouped_fp8_fp4_gemm_nt_contiguous`(同样写死 ue8m0),用纯 PyTorch 反量化参考(w = e2m1×e4m3sf×global,激活动态量化),权重构造新增 `_cast_weights_to_nvfp4`。
- 仓库里已有的 e4m3 相关工具只有 `tests/utils.py:57-69` `to_cublaslt_vec16_sf_layout`(ue4m3 编码参考)和 `cublaslt_nvfp4_gemm_nt`(dense GEMM,可作小规模交叉验证)。

---

## 4. 开发流程(建议)

1. **小步快跑,先单测 MMA 数值**:不要一上来改 mega 全链路。先写最小验证——可以新增/改造一个非 mega 的 grouped GEMM(`sm100_fp8_fp4_gemm_1d1d.cuh`,同样写死 ue8m0)加 nvfp4 分支,验证 instr desc + SF 布局 + UTCCP/TMEM 字序正确,再回移植 mega。TMEM/UTCCP 在 vec16 下的字序是最大风险点。
2. **编辑→部署→编译→测试循环**:
   ```bash
   # 本地编辑 .work/DeepGEMM/<path> 后:
   cat .work/DeepGEMM/<path> | ssh -p 2222 ruizi@root@10.0.3.117@jumpcg.ppio.cloud \
     'kubectl -n ruizi-k3pd exec -i k3-dev-56db6fdb4c-zfmkf -- bash -c "cat > /root/DeepGEMM/<path>"'
   # device 代码(.cuh)是 JIT 的,不用重编 wheel,直接重跑测试即可;
   # host 代码(csrc/apis、jit_kernels)变了要重跑 §2.2 的 wheel 编译(约几分钟)。
   ```
3. **测试命令**(在 pod):
   ```bash
   cd /root/DeepGEMM/tests && python3 test_mega_moe.py --num-processes 4 --num-experts 256 --activation swiglu --mma-type fp4xfp4
   # 单进程调试: --local-rank-idx 0(配 cuda-gdb/compute-sanitizer)
   ```
4. **GLM 真实权重验证**:写脚本从 `/data/models/RedHatAI/GLM-5.3-Flash-NVFP4` 的 safetensors 读一层(`model.language_model.layers.3.mlp.experts.*` 的 `weight_packed/weight_scale/weight_global_scale`,hidden=4096、inter=2048),构造 mega moe 输入,对比 PyTorch 反量化参考。
5. **回归**:原有 `--mma-type fp8xfp4`/`fp8xfp8`/`bf16xbf16` 与 swiglu/situ 用例必须全绿(模板双实例化,不破坏旧路径)。
6. 本地 git 及时 commit(`.work/DeepGEMM`),pod 侧改动建议也定期 `git -C /root/DeepGEMM diff > patch` 备份到 /data(pod 重建会丢 /root)。

## 5. 验收标准

- [ ] `test_mega_moe.py --mma-type fp4xfp4` 合成数据通过(与 PyTorch 反量化参考误差口径对齐现有 fp8xfp4 用例)
- [ ] GLM-5.3-Flash-NVFP4 真实 expert 权重算子级 e2e 通过
- [ ] per-expert global scale 生效(构造 g≠1 的用例验证)
- [ ] 既有 mxfp4/fp8/bf16 路径回归全绿
- [ ] 性能 smoke:fp4xfp4 vs fp8xfp4 时延对比记录(dispatch 带宽应减半受益)

## 6. 明确不做(本期)

- vLLM 侧接线(`prepare_megamoe_inputs` 的 nvfp4 激活量化 Triton kernel、模型接入、checkpoint 加载映射)——后续任务。
- shared experts 保持 fp8/ue8m0 不动(独立 instr desc,与 routed 解耦)。
- situ 激活 fork 已支持(`mega_moe.hpp:43,181`),无需处理;但 situ 变体 kernel 的 nvfp4 实例化要与 swiglu 版同步改。
