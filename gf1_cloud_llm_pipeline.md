# GF-1 云/非云分割 + 本地多模态 LLM（Qwen2-VL）全自动 Router + Judge 方案（无 AOI / 无人工）

> **说明**：当前代码实现已重构为“云 vs 非云”二分类，`shadow` 标签被视为非云。下文中涉及云影的段落为历史设计背景，仅作参考。

> 目标：在 **GF-1 WFV（4-band：B,G,R,NIR）** 场景下，实现“**分割网络产出 mask** → **结构化特征** → **LLM 路由器自动分流** → **全自动判别器（Judge）给最终 ACCEPT/REJECT**”，
> - **不使用 AOI**
> - **Judge 不依赖人工**
> - 通过“自动升级验证链路（ESCALATE）”替代人工复核
> - 全链路可回放、可审计、可稳定上线

---

## 1. 总览

### 1.1 设计原则
- **像素级分割**：由专门的分割网络完成（CloudSegNet）。
- **LLM 不做像素分割**：LLM 更适合做 **路由策略、解释、灰区核验**。
- **最终判定由可控融合器给出**：LLM 负责输出可审计的 `reasons(JSON)`，但 **不允许改阈值/规则**。
- **灰区处理**：不是进入人工，而是进入 **自动升级链路**（second-check + patch 多模态核验）。

### 1.2 模块清单
- **CloudSegNet**：云/云影分割网络（4-band 输入）。
- **Metrics Builder**：从概率图/掩码计算全景特征（无 AOI）。
- **Router**：LLM/轻量分类器，输出 `FAST_ACCEPT / FAST_REJECT / ESCALATE`。
- **Second Checker（可选但推荐）**：传统方法或第二个轻量网络，提供 second opinion。
- **Patch Sampler**：从不确定边界/分歧区域抽 K 个 patch，并生成 RGB / NIRRG / overlay。
- **Patch Verifier（MLLM，可选但推荐）**：Qwen2-VL 对 patch 做“云/云影/清晰/高亮混淆”核验。
- **Auto-Judge Fusion**：融合主分割特征 + second-check + patch 核验，输出最终 ACCEPT/REJECT。

---

## 2. 系统结构图（推理链路，无 AOI / 无人工）

```
GF-1 Scene (B,G,R,NIR)
   |
   v
[1] CloudSegNet 分割
   - 输出 P_cloud, P_shadow (概率图)
   |
   v
[2] Post & Metrics
   - 阈值化 -> mask_cloud/shadow
   - 形态学/连通域/不确定性
   - 生成全景 features JSON（无 AOI）
   |
   v
[3] Router（本地 Qwen2-VL，只吃文本JSON）
   - route = FAST_ACCEPT | FAST_REJECT | ESCALATE
   |
   +-------------------------+
   |                         |
   v                         v
[4] Second Checker(自动)     [5] Patch Sampler(自动)
   - second mask / prob         - 抽K个patch（不确定边界/分歧区/最大云团边缘）
   - agreement 指标             - 生成 RGB / NIRRG / overlay
   |                         |
   +-----------+-------------+
               v
[6] Auto-Judge Fusion（全自动）
   - 输出: ACCEPT / REJECT（可选 REJECT_SAFE）
   - LLM仅生成 reasons(JSON)，最终判定由融合器产出
```

---

## 3. CloudSegNet 网络结构（推荐）

### 3.1 输入与输出
- 输入：`H×W×4`（B,G,R,NIR）
- 输出：
  - `P_cloud(x,y) = sigmoid(logits_cloud)`
  - `P_shadow(x,y) = sigmoid(logits_shadow)`

> 使用 **双头二分类**（cloud head + shadow head），比三分类 softmax 更便于处理极端不平衡的云影，并能单独调阈值/权重。

### 3.2 结构图（SegFormer 风格）

```
Input: 512×512×4
  |
  v
Encoder: MiT (Hierarchical Transformer)
  F1: 1/4
  F2: 1/8
  F3: 1/16
  F4: 1/32
  |
  v
Decoder: SegFormer Head (MLP fusion)
  - project F1..F4 -> same dim
  - upsample -> 1/4 and fuse
  - conv + upsample -> full res
  |
  +-------------------+
  |                   |
  v                   v
Cloud Head         Shadow Head
logits_cloud       logits_shadow
```

### 3.3 推荐损失
- `Loss_cloud = BCEWithLogits + Dice`
- `Loss_shadow = w * (BCEWithLogits + Dice)`，`w=2~5`（视云影比例）
- 可选：`BoundaryLoss`（薄云边界/云影边界更稳）

总损失：
```
Loss = Loss_cloud + λ * Loss_shadow + μ * BoundaryLoss
```

---

## 4. 特征工程规范（无 AOI，全景特征）

> **Router/Judge** 的输入不直接是整景像素，而是 **mask 后的结构化特征**（可审计、可回放、吞吐高）。

### 4.1 基础掩码生成
- 云掩码：`mask_cloud = (P_cloud > t_c)`
- 云影掩码：`mask_shadow = (P_shadow > t_s)`
- 阈值 `t_c/t_s`：在验证集上扫描确定（S1.4）。

### 4.2 建议特征字段（scene 级）

**覆盖率与置信度**
- `cloud_frac_full = mean(mask_cloud)`
- `shadow_frac_full = mean(mask_shadow)`
- `cloud_conf_mean = mean(P_cloud | mask_cloud)`
- `shadow_conf_mean = mean(P_shadow | mask_shadow)`

**不确定性**
- `entropy_mean = mean( -p log p -(1-p) log(1-p) )`（对 `P_cloud`，可再加 shadow）
- `boundary_uncertainty`：在边界环带内的平均熵/低置信比例
  - 边界环带：`ring = dilate(mask_cloud,r) XOR erode(mask_cloud,r)`，`r=3~7px`

**连通域与形态**
- `num_cloud_cc`
- `largest_cloud_cc_frac = area(largest_cc) / (H*W)`
- `cc_area_p90`, `cc_area_max`
- `fragmentation = num_cloud_cc / (cloud_frac_full + eps)`
- 可选：`compactness_mean`、`elongation_mean`（用于区分条带噪声/纹理误检）

**质量标记（可选但很有用）**
- `stripe_score`（条带噪声）
- `overexposure_ratio`（过曝比例）
- `glare_like_ratio`（高亮地物/反光占比的粗略信号，如高 DN 区域占比）

### 4.3 结构化特征 JSON 模板
```json
{
  "scene_id": "xxx",
  "stats": {
    "cloud_frac_full": 0.23,
    "shadow_frac_full": 0.06,
    "cloud_conf_mean": 0.81,
    "shadow_conf_mean": 0.63,
    "entropy_mean": 0.18,
    "boundary_uncertainty": 0.11,
    "num_cloud_cc": 41,
    "largest_cloud_cc_frac": 0.04,
    "cc_area_p90": 2100,
    "cc_area_max": 9800,
    "fragmentation": 178.2
  },
  "quality": {
    "stripe_score": 0.02,
    "overexposure_ratio": 0.01,
    "glare_like_ratio": 0.07
  },
  "thresholds": {
    "t_cloud": 0.5,
    "t_shadow": 0.5
  },
  "policy_id": "global_screening_v1"
}
```

---

## 5. Router（全自动路由器：FAST_ACCEPT / FAST_REJECT / ESCALATE）

### 5.1 Router 的职责
Router **不输出最终 ACCEPT/REJECT 的业务结论**，而是输出：
- 能否走 **快速链路**（FAST_ACCEPT / FAST_REJECT）
- 是否需要 **升级验证链路**（ESCALATE）

> ESCALATE 不代表人工复核，而是触发 **second-check + patch 核验**。

### 5.2 Router 输出 schema
```json
{
  "route": "FAST_ACCEPT | FAST_REJECT | ESCALATE",
  "why": ["...短理由..."],
  "next": {
    "run_second_check": true,
    "sample_patches": {"enabled": true, "k": 12, "strategy": "highest_uncertainty_boundary"}
  }
}
```

### 5.3 一个可用的默认路由策略（policy_v1）
（你可以在验证集上再调）

- **FAST_ACCEPT**
  - `cloud_frac_full <= 0.15`
  - `boundary_uncertainty <= 0.06`
  - `entropy_mean <= 0.12`

- **FAST_REJECT**
  - `cloud_frac_full >= 0.35` **或**
  - `shadow_frac_full >= 0.12`

- **ESCALATE（灰区）**
  - 其它情况（尤其临界云量、边界不确定高、碎片化高）

> Router 的设计应偏保守：宁可 ESCALATE 多一点，也别把灰区直接 FAST_ACCEPT。

### 5.4 Router 的实现选证明细节
- **强推荐：线上 route 用轻量分类器（XGBoost/MLP）**（高吞吐）
- **LLM Router（Qwen2-VL）**：只用来生成 route + reasons（更可解释、策略更易迭代）
- 也可以直接用 LLM 做 route，但务必：`temperature=0 + 强制 JSON schema`。

---

## 6. Second Checker（全自动 second opinion）

> 目的：在 GF-1（缺少 SWIR/TIR）导致灰区多的情况下，为 Auto-Judge 提供“第二观点”，提升稳健性。

### 6.1 选项（择一）
- **传统/规则云检**：光谱阈值 + 纹理 + 形态 + 云影几何投影（适合 GF-1 的经典思路）
- **第二个轻量分割模型**：不同 backbone/不同训练策略作为对照（最容易工程化）
- **蒸馏模型**：从主模型蒸馏一个更轻版本做校验器

### 6.2 second-check 输出特征
- `cloud_frac_full_2`
- `shadow_frac_full_2`
- `agreement_iou_cloud = IoU(mask_cloud, mask_cloud_2)`
- `agreement_iou_shadow`
- 可选：`disagreement_area = mean(mask_cloud XOR mask_cloud_2)`

---

## 7. Patch Sampler（全自动抽样与证据构建）

> 目的：让 MLLM 只看“最有信息量”的局部，而不是整景。

### 7.1 抽样策略（推荐混合）
总数 `K=8~20`（默认 12）：
1) **最高不确定边界**（50%）
2) **最大云团边缘**（30%）
3) **分歧区域**：主分割 vs second-check 差异最大（20%）

### 7.2 Patch 视图（每个 patch 2~3 张图）
- `RGB`（从 B,G,R 组成）
- `NIRRG`（NIR,R,G 伪彩；或 NIR 灰度）
- `overlay`（把预测 mask 半透明叠加到 RGB）

### 7.3 Patch 抽样伪代码
```python
def sample_patches(P_cloud, mask_cloud, mask_cloud_2=None, K=12, patch=512):
    # 1) 计算边界环带
    ring = dilate(mask_cloud, r=5) ^ erode(mask_cloud, r=5)

    # 2) 不确定性（熵）
    p = clip(P_cloud, 1e-6, 1-1e-6)
    entropy = -(p*log(p) + (1-p)*log(1-p))

    # 3) 候选中心：ring 内 entropy 最大的点
    cand1 = topk_points(entropy * ring, k=K//2)

    # 4) 最大连通域边缘
    largest_cc = get_largest_cc(mask_cloud)
    edge_largest = boundary(largest_cc)
    cand2 = topk_points(entropy * edge_largest, k=K*3//10)

    # 5) 分歧区域（有 second-check 时）
    cand3 = []
    if mask_cloud_2 is not None:
        disagree = mask_cloud ^ mask_cloud_2
        cand3 = topk_points(entropy * disagree, k=K - len(cand1) - len(cand2))

    centers = merge_and_dedup(cand1 + cand2 + cand3)
    return crop_patches(centers, patch_size=patch)
```

---

## 8. Patch Verifier（Qwen2-VL：自动核验器，可选但推荐）

> 注意：这不是人工 Judge，而是自动“局部证据核验器”。它对 GF-1 灰区（薄云/雾霾/高亮地物混淆）很有帮助。

### 8.1 输入与输出
输入：每个 patch 的 `{RGB, NIRRG, overlay}` + 简短指令  
输出（建议固定 JSON）：
```json
{
  "cloud_like_prob": 0.0,
  "shadow_like_prob": 0.0,
  "confusing_surface_prob": 0.0,
  "notes": "一句话"
}
```

### 8.2 训练标签（无需人工）
用 GT 自动生成 patch 标签（监督微调/LoRA）：
- `gt_cloud_ratio = mean(gt_cloud_mask in patch)`
- `gt_shadow_ratio = mean(gt_shadow_mask in patch)`
- 规则示例：
  - `gt_cloud_ratio > 0.5 -> CLOUD`
  - `gt_shadow_ratio > 0.5 -> SHADOW`
  - 否则 `CLEAR`
并额外采样灰区：从预测高熵边界、分歧区域采样，让模型学会判别“像云但不是云”的局部。

---

## 9. Auto-Judge Fusion（全自动最终判定）

> 核心：最终决策 **由融合器** 输出，避免 LLM 自由发挥。LLM 只提供解释 reasons(JSON)。

### 9.1 需要的融合输入
- 主分割特征：`cloud_frac_full`, `shadow_frac_full`, `boundary_uncertainty`, `fragmentation`, ...
- second-check：`agreement_iou_cloud/shadow`, `delta_cloud_frac`, ...
- patch 核验：聚合统计（如平均 cloud_like_prob、top-3 最大值等）

### 9.2 一个可上线的融合得分
```text
score =
  a * cloud_frac_full
+ b * shadow_frac_full
+ c * boundary_uncertainty
+ d * (1 - agreement_iou_cloud)         # 有 second-check 时
+ e * patch_cloud_like_mean             # 有 patch verifier 时
+ f * patch_confusing_surface_mean      # 反向项：高混淆 -> 更保守（可加权）
```

判定：
- `score < T_accept -> ACCEPT`
- `score >= T_reject -> REJECT`
- 中间段：再跑一次 ESCALATE（如果还没跑），仍无法消歧时可选择 **REJECT_SAFE**（保守拒收），保持全自动。

### 9.3 融合权重如何训练（无需人工）
在验证集上，以 GT 为监督，做一个简单的校准/拟合：
- 目标：最大化 `Accept/Reject` 的业务指标（比如 Reject 的精确率更重要）
- 方法：
  - Logistic Regression / Linear SVM 拟合权重
  - 或直接用网格搜索调 `a..f` 与阈值 `T_accept/T_reject`
- 不需要人工标注，只用数据集 GT。

---

## 10. 训练流程（端到端，无人工）

### S1：训练 CloudSegNet（监督分割）
1) 切 patch（512×512）+ 混合采样（云附近/随机）
2) 训练：AdamW + cosine + warmup + mixed precision
3) 验证：IoU/F1（cloud/shadow）
4) 阈值标定：`t_c/t_s` 在 val 上扫描
5) 概率校准（可选但建议）：temperature scaling

### S2：离线跑全数据，生成特征库
对 train/val/test 每景：
- 输出 P_cloud/P_shadow、mask、所有全景特征 JSON
- 保存中间产物（便于回放/复现实验）

### S3：构建 Router 训练集并训练（无需人工）
1) 用 GT 计算每景误差：`IoU_cloud/shadow`, `abs(frac_err)` 等
2) 以“风险规则”自动生成 route 标签（FAST/ESCALATE）
3) 训练轻量 Router（XGBoost/MLP）
4) （可选）对 Qwen2-VL 做 LoRA：输入 features JSON 输出 route JSON（文本-only）

### S4：构建 Patch Verifier 训练集并训练（可选但推荐）
1) 对 ESCALATE 样本抽 patch（不确定边界/分歧区）
2) 用 GT 自动贴 patch 标签（CLOUD/SHADOW/CLEAR）
3) 用 Qwen2-VL 做 LoRA/SFT，让它输出核验概率 JSON

### S5：训练/拟合 Auto-Judge Fusion（无需人工）
1) 用 val 集拟合权重与阈值（LogReg / 网格搜索）
2) 固化 policy 版本（`policy_id`）

### S6：端到端评估
- 分割指标：IoU/F1（cloud/shadow）
- 决策指标：Accept/Reject 准确率、Reject 精确率（保守拒收策略下很关键）
- 成本指标：ESCALATE 比例、平均 patch 数、吞吐与延迟

---

## 11. 本地部署（Qwen2-VL + vLLM 推荐）

### 11.1 为什么用 AWQ 7B
- Router/Judge 场景文本短、图像 patch 少，7B AWQ 很容易在 12–16GB 显存上跑起来。
- 首发建议：`Qwen2-VL-7B-Instruct-AWQ`（之后再根据吞吐升级/降级）。

### 11.2 vLLM 启动示例
```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct-AWQ \
  --task generate \
  --dtype auto \
  --max-model-len 8192 \
  --allowed-local-media-path /data/mm \
  --limit-mm-per-prompt image=5
```

### 11.3 Router/Judge 调用约束
- `temperature=0`
- 强制输出 JSON（schema 校验，不合格就重试一次）
- Router 默认不带图片（只带 features JSON）
- 只有 ESCALATE 才带 patch 图像给 patch verifier（或 Judge 解释器）

---

## 12. Prompt 模板（强约束 JSON）

### 12.1 Router Prompt（只吃文本）
```text
你是遥感云检测的自动路由器。
输入是一个 JSON（全景特征、阈值、policy_id）。
你只能输出一个 JSON，符合下面 schema：
{ "route": "...", "why": ["..."], "next": {...} }
禁止输出多余文字。禁止修改阈值。

根据 policy_id=global_screening_v1 进行路由：
- FAST_ACCEPT: cloud_frac_full<=0.15 且 boundary_uncertainty<=0.06 且 entropy_mean<=0.12
- FAST_REJECT: cloud_frac_full>=0.35 或 shadow_frac_full>=0.12
- 其它：ESCALATE，并要求 run_second_check=true 且 sample_patches.enabled=true (k=12)
```

### 12.2 Patch Verifier Prompt（多图）
```text
你是遥感云/云影patch核验器。输入包含：RGB、NIRRG、overlay 三张图。
请判断该patch更像：云、云影、清晰地表、或高亮地物/雾霾等混淆。
只输出JSON：
{
  "cloud_like_prob": 0~1,
  "shadow_like_prob": 0~1,
  "confusing_surface_prob": 0~1,
  "notes": "一句话"
}
禁止输出多余文字。
```

### 12.3 Judge 解释器（可选，仅负责 reasons，不负责最终决策）
```text
你是决策解释器。输入给出：features JSON + second-check一致性 + patch核验聚合。
你只能输出 reasons JSON：列出命中阈值与证据数字。
禁止更改阈值、禁止给出最终ACCEPT/REJECT（由融合器输出）。
```

---

## 13. 产物与版本管理（建议）
- `seg_model.pt`：CloudSegNet 权重
- `calibration.json`：阈值/概率校准参数（t_c, t_s, temp 等）
- `policy_v1.json`：路由与融合策略版本化配置
- `router_model.bin`：轻量 Router（可选）
- `patch_verifier_lora/`：Qwen2-VL LoRA（可选）
- `fusion_weights.json`：融合器权重 + 阈值
- `logs/`：每景 features + route + 决策 + 关键 patch 索引（可回放）

---

## 14. 最小可行落地（MVP）路线图
1) 只训练 CloudSegNet + 阈值标定（S1）
2) 做 Metrics Builder（S2）
3) Router 先用规则实现（不需要 LLM）
4) Auto-Judge 先用简单融合得分（无 second-check，无 patch verifier）
5) 再逐步加入：second-check → patch verifier → Router/Verifier LoRA

---

## 15. 你需要我补充的（可选）
如果你希望我把它进一步“落到代码级别”，我可以按你当前框架（PyTorch/Lightning/MMseg 等）补：
- 完整的 feature 计算实现（含连通域/边界环带/熵）
- vLLM 的请求封装（Router + patch verifier + reasons）
- fusion 权重拟合脚本（LogReg + 阈值搜索）
- GF-1 大景滑窗推理与拼接策略（含重叠、边缘融合）
