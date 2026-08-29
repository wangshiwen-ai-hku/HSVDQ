# V3-OAR：面向 activation outlier 的分组与 FP 低秩路由

> 状态：新 V3 设计稿。原 teacher--student trajectory-V3 已验证失败并废弃；本文不使用跨层
> trajectory correction、cumulative teacher target 或 paired block trajectory。新 V3 只在已验证
> 有效的 V2 / V2-plus 上增加局部、可回退的优化。

实现状态（当前分支）：`v3_outlier_routing.py` 已包含 eager grouping、tail-aware `D`、
精确 loss 分解、fixed-code rank-r routing 和 randomized block Hadamard；`HSVQuantLinear`
支持可选静态 permutation 与 block Hadamard。native CUDA 暂时显式拒绝这两种 transform，
避免无意破坏连续 K layout。可复现实验见
`scripts/benchmarks/toy_v3_outlier_routing.py` 与
`hsvdquant/toy/results/v3_outlier_routing/report.md`。

## 1. 目标与结论

新 V3 同时做两件互补的事：

1. **降低 activation quantizer 本身的损失**：保持硬件 group size（首选 `g=128`）不变，离线选择
   静态 channel permutation，使 tail channel 被隔离到少数组，减少 shared max 对 body channel 的污染；
   随后在新分组上重解 smoothing `D`。必要时才加入 group-local Hadamard。
2. **把仍然存在、但可预测的 outlier 输出误差吸收到已有 FP16 低秩分支**：保留 V2-plus 的
   `L1/L2 + W4A4 residual` 结构和总 rank，用 activation quantization 的真实交叉协方差监督
   `L1/L2` 的联合旋转，而不是仅用 activation 幅值或对 frozen-code `L2` 做普通 refit。

新 V3 不增加第三条推理分支，不增加 rank；基础版本也不要求改变 `g=128` 的 MMA 主循环。
base V3 的核心部署变化是一份静态 permutation，以及与之相匹配的 residual weight packing；
V3-H 额外需要在 activation pack 前融合 group-local FWHT。

---

## 2. 为什么原 V2 模型仍漏掉 outlier

考虑一层 `Y = XW`。沿用 H-SVDQuant 记号：

\[
U=XD^{-1},\qquad \widetilde W=DW,\qquad B=L_1L_2.
\]

令 \(P\) 为离线选择的 channel permutation，\(V=UP\)。FP 分支仍可在原物理顺序计算
\(UB\)；只把量化 residual 写到 permutation 后的坐标：

\[
R_P=P^\top(\widetilde W-B),\qquad Q=Q_W(R_P).
\]

部署输出为

\[
\widehat Y=UB+Q_{A,G}(V)Q,
\]

其中 \(G\) 是 permutation 后的连续、等长 group。定义

\[
E_A=V-Q_{A,G}(V),\qquad \Delta=R_P-Q.
\]

则精确的局部输出误差是

\[
Y-\widehat Y=V\Delta+E_AQ. \tag{1}
\]

给 token 权重 \(\Omega\succeq0\)、输出敏感度矩阵 \(\Gamma\succeq0\)，定义

\[
\mathcal J=\left\|\Omega^{1/2}(Y-\widehat Y)\Gamma^{1/2}\right\|_F^2.
\]

令

\[
H=V^\top\Omega V,\quad
C=V^\top\Omega E_A,\quad
\Sigma_A=E_A^\top\Omega E_A,
\]

代入式 (1) 得到精确分解

\[
\boxed{
\mathcal J=
\operatorname{tr}\!\left[\Gamma\Delta^\top H\Delta\right]
+2\operatorname{tr}\!\left[\Gamma\Delta^\top C Q\right]
+\operatorname{tr}\!\left[\Gamma Q^\top\Sigma_AQ\right].
} \tag{2}
\]

V2 的 `F_W + lambda F_A` 对应式 (2) 的第一项和第三项，并在 additive、zero-mean、
independent noise 假设下丢掉第二项。对普通 rounding noise 这是合理近似；对由 group max
驱动的确定性 tail error，\(C=V^\top\Omega E_A\) 往往不小：

- outlier 抬高 group scale 后，body channel 的 rounding residual 会随该 outlier pattern 重复出现；
- skew、稀疏和 code occupancy 不均使误差条件均值不为零；
- 因而部分 \(E_AQ\) 可以被输入 \(V\) 线性预测，正是现有 FP 低秩分支能够吸收的部分。

**新 V3 的理论核心不是再加一个任意 loss，而是恢复被 V2 忽略的交叉项 \(C\)。**

---

## 3. V3-A：固定 `g=128` 的 outlier-aware index grouping

### 3.1 为什么按物理位置连续分组会递减

对 permutation 后的一个 group \(G\)，对称均匀 A4 的量化步长为

\[
s_{tG}=\frac{\max_{i\in G}|V_{ti}|}{\kappa_A},\qquad \kappa_A=2^{b_A-1}-1.
\]

一个极端 channel 会把同组其余 \(g-1\) 个 channel 的步长一起放大。减小 `g` 有效，是因为
它降低了一个 outlier 可以污染的 channel 数量；但 native 连续分组只是在物理轴上切得更细，
无法保证 tail channel 与 body channel 分开，所以收益必然逐渐递减。

这也解释了为什么实验中 **g-half 在未加 Hadamard 的 V3 之前仍然最好**，虽然它不能直接选择
outlier。若某 token 的一个 channel 为 \(M\)，其余为小 body，outlier 自己通常正好映射到
量化端点，误差反而接近 0；主要损失来自它把同组 body 的步长抬成
\(\Delta\approx M/\kappa_A\)。高分辨率近似下，该组 body 总误差为

\[
\mathbb E\|e_{\rm body}\|_2^2
\approx (g-1)\frac{M^2}{12\kappa_A^2}. \tag{3a}
\]

把 `g` 减半无需知道 outlier 在哪里，就把一次异常的“污染半径”从约 \(g-1\) 降到
\(g/2-1\)，同时为每个 token 加倍 scale 数。相反，静态 permutation 只能隔离跨 token
稳定的 tail channel；它不降低 \(M\)，也无法处理 token-dependent tail location，且被移入
tail group 的 body 仍共享大 scale。`L1/L2` 可以吸收可预测的输出误差，却不能改变 A4
quantizer 的这个 operator geometry。因此 g-half 是强而稳定的 reference，不是“直接找到
outlier”才有效。

### 3.2 不能只按 `amax` 排序

仅将 channel 按 `amax` 排序是可用的 warm start，但不是目标函数。量化误差最终经过 residual
weight 投影到输出；同样大的 activation error，如果对应的 \(Q_{i,:}\) 很小，影响也可能很小。
此外，两个 tail channel 的极值是否发生在相同 token 上，会改变 group max 的出现频率。

对候选 group \(G\) 定义直接的 output-aware 代价

\[
c(G)=\sum_t\omega_t
\left\|
\big(V_{t,G}-Q_{A,G}(V_{t,G})\big)
Q_{G,:}\Gamma^{1/2}
\right\|_2^2. \tag{3}
\]

忽略不同 group 量化误差之间较小的交叉协方差后，分组问题为

\[
\min_{\{G_k\}}\sum_k c(G_k),
\qquad |G_k|=g,\quad \bigsqcup_kG_k=[c]. \tag{4}
\]

式 (3) 同时包含三类信息：

- activation 的幅值与 tail token 共现模式；
- residual row 的输出敏感度 \(\|Q_{i,:}\Gamma^{1/2}\|_2\)；
- 当前 `D`、codes 和 group scale 的真实相互作用。

因此它能解释“sort-by-amax 有的层改善、有的层退化”：排序只使用了第一类信息的一部分。

### 3.3 阶梯分组的可执行形式

不使用不规则 group size；使用**固定容量、阶梯式 channel tier + 代价细化**：

1. 用 robust tail score（例如 `p99.9(|V_i|)`、top-token energy、spikiness）将 channel 分成
   extreme / high / body 三到五档；
2. 先把幅值相近的 channel 放在相同档，避免 extreme 与 body 共用 scale；
3. 在每一档内按 tail-token signature 和
   \(\|Q_{i,:}\Gamma^{1/2}\|_2\) 聚类；
4. 保持每组恰好 128 个 channel，用 swap / local search 直接下降式 (3)；
5. 用独立 calibration split 计算完整式 (1)，只有 held-out output loss 下降才接纳该层的 \(P\)。

这比“不同组用 32/64/128 的不规则阶梯”更适合当前 runtime：逻辑上的 channel 是精选的，
物理存储后仍是连续 `g=128`。

### 3.4 permutation 与推理代价

数学上，静态 permutation 完全等价：

\[
VR_P=(UP)\big(P^\top(\widetilde W-B)\big)=U(\widetilde W-B).
\]

推理困难来自内存布局，而非 GEMM 数学：当前 kernel 按连续 K 读取 128 个值并共享 scale；
任意 index gather 会破坏 coalesced load。部署优先级应为：

1. **producer folding**：把 \(P\) 融入前一层输出 channel、LayerNorm 输出或相邻 projection
   的权重布局；Q/K/V 与 gate/up 可共享一次 reorder；
2. **quantize-and-pack fusion**：在 activation quantization/packing kernel 中读取静态 index，
   同时将 residual weight 按 \(P\) 预打包；不产生额外的完整 FP16 tensor；
3. **locality-constrained P**：如果 gather 开销过大，只允许在 512/1024-channel macrotile 内跨
   `g=128` 重组；牺牲少量最优性换取 cache locality；
4. 不能被融合且 held-out gain 小于延迟门槛的层回退到 identity。

因此，不使用 index grouping 的常见原因是 kernel/layout 工程约束，不是理论上无效。

---

## 4. V3-B：在新分组上重解 tail-aware smoothing `D`

在固定 \(P,B,Q\) 下，原有 diagonal-noise 近似可推广为

\[
F_A^{\mathrm{diag}}(D,P)
=\frac{1}{12\kappa_A^2}
\sum_G
\underbrace{\mathbb E_t\max_{i\in G}\frac{X_{ti}^2}{d_i^2}}_{\text{group peak}}
\underbrace{\sum_{i\in G}
\|Q_{i,:}\Gamma^{1/2}\|_2^2}_{\text{output-sensitive residual mass}}. \tag{5}
\]

式 (5) 比只看 activation range 更合理，但平均 MSE 仍会低估极少数 massive token。令

\[
\ell_t=\frac{\|(E_AQ)_{t,:}\Gamma^{1/2}\|_2^2}
{\|Y_{t,:}\Gamma^{1/2}\|_2^2+\epsilon},
\]

使用 mean + tail risk：

\[
\mathcal L_{\mathrm{op}}
=F_W+\lambda_{\mathrm{mean}}\,\mathbb E_t\ell_t
+\lambda_{\mathrm{tail}}\,\operatorname{CVaR}_{\tau}(\ell), \tag{6}
\]

\[
\operatorname{CVaR}_{\tau}(\ell)
=\min_{z}\left[z+\frac{1}{1-\tau}\mathbb E_t(\ell_t-z)_+\right].
\]

推荐 `tau=0.99` 起步。训练时可将 CVaR 写成 IRLS token 权重 \(\Omega\)，继续复用 V2 的
log-space `D` 优化；但必须在 held-out split 同时报 mean、p99/CVaR 和 A16 weight loss。

关键顺序是 **先选 P，再重解 D，再重算 codes**。如果先优化 D 再换 group，D 所看到的
within-group max 已经过时。

---

## 5. V3-C：用交叉项监督 L1/L2，把 outlier 路由到 FP 分支

### 5.1 固定 codes 时的精确 rank-r 解

令 \(W_P=P^\top\widetilde W\)、\(B_P=P^\top B\)。固定 \(D,P,Q\) 后，FP 分支应拟合

\[
T=VW_P-Q_{A,G}(V)Q
=V(W_P-Q)+E_AQ. \tag{7}
\]

于是 `L1/L2` 的正确局部问题不是只重构 weight，而是 reduced-rank regression：

\[
\min_{\operatorname{rank}(B_P)\le r}
\left\|\Omega^{1/2}(T-VB_P)\Gamma^{1/2}\right\|_F^2. \tag{8}
\]

无 rank 约束的目标为

\[
B_{\mathrm{uc}}
=H^{-1}V^\top\Omega T
=\underbrace{W_P-Q}_{\text{weight residual}}
+\underbrace{H^{-1}CQ}_{\text{predictable activation-error correction}}. \tag{9}
\]

式 (9) 清楚说明如何“增强 L1/L2 信息”：

- `L1` 不再只看 \(H\) 或 \(\Sigma_A^{1/2}W\)，而是选择能预测 \(E_AQ\) 的输入方向；
- `L2` 学习这些 FP feature 应该向哪些 output channel 输出，并由 \(\Gamma\) 加权；
- `L1/L2` 共同旋转，仍保持 rank `r`，而不是给 frozen `L1` 单独重拟合 `L2`。

式 (8) 有闭式 rank-r 解。若

\[
H^{1/2}B_{\mathrm{uc}}\Gamma^{1/2}=U_rS_rV_r^\top+\text{tail},
\]

则

\[
B_P^*=H^{-1/2}U_rS_rV_r^\top\Gamma^{-1/2}, \tag{10}
\]

可取

\[
L_{1,P}=H^{-1/2}U_rS_r^{1/2},\qquad
L_2=S_r^{1/2}V_r^\top\Gamma^{-1/2},\qquad
L_1=PL_{1,P}. \tag{11}
\]

这一步在 fixed-code 条件下全局最优，而且旧的 rank-r branch 是可行点，所以 calibration loss
不会上升。随后重新构造 residual 并 GPTQ；若 requantization 后 held-out loss 上升，回退。

### 5.2 只吸收可泛化的 outlier，而不是量化噪声

直接使用 calibration 上的 \(E_AQ\) 可能让 branch 拟合随机 rounding noise。把它分解为

\[
E_AQ=m_\Phi+\xi,\qquad
m_\Phi=\mathbb E[E_AQ\mid\Phi],\qquad
\mathbb E[\xi\mid\Phi]=0. \tag{12}
\]

\(\Phi\) 只使用 quantizer 已有的局部信息，例如 group scale、code occupancy、tail tier、
top-channel identity 和 saturation ratio。用 cross-fitting 在一半 token 上估计 \(m_\Phi\)，在另一半
token 上生成监督，然后交换两半。运行时不需要预测器；它只用于校准时判断哪部分误差稳定。

将式 (9) 改为

\[
B_{\mathrm{uc}}^{\mathrm{red}}
=W_P-Q+H^{-1}V^\top\Omega m_\Phi. \tag{13}
\]

只有 \(m_\Phi\) 在 held-out 上的 explained energy 为正，才允许它占用 rank。一个自然的
rank 分配判据是比较

\[
s_k^2\!\left(H^{-1/2}V^\top\Omega m_\Phi\Gamma^{1/2}\right)
\]

与当前 weight-reconstruction 的下一条边际奇异值；前者更大时，才把该方向旋入 FP branch。

### 5.3 保护已经验证有效的 V2-plus

以 V2-plus checkpoint 为 baseline \((D_0,B_0,Q_0)\)。任何 V3 branch candidate 必须满足：

\[
F_W(B,Q)\le(1+\varepsilon_W)F_W(B_0,Q_0), \tag{14}
\]

并且在独立 token split 上同时满足：

- A4 output loss / CVaR 下降；
- A16 output loss 不超过 trust region；
- 分组、D、branch、requantization 全部完成后的端到端局部 loss 下降。

用 `alpha in {0, 1/8, 1/4, 1/2, 1}` 对 correction 做 backtracking；每个 `alpha` 都重新做
rank-r retraction，而不是直接相加成 rank-2r。`alpha=0` 永远保留，所以 V3 是可回退增强。

---

## 6. output-channel loss 与分布对齐应该怎样加入

不建议直接按当前 error 最大的 output channel 设永久权重，这会产生自激式过拟合。优先使用：

1. **token tail 权重**：式 (6) 的 CVaR，解决 massive activation 的稀有性；
2. **下游敏感度 \(\Gamma\)**：首选 diagonal / low-rank Gauss--Newton 或 Fisher 近似；没有时用
   \(\Gamma=I\)；
3. **paired output regression**：式 (8) 比只匹配输出均值/方差更强，因为它逐 token 对齐。

如果仍观察到系统性 bias，可加入很小的条件矩 loss：

\[
\mathcal L_{\mathrm{mom}}
=\|\mu_{\widehat Y\mid\mathrm{tail}}-\mu_{Y\mid\mathrm{tail}}\|_\Gamma^2
+\eta\|\Gamma^{1/2}(C_{\widehat Y\mid\mathrm{tail}}-C_{Y\mid\mathrm{tail}})
\Gamma^{1/2}\|_F^2. \tag{15}
\]

它只作 tie-breaker，不能替代式 (8)。另外，纯 diagonal \(\Gamma\) 对逐 output 独立的 weight code
最优解影响有限，但会显著影响共享 rank 的 L1/L2 方向选择，因此应主要用在 branch 和 admission。

---

## 7. 与 Hadamard / rotation 的关系

Permutation、g-half 与 Hadamard 解决不同问题：

- permutation 保持每个数值不变，\(\|xP\|_\infty=\|x\|_\infty\)；它只能把 outlier 污染
  **隔离**到少数组；
- signed Hadamard 会混合坐标。对固定向量 \(x\)，随机符号 Hadamard 的最大坐标以高概率约为
  \(O(\|x\|_2\sqrt{\log c/c})\)，因此能把单个巨大峰 **摊平**；代价是在线 transform、
  weight-side range 变化和额外 kernel 复杂度。

本实现对每个 group 使用 \(R_G=D_sH_G\)，其中 \(D_s\) 是离线选择的 Rademacher 符号，
\(H_G\) 是 normalized Hadamard。量化 residual path 变成

\[
Q_A(V_GR_G)\;Q_W\!\left(R_G^\top R_{P,G}\right), \qquad
(V_GR_G)(R_G^\top R_{P,G})=V_GR_{P,G}. \tag{15a}
\]

对单个幅值 \(M\) 的 spike，变换后每个坐标约为 \(M/\sqrt g\)，因此它不再用 \(M\)
决定整组步长。与 g-half 只减少受污染 channel 数相比，Hadamard 直接降低组内
\(\ell_\infty/\ell_2\)；但它可能增大 transformed weight range，所以符号、weight codes 与
最终输出必须联合重算并经 held-out admission。

所以不是“为什么不用思路 1”，而是：RPTQ 类方法确实使用 reorder，DuQuant 进一步组合了
rotation 与 permutation。对当前 H-SVDQuant，推荐顺序为：

1. 先做固定 `g=128` 的 output-aware permutation；
2. 重解 D；
3. 若某组 held-out `peak/RMS` 仍过高，再只对该组加入 128-point block Hadamard：

\[
(V_GR_G)(R_G^\top R_{P,G})=V_GR_{P,G},\qquad R_G=D_sH_G,
\]

   它只作用于 quant residual path，FP branch 不需要变化；
4. Hadamard candidate 同样必须通过 latency + held-out loss admission。

这形成轻量 base V3（P + D + FP routing）与可选 V3-H，而不是一开始就让所有层承担旋转成本。

---

## 8. 完整的 V3-OAR 交替算法

### 输入

V2-plus checkpoint、calibration token、固定 `g=128`、总 rank `r`。

### 每层步骤

1. **Baseline**：加载 V2-plus 的 \(D,B,Q\)，记录 A16、A4 mean、p99/CVaR。
2. **Group proposal**：用 robust tail tier 生成 `P_sort`，再用式 (3) 做 balanced swap refinement，
   同时保留 identity 和 sort-only。
3. **Group admission**：每个 P 都重排 residual rows、重新 GPTQ，在 held-out 上选最优；不改善则
   identity。
4. **D update**：在选中的 P 上最小化式 (6)，然后重算 B/Q；带 A16 trust region。
5. **Cross-fitted target**：计算 \(E_AQ\)、\(C\)、\(m_\Phi\) 和 reducible-energy ratio。
6. **Joint L1/L2 rotation**：按式 (10)--(13) 求 rank-r candidate，backtracking 后重新 GPTQ。
7. **Optional H**：仅对仍有严重 within-group peak 的 admitted 层尝试 group-local Hadamard。
8. **Final admission**：完整 forward 在 held-out 上必须优于 V2-plus；否则逐块回退到最近的安全状态。

### 推荐总目标

\[
\boxed{
\min_{D,P,\operatorname{rank}(B)\le r,Q}
\mathbb E_t\ell_t
+\lambda_{\mathrm{tail}}\operatorname{CVaR}_{0.99}(\ell)
\quad
\text{s.t. A16 trust、fixed }g=128\text{、W4 codes。}
} \tag{16}
\]

式 (2) 用于解释与 block solver；式 (16) 的真实 paired output loss 用于最终 admission。

---

## 9. 最小消融矩阵与证伪条件

| 变体 | grouping | D | L1/L2 | 目的 |
|---|---|---|---|---|
| V2 | native | V2 | V2 | 已验证 baseline |
| V2-plus | native | V2+ | reducible update | 当前强 baseline |
| V3-P-sort | amax tier | re-solve | V2+ | 验证简单阶梯分组 |
| V3-P-cost | 式 (3) | re-solve | V2+ | 验证 output-aware grouping |
| V3-R | native | V2+ | 式 (13) | 隔离 FP routing 增益 |
| V3-full | 式 (3) | 式 (6) | 式 (13) | 新 V3 |
| V3-H | V3-full + H | 式 (6) | 式 (13) | 剩余 peak 的上限 |
| g64 reference | native g64 | re-solve | 式 (13) | 与增 scale 数量比较 |

必须报告：

- A4-vs-A16 local output NMSE：mean、p99、CVaR99；
- 式 (2) 三项，尤其交叉项占比；
- group pollution：`max/RMS`、scale、code occupancy、clean-channel error；
- \(m_\Phi\) 的 held-out explained energy，以及 FP branch 实际吸收比例；
- A16/FW regression；
- Wikitext/C4 PPL 与实际 native latency。

以下任一结果会证伪对应假设：

1. `P-cost` 在 held-out 上不优于 identity/sort：该层 tail 不是稳定 channel-structured，禁用 P；
2. \(\|H^{-1/2}CQ\|\) 大但 cross-fit explained energy 接近 0：交叉项是样本噪声，禁用 routing；
3. branch calibration gain 在 requantization 后消失：Q/B 交替未收敛，需要增加一次 joint iteration，
   不能提高 rank 掩盖；
4. V3-full 不优于 g64：说明额外 scale 的 operator gain仍不可由 P+D+rank-r 模拟；此时应投入
   g64 native kernel，而不是继续加 loss；
5. PPL 不随 local CVaR/NMSE 改善：\(\Gamma=I\) 不足，应先升级下游敏感度，而非恢复 trajectory。

当前 toy 的四个 seed（17/3/11/29）中，V3-H 在三类 case 均 4/4 通过 held-out admission。
平均 test NMSE 相对 g-half 分别下降 8.3%（stable tail）、18.7%（sensitivity mismatch）和
15.0%（tail location shift）；相对 V2-plus 分别下降 37.2%、49.5% 和 38.4%。这是
checkpoint-free synthetic evidence，说明 block Hadamard 能补足 permutation 无法降低组内峰值的
缺口；它不能替代真实模型 PPL、CUDA latency 与逐层 admission。

---

## 10. 实现优先级

1. **先做 eager oracle**：支持任意 P、式 (3) 分组、式 (2) 诊断、式 (13) rank-r 更新；验证新
   V3 是否超过 V2-plus。
2. **再做 native permutation packing**：checkpoint 保存 `activation_permutation`，离线重排 packed
   residual；quantize/pack kernel 支持 producer-folded 或 indexed load。
3. **再融合 native group-local Hadamard**：eager oracle 与 checkpoint state 已实现；native 需要
   在 A4 quantize/pack kernel 内融合 FWHT，并测量它是否抵消 local loss gain。

不要先实现不规则 group size、第三条 sparse FP 分支或新的 trajectory teacher；它们都会同时改变
算法与 runtime，无法判断新 V3 的真实增益来源。

## 参考工作（用于定位，不作为本文假设）

- [SmoothQuant](https://arxiv.org/abs/2211.10438)：diagonal smoothing，把 activation
  quantization 难度迁移到 weights。
- [RPTQ](https://arxiv.org/abs/2304.01089)：按 channel range 重排并聚类量化，说明 index
  grouping 可以通过图级融合降低开销。
- [QuaRot](https://arxiv.org/abs/2404.00456) / [SpinQuant](https://arxiv.org/abs/2405.16406)：
  用正交旋转摊平 outlier；SpinQuant 用 calibration loss 学习旋转。
- [DuQuant](https://arxiv.org/abs/2406.01721)：组合 outlier-aware block rotation、zigzag
  permutation 与第二次 rotation。
- [DFRot](https://arxiv.org/abs/2412.00648)：强调 massive-token 是 long-tail 问题，并使用
  tail-weighted rotation loss。
