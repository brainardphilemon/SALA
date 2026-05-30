---
title: supplementary.md

---

## Hyperparameter Sensitivity

In all experiments, the layer-wise projectors are trained using a shared set of hyperparameters across layers, with the candidate dimension set fixed to $`\mathcal{K}=\{8, 16, 32, 64, 128, 256, 512\}`$.

### TruthfulQA as the Held-out Test Domain

We use the TruthfulQA held-out test setting to assess hyperparameter sensitivity. Overall, the landscape is structured yet robust: the AUROC varies across the $`(\lambda_{s}, \lambda_{a})`$ grid, but we do not observe abrupt collapses or cliff-like failures. Instead, performance changes smoothly, and a broad range of configurations remains competitive. 

Importantly, despite these fluctuations, the sweep consistently stays above the strongest baseline reported in our main results. This indicates that sensitivity manifests as variation around a strong baseline rather than a brittle dependence on a single optimum. Most settings fall within a relatively narrow band, with the best-performing region centered around moderate regularization and small-to-moderate separation.

<p align="center">
  <img src="assets/figures/sweep_heatmap_13.png" alt="Heatmap TruthfulQA" width="70%">
</p>

**Figure 1.1:** Heatmap of test-domain AUROC over the sweep of the separation weight $`\lambda_{s}`$ and the regularization weight $`\lambda_{a}`$ (with $`\lambda_{v}`$ fixed). Higher values indicate better cross-domain performance.

**Sensitivity to $`\lambda_{a}`$ (fixed $`\lambda_{s}`$)** The results exhibit a non-monotonic dependence on the regularization weight $`\lambda_{a}`$. Across multiple fixed $`\lambda_{s}`$ values, moving from zero to a small-to-moderate $`\lambda_{a}`$ improves the AUROC, while overly strong regularization degrades performance. Crucially, this degradation is gradual rather than catastrophic; the curves bend downward smoothly without collapsing. This pattern supports a practical tuning strategy using a coarse grid over $`\lambda_{a}`$, as moderate misspecification still preserves the performance gains over the best baseline.

<p align="center">
  <img src="assets/figures/sweep_slice_vs_reg_13.png" alt="Slice vs reg TruthfulQA" width="70%">
</p>

**Figure 1.2:** One-dimensional slices of the sweep showing AUROC changes as a function of $`\lambda_{a}`$.

**Sensitivity to $`\lambda_{s}`$ (fixed $`\lambda_{a}`$)** Similarly, increasing the separation weight $`\lambda_{s}`$ yields non-monotonic effects. The optimal $`\lambda_{s}`$ relies on the chosen $`\lambda_{a}`$: for a moderate $`\lambda_{a}`$, a larger $`\lambda_{s}`$ is beneficial, whereas at extreme values of $`\lambda_{a}`$, the optimum shifts toward an intermediate $`\lambda_{s}`$. The curves around their maxima are relatively flat, implying a reasonably wide robust range for $`\lambda_{s}`$. The performance consistently exceeds the best baseline, confirming that the method is not overly brittle to tuning errors.

<p align="center">
  <img src="assets/figures/sweep_slice_vs_sep_13.png" alt="Slice vs sep TruthfulQA" width="70%">
</p>

**Figure 1.3:** One-dimensional slices of the sweep showing AUROC changes as a function of $`\lambda_{s}`$.

**Top-performing Configurations** To further quantify the concentration of the optimum, we evaluate the top-10 hyperparameter settings. The best configuration achieves an AUROC of 0.8038 at $`(\lambda_{s}, \lambda_{a}) = (0.4, 0.05)`$, with several nearby configurations yielding comparable results (e.g., $`(0.4, 0.1)`$ and $`(0.4, 0)`$). The tail of the top-10 list maintains strong performance, indicating that the optimum is not an isolated spike. These results suggest that tuning primarily refines performance within an already robust regime; the gains are stable and do not hinge on a fragile configuration.

<p align="center">
  <img src="assets/figures/sweep_top10_13.png" alt="Top 10 TruthfulQA" width="70%">
</p>

**Figure 1.4:** The top-10 hyperparameter configurations from the sweep, ranked by test-domain AUROC on TruthfulQA.

---

### Additional Verification: NQ Open as the Held-out Test Domain

To demonstrate that our sensitivity observations generalize beyond a single dataset, we conduct an additional sweep on NQ Open as a second representative test domain. The qualitative conclusions remain consistent: the response surface is structured but not fragile, and performance varies smoothly without cliff-like failures. 

<p align="center">
  <img src="assets/figures/sweep_heatmap_15.png" alt="Heatmap NQ Open" width="45%">
  <img src="assets/figures/sweep_slice_vs_reg_15.png" alt="Slice vs reg NQ Open" width="45%">
</p>
<p align="center">
  <img src="assets/figures/sweep_slice_vs_sep_15.png" alt="Slice vs sep NQ Open" width="45%">
</p>

**Figure 1.5:** (Top Left) Test-domain AUROC on NQ Open over a grid of $`(\lambda_{s}, \lambda_{a})`$. (Top Right & Bottom) Corresponding one-dimensional slices of the sweep.

Overall, these supplementary results corroborate that the proposed method is resilient to moderate hyperparameter misspecification.


## 2. Additional Analyzing Plots

### 2.1 Layer-wise Probing Performance (In-Domain)

<p align="center">
  <img src="assets/figures/math_layerwise_in_domain_auroc_bars.png" width="48%" />
  <img src="assets/figures/theoremqa_layerwise_in_domain_auroc_bars.png" width="48%" />
  <br>
  <img src="assets/figures/mgsm_layerwise_in_domain_auroc_bars.png" width="48%" />
  <img src="assets/figures/SVAMP_layerwise_in_domain_auroc_bars.png" width="48%" />
</p>

> **Figure 2.1: Layer-wise probing performance varies substantially across domains.** For each mathematical dataset, we train a linear probe at each Transformer layer for hallucination detection and report the in-domain AUROC. This in-domain probing exhibits *layer-index shift*: the layer that achieves the best in-domain performance changes markedly across datasets, suggesting that there is no single probing layer that is consistently optimal.

<br>

<p align="center">
  <img src="assets/figures/indomain_best_layer_bars_7datasets.png" width="80%" />
</p>

> **Figure 2.2: Layer-index shift in the in-domain case.** For each domain, we report the layer index that attains the highest in-domain AUROC, showing that the best-separable layer varies across domains rather than concentrating at a fixed layer.


---

### 2.2 Further Diagnostics of the Proposed Instability Index

In the following figures, we visualize our layer-wise dimension scoring function and show that, in most cases, it selects dimensions that closely match the empirically optimal choices.

<p align="center">
  <img src="assets/figures/layer00_tradeoff.png" width="48%" />
  <img src="assets/figures/layer10_tradeoff.png" width="48%" />
  <br><b>Left:</b> Layer 1. <b>Right:</b> Layer 11.<br><br>
  <img src="assets/figures/layer11_tradeoff.png" width="48%" />
  <img src="assets/figures/layer23_tradeoff.png" width="48%" />
  <br><b>Left:</b> Layer 12. <b>Right:</b> Layer 24.
</p>

> **Figure 2.3:** Proxy score and test AUROC across dimensions $`d`$, with $`d^*`$ and $`d^\dagger`$ indicated by vertical markers.


---

### 2.3 Cross-Domain Probing and Layer-Index Shift

<p align="center">
  <img src="assets/figures/bars_train_math.png" width="48%" />
  <img src="assets/figures/bars_train_SVAMP.png" width="48%" />
  <br>
  <img src="assets/figures/bars_train_mgsm.png" width="48%" />
  <img src="assets/figures/bars_train_theoremqa.png" width="48%" />
</p>

> **Figure 2.4: Cross-domain probing exhibits "layer-index shift" on four mathematical benchmarks (no universal best layer).** Each subplot trains a linear probe for hallucination detection on one training domain (title) and evaluates it on the remaining test domains (y-axis).

<br>

<p align="center">
  <img src="assets/figures/cross_domain_4panel_qa.png" width="90%" />
</p>

> **Figure 2.5: Cross-domain layer-index shift.** The figure contains four panels, each trained on a different training domain (shown in the panel title). Within each panel, the x-axis enumerates the three held-out test domains and the y-axis indexes transformer layers. Each cell reports the AUROC obtained by a model trained on the training domain at the corresponding layer and evaluated on the given test domain.



---

## 3. Theoretical Proofs

In this section, we provide the complete theoretical derivations and mathematical proofs omitted from the main text.

### 3.1 Proof of Lemma 1 (Source Domain Variance)

> **Lemma 1.** Given $`M`$ training domains $`\mathbb{P}_{Q,T}^1, \ldots, \mathbb{P}_{Q,T}^M`$, and $`N`$ test domains $`\mathbb{P}_{Q,T}^{M+1}, \ldots, \mathbb{P}_{Q,T}^{M+N}`$, for any $`e, e' \in\{1,\ldots,M\}`$ and any $`t\in \{M+1,...,M+N\}`$, we have
>
> $$\big|R_{d_{\ell}}^{e}-R_{d_{\ell}}^{e'} \big| \leq \sqrt{2M}\beta_{d_{\ell}}$$
>
> $$\big|R_{d_{\ell}}^t-R_{d_{\ell}}^e\big|\leq \min_{e'\in \{1, \ldots, M\}}\big|R_{d_{\ell}}^t-R_{d_{\ell}}^{e'}\big|+\sqrt{2M}\beta_{d_{\ell}}$$

**Proof.**
Let $`\bar{R}_{d_{\ell}}=\frac{1}{M}\sum_{e=1}^M R_{d_{\ell}}^e`$ and define the centered quantity $`r_{d_{\ell}}^e=R_{d_{\ell}}^e-\bar{R}_{d_{\ell}}`$. Then $`\sum_{e=1}^M r_{d_{\ell}}^e=0`$ and, by definition,

$$
(\beta_{d_{\ell}})^2=\frac{1}{M}\sum_{e=1}^M \big(r_{d_{\ell}}^e\big)^2
$$

Moreover, for any $`e,e'\in\{1,\ldots,M\}`$,

$$
\big|r_{d_{\ell}}^{e}-r_{d_{\ell}}^{e'}\big|=\big|(R_{d_{\ell}}^{e}-\bar{R}_{d_{\ell}})-(R_{d_{\ell}}^{e'}-\bar{R}_{d_{\ell}})\big|=\big|R_{d_{\ell}}^{e}-R_{d_{\ell}}^{e'}\big|
$$

Using the basic inequality $`(a-b)^2 \le 2(a^2+b^2)`$, we have

$$
\big(r_{d_{\ell}}^{e}-r_{d_{\ell}}^{e'}\big)^2\le 2\Big(\big(r_{d_{\ell}}^{e}\big)^2+\big(r_{d_{\ell}}^{e'}\big)^2\Big)\le 2\sum_{e=1}^M \big(r_{d_{\ell}}^e\big)^2
$$

Substituting $`\sum_{e=1}^M (r_{d_{\ell}}^e)^2=M(\beta_{d_{\ell}})^2`$ yields

$$
\big(r_{d_{\ell}}^{e}-r_{d_{\ell}}^{e'}\big)^2\le 2M(\beta_{d_{\ell}})^2
$$

Taking square roots and using $`\big|r_{d_{\ell}}^{e}-r_{d_{\ell}}^{e'}\big|=\big|R_{d_{\ell}}^{e}-R_{d_{\ell}}^{e'}\big|`$ completes this step:

$$
\big|R_{d_{\ell}}^{e}-R_{d_{\ell}}^{e'}\big|\le \sqrt{2M}\,\beta_{d_{\ell}}
$$

Then, let

$$
\bar{R}_{d_{\ell}}=\frac{1}{M}\sum_{e=1}^M R_{d_{\ell}}^e
$$

and standard deviation

$$
\beta_{d_{\ell}}=\sqrt{\frac{1}{M}\sum_{e=1}^M (R_{d_{\ell}}^e-\bar{R}_{d_{\ell}})^2}
$$

Fix any $`e\in\{1,\ldots,M\}`$. For any $`e'\in\{1,\ldots,M\}`$, by the triangle inequality,

$$
|R_{d_{\ell}}^t-R_{d_{\ell}}^e|\le |R_{d_{\ell}}^t-R_{d_{\ell}}^{e'}|+|R_{d_{\ell}}^{e'}-R_{d_{\ell}}^e|
$$

Taking the minimum over $`e'`$ on the right hand, then

$$
|R_{d_{\ell}}^t-R_{d_{\ell}}^e|\le \min_{{e'\in\{1,\ldots,M\}}}|R_{d_{\ell}}^t-R_{d_{\ell}}^{e'}|+\max_{{e'\in\{1,\ldots,M\}}}|R_{d_{\ell}}^{e'}-R_{d_{\ell}}^e|
$$

It remains to bound $`\max_{{e'\in\{1,\ldots,M\}}}|R_{d_{\ell}}^{e'}-R_{d_{\ell}}^e|`$ by $`\beta_{d_{\ell}}`$. This follows from the result above: for any $`e,e'`$,

$$
|R^{e'}_{d_{\ell}}-R^e_{d_{\ell}}|\le \sqrt{2M}\,\beta_{d_{\ell}}
$$

Therefore,

$$
\max_{e'\in\{1,\ldots,M\}}|R^{e'}_{d_{\ell}}-R^e_{d_{\ell}}|\le \sqrt{2M}\,\beta_{d_{\ell}}
$$

Hence,

$$
|R_{d_{\ell}}^t-R_{d_{\ell}}^e|\le \min_{e'\in\{1,\ldots,M\}}|R_{d_{\ell}}^t-R_{d_{\ell}}^{e'}|+\sqrt{2M}\,\beta_{d_{\ell}}
$$

---

### 3.2 Proof of Bounded Score Gap and Total Variation (Lemma 2)

**Lemma 2.** Under the two assumptions described as follows:
* **(A1) Bounded score gap:** $`\big|D_{d_{\ell}}^e\big|\leq B_{D}`$ almost surely for some constant $`B_{D}>0`$.
* **(A2) Non-degeneracy:** There exists $`c>0`$ such that for all $`e\in\{1,\ldots,M\}`$, $`(M_{d_{\ell}}^e)^2+V_{d_{\ell}}^e\geq c`$.

There exists a uniform constant $`C`$, such that for any two domain indices $`e,e'`$,

$$
\big|R_{d_{\ell}}^e-R_{d_{\ell}}^{e'}\big|\leq C d_{\rm TV}(\mathbb{P}_{Q,T}^e\otimes \mathbb{P}_{Q,H}^e,\mathbb{P}_{Q,T}^{e'}\otimes \mathbb{P}_{Q,H}^{e'})
$$

where $`d_{\rm TV}(\cdot,\cdot)`$ denotes total variation distance, and $`\mathbb{P}_{Q,T}^e\otimes \mathbb{P}_{Q,H}^e`$ and $`\mathbb{P}_{Q,T}^{e'}\otimes \mathbb{P}_{Q,H}^{e'}`$ denote product measures, indicating that truthful and hallucinated question-answer pairs are sampled independently from domains $`e`$ and $`e'`$, respectively.

**Proof.**
We want to prove that

$$
\big|R_{d_{\ell}}^e-R_{d_{\ell}}^{e'}\big|\leq C\, d_{\rm TV}\!\Big(\mathbb{P}_{Q,T}^e\otimes \mathbb{P}_{Q,H}^e,\; \mathbb{P}_{Q,T}^{e'}\otimes \mathbb{P}_{Q,H}^{e'}\Big)
$$

where $`R_{d_{\ell}}^e=\frac{(M_{d_{\ell}}^e)^2}{(M_{d_{\ell}}^e)^2+V_{d_{\ell}}^e}`$ with $`M_{d_{\ell}}^e=\mathbb{E}[D_{d_{\ell}}^e]`$ and $`V_{d_{\ell}}^e=\mathrm{Var}(D_{d_{\ell}}^e)`$.

We first state a basic auxiliary lemma:
> **Auxiliary Lemma (TV controls expectation differences of bounded functions).** Let $`f`$ be measurable and bounded with $`\|f\|_{\infty}\le B`$. Then for any probability measures $`P,Q`$,
>
> $$\big|\mathbb E_{P}[f]-\mathbb E_{Q}[f]\big|\le 2B\,d_{\rm TV}(P,Q)$$
>
> where $`d_{\rm TV}(P,Q)=\sup_{A} |P(A)-Q(A)|`$.
> 
> *Proof of Auxiliary Lemma:* Define $`g=(f+B)/(2B)`$. Then $`0\le g\le 1`$ and $`\mathbb E_{P}[f]-\mathbb E_{Q}[f]=2B\big(\mathbb E_{P}[g]-\mathbb E_{Q}[g]\big)`$. For any $`0\le g\le 1`$,
>
> $$\big|\mathbb E_{P}[g]-\mathbb E_{Q}[g]\big|\le \sup_{0\le h\le 1}\big|\mathbb E_{P}[h]-\mathbb E_{Q}[h]\big|=\sup_{A}|P(A)-Q(A)|=d_{\rm TV}(P,Q)$$

First, we decompose $`|R_{d_{\ell}}^e-R_{d_{\ell}}^{e'}|`$ to $`|M_{d_{\ell}}^e-M_{d_{\ell}}^{e'}|`$ and $`|V_{d_{\ell}}^e-V_{d_{\ell}}^{e'}|`$. Define $`g(M,V)=M^2/(M^2+V)`$, so that $`R_{d_{\ell}}^e=g(M_{d_{\ell}}^e,V_{d_{\ell}}^e)`$. On the region $`M^2+V>0`$, $`g`$ is continuously differentiable with:

$$
\begin{aligned}
\frac{\partial g}{\partial M}(M,V)
&=
\frac{2MV}{(M^2+V)^2},
\\
\frac{\partial g}{\partial V}(M,V)
&=
-\frac{M^2}{(M^2+V)^2}.
\end{aligned}
$$

By the multivariate mean value theorem, there exists $`(\tilde M_{d_{\ell}},\tilde V_{d_{\ell}})`$ lying on the line segment connecting $`(M_{d_{\ell}}^e,V_{d_{\ell}}^e)`$ and $`(M_{d_{\ell}}^{e'},V_{d_{\ell}}^{e'})`$ such that:

$$
\begin{aligned}\left|g(M_{d_{\ell}}^e,V_{d_{\ell}}^e)-g(M_{d_{\ell}}^{e'},V_{d_{\ell}}^{e'})\right|&\le\left|\frac{\partial g}{\partial M}(\tilde M_{d_{\ell}},\tilde V_{d_{\ell}})\right|\cdot \left|M_{d_{\ell}}^e-M_{d_{\ell}}^{e'}\right|\\&+\left|\frac{\partial g}{\partial V}(\tilde M_{d_{\ell}},\tilde V_{d_{\ell}})\right|\cdot \left|V_{d_{\ell}}^e-V_{d_{\ell}}^{e'}\right|\end{aligned}
$$

Under (A1), $`|M^e_{d_{\ell}}|=|\mathbb E[D^e_{d_{\ell}}]|\le \mathbb E|D^e_{d_{\ell}}|\le B_{D}`$, and $`0\le V^e_{d_{\ell}}\le \mathbb E[(D^e_{d_{\ell}})^2]\le B_{D}^2`$, hence $`|\tilde M_{d_{\ell}}|\le B_{D}`$ and $`0\le \tilde V_{d_{\ell}}\le B_{D}^2`$.

Under (A2), $`\tilde M_{d_{\ell}}^2+\tilde V_{d_{\ell}}\ge c`$. Therefore,

$$
\begin{aligned}
\lvert \frac{\partial g}{\partial M}(\tilde M_{d_{\ell}},\tilde V_{d_{\ell}}) \rvert
&=
\frac{2\lvert \tilde M_{d_{\ell}} \rvert \tilde V_{d_{\ell}}}
{(\tilde M_{d_{\ell}}^2+\tilde V_{d_{\ell}})^2}
\le
\frac{2B_{D}\cdot B_{D}^2}{c^2}
=
\frac{2B_{D}^3}{c^2},
\\
\lvert \frac{\partial g}{\partial V}(\tilde M_{d_{\ell}},\tilde V_{d_{\ell}}) \rvert
&=
\frac{\tilde M_{d_{\ell}}^2}
{(\tilde M_{d_{\ell}}^2+\tilde V_{d_{\ell}})^2}
\le
\frac{B_{D}^2}{c^2}.
\end{aligned}
$$

Hence,

$$
|R^e_{d_{\ell}}-R^{e'}_{d_{\ell}}|\le\frac{2B_{D}^3}{c^2}|M^e_{d_{\ell}}-M^{e'}_{d_{\ell}}|+\frac{B_{D}^2}{c^2}|V^e_{d_{\ell}}-V^{e'}_{d_{\ell}}|
$$

Then, let $`\mathbb{P}^e=\mathbb{P}_{Q,T}^e\otimes\mathbb{P}_{Q,H}^e`$ and $`\mathbb{P}^{e'}=\mathbb{P}_{Q,T}^{e'}\otimes\mathbb{P}_{Q,H}^{e'}`$. By Assumption 1, we view $`D^e_{d_{\ell}}`$ as $`f(\mathbf Q,\mathbf T,\mathbf Q',\mathbf H)`$ with $`\|f\|_{\infty}\le B_{D}`$. According to the Auxiliary Lemma,

$$
|M^e_{d_{\ell}}-M^{e'}_{d_{\ell}}|=\big|\mathbb E_{\mathbb{P}^e}[f]-\mathbb E_{\mathbb{P}^{e'}}[f]\big|\le 2B_{D}\,d_{\mathrm{TV}}(\mathbb{P}^e,\mathbb{P}^{e'})
$$

For $`|V^e_{d_{\ell}}-V^{e'}_{d_{\ell}}|`$, denote $`V^e_{d_{\ell}}=\mathbb E_{\mathbb{P}^e}[f^2]-(M^e_{d_{\ell}})^2`$ and $`V_{d_{\ell}}^{e'}=\mathbb E_{\mathbb{P}^{e'}}[f^2]-(M^{e'}_{d_{\ell}})^2`$, so

$$
|V^e_{d_{\ell}}-V^{e'}_{d_{\ell}}|\le\big|\mathbb E_{\mathbb{P}^e}[f^2]-\mathbb E_{\mathbb{P}^{e'}}[f^2]\big|+|(M^e_{d_{\ell}})^2-(M^{e'}_{d_{\ell}})^2|
$$

Since $`\|f^2\|_{\infty}\le B_{D}^2`$, then according to the Auxiliary Lemma,

$$
\big|\mathbb E_{\mathbb{P}^e}[f^2]-\mathbb E_{\mathbb{P}^{e'}}[f^2]\big|\le 2B_{D}^2\,d_{\mathrm{TV}}(\mathbb{P}^e,\mathbb{P}^{e'})
$$

Moreover,

$$
\begin{aligned}|(M^e_{d_{\ell}})^2-(M^{e'}_{d_{\ell}})^2|&=|M^e_{d_{\ell}}-M^{e'}_{d_{\ell}}|\cdot |M^e_{d_{\ell}}+M^{e'}_{d_{\ell}}|\\&\le(|M^e_{d_{\ell}}|+|M^{e'}_{d_{\ell}}|)\,|M^e_{d_{\ell}}-M^{e'}_{d_{\ell}}|\\&\le 2B_{D}\,|M^e_{d_{\ell}}-M^{e'}_{d_{\ell}}|\end{aligned}
$$

Combining this with the prior mean bound,

$$
|(M^e_{d_{\ell}})^2-(M^{e'}_{d_{\ell}})^2|\le 2B_{D}\cdot (2B_{D}\,d_{\mathrm{TV}}(\mathbb{P}^e,\mathbb{P}^{e'}))= 4B_{D}^2\,d_{\mathrm{TV}}(\mathbb{P}^e,\mathbb{P}^{e'})
$$

Substituting the squared bounds back gives

$$
|V^e_{d_{\ell}}-V^{e'}_{d_{\ell}}|\le(2B_{D}^2+4B_{D}^2)\,d_{\mathrm{TV}}(\mathbb{P}^e,\mathbb{P}^{e'})= 6B_{D}^2\,d_{\mathrm{TV}}(\mathbb{P}^e,\mathbb{P}^{e'})
$$

Plugging these back into the partial derivatives expansion,

$$
\begin{aligned}|R^e_{d_{\ell}}-R^{e'}_{d_{\ell}}|&\le\frac{2B_{D}^3}{c^2}\cdot (2B_{D}\,\mathrm{TV}(\mathbb{P}^e,\mathbb{P}^{e'}))+\frac{B_{D}^2}{c^2}\cdot (6B_{D}^2\,\mathrm{TV}(\mathbb{P}^e,\mathbb{P}^{e'}))\end{aligned}
$$

Therefore,

$$
|R^e_{d_{\ell}}-R^{e'}_{d_{\ell}}|\le \frac{10B_{D}^4}{c^2}\,d_{\mathrm{TV}}(\mathbb{P}^e,\mathbb{P}^{e'})
$$

---

### 3.3 Proof of Test Domain Bound

Before stating the final theorem, we formally introduce two necessary assumptions regarding the domain distributions.

> **Assumption 1 (Meta Assumption [Dong et al., 2021]).** The distributions $`\{\mathbb{P}_{Q,T}^e \otimes \mathbb{P}_{Q,H}^e\}_{e=1}^{M}`$ are assumed to be drawn independently and identically distributed from a meta distribution $`\mathscr{P}`$ over the joint distribution space $`\mathcal{P}_{Q,T,Q',H}`$, where $`\mathbb{P}_{Q,T}^e \otimes \mathbb{P}_{Q,H}^e`$ is the product distribution of $`\mathbb{P}_{Q,T}^e`$ and $`\mathbb{P}_{Q,H}^e`$.
> 
> **Assumption 2 (Regular Distribution [Dong et al., 2021]).** $`\mathbb{P}_{Q,T}^e \otimes \mathbb{P}_{Q,H}^e`$ is a regular distribution, i.e., for any $`\epsilon>0`$, $`\mathscr{P}(\mathcal{N}^{e}_{\epsilon})>0`$, where $`\mathscr{P}`$ is the meta distribution introduced in Assumption 1 and $`\mathcal{N}^{e}_{\epsilon} = \{\mathbb{P}\in \mathcal{P}_{Q,T,Q',H}:d_{\rm TV}(\mathbb{P},\mathbb{P}_{Q,T}^e \otimes \mathbb{P}_{Q,H}^e)<\epsilon\}`$, here $`d_{\rm TV}(\cdot , \cdot)`$ is the total variation distance.

*Remark:* It is straightforward to verify that if the meta distribution $`\mathscr{P}`$ is discrete or continuous with a strictly positive density on its support, then Assumption 2 holds for any distribution sampled from $`\mathscr{P}`$.

> **Theorem (Test Domain Bound).** Given a test domain $`\mathbb{P}_{Q,T}^t`$ with the corresponding hallucinated-answer distribution $`\mathbb{P}_{Q,H}^t`$, suppose that Assumption 1 holds and that the product distribution $`\mathbb{P}_{Q,T}^t \otimes \mathbb{P}_{Q,H}^t`$ is regular, i.e., Assumption 2 holds for $`\mathbb{P}_{Q,T}^t \otimes \mathbb{P}_{Q,H}^t`$. Then, under the stated conditions, with probability at least $`1 - \big(1 - \mathscr{P}(\mathcal{N}^{t}_{\epsilon})\big)^M > 0`$, the following holds: for any $`e \in \{1,\ldots,M\}`$,
>
> $$\big|R_{d_{\ell}}^t - R_{d_{\ell}}^e\big| \le C\,\epsilon + \sqrt{2M}\,\beta_{d_{\ell}}$$
>
> where $`C>0`$ is a uniform constant.

**Proof.** According to Lemma 1 and Lemma 2, we can prove that:

$$
\begin{aligned}\big|R_{d_{\ell}}^t-R_{d_{\ell}}^e\big|\leq\sqrt{2M}\beta_{d_{\ell}}+ C \min_{e'\in \{1, \ldots, M\}}d_{\rm TV}(\mathbb{P}_{Q,T}^{t} \otimes \mathbb{P}_{Q,H}^{t},\mathbb{P}_{Q,T}^{e'} \otimes \mathbb{P}_{Q,H}^{e'})\end{aligned}
$$

It is easy to check that with the probability at least $`1 - \big(1 - \mathscr{P}(\mathcal{N}^{t}_{\epsilon})\big)^M > 0`$, there exists some $`e^*\in \{1,...,M\}`$ such that $`\mathbb{P}_{Q,T}^{e^*} \otimes \mathbb{P}_{Q,H}^{e^*}\in \mathcal{N}^{t}_{\epsilon}`$, which implies that:

$$
\begin{aligned}\min_{e'\in \{1, \ldots, M\}}d_{\rm TV}(\mathbb{P}_{Q,T}^{t} \otimes \mathbb{P}_{Q,H}^{t},\mathbb{P}_{Q,T}^{e'} \otimes \mathbb{P}_{Q,H}^{e'})&\leq d_{\rm TV}(\mathbb{P}_{Q,T}^{t} \otimes \mathbb{P}_{Q,H}^{t},\mathbb{P}_{Q,T}^{e^*} \otimes \mathbb{P}_{Q,H}^{e^*})\\&\leq \epsilon\end{aligned}
$$

This completes the proof.