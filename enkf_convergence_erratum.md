# Erratum: gap in the proof of EnKF convergence

**Concerns**

- M. D. Butala, J. Yun, Y. Chen, R. A. Frazin, F. Kamalabadi, "Asymptotic convergence of the ensemble Kalman filter," *ICIP 2008*, pp. 825--828. (thesis ref. [121])
- M. D. Butala, R. A. Frazin, Y. Chen, F. Kamalabadi, "Tomographic imaging of dynamic objects with the ensemble Kalman filter," *IEEE TIP* **18**(8):1573--1587, 2009. (thesis ref. [6] in Mandel et al.)
- M. D. Butala, PhD thesis, App. A (Convergence Proof), proof of Thm 4.1.

**Claimed by** J. Mandel, L. Cobb, J. D. Beezley, "On the convergence of the ensemble Kalman filter,"
*Appl. Math.* **56**(6):533--541, 2011 (arXiv:0901.2951v2, p. 2):

> "The proof of EnKF convergence in [6] has a gap; it assumes that certain covariances derived
> from the ensemble exist, which is not guaranteed without an L2 bound."

**Verdict.** The gap is real and Mandel's description is accurate. It is localized to the
cross-term argument in Step 2 of App. A, eqs. (A.13)--(A.18) (and the analogous step in
(A.22)). It is repaired by inserting one a priori moment-bound lemma (below). The
theorem statement survives unchanged; the proof can in fact be strengthened from
convergence in probability to $L^p$ convergence.

---

## 1. Notation (thesis App. A)

Ensemble size $L$, state dim $N$, time index $i$. Members $\hat x^{l}_{i|i-1}$, sample mean
$\hat x_{i|i-1}$, sample covariance $\hat P^{f}_{i|i-1}$ as in (4.17)--(4.20). Taper $C_i$, gain
$\hat K_i$ as in (4.19). Perturbed measurements $y^{l}_{i} = y_i + v^{l}_{i}$ with
$v^{l}_{i} \stackrel{\text{iid}}{\sim} N(0,R_i)$; process noise $u^{l}_{i} \stackrel{\text{iid}}{\sim} N(0,Q_i)$.
Anomalies $z^{l}_{i|i-1} = \hat x^{l}_{i|i-1} - \hat x_{i|i-1}$, so $\sum_{l=1}^{L} z^{l}_{i|i-1} = 0$.

Induction hypotheses used in Step 2:

$$
\hat x_{i|i-1} \xrightarrow{p} x^{\infty}_{i|i-1}, \qquad
\hat P^{f}_{i|i-1} \xrightarrow{p} P^{\infty}_{i|i-1}. \tag{A.4--A.5}
$$

## 2. The gap

The posterior sample covariance (A.10) contains the cross terms

$$
\frac{1}{L-1}\sum_{l=1}^{L} z^{l}_{i|i-1}\,(w^{l}_{i})^{T},
\qquad w^{l}_{i} := y^{l}_{i} - \bar y_{i},
\tag{A.12}
$$

which must be shown to converge in probability to $0$. Componentwise, (A.13)--(A.15) reduce
this to the average of

$$
W_{L,l} \;:=\; z^{l}_{i|i-1,m}\, v^{l}_{i,n}.
$$

Two steps of the argument presuppose that these ensemble-derived random variables have
finite, $L$-uniform second moments:

1. **(A.17)** asserts
 $\operatorname{Cov}\!\big(z^{k}_{i|i-1,m} v^{k}_{i,n},\, z^{l}_{i|i-1,m} v^{l}_{i,n}\big) = 0$ for $k \neq l$.
 A *covariance* is being computed; it exists only if $W_{L,k} W_{L,l}$ and $W_{L,l}^2$ are
 integrable. This is exactly Mandel's "certain covariances derived from the ensemble."

2. **(A.18)** invokes the $L^2$ weak law of large numbers (Durrett, *Probability: Theory and
 Examples*, 2nd ed., p. 36) on $\sum_l W_{L,l}$. That result requires
 $\operatorname{Var}(W_{L,l}) \le C$ **uniformly in $L$**, so that
 $\operatorname{Var}\big((L-1)^{-1}\sum_l W_{L,l}\big) \le C/(L-1) \to 0$.

Neither is established anywhere. The induction hypotheses (A.4)--(A.5) assert only
*convergence in probability* of the sample mean and sample covariance. That places no
constraint on the moments of the members: $\hat P^{f}_{i|i-1}$ is a.s. finite for every finite $L$
under no moment assumption at all, and (A.4)--(A.5) are compatible with
$\sup_L \mathbb{E}\big[(z^{L}_{i|i-1,m})^{2}\big] = \infty$. In that case
$\mathbb{E}[W_{L,l}^2] = \mathbb{E}[(z^{l}_m)^2]\,\mathbb{E}[(v^{l}_n)^2]$ can be infinite or grow with $L$,
(A.17) is meaningless, the $L^2$ WLLN does not apply, and (A.12) is not shown to vanish.

The same omission propagates to the cross terms of (A.22) in Step 3 ("by a similar
argument used to show that (A.12) converges in probability to the matrix 0").

**Secondary defect (not the gap).** (A.16) cites "the WLLN" for
$(L-1)^{-1}\sum_l z^{l}_{i|i-1,m}[y_i]_n \to 0$. The summands are *not* i.i.d. (each $z^{l}$
contains the shared sample mean $\hat x_{i|i-1}$), so the i.i.d. WLLN does not apply as cited,
and the displayed LHS of (A.16) writes $[y^{l}_i]_n$ where the argument requires $[y_i]_n$.
The conclusion is nonetheless correct and needs no LLN: $\sum_l z^{l}_{i|i-1} = 0$ implies
$\sum_l z^{l}_{i|i-1,m}[y_i]_n = 0$ *identically*.

## 3. Repair

### Lemma 1 (a priori $L^p$ bound; = Mandel--Cobb--Beezley Lemma 5, extended to tapers)

*For every time index $i$ and every $p \in [1,\infty)$ there is a constant $c(i,p) < \infty$,
independent of $L$, such that for all $L \ge 2$ and all $l$*

$$
\|\hat x^{l}_{i|i-1}\|_p,\; \|\hat x^{l}_{i|i}\|_p,\; \|\hat P^{f}_{i|i-1}\|_p,\; \|\hat K_{i}\|_p
\;\le\; c(i,p).
$$

*Proof.* Induction on $i$. The initial ensemble (4.18) is Gaussian, so all moments exist
with bounds independent of $L$. Assume the bound at $i-1$.

- *Forecast (4.22):* $\|\hat x^{l}_{i|i-1}\|_p = \|F_{i-1}\hat x^{l}_{i-1|i-1} + u^{l}_{i-1}\|_p \le \|F_{i-1}\|\,c(i{-}1,p) + \|u\|_p$.
- *Mean:* $\|\hat x_{i|i-1}\|_p \le L^{-1}\sum_l \|\hat x^{l}_{i|i-1}\|_p \le c$ by Jensen.
- *Anomalies and sample covariance:* $\|z^{l}\|_p \le 2c$, and by the Cauchy--Schwarz
 bound $\|WZ\|_p \le \|W\|_{2p}\|Z\|_{2p}$ (Mandel (4.1)),
 $\|\hat P^{f}_{i|i-1}\|_p \le (L-1)^{-1}\sum_l \|z^{l}(z^{l})^{T}\|_p \le \|z^{1}\|_{2p}^{2} \le c'$.
- *Taper:* $P \mapsto C_i \circ P$ is linear and bounded in any matrix norm (tapers obey
 $|[C_i]_{jk}| \le 1$; more generally $\|C \circ P\| \le \|C\|_{\max}\,\|P\|$), so
 $\|C_i \circ \hat P^{f}_{i|i-1}\|_p \le c''$.
- *Gain:* $H_i(C_i\circ\hat P^{f})H_i^{T} + R_i \succeq R_i \succ 0$, hence
 $\big\|\big(H_i(C_i\circ\hat P^{f})H_i^{T}+R_i\big)^{-1}\big\| \le \|R_i^{-1}\|$ **deterministically**
 (no inverse moments are needed), and
 $\|\hat K_i\|_p \le c''\,\|H_i^{T}\|\,\|R_i^{-1}\|$.
- *Analysis (4.21):* with $y^{l}_i = y_i + v^{l}_i$,
 $\|\hat x^{l}_{i|i}\|_p \le \|\hat x^{l}_{i|i-1}\|_p + \|\hat K_i y^{l}_i\|_p + \|\hat K_i H_i \hat x^{l}_{i|i-1}\|_p \le c(i,p)$
 by (4.1) and the induction hypothesis (which supplies every $2p$ as well). $\square$

Note the two load-bearing facts: $R_i \succ 0$ makes the random inverse uniformly bounded,
and the hypothesis "for all $p$ at once" absorbs the $p \mapsto 2p$ losses of Cauchy--Schwarz.

### Lemma 2 (the missing step)

*Fix $i$ and let $W_{L,l} = z^{l}_{i|i-1,m} v^{l}_{i,n}$. Then $W_{L,l} \in L^2$ with*
$\sup_{L,l} \mathbb{E}[W_{L,l}^{2}] \le C(i) < \infty$, *and $\{W_{L,l}\}_{l=1}^{L}$ is pairwise
uncorrelated with mean zero. Consequently $(L-1)^{-1}\sum_{l} W_{L,l} \to 0$ in $L^2$ and
in probability.*

*Proof.* $z^{l}_{i|i-1}$ is measurable w.r.t. the $\sigma$-algebra generated by the forecast
ensemble, and $v^{l}_{i}$ w.r.t. that generated by $\{v^{k}_{i}\}_k$; these are independent
(the initial ensemble, $\{u^{l}\}$, $\{v^{l}\}$ are mutually independent). Hence by Lemma 1

$$
\mathbb{E}[W_{L,l}^{2}] = \mathbb{E}\big[(z^{l}_{i|i-1,m})^{2}\big]\,\mathbb{E}\big[(v^{l}_{i,n})^{2}\big]
\le c(i,2)^{2}\,[R_i]_{nn} =: C(i),
$$

which is the integrability that (A.17) requires. For $k \neq l$,
$\mathbb{E}[W_{L,k}W_{L,l}] = \mathbb{E}[z^{k}z^{l}]\,\mathbb{E}[v^{k}v^{l}] = 0$ since the $v^{l}$ are
i.i.d. mean zero, and $\mathbb{E}[W_{L,l}]=0$. The $L^2$ WLLN (Durrett, p. 36) now applies:
$\operatorname{Var}\big((L-1)^{-1}\sum_l W_{L,l}\big) \le C(i)/(L-1) \to 0$. $\square$

### Completion of the proof of Theorem 4.1

- **Step 1** is unchanged: (4.18) is Gaussian, so the i.i.d. WLLN applies to the sample mean
 and (entrywise, via $x_jx_k \in L^1$) to the sample covariance.
- **Step 2**, eqs. (A.12)--(A.18): replace (A.17)--(A.18) by Lemma 2. The cross terms
 (A.12) then converge in probability to $0$; the first term of (A.10) converges by (A.5) and
 continuity of the Hadamard product; the fourth by the continuous mapping theorem and
 $R_i \succ 0$. Eq. (A.19) follows. For (A.16), delete the WLLN citation and use
 $\sum_l z^{l}_{i|i-1}=0$ as noted in §2.
- **Step 3**, eq. (A.22): the cross terms in
 $\sum_l z^{l}_{i|i}(u^{l}_i - \bar u_i)^{T}$ and its transpose vanish by the *same* argument
 as Lemma 2 ($z^{l}_{i|i} \perp u^{l}_i$, $\sup_L\|z^{l}_{i|i}\|_2 < \infty$ by Lemma 1). The
 remaining terms converge by (A.19) and the i.i.d. WLLN on $\{u^{l}_i\}$. Eqs. (A.21), (A.23)
 follow, closing the induction.

### Strengthening

Lemma 1 gives uniform $L^p$ bounds on the members, hence uniform integrability. As in
Mandel--Cobb--Beezley Thm 1, convergence in probability may then be upgraded to

$$
\hat x^{l}_{i|i} \to x^{\infty}_{i|i} \quad \text{in } L^{q} \text{ for every } q < p,
\qquad \hat x_{i|i},\ \hat P^{f}_{i|i} \to x^{\infty}_{i|i},\ P^{\infty}_{i|i} \text{ in } L^{q},
$$

at each fixed time index $i$. Corollary 4.2 (unlocalised EnKF $\to$ KF/LMMSE) is unaffected.

## 4. Remarks

1. **Scope of Mandel--Cobb--Beezley.** Their paper treats only the *unlocalised stochastic*
 EnKF (Burgers--van Leeuwen--Evensen); they state explicitly that square-root variants "are
 not the subject of this paper." Theorem 4.1's tapered limit (EnKF $\to$ LKF, and hence the
 quantification of *localization bias* via KF $\ne$ LKF) is outside their scope. The repair
 above is written for the tapered case and covers that content.

2. **Nature of the gap.** It is a missing hypothesis, not a false conclusion. No step of the
 argument is wrong once $L$-uniform second moments are available; the fix is a one-lemma
 insertion, not a restructuring.

3. **Horizon.** Theorem 4.1 is per fixed time index $i$ (finite horizon, induction on $i$).
 Time-uniform / infinite-horizon accuracy is a separate question; see Cheng, Sanz-Alonso &
 Waniorek, arXiv:2609.23927 (2026) for deterministic square-root EnKFs, which also
 separates sampling error from localization bias.
