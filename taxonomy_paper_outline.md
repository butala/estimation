# Outline — "What does your smoother need you to invert?"

A taxonomy of Kalman and ensemble smoothers, organized by invertibility.
Working document; target a first draft in 3--4 weeks of writing.

---

## 0. Pitch

**Question the reader actually has:** *I need to smooth. Which recursion do I use, and
what is going to break?*

**Thesis:** the choice of smoothing recursion is dictated almost entirely by **which
matrix you are required to invert** — and the three classical recursions fail in three
different, complementary regimes. Organizing the comparison along that axis yields

1. a near-decision procedure (a table plus a flowchart),
2. an explanation of why each community settled on the recursion it did, and
3. an explanation of why the same "no inversion" property keeps getting rediscovered
   (three times in 2026 alone, in literatures with disjoint bibliographies).

### Title candidates

1. **What Does Your Smoother Need You to Invert? A Taxonomy of Kalman and Ensemble Smoothers**
   — the question, which is also the search query. Preferred.
2. *Invertibility Constraints in Kalman and Ensemble Smoothers: a Taxonomy, a Derivation, and a Recommendation*
   — drier, more formal venue-friendly.
3. *Smoothing Without Inversion: the Bryson--Frazier Route and its Ensemble Form*
   — narrower; leads with the answer rather than the question. Better as a companion note.

### Abstract sketch (150 words)

> Fixed-interval smoothing for linear--Gaussian state-space models admits three
> classical recursions — Rauch--Tung--Striebel (RTS), Mayne--Fraser--Potter (MFP) and
> Bryson--Frazier (BF) — and a growing family of ensemble approximations. The
> recursions are usually presented as interchangeable, and each community has adopted
> one as default. They are not interchangeable: each must invert a different matrix, and
> each is therefore *unevaluable* in a different regime. RTS and the ensemble
> smoothers built on it require the inverse of the forecast (co)variance and fail when
> the representation is rank-deficient; MFP requires the inverse of the dynamics and
> fails for dissipative physics; BF requires only the inverse of the innovation
> covariance, which is nonsingular by construction. We derive all three from a single
> constrained least-squares problem, which makes the constraints transparent; tabulate
> the ensemble descendants along the same axes; and demonstrate two regimes — ensemble
> size below state dimension, and non-invertible dynamics — in which the standard
> alternatives cannot be evaluated at all. Software and closed-form validation accompany
> the paper.

### Audience & venue

Primary audience: data assimilation and state-estimation practitioners and method
developers — the people who have implemented a lag-one ensemble Kalman smoother and hit
the \(P^{-1}\) wall. Secondary: the Gaussian-process / probabilistic-numerics and
continuous-time communities, where the same property is being rediscovered (Section 6).

**Target: *Nonlinear Processes in Geophysics*.** Open access, method + theory + numerics
is exactly its shape, and it is where the DA community looks. Alternates: *QJRMS* (more
applied), *SIAM/ASA JUQ* (if we want to lean harder on the derivation).

**Companion, later:** a short *SIAM Review* "Features" / *BAMS* piece — one table, one
worked example, the flowchart. That is the visibility vehicle; this paper is the archival
one.

---

## 1. Contributions (what is new here, honestly)

| # | Contribution | Novelty |
|---|---|---|
| C1 | **The invertibility taxonomy** (Tables 1--2) | Synthesis, but not available anywhere; the citable object |
| C2 | **One derivation from which all three recursions fall out** — the constrained least-squares / Hamiltonian two-point boundary-value problem, with *sweep* \(\to\) BF and *Riccati transformation* \(\to\) RTS | Novel as an *explanatory* device: it makes the invertibility requirements visible as artifacts of the solution method, not of the problem |
| C3 | **Two discriminating experiments** in which a competitor is not merely worse but **unevaluable**: \(L<N\) (rank-deficient forecast covariance) and non-invertible \(F\) | Novel; the literature compares accuracy, not well-posedness |
| C4 | **Consolidation of three literatures** with disjoint bibliographies (DA; GP regression / probabilistic numerics; continuous-time stochastic analysis) | Synthesis; the "why now" |
| C5 | **Reproducible software with closed-form validation** | Enabling |

Be explicit in the paper that C1, C4 are synthesis and C2, C3 are the new work. A
taxonomy paper loses credibility if it oversells.

---

## 2. Section-by-section

### 1. Introduction
- The folklore problem: RTS is the default in GP regression and control (Särkkä's text);
  the lag-one cross-covariance smoother is the default in data assimilation
  (van Leeuwen--Evensen); MFP is the default in two-filter derivations. Nobody compares
  them on the axis that decides the choice.
- **The claim:** the recursions are three solution methods for *one* optimization problem,
  and their constraints are artifacts of the solution method. State the three
  constraints as a teaser:
  - RTS: \(P_{i|i-1}^{-1}\) — \(N\times N\), singular whenever the forecast is
    rank-deficient (e.g. an ensemble with \(L-1<N\))
  - MFP: \(F_i^{-1}\) — false for most dissipative physics
  - BF: \(R_{e,i}^{-1}\) — \(M\times M\), \(\succeq R_i \succ 0\) by construction
- Contributions (C1--C5) as a bulleted list.
- Roadmap.

### 2. The smoothing problem and one derivation of three recursions
*Claim: all three are the same two-point boundary-value problem, solved differently; the
constraints are visible in the solution method.*

- **2.1** Model and target: \(x_{i+1}=F_ix_i+u_i,\ y_i=H_ix_i+v_i\), and the smoothing
  distribution's first two moments.
- **2.2** Smoothing as constrained least squares (the cost function already in the
  author's notes: \(\|x_1\|_{\Pi_1^{-1}}^2+\sum\|y_i-H_ix_i\|_{R_i^{-1}}^2+\sum\|u_i\|_{Q_i^{-1}}^2\)
  subject to \(x_{i+1}=F_ix_i+u_i\)). Introduce Lagrange multipliers \(\lambda_i\) →
  the **Hamiltonian two-point boundary-value problem**. *(This answers the open
  question in the author's note about what the adjoint is: \(\lambda_i\) is the Lagrange
  multiplier, and the dual problem is the smoothing cost.)*
- **2.3** Two ways to solve the same BVP:
  - a *sweep* (forward--backward elimination on the Hamiltonian system) → **Bryson--Frazier**
  - a *Riccati transformation* (solve for the covariance recursion first) → **RTS**
  - and the *two-filter / combine* route → **Mayne--Fraser--Potter**
  - Where \(P^{-1}\), \(F^{-1}\), \(R_e^{-1}\) enter each, and why.
- **2.4 Table 1** — the exact smoothers.

| | inverts | size | fails when | passes | stores |
|---|---|---|---|---|---|
| RTS | \(P_{i\mid i-1}\) | \(N^3\) | forecast covariance singular/ill-conditioned | \(O(N^3)\) or \(O(N^2M)\) | \(\hat x_{i\mid i},P_{i\mid i}\) |
| MFP | \(F_i\) | \(N^3\) | \(F\) not invertible (dissipative models) | \(O(N^3)\) | two filter passes |
| BF | \(R_{e,i}\) | \(M^3\) | never (\(R_{e,i}\succeq R_i\succ0\)) | \(O(M^3+N^2)\) | \(\hat x_{i\mid i},P_{i\mid i},K_i,e_i,R_{e,i}\) |

*Optional appendix:* the continuous-time analogue (Kalman--Bucy / RTS / BF), for
completeness and to make contact with Kurisaki (2026).

### 3. Ensemble descendants
*Claim: the ensemble methods inherit the constraint of the exact recursion they
approximate, and the ones that escape it pay in sweeps or in what they return.*

- **3.1** Why \(P^{-1}\) gets *worse* in the ensemble setting: \(P^f = X X^\top/(L-1)\)
  has rank \(\le L-1\), so the inverse does not merely become ill-conditioned — it does
  not exist whenever \(L-1<N\). This is the crux and is worth stating as a proposition.
- **3.2** The lag-one cross-covariance smoother (van Leeuwen & Evensen 1996; Evensen &
  van Leeuwen 2000): \(x^s_i = x^a_i + C_{i,i+1}(P^f_{i+1})^{-1}(x^s_{i+1}-x^f_{i+1})\).
  Inherits the RTS constraint. Options at \(L<N\): pseudo-inverse (ad hoc), square-root
  forms (extra cost), or — see 3.4.
- **3.3** The iterative family (ES-MDA; IEnKS; SIEnKS): escape inversion by **iterating**
  a variational/optimization formulation. Cost: multiple forward sweeps. Return: a
  minimizer of a cost, not the moments of the smoothing distribution. (That is a
  legitimate trade, and the paper should say so rather than dismiss it.)
- **3.4** **The ensemble Bryson--Frazier smoother / LEnKS** (Butala, Fathpour & Bhatt
  2012): the Monte Carlo form of §2's BF recursion — an *adjoint ensemble*
  \(\tilde\Lambda_i\) with \(\mathbb E[\tilde\Lambda_i\tilde\Lambda_i^\top/(L-1)]=\Lambda_i\)
  — plus covariance tapering in the gain and in stage 3, plus a convergence theorem
  (EnKS \(\to\) LKS as \(L\to\infty\)) **including the smoothed error covariance**.
  No \(P^{-1}\), no \(F^{-1}\), one forward and one backward pass, no adjoint physics
  model. Derivation of the ensemble adjoint recursion as the unbiased-moment completion
  of the exact \(\Lambda\) recursion.
- **3.5 Table 2** — the taxonomy. Columns: inverts · requires \(L>N\)? · requires
  \(F^{-1}\)? · sweeps · adjoint/tangent-linear physics? · localization? · returns
  smoothing moments or a minimizer? · convergence theory incl. smoothed covariance?

This table is the paper's product. It should fit on one page and be photocopiable.

### 4. Two regimes where the alternatives are not worse but unevaluable
*Claim: the constraints are not academic — they decide whether a method can be run at all.*

Worked on the software in §7. **The validation harness comes first**: §4.0 confirms every
implementation reproduces the closed-form Bayes posterior (\(\Lambda^{-1}b\), \(\Lambda^{-1}\)
for a scalar random walk with a path-Laplacian precision) to machine precision. That
establishes the implementations are right before we compare them.

- **4.1 Example A — \(L-1<N\).** Lorenz (1996), \(N=40\), \(L=20\) and \(45\).
  - the lag-one EKS is **not evaluable** (\((P^f_{i+1})^{-1}\) is singular);
  - LEnKS returns an estimate, and the taper measurably helps (our measurement: RMSE
    1.42 vs 2.97 at \(L=45\));
  - quantify the truncation an \(L\)-member ensemble must perform (rank of the analysis
    covariance is \((L-1)+M\) with a taper, \(\le L-1\) without — measured 0.0000 vs 0.0938
    relative mismatch against the tapered Kalman target at \(L=45\) vs \(L=20\)).
- **4.2 Example B — non-invertible \(F\).** A deliberately dissipative model (e.g.
  \(F=\mathrm{diag}(1,0)\), or a two-mode advection--diffusion step with heavy damping).
  - MFP is **not evaluable**;
  - RTS and BF agree with the closed-form posterior.
- **4.3 Example C — all methods defined.** The random walk of §4.0: all agree with each
  other and with Bayes' rule. *(This is the "no method is being straw-manned" example.)*

**Figure 1:** error of each method across a sweep of \(L/N\) and of \(\|F\|\)-conditioning,
with the "unevaluable" region shaded. This is the picture people will remember.

### 5. A decision procedure
*Claim: the table can be turned into a flowchart that is nearly deterministic.*

- **Figure 2 (flowchart):**
  1. Can you form and invert the forecast covariance? (i.e. is \(P\) explicit and nonsingular?)
     → RTS (simplest) or BF (cheaper, more stable).
  2. If \(P\) is only available through an ensemble: is \(L-1 \ge N\)?
     - yes, and few sweeps acceptable → lag-one EKS or ES-MDA;
     - **no** → **LEnKS** (or a pseudo-inverse variant, flagged as ad hoc).
  3. Is \(F\) invertible? if not, MFP is out regardless.
  4. Is the problem high-dimensional with short correlation length? → insist on a
     localized form (LEnKS), i.e. a form with tapering in the gain *and* the third stage.
  5. Do you need the smoothing *moments* (uncertainty quantification) or only a
     minimizer (history matching)? — decides BF-family vs the iterative smoothers.
- Practical notes: the third-stage tapers \(C'_i, C''_i\); the cost comparison
  (one forward + one backward vs \(N_{\rm iter}\) forward sweeps); memory (what must be
  written to disk in stage 1).

### 6. Why this is being rediscovered: three literatures, one property
*Claim: the same advantage is being found independently because each field hit its own
version of the wall and none of them can see the others' literature.*

- **Data assimilation** (2012): the wall is \(L\ll N\) and huge \(N\). Result: LEnKS.
- **GP regression / probabilistic numerics** (2026, Colemont et al.): the state-space GP
  trick (Hartikainen--Särkkä 2010) made RTS the default; pushing accuracy makes the
  posterior variance small and \(P^-\) ill-conditioned. Result: the *modified* BF
  (Bierman 1973) as a robust RTS alternative, plus **hyperparameter gradients** from the
  smoother's intermediates. Bibliography contains no DA-side work (checked).
- **Continuous-time stochastic analysis** (2026, Kurisaki): the objection to BF is that
  it had no derivation avoiding invertibility. Result: an **Ornstein--Uhlenbeck pathwise
  representation**, a first rigorous continuous-time derivation, and **pathwise
  sampling** of the smoothing distribution.
- **The upshot:** complementary, not competing. Table 3 = the union of the three
  contributions along the taxonomy's columns. What none of the 2026 work has:
  localization, the ensemble form, a convergence theorem with the smoothed covariance.
  What this paper does not have: hyperparameter gradients, pathwise sampling, a
  continuous-time derivation — each named as future work (§8).
- This section is short (2--3 pages) but it is the reason the paper will be read now.

### 7. Software and reproducibility
- Everything in §4 is a test in the `estimation` library: the closed-form random-walk
  posterior is the reference, and the Lorenz-96 comparison is a script.
- State the validation results concretely (machine-precision agreement for the exact BF
  form; \(1/\sqrt L\) convergence of the ensemble form).
- Link the repository; note the tests as the paper's reproducibility artifact.

### 8. Discussion, limitations, open problems
- **Nonlinear dynamics.** All of the above is linear--Gaussian. The ensemble/iterative
  methods handle nonlinearity by running the model on members; the smooth-optimization
  view (Bocquet) is the alternative. Say plainly what §2--§5 do and do not settle.
- **The dual question, now partly answered.** \(\lambda_i\) is the Lagrange multiplier of
  the smoothing cost (§2.2), which means the BF adjoint is *already* gradient
  information. Conjecture/open problem: the LEnKS adjoint ensemble should yield
  derivatives of the smoothing cost w.r.t. parameters — the data-assimilation analogue of
  Colemont et al.'s hyperparameter gradients. This is the natural next paper.
- **Continuous-time form** of the ensemble BF smoother (open; Kurisaki's is deterministic).
- **Localization bias** as a quantifiable object (the \(\|\mathrm{LEnKS}-\mathrm{LKS}\|\)
  term) — connect to the author's broader convergence program.

### Appendix A
The Hamiltonian / two-point BVP derivation in full (already in the author's notes).

### Appendix B
The closed-form random-walk posterior: \(\Lambda = T/q + I/r + e_0e_0^\top/P_0\) with the
path Laplacian \(T\); smoothed mean \(\Lambda^{-1}b\), covariance \(\Lambda^{-1}\).

### Appendix C
The ensemble adjoint recursion and the moment identity
\(\mathbb E[\tilde\Lambda_i\tilde\Lambda_i^\top/(L-1)]=\Lambda_i\) (including the
\(H_i^\top\) on the noise term, which is forced by that identity and which the journal
form of the recursion does not display).

---

## 3. Tables and figures (the deliverable inventory)

| Item | Content | Status |
|---|---|---|
| **Table 1** | exact smoothers: inverts · size · failure mode · cost · storage | needs writing (from §2.3) |
| **Table 2** | **the taxonomy** — ensemble methods along the same axes + sweeps, localization, moments-or-minimizer, convergence theory | needs writing; this is the paper |
| Table 3 | the three 2026/2012 contributions side by side | partially drafted (above) |
| **Figure 1** | error / well-posedness over a sweep of \(L/N\) and \(F\)-conditioning, "unevaluable" region shaded | needs the sweep script |
| **Figure 2** | the decision flowchart | needs drawing |
| Figure 3 | Lorenz-96: filtered RMSE with and without taper at \(L=45\) | **exists** (`tests/test_lorenz_96.py --figures`) |
| Figure 4 | random walk: filter vs smoother vs joint posterior, \(\pm2\sigma\) | **exists** (`tests/test_linear_gaussian_smoother.py --figures`) |

---

## 4. What already exists vs. what must be written

**Already exists** (in `estimation`, and validated):
- `KF`, `SquareRootKF`, `UDKF` — the exact filter, three representations
- `rts_smooth` — the exact reference
- `EnKS`/`LEKS` — the ensemble BF smoother, with the adjoint ensemble
- `EnKF`/`EnSRF`/`EAKF`/`ETKF`/`LETKF` — the forward passes
- **The closed-form random-walk validation** (21 checks) — this is §4.0
- **The Lorenz-96 localization demonstration** (taper halves the RMSE at \(L=45\))
- Figures 3--4

**Must be written:**
1. §2's derivation written up as prose (it exists in the author's notes as formulas +
   open questions — those become the text)
2. The MFP implementation (a few lines: a second filter pass and a combine step) and
   Example B's non-invertible-\(F\) problem
3. The lag-one cross-covariance smoother implementation, to be shown *unevaluable* at
   \(L<N\)
4. The Figure 1 sweep script
5. Figure 2 (flowchart) — a drawing tool, not code
6. §1, §5, §6, §8 prose

---

## 5. Effort

| | |
|---|---|
| §2 derivation write-up | ~1 week (the math is in hand) |
| §3 taxonomy + Table 2 | ~3--4 days |
| §4 numerics (two new implementations + two sweeps + figures) | ~1--1.5 weeks |
| §1, §5, §6, §8 prose + Figure 2 | ~1 week |
| **Total drafting** | **~3--4 weeks** |
| Internal review + revision | ~1--2 weeks |
| **Companion short piece** (SIAM Review / BAMS) | +3--4 days once the paper exists |

Roughly a two-month calendar to submission if it is the main thing; longer alongside
other work.

---

## 6. Anticipated review objections, and the answers

| Objection | Answer |
|---|---|
| "This is a survey." | C2 and C3 are not. The one-derivation viewpoint makes the constraints *visible as artifacts of the solution method*, and §4 compares well-posedness, not accuracy — a distinction the literature has not drawn. Say this in the introduction. |
| "BF is classical (Bryson & Ho; Bierman 1973)." | Correct, and stated. The contribution is the *ensemble form with localization and a convergence theorem* (LEnKS 2012), the taxonomy, and the discriminating experiments. Cite Bierman and Bryson--Ho as the ancestors. |
| "The iterative smoothers (ES-MDA/IEnKS) are what people use." | Include them fairly in Table 2 as a different trade (sweeps for robustness to nonlinearity; a minimizer rather than moments). §5's flowchart routes to them explicitly when that is what the user wants. |
| "Why not just use a pseudo-inverse of \(P^f\)?" | Mention it in §3.2 as the standard workaround, and note it is ad hoc: the result depends on the truncation threshold, and it silently changes the estimator. Contrast with the BF form, which needs no such choice. |
| "Only linear--Gaussian." | §8, plainly. Note that the same critique applies to every exact smoother in §2, and that the ensemble/iterative family's claim to nonlinearity is a different mechanism (running the model on members), which §3.3 states. |
| "The 2026 papers cover this." | §6. They cover GP conditioning and continuous-time foundations; neither has an ensemble form, localization, or a convergence theorem, and neither cites the DA literature. The taxonomy is what makes the three visible together. |

---

## 7. One paragraph to get right first

The single most quotable claim, worth writing before anything else:

> The three classical smoothing recursions are not interchangeable implementations of
> one algorithm; they are three solution methods for one two-point boundary-value
> problem, and each must invert a different matrix to solve it. In the ensemble setting
> the forecast-covariance inverse does not merely become ill-conditioned — it does not
> exist. Choosing a smoother is therefore not a matter of accuracy but of **which
> constraint you can satisfy**: the size of the ensemble relative to the state, the
> invertibility of the dynamics, and whether you need the moments of the smoothing
> distribution or only a minimizer.
