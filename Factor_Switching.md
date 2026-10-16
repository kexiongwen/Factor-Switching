# Factor Switching

Rotation, sign and permutation post-processing of factor loadings.

[TOC]

---

## 1. Factor Model

Notation and model setting used throughout.

### 1.1 Model setting

Let $Y_i=(Y_{i1},\ldots,Y_{ip})^\top$ denote the $i$-th observation of a random sample of $p$-dimensional responses, $Y_i\in\mathbb{R}^{p}$, $i=1,\ldots,n$. Let $\mathcal{N}_p(\mu,\boldsymbol{\Sigma})$ denote the $p$-dimensional normal distribution with mean $\mu=(\mu_1,\ldots,\mu_p)\in\mathbb{R}^p$ and covariance matrix $\boldsymbol{\Sigma}$, and let $\mathbf{I}_p$ denote the $p\times p$ identity matrix.

In the typical Factor Analysis (FA) model — also referred to as the *strict factor structure* — $Y_i$ is expressed as a linear combination of a latent vector of factors $F_i\in\mathbb{R}^{q}$:

$$
Y_i = \mu + \boldsymbol{\Lambda} F_i + \varepsilon_i,\quad i = 1,\ldots,n,
\tag{1}
$$

where $q>0$ denotes the number of factors, which is assumed fixed and $q<p$. The $p\times q$ matrix $\boldsymbol{\Lambda}=(\lambda_{rj})$ contains the factor loadings, while $\mu=(\mu_1,\ldots,\mu_p)$ contains the marginal mean of $Y_i$. The unobserved vector of factors $F_i=(F_{i1},\ldots,F_{iq})^\top$ lies on a lower-dimensional space, that is, $q<p$, and consists of uncorrelated and homoscedastic features:

$$
F_i \sim \mathcal{N}_q(0_q,\mathbf{I}_q),
\tag{2}
$$

independent for $i=1,\ldots,n$, where $0_q:=(0,\ldots,0)^\top$ and $\mathbf{I}_q$ denotes the $q\times q$ identity matrix.

The terms $\varepsilon_i$ (commonly referred to as *uniquenesses* or *idiosyncratic errors*) are independent of $F_i$, so $\mathrm{Cov}(F_{ij},\varepsilon_{ik})=0$ for all $j=1,\ldots,q$ and $k=1,\ldots,p$. They are normally distributed:

$$
\varepsilon_i\sim \mathcal{N}_p(0_p,\boldsymbol{\Sigma}),
\tag{3}
$$

independent for $i=1,\ldots,n$. Furthermore, $\varepsilon_i$ consists of independent random variables $\varepsilon_{i1},\ldots,\varepsilon_{ip}$, that is,

$$
\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1^2,\ldots,\sigma_p^2).
\tag{4}
$$

The distribution of $Y_i$ conditional on $F_i$ is

$$
Y_i \mid F_i \sim \mathcal{N}_p(\mu+\boldsymbol{\Lambda} F_i,\boldsymbol{\Sigma}),
\quad \text{independent for } i=1,\ldots,n,
\tag{5}
$$

with corresponding marginal distribution

$$
Y_i \sim \mathcal{N}_p(\mu,\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}),
\quad \text{iid for } i=1,\ldots,n.
\tag{6}
$$

Without loss of generality we may assume $\mu=0_p$, although this is not a necessary assumption.

### 1.2 Assumptions at a glance

1. **Latent linear structure (1).** Each $p$-dimensional response is a linear function of $q<p$ latent factors plus additive error, $Y_i=\mu+\boldsymbol{\Lambda}F_i+\varepsilon_i$; $q$ is treated as fixed and known.
2. **Factor distribution (2).** $F_i\sim\mathcal{N}_q(0_q,\mathbf{I}_q)$, drawn independently across observations: factors are uncorrelated, homoscedastic and standardized (zero mean, unit variance).
3. **Error distribution (3).** $\varepsilon_i\sim\mathcal{N}_p(0_p,\boldsymbol{\Sigma})$, independent across $i$ and independent of the factors.
4. **Diagonal error covariance (4).** $\boldsymbol{\Sigma}=\mathrm{diag}(\sigma_1^2,\ldots,\sigma_p^2)$: uniquenesses are mutually independent, which is what makes this a *strict* factor structure.
5. **Conditional law (5).** Given $F_i$, $Y_i$ is normal with mean $\mu+\boldsymbol{\Lambda}F_i$ and covariance $\boldsymbol{\Sigma}$, independently across $i$.
6. **Marginal law (6).** Integrating out $F_i$ gives $Y_i\sim\mathcal{N}_p(\mu,\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma})$ iid, hence $\mathrm{Cov}(Y_i)=\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}$: the latent factors are the only source of correlation among the measurements, the defining characteristic of factor analytic models.

---

## 2. Identifiability Problems

Two sources of identifiability problems affect the model of Section 1: one for $\boldsymbol{\Sigma}$ (the *uniqueness problem*, Section 2.1) and one for the factor loadings $\boldsymbol{\Lambda}$ (orthogonal invariance, Section 2.2).

### 2.1 Uniqueness problem: identifiability of $\boldsymbol{\Sigma}$

Given that there are $q$ factors, the number of free parameters in the covariance matrix $\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}$ is $p+pq-\tfrac{1}{2}q(q-1)$, while the number of free parameters in the unconstrained covariance matrix of $Y_i$ is $\tfrac{1}{2}p(p+1)$. Hence, if a strict factor structure is present in the data, the number of parameters in the covariance matrix is reduced by

$$
\frac{1}{2}p(p+1)-\left[p+pq-\frac{1}{2}q(q-1)\right]
=\frac{1}{2}\left[(p-q)^2-(p+q)\right].
\tag{7}
$$

The last expression is positive if $q<\phi(p)$, where

$$
\phi(p):=\frac{2p+1-\sqrt{8p+1}}{2},
\tag{8}
$$

a quantity known as the Ledermann bound. Under the usual genericity and rank conditions, $q<\phi(p)$ suffices for $\boldsymbol{\Sigma}$ to be almost surely unique. We assume that the number of latent factors does not exceed $\phi(p)$; the equality $q=\phi(p)$ is a boundary case not covered by this strict-inequality sufficient condition.

**Isotropic error model.** Non-identifiability does not necessarily take place when $q>\phi(p)$. A special case of the general FA model assumes an isotropic error model, that is, $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}_p$ in Equation (4), resulting in the probabilistic principal component analysis framework. In such a case, the number of factors can be as large as $p-1$.

**Heywood cases.** A related issue in estimating $\boldsymbol{\Sigma}$ is the phenomenon of so-called Heywood cases. Unconstrained likelihood optimization may produce negative estimates of one or more idiosyncratic variances, while optimization constrained to nonnegative variances may yield a boundary estimate of zero. An inverse-gamma prior with positive shape and scale parameters assigns support on $(0,\infty)$, so a proper posterior under this prior excludes nonpositive variances. Nevertheless, the posterior may have substantial mass or a mode near zero. A zero residual variance, if permitted as a boundary case of the model, would mean that the variation of the manifest variable in question was wholly explained by the latent factors.

Adjusting inverse-gamma shape and scale parameters can suppress posterior mass near zero, but does not impose a strictly positive lower bound: the prior density remains positive for every $\sigma_r^2>0$. If a hard bound $\sigma_r^2\geq\delta>0$ is required, it must be imposed through truncation or another prior whose support enforces that bound.

### 2.2 Orthogonal invariance: identifiability of $\boldsymbol{\Lambda}$

Given identifiability of $\boldsymbol{\Sigma}$, the second source of identifiability problems is related to orthogonal transformations of the matrix of factor loadings, which is the main focus here. A square matrix $\mathbf{O}$ is an *orthogonal* matrix if and only if $\mathbf{O}^{\top}=\mathbf{O}^{-1}$, that is, its inverse equals its transpose. The determinant of any orthogonal matrix is equal to $1$ or $-1$. Orthogonal matrices represent rotations which may be proper (if the determinant is positive) or improper (otherwise).

Consider a $q\times q$ orthogonal matrix $\mathbf{O}$ and define $\widetilde{F}_i=\mathbf{O}F_i$. It follows that the representation

$$
Y_i=\mu+\boldsymbol{\Lambda}\mathbf{O}^{\top}\widetilde{F}_i+\varepsilon_i
\tag{9}
$$

leads to the same marginal distribution of $Y_i$ as the one in Equation (6). The marginal likelihood is therefore invariant under $\boldsymbol{\Lambda}\mapsto\boldsymbol{\Lambda}\mathbf O^\top$. A proper posterior inherits this symmetry if the joint prior on the loadings and other model parameters is also invariant under this transformation, with the other parameters held fixed. For a posterior including the latent factors, the corresponding joint transformation is $\boldsymbol{\Lambda}\mapsto\boldsymbol{\Lambda}\mathbf O^\top$, $F_i\mapsto\mathbf O F_i$. For $q\geq2$, continuous symmetry produces orbits of equivalent parameter values; a posterior maximum on a nontrivial orbit is not an isolated mode. A non-invariant prior can break posterior symmetry without changing the non-identifiability of the likelihood.

The orthogonal invariance above follows from assumption (2), which yields the marginal covariance matrix $\boldsymbol{\Lambda}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}$ in Equation (6). If instead $F_i\sim\mathcal{N}_q(0_q,\boldsymbol{\Psi})$, where $\boldsymbol{\Psi}$ is a fixed positive definite diagonal matrix with pairwise distinct entries, the marginal covariance becomes $\boldsymbol{\Lambda}\boldsymbol{\Psi}\boldsymbol{\Lambda}^\top+\boldsymbol{\Sigma}$. The only Euclidean orthogonal transformations satisfying $\mathbf O\boldsymbol{\Psi}\mathbf O^\top=\boldsymbol{\Psi}$ are then diagonal sign matrices. Thus fixing distinct factor variances removes continuous orthogonal symmetries of the factor distribution, but preserves sign ambiguity. Ordinary column permutations remain symmetries if the diagonal entries of $\boldsymbol{\Psi}$ are permuted together with the loading columns; fixing their order excludes these simultaneous relabelings.

This restriction to orthogonal transformations does not establish general identifiability for unrestricted loadings. For any orthogonal $\mathbf U$, define

$$
\mathbf B=\boldsymbol{\Psi}^{1/2}\mathbf U\boldsymbol{\Psi}^{-1/2},
\qquad
\mathbf B\boldsymbol{\Psi}\mathbf B^\top=\boldsymbol{\Psi}.
$$

Then $\boldsymbol{\Lambda}\mathbf B$ yields the same marginal covariance as $\boldsymbol{\Lambda}$, and the accompanying factor transformation is $F_i\mapsto\mathbf B^{-1}F_i$. For $q\geq2$, these transformations form a continuous family and are generally nonorthogonal when the factor variances are distinct. Fixed unequal factor variances alone therefore do not remove continuous non-identifiability. For $q=1$, the family reduces to sign changes.

For the standardized-factor model (2), the approach followed here starts with unconstrained loading draws and seeks a common orientation *ex post*, by post-processing the MCMC sample. This selects a representation for posterior summaries; it does not make the original statistical model identifiable. The orthogonal RSP transformations below are specified for model (2), not for an arbitrary fixed unequal $\boldsymbol{\Psi}$.

---

## 3. Method

### 3.1 Notation and definitions

Let $\mathcal{T}_q$ denote the set of all $q!$ permutations of $\{1,\ldots,q\}$. For a given $\nu=(\nu_1,\ldots,\nu_q)\in\mathcal{T}_q$, consider the matrix $\dot{\boldsymbol{\Lambda}}$ where the $j$-th column of $\boldsymbol{\Lambda}$ is mapped to the $\nu_j$-th column of $\dot{\boldsymbol{\Lambda}}$, for $j=1,\ldots,q$. This transformation can be expressed using a $q\times q$ *permutation matrix* $\mathbf{P}$ whose $i,j$ element is equal to

$$
(\mathbf{P})_{ij}=
\begin{cases}
1, & \text{if } \nu_i=j,\\
0, & \text{otherwise},
\end{cases}
\quad i=1,\ldots,q;\ j=1,\ldots,q.
$$

The transformed matrix is simply written as $\dot{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf{P}$. A permutation matrix is a special case of an orthogonal matrix.

For example, assume that $\boldsymbol{\Lambda}$ has three columns and that $\nu=(3,1,2)\in\mathcal{T}_3$. This means that columns 1, 2, 3 become columns 3, 1, 2, respectively. This transformation corresponds to the $3\times 3$ permutation matrix

$$
\mathbf{P}=
\begin{pmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix}.
$$

Obviously, $\boldsymbol{\Lambda}\mathbf{P}$ results in the desired reordering of columns:

$$
\dot{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf{P}=
\begin{pmatrix}
\lambda_{11} & \lambda_{12} & \lambda_{13}\\
\vdots & \vdots & \vdots \\
\lambda_{p1} & \lambda_{p2} & \lambda_{p3}
\end{pmatrix}
\begin{pmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix}
=
\begin{pmatrix}
\lambda_{12} & \lambda_{13} & \lambda_{11}\\
\vdots & \vdots & \vdots \\
\lambda_{p2} & \lambda_{p3} & \lambda_{p1}
\end{pmatrix}.
$$

A *signed permutation matrix* is a square matrix which has precisely one nonzero entry in every row and column and whose only nonzero entries are $1$ and/or $-1$. A $q\times q$ signed permutation matrix $\mathbf{Q}$ can be expressed as

$$
\mathbf{Q}=\mathbf{S}\mathbf{P},
\tag{10}
$$

where $\mathbf{S}=\mathrm{diag}(s_1,\ldots,s_q)$ is a $q\times q$ diagonal matrix with diagonal entries equal to $s_j\in\{-1,1\}$, $j=1,\ldots,q$, and $\mathbf{P}$ is a $q\times q$ permutation matrix. Equation (10) uses the column-mapping convention of this subsection; the assignment convention of Section 3.2.2 uses its transpose. For example,

$$
\mathbf{Q}=
\begin{pmatrix}
0 & 0 & -1\\
-1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix}
=
\begin{pmatrix}
-1 & 0 & 0\\
0 & -1 & 0\\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{pmatrix}.
$$

It is evident that applying a signed permutation $\mathbf{Q}$ to a $p\times q$ matrix $\boldsymbol{\Lambda}$ results in a transformed matrix $\mathring{\boldsymbol{\Lambda}}$ arising from the corresponding signed permutation of its columns. For example

$$
\mathring{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf{Q}
=\boldsymbol{\Lambda}\mathbf{S}\mathbf{P}=
\begin{pmatrix}
-\lambda_{12} & \lambda_{13} & -\lambda_{11}\\
\vdots & \vdots & \vdots \\
-\lambda_{p2} & \lambda_{p3} & -\lambda_{p1}
\end{pmatrix}
$$

is generated by first switching the signs of the first two columns and then reordering the columns according to $\nu=(3,1,2)$.

Operations and/or special matrices of factor loadings will be denoted as follows; symbols introduced in Section 3.2 are included here for reference:

| Symbol | Meaning |
|---|---|
| $\widetilde{\boldsymbol{\Lambda}}$ | varimax rotated |
| $\dot{\boldsymbol{\Lambda}}$ | column-permuted |
| $\mathring{\boldsymbol{\Lambda}}$ | sign-permuted (a signed permutation of the columns) |
| $\boldsymbol{\Lambda}^\star$ | reference matrix of factor loadings |
| $\mathbf{O}$ | orthogonal matrix; $\mathbf{O}^{(t)}$ is the varimax rotation at draw $t$ |
| $\mathbf{P}$ | column-mapping permutation matrix of Section 3.1 |
| $\mathbf{S}$ | sign matrix, $\mathrm{diag}(s_1,\ldots,s_q)$ with $s_j\in\{-1,1\}$ |
| $\mathbf{Q}$ | signed permutation matrix; $\mathbf Q=\mathbf S\mathbf P$ for column mapping and $\mathbf Q=\mathbf P^\top\mathbf S$ for assignment |
| $\mathbf{R}^{(t)}$ | total orthogonal transform at draw $t$, $\mathbf R^{(t)}=\mathbf O^{(t)}\mathbf Q^{(t)}$ |
| $\mathcal{T}_q$ | set of all $q!$ permutations of $\{1,\ldots,q\}$ |
| $\mathcal{Q}_q$ | set of all $q\times q$ signed permutation matrices |
| $s$ | sign vector $(s_1,\ldots,s_q)$, $s_j\in\{-1,1\}$ |
| $\nu$ | permutation vector; read as an assignment (see the note below) |
| $\Psi$ | objective function of the signed-permutation stage |
| $\mathcal{L}^{(t)}_{s,\nu}$ | per-draw loss of the signed-permutation stage |

In the signed-permutation stage, $\nu$ is read as an assignment: reference column $j$ is matched to column $\nu_j$ of the draw.

### 3.2 Rotation-Sign-Permutation (RSP) post-processing algorithm

The RSP algorithm seeks a common orientation for an MCMC sample of factor loadings in two stages: a per-draw varimax iteration, followed by a recursive signed permutation alignment of the whole sample towards a common reference. These stages do not guarantee a unique orientation or complete removal of rotational discrepancies.

#### 3.2.1 Varimax rotation

Given a $p\times q$ matrix $\boldsymbol{\Lambda}$ of factor loadings, the varimax problem is to find a $q\times q$ rotation matrix $\boldsymbol{\Phi}$ such that the sum of the within-factor variances of squared factor loadings of the rotated matrix of loadings $\widetilde{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\boldsymbol{\Phi}$ is maximized. That is, the optimization problem is summarized by

$$
\begin{aligned}
\text{maximize}\quad & \frac{1}{4}\sum_{j=1}^{q}\left[\left(\sum_{r=1}^{p}\widetilde\lambda^4_{rj}\right)-\frac{1}{p}\left(\sum_{r=1}^{p}\widetilde\lambda^2_{rj}\right)^2\right]\\
\text{subject to}\quad & \boldsymbol{\Phi}^\top \boldsymbol{\Phi} = \mathbf{I}_q\\
\text{where}\quad & \widetilde\lambda_{rj}=\sum_{k=1}^{q}\lambda_{rk}\phi_{kj},\quad r=1,\ldots,p;\ j=1,\ldots,q.
\end{aligned}
\tag{11}
$$

The original approach for solving the varimax problem increases the objective function by successively rotating pairs of factors; subsequent developments based on matrix formulations rotate all factors simultaneously in order to improve the objective function.

In the implementation, a single-start varimax iteration is run separately for each MCMC draw, using the relative stagnation parameter $\varepsilon=10^{-5}$ described in Section 4.1. This parameter is not an accuracy bound for maximizing (11). Given simulated loadings $\boldsymbol{\Lambda}^{(t)}=(\lambda_{rj}^{(t)})$, $t=1,\ldots,T$, we denote the rotated output returned by the iteration as $\widetilde{\boldsymbol{\Lambda}}^{(t)}=(\widetilde\lambda_{rj}^{(t)})$, where $T$ is the size of the MCMC sample. Section 4.1 describes the updates, stopping rule and limitations.

For draws that differ only by orthogonal transformations, the remaining ambiguity reduces to signed permutations under two additional conditions: each varimax run attains a global maximum of (11), and the maximizing rotated loading matrices are unique up to signed permutations. Neither condition holds in general. In particular, different local optima or inequivalent global maxima can leave rotational discrepancies that the signed-permutation stage cannot remove. Ordinary posterior draws also vary in ways other than orientation, so exact agreement of all aligned draws is not expected.

#### 3.2.2 Signed permutations towards a reference

After the per-draw varimax iteration, the second stage minimizes the squared distance of the rotated output to a reference matrix $\boldsymbol{\Lambda}^{\star}$ over signed permutations. The minimum need not be small or zero. For the moment $\boldsymbol{\Lambda}^{\star}$ is assumed fixed; this assumption is relaxed in Section 3.2.3. Let $\mathcal{Q}_q$ denote the set of $q\times q$ signed permutation matrices. The optimization problem is stated as

$$
\begin{aligned}
\text{minimize}\quad & \sum_{t=1}^{T}\left\|\widetilde{\boldsymbol{\Lambda}}^{(t)}\mathbf{Q}^{(t)}-\boldsymbol{\Lambda}^{\star}\right\|^2\\
\text{subject to}\quad & \mathbf{Q}^{(t)}\in\mathcal{Q}_q,\quad t=1,\ldots,T,
\end{aligned}
\tag{12}
$$

where $\|\mathbf{A}\|=\sqrt{\sum_i\sum_j\alpha_{ij}^2}$ denotes the Frobenius norm on the matrix space. Notice that (12), with the signed-permutation constraint relaxed to $\mathbf{Q}^{\top}\mathbf{Q}=\mathbf{I}_q$, is the Orthogonal Procrustes problem, for which an analytical solution is available: for the $q\times q$ cross-product $\mathbf{C}^{(t)}=\boldsymbol{\Lambda}^{\star\top}\widetilde{\boldsymbol{\Lambda}}^{(t)}$ with singular value decomposition $\mathbf{C}^{(t)}=\mathbf{U}\mathbf{D}\mathbf{V}^{\top}$, the minimizer of $\|\widetilde{\boldsymbol{\Lambda}}^{(t)}\mathbf{Q}-\boldsymbol{\Lambda}^{\star}\|$ over all orthogonal $\mathbf{Q}$ is $\mathbf{Q}^{(t)}=\mathbf{V}\mathbf{U}^{\top}$ (improper rotations, with determinant $-1$, are allowed). The problem addressed here retains the restricted discrete parameter space $\mathcal{Q}_q$ rather than the set of all orthogonal matrices.

Parameterize an assignment signed permutation by the sign vector $s=(s_1,\ldots,s_q)$ and the permutation vector $\nu=(\nu_1,\ldots,\nu_q)\in\mathcal{T}_q$: reference column $j$ is matched to column $\nu_j$ of the rotated draw, scaled by the sign $s_j$. Its matrix $\mathbf Q$ is defined entrywise by $(\mathbf Q)_{\nu_j,j}=s_j$, with all other entries equal to zero. If $\mathbf P$ and $\mathbf S$ are the matrices built from the same pair under the column-mapping convention of Section 3.1, then $\mathbf Q=\mathbf P^\top\mathbf S=(\mathbf S\mathbf P)^\top$; the multiplication order is essential because signs and permutations do not generally commute. With this parameterization,

$$
\left\|\widetilde{\boldsymbol{\Lambda}}\mathbf{Q}-\boldsymbol{\Lambda}^{\star}\right\|^2
=\sum_{r=1}^{p}\sum_{j=1}^{q}\left(s_j\widetilde\lambda_{r\nu_j}-\lambda^{\star}_{rj}\right)^2.
$$

Thus, (12) can also be written as

$$
\begin{aligned}
\min\ \Psi(\boldsymbol{\Lambda}^{\star},s,\nu)
&=\sum_{t=1}^{T}\mathcal{L}_{s^{(t)},\nu^{(t)}}^{(t)}\\
&=\sum_{t=1}^{T}\sum_{r=1}^{p}\sum_{j=1}^{q}\left(s_j^{(t)}\widetilde\lambda_{r\nu_j^{(t)}}^{(t)}-\lambda^{\star}_{rj}\right)^2\\
\text{subject to}\quad & s_j^{(t)}\in\{-1,1\},\quad t=1,\ldots,T;\ j=1,\ldots,q\\
\text{and}\quad & \nu^{(t)}\in\mathcal{T}_q,\quad t=1,\ldots,T,
\end{aligned}
\tag{13}
$$

where we have also defined the per-draw loss

$$
\mathcal{L}_{s,\nu}^{(t)}:=\sum_{r=1}^{p}\sum_{j=1}^{q}\left(s_j\widetilde\lambda_{r\nu_j}^{(t)}-\lambda^{\star}_{rj}\right)^2.
\tag{14}
$$

**Index convention.** The assignment reading of $\nu$ stated above is used throughout this section; it is the inverse of the column mapping of Section 3.1. Both readings span the same family of signed permutations, so the optimal value of $\Psi$ and the optimal aligned draws are identical; only the correspondence between a stored pair $(s,\nu)$ and a signed permutation matrix differs.

Obviously, $\Psi$ in (13) also depends on $\{\widetilde{\boldsymbol{\Lambda}}\}_{t=1}^{T}$, but this is already fixed at this second stage, hence it is omitted from the notation on the left-hand side. The reference loading matrix $\boldsymbol{\Lambda}^{\star}=(\lambda^{\star}_{rj})$ is not known, thus it is approximated by a recursive algorithm, in the spirit of the label switching problem in Bayesian mixture analysis.

#### 3.2.3 The recursive algorithm

The Varimax Rotation-Sign-Permutation (RSP) post-processing algorithm alternates two coordinate-wise minimizations of $\Psi(\boldsymbol{\Lambda}^{\star},s,\nu)$:

- the **Sign-Permutation (SP) step** minimizes with respect to $(s,\nu)$ for a fixed reference matrix;
- the **Reference-Loading Matrix Estimation (RLME) step** minimizes with respect to $\boldsymbol{\Lambda}^{\star}$ for fixed signed permutations.

Although each SP subproblem is solved exactly, the outer algorithm is an alternating minimization and does not guarantee a global optimum of (13).

**Algorithm 1 (Varimax-RSP, current implementation).**

- **Input:** simulated MCMC sample of factor loadings $\{\boldsymbol{\Lambda}^{(t)},t=1,\ldots,T\}$.
- **Output:** the aligned sample $\{\mathring{\boldsymbol{\Lambda}}^{(t)},t=1,\ldots,T\}$ and its final reference matrix, together with the diagnostics described below.
- **Step 1 — Varimax rotation:** for each $t=1,\ldots,T$, run the iteration of Section 4.1 on $\boldsymbol{\Lambda}^{(t)}$ to obtain $\widetilde{\boldsymbol{\Lambda}}^{(t)}$ and its rotation matrix; optimality for (11) is not certified.
- **Step 2 — Reference initialization.** An explicitly supplied reference takes priority. Otherwise, initialize $\mathbf M_0=\boldsymbol{\Lambda}^{\star}$ by one of the following rules:
  - **`mean`:** set $\mathbf M_0=T^{-1}\sum_{t=1}^{T}\widetilde{\boldsymbol{\Lambda}}^{(t)}$. This is the original R behavior, equivalently obtained by initializing all signed permutations to the identity and applying an initial RLME update.
  - **`auto` (default) or `draw`:** select the representative draw $t_0=\operatorname*{argmax}_t\|\widetilde{\boldsymbol{\Lambda}}^{(t)}\|_F$. Using $\widetilde{\boldsymbol{\Lambda}}^{(t_0)}$ as a temporary reference, perform one complete SP--RLME sweep over all draws and set $\mathbf M_0$ to the resulting aligned mean; it is not generally the representative draw itself. `auto` additionally uses the near-zero mean-cancellation diagnostic for verbose logging. Both rules otherwise follow the same representative initialization, and the diagnostic never acts as a condition that switches initialization rules. For all-zero input, the mean reference is retained.
- **Step 3 — Alternating SP--RLME sweeps.** For $k=1,\ldots,K$, with default $K=\texttt{max_iter}=100$:
  - **3.1 SP step.** Given $\mathbf M_{k-1}$, solve (15) independently for every MCMC draw $t=1,\ldots,T$:

    $$
    \left(s^{(t)},\nu^{(t)}\right)=\operatorname*{argmin}_{s,\nu}\left\{\mathcal{L}_{s,\nu}^{(t)}:s\in\{-1,1\}^q,\ \nu\in\mathcal{T}_q\right\}.
    \tag{15}
    $$

    Let $\mathring{\boldsymbol{\Lambda}}_k^{(t)}$ denote the resulting aligned draw and define the pre-update loss $f_k^{\mathrm{pre}}=\sum_{t=1}^{T}\|\mathring{\boldsymbol{\Lambda}}_k^{(t)}-\mathbf M_{k-1}\|_F^2$.
  - **3.2 RLME step.** Update the reference and evaluate the post-update loss:

    $$
    \mathbf M_k=\frac{1}{T}\sum_{t=1}^{T}\mathring{\boldsymbol{\Lambda}}_k^{(t)},
    \qquad
    f_k^{\mathrm{post}}=\sum_{t=1}^{T}\|\mathring{\boldsymbol{\Lambda}}_k^{(t)}-\mathbf M_k\|_F^2.
    $$

    Writing $\mathbf M_k=(\lambda_{k,rj})$, its elementwise form $\lambda_{k,rj}=T^{-1}\sum_t s_{k,j}^{(t)}\widetilde\lambda_{r\nu_{k,j}^{(t)}}^{(t)}$ is the exact minimizer with respect to the reference for fixed signed permutations.
  - **3.3 Stopping and status.** The stopping sweep still performs the RLME update. Let $\theta=\texttt{threshold}$, with default $\theta=10^{-6}$, and let $\varepsilon_{\mathrm{dtype}}$ be the working-precision machine epsilon. Define $\rho=\max\{10^{-9},10\varepsilon_{\mathrm{dtype}}\sqrt{Tpq}\}$ and $\eta=100\varepsilon_{\mathrm{dtype}}^2 Tpq(\max_{t,r,j}|\lambda_{rj}^{(t)}|)^2$, using the original input loadings for the data scale. The latter is an implementation tolerance for detecting anomalies, not a rigorous rounding-error bound.
    - At every sweep, nonfinite pre- or post-update loss, aligned loadings, or updated reference takes precedence over the stopping rules below: the run returns `status_code = 4`. Final returned arrays and any requested transforms are checked as well; `diagnostics.nonfinite_fields` identifies the affected quantities. A nonfinite initial unaligned loss alone does not preclude a subsequent finite alignment.
    - Otherwise, for $k\geq2$, an anomaly is declared if $f_k^{\mathrm{pre}}-f_{k-1}^{\mathrm{pre}}>\max\{\rho|f_{k-1}^{\mathrm{pre}}|,\eta\}$; the run stops with `status_code = 3`.
    - Otherwise, for $k\geq2$, the run stops with `status_code = 1` (`converged`) when $f_{k-1}^{\mathrm{pre}}-f_k^{\mathrm{pre}}<\theta Tpq$. Thus sufficiently small positive improvements also stop the algorithm.
    - If no rule has fired after $K$ sweeps, the run ends with `status_code = 2` (`max_iter`).

The representative-initialization sweep in Step 2 is not included in `n_iter` or in the loss histories.

The `converged` flag and `status_code` above describe only the outer SP--RLME stopping rule. They do not certify convergence or optimality of the preceding varimax iterations, or a global minimum of (13).

The floating-point implementation uses binary exponent scaling for residuals, means, and norms. SP retains a separate exponent for each pair cost and preserves the original assignment weights; a wide dynamic range can require additional assignment solves using a feasible-cost bound. These safeguards address the recorded range and cancellation regressions, but do not make arbitrary floating-point inputs equivalent to exact arithmetic. A loss outside the output dtype's range can still overflow or round to zero.

**Result and indexing conventions.** The implementation returns the following fields:

| Field | Meaning |
|---|---|
| `aligned_loadings` | final aligned draws $\mathring{\boldsymbol{\Lambda}}^{(t)}$ |
| `mean_loadings` | final reference $\mathbf M$ |
| `signs`, `permutations` | final signs and zero-based assignments; adding one to every permutation entry gives the mathematical index $\nu_j$ |
| `loss_initial` | initial unaligned loss $\sum_t\|\widetilde{\boldsymbol{\Lambda}}^{(t)}-\mathbf M_0\|_F^2$ |
| `loss_final` | final post-update loss $f^{\mathrm{post}}$ |
| `loss_history`, `loss_history_post` | pre- and post-update losses; their first `n_iter` entries are valid |
| `n_iter`, `converged`, `status_code` | number of outer sweeps and outer stopping diagnostics |
| `varimax_n_iter` | per-draw `int32` counts for the preceding varimax stage, or `None` if that stage was skipped; a count of 1000 means the iteration budget was exhausted, not that a maximum was reached |
| `transforms` | optional total orthogonal transforms $\mathbf R^{(t)}$, returned when `return_transforms=True` |
| `diagnostics`, `metadata` | configuration and caller-supplied MCMC information |

Let $\mathbf{O}^{(t)}$ denote the varimax rotation matrix at Step 1 — the matrix denoted $\boldsymbol{\Phi}$ in (11). Let $\mathbf Q^{(t)}$ be the assignment signed-permutation matrix attached to $(s^{(t)},\nu^{(t)})$, so $(\mathbf Q^{(t)})_{\nu_j^{(t)},j}=s_j^{(t)}$; equivalently, $\mathbf Q^{(t)}=(\mathbf P^{(t)})^\top\mathbf S^{(t)}$ when $\mathbf P^{(t)}$ is the column-mapping matrix of Section 3.1. Define the total orthogonal transform $\mathbf R^{(t)}=\mathbf O^{(t)}\mathbf Q^{(t)}$. The overall transformations of the factor loadings and of the column-vector latent factors are

$$
\mathring{\boldsymbol{\Lambda}}^{(t)}
=\boldsymbol{\Lambda}^{(t)}\mathbf R^{(t)}
=\boldsymbol{\Lambda}^{(t)}\mathbf O^{(t)}\mathbf Q^{(t)},\quad t=1,\ldots,T,
\tag{16}
$$

$$
\mathring{F}_i^{(t)}
=(\mathbf R^{(t)})^\top F_i^{(t)}
=(\mathbf Q^{(t)})^\top(\mathbf O^{(t)})^\top F_i^{(t)},
\quad i=1,\ldots,n,
\tag{17}
$$

for MCMC draw $t=1,\ldots,T$. For row-stored scores $f_i^{(t)}=(F_i^{(t)})^\top$, the equivalent rule is $\mathring f_i^{(t)}=f_i^{(t)}\mathbf R^{(t)}$. These conventions preserve the fitted common component: $\mathring{\boldsymbol{\Lambda}}^{(t)}\mathring F_i^{(t)}=\boldsymbol{\Lambda}^{(t)}F_i^{(t)}$. The required matrices are derived from the loadings alone and are available when `return_transforms=True`.

The SP--RLME stage (Steps 2--3) can also serve as a stand-alone algorithm, without the varimax rotation in Step 1; an application is the comparison of multiple MCMC chains. Strategies for minimizing expression (15) are discussed in Section 4.2.

---

## 4. Computational Strategies

### 4.1 Iterative optimization of the varimax criterion

A single-start matrix-form varimax iteration is used independently for each MCMC draw to seek a maximizer of (11). This approach updates all $q$ factors simultaneously rather than successively rotating pairs of factors. The implementation uses the unnormalized loadings, corresponding to `normalize = FALSE`: no row (Kaiser) normalization is applied. Denote the objective function in (11) by $\mathcal V(\mathbf O)$, where $\mathbf O\in\mathbb R^{q\times q}$ and $\mathbf O^\top\mathbf O=\mathbf I_q$.

#### Matrix gradient and orthogonal update

Let $\mathbf O_k$ be the orthogonal rotation at iteration $k$, starting from $\mathbf O_0=\mathbf I_q$. Define the current rotated matrix, the column sums of its squared elements, and the matrix $\mathbf B_k$ by

$$
\begin{aligned}
\mathbf Z_k&=\boldsymbol{\Lambda}\mathbf O_k,\\
c_{kj}&=\sum_{r=1}^{p}z_{rkj}^2,\qquad
\mathbf C_k=\operatorname{diag}(c_{k1},\ldots,c_{kq}),\\
\mathbf B_k
&=\boldsymbol{\Lambda}^{\top}
\left(\mathbf Z_k^{\odot 3}-\frac{1}{p}\mathbf Z_k\mathbf C_k\right)
=\nabla\mathcal V(\mathbf O_k),
\end{aligned}
\tag{18}
$$

where $\mathbf Z_k^{\odot 3}$ denotes the elementwise cube of $\mathbf Z_k$. With the factor $1/4$ included in (11), $\mathbf B_k$ is exactly the Euclidean matrix gradient of the varimax criterion at the current rotation. If the objective is multiplied by four, the gradient is $4\mathbf B_k$; this positive rescaling does not change the orthogonal update in (19).

The next rotation is obtained by maximizing the associated linear functional over the set of orthogonal matrices:

$$
\mathbf O_{k+1}
=\operatorname*{argmax}_{\mathbf O^\top\mathbf O=\mathbf I_q}
\langle\mathbf B_k,\mathbf O\rangle_F
=\operatorname*{argmax}_{\mathbf O^\top\mathbf O=\mathbf I_q}
\operatorname{tr}(\mathbf B_k^\top\mathbf O).
\tag{19}
$$

This problem has an explicit singular-value solution. If

$$
\mathbf B_k=\mathbf U_k\mathbf D_k\mathbf V_k^\top,
\qquad
\mathbf O_{k+1}=\mathbf U_k\mathbf V_k^\top,
\qquad
d_k:=\operatorname{tr}(\mathbf O_{k+1}^\top\mathbf B_k)
=\operatorname{tr}(\mathbf D_k),
\tag{20}
$$

where $\mathbf D_k$ contains the singular values of $\mathbf B_k$. Indeed, von Neumann's trace inequality gives $\langle\mathbf B_k,\mathbf O\rangle_F\leq\operatorname{tr}(\mathbf D_k)$ for every orthogonal $\mathbf O$, with equality at $\mathbf O=\mathbf U_k\mathbf V_k^\top$. The matrix $\mathbf U_k\mathbf V_k^\top$ replaces the current rotation; it is not additionally multiplied by $\mathbf O_k$.

#### Polar decomposition and the QDWH alternative

The update in (20) can equivalently be expressed through the polar decomposition

$$
\mathbf B_k=\mathbf W_k\mathbf H_k,
\qquad
\mathbf W_k=\mathbf U_k\mathbf V_k^\top,
\qquad
d_k=\langle\mathbf W_k,\mathbf B_k\rangle_F=\operatorname{tr}(\mathbf D_k),
\tag{21}
$$

where $\mathbf W_k$ is orthogonal and $\mathbf H_k$ is symmetric positive semidefinite. For nonsingular $\mathbf B_k$, the orthogonal polar factor can also be written as $\mathbf W_k=\mathbf B_k(\mathbf B_k^\top\mathbf B_k)^{-1/2}$.

Two numerical routes are used to obtain $\mathbf W_k$:

1. **SVD route.** Compute the singular value decomposition of $\mathbf B_k$ and set $\mathbf W_k=\mathbf U_k\mathbf V_k^\top$. This is the R-faithful reference iteration corresponding to R's `stats::varimax`, selected explicitly with `polar="svd"`.
2. **QDWH route (current default, `polar="qdwh"`).** Compute the same orthogonal polar factor using the QR-based dynamically weighted Halley (QDWH) iteration, without explicitly forming an SVD. The stopping quantity is evaluated as $d_k=\langle\mathbf W_k,\mathbf B_k\rangle_F$, which equals the sum of the singular values when $\mathbf W_k$ is the exact polar factor.

For an exactly zero or non-finite $\mathbf B_k$, the QDWH implementation keeps the current rotation for that update. It also checks convergence of the inner QDWH iteration, finiteness, and orthogonality of the returned rotation. The default orthogonality tolerance is $10^{-3}$ for float32 and $10^{-9}$ for float64. A draw failing these checks is recomputed using the SVD route; accepted draws are not recomputed. These checks concern the numerical polar updates, not optimality of the varimax objective.

#### Scale invariance and numerical scaling

The varimax maximizer is invariant under a positive uniform rescaling of the loadings. For each draw, the iteration reduces the risk of overflow or underflow in the fourth-order intermediates in (18) by using

$$
\boldsymbol{\Lambda}_{\alpha}=\frac{\boldsymbol{\Lambda}}{\alpha},
\qquad \alpha=\alpha_1\alpha_2>0,
\qquad
\mathbf B_{\alpha,k}=\alpha^{-4}\mathbf B_k,
\qquad
\operatorname{polar}(\mathbf B_{\alpha,k})=\operatorname{polar}(\mathbf B_k)
\quad(\det\mathbf B_k\neq0),
\tag{22}
$$

where $\mathbf B_{\alpha,k}$ is constructed from $\boldsymbol{\Lambda}_{\alpha}$ and $\operatorname{polar}(\cdot)$ denotes the orthogonal polar factor. For nonsingular $\mathbf B_k$, the unique orthogonal polar factor is unchanged. For singular matrices, positive scaling preserves the set of admissible polar factors, and any fixed corresponding SVD/polar branch transforms in the same way. Consequently, scaling changes the singular values but not the orthogonal update or the maximizing rotation. The final rotated loadings are formed on the original scale as $\boldsymbol{\Lambda}\mathbf O_k$.

For each draw with $a=\max_{r,j}|\lambda_{rj}|>0$, the scale is chosen as a power of two based on $a$. Writing $e=\lfloor\log_2a\rfloor$, $e_1=\lfloor e/2\rfloor$ and $e_2=e-e_1$, the implementation uses $\alpha_1=2^{e_1}$ and $\alpha_2=2^{e_2}$ and divides successively by these two factors. For normal binary floating-point values whose exponents remain within the representable range, multiplication or division by a power of two introduces no additional rounding. Splitting the exponent protects the scaling itself at extreme magnitudes, although subnormal and underflow effects cannot be eliminated universally. For $a=0$, the implementation uses $\alpha_1=\alpha_2=1$. This numerical scaling is distinct from Kaiser normalization.

#### Stopping rule and output

The scalar $d_k$ in (20)–(21) is the sum of the singular values of $\mathbf B_k$, not the varimax objective itself. Successive values are used to detect relative stagnation. After the current update, the iteration stops when

$$
d_k<d_{k-1}(1+\varepsilon),
\qquad \varepsilon=10^{-5},
\tag{23}
$$

or when 1000 updates have been performed. For nonzero draws with $q\geq2$, the initial zero convention ensures that the first update is executed. If $\mathbf O_{\mathrm{final}}$ is the final rotation, the returned loadings and rotation are $\widetilde{\boldsymbol{\Lambda}}=\boldsymbol{\Lambda}\mathbf O_{\mathrm{final}}$ and $\mathbf O_{\mathrm{final}}$, respectively. For $q<2$ or an exactly zero draw, no iteration is needed and the input is returned with an identity rotation and count zero.

For $q\geq2$, this stopping rule certifies neither stationarity nor a local or global maximum of (11). The iteration may stagnate at a stationary point that is not a local maximum, or reach the update limit. In particular, $\varepsilon=10^{-5}$ controls the comparison of successive $d_k$ values; it does not bound the error in the varimax objective or the distance to a maximizing rotation. Even an exact global maximizer need not be unique up to signed permutations. Thus the sufficient conditions stated in Section 3.2.1 are not guaranteed by the iteration.

### 4.2 Solving the sign-permutation problem

The SP step in (15) is a discrete optimization problem over $2^q q!$ signed permutations. For a fixed reference matrix, the total objective in (13) separates across MCMC draws, so each draw can be aligned independently. Whether it follows the varimax step or is used as a stand-alone algorithm, the key computational device is to eliminate the signs analytically and then solve one linear assignment problem per draw. The default `sp_path="elim"` uses this elimination route; `sp_path="enum"` is retained for legacy compatibility and validation.

To simplify the notation, suppress the draw index $t$ and write $\mathbf A=(a_{rj})$ for the loadings being aligned and $\mathbf M=(m_{rj})$ for the current reference $\boldsymbol{\Lambda}^{\star}$. Under the assignment convention of Section 3.2.2, reference column $j$ is matched to column $\nu_j$ of $\mathbf A$.

#### Analytical elimination of the signs

For a proposed match between reference column $j$ and draw column $k$, evaluate the two possible signed squared residuals directly:

$$
c_{jk}^{(+)}
=\sum_{r=1}^{p}(a_{rk}-m_{rj})^2,
\qquad
c_{jk}^{(-)}
=\sum_{r=1}^{p}(a_{rk}+m_{rj})^2,
\tag{24}
$$

where $c_{jk}^{(+)}$ corresponds to $s_j=+1$ and $c_{jk}^{(-)}$ to $s_j=-1$. The optimal sign and minimum residual for this particular match are

$$
c_{jk}=\min\left\{c_{jk}^{(+)},c_{jk}^{(-)}\right\},
\qquad
\eta_{jk}=
\begin{cases}
+1,&c_{jk}^{(+)}\leq c_{jk}^{(-)},\\
-1,&c_{jk}^{(+)}>c_{jk}^{(-)}.
\end{cases}
\tag{25}
$$

Thus, a zero residual difference is resolved in favour of the positive sign, and the chosen sign is always in $\{-1,+1\}$.

For a fixed permutation $\nu$, the signs affect different columns independently. Consequently, the sign minimization can be interchanged with the permutation minimization without approximation:

$$
\boxed{
\min_{\substack{s\in\{-1,+1\}^q\\\nu\in\mathcal T_q}}
\mathcal L_{s,\nu}
=
\min_{\nu\in\mathcal T_q}
\sum_{j=1}^{q}c_{j,\nu_j}.
}
\tag{26}
$$

The joint discrete problem therefore reduces to a minimization over permutations using the precomputed pairwise costs $c_{jk}$.

#### Reduction to a linear assignment problem

Equation (26) is a square linear assignment problem on the $q\times q$ cost matrix $\mathbf C=(c_{jk})$. Its rows index reference columns and its columns index draw columns; a feasible assignment matches every reference column to exactly one draw column and vice versa. Once the assignment has been solved, the signs, aligned loadings, and per-draw loss are recovered as

$$
\nu^\star=\operatorname*{argmin}_{\nu\in\mathcal T_q}
\sum_{j=1}^{q}c_{j,\nu_j},
\qquad
s_j^\star=\eta_{j,\nu_j^\star},
\qquad
\mathring a_{rj}=s_j^\star a_{r\nu_j^\star},
\qquad
\mathcal L^\star=\sum_{j=1}^{q}c_{j,\nu_j^\star}.
\tag{27}
$$

Hence only one assignment problem is required for each draw and each outer SP sweep. The final operation in (27) is precisely a column gather followed by column-wise sign multiplication.

Three assignment solvers are available:

1. **Exhaustive permutation search.** All $q!$ permutations are evaluated. This is used for small $q$ and gives the lexicographically smallest optimal permutation under the exhaustive tie rule.
2. **Hungarian algorithm.** The Optax Hungarian solver provides a polynomial-time assignment method with typical $O(q^3)$ complexity and is the default production route for larger $q$.
3. **SciPy assignment solver.** `scipy.optimize.linear_sum_assignment` provides an independent host-side backend and validation route, selected explicitly with `assignment_backend="scipy"`; it is not an automatic fallback from Optax.

Under the default dispatch rule, exhaustive search is used for $q\leq7$ and the Optax Hungarian solver for $q>7$; this threshold is configurable. Constructing the pairwise costs requires $O(pq^2)$ operations per draw. Batch chunking only changes how these independent problems are executed and does not alter the optimization problem.

#### Equivalent inner-product formulation and direct residual evaluation

The sign elimination can also be understood algebraically through the pairwise inner products

$$
g_{jk}=\sum_{r=1}^{p}m_{rj}a_{rk},
\qquad
c_{jk}^{(+)}-c_{jk}^{(-)}=-4g_{jk},
\qquad
c_{jk}=\sum_{r=1}^{p}a_{rk}^2+\sum_{r=1}^{p}m_{rj}^2-2|g_{jk}|.
\tag{28}
$$

Thus $\eta_{jk}$ is the sign of $g_{jk}$, with the positive sign chosen when $g_{jk}=0$. Moreover, every permutation uses every reference column and every draw column exactly once, so the two squared-norm sums are constant across feasible assignments. Therefore,

$$
\min_{\nu\in\mathcal T_q}\sum_{j=1}^{q}c_{j,\nu_j}
=
\|\mathbf A\|_F^2+\|\mathbf M\|_F^2
-2\max_{\nu\in\mathcal T_q}\sum_{j=1}^{q}|g_{j,\nu_j}|.
\tag{29}
$$

Although (29) is algebraically equivalent, the implementation deliberately constructs the assignment costs from the direct squared differences in (24), rather than from expanded squared norms or $-|g_{jk}|$. This avoids the catastrophic cancellation that can occur when subtracting nearly equal large quantities in the expanded formulation; it does not change the underlying discrete optimization problem.

#### Exhaustive sign enumeration as a validation route

For validation and reproduction of the R enumeration order, an alternative route enumerates all $2^q$ sign vectors. For any fixed $s\in\{-1,+1\}^q$, define

$$
c_{jk}^{(s)}=\sum_{r=1}^{p}(a_{rk}-s_jm_{rj})^2,
\qquad
\mathcal L^\star(s)=
\min_{\nu\in\mathcal T_q}\sum_{j=1}^{q}c_{j,\nu_j}^{(s)},
\qquad
\mathcal L^\star=\min_{s\in\{-1,+1\}^q}\mathcal L^\star(s).
\tag{30}
$$

Each minimization over $\nu$ is again a linear assignment problem. This enumeration route therefore solves $2^q$ assignment problems per draw, compared with only one in the elimination route. Strict loss comparisons retain the first minimum in the prescribed sign-enumeration order. The two routes solve the same fixed-reference optimization problem, although their selected signs or permutations may differ under ties.

Both routes are exact for the SP subproblem with $\mathbf M$ held fixed. The subsequent RLME update replaces $\mathbf M$ by the mean of the aligned draws and the two steps are repeated. Accordingly, exactness of each SP solve does not imply that the overall alternating RSP algorithm attains a global optimum over the reference and all signed permutations.

## References

1. Papastamoulis, P., & Ntzoufras, I. (2022). On the identifiability of Bayesian factor analytic models. *Statistics and Computing*, **32**(2), 23.

2. Rey-Sáez, R., & Revuelta, J. (2026). An Efficient Rotation-Sign-Permutation Algorithm to Solve Rotational Indeterminacy in Bayesian Exploratory Factor Analysis. *Bayesian Analysis*, **1**(1), 1–23.
