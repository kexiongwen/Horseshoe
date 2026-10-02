# Scalable Approximate MCMC for the Horseshoe Prior: Model, Algorithms, Derivations

> **Corrections relevant to what is kept.**
>
> 1. Eigenvalues of $M_{\xi,\delta}$: $1+\xi^{-1}s_i^2$ for the singular values $s_i$ of $W_SD_S^{1/2}$, plus $1$ with multiplicity $N-s_\delta$ — not $1+s^2$.
> 2. Eq. (9): the $(S,S)$ block of $\Sigma_\delta$ carries the positive semidefinite term $\Delta$; equivalently (30) reads $\Sigma_\delta = \Gamma - X - X' + Y$.
> 3. Eqs. (6)–(7): the active set is defined with $\xi_{\min}=\min(\xi,\xi^*)$ — equivalently $\eta_j^{-1}\max(\xi^{-1},\xi^{*-1})>\delta$ — because the same $D_\delta$ serves both terms of the MH ratio in step 2 of (4).
> 4. Truncation: the convergence analysis uses $\eta_j\ge b$ and $\xi^{-1/2}\in[a_\xi,b_\xi]$; everything below is the main-text case $b=0$, $a_\xi=0$, $b_\xi=\infty$.

---

## 1. Model

**Likelihood.** For $z\in\mathbb{R}^N$, $W\in\mathbb{R}^{N\times p}$ and $\beta\in\mathbb{R}^p$ sparse,

$$
L(z\mid W\beta,\sigma^2) = (2\pi\sigma^2)^{-N/2}\exp\Big\{-\frac{1}{2\sigma^2}(z-W\beta)'(z-W\beta)\Big\}.\tag{1}
$$

**Prior.** A global–local Gaussian scale mixture on $\beta$:

$$
\beta_j\mid\sigma^2,\eta,\xi \stackrel{iid}{\sim} \mathrm{N}\big(0,\sigma^2\xi^{-1}\eta_j^{-1}\big),\quad
\eta_j^{-1/2}\stackrel{iid}{\sim}\upsilon_L,\quad
\xi^{-1/2}\sim\upsilon_G,\quad
\sigma^2\sim\mathrm{InvGamma}(\omega/2,\omega/2),\tag{2}
$$

with $\upsilon_L,\upsilon_G$ densities on $\mathbb{R}_+$, $j=1,\dots,p$. Writing $\lambda_j=\eta_j^{-1/2}$ for the local scales and $\tau=\xi^{-1/2}$ for the global scale, (2) says $\beta_j\mid\sigma^2,\tau,\lambda_j\sim\mathrm{N}(0,\sigma^2\tau^2\lambda_j^2)$, i.e. the prior is conditionally Gaussian given the scales.

**Horseshoe case.** $\upsilon_L=\upsilon_G=$ standard half-Cauchy, so $\lambda_j\stackrel{iid}{\sim}\mathrm{C}^+(0,1)$ and $\tau\sim\mathrm{C}^+(0,1)$. The marginal prior on each $\beta_j$ (integrating out $\lambda_j$) then has a pole at zero and Cauchy-like tails, which is what makes it adaptive to sparsity.

**Roles of the two scales.** $\xi$ (global precision) controls how many components are signals; the $\eta_j$ (local precisions) decide which components are nulls. Components with large $\xi\eta_j$ are pinned near zero by the prior, which is precisely the structure the approximate algorithm exploits in §2.2: the posterior is tightly concentrated near the origin in a subspace of dimension $\approx p-s$, where $s$ is the unknown number of non-nulls.

---

## 2. Algorithms

The following quantities recur throughout. With $D=\operatorname{diag}(\eta_j^{-1})$ and $\Gamma := \xi^{-1}D$,

$$
D=\operatorname{diag}(\eta_j^{-1}),\qquad M_\xi = I_N+\xi^{-1}WDW' = I_N + W\Gamma W',\qquad
p(\xi\mid\eta)=|M_\xi|^{-1/2}\Big(\frac{\omega}{2}+\frac{1}{2}z'M_\xi^{-1}z\Big)^{-(N+\omega)/2}\frac{1}{\sqrt{\xi}(1+\xi)}.\tag{3}
$$

### 2.1. Exact algorithm

The exact sampler is a blocked Metropolis-within-Gibbs scheme targeting the horseshoe posterior, with $\eta$ drawn exactly and $(\beta,\sigma^2,\xi)$ blocked:

$$
\begin{aligned}
&\text{1. sample } \eta\sim p(\eta\mid\xi,\beta,\sigma^2)\propto\prod_{j=1}^p\frac{1}{1+\eta_j}e^{-\beta_j^2\xi\eta_j/(2\sigma^2)},\\
&\text{2. propose } \log(\xi^*)\sim\mathrm{N}(\log\xi,s),\ \text{accept }\xi^*\text{ w.p. } \frac{p(\xi^*\mid\eta)\,\xi^*}{p(\xi\mid\eta)\,\xi},\\
&\text{3. sample } \sigma^2\mid\eta,\xi\sim\mathrm{InvGamma}\Big(\frac{\omega+N}{2},\frac{\omega+z'M_\xi^{-1}z}{2}\Big),\\
&\text{4. sample } \beta\mid\eta,\xi,\sigma^2\sim\mathrm{N}\Big((W'W+\Gamma^{-1})^{-1}W'z,\ \sigma^2(W'W+\Gamma^{-1})^{-1}\Big).
\end{aligned}\tag{4}
$$

The chain defined by (4) is denoted $\mathcal{P}$. Step 1 is done with the exact rejection sampler of Appendix S1; the point of blocking $(\beta,\sigma^2,\xi)$ in step 2 is that the acceptance ratio uses the marginal $p(\xi\mid\eta)$ rather than $p(\xi\mid\eta,\beta,\sigma^2)$, which mixes the global scale far better.

#### Derivation of the $\xi$ target (3)

$p(\xi\mid\eta)\propto p(\xi)\iint p(z\mid\beta,\sigma^2)p(\beta\mid\sigma^2,\eta,\xi)p(\sigma^2)\,d\beta\,d\sigma^2$. Given $(\eta,\xi)$, the $\beta$-precision is $\Gamma^{-1}+W'W$, so completing the square with $\Sigma=(W'W+\Gamma^{-1})^{-1}$, $\mu=\Sigma W'z$ and integrating $\beta$ out leaves

$$
(2\pi\sigma^2)^{-N/2}|M_\xi|^{-1/2}\exp\Big\{-\frac{z'M_\xi^{-1}z}{2\sigma^2}\Big\},
$$

using $z'z-z'W\Sigma W'z = z'M_\xi^{-1}z$ and $|\Gamma|^{-1/2}|\Sigma|^{1/2}=|M_\xi|^{-1/2}$. Multiplying by the InvGamma$(\omega/2,\omega/2)$ density and integrating $\sigma^2$,

$$
\int(\sigma^2)^{-(N+\omega)/2-1}e^{-(z'M_\xi^{-1}z+\omega)/(2\sigma^2)}d\sigma^2 \ \propto\ \Big(\frac{\omega}{2}+\frac{z'M_\xi^{-1}z}{2}\Big)^{-(N+\omega)/2}.
$$

Finally $\xi^{-1/2}\sim\mathrm{C}^+(0,1)$ gives $p(\xi)\propto\xi^{-1/2}/(1+\xi)$, so the product is exactly (3). Two by-products of the same calculation are used below: taking the $\sigma^2$-integrand *before* integrating gives step 3 of (4), and the same completion of squares gives the full conditional in step 4.

The factor $\xi^*/\xi$ in the acceptance ratio of step 2 is the Jacobian of the log-scale random walk, not part of the target: for $\log\xi^*\sim\mathrm{N}(\log\xi,s)$ the proposal satisfies $q(\xi^*\mid\xi)\propto1/\xi^*$ and $q(\xi\mid\xi^*)\propto1/\xi$, so

$$
\frac{p(\xi^*\mid\eta)\,q(\xi\mid\xi^*)}{p(\xi\mid\eta)\,q(\xi^*\mid\xi)}=\frac{p(\xi^*\mid\eta)\,\xi^*}{p(\xi\mid\eta)\,\xi}.
$$

#### Derivation of step 1, and why it can be sampled exactly

The prior marginals are $p(\eta_j)\propto\eta_j^{-1/2}/(1+\eta_j)$ (from $\eta_j^{-1/2}\sim\mathrm{C}^+(0,1)$) and $p(\beta_j\mid\eta_j,\xi,\sigma^2)$ has density $\propto\eta_j^{1/2}e^{-m_j\eta_j}$ with

$$
m_j=\frac{\xi\beta_j^2}{2\sigma^2}.
$$

The factors $\eta_j^{\pm1/2}$ cancel, so

$$
p(\eta_j\mid\xi,\beta,\sigma^2)\propto\frac{e^{-m_j\eta_j}}{1+\eta_j},\qquad \eta_j>0,
$$

and the $\eta_j$ are conditionally independent. The log-density is $-m_jt-\log(1+t)$, whose second derivative $1/(1+t)^2$ is positive, so the full conditional is **log-convex**; equivalently $f(t)=m_jt+\log(1+t)$ is increasing and concave. Because $f$ is concave, a piecewise-linear lower bound $f_L\le f$ can be built from its chords; $e^{-f_L}$ is then an upper envelope of the target density that is a mixture of truncated exponentials, samplable by inversion and accepted with high probability. This gives an exact rejection sampler — see Appendix S1.

#### Derivation of the $\beta$ sampler (5)

Because $\Gamma$ is diagonal and $\beta\mid\sigma^2,\eta,\xi$ is Gaussian, step 4 can be performed without ever forming a $p\times p$ inverse. By Woodbury,

$$
\mu=\Sigma W'z=\Gamma W'M_\xi^{-1}z,\qquad \Sigma=(W'W+\Gamma^{-1})^{-1}=\Gamma-\Gamma W'M_\xi^{-1}W\Gamma .
$$

Sample

$$
u\sim\mathrm{N}(0,\Gamma),\quad f\sim\mathrm{N}(0,I_N)\ \text{independently},\qquad
v=Wu+f,\qquad v^*=M_\xi^{-1}(z/\sigma-v),\qquad \beta=\sigma\big(u+\Gamma W'v^*\big).\tag{5}
$$

*Verification.* Substituting $v$ and $v^*$,

$$
\beta=\Gamma W'M_\xi^{-1}z+\sigma\big(I_p-\Gamma W'M_\xi^{-1}W\big)u-\sigma\Gamma W'M_\xi^{-1}f =: \mu+\sigma g .
$$

The mean is $\mu$ by the identity above. For the covariance, using $\operatorname{Var}(u)=\Gamma$, $\operatorname{Var}(f)=I_N$ and $u\perp f$,

$$
\operatorname{Var}(g)=\big(I_p-\Gamma W'M_\xi^{-1}W\big)\Gamma\big(I_p-\Gamma W'M_\xi^{-1}W\big)'+\Gamma W'M_\xi^{-2}W\Gamma
=\Gamma-\Gamma W'M_\xi^{-1}W\Gamma,
$$

where the last step expands the first term and combines the two quadratic pieces via $W\Gamma W'+I_N=M_\xi$:

$$
\Gamma W'M_\xi^{-1}W\Gamma W'M_\xi^{-1}W\Gamma+\Gamma W'M_\xi^{-2}W\Gamma=\Gamma W'M_\xi^{-1}\big(W\Gamma W'+I_N\big)M_\xi^{-1}W\Gamma=\Gamma W'M_\xi^{-1}W\Gamma .
$$

Hence $\beta\sim\mathrm{N}(\mu,\sigma^2\Sigma)$, as required. Note that $z$ and $\sigma$ enter (5) only through the vector $z/\sigma$, and that the $p$-vector $u$ is used only through $Wu$: the update costs $O(Np)$ plus $N\times N$ linear algebra, and never forms or inverts a $p\times p$ matrix.

#### Cost

Forming $M_\xi$ requires $WDW'$, costing $O(N^2p)$; on top of that, $|M_\xi|$, $z'M_\xi^{-1}z$ and the solve $M_\xi v^*=(z/\sigma-v)$ each cost $O(N^3)$. For $p>N$ the $O(N^2p)$ product dominates every other operation in (4) — this is the bottleneck the approximation removes.

### 2.2. Approximate algorithm

**Idea.** For a coordinate to be shrunk to near zero, the precision $\xi\eta_j$ must be large, i.e. $\Gamma_j=\xi^{-1}\eta_j^{-1}$ small; then the $j$-th column of $W$ contributes almost nothing to $\xi^{-1}WDW'$. Once the chain has begun to converge, $\xi^{-1}WDW'$ is therefore well approximated by hard-thresholding $D$:

$$
M_\xi\approx M_{\xi,\delta}:=I_N+\xi^{-1}WD_\delta W',\qquad
D_\delta=\operatorname{diag}\big(\eta_j^{-1}\mathbf{1}\{\xi_{\min}^{-1}\eta_j^{-1}>\delta\}\big),\tag{6}
$$

for a "small" $\delta$, where $\xi_{\min}=\min(\xi,\xi^*)$ in step 2 of (4). The active set is

$$
S=\{j:\xi_{\min}^{-1}\eta_j^{-1}>\delta\},\qquad s_\delta=|S|=\sum_{j=1}^p\mathbf{1}\{\xi_{\min}^{-1}\eta_j^{-1}>\delta\},\tag{7}
$$

so that $M_{\xi,\delta}=I_N+W_S\Gamma_SW_S'$, written $M_S$ for short, where $W_S$ and $\Gamma_S=\xi^{-1}D_S$ are the column/diagonal sub-matrices indexed by $S$.

*Why $\xi_{\min}$.* The set must contain every coordinate that is active under *either* $\xi$ or the proposal $\xi^*$, because the same $D_\delta$ is used in both the numerator and the denominator of the MH ratio in step 2 of (4). Equivalently the criterion is $\eta_j^{-1}\max(\xi^{-1},\xi^{*-1})>\delta$.

The approximate algorithm $\mathcal{P}_\epsilon$ is (4) with exactly two changes:

1. $M_\xi$ is replaced by $M_{\xi,\delta}$ everywhere it appears in (4);
2. in the final step of (5), $DW'$ is replaced by $D_\delta W'$.

Coordinates outside $S$ are **not** deleted from the model: they are still sampled, from a Gaussian approximation to the exact full conditional, so their values keep updating, the active set changes from iteration to iteration, and posterior uncertainty about which variables are signals is retained.

**Cost.** With $S$ as in (7), $W D_\delta W'=W_SD_SW_S'$ costs $O(N^2s_\delta)$. When $s_\delta<N$, $M_{\xi,\delta}$ need not be formed at all; the Woodbury identity gives

$$
M_{\xi,\delta}^{-1}=\big(I_N+\xi^{-1}W_SD_SW_S'\big)^{-1}=I_N-W_S\big(\xi D_S^{-1}+W_S'W_S\big)^{-1}W_S',
$$

so $z'M_{\xi,\delta}^{-1}z$ and $M_{\xi,\delta}^{-1}(z/\sigma-v)$ require only $s_\delta\times s_\delta$ solves, at cost $O(s_\delta^3\vee s_\delta N)$. The determinant is handled by an SVD of $W_SD_S^{1/2}$, $O(s_\delta^2N)$: the non-zero eigenvalues of $W_SD_SW_S'$ equal those of $D_S^{1/2}W_S'W_SD_S^{1/2}$, so if $s_1,\dots,s_{s_\delta}$ are the singular values of $W_SD_S^{1/2}$ then

$$
|M_{\xi,\delta}|=\prod_{i=1}^{s_\delta}\big(1+\xi^{-1}s_i^2\big),
$$

the remaining $N-s_\delta$ eigenvalues of $M_{\xi,\delta}$ being $1$. Adding the $O(Np)$ cost of $Wu$ in (5), the per-step cost of the approximate algorithm for $s_\delta<N$ is

$$
O\big((s_\delta^2\vee p)\,N\big),
$$

i.e. of the same order as one iteration of coordinate descent for the Lasso.

**Effect on $\beta$.** With $\Gamma_\delta=\xi^{-1}D_\delta$ and $M_\delta:=M_{\xi,\delta}$, the modified update (5) sets

$$
\beta=\Gamma_\delta W'M_\delta^{-1}z+\sigma\big(u-\Gamma_\delta W'M_\delta^{-1}v\big),
$$

which is still Gaussian: $\beta\sim\mathrm{N}(\mu_\delta,\sigma^2\Sigma_\delta)$ with

$$
\mu_\delta=(\mu_S;0_{(p-s_\delta)\times1}),\qquad \mu_S=(W_S'W_S+\Gamma_S^{-1})^{-1}W_S'z,\tag{8}
$$

and (see Appendix A)

$$
\Sigma_\delta=\begin{bmatrix}(W_S'W_S+\Gamma_S^{-1})^{-1}+\Delta & -\Gamma_SW_S'M_S^{-1}W_{S^c}\Gamma_{S^c}\\ -\Gamma_{S^c}W_{S^c}'M_S^{-1}W_S\Gamma_S & \Gamma_{S^c}\end{bmatrix},\qquad
\Delta=\Gamma_SW_S'M_S^{-1}W_{S^c}\Gamma_{S^c}W_{S^c}'M_S^{-1}W_S\Gamma_S,\tag{9}
$$

equivalently $\Sigma_\delta=\Gamma-X-X'+Y$ with $X=\Gamma_\delta W'M_\delta^{-1}W\Gamma$, $X'=X^\top$, $Y=\Gamma_\delta W'M_\delta^{-1}MM_\delta^{-1}W\Gamma_\delta$ and $M=I_N+W\Gamma W'$.

Three consequences, in words:

- $\mathbb{E}(\beta_{S^c})=0$: thresholded coordinates are centred at zero, as in a model containing only the active set.
- The marginal of $\beta_S$ is $\mathrm{N}\big((W_S'W_S+\Gamma_S^{-1})^{-1}W_S'z,\ \sigma^2[(W_S'W_S+\Gamma_S^{-1})^{-1}+\Delta]\big)$. It differs from the full conditional that would obtain under the active-only model by the positive semidefinite term $\Delta$, which arises because the thresholded coordinates still propagate through $v=Wu+f$. Since $\Delta$ is small whenever the thresholded $\Gamma_j$ are small, the marginal is close to — but not equal to — the active-only full conditional.
- The cross-block covariance $-\Gamma_SW_S'M_S^{-1}W_{S^c}\Gamma_{S^c}$ is retained, so $\beta_S$ and $\beta_{S^c}$ remain dependent.

---

## Appendix A. Derivation of (8) and (9)

Since $\Gamma_\delta$ is diagonal with $\Gamma_\delta=\operatorname{diag}(\Gamma_j\mathbf{1}\{j\in S\})$, we have $\Gamma_\delta W'=(\Gamma_SW_S';\ 0_{(p-s_\delta)\times N})$ and $W\Gamma_\delta W'=W_S\Gamma_SW_S'$, hence $M_\delta=M_S$. The mean is immediate: $\mu_\delta=(\mu_S;0)$ with

$$
\mu_S=\Gamma_SW_S'(I_N+W_S\Gamma_SW_S')^{-1}z=(W_S'W_S+\Gamma_S^{-1})^{-1}W_S'z,
$$

the second equality being Woodbury.

For the covariance, $\beta-\mu_\delta=\sigma(u-\Gamma_\delta W'M_\delta^{-1}v)$ and

$$
u-\Gamma_\delta W'M_\delta^{-1}v=\big(u_S-\Gamma_SW_S'M_S^{-1}v;\ u_{S^c}\big),
$$

because the second block of $\Gamma_\delta W'$ vanishes. Write $v=Wu+f=W_Su_S+W_{S^c}u_{S^c}+f$; since $\Gamma$ is diagonal, $u_S\perp u_{S^c}$ and

$$
\operatorname{cov}(u_S,v)=\Gamma_SW_S',\qquad \operatorname{cov}(u_{S^c},v)=\Gamma_{S^c}W_{S^c}'.
$$

The three blocks follow.

1. $(S,S)$. Using $\operatorname{Var}(v)=M=M_S+W_{S^c}\Gamma_{S^c}W_{S^c}'$ (the full $M$, not $M_S$),

$$
\operatorname{cov}\big(u_S-\Gamma_SW_S'M_S^{-1}v\big)=\Gamma_S-\Gamma_SW_S'M_S^{-1}W_S\Gamma_S+\Delta=(W_S'W_S+\Gamma_S^{-1})^{-1}+\Delta,
$$

with

$$
\Delta=\Gamma_SW_S'M_S^{-1}W_{S^c}\Gamma_{S^c}W_{S^c}'M_S^{-1}W_S\Gamma_S .
$$

The extra $\Delta$ appears because $\operatorname{Var}(v)=M$ is the *full* $M$, not $M_S$: the terms combine as

$$
\Gamma_S-2\Gamma_SW_S'M_S^{-1}W_S\Gamma_S+\Gamma_SW_S'M_S^{-1}M_SM_S^{-1}W_S\Gamma_S+\Delta=\Gamma_S-\Gamma_SW_S'M_S^{-1}W_S\Gamma_S+\Delta=(W_S'W_S+\Gamma_S^{-1})^{-1}+\Delta,
$$

where the last equality is Woodbury and the single surviving $\Delta$ is the contribution of the thresholded block $W_{S^c}\Gamma_{S^c}W_{S^c}'$ inside $\operatorname{Var}(v)$.
2. $(S,S^c)$. By $u_S\perp u_{S^c}$,

$$
\operatorname{cov}\big(u_S-\Gamma_SW_S'M_S^{-1}v,\ u_{S^c}\big)=-\Gamma_SW_S'M_S^{-1}W_{S^c}\Gamma_{S^c}.
$$

3. $(S^c,S^c)$. $\operatorname{cov}(u_{S^c})=\Gamma_{S^c}$.

Equivalently, in full-matrix form,

$$
\Sigma_\delta=\Gamma-X-X'+Y,\qquad X=\Gamma_\delta W'M_\delta^{-1}W\Gamma,\quad X'=\Gamma W'M_\delta^{-1}W\Gamma_\delta,\quad Y=\Gamma_\delta W'M_\delta^{-1}MM_\delta^{-1}W\Gamma_\delta,
$$

which agrees with the block form (9). (The original printing replaced $X+X'$ by $2\Gamma W'M_\delta^{-1}W\Gamma_\delta$, which is not symmetric and hence cannot be a covariance.)

---

## Appendix S1. Exact rejection sampler for the local scales

Step 1 of (4) requires draws from

$$
h_\varepsilon(t)=C_\varepsilon\frac{e^{-\varepsilon t}}{1+t},\qquad t>0,\qquad \varepsilon=m_j=\frac{\xi\beta_j^2}{2\sigma^2},
$$

with normalizing constant $C_\varepsilon=e^{-\varepsilon}/E_1(\varepsilon)$, where $E_1(x)=\int_x^\infty e^{-t}/t\,dt=\Gamma(0,x)$ is the exponential integral; $C_\varepsilon$ is increasing in $\varepsilon$, with $C_1\approx1.6$ and $C_\varepsilon<1$ for $\varepsilon<0.40$. The negative log-density up to constants is

$$
f(x)=\varepsilon x+\log(1+x),\qquad x>0,
$$

which is increasing and concave.

**Envelope.** Fix $0<a<1<b$ and set $A=f(a/\varepsilon)$, $I=f(1/\varepsilon)$, $B=f(b/\varepsilon)$,

$$
\lambda_2=\frac{I-A}{(1-a)/\varepsilon},\qquad \lambda_3=\frac{B-I}{(b-1)/\varepsilon},\qquad
f_L(x)=\begin{cases}\log(1+x), & x\in[0,a/\varepsilon),\\ A+\lambda_2(x-a/\varepsilon), & x\in[a/\varepsilon,1/\varepsilon),\\ I+\lambda_3(x-1/\varepsilon), & x\in[1/\varepsilon,b/\varepsilon),\\ B+\varepsilon(x-b/\varepsilon), & x\ge b/\varepsilon.\end{cases}
$$

$f_L$ is increasing, piecewise linear on $[a/\varepsilon,\infty)$, equals $\log(1+x)$ on $[0,a/\varepsilon)$, interpolates $f$ at $a/\varepsilon,1/\varepsilon,b/\varepsilon$, and continues with slope $\varepsilon$ beyond $b/\varepsilon$. It has a downward jump at $a/\varepsilon$ and is continuous elsewhere. On the first piece $f_L\le f$ by construction; on $[a/\varepsilon,\infty)$ concavity of $f$ forces the chords to lie below $f$. Hence $f_L\le f$ globally and $h_L\propto e^{-f_L}$ is an upper envelope of $h_\varepsilon$. The tail piece is calibrated so that $\Pr_{h_\varepsilon}(T>b/\varepsilon)$ obeys the same $e^{-b}$ bound as an $\mathrm{Expo}(\varepsilon)$ variate, up to the factor $C_\varepsilon/b$.

**Sampling.** With $\nu=\int_0^\infty e^{-f_L}$, $h_L=e^{-f_L}/\nu$ is the four-component mixture

$$
h_L=\frac{\nu_1}{\nu}h_1+\frac{\nu_2}{\nu}h_2+\frac{\nu_3}{\nu}h_3+\frac{\nu_4}{\nu}h_4,
$$

$$
\nu_1=\log(1+a/\varepsilon),\quad
\nu_2=\lambda_2^{-1}e^{-A}\big[1-e^{-(I-A)}\big],\quad
\nu_3=\lambda_3^{-1}e^{-I}\big[1-e^{-(B-I)}\big],\quad
\nu_4=\varepsilon^{-1}e^{-B},
$$

where $h_1(x)\propto\mathbf{1}_{[0,a/\varepsilon)}(x)/(1+x)$ and $h_2,h_3,h_4$ are truncated exponentials on $[a/\varepsilon,1/\varepsilon)$, $[1/\varepsilon,b/\varepsilon)$ and $[b/\varepsilon,\infty)$ with rates $\lambda_2,\lambda_3,\varepsilon$ and truncation masses $H_2=1-e^{-(I-A)}$, $H_3=1-e^{-(B-I)}$.

All four are sampled by inversion. For $\mathrm{Expo}(\lambda,\underline{v},\bar v)$ with mass $H=1-e^{-\lambda(\bar v-\underline{v})}$,

$$
x=\underline{v}+\frac{-\log(1-uH)}{\lambda},\qquad u\sim\mathrm{U}(0,1),
$$

and for $h_1$ the cdf is $\log(1+x)/\nu_1$, so $x=e^{u\nu_1}-1=(1+a/\varepsilon)^u-1$. The rejection step is

$$
\text{draw } z\sim h_L,\ u\sim\mathrm{U}(0,1);\quad\text{accept } z \text{ if } u<e^{-(f-f_L)(z)} .
$$

Because $f_L\le f$, the sampler is exact, with acceptance probability $\int_0^\infty e^{-f(x)}dx\big/\nu=e^{\varepsilon}E_1(\varepsilon)/\nu$; it is controlled by the choice of $a$ and $b$, and the original illustrates $\varepsilon=10^{-4}$, $a=1/5$, $b=10$ (Figure S1). The construction is stated in the original for $\varepsilon\in(0,1)$.

**Truncated variant.** For the modified prior with lower truncation $\eta_j\ge b$ (analysis only; the algorithm above uses $b=0$), the target is $h_\varepsilon$ restricted to $(b,\infty)$. Rejecting draws $\eta_j\le b$ from the untruncated sampler is exact but needs $\sim e^{\tilde m_jb}$ retries when $\tilde m_jb\gg1$; for $\tilde m_j>1$ propose instead from the exponential restricted to $(b,\infty)$,

$$
t=b-\frac{\log(1-u)}{\tilde m_j},\qquad u\sim\mathrm{U}(0,1),
$$

and accept with probability $(1+b)/(1+t)$ — the ratio of the target to this proposal is proportional to $(1+t)^{-1}$, maximized at $t=b$. Here $\tilde m_j$ is $m_j$ evaluated at the current state, and this $b$ is the prior's truncation parameter, unrelated to the envelope breakpoint $b$ of the preceding paragraph.
