
min f(x): || $\nabla$f($x_1$) - $\nabla$f($x_2$)||$_2$ $\leq$ L . ||$x_1 - x_2$||
$x_{t+1} = x_t - \eta \nabla f(x_t)$
f(x) is convex
$min_t || \nabla f(x_t)||_2 \leq O(\frac{1}{\sqrt t})$
$||x_{t+1} - x*||_2^2 =^{GD} ||x_t - \eta \nabla f(x+t) - x*||_2^2$
						= $||x_t - x*||_2^2 - 2\eta <\nabla f(x_t), x_t - x*>$
From L-Lipschitz: $\frac{1}{L} || \nabla f(x_1) - \nabla f(x_2)||_2 \leq <\nabla f(x_1) - \nabla f(x_2), x* - x_t>$
At x1=x* and x2=xt
$\frac{1}{L} ||\nabla f(x_2)||_2^2 \leq <-\nabla f(x_t), x* - x_t>$
$<\nabla f(x_t), x* - x_t> \leq \frac{1}{L} ||\nabla f(x_2)||_2^2$
$||x_{t+1} - x*||_2^2 \leq || x_t - x*||_2^2 + \eta^2 || \nabla f(x_t)||_2^2 - \frac{2\eta}{L} ||\nabla f(x_t)||_2^2$ 
					$=|| x_t - x*||_2^2 - \eta (\frac{2}{L} - \eta) ||\nabla f(x_t)||_2^2$
This shows that $f(x_{t+1}) \leq f(x_t))$ which is similar to non-convex optimizations
which only tells us that it always climbs down from the positive values
* From Lipschitz, 
$f(x_t) \leq f(x*) - \eta (\frac{2}{L} - \eta) ||\nabla f(x_t)||_2^2$
* From Convex optimization as seen above,

$f(x_t) - f(x*) \leq <\nabla f(x_t), x_t - x*>$
From Cauchy Scwartz,
$f(x_t) - f(x*) \leq ||x_t - x*||_2 . ||\nabla(x_t)||_2$

Combining both,
$f(x_{t+1}) - f(x*) \leq f(x_t) - f(x*) - \eta (\frac{2}{L} - \eta) \frac{f(x_t) - f(x*)^2}{||x_t - x*||_2^2}$
With $\eta = \frac{1}{L}$

$f(x_t) - f(x*) \leq \frac{2L(f(x_0)-f(x*).||x_0-x*||_2^2)}{2L||x_0 - x*||_2^2 + \textbf{T} (f(x_0) - f(x*))}$ = O(1/T)
$\implies$ O(1/T) making it a faster convergence than non-convex optimization

### Strong Convexity
A strong convexity means that the curve is **always** steep enough to make enough progress, ie, we can lower bound to a quadratic function

##### Characteristics
* $\nabla^2 f(x) \succeq \mu I$
	* Lowerbound

### Gains from this

$min f(x):$ L - Smooth, 
				$\mu$ - Strongly convex
$||x_{t+1} - x*||_2^2 \leq || x_t - x*||_2^2 + \eta^2 || \nabla f(x_t)||_2^2 - \frac{2\eta}{L} ||\nabla f(x_t)||_2^2$ 
$<\nabla f(x) - \nabla f(y)> \geq \frac{\mu L}{\mu + L} ||x - y||_2^2 + \frac{1}{\mu + L} ||\nabla f(x) - \nabla f(y)||_2^2$
at y = x* and x = x_t
$<-\nabla f)x_t), x* - x_t> \geq \frac{\mu L}{\mu + L} ||x_t - x*||_2^2 + \frac{1}{\mu + L} ||\nabla f(x)_t||_2^2$
$||x_{t+1} 0 x*||_2^2 \geq .. (1 - \frac{2\mu \eta L}{\mu +L})||x_t - x*||_2^2$
if  $\eta \leq \frac{2}{\mu + L}$
	$\leq (1 - \frac{2\mu \eta L}{\mu +L})||x_t - x*||_2^2$
	= O($log \frac{1}{\epsilon}$)

# Lecture 8 - Convex optimizations contd.(15/09/22)
* Gradient descent has an upper bound of $O(\frac{1}{T})$ and a lower bound of $O(\frac{1}{T^2})$
* $||x_t - x*||_2^2 \leq (\frac{k-1}{k+1})^T ||x_0 - x*||_2^2$  such that $\kappa=\frac{L}{\mu}$ 
	* L -> Smoothness
	* $\mu$ -> Strongly convex
* minimizing f(x) brings a upper bound of O($\kappa log \frac{1}{\epsilon}$) vs a Lower bound of O($\sqrt{\kappa}log\frac{1}{\epsilon}$)
* PL inequality with Lipschitz aids gradient descent with non-convex optimizations
* We can also project a point to the nearest convex set (Projecting gradient descent at each time step to maintain the constraints with the problems)
# Lecture 9 - Convex optimizations contd.
### Convex sets
All points in the set follows $\alpha x + (1-\alpha) y \in C$
$\Pi_C(x) = arg min_{y \in C} \parallel x - y \parallel_2^2$
	Euclidean distance is used in general for the projection 

#### Properties of Convex sets

$\parallel x - \Pi_C(x) \parallel_2^2 \leq \parallel x - y \parallel_2^2$
$<\Pi_C(x) - y, \Pi_C(x) - x> \leq 0$
$\parallel \Pi_C(y) - \Pi_C(x) \parallel_2^2 \leq \parallel x - y \parallel_2^2$

#### Projected Gradient Descent
$x_{t+1} = \Pi_C(x_t - \eta \nabla f(x_t))$
Step 1: $x' = x_t - \eta \nabla f(x_t)$
Step 2: $x_{t+1} = \Pi_C(x')$

**DEMO TIME**


# CH4: Frank Wolfe Conditional Gradient Method

$min_{x \in C}f(x)$: L Smoothness

1. $f(x) \leq f(x_t) + <\nabla f(x_t), x - x_t> + \frac{L}{2} \parallel x - x_t \parallel_2^2$
2. $min_{x \in C} [f(x_t) + <\nabla f(x_t), x - x_t> + \frac{L}{2} \parallel x - x_t \parallel_2^2]$
	$min_{x \in C} [\frac{1}{L}\parallel \nabla f(x_t) \parallel_2^2 + <\nabla f(x_t), x - x_t> + \frac{L}{2} \parallel x - x_t \parallel_2^2]$ (Since gradient at xt is independent of x)
	$min_{x \in C} [\frac{L}{2} \parallel x - (x_t - \frac{1}{L} \nabla f(x_t))\parallel_2^2]$
	$min_{x \in C} [ \parallel x - (x_t - \frac{1}{L} \nabla f(x_t))\parallel_2^2]$
	$min_{x \in C} \parallel x - Y\parallel_2^2$ Where Y = $(x_t - \frac{1}{L} \nabla f(x_t))$
	
#### Frank Wolfe algorithm
1. $f(x) = f(x_t) + <\nabla f(x_t), x - x_t>$
	1. We can use this since its constrained by convexity and L-Smoothness
2. $min_{x \in C} [f(x_t) + <\nabla f(x_t), x - x_t>]$
3. $min_{x \in C} [<\nabla f(x_t), x - x_t>]$ -> Answer is $s_t$
	1. This leads to the corners of the set which are the sparse solutions

# Lecture 10: Frank Wolfe contd.

4. $x_{t+1} = x_t + \eta d_t = x_t + \eta_t(s_t - x_t)$
	1. $= (1 - \eta_t)x_t + \eta_t . s_t$
5. $\eta_t = \frac{2}{t+2}$
6. f: C -> R, L-Smooth, $\exists x* \in C$. Then f satisfies:
	1. $f(x_{t+1}) - f(x*) \leq \frac{2 LD^2}{t+2} = O(\frac{1}{t})$ 
	2. where $\eta_t = \frac{2}{t+2}$
	3. D = $max_{x, y in C} \parallel x - y \parallel_2$ (Diameter of C)
		1. This projection works efficiently per iteration than the projected gradient descent
		2. O(1/t) vs O(1/$\sqrt t$)
		3. Q. Assume that $C = {x \in R: \parallel x \parallel_1 \leq \lambda}$
			1. FW: 
				1. $s_t = min {<\nabla f(x_t), x>} ST \parallel x \parallel_1 \leq \lambda$
				2. $x_{t+1} = (1 - \eta_t)x_t + \eta_t . s_t$ where $\eta_t = \frac{2}{t+2}$
			2. VERSUS Projected Gradient Descent
				1. $x'_{t+1} = x_t -\eta \nabla f(X_t)$
				2. $x_{t+1} = min \parallel x - x'_{t+1} \parallel_2^2 ST \parallel x \parallel \leq \lambda$
					1. O(plogp) for the projection step(2)
7. $s_t = min {<\nabla f(x_t), x>} ST \parallel x \parallel_1 \leq \lambda$
	1. Without constraint $<\nabla f(x_t), x>$
	2. Minimize it as much as possible
		1. Go in the opposite direction as much as we can =>  <., .> -> - $\infty$
		2. <a, x> = $\sum a_i x_i$ with $\parallel x \parallel_1 \leq \lambda$ 
			1. Which minimizes to $x_i$ = -$\lambda$
				1. $s_t = -\lambda sign(<\nabla f(x_t), e_i*>).e_i*$
			2. We will traverse the vector to find the max value here
			3. Makes it O(p) for the projection step
8. $min_{x \in R^{p*p}} f(x)$ ST $\parallel X \parallel_* \leq 1$
	1. $\parallel X \parallel_* = \sum \sigma_i(x)$
	2. A low rank matrix appears
		1. Projected gradient descent:
			1.  $x'_{t+1} = x_t -\eta \nabla f(X_t)$
			2. $x_{t+1} = min \parallel x - x'_{t+1} \parallel_F^2 ST \parallel x \parallel_* \leq 1$
				1. $O(p^3)$ step making it very expensive
		2. Frank Wolfe:
			1. $s_t = min {<\nabla f(x_t), x>} ST \parallel x \parallel_* \leq 1$
			2. $x_{t+1} = (1 - \eta_t)x_t + \eta_t . s_t$ where $\eta_t = \frac{2}{t+2}$
# Lecture 11: Frank Wolfe method contd. and power-iteration
1. $s_t = -1 . u_1 v_1^T$ where u and v are left and right singular vectors of $\nabla f(x_t)$
2. This shows that the second step jus adds a rank 1 matrix per iteration.
	1. Makes it a $O(p^2)$
3. *The frank-wolfe algorithm performs worse than projected gradient descent when dealing with non-convex optimizations.*

### Power iteration method

Let $A \in R^{p*p}$ real, symmetric matrix
$q \in R^p$ is the variable to find the largest eigenvector
$A = U \Lambda U^T$ (Eigenvalue decomposition)
$diag(\Lambda) = (\lambda_1, \lambda_2, ..., \lambda_p)\ where\ |\lambda_1| \geq |\lambda_2| \geq ... \geq |\lambda_p|$
POWER ITERATION:
	1. $q_{t+1} = \frac{A q_t}{\parallel Aq_t \parallel_2}$
		1. $q_{t+1} = \frac{A^t q_0}{\parallel A^t q_0 \parallel_2}$
	2. max $q^T A q$
		1. $\nabla = Aq$
			1. We keep calculating the gradient and using that in coming steps
		2. ST $\parallel q \parallel_2 = 1$
	3. Equal complexity compared to gradient descent
	4. Non-convex operation but still converges in linear time O($log(\frac{1}{\epsilon})$)
		1. Per iteration time complexity of O($p^2$)
	5. For getting all eigenvalues, we can run this p times making it $O(p^3)$
	6. If q is orthogonal to the required eigenvector, the algorithm never converges(but never happens in random init)

# Lecture 12 : Ch5 - Beyond gradients

### Newton's method
1. From Taylors series expansion in second order:
	1. $f(x + \Delta x) = f(x) + <\nabla f(x), \Delta x> + \frac{1}{2} <\nabla^2 f(x) \Delta x, \Delta x>$
		1. $f(x + \nabla x) \leq f(x) + <\nabla f(x), \nabla x> + \frac{L}{2} \parallel (\nabla x)\parallel_2^2$ 
			1. From lipschitz where $\nabla^2 \leq LI$
		2. We can decide to compute the hessian instead of using lipschitz as an upper bound
	2. If we equate derivatives to zero:
		1. $\nabla_{\Delta x} f(x + \Delta x) = 0 \implies \Delta x = -(\nabla^2 f(x))^{-2} \nabla f(x)$
		2. $H_t = \nabla^2 f(x_t)$
	3. For an iteration, by newtons method:
		1. $x_{t+1} = x_t - \eta H_t^{-1}\nabla f(x_t)$
			1. H is not invertible if the matrix is positive semi-definite as the determinant of Hessian becomes 0 at the points where eigenvalues defines as 0
		2. Example: $min \frac{1}{2} \parallel b - Ax \parallel_2^2, A \in R^{pxp}$
			1. $\nabla^2 f(x) = A^TA$
			2. $x_{t+1} = x_t - \eta (A^TA)^{-1} (-A^T(b - Ax_t))$
			3. $x_{t+1} = x_t + \eta (A^TA)^{-1}(A^TAx* - A^TAx_t) = x_t + \eta(x* - x_t)$
				1. If $\eta = 1$
				2. $x_{t+1} = x*$
					1. Which means that it can converge in a single step
				2. If we are close enough ($\parallel x_0 - x* \parallel_2 < \frac{2\mu}{3M}$ where $\nabla^2f(x*) \succcurlyeq \mu I$ and $\parallel \nabla^2 f(x*) - \nabla f(x) \parallel_2 \leq M \parallel x - y \parallel_2$)
					1. Newtons method converges quadratically with O(log log($\frac{1}{\epsilon}$)) from the equation $\parallel x_{t+1} - x* \parallel_2 \leq \frac{M \parallel x_t - x* \parallel_2^2}{2(\mu - M\parallel x_t - x* \parallel_2)}$
						1. Due to the norm squares compared to just norm
					2. But before it is close enough, it starts sublinear

# Chapter 13: BFGS (Quasi newtons method)

1. Approximate $H_{t+1}$ to $g_{t+1}$ such that
	1. $\nabla g_{t+1}(0) = \nabla f(x_{t+1})$
	2. $\nabla g_{t+1}(-\Delta x) = \nabla f(x_t)$
2. From 2. and Taylors expansion,
	1. $H_{t+1} \Delta x = \nabla f(x_{t+1}) - \nabla f(x_t)$[ Secant method ]
	2. $\Delta x^T H_{t+1} \Delta x = \Delta x^T (\nabla f(x_{t+1}) - \nabla f(x_t)) \succ 0$ if H is positive definite (Assumption from convex)
3. BFGS states
	1. $min_{B \succ 0} \parallel B - B_t \parallel_F^2$
	2. ST. $B = B^T$
	3. $\Delta x = B(\nabla f(x_{t+1}) - \nabla f(x_t))$
		1. Has a closed form solution
		2. $B_{t+1} = (I - \frac{s_t y_t^T}{s_t^T y_t}) B_t (I - \frac{y_t s_t^T}{s_t^T y_t}) + \frac{s_t s_t^T}{s_t^Ty_t}$
		3. Where $y_t = \nabla f(x_{t+1}) - \nabla f(x_t)$, $s_t = \Delta x$
4. SR-1 method works on Rank 1 instead of rank 2
	1. $H_{t+1} = H_t + \sigma v v^T$ where $\sigma = +- 1$ (and secant eq satisfied)
	2. SR1 shows that
		1. $B_t+1 = B_t + \frac{(s_t - B_t y_t)(s_t - B_ty_t)^T}{(s_t - B_ty_t)^T y_t}$
		2. Useful for indefinite Hessian approximations in non-convex (BFGS needs convex)
		3. Failed neural network training tho :(

For strongly convex problems
1. GD: $\parallel x_{k+1} - x* \parallel_2 \leq c \parallel x_k - x* \parallel$ , 0<c<1   -> Linear
2. Newtons: $\parallel x_{k+1} - x* \parallel_2 \leq c \parallel x_k - x* \parallel^2$ , 0<c<1  -> Quadratic
3. BFGS: $\parallel x_{k+1} - x* \parallel_2 \leq c_k \parallel x_k - x* \parallel$ , $c_k \implies 0$   -> Super-Linear
	1. ck decreases over iteration but not as fast as square in newtons
4. ExtraGradient($min_x max_y f(x, y)$): $X_{t+1} = x_t - \eta \nabla f(x_t - \gamma \nabla f(x_t))$ (Out of syllabus)
	1. Faster than GD in this scenario (GD diverges in minmax) and constant diff with minimization problems

What to do when not even gradient is available?
Bisection method, genetic agorithms, simulated annealing, etc

Finite differences method is used instead
Mainly used in adversarial training

# Chapter 14: Accelerated methods

### Momentum Acceleration

We move based on
1. $x_{t+1} = x_t - \eta \nabla f(x_t) + \beta (x_t - x_{t-1})$
, including a momentum factor
This has same time complexity as gradient descent but doubles the space it takes.

When we take L-smooth and $\mu$ convex,
$$\begin{bmatrix}
x_{t+1} - x* \\
x_t - x* 
\end{bmatrix} = 
\begin{bmatrix} 
x_t - \eta \nabla f(x_t) + \beta (x_t - x_{t-1}) - x* \\
x_t - x*
\end{bmatrix} $$$$= 
\begin{bmatrix}
x_t + \beta(x_t - x_{t-1} - x*) \\
x_t - x*
\end{bmatrix} - \eta
\begin{bmatrix}
\nabla f(x_t) \\
0
\end{bmatrix}$$= $$\parallel\begin{bmatrix}
x_{t+1} - x* \\
x_t - x* 
\end{bmatrix}\parallel_2 = 
\parallel \begin{bmatrix}
(1+\beta)I & -\beta I \\
I & 0
\end{bmatrix} \begin{bmatrix}
x_t - x* \\
x_{t-1} - x*
\end{bmatrix} - \eta \begin{bmatrix}
\nabla^2 f(x_t)(x_t - x*) \\
0
\end{bmatrix} \parallel_2
$$
$$
\parallel\begin{bmatrix}
x_{t+1} - x* \\
x_t - x* 
\end{bmatrix}\parallel_2 = 
\parallel \begin{bmatrix}
(1+\beta)I - \eta \nabla^2 f(x_t) & -\beta I \\
I & 0
\end{bmatrix} \begin{bmatrix}
x_t - x* \\
x_{t-1} - x*
\end{bmatrix} \parallel_2
$$

Applying cauchy schwartz to this:

$$

\parallel\begin{bmatrix}
x_{t+1} - x* \\
x_t - x* 
\end{bmatrix}\parallel_2 \ \leq \ 
\parallel \begin{bmatrix}
(1+\beta)I - \eta \nabla^2 f(x_t) & -\beta I \\
I & 0
\end{bmatrix} \parallel_2 \ .\parallel \begin{bmatrix}
x_t - x* \\
x_{t-1} - x*
\end{bmatrix} \parallel_2
$$
Where the first matrix becomes the spectrum of a matrix (it depends on the eigenvalues)
$\phi(\nabla^2f(.), \beta, \eta)$

$\nabla^2 f(.) > 0, \nabla^2f(.)=U \Lambda U^T$
$$ 
\parallel \begin{bmatrix}
(1+\beta)I - \eta U \Lambda U^T & -\beta I \\
I & 0
\end{bmatrix} \parallel_2 = \parallel \begin{bmatrix}
U^T & 0 \\
0 & U^T
\end{bmatrix}
\begin{bmatrix}
... \\ ...
\end{bmatrix}
\begin{bmatrix}
U & 0 \\
0 & U
\end{bmatrix} \parallel_2

$$
$$
= \parallel \begin{bmatrix}
(1 + \beta)I - \eta \Lambda & -\beta I \\
I & 0
\end{bmatrix} \parallel_2 = max_i \parallel \begin{bmatrix}
1 + \beta - \eta \lambda_i & -\beta \\
1 & 0
\end{bmatrix} \parallel_2
$$
Since everything in LHS is diagonal

$\leq max{|1 - \sqrt{\eta \mu} | , |1 - \sqrt{\eta L}}|$

To solve,
$\xi^2 - (1 + \\beta - \mu \lambda_i)\xi + \beta = 0$

$\eta = \frac{4}{(\sqrt{\mu} + \sqrt{L})^2}, \beta = \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}$
$\lambda \in [\mu, L]$


$$
\parallel\begin{bmatrix}
x_{t+1} - x* \\
x_t - x* 
\end{bmatrix}\parallel_2  = \leq max{|1 - \sqrt{\eta \mu} | , |1 - \sqrt{\eta L}}| \begin{bmatrix}
x_t - x* \\
x_{t-1} - x*
\end{bmatrix}
$$


$$
\parallel\begin{bmatrix}
x_{t+1} - x* \\
x_t - x* 
\end{bmatrix}\parallel_2  = \leq (\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1 })\begin{bmatrix}
x_t - x* \\
x_{t-1} - x*
\end{bmatrix}
$$

