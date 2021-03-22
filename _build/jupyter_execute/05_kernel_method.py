# Kernel Method

## Polynomial Regression

basic linear regression with one variable:

$$y=\theta_{0} + \theta_{1}x$$

what if linear model could not nicely fit training examples?<br>
we can naturally extend linear model to polynomial model, for example:

$$y=\theta_{0} + \theta_{1}x + \theta_{2}x^{2} + \theta_{3}x^{3}$$

this method can be conclude as: map original attibutes x to some new set of quantities $\phi(x)$ (called features), and use the same set of model.

### least mean squares with features
let $\phi : \mathbb{R}^{d} \to \mathbb{R}^{p}$ be a feature map, then original batch gradient descent:

$$\theta := \theta + \alpha\sum_{i=1}^{n}(y^{(i)} - \theta^{T}x^{(i)})x^{(i)}$$

using features:

$$\theta := \theta + \alpha\sum_{i=1}^{n}(y^{(i)} - \theta^{T}\phi(x^{(i)}))\phi(x^{(i)})$$

the above becomes computationally expensive when $\phi(x)$ is high dimensional.<br>
but we can observe that, if at some point , $\theta$ can be represented as:

$$\theta = \sum_{i=1}^{n}\beta_{i}\phi(x^{(i)})$$

then in the next round:

$$
\begin{equation}
\begin{split}
\theta &:= \theta + \alpha\sum_{i=1}^{n}(y^{(i)} - \theta^{T}\phi(x^{(i)}))\phi(x^{(i)}) \\
&=\sum_{i=1}^{n}\beta_{i}\phi(x^{(i)}) + \alpha\sum_{i=1}^{n}(y^{(i)} - \theta^{T}\phi(x^{(i)}))\phi(x^{(i)}) \\
&=\sum_{i=1}^{n}(\beta_{i} + \alpha(y^{(i)} - \theta^{T}\phi(x^{(i)})))\phi(x^{(i)})
\end{split}
\end{equation}
$$

$\theta$ can be also represented as a linear representation of $\phi(x^{(i)})$<br>
we can then derive $\beta$'s update rule:

$$\beta_{i} := \beta_{i} + \alpha(y^{(i)} - \sum_{j=1}^{n}\beta_{j}\phi(x^{(j)})^{T}\phi(x^{(i)}))$$

we only need to compute $\left \langle \phi(x^{(j)}), \phi(x^{(i)}) \right \rangle = \phi(x^{(j)})^{T}\phi(x^{(i)}))$ to update parameters no matter how high the feature dimension p is.

### kernel

we define the kernel corresponding to the feature map $\phi$ as a function that satisfying:

$$K(x, z) := \left \langle \phi(x), \phi(z) \right \rangle$$

define the kernel matrix:

$$K_{ij} = K(x^{(i)},x^{(j)})$$

properties of kernel matrix:

1. symmetric, since $\phi(x)^{T}\phi(z) = \phi(z)^{T}\phi(x)$.
2. positive semidefinite:

$$
\begin{equation}
\begin{split}
z^{T}Kz=&\sum_{i}\sum_{j}z_{i}K_{ij}z_{j}\\
=&\sum_{i}\sum_{j}z_{i}\phi(x^{(i)})^{T}\phi(x^{(j)})z_{j}\\
=&\sum_{i}\sum_{j}z_{i}\sum_{k}\phi_{k}(x^{(i)})\phi_{k}(x^{(j)})z_{j}\\
=&\sum_{k}\sum_{i}\sum_{j}z_{i}\phi_{k}(x^{(i)})\phi_{k}(x^{(j)})z_{j}\\
=&\sum_{k}\left(\sum_{i}z_{i}\phi_{k}(x^{(i)})\right)^2\\
\ge&\ 0
\end{split}
\end{equation}
$$

in the other hand, we have sufficient conditions for valid kernels:

(mercer): let $ K: \mathbb{R}^{d} \times \mathbb{R}^{d} \mapsto \mathbb{R}$. then for K be a valid kernel, it is necessary and sufficient that for any $\left \{ x^{(1)},...,x^{(n)} \right \} $, the corresponding kernel matrix is symmetric positive semi-definite.

proof t.b.c.

## support vector machine

### Margins

consider logistic regression, where the probability $p(y=1|x;\theta)$ is modeled by $h_{\theta}(x)=\sigma(\theta^{T}x)$. we predict 1 on an input x if and only if $\theta^{T}x >= 0.5$, the larger $\theta^{T}x$ is, the more confidence we are.<br>
the distance from the hyperplane is important.

functional margin:

$$\hat{\gamma}^{(i)}=y^{(i)}(w^{T}x^{(i)} + b)$$

geometric margin:

$$\gamma^{(i)}=\frac{y^{(i)}(w^{T}x^{(i)} + b)}{\left \| w \right \| }$$

geometric margin is the euclid distance.

### the optimal margin classifier

we want to maximize geometic margin:

$$
\begin{equation}
\begin{split}
\underset{\gamma, w, b}{max}\ &\gamma \\
s.t\quad &\frac{y^{(i)}(w^{T}x^{(i)} + b)}{\left \| w \right \| } >= \gamma
\end{split}
\end{equation}
$$

without loss of generality, we can set $\gamma\left \| w \right \|=1$, then the above is equivalent to:

$$
\begin{equation}
\begin{split}
\underset{w, b}{min}\ &\frac{1}{2}{\left \| w \right \|}^2 \\
s.t\quad &{y^{(i)}(w^{T}x^{(i)} + b)} >= 1
\end{split}
\end{equation}
$$

### lagrange duality

consider the following primal optimization problem:

$$
\begin{equation}
\begin{split}
\underset{w}{min}\quad &f(w)\\
s.t\quad &g_{i}(w) \le 0,i=1,...,k\\
&h_{i}(w)=0, i=1,...,l
\end{split}
\end{equation}
$$

we define the lagrangian of this optimization problem:

$$L(w,\alpha,\beta)=f(w) + \sum_{i=1}^{k}\alpha_{i}g_{i}(w) + \sum_{i=1}^{l}\beta_{i}h_{i}(w)$$

here $\alpha_{i}, \beta_{i}$ are the lagrange multipliers. consider the quantity:

$$\theta_{P}(w) = \underset{\alpha,\beta:\alpha_{i}\ge{0}}{max}L(w,\alpha,\beta) $$

here $P$ stands for "primal".

let some $w$ be given, if $w$ violates any primal constraints, i.e., if either $g_{i}(w) > 0$ or $h_{i}(w) \ne 0$, then:

$$\theta_{P}(w) = \underset{\alpha,\beta:\alpha_{i}\ge{0}}{max}f(w) + \sum_{i=1}^{k}\alpha_{i}g_{i}(w) + \sum_{i=1}^{l}\beta_{i}h_{i}(w) = \infty$$

conversely, if the constraints are saitistied for particular $w$, then $\theta_{P}(w) = f(w)$, hence:

$$
\theta_{P}(w) = 
\begin{cases}
f(w)\ &\text{if w satisfy constraints}\\
\infty\ &\text{otherwise.}
\end{cases}
$$

with this equation, we have:

$$\underset{w}{min}f(w),\ \text{satisfy constraints} \Leftrightarrow \underset{w}{min}\theta_{P}(w)=\underset{w}{min}\underset{\alpha,\beta:\alpha_{i}\ge{0}}{max}L(w,\alpha,\beta)$$

we define the optimal value of the objective function to be $p^{\ast} = \underset{w}{min}\theta_{P}(w)$.

reverse the min max oder, we define:

$$\theta_{D}(\alpha, \beta) = \underset{w}{min}L(w,\alpha,\beta)$$

D stands for "dual", we can pose the dual optimization problem:

$$\underset{\alpha,\beta:\alpha_{i}\ge{0}}{max}\theta_{D}(\alpha, \beta) = \underset{\alpha,\beta:\alpha_{i}\ge{0}}{max}\underset{w}{min}L(w,\alpha,\beta)$$

we define the optimal $d^{\ast}=\underset{\alpha,\beta:\alpha_{i}\ge{0}}{max}\theta_{D}(\alpha, \beta)$, it can easily be shown that:

$$d^{\ast}=\underset{\alpha,\beta:\alpha_{i}\ge{0}}{max}\theta_{D}(\alpha, \beta) \le \underset{w}{min}\theta_{P}(w) = p^{\ast}$$

however, under certain conditions, we have:

$$d^{\ast} = p^{\ast}$$

so that we can solve the dual problem instead of the primal problem.

(KKT conditions)suppose $f$ and $g_{i}$ are convex, $h_{i}$ are affine, and there exits some $w$ so that $g_{i}(w) < 0$ for all $i$. <br>
under these assumptions, there must exists $w^{\ast}, \alpha^{\ast}, \beta^{\ast}$ so that $w^{\ast}$ is the solution to the primal problem, $\alpha^{\ast},\beta^{\ast}$ are the solutions to the dual problem, and $p^{\ast}=d^{\ast}=L(w^{\ast}, \alpha^{\ast}, \beta^{\ast})$.<br>
moreover, $w^{\ast}, \alpha^{\ast}, \beta^{\ast}$ iff satisfy the KKT conditions:

$$
\begin{equation}
\begin{split}
\frac{\partial}{\partial w_{i}}L(w^{\ast},\alpha^{\ast},\beta^{\ast}) =& 0,\ i=1,...,d\\
\frac{\partial}{\partial \beta_{i}}L(w^{\ast},\alpha^{\ast},\beta^{\ast}) =& 0,\ i=1,...,l\\
\alpha_{i}^{\ast}g_{i}(w) =& 0,\ i=1,...,k\\
g_{i}(w^{\ast}) \le& 0,\ i=1,...,k\\
\alpha_{i}^{\ast} \ge& 0,\ i=1,...,k
\end{split}
\end{equation}
$$

proof t.b.c.

## 2.4 use lagrange duality

previous optimal margin problem:

$$
\begin{equation}
\begin{split}
\underset{w, b}{min}\ &\frac{1}{2}{\left \| w \right \|}^2 \\
s.t\quad &{y^{(i)}(w^{T}x^{(i)} + b)} >= 1
\end{split}
\end{equation}
$$

we can write the constraints as:

$$g_{i}(w) = 1 - y^{(i)}(w^{T}x^{(i)} + b) \le 0$$

this problem satisfy the KKT prerequisites:

1. $\frac{1}{2}{\left \| w \right \|}^2$ is convex.
2. $1 - y^{(i)}(w^{T}x^{(i)} + b)$ is convex.
3. there exits $w, b$, such that $1 - y^{(i)}(w^{T}x^{(i)} + b) < 0$, just increase the order.

write the lagrangian of this problem:

$$
L(w,b,\alpha )=\frac{1}{2}\left \| w \right \|^{2} - \sum_{i=1}^{n}\alpha_{i}\left [ y^{(i)}(w^{T}x^{(i)} + b) - 1 \right ]     
$$

just solve the dual form of the problem, to do so, we need to first minimize $L(w,\alpha,\beta)$ with respect to $w,b$, setting derivatives to 0:

$$\nabla_{w}L(w, b, \alpha) = w - \sum_{i=1}^{n}\alpha_{i}y^{(i)}x^{(i)} = 0$$

this implies:

$$w = \sum_{i=1}^{n}\alpha_{i}y^{(i)}x^{(i)}$$

derivative with respect to b:

$$\nabla_{b}L(w, b, \alpha) = \sum_{i=1}^{n}\alpha_{i}y^{(i)}=0$$

plug these equations back to lagrangian:

$$
\begin{equation}
\begin{split}
L(w, b, \alpha) =& \frac{1}{2}\left \| w \right \|^{2} - \sum_{i=1}^{n}\alpha_{i}\left [ y^{(i)}(w^{T}x^{(i)} + b) - 1 \right ]\\
=& \frac{1}{2}(\sum_{i=1}^{n}\alpha_{i}y^{(i)}x^{(i)})^{T}(\sum_{i=1}^{n}\alpha_{i}y^{(i)}x^{(i)}) - \sum_{i=1}^{n}\alpha_{i}y^{(i)}(\sum_{i=1}^{n}\alpha_{i}y^{(i)}x^{(i)})^{T}x^{(i)} + b\sum_{i=1}^{n}\alpha_{i}y^{(i)} + \sum_{i=1}^{n}\alpha_{i}\\
=& \sum_{i=1}^{n}\alpha_{i} - \sum_{i,j=1}^{n}y^{(i)}y^{(j)}\alpha_{i}\alpha_{j}(x^{(i)})^{T}x^{(i)}
\end{split}
\end{equation}
$$

we thus obtain the following dual optimization problem:

$$
\begin{equation}
\begin{split}
\underset{\alpha}{max}\quad &W(\alpha)=\sum_{i=1}^{n}\alpha_{i} - \sum_{i,j=1}^{n}y^{(i)}y^{(j)}\alpha_{i}\alpha_{j}\left \langle x^{(i)},x^{(j)} \right \rangle \\
s.t\quad &\alpha_{i}\ge{0},\ i=1,...,n \\
&\sum_{i=1}^{n}\alpha_{i}y^{(i)}=0
\end{split}
\end{equation}
$$

this is easier to solve(we'll talk about it later). after abtained $\alpha_{i}^{\ast}$, we have:

$$w^{\ast} = \sum_{i=1}^{n}\alpha_{i}^{\ast}y^{(i)}x^{(i)}$$

go back to the original problem, we get:

$$b^{\ast} = -\frac{max_{i: y^{(i)}=-1}w^{\ast{T}}x^{(i)} + min_{i: y^{(i)}=1}w^{\ast{T}}x^{(i)}}{2} $$

when making predictions, we have:

$$w^{T}x + b = \left ( \sum_{i=1}^{n}\alpha_{i}y^{(i)}x^{(i)}\right )^{T}x + b = \sum_{i=1}^{n}\alpha_{i}y^{(i)}\left \langle x^{(i)},x \right \rangle + b$$

we only needed inner product.

### non-separable case

so far, we assumed the data is linearly separable.

while mapping data to a high dimensional feature space via $\phi$ does not increase the likelihood that the data is separable.

so we need to make the algorithm work for non-linearly separable datasets.

we reformulate our optimization as follows:

$$
\begin{equation}
\begin{split}
\underset{w, b}{min}\ &\frac{1}{2}{\left \| w \right \|}^2 + C\sum_{i=1}^{n}\xi_{i}\\
s.t\quad &{y^{(i)}(w^{T}x^{(i)} + b)} >= 1 - \xi_{i},\ i=1,...,n\\
&\xi_{i} \ge 0,\ i=1,...,n.
\end{split}
\end{equation}
$$

the cost of outlier $C\xi_{i}$.

the lagrangian:

$$
    L(w,b,\xi,\alpha,r )=\frac{1}{2}\left \| w \right \|^{2} + C\sum_{i=1}^{n}\xi_{i} - \sum_{i=1}^{n}\alpha_{i}\left [ y^{(i)}(w^{T}x^{(i)} + b) - 1 + \xi_{i}\right ] -\sum_{i=1}^{n}r_{i}\xi_{i}     
$$

here $\alpha_{i},r_{i}$ are our lagrange multipliers.

the dual form of the problem:

$$
\begin{equation}
\begin{split}
\underset{\alpha}{max}\quad &W(\alpha)=\sum_{i=1}^{n}\alpha_{i} - \sum_{i,j=1}^{n}y^{(i)}y^{(j)}\alpha_{i}\alpha_{j}\left \langle x^{(i)},x^{(j)} \right \rangle \\
s.t\quad &0 \le \alpha_{i}\le{C},\ i=1,...,n \\
&\sum_{i=1}^{n}\alpha_{i}y^{(i)}=0
\end{split}
\end{equation}
$$

the same as before except $\alpha_{i}$'s constraints, the calculation for $b^{\ast}$ has to be modified, t.b.c.

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, 2:]
y = (iris["target"] == 2).astype(np.int)

svm_clf = Pipeline([("scaler", StandardScaler()),
                    ("linear_svc", LinearSVC(C=1, loss="hinge"))])

svm_clf.fit(X, y)

### SMO algorithm

consider when trying to solve the unconstrained optimization problem:

$$\underset{\alpha}{max}\ W(\alpha_{1},...,\alpha_{n})$$

we have the coordinate ascent algorithm:

$$
\begin{equation}
\begin{split}
&\text{loop until convergence:}\\
&\qquad \text{for }i=1,...,n:\\
&\qquad\qquad \alpha_{i}:=\underset{\hat{\alpha_{i}}}{argmax}
W(\alpha_{1},...,\alpha_{i-1},\hat{\alpha_{i}},\alpha_{i+1},...,\alpha_{n}).
\end{split}
\end{equation}
$$

we can not directly use coordinate ascent facing SVM dual problem:

$$
\begin{equation}
\begin{split}
\underset{\alpha}{max}\quad &W(\alpha)=\sum_{i=1}^{n}\alpha_{i} - \sum_{i,j=1}^{n}y^{(i)}y^{(j)}\alpha_{i}\alpha_{j}\left \langle x^{(i)},x^{(j)} \right \rangle \\
s.t\quad &0 \le \alpha_{i}\le{C},\ i=1,...,n \\
&\sum_{i=1}^{n}\alpha_{i}y^{(i)}=0
\end{split}
\end{equation}
$$

if we want to update some $\alpha_{i}$, we must update at least two of them simutaneously.

coordinate ascent change to:

$$
\begin{equation}
\begin{split}
&\text{loop until convergence:}\\
&\qquad 1.\text{select some pair } \alpha_{i}, \alpha_{j}.\\
&\qquad 2.\text{optimize } W(\alpha) \text{ with respect to } \alpha_{i}, \alpha_{j}, \text{while holding other } \alpha_{k} \text{ fixed}.
\end{split}
\end{equation}
$$

the reason that SMO is an efficient is that the update to $\alpha_{i}, \alpha_{j}$ can be computed efficiently.

take $i,j=1,2$ as example, SMO step is like:

$$max\ W(\alpha_{1},\alpha_{2},...,\alpha_{n}) \text{ while } $$
$$\alpha_{3},...,\alpha_{n} \text{ fixed}$$
$$\alpha_{1}y^{(1)} + \alpha_{1}y^{(1)} = \zeta$$
$$0 \le \alpha_{1} \le{C},0 \le \alpha_{2} \le{C},$$

this can be change to:

$$max\ W(\alpha_{1},(\zeta - \alpha_{1}y^{(1)})y^{(2)},...,\alpha_{n}) \text{ while } L\le \alpha_{1}\le H$$

this is direct quadratic optimization, easy to solve.

remained questions:

1. the choice of $\alpha_{i},\alpha_{j}$ , this is heuristic.
2. how to update b.

t.b.c according to Platt's paper.

## SVM with kernels

let $\phi : \mathbb{R}^{d} \to \mathbb{R}^{p}$ be a feature map, the original dual form of the problem:

$$
\begin{equation}
\begin{split}
\underset{\alpha}{max}\quad &W(\alpha)=\sum_{i=1}^{n}\alpha_{i} - \sum_{i,j=1}^{n}y^{(i)}y^{(j)}\alpha_{i}\alpha_{j}\left \langle x^{(i)},x^{(j)} \right \rangle \\
s.t\quad &\alpha_{i}\ge{0},\ i=1,...,n \\
&\sum_{i=1}^{n}\alpha_{i}y^{(i)}=0
\end{split}
\end{equation}
$$

now change to:

$$
\begin{equation}
\begin{split}
\underset{\alpha}{max}\quad &W(\alpha)=\sum_{i=1}^{n}\alpha_{i} - \sum_{i,j=1}^{n}y^{(i)}y^{(j)}\alpha_{i}\alpha_{j}\left \langle \phi(x^{(i)}),\phi(x^{(j)}) \right \rangle \\
s.t\quad &\alpha_{i}\ge{0},\ i=1,...,n \\
&\sum_{i=1}^{n}\alpha_{i}y^{(i)}=0
\end{split}
\end{equation}
$$

we only need to know $\left \langle \phi(x^{(i)}),\phi(x^{(j)}) \right \rangle$ to optimize.

when predicting:

$$w^{T}\phi(x) + b = \left ( \sum_{i=1}^{n}\alpha_{i}y^{(i)}\phi(x^{(i)})\right )^{T}x + b = \sum_{i=1}^{n}\alpha_{i}y^{(i)}\left \langle \phi(x^{(i)}),\phi(x) \right \rangle + b$$

"""PolynomialFeatures + LinearSVC"""
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)

"""poly SVC"""
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X, y)

"""rbf SVC"""
from sklearn.svm import SVC

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)

"""SVM can support regression"""
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_poly_reg.fit(X, y)

