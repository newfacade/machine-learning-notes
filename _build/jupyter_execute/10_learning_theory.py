# Learning Theory

## bias and variance

we define bias of a model to be the expected generalization error even if we were to fit it to a very large (say, infinity) training set.

variance is about the expected distance of a model from this setting's optimal model.

(hoeffding inequality) let $Z_{1},...,Z_{n}$ be $n$ iid random variables draw from Bernoulli ($\phi$) distribution, let $\hat{\phi} = \frac{1}{n}\sum_{i=1}^{n}Z_{i}$ be the mean of these random variables, let $\gamma > 0$ be fixed, then:

$$P(|\phi - \hat{\phi}| > \gamma) \le 2exp(-2\gamma^{2}n)$$

proof t.b.c.

## errors

to simplify our exposition, we restrict to binary classification problem.

assume training set $S = \left\{(x^{(i)}, y^{(i)})\right\}_{i=1}^{n}$ draw iid from some distribution $\mathcal{D}$.

then for a hypothesis $h$, we define the training error (also called the empirical risk or empirical error) to be:

$$\hat{\epsilon}(h) = \frac{1}{n}\sum_{i=1}^{n}1\left\{h(x^{(i)} \ne y^{(i)})\right\}$$

we define the generalization error to be:

$$\epsilon(h) = P_{(x,y)\sim\mathcal{D}}(h(x) \ne y)$$

our goal is to minimize the generalization error.

while we can only minimize training error in practice, our training process can often be formalize as the process of empirical risk minimization (ERM):

$$\hat{\theta} = \underset{\theta}{argmin}\ \hat{\epsilon}(h_{\theta})$$

we denote hypothesis class $\mathcal{H}$ to be the set of all classifier considered by a learning algorithm, thus ERM can be rewritten as:

$$\hat{h} = \underset{h \in \mathcal{H}}{argmin}\ \hat{\epsilon}(h)$$

the optimal model is:

$$h^{\ast} = \underset{h \in \mathcal{H}}{argmin}\ {\epsilon}(h)$$

we want to give guarantee on the approximation of $h^{\ast}$ by $\hat{h}$. 

this is no easy, we consider it in two cases.

## the case of finite hypothesis class

take any fixed $h_{i} \in H$

define a bernoulli random variable $Z = 1\left\{h_{i}(x)\ne y\right\}$

similarily $Z_{j} = 1\left\{h_{i}(x^{(j)})\ne y^{(j)}\right\}$, then $Z$ and $Z_{j}$ have the same distribution.

$$\epsilon(h_{i}) = E(Z) = E(Z_{j})$$

and

$$\hat\epsilon(h_{i}) = \frac{1}{n}\sum_{j=1}^{n}Z_{j}$$

we can then apply the hoeffding inequality:

$$P(|\epsilon(h_{i}) - \hat\epsilon(h_{i})| > \gamma) \le 2exp(-2\gamma^{2}n)$$

this shows that for particular $h_{i}$, training error will be close to generalization error with high probability if $n$ is large.

further, use the finiteness of $\mathcal{H}$, we derive the uniform convergence result:

$$P(\exists h\in\mathcal{H}.|\epsilon(h_{i}) - \hat\epsilon(h_{i})| > \gamma) \le 2k\ exp(-2\gamma^{2}n)$$

that is:

$$P(\forall h\in\mathcal{H}.|\epsilon(h_{i}) - \hat\epsilon(h_{i})| \le \gamma) > 1 - 2k\ exp(-2\gamma^{2}n)$$

what we did was, for particular values of $n$ and $\gamma$, give a lower bound on the probability that for all $h\in \mathcal{H}$, $|\epsilon(h) - \hat\epsilon(h)| \le \gamma$.

our estimate can be represented by $(n, \gamma, \delta)$, where $\delta = 2k\ exp(-2\gamma^{2}n)$.

now with the probability at least $1 - \delta$, we have:

$$
\begin{equation}
\begin{split}
\epsilon{(\hat{h})}\ \le& \ \hat\epsilon{(\hat{h})} + \gamma\\
\le& \ \hat\epsilon{(h^{\ast})} + \gamma\\
\le& \ \epsilon{(h^{\ast})} + 2\gamma
\end{split}
\end{equation}
$$

this is what we want.

theorem. let $|\mathcal{H}| = k$, and let $n, \delta$ be fixed. then with the probabilty at least $1 - \delta$, we have:

$$\epsilon(\hat{h})\ \le\ \epsilon(h^{\ast}) + 2\sqrt{\frac{1}{2n}log\frac{2k}{\delta}}$$

in other words, fix $\gamma, \delta$. to guarantee the results, it suffices that:

$$n \ge \frac{1}{2\gamma^{2}}log\frac{2k}{\delta} = O(\frac{1}{\gamma^{2}}log\frac{k}{\delta})$$

## the case of infinite hypothesis class

intuitive argument:

suppose $\mathcal{H}$ is parameterized by $d$ real numbers. IEEE double-precision uses 64 bits to represent a floating point number.

now each model in $\mathcal{H}$ is parameterized by $64d$ bits, our hypothesis class really consist of at most $k=2^{64d}$ different hypotheses.

use the finite case result, to make the guarantee, it suffices:

$$n \ge O(\frac{1}{\gamma^{2}}log\frac{2^{64d}}{\delta}) = O(\frac{d}{\gamma^{2}}log\frac{1}{\delta}) = O_{\gamma, \delta}(d)$$

that is, to learn "well" using a hypothesis class of $d$ parameters, generally we are going to need $O(d)$ training examples.

but $d$ can be different for the same $\mathcal{H}$, for example:

$$h_{\theta}(x) = 1\left\{\theta_{0} + \theta_{1}x_{1} + ... + \theta_{d}x_{d} \ge 0\right\}$$

$$h_{u, v}(x) = 1\left\{(u_{0}^2 - v_{0}^2) + (u_{1}^2 - v_{1}^2)x_{1} + ... + (u_{d}^2 - v_{d}^2)x_{d} \ge 0\right\}$$

they both represent the class of linear classifiers in $\mathbb{R}^{d}$, but one represent by $d + 1$ parameters, the other by $2d + 2$ parameters.

now we give some definitions to clear it out.

given a set $S = \left\{x^{(1)}, ..., x^{(D)}\right\}$ of points $x^{(i)} \in \mathcal{X}$, we say that $\mathcal{H}$ shatters $S$ if $\mathcal{H}$ can realize any labeling on $S$.

we define the vapnik-chervonenkis dimension of $\mathcal{H}$, written $VC(\mathcal{H})$, to be the size of the largest set that is shattered by $\mathcal{H}$.

for approporiate representation, $VC(\mathcal{H})$ is roughly linear in the number of parameters.

theorem. let $\mathcal{H}$ be given, $D = VC(\mathcal{H})$, then with probability at least $1 - \delta$, we have that for all $h \in \mathcal{H}$:

$$|\epsilon(h) - \hat\epsilon(h)| \le O\left(\sqrt{\frac{D}{n}log\frac{n}{D} + \frac{1}{n}log\frac{1}{\delta}}\right)$$

thus, we also have:

$$\epsilon(\hat{h}) \le \epsilon(h^{\ast}) + O\left(\sqrt{\frac{D}{n}log\frac{n}{D} + \frac{1}{n}log\frac{1}{\delta}}\right)$$

proof t.b.c

