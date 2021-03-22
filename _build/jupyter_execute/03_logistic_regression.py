# Logistic Regression

## Regression and Classification

regression answers how much? or how many? questions, for example:

1. predict the number of dollors at which a house will be sold.
2. predict the revenue of a restaurant.

in practice, we are more often interested in classification: asking not "how much", but "which one":

1. does this email belong in the spam foler or the inbox?
2. does this image depict a donkey, a dog, a cat, or a rooster?
3. which movie is Jean most likely to watch next?

linear regression not fit to solve classification problem for two reasons:

1. linear regression range in $\mathbb{R}$, while classification label is discrete.
2. linear regression uses euclid distance, result in d(class 3, class 1) > d(class 2, class 1), this is often not true in classification.

## Logistic Regression Model

we first foucus on binary classification

suppose dataset $D=\left \{ (x^{(1)},y^{(1)}),...,(x^{(n)},y^{(n)}) \right \} $, where $x^{(i)} \in \mathbb{R}^{d},\ y^{(i)} \in \left \{ 0, 1\right \}$.

to tackle this problem, after deriving $\theta^{T}x$

we uses a function that transform value in $\mathbb{R}$ into value in $[0, 1]$ and view this as the probability of being positive

we choose:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

as that function, it is the so called sigmoid function, we choose sigmoid function because:

1. it is a monotony increase, differentiable, symmetric function that maps $\mathbb{R}$ into $[0,1]$.
2. simple form
3. derivative can calculate easily: ${\sigma}'(x) = \sigma(x)(1 - \sigma(x))$

now the model is:

$$h_{\theta}(x) = \frac{1}{1 + exp(-\theta^{T}x)}$$

this is the logistic regression model.

additionally, in our view, we have:


$$h_{\theta}(x) = p(y=1|x)$$

## Entropy

self information $I(x)$ indicates the amount of information an event $x$ to happen.

we want $I(x)$ to satisfy:

1. $I(x) \ge 0$
2. $I(x) = 1 \text{ if }p(x)=1$
3. $\text{if }p(x_{1}) > p(x_{2}) \text{, then } I(x_{1}) < I(x_{2})$
4. $I(x_{1}, x_{2}) = I(x_{1}) + I(x_{2}) \text{ for independent }x_{1},x_{2}$

this leads to $I(x) = -log\ p(x)$

while self-information measures the information of a single discrete event, entropy measures the information of a random variable:

$$
\begin{equation}
\begin{split}
H(X)=&E(I(x))\\
=&E(-log\ p(x))\\
=&-\sum_{x \in \mathcal{X}}log\ p(x)
\end{split}
\end{equation}
$$

it is exactly the optimal encoding length of $X$.

cross entropy $H(p, q)$ is the encoding length of $p$ by optimal encoding of $q$:

$$H(p,q)=E_{p}\left[-log\ q(x)\right] = -\sum_{x}p(x)log\ q(x)$$

fix $p$, the closer $q$ is to $p$, the less is $H(p,q)$. 

we can use $H(p,q)$ to define the distance of $q$ to $p$.

turn to our binary classification problem, for $y^{(i)} \in \left\{0,1\right\}, \hat{y}^{(i)} \in (0, 1)$, we have:

$$
H(y^{(i)}, \hat{y}^{(i)}) =
\begin{cases}
-log(\hat{y}^{(i)}),\text{ if } y^{(i)} = 1\\
-log(1 - \hat{y}^{(i)}),\text{ if } y^{(i)} = 0
\end{cases}
$$

combine the two cases:

$$H(y^{(i)}, \hat{y}^{(i)}) = -y^{(i)}log(\hat{y}^{(i)}) - (1-y^{(i)})log(1 - \hat{y}^{(i)})$$

we use cross entropy to define the loss of logistic regression model:

$$
\begin{equation}
\begin{split}
J(\theta) =& \sum_{i=1}^{n}H(y^{(i)}, H(\hat{y}^{(i)})) \\
=& \sum_{i=1}^{n}-y^{(i)}log(\hat{y}^{(i)}) - (1-y^{(i)})log(1 - \hat{y}^{(i)}) \\
=& \sum_{i=1}^{n}-y^{(i)}log(h_{\theta}(x^{(i)})) - (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)}))
\end{split}
\end{equation}
$$

this is the so called logistic regression.

in addition, we can write cross entropy loss in matrix form:

$$J(\theta) = -y^{T}log(\sigma(\theta^{T}X)) - (1 - y)^{T}log(1 - \sigma(\theta^{T}X))$$

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris["data"][:, 2:]
y = (iris["target"] == 2).astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_test)
accuracy_score(y_test, y_pred)

## Probability Interpretation of Cross Entropy Loss

as we suppose

$$p(y=1|x) = h_{\theta}(x)$$

and

$$p(y=0|x) = 1 - h_{\theta}(x)$$

combine these two, in our prediction:

$$p(y^{(i)}|x^{(i)}) = \left [ h_{\theta}(x^{(i)}) \right ]^{y^{(i)}}\left [ 1 - h_{\theta}(x^{(i)}) \right ]^{1 - y^{(i)}} $$

log likelikhood function:

$$
\begin{equation}
\begin{split}
L(\theta) &= log\prod_{i=1}^{n} \left [ h_{\theta}(x^{(i)}) \right ]^{y^{(i)}}\left [ 1 - h_{\theta}(x^{(i)}) \right ]^{1 - y^{(i)}} \\
&= \sum_{i=1}^{n}\left [y^{(i)}log\ h_{\theta}(x^{(i)}) + (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)})) \right ]
\end{split}
\end{equation}
$$

maximum the above likelihood is equal to minimize the cross entropy loss:

$$J(\theta) = \sum_{i=1}^{n}-y^{(i)}log(h_{\theta}(x^{(i)})) - (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)}))$$

## Update-rule

we have:

$$
\begin{equation}
\begin{split}
\frac{\partial }{\partial \theta_{j}}J(\theta ) &= \frac{\partial }{\partial \theta_{j}}\sum_{i=1}^{n}-log(\sigma(\theta^{T}x^{(i)}))y^{(i)} - log(1 - \sigma(\theta^{T}x^{(i)}))(1 - y^{(i)}) \\
&= \sum_{i=1}^{n} \left (-y^{(i)}\frac{1}{\sigma(\theta^{T}x^{(i)})} + (1 - y^{(i)})\frac{1}{1 - \sigma(\theta^{T}x^{(i)})} \right )\frac{\partial }{\partial \theta_{j}}\sigma(\theta^{T}x^{(i)})\\
&=\sum_{i=1}^{n} \left (-y^{(i)}\frac{1}{\sigma(\theta^{T}x^{(i)})} + (1 - y^{(i)})\frac{1}{1 - \sigma(\theta^{T}x^{(i)})} \right )\sigma(\theta^{T}x^{(i)})(1-\sigma(\theta^{T}x^{(i)}))\frac{\partial }{\partial \theta_{j}}\theta^{T}x^{(i)} \\
&=\sum_{i=1}^{n}(\sigma(\theta^{T}x^{(i)}) - y^{(i)})x_{j}^{(i)} \\
&=\sum_{i=1}^{n}(h_{\theta}(x^{(i)}) - y^{(i)})x_{j}^{(i)}
\end{split}
\end{equation}
$$

the same form with linear regression!

as linear regression, we have the update rule for logistic regression:

$$\theta_{j}: =\theta_{j} - \alpha\sum_{i=1}^{n} (h_{\theta }(x^{(i)}) - y^{(i)})x_{j}^{(i)} $$

combine all dimensions, we have:

$$\theta: =\theta - \alpha\sum_{i=1}^{n} (h_{\theta }(x^{(i)}) - y^{(i)})\cdot x^{(i)} $$

write in matrix form：
$$
\frac{\partial }{\partial \theta}J(\theta ) = X^{T}(\sigma(X\theta) -y)
$$
matrix form of update formula：
$$
\theta: =\theta - \alpha X^{T}(\sigma(X\theta)-\mathbf{y} )
$$

## Regularization

like linear regression, we add penalty term on $J(\theta)$ for regularization

$l2$ penalty：
$$J(\theta) := J(\theta) + \lambda \left \| \theta \right \|_{2}^{2} $$

$l1$ penalty：
$$J(\theta) := J(\theta) + \lambda \left \| \theta \right \|_{1} $$

## Softmax Regression

now we turn to multi-class classification.

we start off with a simple image classification problem, each input consists of a $2\times{2}$ grayscale image, represent each pixel with a scalar, giving us features $\left\{x_{1},x_{2},x_{3}, x_{4}\right\}$. assume each image belong to one among the categories "cat", "chiken" and "dog".

we have a nice way to represent categorical data: the one-hot encoding, i.e a vector with as many components as we have categories, the component corresponding to particular instance's category is 1 and all others are 0.

for our problem, "cat" represents by $(1,0,0)$, "chicken" by $(0, 1, 0)$, "dog" by $(0, 0, 1)$.

to estimate the conditional probabilities of all classes, we need a model with multiple outputs, one per class.

address classification with linear models, we will need as many affine functions as we have outputs:

$$o_{1} = x_{1}w_{11} + x_{2}w_{12} + x_{3}w_{13} + x_{4}w_{14}$$
$$o_{2} = x_{1}w_{21} + x_{2}w_{22} + x_{3}w_{23} + x_{4}w_{24}$$
$$o_{3} = x_{1}w_{31} + x_{2}w_{32} + x_{3}w_{33} + x_{4}w_{34}$$

depict as:

![jupyter](./images/softmaxreg.svg)

we would like output $\hat{y_{j}}$ to be interpreted as probability that a given item belong to class $j$.

to transform our current outputs $\left\{o_{1},o_{2},o_{3},o_{4}\right\}$ to probability distribution $\left\{\hat{y}_{1},\hat{y}_{2},\hat{y}_{3},\hat{y}_{4}\right\}$, we use the softmax operation:

$$\hat{y}_{j} = \frac{exp(o_{j})}{\sum_{k}exp(o_{k})}$$

when predicting:

$$\text{predict class} = \underset{j}{argmax}\ \hat{y}_{j} = \underset{j}\ o_{j}$$

$\hat{y}_{j}$ is necessary when compute loss.

as logistic regression, we use the cross entropy loss:

$$H(y,\hat{y}) = -\sum_{k}y_{j}log\ \hat{y}_{j} = -log\ \hat{y}_{\text{category of y}}$$

now complete the construction of softmax regression.

y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X_train, y_train)

y_pred = softmax_reg.predict(X_test)
accuracy_score(y_test, y_pred)

