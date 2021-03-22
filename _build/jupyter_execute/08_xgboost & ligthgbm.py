# XGBoost & LightGBM

## XGBoost

mathematically, we can write our ensemble tree model in the form

$$\hat{y}_{i} = \sum_{k=1}^{K}f_{k}(x_{i}), f_{k}\in\mathcal{F}$$

where $K$ is the number of trees, $f$ is a function in the function space $\mathcal{F}$, and $\mathcal{F}$ is the set of all possible CARTs.

the objective function to be optimized is given by

$$\mbox{obj}(\theta) = \sum_{i=1}^{n}l(y_{i}, \hat{y}_{i}) + \sum_{k=1}^{K}\Omega(f_{k})$$

where $l$ is the loss term, $\Omega$ is the regularization term.

### Additive Learning

what are the parameters of trees? you can find that what we need to learn are those functions $f_{i}$, each containing the structure of the tree and the leaf scores.

learning tree strcture is much harder than traditional optimization problem where you can simply take the gradient, instead, we use an additive strategy: fix what we learned, and add one new tree at a time. we write the prediction value at step $t$ as $\hat{y}_{i}^{(t)}$, then

$$
\begin{equation}
\begin{split}
\mbox{obj}^{(t)} =& \sum_{i=1}^{n}l(y_{i}, \hat{y}_{i}^{(t)}) + \sum_{k=1}^{t}\Omega(f_{k}) \\
=& \sum_{i=1}^{n}l(y_{i}, \hat{y}_{i}^{(t - 1)} + f_{t}(x_{i})) + \Omega(f_{t}) + \mbox{constant}
\end{split}
\end{equation}
$$

$$\hat{y}_{i}^{(0)} = 0$$

in general case($l$ arbitrary), we take the taylor expansion of the loss function up to the second order

$$\mbox{obj}^{(t)} \approx \sum_{i=1}^{n}[l(y_{i}, \hat{y}_{i}^{(t - 1)}) + g_{i}f_{t}(x_{i}) + \frac{1}{2}h_{i}f_{t}(x_{i})^{2}] + \Omega(f_{t}) + \mbox{constant}$$

where $g_{i}$ and $h_{i}$ are defined as

$$g_{i} = \frac{\partial l(y_{i}, \hat{y}_{i}^{(t - 1)})}{\partial \hat{y}_{i}^{(t-1)}}, h_{i} = \frac{\partial^{2} l(y_{i}, \hat{y}_{i}^{(t - 1)})}{{\partial \hat{y}_{i}^{(t-1)}}^2}$$

after removing all the constants, the specific objective at step $t$ becomes

$$\sum_{i=1}^{n}[g_{i}f_{t}(x_{i}) + \frac{1}{2}h_{i}f_{t}(x_{i})^{2}] + \Omega(f_{t})$$

this becomes our optimization goal for the new tree.

### Model Complexity

we need to define the complexity of the tree $\Omega(f)$, in order to do so, let us first refine the definition of the tree $f(x)$ as

$$f(x) = w_{q(x)}, w \in \mathbb{R}^{T}, q: \mathbb{R}^{d} \to \{1,2,...,T\}$$

where $w$ is the vector of scores on leaves, $q$ is a function assigning each data point to the corresponding leaf, and $T$ is the number of leaves. in XGBoost, we define the complexity as

$$\Omega(f) = \gamma{T} + \frac{1}{2}\lambda\sum_{j=1}^{T}w_{j}^{2}$$

this works well in practice.

### The Structure Score

now we can write the objective value with the $t$-th tree as:

$$
\begin{equation}
\begin{split}
\mbox{obj}^{(t)} = & \sum_{i=1}^{n}[g_{i}f_{t}(x_{i}) + \frac{1}{2}h_{i}f_{t}(x_{i})^{2}] + \gamma{T} + \frac{1}{2}\lambda\sum_{j=1}^{T}w_{j}^{2} \\
=& \sum_{j=1}^{T}[(\sum_{i\in{I_{j}}}g_{i})w_{j} + \frac{1}{2}(\sum_{i\in{I_j}}h_{i} + \lambda)w_{j}^2] + \gamma{T}
\end{split}
\end{equation}
$$

where $I_{j} = \{i|q_{i}=j\}$ is the set of indices of data-points assign to the $j$-th leaf.

we could further compress the expression by defining $G_{j} = \sum_{i\in{I_{j}}}g_{i}, H_{j} = \sum_{i\in{I_j}}h_{i}$:

$$\mbox{obj}^{(t)} = \sum_{j=1}^{T}[G_{j}w_{j} + \frac{1}{2}(H_{j} + \lambda)w_{j}^2] + \gamma{T}$$

in this equation, $w_{j}$ are independent with respect to each other, the form $G_{j}w_{j} + \frac{1}{2}(H_{j} + \lambda)w_{j}^2$ is quadratic and the best $w_{j}$ for a given $q(x)$ and the best objective reduction we can get is:

$$w_{j}^{\ast} = -\frac{G_{j}}{H_{j} + \lambda}$$

$$\mbox{obj}^{\ast} = -\frac{1}{2}\sum_{i=1}^{T}\frac{G_{j}^2}{H_{j} + \lambda} + \gamma{T}$$

the last equation measures how good a tree structure $q(x)$ is.

### Learn the Tree Structure

now we have a way to measure how good a tree is, ideally we would enumerate all possible trees and pick the best one, but not practical.

instead we will try to optimize **one level** of the tree at a time, specifically we try to split a leaf into two leaves, and the score gain is:

$$Gain = \frac{1}{2}\left [\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right ] - \gamma$$

if the first part of $Gain$ is smaller than $\gamma$, we would do better not add that branch, this is exactly pruning!

for real valued data, we places all instances in sorted order(by the split feature), then a left to right scan is sufficient to calculate the structure score of all possible split solutions, and we can find the best split efficiently. 

in practice, since it is intractable to enumerate all possible tree structures, we add one split at a time, this approach works well at most of the time.

### XGBoost Practice

"""quadratic dataset"""
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

"""basic xgboost"""
import xgboost
from sklearn.metrics import mean_squared_error

xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
mean_squared_error(y_pred, y_val)

"""xgboost automatically taking care of early stopping"""
xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)
mean_squared_error(y_pred, y_val)

## LightGBM

XGBoost uses level-wise tree growth:

![jupyter](images/level.png)

while LightGBM uses leaf-wise tree growth:

![jupyter](images/leaf.png)

we can formalize leaf-wise tree growth as:

$$(p_m, f_{m}, v_{m}) = \underset{(p,f,v)}{argmin}\ L(T_{m-1}(X).split(p, f, v), Y)$$

$$T_{m}(X) = T_{m-1}(X).split(p_m, f_{m}, v_{m})$$

finding the best split is costly, in XGBoost, we enumerate all features and all thresholds to find the best split.

LightGBM optimize that by using the histogram algorithm:



