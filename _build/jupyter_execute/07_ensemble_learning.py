# Ensemble Learning

ensemble learning: combine multiple weak leaners to form a strong learner.

bagging is one approach of ensemble learning.

it uses some randomness, e.g sampling at random, then generate a bunch of different base learner. 

when we say sampling at random, we mean sampling with replacement. 

we usually select $m$ samples with replacement from m samples.

when predicting:

1. if classification, use vote.
2. if regression, use mean.

### voting

"""make moons dataset"""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # default test_size = 0.25

"""create ensemble learning"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting="hard")

"""training and testing"""
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

### BaggingClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)

"""set oob_score=True, so each sample can be evalute on estimators whose bag does not include that sample"""
bag_oob_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, bootstrap=True, n_jobs=-1, random_state=42, oob_score=True)
bag_oob_clf.fit(X_train, y_train)
bag_oob_clf.oob_score_

import numpy as np

y_oob_pred = bag_oob_clf.predict(X_test)
np.all(y_oob_pred == y_pred)

## random foreset

random foreset is one implementation of bagging with decision tree as base learner.

besides sampling at random, random foreset added some random when split.

rather than using all $d$ features when split, we randomly select it's $k$ features subset, and select from this subset.

often, we set $k = log_{2}d$.

in one word:

random foreset $=$ decision tree $+$ bagging $+$ use random subset when split.

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred_rf)

"""roughly equivalent"""
bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16), n_estimators=500,
                            max_samples=1.0, bootstrap=True, n_jobs=-1)

"""RandomForest automaticly measures feature importance = weighted average of node's impurity reduce"""
from sklearn.datasets import load_iris

iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

## boosting

boosting is another approach of ensemble learning.

boosting first train a base learner which is rather weak, then adjust that leaner with the knowledge of current prediction to make it stronger.

for example, adjust the traing set distribution, focus on samples that previous predict wrong.

## adaboost

suppose a binary classification problem, dataset is $\left\{(x_{1}, y_{1}),...,({x_{N},y_{N}})\right\}$, $y_{i} \in \left\{-1, 1\right\}$, total $N$ samples.

initial sample distribution is $D_{1} = (\frac{1}{N},...,\frac{1}{N}) = (w_{1,1},...,w_{1,N})$

for $m=1,...,M$:

1. train $G_{m}: \mathcal{X} \rightarrow \left\{-1, 1\right\}$ based on $D_{m} = (w_{m,1},...,w_{m,N})$

2. compute misclassify error:
$$e_{m} = \sum_{i=1}^{N}P(G_{m}(x_{i}) \ne y_{i}) = \sum_{i=1}^{N}w_{m,i}I(G_{m}(x_{i} \ne y_{i}))$$
3. update distribution $D_{m}$ to $D_{m+1}$:
$$
w_{m+1, i} =
\begin{cases}
\frac{w_{m,i}}{Z_{m}} \\
{\frac{1 - e_{m}}{e_{m}}}\frac{w_{m,i}}{Z_{m}}
\end{cases}
$$
where $Z_{m}$ is the normalization factor that makes $D_{m+1}$ a proper distribution.<br>
this step simply enlarge the previous wrong predicted sample's distribution by a factor $\frac{1 - e_{m}}{e_{m}}$.

then construct a linear combination of $G_{m}$:

$$f(x) = \sum_{m=1}^{M}log\ \frac{1 - e_{m}}{e_{m}}G_{m}(x)$$

finally the resulting classifier:

$$G(x) = sign(f(x)) = sign\left(\sum_{i=1}^{N}log\ \frac{1 - e_{m}}{e_{m}}G_{m}(x)\right)$$

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)

## forward stage wise algorithm

consider addictive model:

$$f(x) = \sum_{m=1}^{M}\beta_{m}b(x; \gamma_{m})$$

here $b$ is the base model, $\gamma_{m}$ is it's paramter.

given loss function $L$, we try to solve the following optimization problem:

$$\underset{\beta, \gamma}{min}\sum_{i=1}^{N}L\left(y_{i}, \sum_{m=1}^{M}\beta_{m}b(x; \gamma_{m})\right)$$

this is usually very complex.

foward stage wise algorithm simplify it by going only one stage at a time:

init $f_{0}(x) = 0$.

for $m=1,...,M$:

1. do the following one step optimization:
$$(\beta_{m}, \gamma_{m}) = \underset{\beta,\gamma}{argmin}\sum_{i=1}^{N}L(y_{i}, f_{m-1}(x_{i}) + \beta{b(x_{i},\gamma)})$$
2. update:
$$f_{m}(x) = f_{m-1}(x) + \beta_{m}b(x;\gamma_{m})$$

finally the resulting addictive model:

$$f(x) = f_{M}(x) = \sum_{m=1}^{M}\beta_{m}b(x; \gamma_{m})$$

adaboost $\Leftrightarrow$ foward stage wise algorithm when $L(y, f(x)) = exp(-yf(x))$, t.b.c.

## boosting tree

boosting tree model:

$$f(x) = \sum_{m=1}^{M}T(x; \theta_{m})$$

where $T(x; \theta_{m})$ is a decision tree.

boosting tree use the forward stage wise algorithm.

for binary classification boosting tree, we use $L(y, f(x)) = exp(-yf(x))$, so equivalent to adaboost where base model is decision tree.

for regression problem, we use $L(y, f(x)) = (y - f(x))^2$, i.e square error.

then the forward stage wise algorithm:

$$\hat\theta_{m} = \underset{\theta_{m}}{argmin}\sum_{i=1}^{N}L(y_{i}; f_{m-1}(x) + T(x_{i};\theta_{m}))$$

$$L(y_{i}; f_{m-1}(x) + T(x_{i};\theta_{m})) = [y - f_{m-1}(x) - T(x;\theta_{m})]^{2} = [r - T(x;\theta_{m})]^{2}$$

here $r = y - f_{m-1}(x)$ is the residual.

so for regression tree, we just need to fit the residual!

## GBDT-gradient boosting decision tree

while we can easily proceed in forward stage wise algorithm if loss function is square or exponential, for generic loss, this is not easy.

we use the gradient of the loss function to proceed  in generic case, that is GBDT.

GBDT for regression:

input: training set $T = \left\{(x_{1},y_{1}),...,(x_{N},y_{N})\right\}$, $x_{i} \in \mathcal{X} \in \mathbb{R}^{n}$, $y_{i} \in \mathcal{Y} \in \mathbb{R}$, loss function $L(y, f(x))$.

output: regression tree $\hat{f}(x)$.

init 
$$f_{0}(x) = \underset{c}{argmin}\sum_{i=1}^{N}L(y_{i}, c)$$

for $m=1,...,M$.

1. for $i=1,...,N$, compute
$$r_{mi} = -\left[\frac{\partial{L(y_{i}, f(x_{i}))}}{\partial f(x_{i})}\right]_{f = f_{m-1}}$$

2. generate a tree that fits $r_{mi}$, denote it's leaf nodes areas as $R_{mi},j=1,...,J$.

3. for $c=1,...,J$, compute
$$c_{mj} = \underset{c}{argmin}\sum_{x_{i} \in R_{mj}}L(y_{i}, f_{m-1}(x_{i}) + c)$$

4. update $f_{m}(x) = f_{m-1}(x) + \sum_{j=1}^{J}c_{mj}I(x \in R_{mj})$.

finally the resulting regression tree:

$$\hat{f}(x) = f_{M}(x) = \sum_{m=1}^{M}\sum_{j=1}^{J}c_{mj}I(x \in R_{mj})$$

in one word, while we want to minimize $L(y_{i}, f_{m-1}(x_{i}) + c)$, we just pick it's negative gradient like gradient descent:

$$c = -\left[\frac{\partial{L(y_{i}, f(x_{i}))}}{\partial f(x_{i})}\right]_{f = f_{m-1}}$$

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
y_pred

"""basic gbrt"""
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
gbrt.predict(X_new)

### early stopping

"""use staged_predict to find the best n_estimators"""
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

"""use warm_start to learn incrementally, stop while error does not improve for 5 iters"""
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping

## Exercise

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

[estimator.score(X_val, y_val) for estimator in estimators]

voting_clf = VotingClassifier([
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
])

voting_clf.fit(X_train, y_train)

voting_clf.score(X_val, y_val)

voting_soft_clf = VotingClassifier([
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("mlp_clf", mlp_clf),
], voting="soft")
voting_soft_clf.fit(X_train, y_train)

voting_soft_clf.score(X_val, y_val)

### stacking

X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)

X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
    
y_pred = rnd_forest_blender.predict(X_test_predictions)
accuracy_score(y_test, y_pred)

