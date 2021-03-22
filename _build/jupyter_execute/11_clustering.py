# Clustering

## k-means

### k-means algorithm
in the clustering problem, we are given a training set $\left\{x^{(1)},...,x^{(n)}\right\}$, and want to group the data into a few cohesive "clusters".

here, $x^{(i)} \in \mathbb{R}^{d}$ as usual, but no labels $y^{(i)}$ are given, so this is an unsupervised learning problem.

k-means clustering algorithm:

1. initialize cluster centroids $\mu_{1},...,\mu_{k} \in \mathbb{R}^{d}$ randomly.
2. repeat until convergence:<br>
$\quad$ for each $i$, set: 
$$c^{(i)} := \underset{j}{argmin}\left \|x^{(i)} - \mu_{j} \right \|^{2} $$
$\quad$ for each $j$, set:
$$\mu_{j} := \frac{\sum_{i=1}^{n}1\left\{c^{(i)}=j\right\}x^{(i)}}{\sum_{i=1}^{n}1\left\{c^{(i)}=j\right\}}$$

k is the number of clusters we want to find.

### convergence

define the distortion function to be:

$$J(c,\mu) = \sum_{i=1}^{n}\left \|x^{(i)} - \mu_{c^{(i)}} \right \|^{2}$$

thus $J$ measures the sum of squared distances between each training example $x^{(i)}$ and the nearest cluster centroid $\mu_{c^{(i)}}$

k-means is exactly coordinate descent on $J$.

the first step of k-means inner-loop:

$\quad$ minimize $J$ with respect to $c$ while holding $\mu$ fixed.

the second step of k-means inner-loop:

$\quad$ minimize $J$ with respect to $\mu$ while holding $c$ fixed.

but the distortion function $J$ is non-convex, so k-means may converge to local minimal.

to tackle this local-minimal problem, we commonly run k-means many times, and choose the one gives the lowest $J(c, \mu)$

### practice

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

from sklearn.linear_model import LogisticRegression

n_labeled = 50
log_reg = LogisticRegression()
log_reg.fit(X_train[: n_labeled], y_train[: n_labeled])

log_reg.score(X_test, y_test)

from sklearn.cluster import KMeans

k=50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
X_digits_dist.shape  # sample distance from each centroid

import numpy as np

representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

y_representative_digits = y_train[representative_digit_idx]

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)

y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
    
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train_propagated)

log_reg.score(X_test, y_test)

percentile_closest = 75

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]  # distance of each sample
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1
    
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

log_reg.score(X_test, y_test)

np.mean(y_train_partially_propagated == y_train[partially_propagated])

