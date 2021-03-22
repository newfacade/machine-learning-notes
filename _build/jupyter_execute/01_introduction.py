# Introduction

## What is machine learning

Machine learning is the science (and art) of programming computers so they can **learn from data.**

A slightly more general definition by Arthur Samuel in 1959:

**Field of study that gives computers the ability to learn without beging explicitly programmed.**

Common definition by Tom Mitchell in 1988:

**A computer program is said to learn from experience E with respect to some task T and some performace measure P, if its performance on T, as measured by P, improves with experience E.**

## Why use machine learning

machine learning especially shines in these four fields:

1. problems for which existing solutions require a lot of fine-tuning or long lists of rules, e.g spam-filter.
2. complex problems for which using a traditional approach yields no good solution, e.g speech-recognition.
3. a machine learning system can adapt to new data.
4. getting insights about complex problems and large amounts of data.

## Examples of applications

Basic machine learning:

1. basic regression
2. basic classification
3. dimension reduction & visualization
4. clustering
5. anomaly detection

Computer vision:

1. image classification, typically use CNNs.
2. semantic segmentation, e.g detecting tumors in scan, use CNNs.

Natural Language Processing:

1. text classification(include text sentiment analysis), typically use RNNs, Transformers.
2. text summerization, use RNNS, Transformers.
3. chatbot or personal assistant, this involves many NLP components, including NLU and question answering.
4. speech recognition

Recommender System

Reinforcement Learning, e.g AlphaGo.

## Types of machine learning

by whether or not they are trained with human supervision:

supervised learning: the training set you feed to the algorithm includes the desired solutions, called labels.

unsupervised learning: the training data is unlabeled.

semi-supervised learning: can deal with data that's partially labeled.

reinforcement learning: the learning system called an agent can observe the evironment, select and perform actions, and get reward in return. it must learn by itself what is the best strategy, called policy, to get the most reward over time.

by whether or not they can learn incrementally on the fly (online versus batch learning)

by whether they work by simply comparing new data points to known data points (like KNN), or instead by detecting patterns in the training data and build a predictive model. (instance-based versus model-based learning)

## Main challenges of machine learning

"bad data" aspect:

insufficient quantity of training data.

nonrepresentative training data.

poor-quality data.

irrelevant features.

"bad algorithm" aspect:

overfitting: model performs well on the training data, but it does not generalize well.

underfitting: model is too simple to learn the underlying structure of the data.

## Testing and Validation

the error rate on new cases is called the generalization error.

we split data into training set and test set, train the model using training set, test it using test set (use error on test set to estimate generalization error).

in hyperparameter tuning and model selection, once we use test set to select models, then test error can not be use to estimate generalization error.

**because you adapted the model and hyperparameters to produce the best models for that particular set.**

a common solution is called hold-out validation: hold out part of training set to evaluate several candidate models, after picking the best model, we typically retrain that best-model using the whole training set.

suppose we want to choose a model from $\mathcal{M} = \left\{M_{1},...,M_{d}\right\}$.

hold-out validation:

1. randomly split dataset $S$ into $S_{train}$ (say 70% of the data) and $S_{val}$ (say 30% of the data).
2. train each model $M_{i}$ on $S_{train}$ only, get $h_{i}$.
3. use $h_{i}$ to compute error on $S_{val}$, choose the corresponding model that minimize this error.

problems: wastes $S_{val}$ data, only evaluate on $S_{val}$ may add bias.

### k-fold cross validation

one solution of these problems is k-fold cross validation:

1. randomly split $S$ into k even disjoint subsets $S_{1},...,S_{k}$.
2. for each model $M_{i}$<br>
$\quad$ for $j=1,...,k$, train on all data except $S_{j}$ to get hypothesis $h_{ij}$<br>
$\quad$ test $h_{ij}$ on $S_{j}$ to get $\hat\epsilon_{S_{j}}(h_{ij})$<br>
estimate generalization error on $M_{i}$ as $\sum_{j=1}^{k}\hat\epsilon_{S_{j}}(h_{ij})$.
3. choose the model that minimize the estimated generalization error.

$k=10$ is a commonly used choice.

disadvantage: computationally expensive.

### data mismatch

in some cases, it's easy to get a large amount of data for training, but this data won't be perfectly representative of the data in production.

in this case, the most important rule is that validation set and test set must be as representative as possible.

typically, mismatch data is separated into train set and train-dev set. we first train the model no train set, then evaluate model on train-dev set.

when $P(validation) << P(train)$

$\quad$if $P(train\_dev) << P(train):$

$\quad\quad$overfitting is a cause

$\quad$elif $P(train\_dev) \approx P(train):$

$\quad\quad$cause is data mismatch

## feature selection

we can use validation to select features.

foward search:

1. initialize $\mathcal{F} = \emptyset$
2. repeatedly:<br>
$\quad$add each un-included feature to $\mathcal{F}$, then use cross validation to pick the best feature, add this best feature to $\mathcal{F}$.
3. end until $\mathcal{F}$ include all features.

disadvantage: computationally expensive. 

other heuristic filter feature selection:

compute correlation between $X_{i}$ and $Y$, then pick $k$ most correlated features.

$$r(X, Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}$$

in practice, we commonly replace correlation with  mutual-information.

$$I(X, Y) = \sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}}p(x,y)log\frac{p(x, y)}{p(x)p(y)} = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

if $X, Y$ are independent, $I(X,Y)=0$.

## no free lunch theorem

**demonstrate that if you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.**

