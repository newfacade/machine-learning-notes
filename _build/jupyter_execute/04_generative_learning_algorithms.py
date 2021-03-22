# Generative Learning Algorithms

## Discriminative & Generative

discriminative：try to learn $p(y|x)$ directly, such as logistic regression<br>
or try to learn mappings from the space of inputs to the labels $\left \{ 0, 1\right \}$ directly, such as perceptron

generative：algorithms that try to model $p(x|y)$ and $p(y)$(called class prior), such as guassian discriminant analysis and naive bayes

when predicting, use bayes rule：

$$p(y|x)=\frac{p(x|y)p(y)}{p(x)}$$

then：

$$\hat{y}=\underset{y}{argmax}\ \frac{p(x|y)p(y)}{p(x)}= \underset{y}{argmax}\ {p(x|y)p(y)}$$

## Guassian Discriminant Analysis(GDA)

multivariant normal distribution

guassian distribution is parameterized by a mean vector $\mu \in \mathbb{R}^{d}$ and a covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$, where $\Sigma >= 0$ is symmetric and positive semi-definite. also written $\mathcal{N}(\mu,\Sigma)$, it's density is given by：

$$p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}\left | \Sigma \right |^{1/2} }exp\left ( -\frac{1}{2}(x-\mu)^{T}{\Sigma}^{-1}(x-\mu)\right )$$

unsurprisingly, for random variable $X \sim \mathcal{N}(\mu,\Sigma)$:

$$E[X] = \int_{x}xp(x;\mu, \Sigma)dx = \mu$$
$$Cov(X) = E[(X - E(X))(X - E(X))^{T}] = \Sigma$$

GDA:

when we have a classification problem in which the input features x are continous, we can use GDA, which model $p(x|y)$ using a multivariant normal distribution:

$$y\sim Bernoulli(\phi)$$
$$x | y=0 \sim \mathcal{N}(\mu_{0},\Sigma) $$
$$x | y=1 \sim \mathcal{N}(\mu_{1},\Sigma) $$

writing out the distribution:

$$p(y) = \phi^{y}(1 - \phi)^{1 - y}$$

$$p(x| y=0)=\frac{1}{(2\pi)^{d/2}\left | \Sigma \right |^{1/2} }exp\left (-\frac{1}{2}(x-\mu_{0})^{T}{\Sigma}^{-1}(x-\mu_{0})\right )$$

$$p(x| y=1)=\frac{1}{(2\pi)^{d/2}\left | \Sigma \right |^{1/2} }exp\left (-\frac{1}{2}(x-\mu_{1})^{T}{\Sigma}^{-1}(x-\mu_{1})\right )$$

the log-likelihood of the data is given by：

$$
\begin{equation}
\begin{split}
l(\phi,\mu_{0},\mu_{1},\Sigma) &= log\prod_{i=1}^{n}p(x^{(i)},y^{(i)};\phi,\mu_{0},\mu_{1},\Sigma) \\
&= log\prod_{i=1}^{n}p(x^{(i)}|y^{(i)};\mu_{0},\mu_{1},\Sigma)p(y^{(i)};\phi)
\end{split}
\end{equation}
$$

we find the maximum likelihood estimate of the parameters are：

$$\phi = \frac{1}{n}\sum_{i=1}^{n}1\left \{ y^{(i)}=1 \right \}$$

$$\mu_{0} = \frac{\sum_{i=1}^{n}1\left \{ y^{(i)}=0 \right \}x^{(i)}  }{\sum_{i=1}^{n}1\left \{ y^{(i)}=0 \right \}} $$

$$\mu_{1} = \frac{\sum_{i=1}^{n}1\left \{ y^{(i)}=1 \right \}x^{(i)}  }{\sum_{i=1}^{n}1\left \{ y^{(i)}=1 \right \}} $$

$$\Sigma=\frac{1}{n}\sum_{i=1}^{n}(x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^{T}$$

as we wanted

## GDA and logistic regression

the GDA model has an interesting relationship to logistic regression.

if we view the quantity $p(y=1|x; \phi,\Sigma,\mu_{0},\mu_{1})$ as a function of $x$, we'll find that it can be expressed in the form:

$$
p(y=1|x; \phi,\Sigma,\mu_{0},\mu_{1}) = \frac{1}{1 + exp(-\theta^{T}x)}
$$

where $\theta$ is some appropriate function of $\phi,\Sigma,\mu_{0},\mu_{1}$.<br>
proof:

$$
\begin{equation}
\begin{split}
p(y=1|x; \phi,\Sigma,\mu_{0},\mu_{1}) &= \frac{p(y=1, x)}{p(x)} \\
&=\frac{p(x|y=1)p(y=1)}{p(x|y=1)p(y=1) + p(x|y=1)p(y=1)} \\
&=\frac{\phi\cdot exp\left (-\frac{1}{2}(x-\mu_{1})^{T}{\Sigma}^{-1}(x-\mu_{1})\right )}{\phi\cdot exp\left (-\frac{1}{2}(x-\mu_{1})^{T}{\Sigma}^{-1}(x-\mu_{1})\right ) + (1 - \phi)\cdot exp\left (-\frac{1}{2}(x-\mu_{0})^{T}{\Sigma}^{-1}(x-\mu_{0})\right )} \\
&=\frac{1}{1 + \frac{1 - \phi}{\phi}exp\left(-\frac{1}{2}\left((x-\mu_{0})^{T}{\Sigma}^{-1}(x-\mu_{0}) - (x-\mu_{1})^{T}{\Sigma}^{-1}(x-\mu_{1})\right)\right)} \\
&=\frac{1}{1 + \frac{1 - \phi}{\phi}exp\left(-\frac{1}{2}\left((\mu_{1}^T -\mu_{0}^T)\Sigma^{-1}x + x^{T}\Sigma^{-1}(\mu_{1}-\mu_{0}) + (\mu_{0}^{T}\Sigma^{-1}\mu_{0} - \mu_{1}^{T}\Sigma^{-1}\mu_{1})\right)\right)}
\end{split}
\end{equation}
$$

because of $x^{T}a=a^{T}x$, we can express the above as:

$$\frac{1}{1 + exp(-\theta^{T}x)}$$

so the separating surface of GDA is $\theta^{T}x=0$.

as logistic regression, GDA can be interpreted by logistic model, but with different optimization strategy.

## Naive Bayes

in GDA，x is continous, when x is discrete, we can use naive bayes.

suppose $x=(x_{1}, x_{2},..., x_{d})$, for simplicity, we assume $x_{j}$ binary here, we of course have：

$$
\begin{equation}
\begin{split}
p(x|y) &= p(x_{1},x_{2},...,x_{d}|y) \\
&= p(x_{1}|y)p(x_{2}|y,x_{1})...p(x_{d}|y,x_{1},...,x_{d-1})
\end{split}
\end{equation}
$$

we will assume that $x_{j}$'s are conditionally independent given $y$. this assumption is called the naive bayes assumption, and the resulting algorithm is called the naive bayes classifier：

$$
\begin{equation}
\begin{split}
p(x|y) &= p(x_{1}|y)p(x_{2}|y,x_{1})...p(x_{d}|y,x_{1},...,x_{d-1}) \\
&= p(x_{1}|y)p(x_{2}|y)...p(x_{d}|y)
\end{split}
\end{equation}
$$

our model is parameterized by:

$$y\sim Bernoulli(\phi)$$
$$x_{j}|y=1 \sim Bernoulli(\phi_{j|y=1})$$
$$x_{j}|y=0 \sim Bernoulli(\phi_{j|y=0})$$

note:

$$k = \sum_{i=1}^{n}1\left\{y^{(i)}=1\right\}$$
$$s_{j} = \sum_{i=1}^{n}1\left\{x_{j}^{(i)}=1 \wedge y^{(i)}=1\right\}$$
$$t_{j} = \sum_{i=1}^{n}1\left\{x_{j}^{(i)}=1 \wedge y^{(i)}=0\right\}$$

we can write down the joint log likelihood of the data:

$$
\begin{equation}
\begin{split}
\mathcal{L}(\phi, \phi_{j|y=1}, \phi_{j|y=0}) &= log\prod_{i=1}^{n}p(x^{(i)},y^{(i)})\\
&=\sum_{i=1}^{n}log(p(x^{(i)}, y^{(i)})) \\
&=\sum_{i=1}^{n}log(p(y^{(i)})\prod_{j=1}^{d}p(x_{j}^{(i)}|y^{(i)})) \\
&=\sum_{y^{(i)}=1}(log(\phi) + \sum_{j=1}^{d}log(p(x_{j}^{(i)}|y=1))) + \sum_{y^{(i)}=0}(log(1 - \phi) + \sum_{j=1}^{d}log(p(x_{j}^{(i)}|y=0))) \\
&=k\ log(\phi) + (n-k)log(1 - \phi) + \sum_{j=1}^{d}(s_{j}\ log(\phi_{j|y=1}) + (k-s_{j})log(1 - \phi_{j|y=1}) + t_{j}\ log(\phi_{j|y=0}) + (n -k - t_{j})log(1 - \phi_{j|y=0})
\end{split}
\end{equation}
$$

maximizing this equal to maximize each part, we derive:

$$\phi=\frac{k}{n}$$
$$\phi_{j|y=1} = \frac{s_{j}}{k}$$
$$\phi_{j|y=0} = \frac{t_{j}}{n-k}$$

as expected

## Laplace Smoothing

there is problem with the naive bayes in its current form

if $x_{j}=1$ never occur in the training set, then $p(x_{j}=1|y=1)=0,p(x_{j}=1|y=0)=0$.

hence when predicting a sample $x$ with $x_{j}=1$, then:

$$
\begin{equation}
\begin{split}
p(y=1|x) &= \frac{\prod_{k=1}^{d}p(x_{k}|y=1)p(y=1)}{\prod_{k=1}^{d}p(x_{k}|y=1)p(y=1) + \prod_{k=1}^{d}p(x_{k}|y=0)p(y=0)} \\
&= \frac{0}{0}
\end{split}
\end{equation}
$$

does't know how to predict.

to fix this problem, instead of:

$$\phi_{j|y=1}=\frac{\sum_{i=1}^{n}1\left\{x_{j}^{(i)}=1 \wedge y^{(i)}=1\right\}}{\sum_{i=1}^{n}1\left\{y^{(i)}=1\right\}}$$

$$\phi_{j|y=0}=\frac{\sum_{i=1}^{n}1\left\{x_{j}^{(i)}=1 \wedge y^{(i)}=0\right\}}{\sum_{i=1}^{n}1\left\{y^{(i)}=0\right\}}$$

we add 1 to the numerator, add 2 to the denominator to:

1. avoid 0 in the parameter
2. $\phi_{j|y=1}$ and $\phi_{j|y=0}$ is still a probability distribution.

i.e:

$$\phi_{j|y=1}=\frac{1 + \sum_{i=1}^{n}1\left\{x_{j}^{(i)}=1 \wedge y^{(i)}=1\right\}}{2 + \sum_{i=1}^{n}1\left\{y^{(i)}=1\right\}}$$

$$\phi_{j|y=0}=\frac{1 + \sum_{i=1}^{n}1\left\{x_{j}^{(i)}=1 \wedge y^{(i)}=0\right\}}{2 + \sum_{i=1}^{n}1\left\{y^{(i)}=0\right\}}$$

## Text Classification

bernoulli event model:<br>
first randomly determined whether a spammer or non-spammer<br>
then runs through the dictionary deciding whether to include each word j.

multinomial event model:<br>
first randomly determined whether a spammer or non-spammer<br>
then each word in the email is generating from some same multinomial distribution independently.

using multinomial event model, if we have training set $\left \{ (x^{(1)},y^{(1)}),...,(x^{(n)},y^{(n)}) \right \}$ where $x^{(i)}=(x_{1}^{(i)},...,x_{d_{i}}^{(i)})\ $(here $d_{i}$ is the number of words in the i-th training example)

using maximum likelihood estimates of parameters like the above:

$$\phi = \frac{1}{n}\sum_{i=1}^{n}1\left \{ y^{(i)}=1 \right \}$$

$$\phi_{k|y=1}=\frac{\sum_{i=1}^{n}\sum_{j=1}^{d_{i}}1\left \{x_{j}^{(i)}=k\wedge y^{(i)}=1 \right \}}{\sum_{i=1}^{n}1\left \{ y^{(i)}=1 \right \}d_{i}}$$

$$\phi_{k|y=0}=\frac{\sum_{i=1}^{n}\sum_{j=1}^{d_{i}}1\left \{x_{j}^{(i)}=k\wedge y^{(i)}=0 \right \}}{\sum_{i=1}^{n}1\left \{ y^{(i)}=0 \right \}d_{i}}$$

add laplace smoothing with respect to multinomial event model:

$$\phi_{k|y=1}=\frac{1 + \sum_{i=1}^{n}\sum_{j=1}^{d_{i}}1\left \{x_{j}^{(i)}=k\wedge y^{(i)}=1 \right \}}{\left | V \right | + \sum_{i=1}^{n}1\left \{ y^{(i)}=1 \right \}d_{i}}$$

$$\phi_{k|y=0}=\frac{1 + \sum_{i=1}^{n}\sum_{j=1}^{d_{i}}1\left \{x_{j}^{(i)}=k\wedge y^{(i)}=0 \right \}}{\left | V \right | + \sum_{i=1}^{n}1\left \{ y^{(i)}=0 \right \}d_{i}}$$

$\left | V \right |$ is the size of the vocabulary

in short:

$$\phi_{k|y=1}=\frac{1 + number\ of\ words\ k\ occur\ in\ spammer}{\left | V \right | + number\ of\ words\ in\ spammer}$$

