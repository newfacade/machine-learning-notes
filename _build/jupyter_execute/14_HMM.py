# Hidden Markov Model

## Definition

Let $X_{n}$ and $Y_{n}$ be discrete-time stochastic process and $n \ge 1$. The pair $(X_{n}, Y_{n})$ is a hidden markov model if:

* $X_{n}$ is a markov process whose behavior is not directly observable("hidden")

* $P(Y_{n} = y_{n}|X_{1}=x_{1},...,X_{n}=x_{n}) = P(Y_{n}=y_{n}|X_{n}=x_{n})$ for every $n \ge 1$

The states of the process $X_{n}$ is called the hidden states, and $P(Y_{n}=y_{n}|X_{n}=x_{n})$ is called emission probability.

## compute probability given model

target: compute $P(O|\lambda)$ for any $O$ given $\lambda$.

### direct approach

$$P(O|\lambda) = \sum_{I}P(O,I|\lambda) =\sum_{I}P(O|I,\lambda)P(I|\lambda)$$

$I$ take on $N^{T}$ sums, computation complexity is $O(TN^{T})$, this does not work.

### forward

probability of $\{o_{1},...,o_{T}\}$ only depends on $\{o_{1},...,o_{T - 1}\}$ and $s_{t}$.


