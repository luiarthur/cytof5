# README

This version is similar to Rui Meng's way of implementing VI.
That is, he writes ELBO = E[p(x|z)] - KL[q(z) || p(z)].
Parameters are not transformed, and so a Jacobian isn't needed.

KL of q and p can be done in closed form if they are exponential
family. At times, an MC integral maybe needed for some terms.

The only thing that needs justification is the sampling from the variational
distribution. In PyTorch, there are samplers for that. They seem to be doing
the right thing. But it needs to be understood. Being able to sample from a
re-parameterized version that doesn't depend on the parameters of interest
should be used. [This paper][1] suggests ways. But it is not clear to me if
PyTorch uses this. I must get to the bottom of this.

I should also implement this with a proper [`nn` module][3] with `forward`
methods (and `backward` methods because of my binary Z matrix).

For `Gamma.rsample` see [this][4] paper.

[1]: https://arxiv.org/pdf/1806.01851.pdf
[2]: https://pytorch.org/docs/stable/distributions.html#gamma
[3]: https://pytorch.org/docs/stable/nn.html
[4]: https://arxiv.org/pdf/1806.01851.pdf
