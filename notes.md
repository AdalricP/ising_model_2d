We have some non-isolated system, it is 'surrounded' by another outer system with a certain temperature (that's the temperature bath.)
Eg. you in the atmosphere.

Probability $p_\alpha$ of being in state $\alpha$ with energy $E_\alpha$ is $$p_\alpha=\frac 1Z e^{-\beta E_\alpha}$$ where $\beta=1/kt$ is the boltzmann constant.

$$Z=\sum_\alpha e^{-\beta E_\alpha}$$ is the *partition function* which is a very special quantity that can tell us a lot about a thermodynamic system.

Remember that $P(a\to b)$ defines the probability of going from state $a$ to state $b$ which obviously implies that when we're at equilibrium (i.e. going from one state to another must be equally likely as it's inverse) which gives us the equation,
$$\sum_v p_v P(v\to u$) = \sum_u p_u P(u\to v)$$ - this can give us the *detailed balance condition*

$$p_uP(u\to v)=p_v P(v\to u)$$

Essentially 'the probability of being in that state in the first place, times the probability of moving to another state will be the same as the probability of being in that state in the first place times moving back to the original state" 

So we have the ising model now, this is the model for a ferromagnetic material, (there aren't many ferromagnetic materials), say this is iron.

You have a grid of points, think a lattice, and each point can either be in a 'up' or 'down' state or +1/-1, so we get the total energy (to derive)
$$E_\mu = sum_{<i, j>}-J\sigma_i\sigma_j$$
(<i, j> ==> nearest-neighbours sum)

For $\alpha$ satisfying a specific configuration of spins, we can use the detailed balance configuration to get,

$$\frac{P(\alpha\to\xi)}{P(\xi\to\alpha)}=\frac{p_\xi}{p_\alpha}=e^{-\beta(E_\xi-E_\alpha)}$$

# THE METROPOLIS ALGORITHM

Idea: We want to find the equilibrium state $\mu$ in a magnet at a particular temperature $\beta$ (how many $\sigma_j$'s are +1 or -1).
We start with some random lattice of spins, some pointy up, some pointy down and make it dance around using the equaion above until it drops into the equilibrium.

Algo of doom
1. Call current state $\mu$
2. pick a random point on the lattice and flip the sign. This new state is $\nu$ we want to find $P(\mu\to\nu)$ to accept the new state, we find that using detailed balance.
3. If $E_\nu > E_\mu$ set $P(\nu\to\mu)=1$ which gives us $P(\mu\to\nu)=e^{-\beta (E_\nu - E_\mu))}$ and similarly for $E_\mu > E_\nu$
4. Change to state $\nu$ with the probability defined above
5. Ad equilbrium

So the only thing we need to evaluate is $-\beta (E_\nu - E_\mu) = -\beta J \sum_{k=1}^4 \sigma_i \sigma_k$


Also to note is that on the boundaries of the grid, things may have less (3 or 2) neighbours. This will be an issue for this model, but not for hypergraphs, where each thing will have the exact same number of 'neighbours' due to hyper-edges.

 

