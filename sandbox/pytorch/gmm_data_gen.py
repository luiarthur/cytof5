import numpy as np
import matplotlib.pyplot as plt

def genData(seed:int=1, nfactor=100):
    np.random.seed(seed)

    # Number of elements in group
    N = [n * nfactor for n in [3, 5, 2]]
    Nsum = sum(N)
    J = len(N)
    mu = [3, 7, -1]
    sd = [.1, .2, .1]
    assert len(mu) == len(sd) == len(N)

    # Create group id's as one list
    group_id = [[j] * N[j] for j in range(J)]
    group_id = sum(group_id, [])
    np.random.shuffle(group_id)

    # Generate Data
    y = [np.random.randn() * sd[j] + mu[j] for j in group_id]

    return {'y': y, 'mu': mu, 'sd': sd, 'group_id': group_id}

if __name__ == '__main__':
    y = genData()['y']

    # Plot data
    plt.hist(y, bins=30, density=True)
    plt.show()
