import numpy as np
import time
import numba

tuner_spec = [
    ('proposal_sd', numba.float64),
    ('acceptance_count', numba.int64),
]

@numba.jitclass(tuner_spec)
class Tuner():
    def __init__(self, proposal_sd):
        self.proposal_sd = proposal_sd
        self.acceptance_count = 1


@numba.njit
def metropolis(x, log_prob, tuner):
    proposal = tuner.proposal_sd * np.random.randn() + x
    acceptance_log_prob = log_prob(proposal) - log_prob(x)
    if acceptance_log_prob > np.log(np.random.rand()):
        x = proposal
    return x
    

@numba.njit(numba.float64(numba.float64, numba.float64, numba.float64))
def lpdf_normal(x, m, s):
    z = (x - m) / s
    return -0.5 * np.log(2*np.pi) - np.log(s) - 0.5 * z**2

@numba.njit(numba.float64(numba.float64))
def lp(x):
    return lpdf_normal(x, 0, 1)


@numba.njit
def metropolis_ntimes(B, lp, tuner):
    x = 0.0
    for i in range(B):
        x = metropolis(x, lp, tuner)
    return x

tuner = Tuner(1.0)

B = int(1e6)
print('compile...')
metropolis(1, lp, tuner)
metropolis_ntimes(1, lp, tuner)

print('time...')
tic = time.time()
x = metropolis_ntimes(B, lp, tuner)
toc = time.time()

mean_time = (toc - tic)
print('mean time: {}'.format(mean_time))
