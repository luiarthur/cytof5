import numba
from numba import jit, njit
import timeit
import numpy as np

class Timer(object):
    """
    Usage:
    with Timer('Model training'):
        time.sleep(2)
        x = 1
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(self.name, end=' ')

        elapsed = time.time() - self.tstart
        print('time: {}s'.format(elapsed))


def slow_sum(x):
    out = 0
    for xi in x:
        out += xi
    return out

@njit(numba.float64(numba.float64[:]))
def numba_sum(x):
    out = 0.0
    for i in range(x.size):
        out += x[i]
    return out



if __name__ == '__main__':
    # generate data
    n = 10000
    x = np.arange(n) * 1.0

    # Number of reps
    B = 100

    # Setup
    setup = 'from __main__ import slow_sum, numba_sum, x'


    # Time things
    time_slow_sum = timeit.timeit('slow_sum(x)', number=B, setup=setup) / B
    time_sum = timeit.timeit('sum(x)', number=B, setup=setup) / B
    time_numba_sum = timeit.timeit('numba_sum(x)', number=B, setup=setup) / B
    time_np_sum = timeit.timeit('x.sum()', number=B, setup=setup) / B

    print('slow_sum: {} | sum: {} | numba_sum: {} | np_sum: {}'.format(time_slow_sum, time_sum, time_numba_sum, time_np_sum))
