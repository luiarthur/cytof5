import numpy as np
from sklearn.linear_model import LinearRegression

def elbow(x, y):
    """
    Find change-point using elbow criterion.

    Example:
    ========

    N = 30
    x1 = np.random.randn(N, 1)
    x2 = np.random.randn(N, 1)
    b1 = [2, 1]
    b2 = [3, -2]
    y1 = b1[0] + b1[1] * x1 + np.random.randn(N, 1) * .1
    y2 = b2[0] + b2[1] * x2 + np.random.randn(N, 1) * .1
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    d = elbow(x, y)
    plt.plot(d); plt.show()
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    mod1 = LinearRegression()
    mod2 = LinearRegression()
    N = y.shape[0]
    total_ss = []
    for i in range(2, N - 2):
        fit1 = mod1.fit(x[:i], y[:i])
        fit2 = mod2.fit(x[i:], y[i:])
        pred1 = mod1.predict(x[:i])
        pred2 = mod2.predict(x[i:])
        ss1 = ((pred1 - y[:i]) ** 2).sum()
        ss2 = ((pred2 - y[i:]) ** 2).sum()
        total_ss.append(ss1 + ss2)
    total_ss = np.array(total_ss)
    return {'total_ss': total_ss, 'knot-loc': total_ss.argmin() + 2}


