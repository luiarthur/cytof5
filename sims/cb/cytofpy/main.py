import torch
from readCB import readCB
from Cytof import Cytof
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)
    CB_FILEPATH = '../data/cb.txt'
    cb = readCB(CB_FILEPATH)
    cb['m'] = []
    tmp_J = 6
    for i in range(len(cb['y'])):
        cb['y'][i] = torch.tensor(cb['y'][i])[:, :tmp_J]
        cb['m'].append(torch.isnan(cb['y'][i]))
        # FIXME: missing values should be imputed
        cb['y'][i][cb['m'][i]] = -3.0

    model = Cytof(data=cb, K=30, L=[5,3])
    # model.debug=True
    out = model.fit(data=cb, niters=10000, lr=1e-2, print_freq=1, eps=1e-2,
                    minibatch_info={'prop': .1}, nmc=1)
    elbo = out['elbo']
    vp = out['vp']
    plt.plot(elbo); plt.show()

    real_param_mean = {}
    for key in vp:
        real_param_mean[key] = vp[key].m
    params = model.to_param_space(real_param_mean)
    # print(params)

    # for key in vp:
    #     print('{} log_s: {}'.format(key, (vp[key].log_s)))

    # TODO. RUN THIS!
    # FIXME: optmizer only accepts iterables of tensors. So, fix init_vp so that
    #        the parameters are tensors, not list of tensors. Consider padding 
    #        the shorter L with zeros.
