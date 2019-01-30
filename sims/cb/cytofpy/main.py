import torch
from readCB import readCB
from Cytof import Cytof
import math

if __name__ == '__main__':
    torch.manual_seed(2)
    CB_FILEPATH = '../data/cb.txt'
    cb = readCB(CB_FILEPATH)
    cb['m'] = []
    for i in range(len(cb['y'])):
        cb['y'][i] = torch.tensor(cb['y'][i])
        cb['m'].append(torch.isnan(cb['y'][i]))
        # FIXME: missing values should be imputed
        cb['y'][i][cb['m'][i]] = -3.0

    model = Cytof(data=cb)
    model.fit(data=cb, niters=1000, lr=1e-1, print_freq=1,
              minibatch_info={'prop': .1}, nmc=1)

    # TODO. RUN THIS!
    # FIXME: optmizer only accepts iterables of tensors. So, fix init_vp so that
    #        the parameters are tensors, not list of tensors. Consider padding 
    #        the shorter L with zeros.
