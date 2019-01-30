import torch
from readCB import readCB
from Cytof import Cytof

if __name__ == '__main__':
    CB_FILEPATH = '../data/cb.txt'
    cb = readCB(CB_FILEPATH)
    for i in range(len(cb['y'])):
        cb['y'][i] = torch.tensor(cb['y'][i])

    model = Cytof(data=cb)
    model.fit(data=cb, niters=1000, lr=1e-1, seed=1, print_freq=1)

    # TODO. RUN THIS!
    # FIXME: optmizer only accepts iterables of tensors. So, fix init_vp so that
    #        the parameters are tensors, not list of tensors. Consider padding 
    #        the shorter L with zeros.
