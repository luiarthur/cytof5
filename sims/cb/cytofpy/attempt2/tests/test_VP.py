import sys
sys.path.append('../')

from VarParams import *

if __name__ == '__main__':
    # VPNormal
    x = VPNormal((3,2,4))
    x.vp
    x.logpdf(1.0)
    x.sample()
