import unittest

import torch
import vb.transformations as trans


class Test_Transformations(unittest.TestCase):
    def approx(self, a, b, eps=1E-8):
        self.assertTrue(abs(a - b) < eps)

    def test_bounded(self):
        p = torch.tensor(.5)
        self.assertTrue(trans.logit(p) == 0)

        q = torch.tensor(10.0)
        self.approx(trans.logit(q, 5, 16), -.1823216, 1e-5)
        self.approx(trans.invlogit(trans.logit(q, 5, 16), 5, 16), q, 1e-5)
        
    def test_simplex(self):
        p_orig = [.1, .3, .6]
        p = torch.tensor(p_orig)
        x = trans.invsoftmax(p)
        p_tti = torch.softmax(x, 0)
        self.assertTrue(p_tti.sum() == 1)
        for j in range(len(p_orig)):
            self.approx(p_tti[j].item(), p_orig[j], 1e-6)

    def test_lpdf_logx(self):
        gam = torch.distributions.gamma.Gamma(2, 3)
        x = torch.tensor(3.)
        z = trans.lpdf_logx(torch.log(x), gam.log_prob)
        print(z)
        self.approx(z, -4.6055508, eps=1e-6)
        
    def test_lpdf_logitx(self):
        # TODO
        pass

    def test_lpdf_logitx(self):
        # TODO
        pass


if __name__ == '__main__':
    unittest.main()
