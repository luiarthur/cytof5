import torch
from torch.distributions import Normal
from torch.nn import Parameter as Param

class LinReg(torch.nn.Module):
    def __init__(self, p):
        super(LinReg, self).__init__()
        self.b = Param(torch.randn(p, requires_grad=True))
        self.log_sig = Param(torch.randn(1, requires_grad=True))

    def forward(self, y, X):
        return Normal(X.matmul(self.b), self.log_sig.exp()).log_prob(y).mean()


if __name__ == '__main__':
    torch.manual_seed(0)
    p = 2
    N = 1000
    b = torch.randn(p)
    sig = torch.tensor(.5)

    X = torch.cat((torch.ones(N, 1), torch.randn(N, p - 1)), 1)
    y = X.matmul(b) + torch.randn(N) * sig
    model = LinReg(p)

    ll = []
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    for t in range(1000):
        loglike = model(y, X)
        loss = -loglike
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ll.append(-loss.item())
        print('iteration: {} / 100 | loglike: {}'.format(t, -loss.item()))

        if t > 10 and abs(ll[-1] / ll[-2] - 1) < 1e-6:
            break

    print(model.state_dict())
    print(b, sig.log())
