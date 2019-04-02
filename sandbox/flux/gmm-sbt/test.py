import torch
from torch.distributions.transforms import StickBreakingTransform

sbt = StickBreakingTransform()
y = torch.ones(3, requires_grad=True)
p = sbt(y)
z = p.log().sum()
z.backward()

print(y.grad)
