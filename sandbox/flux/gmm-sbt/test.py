import torch
from torch.distributions.transforms import StickBreakingTransform

sbt = StickBreakingTransform()
y = torch.ones(3, requires_grad=True)
p = sbt(y)
z = p.log().sum()
z.backward()

print(y.grad)

sbt = StickBreakingTransform()
y = torch.ones((2,3,4)).cumsum(0).cumsum(1).cumsum(2) / 24
y.requires_grad=True
p = sbt(y)
z = p.log().sum()
z.backward()

print(y.grad)


