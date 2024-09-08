#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchviz
from torch.autograd.functional import hessian
import torch.nn.functional as F

# 定义一个两层的神经网络，测试验证多阶求导
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc1.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        self.fc1.bias.data = torch.tensor([-2.0, 1.0], requires_grad=True)
        self.fc2 = nn.Linear(2, 1)
        self.fc2.weight.data = torch.tensor([[1.0, -1.0]], requires_grad=True)
        self.fc2.bias.data = torch.tensor([[5.0]], requires_grad=True)

    def forward(self, x):
        x = x.view(-1, 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x * x
        return x

model = SimpleDNN()
x = torch.tensor([1.0, 5.0], requires_grad=True)
output = model(x)

print(f"output={output}")  # expected 100
d = dict(model.named_parameters())
d["x"] = x

# 期望输出
#一阶导数:       (tensor([[ -20., -100.],
#            [  20.,  100.]], grad_fn=<TBackward0>), tensor([-20.,  20.], grad_fn=<ViewBackward0>), tensor([[-180., -480.]], grad_fn=<TBackward0>), tensor([[-20.]], grad_fn=<AddBackward0>))
#二阶导数:       fc1.weight      tensor([[ 2., 50.],
#            [ 2., 50.]], grad_fn=<CopySlices>)
#二阶导数:       fc1.bias        tensor([2., 2.], grad_fn=<CopySlices>)
#二阶导数:       fc2.weight      tensor([[ 162., 1152.]], grad_fn=<CopySlices>)
#二阶导数:       fc2.bias        tensor([[2.]])
first_grads = torch.autograd.grad(output, model.parameters(), create_graph=True)
torchviz.make_dot(first_grads, params=d).render("first_grads", format="png")
print(f"一阶导数:\t{first_grads}")

l = list(model.parameters())
second_grads = torch.zeros_like(first_grads[0])

for i in range(0, 2):
    for j in range(0, 2):
        second_grads[i][j] = torch.autograd.grad(first_grads[0][i][j], l[0], create_graph=True)[0][i][j]
print(f"二阶导数:\tfc1.weight\t{second_grads}")

second_grads = torch.zeros_like(first_grads[1])
for i in range(0, 2):
    second_grads[i] = torch.autograd.grad(first_grads[1][i], l[1], create_graph=True)[0][i]
print(f"二阶导数:\tfc1.bias\t{second_grads}")

second_grads = torch.zeros_like(first_grads[2])
for i in range(0, 1):
    for j in range(0, 2):
        second_grads[i][j] = torch.autograd.grad(first_grads[2][i][j], l[2], create_graph=True)[0][i][j]
print(f"二阶导数:\tfc2.weight\t{second_grads}")

second_grads = torch.zeros_like(first_grads[3])
for i in range(0, 1):
    second_grads[i] = torch.autograd.grad(first_grads[3][i], l[3], create_graph=True)[0][i]
print(f"二阶导数:\tfc2.bias\t{second_grads}")

# 检查是否像backward那样会进行梯度积累
# 经验证，不存在这种现象
x = None
x = torch.tensor([1.0, 5.0], requires_grad=True)
output = model(x)

print(f"output={output}")  # expected 100
d = dict(model.named_parameters())
d["x"] = x

# 期望输出
#一阶导数:       (tensor([[ -20., -100.],
#            [  20.,  100.]], grad_fn=<TBackward0>), tensor([-20.,  20.], grad_fn=<ViewBackward0>), tensor([[-180., -480.]], grad_fn=<TBackward0>), tensor([[-20.]], grad_fn=<AddBackward0>))
#二阶导数:       fc1.weight      tensor([[ 2., 50.],
#            [ 2., 50.]], grad_fn=<CopySlices>)
#二阶导数:       fc1.bias        tensor([2., 2.], grad_fn=<CopySlices>)
#二阶导数:       fc2.weight      tensor([[ 162., 1152.]], grad_fn=<CopySlices>)
#二阶导数:       fc2.bias        tensor([[2.]])
first_grads = torch.autograd.grad(output, model.parameters(), create_graph=True)
torchviz.make_dot(first_grads, params=d).render("first_grads", format="png")
print(f"一阶导数:\t{first_grads}")

l = list(model.parameters())
second_grads = torch.zeros_like(first_grads[0])

for i in range(0, 2):
    for j in range(0, 2):
        second_grads[i][j] = torch.autograd.grad(first_grads[0][i][j], l[0], create_graph=True)[0][i][j]
print(f"二阶导数:\tfc1.weight\t{second_grads}")

second_grads = torch.zeros_like(first_grads[1])
for i in range(0, 2):
    second_grads[i] = torch.autograd.grad(first_grads[1][i], l[1], create_graph=True)[0][i]
print(f"二阶导数:\tfc1.bias\t{second_grads}")

second_grads = torch.zeros_like(first_grads[2])
for i in range(0, 1):
    for j in range(0, 2):
        second_grads[i][j] = torch.autograd.grad(first_grads[2][i][j], l[2], create_graph=True)[0][i][j]
print(f"二阶导数:\tfc2.weight\t{second_grads}")

second_grads = torch.zeros_like(first_grads[3])
for i in range(0, 1):
    second_grads[i] = torch.autograd.grad(first_grads[3][i], l[3], create_graph=True)[0][i]
print(f"二阶导数:\tfc2.bias\t{second_grads}")

#for i in model.parameters():
#    print(i)
#second_grads = torch.autograd.grad(first_grads, model.parameters(), grad_outputs=dummy)
#print(f"二阶导数:\t{second_grads}")
