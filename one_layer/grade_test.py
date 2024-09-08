#! /usr/bin/python3

import torch

# 创建一个需要梯度的张量
x = torch.tensor([1.0], requires_grad=True)

# 定义一个函数 y = x^3
y = x ** 3

# 计算一阶导数
y.backward(retain_graph=True)
print("一阶导数:", x.grad)  # 一阶导数 dy/dx = 3x^2 = 3

# 清零梯度，否则会被累加上去而变成18
x.grad.zero_()
print(x.grad)
y.backward(torch.tensor([5.0]))
print("一阶微分：", x.grad) # 一阶微分 5 * dy/dx = 15
