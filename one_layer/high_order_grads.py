#! /usr/bin/python3

import torch
import torchviz

x = torch.tensor([1.0], requires_grad=True)
y = x ** 3
print(f"x={x}")
print(f"y={y}")
d = {}
d["x"] = x
d["y"] = y

first_grad = torch.autograd.grad(outputs=y, inputs=x, create_graph=True)[0]
print("一阶导数：", first_grad.item())
torchviz.make_dot(first_grad, params=d).render("first_grad", format="png")

print(f"first_grad={first_grad}")
second_grad = torch.autograd.grad(outputs=first_grad, inputs=x, create_graph=True)[0]
print("二阶导数：", second_grad.item())
torchviz.make_dot(second_grad, params=d).render("second_grad", format="png")

print(f"second_grad={second_grad}")
third_grad = torch.autograd.grad(outputs=second_grad, inputs=x, create_graph=True)[0]
print("三阶导数：", third_grad.item())
torchviz.make_dot(third_grad, params=d).render("third_grad", format="png")

print(f"third_grad={third_grad}")
forth_grad = torch.autograd.grad(outputs=third_grad, inputs=x, create_graph=True)[0]
print("四阶导数：", forth_grad.item())
torchviz.make_dot(forth_grad, params=d).render("forth_grad", format="png")
