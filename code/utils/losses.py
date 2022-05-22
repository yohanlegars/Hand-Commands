import torch

#### CLASSIFICATION LOSSES ####

# Example of target with class indices
loss = torch.nn.CrossEntropyLoss()
input_tensor = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input_tensor, target)

print(f"{input_tensor=}")
print(f"{target=}")
print(f"{output=}")

output.backward()

# Example of target with class probabilities
input_tensor = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input_tensor, target)
output.backward()

print(f"{input_tensor=}")
print(f"{target=}")
print(f"{output=}")


