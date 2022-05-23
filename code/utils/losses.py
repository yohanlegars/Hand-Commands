"""

This section describes and organises the potential loss functions we are considering for the task at hand.


For Label Classification:
Let N be the number of different labels, and B the Batch Size. then 'input' and 'target' are defined by:

    - 'input': a [B, N] tensor, each row containing the probability of belonging to class 'n'
                    (or an unnormalized representation of that probability, ie a score).
        For example, with N = 3 and B = 2, 'input' could be:
                [[0.3, 0.4, 0.3],
                 [0.2, 0.7, 0.1]]
        Or, with just unnormalized scores:
                [[4.25, -12.63, 3.25],
                 [-6.22, 14.44, 13.69]]

    - 'target': a [B, N] one hot encoded tensor, encoding the value of the label
        For example, with N = 3 and B = 2, 'target' could be:
                [[0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]

The potential loss functions considered here are:
    - Cross Entropy: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
    - Binary Cross Entropy: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch.nn.functional.binary_cross_entropy


For Regression / Localization of Sign:


"""

import torch

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


if __name__ == '__main__':
    """
    An example of loss comparisons between random inputs and outputs,
    with shape and format that will be used during regular training.
    """
    B = 2       # Batch Size
    N = 5       # Number of labels

    class_input = torch.randn([B, N], requires_grad=True)
    class_input_softmaxed = torch.softmax(class_input, dim=1)

    class_target = torch.empty(B, dtype=torch.long).random_(N)
    class_target = torch.nn.functional.one_hot(class_target, num_classes=N)

    print(f"{class_input=}")
    print(f"{class_input_softmaxed=}")
    print(f"{class_target=}")








