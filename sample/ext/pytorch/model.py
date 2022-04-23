# pylint: disable=no-member,unused-import,not-callable
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Get Device for Training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Print model
model = NeuralNetwork().to(device)
print(model)

# Use model
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Layers
input_image = torch.rand(3, 28, 28)  # a sample minibatch of 3 images of size 28x28
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(
    input_image
)  # convert each 2D 28x28 image into a contiguous array of 784 pixel values
print(flat_image.size())

layer1 = nn.Linear(
    in_features=28 * 28, out_features=20
)  # applies a linear transformation on the input using its stored weights and biases.
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(
    hidden1
)  # create the complex mappings between the model’s inputs and outputs.
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(  # an ordered container of modules.
    flatten, layer1, nn.ReLU(), nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

softmax = nn.Softmax(
    dim=1
)  # The last linear layer of the neural network returns logits
pred_probab = softmax(logits)

# Print Model Parameters
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
