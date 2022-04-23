import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# ToTensor:
#   converts a PIL image or NumPy ndarray into a FloatTensor,
#   and scales the image’s pixel intensity values in the range [0., 1.]
# Lambda Transforms:
#   define a function to turn the integer into a one-hot encoded tensor.
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)
