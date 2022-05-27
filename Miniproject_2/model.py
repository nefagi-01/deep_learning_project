# For mini-project 2

from .others.implementations import *
from pathlib import Path
import pickle
import torch

def upsampling_kernel_to_padding(kernel_size):
    assert kernel_size % 2 == 1
    return (kernel_size - 1) // 2


def downsampling_kernel_to_padding(kernel_size):
    assert kernel_size % 2 == 0
    return (kernel_size - 2) // 2


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = Sequential(
            Conv2d(3, 16, kernel_size=2, stride=2, padding=downsampling_kernel_to_padding(2), bias=True),
            ReLU(),
            Conv2d(16, 32, kernel_size=2, stride=2, padding=downsampling_kernel_to_padding(2), bias=True),
            ReLU(),
            Upsampling(32, 16, kernel_size=5, padding=upsampling_kernel_to_padding(5), scale_factor=2),
            ReLU(),
            Upsampling(16, 3, kernel_size=5, padding=upsampling_kernel_to_padding(5), scale_factor=2),
            Sigmoid()
        )
        self.optimizer = SGD(self.model.param(), lr=1e-2, momentum=0.9, nesterov=True)
        self.loss = MSE()
        self.batch_size = 4

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model.modules[0].weight, self.model.modules[0].bias = data[0]
            self.model.modules[2].weight, self.model.modules[2].bias = data[1]
            self.model.modules[4].conv2d.weight, self.model.modules[4].conv2d.bias = data[2]
            self.model.modules[6].conv2d.weight, self.model.modules[6].conv2d.bias = data[3]
            self.optimizer = SGD(self.model.param(), lr=1e-2, momentum=0.9, nesterov=True)

    def dump_model(self, path) -> None:
        data = ((self.model.modules[0].weight, self.model.modules[0].bias),
                (self.model.modules[2].weight, self.model.modules[2].bias),
                (self.model.modules[4].conv2d.weight, self.model.modules[4].conv2d.bias),
                (self.model.modules[6].conv2d.weight, self.model.modules[6].conv2d.bias)
                )

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def train(self, train_input, train_target, num_epochs) -> None:
        train_input = train_input.float()
        train_target = train_target.float()
        # train input in range [0, 255], output of network in range [0, 1], train target in range [0, 255]
        train_target = train_target / 255.0  # fix range of train target
        # train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        # train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        for e in range(num_epochs):
            print("EPOCH {}".format(e))
            for inp, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.model.forward(inp)
                loss = self.loss.forward(preds=output, labels=target)
                self.optimizer.zero_grad()
                self.model.backward(self.loss.backward())
                self.optimizer.step()

    def predict(self, test_input) -> torch.Tensor:
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        # test input in range [0, 255], output in range [0, 255]
        test_input = test_input.float()
        return self.model.forward(test_input) * 255.0
