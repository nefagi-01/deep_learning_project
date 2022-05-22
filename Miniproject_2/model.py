# For mini-project 2

from others.implementations import *
from pathlib import Path
from others.psnr import compute_psnr
import torch


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = Sequential(
            Conv2d(3, 8, kernel_size=2, stride=2, padding=0),
            ReLU(),
            Conv2d(8, 32, kernel_size=2, stride=2, padding=0),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            Sigmoid()
        )
        self.optimizer = SGD(self.model.param(), lr=1e-3, momentum=0.9, nesterov=False)
        self.loss = MSE()
        self.batch_size = 5

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        # train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        # train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        for e in range(num_epochs):
            for inp, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.model.forward(inp)
                loss = self.loss.forward(preds=output, labels=target)
                self.optimizer.zero_grad()
                self.model.backward(self.loss.backward())
                self.optimizer.step()
            print("EPOCH {} --- LOSS {}".format(e, loss))

    def predict(self, test_input) -> torch.Tensor:
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        return self.model.forward(test_input)
