# For mini-project 2

from others.implementations import *
from pathlib import Path
from others.psnr import compute_psnr
import torch
import torch.nn as nn
from torch import optim



class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, nesterov=False)
        self.loss = nn.MSELoss()
        self.batch_size = 5

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        # train input in range [0, 255], output of network in range [0, 1], train target in range [0, 255]
        train_target = train_target / 255.0 # fix range of train target
        # train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        # train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        for e in range(num_epochs):
            inp, target = train_input, train_target
            output = self.model.forward(inp)
            loss = self.loss(output, target)
            print("EPOCH {} --- LOSS {}".format(e, loss))
            for inp, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.model(inp)
                loss = self.loss(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        inp, target = train_input, train_target
        output = self.model.forward(inp)
        loss = self.loss(output, target)
        print("FINAL LOSS {}".format(loss))

    def predict(self, test_input) -> torch.Tensor:
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        # test input in range [0, 255], output in range [0, 255]
        return self.model(test_input) * 255.0
