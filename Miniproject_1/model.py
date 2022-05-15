# For mini-project 1
from torch.optim import Adam
from torch import nn
import torch
from others.unet import UNet
from others.rednet import REDNet30


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = UNet()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.mse = nn.MSELoss()
        self.batch_size = 100

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        # train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        # train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        for e in range(num_epochs):
            print(e)
            for inp, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.model(inp)
                loss = self.mse(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, test_input) -> torch.Tensor:
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        return self.model(test_input)
