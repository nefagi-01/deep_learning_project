# For mini-project 1
from pyexpat import model
from torch.optim import Adam
from torch import nn
import torch
from torch.nn import Sequential, Conv2d, ReLU, ConvTranspose2d


class Model():
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        encoder = Sequential(Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1)),
                            ReLU(),
                            Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1)),
                            ReLU(),
                            Conv2d(32, 32, kernel_size=(4, 4), stride=(2, 2)),
                            ReLU(),
                            Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
                            ReLU(),
                            Conv2d(32, 8, kernel_size=(4, 4), stride=(1, 1)))
        decoder = Sequential(ConvTranspose2d(8, 32, kernel_size=(4, 4), stride=(1, 1)),
                            ReLU(),
                            ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
                            ReLU(),
                            ConvTranspose2d(32, 32, kernel_size=(4, 4), stride=(2, 2)),
                            ReLU(),
                            ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(1, 1)),
                            ReLU(),
                            ConvTranspose2d(32, 3, kernel_size=(5, 5), stride=(1, 1)))
        self.model = Sequential(encoder, decoder)                    
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.mse = nn.MSELoss()
        self.batch_size = 100

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        #: train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        #: train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        for e in range(num_epochs):
            if (e%10==0):
                print(e)
            for input, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.model(input)
                loss = self.mse(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, test_input) -> torch.Tensor:
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        return self.model(test_input)
