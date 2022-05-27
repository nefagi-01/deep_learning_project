# For mini-project 1
from torch.optim import Adam
from torch import nn
import torch

from .others.unet import UNet
from pathlib import Path
from torch import clamp


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = UNet(num_features=32)
        # Use GPU if possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device=self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
        self.loss = nn.MSELoss()
        self.batch_size = 50

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path , map_location=device)
        self.model.load_state_dict(model)

    def save_pretrained_model(self) -> None:
        model_path = Path(__file__).parent / "bestmodel.pth"
        torch.save(self.model.state_dict(), model_path)

    def train(self, train_input, train_target, num_epochs) -> None:
        # train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        # train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        # Use GPU if possible
        train_input = train_input.float()
        train_target = train_target.float()
        train_input, train_target = train_input/255.0, train_target/255.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_input = train_input.to(device=device)
        train_target = train_target.to(device=device)
        for e in range(num_epochs):
            for inp, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.model(inp)
                loss = self.loss(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, test_input) -> torch.Tensor:
        print("passos")
        test_input = test_input.float()
        # Use GPU if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_input = test_input.to(device=device)
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        output = self.model(test_input)*255.0
        print("finisco")
        return clamp(output, 0.0, 255.0)
