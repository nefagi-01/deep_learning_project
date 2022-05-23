from model import *
from others.implementations import *
import numpy as np
import itertools
import pickle

# HELPER FUNCTIONS

def sample(tensor1, tensor2, k):
    perm = torch.randperm(tensor1.size(0))
    idx = perm[:k]
    return tensor1[idx], tensor2[idx]


def upsampling_kernel_to_padding(kernel_size):
    assert kernel_size % 2 == 1
    return (kernel_size - 1) // 2

def downsampling_kernel_to_padding(kernel_size):
    assert kernel_size % 2 == 0
    return (kernel_size - 2) // 2

# MODEL CLASS

class TestModel:
    def __init__(self, shallow_channels, deep_channels, lr, momentum, nesterov, batch_size, downsampling_kernel_size, upsampling_kernel_size) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = Sequential(
            Conv2d(3, shallow_channels, kernel_size=downsampling_kernel_size, stride=2, padding=downsampling_kernel_to_padding(downsampling_kernel_size), bias=True),
            ReLU(),
            Conv2d(shallow_channels, deep_channels, kernel_size=downsampling_kernel_size, stride=2, padding=downsampling_kernel_to_padding(downsampling_kernel_size), bias=True),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(deep_channels, shallow_channels, kernel_size=upsampling_kernel_size, stride=1, padding=upsampling_kernel_to_padding(upsampling_kernel_size), bias=True),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(shallow_channels, 3, kernel_size=upsampling_kernel_size, stride=1, padding=upsampling_kernel_to_padding(upsampling_kernel_size), bias=True),
            Sigmoid()
        )
        self.optimizer = SGD(self.model.param(), lr=lr, momentum=momentum, nesterov=nesterov)
        self.loss = MSE()
        self.batch_size = batch_size

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        pass

    def train(self, train_input, train_target, num_epochs):
        # train input in range [0, 255], output of network in range [0, 1], train target in range [0, 255]
        train_target = train_target / 255.0  # fix range of train target
        # train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        # train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        losses = []
        for e in range(num_epochs):
            inp, target = train_input, train_target
            output = self.model.forward(inp)
            loss = self.loss.forward(preds=output, labels=target)
            losses.append(loss)
            print("EPOCH {} --- LOSS {}".format(e, loss))
            for inp, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.model.forward(inp)
                self.loss.forward(preds=output, labels=target)
                self.optimizer.zero_grad()
                self.model.backward(self.loss.backward())
                self.optimizer.step()
                self.optimizer.zero_grad()
        inp, target = train_input, train_target
        output = self.model.forward(inp)
        loss = self.loss.forward(preds=output, labels=target)
        losses.append(loss)
        print("FINAL LOSS {}".format(loss))
        return losses

    def predict(self, test_input) -> torch.Tensor:
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        # test input in range [0, 255], output in range [0, 255]
        return self.model.forward(test_input) * 255.0






# Hyper parameters for tuning
# hidden_channels = [(8, 32), (16, 32), (16, 64), (32, 64), (32, 128)]
# lr_list = [1e-3, 1e-2, 5e-2]
# momentum_list = [0., 0.9]
# nesterov_list = [False, True]
# batch_size_list = [4, 16, 64]
# downsampling_kernel_size_list = [2, 4]
# upsampling_kernel_size_list = [3, 5]

hidden_channels = [(32, 128)]
lr_list = [1e-2]
momentum_list = [0.9]
nesterov_list = [True]
batch_size_list = [4]
downsampling_kernel_size_list = [2]
upsampling_kernel_size_list = [3]

hyperparameters_combinations = list(itertools.product(hidden_channels, lr_list, momentum_list, nesterov_list, batch_size_list, downsampling_kernel_size_list, upsampling_kernel_size_list))
print(f"Number of combinations: {len(hyperparameters_combinations)}")

# Loading data
path_train = '../data/train_data.pkl'
path_val = '../data/val_data.pkl'
noisy_imgs_1, noisy_imgs_2 = torch.load(path_train)
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float(), noisy_imgs_2.float()
test, truth = torch.load(path_val)
test, truth = test.float(), truth.float() / 255.0

samples = 5000

s1, s2 = sample(noisy_imgs_1, noisy_imgs_2, samples)

results = dict()
for (shallow_channels, deep_channels), lr, momentum, nesterov, batch_size, downsampling_kernel_size, upsampling_kernel_size in hyperparameters_combinations:
    epochs = 10
    description = f"shallow_channels={shallow_channels},deep_channels={deep_channels},lr={lr},momentum={momentum},nesterov={nesterov},batch_size={batch_size},downsampling_kernel_size={downsampling_kernel_size},upsampling_kernel_size={upsampling_kernel_size}"
    m = TestModel(shallow_channels, deep_channels, lr, momentum, nesterov, batch_size, downsampling_kernel_size, upsampling_kernel_size)
    m.train(s1, s2, epochs)
    psnr = compute_psnr(m.predict(test) / 255., truth)
    print(description + f",PSNR={psnr.item()}")
    results[description] = (psnr, m)

with open(r"results.pickle", "wb") as output_file:
    pickle.dump(results, output_file)