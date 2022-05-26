import numpy as np
import itertools
import pickle
import heapq
import torch

from model import *
from others.implementations import *
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
    def __init__(self, shallow_channels, deep_channels, lr, momentum, nesterov, batch_size, downsampling_kernel_size,
                 upsampling_kernel_size) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = Sequential(
            Conv2d(3, shallow_channels, kernel_size=downsampling_kernel_size, stride=2,
                   padding=downsampling_kernel_to_padding(downsampling_kernel_size), bias=True),
            ReLU(),
            Conv2d(shallow_channels, deep_channels, kernel_size=downsampling_kernel_size, stride=2,
                   padding=downsampling_kernel_to_padding(downsampling_kernel_size), bias=True),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(deep_channels, shallow_channels, kernel_size=upsampling_kernel_size, stride=1,
                   padding=upsampling_kernel_to_padding(upsampling_kernel_size), bias=True),
            ReLU(),
            NearestUpsampling(scale_factor=2),
            Conv2d(shallow_channels, 3, kernel_size=upsampling_kernel_size, stride=1,
                   padding=upsampling_kernel_to_padding(upsampling_kernel_size), bias=True),
            Sigmoid()
        )
        self.optimizer = SGD(self.model.param(), lr=lr, momentum=momentum, nesterov=nesterov)
        self.loss = MSE()
        self.batch_size = batch_size

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        pass

    def train(self, training_generator, num_epochs):
        # train input in range [0, 255], output of network in range [0, 1], train target in range [0, 255]
        # fix range of train target
        # train_input: tensor of size (N,C,H,W) containing a noisy version of the images
        # train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs the input by their noise.
        losses = []
        for e in range(num_epochs):
            
            for train_input, train_target in training_generator:
                train_input, train_target = train_input[0], train_target[0]
                train_target = train_target / 255.0 
                inp, target = train_input, train_target
                output = self.model.forward(inp)
                loss = self.loss.forward(preds=output, labels=target)
                losses.append(loss)
                print("EPOCH {} --- LOSS {}".format(e, loss))
                for inp, target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                    print("INP2")
                    print(inp.shape)
                    output = self.model.forward(inp)
                    self.loss.forward(preds=output, labels=target)
                    self.optimizer.zero_grad()
                    self.model.backward(self.loss.backward())
                    self.optimizer.step()

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
# hidden_channels = [(8, 32), (16, 32), (16, 64), (32, 64)]
# lr_list = [1e-3]
# momentum_list = [0., 0.9]
# nesterov_list = [False, True]
# batch_size_list = [4, 16, 64]
# downsampling_kernel_size_list = [2, 4]
# upsampling_kernel_size_list = [3, 5]

# hyperparameters_combinations = list(
#     itertools.product(hidden_channels, lr_list, momentum_list, nesterov_list, batch_size_list,
#                       downsampling_kernel_size_list, upsampling_kernel_size_list))
# print(f"Number of combinations: {len(hyperparameters_combinations)}")


#TOP 30
with open('./top.pickle','rb') as f:
    top = pickle.load(f)
parameters = [[param.split('=')[1] for param in x[0].split(',')] for x in top]




# Loading data
# path_train = './data/train_data.pkl'
# path_val = './data/val_data.pkl'

# test, truth = torch.load(path_val)
# test, truth = test.float(), truth.float() / 255.0

# noisy_imgs_1, noisy_imgs_2 = torch.load(path_train)
# noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float(), noisy_imgs_2.float()

# samples = 50000

# # s1, s2 = sample(noisy_imgs_1, noisy_imgs_2, samples)
# s1, s2 = noisy_imgs_1, noisy_imgs_2

results = dict()
# for (shallow_channels,
#      deep_channels), lr, momentum, nesterov, batch_size, downsampling_kernel_size, upsampling_kernel_size in hyperparameters_combinations:
#     epochs = 10
#     description = f"shallow_channels={shallow_channels},deep_channels={deep_channels},lr={lr},momentum={momentum},nesterov={nesterov},batch_size={batch_size},downsampling_kernel_size={downsampling_kernel_size},upsampling_kernel_size={upsampling_kernel_size}"
#     m = TestModel(shallow_channels, deep_channels, lr, momentum, nesterov, batch_size, downsampling_kernel_size,
#                   upsampling_kernel_size)
#     m.train(s1, s2, epochs)
#     psnr = compute_psnr(m.predict(test) / 255., truth)
#     print(description + f",PSNR={psnr.item()}")
#     results[description] = psnr

#     with open(r"results.pickle", "wb") as output_file:
#         pickle.dump(results, output_file)

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,type_):
        'Initialization'
        self.type = type_
        self.list_IDs = [i for i in range(20)]
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        with open(f'./data/{self.type}/' + str(index) + f'_{self.type}_data.pkl', "rb") as f:
            data = pickle.load(f)
            d1, d2 = data
            print(d1.shape)
            print(d2.shape)
        return d1, d2

# Generators
training_set = Dataset('train')
training_generator = torch.utils.data.DataLoader(training_set)

# validation_set = Dataset('validation')
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)


for shallow_channels, deep_channels, lr, momentum, nesterov, batch_size, downsampling_kernel_size, upsampling_kernel_size in parameters:
    print("batch_size: ", batch_size)
    shallow_channels = int(shallow_channels)
    deep_channels = int(deep_channels)
    lr = float(lr)
    momentum = float(momentum)
    nesterov = nesterov == 'True'
    batch_size = int(batch_size)
    downsampling_kernel_size = int(downsampling_kernel_size)
    upsampling_kernel_size = int(upsampling_kernel_size)

    epochs = 20
    description = f"shallow_channels={shallow_channels},deep_channels={deep_channels},lr={lr},momentum={momentum},nesterov={nesterov},batch_size={batch_size},downsampling_kernel_size={downsampling_kernel_size},upsampling_kernel_size={upsampling_kernel_size}"
    m = TestModel(shallow_channels, deep_channels, lr, momentum, nesterov, batch_size, downsampling_kernel_size,
                  upsampling_kernel_size)
    m.train(training_generator ,epochs)
    psnr = compute_psnr(m.predict(test) / 255., truth)
    print(description + f",PSNR={psnr.item()}")
    results[description] = psnr