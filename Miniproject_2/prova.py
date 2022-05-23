from model import *
from others.implementations import *
import torch

def sample(tensor1, tensor2, k):
    perm = torch.randperm(tensor1.size(0))
    idx = perm[:k]
    return tensor1[idx], tensor2[idx]



path_train = '../data/train_data.pkl'
path_val = '../data/val_data.pkl'
noisy_imgs_1, noisy_imgs_2 = torch.load(path_train)
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float(), noisy_imgs_2.float()
test, truth = torch.load(path_val)
test, truth = test.float(), truth.float()

samples = 5000

s1, s2 = sample(noisy_imgs_1, noisy_imgs_2, samples)
t1, t2 = sample(test, truth, samples)
t2 = t2 / 255.

epochs = 10

m = Model()
m.train(s1, s2, epochs)
print(compute_psnr(m.predict(t1) / 255., t2), m)