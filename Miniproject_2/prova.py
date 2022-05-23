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
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.double(), noisy_imgs_2.double()
test, truth = torch.load(path_val)
test, truth = test.double(), truth.double()

s1, s2 = sample(noisy_imgs_1, noisy_imgs_2, 1000)
t1, t2 = sample(test, truth, 1000)

m = Model()
m.train(s1, s2, 10)
print(compute_psnr(m.predict(t1), t2), m)