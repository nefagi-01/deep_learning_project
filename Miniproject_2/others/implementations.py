import torch
from torch.nn.functional import unfold, fold

#module skeleton
class Module(object):
    def forward(self, *x_):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []

#TO IMPLEMENT:
     
# -Convolution layer. [x]
# - Transpose convolution layer, or alternatively a combination of Nearest neighbor upsampling + Convolution. []
# - Upsampling layer, which is usually implemented with transposed convolution, but you can alternatively use a combination of Nearest neighbor upsampling + Convolution for this mini-project. []
# - ReLU [x]
# - Sigmoid [x]
# - A container like torch.nn.Sequential to put together an arbitrary configuration of modules together. [x]
# - Mean Squared Error as a Loss Function [x]
# - Stochastic Gradient Descent (SGD) optimizer [x]


class Upsample(Module):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    
    def rescale(self, x):
        shape = x.size()
        x = x.view(shape[0], shape[1], shape[2] * shape[3], 1)
        ups1 = x
        for i in range(self.scale_factor-1): 
            ups1 = torch.cat((ups1, x), 3) 
        ups1 = ups1.view(shape[0], shape[1], shape[2], self.scale_factor * shape[3]) 
        ups2 = ups1
        for i in range(self.scale_factor-1): 
            ups2 = torch.cat((ups2, ups1), 3) 
        ups2 = ups2.view(shape[0], shape[1], shape[2] * self.scale_factor, shape[3] * self.scale_factor)
        return ups2

    
    def forward(self, x):
        #implementation of nearest neighbour
        self.x = x
        return self.rescale(x)
        
    def backward(self, dl_dout):
        tmp1 = unfold(dl_dout, (self.scale_factor, self.scale_factor), stride = self.scale_factor).reshape(dl_dout.shape[0], dl_dout.shape[1], -1, self.x.shape[2]*self.x.shape[3]).sum(2)
        result = fold(tmp1, (self.x.shape[-2], self.x.shape[-1]), (1,1))
        return result
        
    def param(self):
        return []
    
    def zero_grad(self):
        pass



class Convolution(Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, padding=0, stride = 1):
        #initialize weights
        self.kernel = torch.ones((out_channels, in_channels, kernel_size, kernel_size))

        #initialize gradient vector
        self.dl_dw = torch.empty(self.kernel.size()).fill_(1)

        #stride, padding
        self.stride = stride
        self.padding = padding

    def output_dim(self, prev_dim, kernel_dim, stride = 1):
        return ((prev_dim - kernel_dim)//stride)+1

    def add_padding(self, x, padding, stride=1):
        if stride > 1 :
            tmp = torch.empty(x.shape[0], x.shape[1], (x.shape[2]-1)*stride+1, (x.shape[3]-1)*stride+1).fill_(0)
            tmp[:,:, 0::stride, 0::stride] = x
            x = tmp
        if padding > 0:
            shape = x.shape
            offset = padding*2
            x_pad = torch.empty(shape[0], shape[1], shape[2]+offset, shape[3]+offset).fill_(0)
            x_pad[:, :, padding:-padding, padding:-padding] = x
            return x_pad
        else:
            return x


    def apply_conv(self, x, kernel, stride=1):
        #compute output
        x_unfolded = unfold(x, kernel.shape[-2:], stride=stride)
        conv_output = x_unfolded.transpose(1,2).matmul(kernel.reshape(kernel.size(0), -1).t()).transpose(1,2)
        new_dim = self.output_dim(torch.tensor(x.shape[-2:]), torch.tensor(kernel.shape[-2:]), stride)
        out = fold(conv_output, new_dim, (1,1))
        return out
    
    def forward(self, x):
        #save input for backward pass
        self.x = self.add_padding(x, self.padding)
        return self.apply_conv(self.x, self.kernel, self.stride)
        
    
    def backward(self, dl_dout):
        #compute gradient with respect to weights (kernel)
        self.dl_dw = (self.apply_conv(self.x.transpose(0,1), self.add_padding(dl_dout, 0, self.stride).transpose(0,1))).transpose(0,1)
        #shape may be different because kernel wasn't applied entirely on self.x (e.g: x_shape = (1,3,5,5), kernel_shape = (2,3,2,2), stride = 2 , padding = 0)
        if self.dl_dw.shape != self.kernel.shape:
            self.dl_dw = self.dl_dw[:, :, :self.kernel.shape[-2], :self.kernel.shape[-1]]

        #compute gradient with respect to input
        #rotate kernel by 180 and transpose
        kernel = self.kernel.flip(self.kernel.dim()-2, self.kernel.dim()-1)
        kernel = kernel.transpose(0,1)
        #add padding to dl_dout
        dl_dout_pad = self.add_padding(dl_dout, 1, self.stride)
        #compute backward by convolution
        dl_dx = self.apply_conv(dl_dout_pad, kernel, 1)
        #shape may be different because kernel wasn't applied entirely on self.x (e.g: x_shape = (1,3,5,5), kernel_shape = (2,3,2,2), stride = 2 , padding = 0)
        if dl_dx.shape != self.x.shape:
            dl_dx_pad = torch.empty(self.x.shape).fill_(0)
            dl_dx_pad[:, :, :dl_dx.shape[-2], :dl_dx.shape[-1]] = dl_dx
            dl_dx = dl_dx_pad
       
        return dl_dx


class Sigmoid(Module):
    def forward(self, x):
        self.x = x.clone()
        return torch.div(1, (1+ torch.exp(-self.x)))

    def backward(self, dl_dout):
        sig = torch.div(1, (1+ torch.exp(-self.x)))
        return sig * (1-sig) * dl_dout


class ReLU(Module):
    def __init__(self):
        self.z = None
    
    # x_: the tensor outputed by the current layer
    def forward(self, x_):
        self.z = x_.clone()
        x_[x_ < 0] = 0
        return x_
        
    def backward(self, gradwrtoutput):
        da = gradwrtoutput
        tensor = self.z.clone()
        # g'(z)
        tensor[tensor > 0] = 1
        tensor[tensor < 0] = 0
        # dz[l]
        return da.mul(tensor)
        
    def param(self):
        return []
    
    def zero_grad(self):
        pass

class Sequential(Module):    
    def __init__(self, *layers_):
        self.modules = layers_
        
    # x_: the x data is a minibatch whose columns are features and lines are samples
    def forward(self, x_):
        x = x_
        for module in self.modules:
            x = module.forward(x)
        return x
        
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for module in reversed(self.modules):
            x = module.backward(x)
        return x
        
    # returns a flatened list of each module's parameters
    # each parameter in the list is represented as a tuple containing the parameter tensor (e.g. w)
    # and the gradient tensor (e.g. dl/dw)
    def param(self):
        return [ p for module in self.modules for p in module.param() ]
    
    # sets the gradient of each layer to zero before the next batch can go through the network
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

class LossMSE(Module): 
    def __init__(self):
        self.error = None
        
    def forward(self, preds, labels):
        self.error = preds - labels
        return self.error.pow(2).sum()
        
    def backward(self):
        return 2 * self.error
        
    def param(self):
        return []

class optim_SGD(Module):
    # parameters: the parameters of the Sequential module
    def __init__(self, parameters, learning_rate):
        self.param = parameters #[ p.shallow() for tup in parameters for p in tup ]
        self.lr = learning_rate
        
    # performs a gradient step (SGD) for all parameters
    def step(self):
        for (p, grad_p) in self.param:
            p.sub_(self.lr*grad_p)