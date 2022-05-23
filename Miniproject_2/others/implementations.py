from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
from math import sqrt

# module skeleton
class Module(object):
    def forward(self, *x_):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass


# TO IMPLEMENT:

# -Convolution layer. [x]
# - Transpose convolution layer, or alternatively a combination of Nearest neighbor upsampling + Convolution. []
# - Upsampling layer, which is usually implemented with transposed convolution, but you can alternatively use a combination of Nearest neighbor upsampling + Convolution for this mini-project. []
# - ReLU [x]
# - Sigmoid [x]
# - A container like torch.nn.Sequential to put together an arbitrary configuration of modules together. [x]
# - Mean Squared Error as a Loss Function [x]
# - Stochastic Gradient Descent (SGD) optimizer [x]


class NearestUpsampling(Module):
    def __init__(self, scale_factor):
        self.x = None
        self.scale_factor = scale_factor

    def rescale(self, x):
        shape = x.size()
        tmp = x.view(shape[0], shape[1], shape[2] * shape[3], 1)
        ups1 = x.view(shape[0], shape[1], shape[2] * shape[3], 1)
        for i in range(self.scale_factor - 1):
            ups1 = cat((ups1, tmp), 3)
        ups1 = ups1.view(shape[0], shape[1], shape[2], self.scale_factor * shape[3])
        ups2 = ups1
        for i in range(self.scale_factor - 1):
            ups2 = cat((ups2, ups1), 3)
        ups2 = ups2.view(shape[0], shape[1], shape[2] * self.scale_factor, shape[3] * self.scale_factor)
        return ups2

    def forward(self, x):
        # implementation of nearest neighbour
        self.x = x
        return self.rescale(x)

    def backward(self, dl_dout):
        tmp1 = unfold(dl_dout, (self.scale_factor, self.scale_factor), stride=self.scale_factor).reshape(
            dl_dout.shape[0], dl_dout.shape[1], -1, self.x.shape[2] * self.x.shape[3]).sum(2)
        result = fold(tmp1, (self.x.shape[-2], self.x.shape[-1]), (1, 1))
        return result

    def param(self):
        return []

    def zero_grad(self):
        pass


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):

        self.x = None
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        
        # initialize weights and bias
        self.bias = empty(out_channels) if bias else None
        self.kernel = empty((out_channels, in_channels, *kernel_size))
        self.xavier_init()


        # initialize gradient vectors
        self.dl_dw = empty(self.kernel.size()).fill_(0)
        self.dl_db = empty(out_channels).fill_(0) if bias else None

        # stride
        if type(stride) is int:
            stride = (stride, stride)
        self.stride = stride

        # padding
        if type(padding) is int:
            padding = (padding, padding)
        self.padding = padding

    def xavier_init(self):
        """
            Xavier's initialization for convolutional layers
        """
        in_channels = self.kernel.shape[1]
        out_channels = self.kernel.shape[0]

        # fan_in = filter dimensions * in_channels (number of neurons affecting an output neuron)
        fan_in = in_channels * self.kernel.shape[2] * self.kernel.shape[3]

        # fan_out = filter dimensions * out_channels (number of neurons affected by an input neuron)
        fan_out = out_channels * self.kernel.shape[2] * self.kernel.shape[3]

        # compute Xavier's bound for the uniform distribution
        bound = sqrt(2. / (fan_in + fan_out))

        # initialize weights
        self.kernel.uniform_(-bound, bound)
        self.kernel = self.kernel

        # initialize bias
        if self.bias is not None:
            self.bias.uniform_(-bound, bound)
            self.bias = self.bias


    def add_padding(self, x, padding):
        if padding != (0, 0):
            shape = x.shape
            x_pad = empty(shape[0], shape[1], shape[2] + padding[0] * 2, shape[3] + padding[1] * 2).fill_(0)
            x_pad[:, :, padding[0]:-padding[0], padding[1]:-padding[1]] = x
            return x_pad
        else:
            return x

    def apply_conv(self, x, kernel, stride=(1, 1), include_bias=False):
        # compute output
        x_unfolded = unfold(x, kernel.shape[-2:], stride=stride)
        conv_output = x_unfolded.transpose(1, 2).matmul(kernel.reshape(kernel.shape[0], -1).t()).transpose(1, 2) + (self.bias.view(1, -1, 1) if self.bias is not None and include_bias else 0)
        out = fold(conv_output, ((x.shape[2] - kernel.shape[2]) // stride[0] + 1, (x.shape[3] - kernel.shape[3]) // stride[1] + 1), (1, 1))
        return out

    def forward(self, x):
        # save input for backward pass
        self.x = self.add_padding(x.clone(), self.padding)
        return self.apply_conv(self.x, self.kernel, self.stride, include_bias=self.bias is not None)

    def delation(self, x):
        if self.stride != (1, 1):
            tmp = empty(x.shape[0], x.shape[1], (x.shape[2] - 1) * self.stride[0] + 1,
                        (x.shape[3] - 1) * self.stride[1] + 1).fill_(0)
            tmp[:, :, 0::self.stride[0], 0::self.stride[1]] = x
            x = tmp
        return x

    def backward(self, dl_dout):
        #compute size of not covered part of self.x
        not_covered = (self.x.shape[-2] - (self.stride[-2] * (dl_dout.shape[-2] - 1) + self.kernel.shape[-2]), self.x.shape[-1] - (self.stride[-1] * (dl_dout.shape[-1] - 1) + self.kernel.shape[-1]))
        
        if not_covered != (0,0):
            x = self.x[:,:,:-not_covered[-2],:-not_covered[-1]]
        else:
            x = self.x
        
        # compute gradient with respect to weights (kernel)
        self.dl_dw += (self.apply_conv(x.transpose(0, 1), self.delation(dl_dout).transpose(0, 1))).transpose(0, 1)
      
        # compute gradient with respect to bias
        if self.bias is not None:
            self.dl_db += dl_dout.sum(dim=(dl_dout.dim()-4, dl_dout.dim() - 2, dl_dout.dim() - 1))

        # compute gradient with respect to input
        # rotate kernel by 180 and transpose
        kernel = self.kernel.flip(self.kernel.dim() - 2, self.kernel.dim() - 1)
        kernel = kernel.transpose(0, 1)
         
        dl_dout = self.delation(dl_dout)

        # add padding to dl_dout
        dl_dout_pad = self.add_padding(dl_dout, ((x.shape[-2] - dl_dout.shape[-2]), (x.shape[-1] - dl_dout.shape[-1])))

        # compute backward by convolution
        dl_dx = self.apply_conv(dl_dout_pad, kernel)

        # shape may be different because kernel wasn't applied entirely on self.x (e.g: x_shape = (1,3,5,5), kernel_shape = (2,3,2,2), stride = 2 , padding = 0)
        if dl_dx.shape != self.x.shape:
            dl_dx_pad = empty(self.x.shape).fill_(0)
            dl_dx_pad[:, :, :-not_covered[-2], :-not_covered[-1]] = dl_dx
            dl_dx = dl_dx_pad

        #remove padding from dl_dx
        if self.padding != (0, 0):
            dl_dx = dl_dx[:, :, self.padding[-2]:-self.padding[-2], self.padding[-1]:-self.padding[-1]]
        return dl_dx

    def param(self):
        return [[self.kernel, self.dl_dw], [self.bias, self.dl_db]] if self.bias is not None else [[self.kernel, self.dl_dw]]

    def zero_grad(self):
        self.dl_dw.fill_(0)
        if self.bias is not None:
            self.dl_db.fill_(0)


class Sigmoid(Module):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x.clone()
        temp = (1 + (-self.x).exp()).pow(-1)
        return temp

    def backward(self, dl_dout):
        sig = (1 + (-self.x).exp()).pow(-1)
        return sig * (1 - sig) * dl_dout

    def zero_grad(self):
        pass


class ReLU(Module):
    def __init__(self):
        self.x = None

    # x_: the tensor outputed by the current layer
    def forward(self, x):
        self.x = x.clone()
        temp = x * (x > 0)
        return temp

    def backward(self, dl_dout):
        tensor = (self.x > 0)

        return dl_dout * tensor

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
        return [p for module in self.modules for p in module.param()]

    # sets the gradient of each layer to zero before the next batch can go through the network
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


class MSE(Module):
    def __init__(self):
        self.error = None

    def forward(self, preds, labels):
        self.error = (preds - labels) / preds.size().numel()
        return (preds - labels).pow(2).mean()

    def backward(self):
        return 2 * self.error

    def param(self):
        return []

    def zero_grad(self):
        pass


class SGD:
    # parameters: the parameters of the Sequential module
    def __init__(self, parameters, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum_buffer_list = [None for _ in parameters]

    # performs a gradient step (SGD) for all parameters
    def step(self):
        for i, param_tuple in enumerate(self.parameters):
            param = param_tuple[0]
            d_p = param_tuple[1]

            if self.weight_decay != 0:
                d_p = d_p.add(param, alpha=self.weight_decay)

            if self.momentum != 0:
                buf = self.momentum_buffer_list[i]

                if buf is None:
                    buf = d_p.clone()
                    self.momentum_buffer_list[i] = buf
                else:
                    buf.mul_(self.momentum).add_(d_p, alpha=1 - self.dampening)

                if self.nesterov:
                    d_p = d_p.add(buf, alpha=self.momentum)
                else:
                    d_p = buf

            # Do final update
            param_tuple[0].add_(d_p, alpha=-self.lr)

    def zero_grad(self):
        for param_tuple in self.parameters:
            param_tuple[1].fill_(0)
