import torch
import torch.nn as nn
import torchvision
import settings

###############################################################################
# Functions
###############################################################################

def CVLoss(cof_list):
    """
       Compute the Coefficient of Variation Loss (CVLoss) for gating coefficients,
       which encourages balanced expert utilization.

       Args:
           cof_list: a list of length num_block2.
                     Each element is itself a list of length num_block3,
                     where each tensor has shape [batch_size, num_experts].
                     (i.e., coefficients for each expert per block)

       Returns:
           total_loss: scalar tensor
               The averaged CVLoss across all blocks.
       """
    total_loss = 0
    num_blocks = len(cof_list)

    for cof_T in cof_list:
        # Ensure cof_T is a tensor of shape [num_block3, batch_size, num_experts]
        if isinstance(cof_T, list):
            cof_T = torch.stack(cof_T, dim=0)  # 堆叠成张量

        # Compute standard deviation across experts (dim=2)
        # Result shape: [num_block3, batch_size]
        std = torch.std(cof_T, dim=2)

        # Compute mean across experts (dim=2)
        mean = torch.mean(cof_T, dim=2)

        # Coefficient of Variation (CV) = std / mean
        # Add 1e-6 to avoid division by zero
        cv = std / (mean + 1e-6)

        # Compute squared CV, then average over num_block3 and batch
        block_loss = torch.mean(cv ** 2)

        # Accumulate block loss
        total_loss += block_loss

    # Average loss across all block2 levels
    total_loss /= num_blocks

    return total_loss

def compute_gradient(img):
    gradx=img[...,1:,:]-img[...,:-1,:]
    grady=img[...,1:]-img[...,:-1]
    return gradx,grady

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target) 
        
        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=False)  # .to(device)  # .cuda()
        self.vgg.load_state_dict(torch.load(settings.vgg_model))
        print('Vgg loaded successfully!')
        self.vgg.eval()
        self.vgg = self.vgg.features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg[i](X)
            if (i + 1) in indices:
                out.append(X)

        return out

class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss

def compute_grad(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady

