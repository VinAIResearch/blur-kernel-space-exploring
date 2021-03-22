import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


class SpatialGradientLoss(nn.Module):
    """Super sharp Loss"""

    def __init__(self):
        super(SpatialGradientLoss, self).__init__()

    def diffMap(self, A, alongX):
        B, N, C, H = A.shape
        if alongX:
            return A[:, :, 1:C, :] - A[:, :, 0:C-1, :]
        return A[:, :, :, 1:H] - A[:, :, :, 0:H-1]

    def forward(self, A, B):
        Amap = self.diffMap(A, alongX=True)
        Bmap = self.diffMap(B, alongX=True)
        loss = torch.sum((Amap - Bmap) ** 2)

        Amap = self.diffMap(A, alongX=False)
        Bmap = self.diffMap(B, alongX=False)
        loss += torch.sum((Amap - Bmap) ** 2)
        loss = torch.sqrt(loss)

        return loss


class KLDivergence(nn.Module):
    """KL loss for VAE regularization"""
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, X):
        B, N = X.shape

        mean = X.mean(dim=0).to(X.device)

        var = torch.zeros((N, N)).to(X.device)
        for i in range(B):
            y = X[i, :] - mean
            var += torch.mm(y.resize(N, 1), y.resize(1, N))
        for i in range(N):
            if var[i, i] <= 0:
                print(var[i][i])
        var = var.clamp(min=1e-18) / N

        kl = 0.5 * (-(var.log().trace()) + torch.trace(var)
                    - N + mean.pow(2).sum())

        return kl


class FrobeniousNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super(FrobeniousNorm, self).__init__()

        self.eps = eps

    def forward(self, X):
        B, C, H, W = X.shape
        return torch.sqrt(torch.sum(X ** 2, dim=(1, 2, 3)) + self.eps)
