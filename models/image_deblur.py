import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import utils.util as util
import models.arch_util as arch_util
from models.loss import SSIM, PerceptualLoss, HyperLaplacePenalty
from models.kernel_wizard import KernelWizard
from models.unet_parts import UnetSkipConnectionBlock


class ImageDIP(nn.Module):
    '''
    DIP (Deep Image Prior) for sharp image
    '''
    def __init__(self, opt):
        super(ImageDIP, self).__init__()
        input_nc = opt["nf"]
        output_nc = opt["nf"]
        ngf = opt["nf"]
        norm_layer = arch_util.get_norm_layer(opt["norm"])

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, img):
        return self.model(img, None)


class KernelDIP(nn.Module):
    '''
    DIP (Deep Image Prior) for blur kernel
    '''
    def __init__(self, opt):
        super(KernelDIP, self).__init__()

        norm_layer = arch_util.get_norm_layer('none')
        n_blocks = opt['n_blocks']
        nf = opt['nf']
        padding_type = opt['padding_type']
        use_dropout = opt['use_dropout']
        kernel_dim = opt['kernel_dim']

        input_nc = 64
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, nf, kernel_size=7, padding=0, bias=True),
                 norm_layer(nf),
                 nn.ReLU(True)]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            input_nc = min(nf * mult, kernel_dim)
            output_nc = min(nf * mult * 2, kernel_dim)
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(nf * mult * 2),
                      nn.ReLU(True)]

        for i in range(n_blocks):   # add ResNet blocks
            model += [
                arch_util.ResnetBlock(kernel_dim,
                                      padding_type=padding_type,
                                      norm_layer=norm_layer,
                                      use_dropout=use_dropout,
                                      use_bias=True)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, noise):
        return self.model(noise)


class ImageDeblur:
    def __init__(self, opt):
        self.opt = opt

        self.dtype = torch.cuda.FloatTensor

        # losses
        self.ssim_loss = SSIM().type(self.dtype)
        self.mse = nn.MSELoss().type(self.dtype)
        self.perceptual_loss = PerceptualLoss().type(self.dtype)
        self.laplace_penalty = HyperLaplacePenalty().type(self.dtype)

        self.kernel_wizard = KernelWizard(opt['KernelWizard'])
        self.kernel_wizard.load_state_dict(torch.load(opt['KernelWizard']['pretrained']))

    def reset_optimizers(self):
        self.x_optimizer = torch.optim.Adam(self.x_dip.parameters(), lr=self.opt['x_lr'])
        self.k_optimizer = torch.optim.Adam(self.k_dip.parameters(), lr=self.opt['k_lr'])

        self.x_scheduler = StepLR(self.x_optimizer, step_size=self.num_iters // 5, gamma=0.7)

        self.k_scheduler = StepLR(self.k_optimizer, step_size=self.num_iters // 5, gamma=0.7)

    def prepare_DIPs(self):
        # x is stand for the sharp image, k is stand for the kernel
        self.x_dip = ImageDIP(self.opt['ImageDIP'])
        self.k_dip = ImageDIP(self.opt['KernelDIP'])

        # fixed input vectors of DIPs
        # zk and zx are the length of the corresponding vectors
        self.dip_zk = util.get_noise(8, 'noise', (64, 64)).type(self.dtype)
        self.dip_zx = util.get_noise(8, 'noise', self.opt['img_size']).type(self.dtype)

    def warmup(self, warmup_x, warmup_k):
        # Input vector of DIPs is sampled from N(z, I)
        reg_noise_std = self.opt['reg_noise_std']

        for step in self.opt['num_warmup_iters']:
            self.k_optimizer.zero_grad()
            dip_zk_rand = self.dip_zk + reg_noise_std * torch.randn_like(self.dip_zk)
            k = self.dip_k(dip_zk_rand)

            loss = self.mse(k, warmup_k)
            loss.backward()
            self.k_optimizer.step()

        for step in self.opt['num_warmup_iters']:
            self.x_optimizer.zero_grad()
            dip_zx_rand = self.dip_zx + reg_noise_std * torch.randn_like(self.dip_zx)
            x = self.dip_x(dip_zx_rand)

            loss = self.mse(x, warmup_x)
            loss.backward()
            self.x_optimizer.step()

    def deblur(self, img):
        pass
