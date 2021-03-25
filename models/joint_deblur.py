import torch
import tqdm

from models.image_deblur import ImageDeblur
import utils.util as util


class JointDeblur(ImageDeblur):
    def __init__(self, opt):
        super(JointDeblur, self).__init__(opt)

    def deblur(self, y):
        '''Deblur image

        Args:
            y: Blur image
        '''
        y = util.img2tensor(y)

        self.prepare_DIPs()
        self.reset_optimizers()

        warmup_k = torch.load(self.opt['warmup_k_path'])
        self.warmup(warmup_k, y)

        # Input vector of DIPs is sampled from N(z, I)
        reg_noise_std = self.opt['reg_noise_std']
        for step in tqdm(range(self.opt['num_iters'])):
            dip_zx_rand = self.dip_zx + reg_noise_std * torch.randn_like(self.dip_zx)
            dip_zk_rand = self.dip_zk + reg_noise_std * torch.randn_like(self.dip_zk)

            self.x_scheduler.step()
            self.k_scheduler.step()

            self.x_optimizer.zero_grad()
            self.k_optimizer.zero_grad()

            x = self.x_dip(dip_zx_rand)
            k = self.k_dip(dip_zk_rand)

            with torch.no_grad():
                fake_y = self.kernel_wizard.adaptKernel(x, k)

            if step <= self.opt['num_iters'] // 2:
                total_loss = 6e-1 * self.perceptual_loss(fake_y, y)
                total_loss += 1 - self.ssim_loss(fake_y, y)
                total_loss += 6e-4 * torch.norm(k)
            else:
                total_loss = self.perceptual_loss(fake_y, y)
                total_loss += 5e-2 * self.laplace_penalty(x)
                total_loss += 6e-4 * torch.norm(k)

        return util.tensor2img(x)
