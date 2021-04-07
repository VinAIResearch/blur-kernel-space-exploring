import torch

from models.pulse.pulse import PULSE
from models.pulse.stylegan import G_synthesis
from models.losses.pulse_loss import LossBuilderStyleGAN


class PULSEStyleGAN(PULSE):
    def __init__(self, opt, cache_dir):
        super(PULSEStyleGAN, self).__init__(opt, cache_dir)

    def load_synthesis_network(self):
        self.synthesis = G_synthesis().cuda()
        self.synthesis.load_state_dict(torch.load('experiments/pretrained/stylegan.pt'), strict=True)

    def initialize_latent_space(self):
        batch_size = self.opt['batch_size']

        # Generate latent tensor
        if(self.opt['tile_latent']):
            self.latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            self.latent = torch.randn(
                (batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')

        # Generate list of noise tensors
        noise = []  # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        noise_type = self.opt["noise_type"]
        bad_noise_layers = self.opt["bad_noise_layers"]
        for i in range(18):
            # dimension of the ith noise tensor
            res = (batch_size, 1, 2**(i//2+2), 2**(i//2+2))

            if(noise_type == 'zero' or i in [int(layer) for layer in bad_noise_layers.split('.')]):
                new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif(noise_type == 'fixed'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif (noise_type == 'trainable'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                if (i < self.opt['num_trainable_noise_layers']):
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                else:
                    new_noise.requires_grad = False
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)

        self.latent_x_var_list = [self.latent] + noise_vars
        self.noise = noise

    def initialize_loss(self, ref_im):
        self.loss_builder = LossBuilderStyleGAN(ref_im, self.opt).cuda()

    def get_gen_im(self, latent_in):
        return (self.synthesis(latent_in, self.noise)+1)/2
