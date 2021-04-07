import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import utils.util as util
from models.losses.pulse_loss import LossBuilder
from models.dips import KernelDIP
from models.pulse.spherical_optimizer import SphericalOptimizer
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class PULSE(torch.nn.Module):
    def __init__(self, opt, cache_dir):
        super(PULSE, self).__init__()

        self.opt = opt

        self.verbose = opt["verbose"]
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print("Loading Synthesis Network")

        self.stylegan_ver = opt['stylegan_ver']
        if self.stylegan_ver == 2:
            from models.pulse.stylegan2 import Generator

            self.synthesis = Generator(size=256, style_dim=512, n_mlp=8).cuda()
            self.synthesis.load_state_dict(torch.load("experiments/pretrained/stylegan2.pt")["g_ema"], strict=True)
        else:
            from models.pulse.stylegan import G_synthesis

            self.synthesis = G_synthesis().cuda()
            self.synthesis.load_state_dict(torch.load("experiments/pretrained/stylegan.pt"))

        for param in self.synthesis.parameters():
            param.requires_grad = False

        # Losses
        self.mse = nn.MSELoss().cuda()

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        if Path("experiments/pretrained/gaussian_fit.pt").exists():
            self.gaussian_fit = torch.load("experiments/pretrained/gaussian_fit.pt")
        else:
            if self.verbose:
                print("\tRunning Mapping Network")
            with torch.no_grad():
                torch.manual_seed(0)
                latent = torch.randn((1000000, 512), dtype=torch.float32, device="cuda")
                latent_out = torch.nn.LeakyReLU(5)(self.synthesis.get_latent(latent))
                self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
                torch.save(self.gaussian_fit, "experiments/pretrained/gaussian_fit.pt")
                if self.verbose:
                    print('\tSaved "gaussian_fit.pt"')

    def make_noise(self, num_trainable_noise_layers):
        log_size = int(math.log(256, 2))
        num_layers = (log_size - 2) * 2 + 1

        noises = []
        noise_vars = []  # stores the noise tensors that we want to optimize on

        for i in range(num_layers + 1):
            new_noise = torch.randn(1, 1, 2 ** i, 2 ** i, device="cuda")
            if i < num_trainable_noise_layers:
                new_noise.requires_grad = True
                noise_vars.append(new_noise)
            else:
                new_noise.requires_grad = False

            noises.append(new_noise)

        return noises, noise_vars

    def warmup(self, k_dip, dip_zk, optimizer_k, warmup_k):
        # Input vector of DIPs is sampled from N(z, I)
        reg_noise_std = self.opt["reg_noise_std"]

        print("Warming up k DIP")
        for step in tqdm(range(self.opt["num_warmup_iters"])):
            optimizer_k.zero_grad()
            dip_zk_rand = dip_zk + reg_noise_std * torch.randn_like(dip_zk).cuda()
            k = k_dip(dip_zk_rand)

            loss = self.mse(k, warmup_k)
            loss.backward()
            optimizer_k.step()

    def forward(self, ref_im):
        if self.opt["seed"]:
            seed = self.opt["seed"]
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]

        # Generate latent tensor
        if self.opt["tile_latent"]:
            latent = torch.randn((batch_size, 1, 512), dtype=torch.float, requires_grad=True, device="cuda")
        else:
            latent = torch.randn((batch_size, 14, 512), dtype=torch.float, requires_grad=True, device="cuda")

        # Generate list of noise tensors
        noise = []  # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        # noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(14):
            res = (i + 5) // 2
            res = [1, 1, 2 ** res, 2 ** res]

            noise_type = self.opt["noise_type"]
            bad_noise_layers = self.opt["bad_noise_layers"]
            if noise_type == "zero" or i in [int(layer) for layer in bad_noise_layers.split(".")]:
                new_noise = torch.zeros(res, dtype=torch.float, device="cuda")
                new_noise.requires_grad = False
            elif noise_type == "fixed":
                new_noise = torch.randn(res, dtype=torch.float, device="cuda")
                new_noise.requires_grad = False
            elif noise_type == "trainable":
                new_noise = torch.randn(res, dtype=torch.float, device="cuda")
                if i < self.opt["num_trainable_noise_layers"]:
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                else:
                    new_noise.requires_grad = False
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)
        # noise, noise_vars = self.make_noise(num_trainable_noise_layers)

        var_list = [latent] + noise_vars

        optimizer_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "sgdm": partial(torch.optim.SGD, momentum=0.9),
            "adamax": torch.optim.Adamax,
        }
        optimizer_func = optimizer_dict[self.opt["optimizer_name"]]
        optimizer_x = SphericalOptimizer(optimizer_func, var_list, lr=self.opt["x_lr"])

        steps = self.opt["num_epochs"] * self.opt["num_x_iters"]
        schedule_dict = {
            "fixed": lambda x: 1,
            "linear1cycle": lambda x: (9 * (1 - np.abs(x / steps - 1 / 2) * 2) + 1) / 10,
            "linear1cycledrop": lambda x: (9 * (1 - np.abs(x / (0.9 * steps) - 1 / 2) * 2) + 1) / 10
            if x < 0.9 * steps
            else 1 / 10 + (x - 0.9 * steps) / (0.1 * steps) * (1 / 1000 - 1 / 10),
        }
        schedule_func = schedule_dict[self.opt["lr_schedule"]]
        scheduler_x = torch.optim.lr_scheduler.LambdaLR(optimizer_x.opt, schedule_func)

        # k dip:
        dip_zk = util.get_noise(64, "noise", (64, 64)).cuda().detach()
        k_dip = KernelDIP(self.opt["KernelDIP"]).cuda()
        optimizer_k = torch.optim.Adam(k_dip.parameters(), lr=self.opt["k_lr"])
        scheduler_k = StepLR(optimizer_k, step_size=self.opt["num_epochs"] * self.opt["num_k_iters"] // 5, gamma=0.7)
        self.warmup(k_dip, dip_zk, optimizer_k, torch.load("experiments/pretrained/kernel.pth"))

        loss_builder = LossBuilder(ref_im, self.opt).cuda()

        min_loss = np.inf
        best_summary = ""
        gen_im = None

        if self.verbose:
            print("Optimizing")

        for epoch in range(self.opt["num_epochs"]):
            print("Step: {}".format(epoch + 1))
            # Optimize x
            tq_x = tqdm(range(self.opt["num_x_iters"]))
            for j in tq_x:
                for p in k_dip.parameters():
                    p.requires_grad = False
                latent.requires_grad = True

                optimizer_x.opt.zero_grad()

                # Duplicate latent in case tile_latent = True
                if self.opt["tile_latent"]:
                    latent_in = latent.expand(-1, 14, -1)
                else:
                    latent_in = latent

                kernel = k_dip(dip_zk)
                # Apply learned linear mapping to match latent distribution to that of the mapping network
                latent_in = self.lrelu(latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"])

                # Normalize image to [0,1] instead of [-1,1]
                if self.stylegan_ver == 1:
                    gen_im = (self.synthesis(latent_in, noise) + 1) / 2
                else:
                    gen_im = (self.synthesis([latent_in], input_is_latent=True, noise=noise)[0] + 1) / 2

                # Calculate Losses
                loss, loss_dict = loss_builder(latent_in, gen_im, kernel, epoch)

                msg = " | ".join("{}: {:.4f}".format(k, v) for k, v in loss_dict.items())
                tq_x.set_postfix(loss=msg)

                # Save best summary for log
                if loss < min_loss:
                    min_loss = loss
                    best_summary = f"BEST ({j+1}) | " + " | ".join([f"{x}: {y:.4f}" for x, y in loss_dict.items()])
                    best_summary += " | Kernel norm: {:.2f}".format(torch.norm(kernel))
                    best_im = gen_im.clone()

                loss.backward()
                optimizer_x.step()
                scheduler_x.step()
            if self.opt["save_intermediate"]:
                yield (
                    best_im.cpu().detach().clamp(0, 1),
                    loss_builder.D.test(best_im, kernel).cpu().detach().clamp(0, 1),
                )

            # Optimize k
            tq_k = tqdm(range(self.opt["num_k_iters"]))
            for j in tq_k:
                for p in k_dip.parameters():
                    p.requires_grad = True
                latent.requires_grad = False

                optimizer_k.zero_grad()

                # Duplicate latent in case tile_latent = True
                if self.opt["tile_latent"]:
                    latent_in = latent.expand(-1, 14, -1)
                else:
                    latent_in = latent

                kernel = k_dip(dip_zk)
                # Apply learned linear mapping to match latent distribution to that of the mapping network
                latent_in = self.lrelu(latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"])

                # Normalize image to [0,1] instead of [-1,1]
                gen_im = (self.synthesis([latent_in], input_is_latent=True, noise=noise)[0] + 1) / 2

                # Calculate Losses
                loss, loss_dict = loss_builder(latent_in, gen_im, kernel, epoch)

                msg = " | ".join("{}: {:.4f}".format(k, v) for k, v in loss_dict.items())
                tq_k.set_postfix(loss=msg)

                loss.backward()
                optimizer_k.step()
                scheduler_k.step()

            if self.opt["save_intermediate"]:
                yield (
                    best_im.cpu().detach().clamp(0, 1),
                    loss_builder.D.test(best_im, kernel).cpu().detach().clamp(0, 1),
                )

        yield (
            gen_im.clone().cpu().detach().clamp(0, 1),
            loss_builder.D.test(best_im, kernel).cpu().detach().clamp(0, 1),
        )
