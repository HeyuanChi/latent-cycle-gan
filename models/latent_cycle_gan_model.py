import torch
import torch.nn as nn
import itertools
import numpy as np

from diffusers import DiffusionPipeline

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class LatentCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=10.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for GAN loss')
            parser.add_argument('--alpha_gan', type=float, default=0.2, help='weight for latent GAN loss')
            parser.add_argument('--alpha_A', type=float, default=0.01, help='weight for latent cycle loss (A -> B -> A)')
            parser.add_argument('--alpha_B', type=float, default=0.01, help='weight for latent cycle loss (B -> A -> B)')
            # SD
            parser.add_argument('--stable_diffusion_dir', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
            # LoRA
            parser.add_argument('--lora_A_dir', type=str, default='/root/autodl-tmp/diffusers/outputs/output-vangogh/checkpoint-5000')
            parser.add_argument('--lora_B_dir',   type=str, default='/root/autodl-tmp/diffusers/outputs/output-photo/checkpoint-5000')
            # prompt
            parser.add_argument('--text_prompt_vangogh', type=str, default='a paint of vangogh')
            parser.add_argument('--text_prompt_photo',   type=str, default='a realistic photograph')
            # init with cycle gan
            parser.add_argument('--init_with_cycle_gan', type=bool, default=False)
            parser.add_argument('--cycle_gan_dir', type=str, default='/root/autodl-tmp/pytorch-CycleGAN-and-pix2pix/checkpoints/vangogh2photo/')
            
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # training losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B',
                           'D_LA', 'G_LA', 'cycle_LA', 'D_LB', 'G_LB', 'cycle_LB']
        # images to save/display
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B
        
        # models
        if self.isTrain:
            self.model_names = ['G_A','G_B','D_A','D_B','D_LA','D_LB']
        else:
            self.model_names = ['G_A','G_B']

        # define networks 
        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids
        )
        self.netG_B = networks.define_G(
            opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids
        )
        if self.isTrain and self.opt.init_with_cycle_gan:
            self.load_networks(epoch='latest', model_names=['G_A', 'G_B'], load_dir=self.opt.cycle_gan_dir)
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                opt.norm, opt.init_type, opt.init_gain, self.gpu_ids
            )
            self.netD_B = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                opt.norm, opt.init_type, opt.init_gain, self.gpu_ids
            )
            if self.opt.init_with_cycle_gan:
                self.load_networks(epoch='latest', model_names=['D_A', 'D_B'], load_dir=self.opt.cycle_gan_dir)
            # latent discriminators
            self.netD_LA = networks.define_D(
                input_nc=4, ndf=64, netD='basic', n_layers_D=3, norm=opt.norm,
                init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids
            )
            self.netD_LB = networks.define_D(
                input_nc=4, ndf=64, netD='basic', n_layers_D=3, norm=opt.norm,
                init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids
            )

        # load SD with LoRA
        if self.isTrain:
            # pipeA
            self.pipeA = DiffusionPipeline.from_pretrained(
                self.opt.stable_diffusion_dir,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True
            ).to(self.device)
            self.pipeA.load_lora_weights(opt.lora_A_dir)
            self.pipeA.vae.requires_grad_(False)
            self.pipeA.unet.requires_grad_(False)

            # pipeB
            self.pipeB = DiffusionPipeline.from_pretrained(
                self.opt.stable_diffusion_dir,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True
            ).to(self.device)
            self.pipeB.load_lora_weights(opt.lora_B_dir)
            self.pipeB.vae.requires_grad_(False)
            self.pipeB.unet.requires_grad_(False)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN    = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle  = nn.L1Loss()
            self.criterionIdt    = nn.L1Loss()
            self.criterionCycleL = nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_LA.parameters(), self.netD_LB.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = (self.opt.direction == 'AtoB')
        self.real_A = input['A' if AtoB else 'B'].to(self.device)  # vangogh
        self.real_B = input['B' if AtoB else 'A'].to(self.device)  # photo
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
    
    #---------------------------------------
    # GAN loss for discriminators
    #---------------------------------------
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = 0.5*(loss_D_real + loss_D_fake)
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) 

    #---------------------------------------
    # latent GAN loss for discriminators
    #---------------------------------------
    def backward_D_latent(self, pipe, netD, real, fake, ratio):
        """Calculate latent GAN loss for discriminator D_LA"""
        # encode real with pipe.vae
        z_real = pipe.vae.encode(real.half()).latent_dist.sample() * 0.18215
        # encode fake with pipe.vae
        z_fake = pipe.vae.encode(fake.detach().half()).latent_dist.sample() * 0.18215
        # sample t
        t_max = max(int(1000 * ratio / 4), 1)  # t_max >= 1
        t = np.random.randint(0, t_max)
        # add noise
        noise_real = torch.randn_like(z_real)
        noise_fake = torch.randn_like(z_fake)
        alpha_bar = pipe.scheduler.alphas_cumprod[t]
        z_real_t = alpha_bar.sqrt() * z_real + (1 - alpha_bar).sqrt() * noise_real
        z_fake_t = alpha_bar.sqrt() * z_fake + (1 - alpha_bar).sqrt() * noise_fake
        # discriminate
        pred_real = netD(z_real_t.to(torch.float32))
        loss_D_real_latent = self.criterionGAN(pred_real, True)
        pred_fake = netD(z_fake_t.to(torch.float32))
        loss_D_fake_latent = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D_latent = 0.5*(loss_D_real_latent + loss_D_fake_latent)
        loss_D_latent.backward()
        return loss_D_latent
    
    def backward_D_LA(self, ratio):
        """Calculate latent GAN loss for discriminator D_LA"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_LA = self.backward_D_latent(self.pipeB, self.netD_LA, self.real_B, fake_B, ratio)

    def backward_D_LB(self, ratio):
        """Calculate latent GAN loss for discriminator D_LB"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_LB = self.backward_D_latent(self.pipeA, self.netD_LB, self.real_A, fake_A, ratio)

    #---------------------------------------
    # loss for generators
    #---------------------------------------
    def backward_gan_latent(self, netD, pipe, fake, ratio):
        """Calculate the latent cycle loss for generators G_A and G_B"""
        # encode fake with pipe.vae
        z_fake = pipe.vae.encode(fake.half()).latent_dist.sample() * 0.18215
        # sample t
        t_max = max(int(1000 * ratio / 4), 1)  # t_max >= 1
        t = np.random.randint(0, t_max)
        # add noise
        noise_fake = torch.randn_like(z_fake)
        alpha_bar = pipe.scheduler.alphas_cumprod[t]
        z_fake_t = alpha_bar.sqrt() * z_fake + (1 - alpha_bar).sqrt() * noise_fake
        # discriminate
        pred_fake = netD(z_fake_t.to(torch.float32))
        # loss
        loss_gan_latent = self.criterionGAN(pred_fake, True)
        return loss_gan_latent

    def backward_cycle_latent(self, pipe, prompt, rec, ratio):
        """Calculate the latent cycle loss for generators G_A and G_B"""
        # encode rec with pipe.vae
        z_rec = pipe.vae.encode(rec.half()).latent_dist.sample() * 0.18215
        with torch.no_grad():
            # text prompt
            emb = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            text_emb = pipe.text_encoder(emb)[0]
            real_batch_size = rec.size(0)
            text_emb = text_emb.repeat(real_batch_size, 1, 1)
        # sample t
        t_max = max(int(1000 * ratio / 4), 1)  # t_max >= 1
        t = np.random.randint(0, t_max)
        # add noise
        noise_rec = torch.randn_like(z_rec)
        alpha_bar = pipe.scheduler.alphas_cumprod[t]
        z_rec_t = alpha_bar.sqrt() * z_rec + (1 - alpha_bar).sqrt() * noise_rec
        # unet
        eps_pred_rec = pipe.unet(z_rec_t, t, encoder_hidden_states=text_emb).sample
        # loss
        loss_cycle_latent = self.criterionCycleL(eps_pred_rec, noise_rec)
        return loss_cycle_latent

    def backward_G(self, ratio):
        """Calculate the loss for generators G_A and G_B"""
        lambda_gan = self.opt.lambda_gan
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        alpha_gan = self.opt.alpha_gan
        alpha_A = self.opt.alpha_A
        alpha_B = self.opt.alpha_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * lambda_gan
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * lambda_gan
        # cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # latent GAN loss
        self.loss_G_LA = self.backward_gan_latent(self.netD_LA, self.pipeB, self.fake_B, ratio) * alpha_gan
        self.loss_G_LB = self.backward_gan_latent(self.netD_LB, self.pipeA, self.fake_A, ratio) * alpha_gan
        # latent cycle loss
        self.loss_cycle_LA = self.backward_cycle_latent(self.pipeA, self.opt.text_prompt_vangogh, self.rec_A, ratio) * alpha_A
        self.loss_cycle_LB = self.backward_cycle_latent(self.pipeB, self.opt.text_prompt_photo, self.rec_B, ratio) * alpha_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + \
                      self.loss_G_LA + self.loss_G_LB + self.loss_cycle_LA + self.loss_cycle_LB
        self.loss_G.backward()

    #---------------------------------------
    # optimize_parameters
    #---------------------------------------
    def optimize_parameters(self, ratio=None):
        """Calculate losses, gradients, and update network weights"""
        # forward
        self.forward()
        # Generators
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_LA, self.netD_LB], False)
        self.optimizer_G.zero_grad()
        self.backward_G(ratio)
        self.optimizer_G.step()
        # Discriminators
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_LA, self.netD_LB], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.backward_D_LA(ratio)
        self.backward_D_LB(ratio)
        self.optimizer_D.step()
