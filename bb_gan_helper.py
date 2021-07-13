#See pytorch GAN

import argparse
import os
import numpy as np
import math
import main
import time
import logging
import copy
logger = logging.getLogger("logger")

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False

def train_gan(self, epoch, grads):
    
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator with agg_grad (probably)
    # I need to initialize D with global model M but I dont know how. Is it the gradients? 
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    
    # GAN Pseudo ----------------------------------------------------------- #
    # Considering the following as the auxiliary dataset mentioned in the 
    # paper, however this is not quite correct. An auxiliary dataset should be
    # taken from the same set of data as the train and test sets, and it should
    # have true labels like the test set does. This needs to be implemented
    # according to the DBA retrieval of data, although you may get it to compile
    # and run by simply loading the same mnist dataset again.
    #
    # I could not find enough information on the auxiliary dataset, this was
    # one of my setbacks.
    # ---------------------------------------------------------------------- #
    
    # This is our temporary auxiliary dataset used to train D and G
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # GAN Pseudo ----------------------------------------------------------- #
    # Optimizers note: betas appear here from GAN github, but not in our attack code SGD instantiation. 
    # May need to look into this
    # ---------------------------------------------------------------------- #
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    audit_data = []

    # ----------
    #  Training
    # ----------

    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        
        # GAN Pseudo ----------------------------------------------------------- #
        # According to the Wiley GAN research paper, you must initialize D with M.
        #
        #"M includes the information of user data"
        #"we can use M to initialize D at the beginning of each training iteration to reconstruct participants’ training data. #
        # 
        # Consider reaching out to the authors if you wish to pursue this further.
        # ---------------------------------------------------------------------- #
        # INITIALIZE D WITH M HERE


        # -----------------
        #  Train Generator 
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()      

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        # GAN Pseudo ----------------------------------------------------------- #
        # This is for reference and will need to be implemented differently most likely.
        # the GAN model here must produce audit_data that can be compared with
        # the DBA attack data for the next segment of this GAN defense.
        # ---------------------------------------------------------------------- #
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            audit_data = gen_imgs        
            
    return audit_data


class GAN(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory
    

    def aggregate_gradients(self, client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # GAN Pseudo ----------------------------------------------------------- #
        # You must reshape client_grads to work with the GAN defense algorithm
        # Any commented out code below is for FoolsGold but may be useful as reference.
        # ---------------------------------------------------------------------- #
        #grads is reshaping client_grads. 
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            
            #grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))

            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]

        # GAN Pseudo ----------------------------------------------------------- #
        # See gan function below for structure outline. Note that currently this
        # currently returns blank weights and alphas until the gan function is completed
        # ---------------------------------------------------------------------- #
        wv, alpha = self.gan(grads) 

        logger.info(f'[GAN agg] wv: {wv}')
        self.wv_history.append(wv)


        
        # GAN Pseudo ----------------------------------------------------------- #
        # This is the final portion in the paper's outline of the GAN algorithm,
        # updating the globel model (agg_grads)
        # Any commented out code below is for FoolsGold but may be useful as reference.
        # IMPORTANT Note: the GAN defense algorithm will update differently depending
        # on the variable d_iter, representing the iteration where GAN is properly trained.
        # See paper for more details. 
        # ---------------------------------------------------------------------- #
        
        #aggregate gradients for each layer
        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            
            #IF epoch > d_iter, train with M = M + S/CV
            #ELSE, train with M = M + 1/M summation(ukt). 
            #    (this ELSE update should look very similar to the foolsgold update seen in the foolsgold paper algorithm)
            

            #temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                #temp += wv[c] * client_grad[i].cpu()
            #temp = temp / len(client_grads)
            #agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))

        return agg_grads, wv, alpha

    def gan(self, grads):
        # GAN Pseudo ----------------------------------------------------------- #
        # This function is the heart of the algorithm.
        # step1: update D (Discrimintor) with grads (might be part of gan_helper.train_gan below)
        # step2: Train G, then D, with Xaux and produce an auditing dataset
        #        (These steps are accomplished through train_gan)
        audit_data = train_gan(main.epoch, grads)
        #
        #    (note for step 3, no else statement needed, 
        #    #since the model updates are done outside of this function)
        #
        # step3: 
        # if epochs < required G/D training iterations (d_iter)
        #   # Construct classification model M for each participant
        #   foreach x in audit_data do
        #       L[k][x] = M(x)
        #   end
        #           
        #   # Specify labels for audit_data and use audit data to calculate model accuracy a of M
        #   # Initialise S = 0 and CV = 0
        #   cv = 0
        #   s = 0
        #   for all participants (k=1 to N)
        #       if accuracy a > threshold theta
        #           s = s + parameters updates
        #           cv = cv + 1
        #           
        #   
        #   (remember, the M model updates happen outside of this function.)
        #   (to differentiate the two, return current iterations)
        #    
        # return weights, learning rate alphas, and current iterations 
        #-----------------------------------------------

        return None, None, None #cv, alphas, iterations (and anything else you may need)
#-------------------------------------------------------------------#


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
