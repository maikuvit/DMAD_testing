# File containing all the classes required for the model
import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data.dataset import Dataset


def get_time_embeddings(time_steps, emb_dim):

    assert emb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    factor = 10000 ** ((torch.arange(
        start=0, end=emb_dim // 2, dtype=torch.float32, device=time_steps.device) / (emb_dim // 2))
    )
    

    t_emb = time_steps[:, None].repeat(1, emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class Autoencoder(nn.Module):

    
    def __init__(self, encoder):
        super().__init__()
        
        self.encoder = encoder
        
        self.latent_space = nn.Sequential( # this is to have a flatten representation of the latent space ...
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
             
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3,stride=2, padding=1,  output_padding=1),
        )

        '''
        self.downscaler = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1 ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),

        )

        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256)
        )
        '''

    def forward(self, x):
        x = self.encoder(x)
        #x = self.downscaler(x)
        #print(x.shape)
        x = self.latent_space(x)
        #da ricontrollare le shapes ...
        x = torch.reshape(x, (x.shape[0], 16, 14, 14)) 
        #x = self.upscaler(x)
        x = self.decoder(x)
        return x
    
    def freeze_encoder(self,freeze = True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def freeze_decoder(self,freeze = True):
        for param in self.decoder.parameters():
            param.requires_grad = not freeze

    def get_encoder(self):

        return nn.Sequential(self.encoder,
                             self.downscaler)
    
    def get_decoder(self):
        return nn.Sequential(self.upscaler,
                             self.decoder)


    def get_embeddings(self, x):
        x = self.encoder(x)
        x = self.downscaler(x)
        x = self.latent_space(x)
        return x
    
    def embeddings_to_out(self,x):
        #da ricontrollare le shapes ...
        x = torch.reshape(x, (x.shape[0], 16, 14, 14)) # [1, 16, 14, 14] -> 16 * 14 * 14 = 3136 -> 1 x 56 x 56
        x = self.upscaler(x)
        x = self.decoder(x)
        return x
    
    def freeze_batchNorm(self, Freeze = True):
        
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_ = not Freeze

class FacesDataset(Dataset):
    r"""
    Dataset class to load the Bonafide images. 
    """
    def __init__(self, bonafide_path, morphed_path, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.im_ext = im_ext
        self.bona_fide = self.load_bonafide(bonafide_path)
        self.morphs = self.load_morphs(morphed_path)

    
    def load_bonafide(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        identities = {}
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        
        for p in os.listdir(im_path):
            for i in os.listdir(os.path.join(im_path, p)):
                id = i.split('.')[0] # getting only the number ...
                id = int(id)

                if id not in identities:
                    identities[id] = {}
                identities[id][p] = os.path.join(im_path, p, i)

        print('Found {} identities.'.format(len(identities)))
        return identities
    
    def load_morphs(self, im_path):
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)

        out = []
        for i in os.listdir(im_path): # list of identities 
            for m in os.listdir(os.path.join(im_path, i)): # morphs
                out.append({"morph" : os.path.join(im_path, i, m),
                            "A" : i,
                            "B" : m.split('_')[1].split('.')[0]})
        return out
    
    def __len__(self):
        return len(self.morphs)
    
    def __getitem__(self, index):
        type = "neutral"
    
        item = self.morphs[index]

        morph = Image.open(item["morph"])
        morph = torchvision.transforms.ToTensor()(morph)

        A = int(item["A"])
        im_A = Image.open(self.bona_fide[A][type])  
        im_A = torchvision.transforms.ToTensor()(im_A)

        B = int(item["B"])
        im_B = Image.open(self.bona_fide[B][type])  
        im_B = torchvision.transforms.ToTensor()(im_B)


        # for the https://github.com/serengil/deepface DeepFace model i need the path
        paths = [self.bona_fide[A][type], item["morph"], self.bona_fide[B][type]]

        return im_A, morph, im_B, paths
    
class Demorpher(nn.Module):
    def __init__(self, encoder, decoder , diffusion, scheduler ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.noise_scheduler = scheduler
        self.latent_unet = diffusion
        

    def forward(self, img_A, img_ID, device = "cpu"):

        z_A = self.encoder(img_A)
        z_ID = self.encoder(img_ID)

        z_A = torch.reshape(z_A, (z_A.shape[0], 1, 56, 56))
        z_ID = torch.reshape(z_ID, (z_ID.shape[0], 1, 56, 56))

        print("z_ID nan: ",torch.isnan(z_ID).any())
        print(" z_A nan: ",torch.isnan(z_A).any())
        
        for t in reversed(range(self.noise_scheduler.get_steps())):
            t = torch.as_tensor(t).repeat(z_A.shape[0])
            noise = self.latent_unet(z_ID, t, z_A)
            print("noise nan: ",torch.isnan(noise).any())   
            z_ID, _ = self.noise_scheduler.sample_prev_timestep(z_ID, noise, t)


        z_ID = nn.functional.normalize(z_ID)
        z_ID = torch.reshape(z_ID, (z_ID.shape[0], 16, 14, 14))
        
        out = self.decoder(z_ID)
        
        return out

    def get_unet(self):
        return self.latent_unet

    def get_decoder(self):
        return self.decoder
    
    def freeze_unet(self, Freeze = True):
        for param in self.latent_unet.parameters():
            param.requires_grad = not Freeze

class NoiseScheduler():

    def __init__(self, noise_start, noise_end,steps):
        self.beta_start = noise_start
        self.beta_end = noise_end
        self.noise_steps = steps 
        self.device = "cpu" #default fallback, override in .to(...) method
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

        #define alphas and precompute values ... 
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.inv_sqrt_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)


    def get_noise(self, t):
        return self.betas[t]

    def add_noise(self, img, noise, t):
        shape = img.shape
        batch_size = shape[0]

        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t].reshape(batch_size)
        inv_sqrt_alpha_cumprod = self.inv_sqrt_alpha_cumprod[t].reshape(batch_size)

        for i in range(len(shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            inv_sqrt_alpha_cumprod = inv_sqrt_alpha_cumprod.unsqueeze(-1)

        return sqrt_alpha_cumprod * img + inv_sqrt_alpha_cumprod * noise
    
    def to(self, device):
        self.device = device
        
        self.betas = self.betas.to(device)

        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.inv_sqrt_alpha_cumprod = self.inv_sqrt_alpha_cumprod.to(device)



    def sample_prev_timestep(self,xt,noise_prediction, t):
        
        x0 = (xt - self.inv_sqrt_alpha_cumprod[t] * noise_prediction) / self.sqrt_alpha_cumprod[t]
        x0 = torch.clamp(x0,-1,1)

        mean = xt - ( self.betas[t] * noise_prediction) / self.inv_sqrt_alpha_cumprod[t]
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0: 
            return mean, x0
        
        variance = (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t])
        variance = variance * self.betas[t]
        
        z = torch.randn(xt.shape).to(self.device)

        # i want to free the used memory ...
        del xt, noise_prediction

        return mean + (variance ** .5) * z, x0
    
    def get_steps(self):
        return self.noise_steps

class resnetBlock(nn.Module):
    
        def __init__(self, in_channels, out_channels, embdim, activation = nn.LeakyReLU()):
            super().__init__()
            self.activation = activation
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3,stride = 1,  padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding=1)


            self.time_embedding = nn.Sequential(
                activation,
                nn.Linear(embdim, out_channels)
            )

            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.in_channels = in_channels
            self.out_channels = out_channels
            
            self.res_input_conv = nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x, t_embs):
            input = x
            
            x = self.bn1(x)
            x = self.activation(x)
            x = self.conv1(x)
            x = x + self.time_embedding(t_embs)[:, :, None, None]
            x = self.bn2(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = x + self.res_input_conv(input)
            return x
        
class selfAttentionBlock(nn.Module):

    def __init__(self,in_channels, out_channels, headnums, activation = nn.LeakyReLU()):
        super().__init__()
        self.activation = activation
        self.norm = nn.GroupNorm(8, out_channels)
        self.satt = nn.MultiheadAttention(out_channels, headnums, batch_first=True)
        self.inConv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self,x):
        input = x
        self.b, self.c, self.h, self.w = x.shape
        x = x.reshape(self.b, self.c, self.h * self.w) #reshape to linear ... 

        x = self.norm(x)
        x = x.transpose(1,2) # to put channels as last dimension ...
        x, _ = self.satt(x, x , x)
        x = x.transpose(1, 2).reshape(self.b, self.c, self.h, self.w) #reshape back to original shape
        x = input + x
        return x
    
class downBlock(nn.Module):
    def __init__(self,in_channels, out_channels, emb_dim, downsample ,num_layers = 1):
        super().__init__()
        self.num_layers = num_layers
        self.downsample = downsample

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.downsample else nn.Identity()

        self.resnet = nn.ModuleList([
            resnetBlock(in_channels if i == 0 else out_channels, out_channels, emb_dim)
            for i in range(num_layers)
        ])
        self.selfatt =nn.ModuleList([
            selfAttentionBlock(out_channels, out_channels, 4)
            for _ in range(num_layers)
            ])

    def forward(self, x, t_embs):
        for i in range(self.num_layers):
            x = self.resnet[i](x, t_embs) 
            x = self.selfatt[i](x)
        
        x = self.down_sample_conv(x)
        return x
    
class middleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.initialResnet = resnetBlock(in_channels, out_channels, emb_dim)
        
        self.resnet = nn.ModuleList([
            resnetBlock(out_channels, out_channels, emb_dim)
            for _ in range(num_layers)
        ])
        self.selfatt = nn.ModuleList([
            selfAttentionBlock(out_channels, out_channels, 4)
            for _ in range(num_layers)
        ])

    def forward(self, x, t_embs):

        x = self.initialResnet(x, t_embs)
        
        for i in range(self.num_layers):
            x = self.selfatt[i](x)
            x = self.resnet[i](x, t_embs)
        return x

class upBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, upsample, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.upsample = upsample

        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) if self.upsample else nn.Identity()


        self.resnet = nn.ModuleList([
            resnetBlock(in_channels if i == 0 else out_channels, out_channels, emb_dim)
            for i in range(num_layers)
        ])
        
        self.selfatt = nn.ModuleList([
            selfAttentionBlock(out_channels, out_channels, 4)
            for _ in range(num_layers)
        ])

    def forward(self, x,downblock_out, t_embs):
        x = self.up_sample_conv(x)
        x = x + downblock_out
        for i in range(self.num_layers):
            x = self.resnet[i](x, t_embs)
            x = self.selfatt[i](x)
        return x

class unet(nn.Module):

    def __init__(self, in_channels,activation = nn.LeakyReLU(), layers_activation = nn.LeakyReLU(), num_layers=1):
        super().__init__()
        self.down_channels = [32,64,128,256] 
        self.middle_channels = [256,256,128]
        self.up_channels = [128,64,32,16]
        self.downsample = [True, True, False]
        self.upsample = list(reversed(self.downsample))
        self.emb_dim = 128
        self.num_layers = num_layers
        self.activation = activation

        self.time_proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            self.activation,
            nn.Linear(self.emb_dim, self.emb_dim)
        )

        self.input_conv = nn.Conv2d(in_channels, self.down_channels[0], 3, padding=1)
        self.conditional_emb = nn.Sequential(
            nn.Conv2d(in_channels, self.down_channels[0], 3, padding=1),
            nn.Conv2d(self.down_channels[0], self.down_channels[1], kernel_size=3, stride=2, padding=1) ,
            nn.Conv2d(self.down_channels[1], self.down_channels[2], kernel_size=3, stride=2, padding=1)
            )

        self.down_blocks = nn.ModuleList()

        for i in range(len(self.down_channels ) - 1): # 0 1 2 3 [32,64,128,256]
            self.down_blocks.append( 
                downBlock(self.down_channels[i], self.down_channels[i+1], self.emb_dim,self.downsample[i], num_layers)
            )
           #print("DOWNBLOCKS with shapes: ",self.down_channels[i], self.down_channels[i + 1], i,self.downsample[i])
        self.mid_blocks = nn.ModuleList()

        for i in range(len(self.middle_channels) - 1):
            self.mid_blocks.append(
                middleBlock(self.middle_channels[i], self.middle_channels[i+1], self.emb_dim, num_layers)
            )

        self.up_blocks = nn.ModuleList()
#                self.down_channels = [32,64,128,256] 

        for i in range(len(self.up_channels) - 1):
            self.up_blocks.append(
                upBlock(self.up_channels[i], self.up_channels[i+1], self.emb_dim, self.upsample[i], num_layers)
            )
           #print("UPBLOCKS with shapes: ",self.up_channels[i], self.up_channels[i + 1], i, self.upsample[i])
        

        # final conversion to same shape as input ...
        self.output_norm = nn.GroupNorm(8,self.up_channels[-1])
        self.output_conv = nn.Conv2d(self.up_channels[-1], in_channels, 3, padding=1)

    # adding c to condition ...
    def forward(self, x, t_embs, c = None):
        downblock_outs = []


        x = self.input_conv(x)
        # già qui sale a >3 ...
        
        t_embs = get_time_embeddings(t_embs, self.emb_dim)
        t_embs = self.time_proj(t_embs)

        t_embs = nn.functional.normalize(t_embs)
        
        for i in range(len(self.down_blocks)):
            #qui esplode!!
            x = nn.functional.normalize(x)
            downblock_outs.append(x)
            x = self.down_blocks[i](x, t_embs)

        for i in range(len(self.mid_blocks)):
            x = nn.functional.normalize(x)
            x = self.mid_blocks[i](x, t_embs)
        
        #conditional embeddings ... 

        if c is not None:
            c = self.conditional_emb(c)
            x = torch.add(x, c)
        
        x = nn.functional.normalize(x)

        for i in range(len(self.up_blocks)):
            x = nn.functional.normalize(x)
            dout = downblock_outs.pop()
            x = self.up_blocks[i](x, dout , t_embs)
        x = self.output_norm(x)
        x = self.activation(x)
        x = self.output_conv(x)
        return x