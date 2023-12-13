import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# Define PointNet++ Encoder
class PointNet2Encoder(nn.Module):
    def __init__(self, in_c=59):
        super(PointNet2Encoder, self).__init__()

        self.in_c = 59

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_c, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the VAE model with PointNet++ Encoder
class VAE(nn.Module):
    def __init__(self, args=None):
        super(VAE, self).__init__()
        
        self.args = args
        self.in_c = 59
        # z_dim = args.z_dim

        # Encoder
        self.encoder = PointNet2Encoder(in_c=self.in_c)

        # Latent space dimensions
        self.latent_dim = args.z_dim # 1024

        # Encoder output to mean and variance
        self.fc_mu = nn.Linear(256, self.latent_dim)
        self.fc_logvar = nn.Linear(256, self.latent_dim)

        # Decoder
        sh_n = 10000 // 2 ** 4
        self.sh_n = sh_n
        self.fc_decoder = nn.Linear(self.latent_dim, 512 * sh_n)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 1, stride=2, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 256, 1, stride=2, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 128, 1, stride=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, self.in_c, 1, stride=2, output_padding=1),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_hat = self.fc_decoder(z)
        x_hat = rearrange(x_hat, 'b (c n) -> b c n', n=self.sh_n)
        x_hat = self.decoder(x_hat)

        # organize the output
        x_hat = rearrange(x_hat, 'b c n -> b n c')
        
        xyz_hat = x_hat[:, :, :3]
        opacities_hat = x_hat[:, :, 3:4]
        features_dc_hat = x_hat[:, :, 4:7]
        features_extra_hat = x_hat[:, :, 7:52]
        scales_hat = x_hat[:, :, 52:55]
        rots_hat = x_hat[:, :, 55:] ## might need to norm
        
        features_dc_hat = features_dc_hat.unsqueeze(-1)
        features_extra_hat = rearrange(features_extra_hat, 'b n (d1 d2) -> b n d1 d2', d1=3, d2=15)
        
        dec_dict = {
            'xyz': xyz_hat,
            'opacities': opacities_hat,
            'features_dc': features_dc_hat,
            'features_extra': features_extra_hat,
            'scales': scales_hat,
            'rots': rots_hat,
        }


        return dec_dict

    def forward(self, x):
        
        xyz, opacities, features_dc, features_extra, scales, rots = x['xyz'], x['opacities'], x['features_dc'], x['features_extra'], x['scales'], x['rots']
        
        # features_dc = features_dc.squeeze(-1) # B x N x 3 x 1 -> B x N x 3. remember to convert it back
        ds_d = features_dc.shape[-1]
        extra_d = features_extra.shape[-1]
        features_dc = rearrange(features_dc, 'b n s0 s1 -> b n (s0 s1)')
        features_extra = rearrange(features_extra, 'b n s0 s1 -> b n (s0 s1)')
        
        x = torch.cat([xyz, opacities, features_dc, features_extra, scales, rots], dim=-1)
        x = rearrange(x, 'b n d -> b d n')
        
        # Encode
        mu, logvar = self.encode(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        dec_dict = self.decode(z)

        ret_dict = {
            'mu': mu,
            'logvar': logvar,
        }
        
        return dec_dict, ret_dict
    
# xyz = torch.rand(10000, 3)
# opacities = torch.rand(10000, 1)
# features_dc = torch.rand(10000, 3, 1)
# features_extra = torch.rand(10000, 3, 15)
# scales = torch.rand(10000, 3)
# rots = torch.rand(10000, 4)