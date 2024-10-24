import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        layers = [
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)    

class DecoderBlock(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class VAE(nn.Module):
    def __init__(self, img_ch, img_size, encoder_chs, latent_dims):
        super().__init__()
        self.decoder_chs = encoder_chs[::-1] # Reverse of encoder_chs
        self.latent_img_size = img_size // (2 ** len(encoder_chs))

        self.encoder0 = EncoderBlock(img_ch, encoder_chs[0])
        self.encoder1 = EncoderBlock(encoder_chs[0], encoder_chs[1])
        self.encoder2 = EncoderBlock(encoder_chs[1], encoder_chs[2])

        self.fc_mu = nn.Linear(encoder_chs[2] * (self.latent_img_size ** 2), latent_dims)
        self.fc_var = nn.Linear(encoder_chs[2] * (self.latent_img_size ** 2), latent_dims)

        self.decoder0 = nn.Linear(latent_dims, self.decoder_chs[0] * (self.latent_img_size ** 2))
        self.decoder1 = DecoderBlock(self.decoder_chs[0], self.decoder_chs[1])
        self.decoder2 = DecoderBlock(self.decoder_chs[1], self.decoder_chs[2])

        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_chs[2], self.decoder_chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.decoder_chs[2]),
            nn.ReLU(),    
            nn.Conv2d(self.decoder_chs[2], img_ch, kernel_size=3, padding=1),
            nn.Tanh(),        
        )
    
    def encode(self, x):
        encoder0 = self.encoder0(x)
        encoder1 = self.encoder1(encoder0)
        encoder2 = self.encoder2(encoder1)   

        flatten = torch.flatten(encoder2, start_dim=1)

        mu = self.fc_mu(flatten)
        log_var = self.fc_var(flatten)   
        return mu, log_var

    def decode(self, z):
        decoder0 = self.decoder0(z)
        decoder0 = decoder0.view(-1, self.decoder_chs[0], self.latent_img_size, self.latent_img_size)
        decoder1 = self.decoder1(decoder0)
        decoder2 = self.decoder2(decoder1)
        return self.out(decoder2)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # standard deviation
        eps = torch.randn_like(std) # sample epsilon
        return mu + (eps * std)

    def forward(self, x):
        mu, log_var = self.encode(x)  
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var