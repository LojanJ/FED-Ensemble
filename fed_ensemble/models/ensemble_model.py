from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm

class VAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim//2)
        )
        self.mean = nn.Linear(hidden_dim//2, latent_dim)
        self.logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, feature):
        encoded = self.encoder(feature)
        return self.mean(encoded), self.logvar(encoded)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, feature):
        mu, logvar = self.encode(feature)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GANModel(nn.Module):
    def __init__(self, input_dim, noise_dim, hidden_dim=128):
        super().__init__()

        # Generator and Discriminator with Dropout and LeaklyReLU
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, input_dim),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def generate(self, z):
        return self.generator(z)
    
    def discriminate(self, feature):
        return self.discriminator(feature)

class EnsembleModel(nn.Module):    
    def __init__(self, 
                 lr, 
                 input_dim, 
                 n_models, 
                 latent_dim=32, 
                 noise_dim=100, 
                 device='cpu'):
        super().__init__()
        self.lr = lr
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.device = device
        self.n_models = n_models
        
        # Initialize multiple VAEs and GANs
        self.vaes = nn.ModuleList([
            VAEModel(input_dim, latent_dim).to(device) 
            for _ in range(n_models)
        ])
        
        self.gans = nn.ModuleList([
            GANModel(input_dim, noise_dim).to(device)
            for _ in range(n_models)
        ])
        
        # Learnable ensemble weights
        self.vae_weights = nn.Parameter(torch.ones(n_models)/n_models)
        self.gan_weights = nn.Parameter(torch.ones(n_models)/n_models)
        
        # Model type weights (VAE vs GAN)
        self.model_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # Initialize optimizers
        self.setup_optimizers(self.lr)

    def setup_optimizers(self, lr=0.0001):
        self.vae_optimizers = [
            optim.Adam(vae.parameters(), lr=lr)
            for vae in self.vaes
        ]
        
        self.gen_optimizers = [
            optim.Adam(gan.generator.parameters(), lr=lr)
            for gan in self.gans
        ]
        
        self.disc_optimizers = [
            optim.Adam(gan.discriminator.parameters(), lr=lr)
            for gan in self.gans
        ]
        
        self.weight_optimizer = optim.Adam([
            self.vae_weights,
            self.gan_weights,
            self.model_weights
        ], lr=lr/2)

    def compute_vae_loss(self, feature, recon_feature, mu, logvar):
        recon_loss = F.mse_loss(recon_feature, feature, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.1 * kl_loss

    def compute_gan_loss(self, discriminator, real_data, fake_data):
        real_labels = torch.ones(real_data.size(0), 1).to(self.device)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(self.device)
        
        # Discriminator loss
        d_real_loss = F.binary_cross_entropy(
            discriminator(real_data), real_labels
        )
        d_fake_loss = F.binary_cross_entropy(
            discriminator(fake_data.detach()), fake_labels
        )
        d_loss = d_real_loss + d_fake_loss
        
        # Generator loss
        g_loss = F.binary_cross_entropy(
            discriminator(fake_data), real_labels
        )
        
        return d_loss, g_loss

    def forward(self, grad, noise):
        # Get outputs from all VAEs
        vae_outputs = []
        vae_mus = []
        vae_logvars = []
        for vae in self.vaes:
            recon, mu, logvar = vae(grad)
            vae_outputs.append(recon)
            vae_mus.append(mu)
            vae_logvars.append(logvar)
        
        # Get outputs from all GANs
        gan_outputs = []
        for gan in self.gans:
            gan_outputs.append(gan.generate(noise))
        
        # Compute weighted VAE output
        vae_weights = F.softmax(self.vae_weights, dim=0)
        weighted_vae_output = sum(w * out for w, out in zip(vae_weights, vae_outputs))
        
        # Compute weighted GAN output
        gan_weights = F.softmax(self.gan_weights, dim=0)
        weighted_gan_output = sum(w * out for w, out in zip(gan_weights, gan_outputs))
        
        # Final ensemble output
        model_weights = F.softmax(self.model_weights, dim=0)
        ensemble_output = (
            model_weights[0] * weighted_vae_output + 
            model_weights[1] * weighted_gan_output
        )
        return {
            'ensemble_output': ensemble_output, 'vae_outputs': vae_outputs, 'gan_outputs': gan_outputs, 'vae_mus': vae_mus,
            'vae_logvars': vae_logvars, 'vae_weights': vae_weights,'gan_weights': gan_weights, 'model_weights': model_weights
        }

    def compute_anomaly(self, features, reputation):
        noise = torch.randn(features.size(0), self.noise_dim).to(self.device)
        outputs = self(features, noise)
        
        #Compute reconstruction errors for each model
        vae_errors = [F.mse_loss(recon, features, reduction='none').mean(1)
                    for recon in outputs['vae_outputs']]
        gan_errors = [F.mse_loss(gen, features, reduction='none').mean(1)
                    for gen in outputs['gan_outputs']]
        
        # Compute model disagreement and stability
        vae_std = torch.std(torch.stack(vae_errors), dim=0)
        gan_std = torch.std(torch.stack(gan_errors), dim=0)
        
        # Compute mean errors with stability weighting
        mean_vae_error = torch.mean(torch.stack(vae_errors), dim=0)
        mean_gan_error = torch.mean(torch.stack(gan_errors), dim=0)
        
        # Compute feature statistics
        feature_mean = torch.mean(features, dim=1)
        feature_std = torch.std(features, dim=1)
        feature_stats = torch.abs(feature_mean) + feature_std
        
        # Combine all signals with adaptive weights
        stability_weight = 1.0 / (1.0 + vae_std + gan_std) 
        error_weight = 1.0 / (1.0 + mean_vae_error + mean_gan_error)  
        feature_weight = 1.0 / (1.0 + feature_stats) 
        
        #  Compute final anomaly score
        anomaly_scores = (
            stability_weight * (mean_vae_error + mean_gan_error) +
            error_weight * (vae_std + gan_std) +
            feature_weight * feature_stats
        )
        
        # Normalize scores
        def normalize(scores):
            min_score = torch.min(scores)
            max_score = torch.max(scores)
            norm_scores = (scores - min_score) / (max_score - min_score + 1e-8)
            return norm_scores
        
        norm_scores = normalize(anomaly_scores)
        adjusted_scores = norm_scores / (reputation + 1e-8)  
        
        return norm_scores, adjusted_scores

    def train_model(self, features, epochs=10):
        features = features.reshape(features.shape[0], -1)
        device = self.device
        all_features = torch.tensor(features, dtype=torch.float32, device=device)
        progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

        def train_vae(vae, optimizer, data):
            optimizer.zero_grad()
            augmented_data = data + torch.randn_like(data) * 0.1
            recon, mu, logvar = vae(augmented_data)
            loss = self.compute_vae_loss(data, recon, mu, logvar)
            
            # L2 regularization
            l2_loss = sum(torch.norm(p) for p in vae.parameters())
            loss += l2_loss * 1e-5
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
            optimizer.step()
            return loss.item()

        def train_gan(gan, gen_opt, disc_opt, noise):
            # Train discriminator
            disc_opt.zero_grad()
            real_data = all_features + torch.randn_like(all_features) * 0.08
            fake_data = gan.generate(noise)
            
            d_loss, _ = self.compute_gan_loss(gan.discriminator, real_data, fake_data)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(gan.discriminator.parameters(), max_norm=1.0)
            disc_opt.step()

            # Train generator twice
            gen_loss = 0
            for _ in range(2):
                gen_opt.zero_grad()
                fake_data = gan.generate(noise)
                _, g_loss = self.compute_gan_loss(gan.discriminator, real_data, fake_data)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), max_norm=1.0)
                gen_opt.step()
                gen_loss += g_loss.item()
                
            return d_loss.item(), gen_loss/2

        with ThreadPoolExecutor() as executor:
            for epoch in progress_bar:
                noise = torch.randn(all_features.size(0), 
                                    self.noise_dim).to(device)
                
                # Parallel training
                vae_futures = [executor.submit(
                    train_vae, vae, opt, all_features) 
                    for vae, opt in zip(self.vaes, self.vae_optimizers)]
                vae_losses = [f.result() for f in vae_futures]
                vae_epoch_loss = sum(vae_losses) / len(self.vaes)

                gan_futures = [executor.
                            submit(train_gan, gan, gen_opt, disc_opt, noise)
                            for gan, gen_opt, disc_opt in
                            zip(self.gans, self.gen_optimizers, self.disc_optimizers)]
                gan_losses = [f.result() for f in gan_futures]
                gan_epoch_loss = sum(d_loss + g_loss 
                                    for d_loss, g_loss in gan_losses) / len(self.gans)

                # Update ensemble weights
                self.weight_optimizer.zero_grad()
                outputs = self(all_features, noise)
                ensemble_loss = F.mse_loss(outputs['ensemble_output'], all_features)
                
                ensemble_weights_reg = (torch.norm(self.vae_weights) + 
                                    torch.norm(self.gan_weights) + 
                                    torch.norm(self.model_weights)) * 1e-5
                (ensemble_loss + ensemble_weights_reg).backward()
                self.weight_optimizer.step()

            # Log training progress
                progress_bar.set_postfix({
                    'Ensemble Loss': f"{ensemble_loss.item():.4f}",
                    'VAE Loss': f"{vae_epoch_loss:.4f}",
                    'GAN Loss': f"{gan_epoch_loss:.4f}"
                })