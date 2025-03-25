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

        # Generator with Dropout and LeaklyReLU
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
        
        # Discrimnator with Dropout and LeaklyReLU
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
    def __init__(self, lr, input_dim, latent_dim=32, noise_dim=100, n_models=4, device='cpu'):
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
            'ensemble_output': ensemble_output,
            'vae_outputs': vae_outputs,
            'gan_outputs': gan_outputs,
            'vae_mus': vae_mus,
            'vae_logvars': vae_logvars,
            'vae_weights': vae_weights,
            'gan_weights': gan_weights,
            'model_weights': model_weights
        }

    def compute_anomaly_score(self, features):
        noise = torch.randn(features.size(0), self.noise_dim).to(self.device)
        outputs = self(features, noise)
        
        # Compute reconstruction errors for each model
        vae_errors = [F.mse_loss(recon, features, reduction='none').mean(1)
                     for recon in outputs['vae_outputs']]
        gan_errors = [F.mse_loss(gen, features, reduction='none').mean(1)
                     for gen in outputs['gan_outputs']]
        
        # Compute model disagreement
        vae_std = torch.std(torch.stack(vae_errors), dim=0)
        gan_std = torch.std(torch.stack(gan_errors), dim=0)
        
        # Compute mean errors
        mean_vae_error = torch.mean(torch.stack(vae_errors), dim=0)
        mean_gan_error = torch.mean(torch.stack(gan_errors), dim=0)
        
        # Combine mean errors and disagreements
        anomaly_scores = (0.5 * mean_vae_error + 0.5 * mean_gan_error +
                        0.25 * vae_std + 0.25 * gan_std)
        
            # Normalize scores 
        min_score = torch.min(anomaly_scores)
        max_score = torch.max(anomaly_scores)
        anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score + 1e-8)

        return anomaly_scores

    def train_model(self, features, baseline_established, epochs=10):
        features = features.reshape(features.shape[0], -1)
        device = self.device
        all_features = torch.tensor(features, dtype=torch.float32, device=device)
        progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")


        for epoch in progress_bar:
            noise = torch.randn(all_features.size(0), self.noise_dim).to(device)
            
            # Train VAEs with augmented data
            vae_epoch_loss = 0
            for vae_idx, (vae, optimizer) in enumerate(zip(self.vaes, self.vae_optimizers)):
                optimizer.zero_grad()
                
                # Data augmentation for robustness
                augmented_features = all_features + torch.randn_like(all_features) * 0.1
                
                recon, mu, logvar = vae(augmented_features)
                loss = self.compute_vae_loss(all_features, recon, mu, logvar)

                # Add L2 regularization
                l2_loss = 0
                for param in vae.parameters():
                    l2_loss += torch.norm(param)
                
                loss = loss + (l2_loss * 1e-5)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
                optimizer.step()
                vae_epoch_loss += loss.item()
            
            vae_epoch_loss /= len(self.vaes)
            
            # Train GANs
            gan_epoch_loss = 0
            for gan_idx, (gan, gen_opt, disc_opt) in enumerate(zip(self.gans, self.gen_optimizers, self.disc_optimizers)):
               
                # Train discriminator
                disc_opt.zero_grad()
                
                # Data Augmentation
                augmented_features = all_features + torch.randn_like(all_features) * 0.08
                fake_data = gan.generate(noise)

                d_loss, _ = self.compute_gan_loss(gan.discriminator, all_features, fake_data)

                # L2 regularization
                d_reg_loss = 0
                for param in gan.discriminator.parameters():
                    d_reg_loss += torch.norm(param)
                
                # Combined loss with regularization
                d_loss = d_loss + (d_reg_loss * 1e-5)
                d_loss.backward()

                torch.nn.utils.clip_grad_norm_(gan.discriminator.parameters(), max_norm=1.0)
                disc_opt.step()
                
                # Train generator (multiple steps for better convergence)
                gen_loss = 0
                for _ in range(2):  # Multiple generator updates per discriminator update
                    gen_opt.zero_grad()
                    fake_data = gan.generate(noise)
                    _, g_loss = self.compute_gan_loss(gan.discriminator, all_features, fake_data)
                    g_reg_loss = 0
                    for param in gan.generator.parameters():
                        g_reg_loss += torch.norm(param)

                    g_loss = g_loss + (1e-5 * g_reg_loss) * 0.5  # Lower weight for generator regularization
                    g_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), max_norm=1.0)
                    gen_opt.step()
                    gen_loss += g_loss.item()
                
                gan_epoch_loss += (d_loss.item() + gen_loss/2)
            
            gan_epoch_loss /= len(self.gans)
            
            # Update ensemble weights
            self.weight_optimizer.zero_grad()
            outputs = self(all_features, noise)
            ensemble_loss = F.mse_loss(outputs['ensemble_output'], all_features)

            ensemble_weights_reg = (torch.norm(self.vae_weights) + 
                       torch.norm(self.gan_weights) + 
                       torch.norm(self.model_weights)) * 1e-5

            # Combined loss with regularization
            ensemble_total_loss = ensemble_loss + ensemble_weights_reg
            ensemble_total_loss.backward()
            self.weight_optimizer.step()
            

            # Early stopping based on validation performance
            if epoch > 5 and epoch % 5 == 0:
                # Compute validation scores on a subset of the data
                val_indices = torch.randperm(len(all_features))[:min(100, len(all_features)//4)]
                val_features = all_features[val_indices]
                val_noise = torch.randn(val_features.size(0), self.noise_dim).to(device)
                
                # Get outputs from the ensemble
                with torch.no_grad():
                    val_outputs = self(val_features, val_noise)
                    val_recon_error = F.mse_loss(val_outputs['ensemble_output'], val_features).item()
                
                # Store validation error history if not already tracking
                if not hasattr(self, 'val_error_history'):
                    self.val_error_history = []
                    self.previous_val_loss = float('inf')
                
                self.val_error_history.append(val_recon_error)
                
                # Check if validation error is increasing (sign of overfitting)
                if len(self.val_error_history) >= 2:
                    if val_recon_error > self.previous_val_loss * 1.2:  # 20% increase threshold
                        print(f"Early stopping at epoch {epoch}: validation error increased from {self.previous_val_loss:.4f} to {val_recon_error:.4f}")
                        break
                
                # Update previous validation loss
                self.previous_val_loss = val_recon_error
                
                # Log validation performance
                progress_bar.set_postfix({
                    'Ensemble Loss': f"{ensemble_loss.item():.4f}",
                    'VAE Loss': f"{vae_epoch_loss:.4f}",
                    'GAN Loss': f"{gan_epoch_loss:.4f}",
                    'Val Error': f"{val_recon_error:.4f}"
                })
            