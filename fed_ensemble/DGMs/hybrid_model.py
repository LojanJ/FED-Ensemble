from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class EnsembleModel(nn.Module):
    def __init__(self, input_dim, latent_dim, noise_dim, device, lr=0.001, gamma=0.1):
        super(EnsembleModel, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.batch_norm = nn.BatchNorm1d(input_dim)
        # Add feature normalization layers
        
        # Initialize models
        self.projection = None
        self.encoder, self.mean, self.logvar = self.create_encoder(input_dim, latent_dim)
        self.decoder = self.create_decoder(latent_dim, input_dim)
        self.generator = self.create_generator(noise_dim, input_dim)
        self.discriminator = self.create_discriminator(input_dim)
        
        # Move models to device
        self.to(device)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5], device=device))

        # Optimizers
        self.optimizers = {
            'vae': optim.AdamW(
                list(self.encoder.parameters()) +
                list(self.decoder.parameters()) +
                list(self.mean.parameters()) +
                list(self.logvar.parameters()), 
                lr=lr,
                weight_decay=0.1),
            'gen': optim.Adam(self.generator.parameters(), lr=lr, weight_decay=0.1),
            'disc': optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=0.1),
            'weights': optim.Adam([self.ensemble_weights], lr=lr/2, weight_decay=0.1)
        }

        # Learning rate schedulers
        self.schedulers = {
            key: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=gamma)
            for key, opt in self.optimizers.items()
        }

    @staticmethod
    def create_encoder(input_dim, latent_dim):
        encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False)
        )
        mean = nn.Linear(64, latent_dim)
        logvar = nn.Linear(64, latent_dim)
        return encoder, mean, logvar

    @staticmethod
    def create_decoder(latent_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, output_dim)
        )

    @staticmethod
    def create_generator(noise_dim, output_dim):
        return nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    @staticmethod
    def create_discriminator(input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _initialize_projection(self, input_shape):
        actual_input_dim = input_shape[-1]
        self.projection = nn.Sequential(
            nn.Linear(actual_input_dim, self.input_dim),
            nn.ReLU(),
            nn.LayerNorm(self.input_dim)
        ).to(self.device)

        # Initialize weights properly
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)

        # Add projection parameters to optimizer
        self.optimizers['projection'] = torch.optim.Adam(
            self.projection.parameters(),
            lr=self.optimizers['vae'].param_groups[0]['lr'],
            weight_decay=0.1
        )
        self.schedulers['projection'] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['projection'],
            step_size=10,
            gamma=self.schedulers['vae'].gamma
        )
        return self.projection
        
    def normalize_features(self, features, update_stats=True):

        features = self._prepare_tensor(features)
        print(features)
        # Force 2D: Add batch dimension if missing
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Shape: (1, input_dim)
        
        # Initialize projection if needed
        if self.projection is None or features.size(-1) != self.input_dim:
            self._initialize_projection(features.shape)
        
        # Apply projection
        features = self.projection(features)  # Shape: (batch_size, output_dim)
        
        # Only apply BatchNorm for batch_size > 1
        if update_stats and features.size(0) > 1:
            features = self.batch_norm(features)

        features = (features - torch.mean(features)) / (torch.std(features) + 1e-8)
        return features
        
    def forward(self, features, noise):
        # VAE outputs
        encoded = self.encoder(features)
        mu = self.mean(encoded)
        logvar = self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        vae_output = self.decoder(z)

        #GAN outputs
        gan_output = self.generator(noise)

        # Ensemble output with softmax weights
        weights = nn.functional.softmax(self.ensemble_weights, dim=0)

        # Create ensemble output without detaching weights
        e_output = weights[0] * vae_output + weights[1] * gan_output

        return {
            'vae_output': vae_output,
            'gan_output': gan_output,
            'ensemble_output': e_output,
            'mu': mu,
            'logvar': logvar,
            'weights': weights,
            "normalized_features": features
        }

    def vae_loss(self, real_data, vae_output, mu, logvar):
      recon = nn.functional.mse_loss(vae_output, real_data, reduction='mean')
      kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
      return recon + 0.1 * kl_div
    
    def compute_disc_loss(self, real_data, fake_data):
      real_preds = self.discriminator(real_data)
      fake_data_pred = self.discriminator(fake_data)
      real_loss = torch.mean(torch.log(real_preds + 1e-8))
      fake_loss = torch.mean(torch.log(1 - fake_data_pred + 1e-8))
      return -(real_loss + fake_loss)

    def compute_gen_loss(self, fake_data):
       fake_preds = self.discriminator(fake_data)
       return -torch.mean(torch.log(fake_preds + 1e-8))

    def train_ensemble(self, all_features, epochs, noise_dim):
        device = self.device
        all_features = torch.tensor(all_features, dtype=torch.float32, device=device)
        all_features = self.normalize_features(all_features)

        progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

        for epoch in progress_bar:
            # Forward pass
            noise = torch.randn(all_features.size(0), noise_dim).to(device)
            outputs = self.forward(all_features, noise)
            
            # Feature extractor update
            self.optimizers['weights'].zero_grad()
            ensemble_reconstr_loss = nn.functional.mse_loss(outputs['vae_output'], all_features)
            weight_reg = torch.abs(outputs['weights'].sum() - 1.0)
            e_loss = ensemble_reconstr_loss + weight_reg
            e_loss.backward(retain_graph=True)
            self.optimizers['weights'].step()
            self.schedulers['weights'].step()

            # VAE update
            self.optimizers['vae'].zero_grad()
            vae_loss_val = self.vae_loss(all_features, outputs['vae_output'], outputs['mu'], outputs['logvar'])
            vae_loss_val.backward(retain_graph=True)
            self.optimizers['vae'].step()
            self.schedulers['vae'].step()

            # Discriminator update
            self.optimizers['disc'].zero_grad()
            d_loss = self.compute_disc_loss(all_features, outputs['gan_output'])
            d_loss.backward(retain_graph=True)
            self.optimizers['disc'].step()
            self.schedulers['disc'].step()

            # Generator update
            self.optimizers['gen'].zero_grad()
            g_loss = self.compute_gen_loss(outputs['gan_output'])
            g_loss.backward()
            self.optimizers['gen'].step()
            self.schedulers['gen'].step()

            # Apply gradient clipping after all updates
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)

            # Update progress bar
            progress_bar.set_postfix({
                'Ensemble Loss': f"{e_loss.item():.4f}",
                'VAE Loss': f"{vae_loss_val.item():.4f}",
                'D Loss': f"{d_loss.item():.4f}",
                'G Loss': f"{g_loss.item():.4f}"
            })

    def compute_anomaly_score(self, features, noise_dim):
        features = self._prepare_tensor(features)
        features = self.normalize_features(features, update_stats=False)
        
        # Get model outputs
        noise = torch.randn(features.size(0), noise_dim, device=self.device)
        outputs = self.forward(features, noise)
        
        # Calculate components with improved normalization
        recon_error = torch.mean((features - outputs['vae_output'])**2, dim=1)
        if not recon_error.std().isnan():
            recon_error = (recon_error - recon_error.mean()) / (recon_error.std() + 1e-8)
        
        kl_div = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - 
                                outputs['logvar'].exp(), dim=1)
        if not kl_div.std().isnan():
            kl_div = (kl_div - kl_div.mean()) / (kl_div.std() + 1e-8)
        
        real_conf = self.discriminator(features).squeeze()
        fake_conf = self.discriminator(outputs['gan_output']).squeeze()
        disc_diff = torch.abs(real_conf - fake_conf)

        if not disc_diff.std().isnan():
            disc_diff = (disc_diff - disc_diff.mean()) / (disc_diff.std() + 1e-8)
        
        # print('RECON_ERROR:', recon_error, 'KL_DIV', kl_div, 'disc_diff', disc_diff )
        # Adjusted weights for better balance
        weights = torch.tensor([0.4, 0.4, 0.2], device=self.device)
        composite_score = (
            weights[0] * recon_error + 
            weights[1] * kl_div + 
            weights[2] * disc_diff
        )
        
        # Apply softer sigmoid scaling
        return torch.sigmoid(composite_score * 0.1).detach().cpu().numpy()
    
    def _prepare_tensor(self, features):
        if not isinstance(features, torch.Tensor):
            if isinstance(features, list):
                features = [f.cpu() if isinstance(f, torch.Tensor) else f for f in features]
                features = torch.tensor(np.array(features), dtype=torch.float32)
            elif isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32)
        return features.to(self.device)