import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from viewpoint information.
    Shared across all Gaussian points in the scene (Broadcasting).
    """
    def __init__(self, hidden_dim, viewpoint_dim=16):
        super().__init__()
        # Input: Flattened R (9) + t (3) + intrinsics (4) = 16
        self.net = nn.Sequential(
            nn.Linear(viewpoint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Output heads for gamma (scale) and beta (shift)
        # Initialize gamma to 1 and beta to 0 for identity start
        self.gamma_head = nn.Linear(64, hidden_dim)
        self.beta_head = nn.Linear(64, hidden_dim)

        # Custom initialization for stability
        with torch.no_grad():
            self.gamma_head.weight.fill_(0)
            self.gamma_head.bias.fill_(1)  # Gamma starts at 1
            self.beta_head.weight.fill_(0)
            self.beta_head.bias.fill_(0)   # Beta starts at 0

    def forward(self, R, t, intrinsics):
        """
        Args:
            R: [B, 3, 3] Rotation matrix
            t: [B, 3] Translation vector
            intrinsics: [B, 4] (fx, fy, cx, cy)
        Returns:
            gamma: [B, 1, hidden_dim] (Unsqueezed for broadcasting)
            beta:  [B, 1, hidden_dim]
        """
        B = R.shape[0]
        
        # Flatten and concatenate inputs
        # R: [B, 3, 3] -> [B, 9]
        # r_flat = R.view(B, -1)
        r_flat = R.reshape(B, -1)
        # t: [B, 3]
        # intrinsics: [B, 4]
        
        view_vec = torch.cat([r_flat, t, intrinsics], dim=1) # [B, 16]
        
        feat = self.net(view_vec)
        
        gamma = self.gamma_head(feat) # [B, H]
        beta = self.beta_head(feat)   # [B, H]
        
        # Add dimension for broadcasting over N points: [B, 1, H]
        return gamma.unsqueeze(1), beta.unsqueeze(1)

class ChannelEncoder(nn.Module):
    def __init__(self, num_vq_groups, codebook_sizes, embedding_dims, cont_dim=4, hidden_dim=64, symbol_dim=16):
        super().__init__()
        
        self.num_vq_groups = num_vq_groups
        self.symbol_dim = symbol_dim # Complex symbols per point
        
        # 3.1 VQ Index Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=codebook_sizes[i], embedding_dim=embedding_dims[i])
            for i in range(num_vq_groups)
        ])
        
        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + cont_dim
        
        # 3.3 Shared MLP (Pre-FiLM)
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 5. Channel Symbol Mapping (Post-FiLM)
        # Maps latent -> Real-valued vector (2 * symbol_dim)
        self.symbol_mapper = nn.Linear(hidden_dim, 2 * symbol_dim)
        
    def forward(self, indices, cont_params, gamma, beta):
        """
        Args:
            indices: [B, N, M] LongTensor
            cont_params: [B, N, D_c] FloatTensor
            gamma, beta: [B, 1, H] FiLM params
        Returns:
            tx_symbols: [B, N, symbol_dim] ComplexTensor
        """
        B, N, M = indices.shape
        
        # 1. Gather Embeddings
        emb_list = []
        for m in range(self.num_vq_groups):
            # indices[..., m]: [B, N]
            emb = self.embeddings[m](indices[..., m]) # [B, N, E_m]
            emb_list.append(emb)
            
        embeds = torch.cat(emb_list, dim=-1) # [B, N, Sum(E_m)]
        
        # 2. Concat with Continuous Params
        # cont_params: [B, N, D_c]
        semantic_vec = torch.cat([embeds, cont_params], dim=-1) # [B, N, Input_Dim]
        
        # 3. MLP Processing
        feature = self.pre_mlp(semantic_vec) # [B, N, H]
        
        # 4. FiLM Modulation (Broadcasting)
        # feature: [B, N, H], gamma: [B, 1, H], beta: [B, 1, H]
        modulated_feature = gamma * feature + beta
        
        # 5. Map to Symbols
        sym_real_flat = self.symbol_mapper(modulated_feature) # [B, N, 2 * S]
        
        # Reshape to complex: [B, N, S, 2] -> [B, N, S] complex
        sym_real_flat = sym_real_flat.view(B, N, self.symbol_dim, 2)
        tx_symbols = torch.complex(sym_real_flat[..., 0], sym_real_flat[..., 1])
        
        # Power Normalization (Unit average power per symbol)
        # E[|x|^2] = 1
        # Calculate current power
        power = torch.mean(torch.abs(tx_symbols)**2, dim=-1, keepdim=True) # [B, N, 1]
        tx_symbols = tx_symbols / (torch.sqrt(power) + 1e-8)
        
        return tx_symbols

class WirelessChannel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, snr_db):
        """
        Args:
            x: [B, N, S] Complex symbols
            snr_db: float or scalar tensor
        Returns:
            y: Received symbols [B, N, S]
            h: Channel coefficients [B, N, S] (Rayleigh)
        """
        B, N, S = x.shape
        device = x.device
        
        # 1. Rayleigh Fading
        # h ~ CN(0, 1) -> Real, Imag ~ N(0, 1/sqrt(2))
        h_real = torch.randn(B, N, S, device=device) / math.sqrt(2)
        h_imag = torch.randn(B, N, S, device=device) / math.sqrt(2)
        h = torch.complex(h_real, h_imag)
        
        # 2. Add Noise
        # Calculate signal power (after fading)
        sig_power = torch.mean(torch.abs(h * x)**2)
        
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = torch.sqrt(noise_power / 2) # /2 for real/imag split
        
        noise = torch.complex(
            torch.randn_like(h_real) * noise_std,
            torch.randn_like(h_imag) * noise_std
        )
        
        y = h * x + noise
        
        return y, h

class ChannelDecoder(nn.Module):
    def __init__(self, num_vq_groups, codebook_sizes, cont_dim=4, hidden_dim=64, symbol_dim=16):
        super().__init__()
        
        self.num_vq_groups = num_vq_groups
        self.symbol_dim = symbol_dim
        
        # 7.1 Input Processing (Complex -> Real)
        input_dim = 2 * symbol_dim
        
        # 7.2 Decode MLP (Pre-FiLM)
        self.dec_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 7.5 Post-FiLM MLP
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 7.6 Output Heads
        # Index prediction heads (Classification)
        self.index_heads = nn.ModuleList([
            nn.Linear(hidden_dim, k) for k in codebook_sizes
        ])
        
        # Continuous parameter head (Regression)
        self.cont_head = nn.Linear(hidden_dim, cont_dim)
        
    def forward(self, y, h, gamma, beta):
        """
        Args:
            y: Received symbols [B, N, S]
            h: Channel state [B, N, S]
            gamma, beta: [B, 1, H] FiLM params (from Viewpoint)
        """
        B, N, S = y.shape
        
        # 1. Perfect Equalization (Zero-Forcing)
        # Avoid division by zero
        h_denom = torch.where(torch.abs(h) < 1e-6, torch.ones_like(h)*1e-6, h)
        x_hat = y / h_denom
        
        # 2. Complex -> Real
        x_real = torch.cat([x_hat.real, x_hat.imag], dim=-1) # [B, N, 2*S]
        
        # 3. Decode MLP
        latent = self.dec_mlp(x_real) # [B, N, H]
        
        # 4. FiLM Demodulation
        # "View-conditioned decoding" - Tells decoder which features are important for this view
        latent_demod = gamma * latent + beta
        
        # 5. Post Processing
        features = self.post_mlp(latent_demod)
        
        # 6. Predictions
        pred_indices_logits = []
        for head in self.index_heads:
            pred_indices_logits.append(head(features)) # [B, N, K_m]
            
        pred_cont_params = self.cont_head(features) # [B, N, D_c]
        
        return pred_indices_logits, pred_cont_params

class ViewConditionedChannelSystem(nn.Module):
    def __init__(self, 
                 num_vq_groups, # number of VQ groups
                 codebook_sizes, # list of codebook sizes for each VQ group
                 embedding_dims=[32, 16, 16, 32], # Optimized for Rotation, Scale, SH_DC, SH_Rest
                 cont_dim=4, # assume 4 continuous parameters to be VQ
                 hidden_dim=128, 
                 symbol_dim=16): 
        super().__init__()
        
        # Validation: Ensure input lists match num_vq_groups
        assert len(codebook_sizes) == num_vq_groups, \
            f"Expected {num_vq_groups} codebook sizes, got {len(codebook_sizes)}"
        assert len(embedding_dims) == num_vq_groups, \
            f"Expected {num_vq_groups} embedding dims, got {len(embedding_dims)}"
        
        self.film_gen = FiLMGenerator(hidden_dim)
        
        self.encoder = ChannelEncoder(
            num_vq_groups, codebook_sizes, embedding_dims, 
            cont_dim, hidden_dim, symbol_dim
        )
        
        self.channel = WirelessChannel()
        
        self.decoder = ChannelDecoder(
            num_vq_groups, codebook_sizes, 
            cont_dim, hidden_dim, symbol_dim
        )
        
    def forward(self, indices, cont_params, R, t, intrinsics, snr_db):
        """
        Full system forward pass.
        
        Args:
            indices: [B, N, M] LongTensor
            cont_params: [B, N, D_c] FloatTensor
            R: [B, 3, 3]
            t: [B, 3]
            intrinsics: [B, 4]
            snr_db: float
            
        Returns:
            output_dict: {
                'pred_indices_logits': list of [B, N, K_m],
                'pred_cont_params': [B, N, D_c]
            }
        """
        
        # 1. Generate FiLM parameters from Viewpoint
        # Shared for Encoder and Decoder
        gamma, beta = self.film_gen(R, t, intrinsics) # [B, 1, H]
        
        # 2. Channel Encoding
        # FiLM is used here to prioritize features relevant to current view
        tx_symbols = self.encoder(indices, cont_params, gamma, beta)
        
        # 3. Wireless Channel
        rx_symbols, h = self.channel(tx_symbols, snr_db)
        
        # 4. Channel Decoding
        # FiLM is used here again to help decoder interpret features in context of view
        pred_logits, pred_cont = self.decoder(rx_symbols, h, gamma, beta)
        
        return {
            'pred_indices_logits': pred_logits,
            'pred_cont_params': pred_cont
        }

    def compute_loss(self, output_dict, target_indices, target_cont, w_idx=1.0, w_cont=1.0):
        """
        Simple reconstruction loss wrapper.
        """
        pred_logits = output_dict['pred_indices_logits']
        pred_cont = output_dict['pred_cont_params']
        
        # 1. Index Loss (Cross Entropy)
        loss_idx = 0.0
        for m, logits in enumerate(pred_logits):
            # logits: [B, N, K_m] -> [B*N, K_m]
            # targets: [B, N, M] -> [..., m] -> [B, N] -> [B*N]
            
            B, N, K = logits.shape
            loss_idx += F.cross_entropy(
                logits.reshape(-1, K), 
                target_indices[..., m].reshape(-1)
            )
            
        # 2. Continuous Params Loss (MSE)
        loss_cont = F.mse_loss(pred_cont, target_cont)
        
        total_loss = w_idx * loss_idx + w_cont * loss_cont
        
        return total_loss, {'loss_idx': loss_idx.item(), 'loss_cont': loss_cont.item()}

