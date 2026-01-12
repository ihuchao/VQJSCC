# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class FiLMGenerator(nn.Module):
#     """
#     Generates FiLM parameters (gamma, beta) from viewpoint information.
#     Shared across all Gaussian points in the scene (Broadcasting).
#     """
#     def __init__(self, hidden_dim, viewpoint_dim=16):
#         super().__init__()
#         # Input: Flattened R (9) + t (3) + intrinsics (4) = 16
#         self.net = nn.Sequential(
#             nn.Linear(viewpoint_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU()
#         )
        
#         # Output heads for gamma (scale) and beta (shift)
#         # Initialize gamma to 1 and beta to 0 for identity start
#         self.gamma_head = nn.Linear(64, hidden_dim)
#         self.beta_head = nn.Linear(64, hidden_dim)

#         # Custom initialization for stability
#         with torch.no_grad():
#             self.gamma_head.weight.fill_(0)
#             self.gamma_head.bias.fill_(1)  # Gamma starts at 1
#             self.beta_head.weight.fill_(0)
#             self.beta_head.bias.fill_(0)   # Beta starts at 0

#     def forward(self, R, t, intrinsics):
#         """
#         Args:
#             R: [B, 3, 3] Rotation matrix
#             t: [B, 3] Translation vector
#             intrinsics: [B, 4] (fx, fy, cx, cy)
#         Returns:
#             gamma: [B, 1, hidden_dim] (Unsqueezed for broadcasting)
#             beta:  [B, 1, hidden_dim]
#         """
#         B = R.shape[0]
        
#         # Flatten and concatenate inputs
#         # R: [B, 3, 3] -> [B, 9]
#         # r_flat = R.view(B, -1)
#         r_flat = R.reshape(B, -1)
#         # t: [B, 3]
#         # intrinsics: [B, 4]
        
#         view_vec = torch.cat([r_flat, t, intrinsics], dim=1) # [B, 16]
        
#         feat = self.net(view_vec)
        
#         gamma = self.gamma_head(feat) # [B, H]
#         beta = self.beta_head(feat)   # [B, H]
        
#         # Add dimension for broadcasting over N points: [B, 1, H]
#         return gamma.unsqueeze(1), beta.unsqueeze(1)

# class ChannelEncoder(nn.Module):
#     def __init__(self, num_vq_groups, codebook_sizes, embedding_dims, cont_dim=4, hidden_dim=64, symbol_dim=16):
#         super().__init__()
        
#         self.num_vq_groups = num_vq_groups
#         self.symbol_dim = symbol_dim # Complex symbols per point
        
#         # 3.1 VQ Index Embeddings
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(num_embeddings=codebook_sizes[i], embedding_dim=embedding_dims[i])
#             for i in range(num_vq_groups)
#         ])
        
#         total_emb_dim = sum(embedding_dims)
#         input_dim = total_emb_dim + cont_dim
        
#         # 3.3 Shared MLP (Pre-FiLM)
#         self.pre_mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
        
#         # 5. Channel Symbol Mapping (Post-FiLM)
#         # Maps latent -> Real-valued vector (2 * symbol_dim)
#         self.symbol_mapper = nn.Linear(hidden_dim, 2 * symbol_dim)
        
#     def forward(self, indices, cont_params, gamma, beta):
#         """
#         Args:
#             indices: [B, N, M] LongTensor
#             cont_params: [B, N, D_c] FloatTensor
#             gamma, beta: [B, 1, H] FiLM params
#         Returns:
#             tx_symbols: [B, N, symbol_dim] ComplexTensor
#         """
#         B, N, M = indices.shape
        
#         # 1. Gather Embeddings
#         emb_list = []
#         for m in range(self.num_vq_groups):
#             # indices[..., m]: [B, N]
#             emb = self.embeddings[m](indices[..., m]) # [B, N, E_m]
#             emb_list.append(emb)
            
#         embeds = torch.cat(emb_list, dim=-1) # [B, N, Sum(E_m)]
        
#         # 2. Concat with Continuous Params
#         # cont_params: [B, N, D_c]
#         semantic_vec = torch.cat([embeds, cont_params], dim=-1) # [B, N, Input_Dim]
        
#         # 3. MLP Processing
#         feature = self.pre_mlp(semantic_vec) # [B, N, H]
        
#         # 4. FiLM Modulation (Broadcasting)
#         # feature: [B, N, H], gamma: [B, 1, H], beta: [B, 1, H]
#         modulated_feature = gamma * feature + beta
        
#         # 5. Map to Symbols
#         sym_real_flat = self.symbol_mapper(modulated_feature) # [B, N, 2 * S]
        
#         # Reshape to complex: [B, N, S, 2] -> [B, N, S] complex
#         sym_real_flat = sym_real_flat.view(B, N, self.symbol_dim, 2)
#         tx_symbols = torch.complex(sym_real_flat[..., 0], sym_real_flat[..., 1])
        
#         # Power Normalization (Unit average power per symbol)
#         # E[|x|^2] = 1
#         # Calculate current power
#         power = torch.mean(torch.abs(tx_symbols)**2, dim=-1, keepdim=True) # [B, N, 1]
#         tx_symbols = tx_symbols / (torch.sqrt(power) + 1e-8)
        
#         return tx_symbols

# class WirelessChannel(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, x, snr_db):
#         """
#         Args:
#             x: [B, N, S] Complex symbols
#             snr_db: float or scalar tensor
#         Returns:
#             y: Received symbols [B, N, S]
#             h: Channel coefficients [B, N, S] (Rayleigh)
#         """
#         B, N, S = x.shape
#         device = x.device
        
#         # 1. Rayleigh Fading
#         # h ~ CN(0, 1) -> Real, Imag ~ N(0, 1/sqrt(2))
#         h_real = torch.randn(B, N, S, device=device) / math.sqrt(2)
#         h_imag = torch.randn(B, N, S, device=device) / math.sqrt(2)
#         h = torch.complex(h_real, h_imag)
        
#         # 2. Add Noise
#         # Calculate signal power (after fading)
#         sig_power = torch.mean(torch.abs(h * x)**2)
        
#         snr_linear = 10 ** (snr_db / 10.0)
#         noise_power = sig_power / snr_linear
#         noise_std = torch.sqrt(noise_power / 2) # /2 for real/imag split
        
#         noise = torch.complex(
#             torch.randn_like(h_real) * noise_std,
#             torch.randn_like(h_imag) * noise_std
#         )
        
#         y = h * x + noise
        
#         return y, h

# class ChannelDecoder(nn.Module):
#     def __init__(self, num_vq_groups, codebook_sizes, cont_dim=4, hidden_dim=64, symbol_dim=16):
#         super().__init__()
        
#         self.num_vq_groups = num_vq_groups
#         self.symbol_dim = symbol_dim
        
#         # 7.1 Input Processing (Complex -> Real)
#         input_dim = 2 * symbol_dim
        
#         # 7.2 Decode MLP (Pre-FiLM)
#         self.dec_mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
        
#         # 7.5 Post-FiLM MLP
#         self.post_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
        
#         # 7.6 Output Heads
#         # Index prediction heads (Classification)
#         self.index_heads = nn.ModuleList([
#             nn.Linear(hidden_dim, k) for k in codebook_sizes
#         ])
        
#         # Continuous parameter head (Regression)
#         self.cont_head = nn.Linear(hidden_dim, cont_dim)
        
#     def forward(self, y, h, gamma, beta):
#         """
#         Args:
#             y: Received symbols [B, N, S]
#             h: Channel state [B, N, S]
#             gamma, beta: [B, 1, H] FiLM params (from Viewpoint)
#         """
#         B, N, S = y.shape
        
#         # 1. Perfect Equalization (Zero-Forcing)
#         # Avoid division by zero
#         h_denom = torch.where(torch.abs(h) < 1e-6, torch.ones_like(h)*1e-6, h)
#         x_hat = y / h_denom
        
#         # 2. Complex -> Real
#         x_real = torch.cat([x_hat.real, x_hat.imag], dim=-1) # [B, N, 2*S]
        
#         # 3. Decode MLP
#         latent = self.dec_mlp(x_real) # [B, N, H]
        
#         # 4. FiLM Demodulation
#         # "View-conditioned decoding" - Tells decoder which features are important for this view
#         latent_demod = gamma * latent + beta
        
#         # 5. Post Processing
#         features = self.post_mlp(latent_demod)
        
#         # 6. Predictions
#         pred_indices_logits = []
#         for head in self.index_heads:
#             pred_indices_logits.append(head(features)) # [B, N, K_m]
            
#         pred_cont_params = self.cont_head(features) # [B, N, D_c]
        
#         return pred_indices_logits, pred_cont_params

# class ViewConditionedChannelSystem(nn.Module):
#     def __init__(self, 
#                  num_vq_groups, # number of VQ groups
#                  codebook_sizes, # list of codebook sizes for each VQ group
#                  embedding_dims=[32, 16, 16, 32], # Optimized for Rotation, Scale, SH_DC, SH_Rest
#                  cont_dim=4, # assume 4 continuous parameters to be VQ
#                  hidden_dim=128, 
#                  symbol_dim=16): 
#         super().__init__()
        
#         # Validation: Ensure input lists match num_vq_groups
#         assert len(codebook_sizes) == num_vq_groups, \
#             f"Expected {num_vq_groups} codebook sizes, got {len(codebook_sizes)}"
#         assert len(embedding_dims) == num_vq_groups, \
#             f"Expected {num_vq_groups} embedding dims, got {len(embedding_dims)}"
        
#         self.film_gen = FiLMGenerator(hidden_dim)
        
#         self.encoder = ChannelEncoder(
#             num_vq_groups, codebook_sizes, embedding_dims, 
#             cont_dim, hidden_dim, symbol_dim
#         )
        
#         self.channel = WirelessChannel()
        
#         self.decoder = ChannelDecoder(
#             num_vq_groups, codebook_sizes, 
#             cont_dim, hidden_dim, symbol_dim
#         )
        
#     def forward(self, indices, cont_params, R, t, intrinsics, snr_db):
#         """
#         Full system forward pass.
        
#         Args:
#             indices: [B, N, M] LongTensor
#             cont_params: [B, N, D_c] FloatTensor
#             R: [B, 3, 3]
#             t: [B, 3]
#             intrinsics: [B, 4]
#             snr_db: float
            
#         Returns:
#             output_dict: {
#                 'pred_indices_logits': list of [B, N, K_m],
#                 'pred_cont_params': [B, N, D_c]
#             }
#         """
        
#         # 1. Generate FiLM parameters from Viewpoint
#         # Shared for Encoder and Decoder
#         gamma, beta = self.film_gen(R, t, intrinsics) # [B, 1, H]
        
#         # 2. Channel Encoding
#         # FiLM is used here to prioritize features relevant to current view
#         tx_symbols = self.encoder(indices, cont_params, gamma, beta)
        
#         # 3. Wireless Channel
#         rx_symbols, h = self.channel(tx_symbols, snr_db)
        
#         # 4. Channel Decoding
#         # FiLM is used here again to help decoder interpret features in context of view
#         pred_logits, pred_cont = self.decoder(rx_symbols, h, gamma, beta)
        
#         return {
#             'pred_indices_logits': pred_logits,
#             'pred_cont_params': pred_cont
#         }

#     def compute_loss(self, output_dict, target_indices, target_cont, w_idx=1.0, w_cont=1.0):
#         """
#         Simple reconstruction loss wrapper.
#         """
#         pred_logits = output_dict['pred_indices_logits']
#         pred_cont = output_dict['pred_cont_params']
        
#         # 1. Index Loss (Cross Entropy)
#         loss_idx = 0.0
#         for m, logits in enumerate(pred_logits):
#             # logits: [B, N, K_m] -> [B*N, K_m]
#             # targets: [B, N, M] -> [..., m] -> [B, N] -> [B*N]
            
#             B, N, K = logits.shape
#             loss_idx += F.cross_entropy(
#                 logits.reshape(-1, K), 
#                 target_indices[..., m].reshape(-1)
#             )
            
#         # 2. Continuous Params Loss (MSE)
#         loss_cont = F.mse_loss(pred_cont, target_cont)
        
#         total_loss = w_idx * loss_idx + w_cont * loss_cont
        
#         return total_loss, {'loss_idx': loss_idx.item(), 'loss_cont': loss_cont.item()}






























































# channel_coding.py (Revised Version)
# Changes:
# - Ensured consistent device placement.
# - Minor fixes for stability (e.g., avoid division by zero).
# - No major structural changes, as the core logic is sound, but ensured compatibility with JSCC2.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FiLMGenerator(nn.Module):
    def __init__(self, hidden_dim, viewpoint_dim=16, enable_film=True):  # ✅ 新增参数
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enable_film = enable_film  # ✅ 控制是否启用视角自适应
        
        if not enable_film:
            print("⚠️ FiLM disabled: Using identity transformation (gamma=1, beta=0)")
            return  # 不初始化网络
        
        # 原有网络定义...
        self.net = nn.Sequential(
            nn.Linear(viewpoint_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.gamma_head = nn.Linear(64, hidden_dim)
        self.beta_head = nn.Linear(64, hidden_dim)

        # 初始化为恒等变换
        with torch.no_grad():
            self.gamma_head.weight.fill_(0)
            self.gamma_head.bias.fill_(1)  # gamma = 1
            self.beta_head.weight.fill_(0)
            self.beta_head.bias.fill_(0)   # beta = 0

    def forward(self, R, t, intrinsics):
        """
        Returns:
            gamma: [B, 1, H] - 如果 disable，则全为 1
            beta:  [B, 1, H] - 如果 disable，则全为 0
        """
        B = R.shape[0]
        device = R.device
        
        # ✅ 如果禁用 FiLM，返回恒等变换
        if not self.enable_film:
            gamma = torch.ones(B, 1, self.hidden_dim, device=device)
            beta = torch.zeros(B, 1, self.hidden_dim, device=device)
            return gamma, beta
        
        # 原有逻辑...
        r_flat = R.reshape(B, -1)
        view_vec = torch.cat([r_flat, t, intrinsics], dim=1)
        
        feat = self.net(view_vec)
        gamma = self.gamma_head(feat)
        beta = self.beta_head(feat)
        
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
            emb = self.embeddings[m](indices[..., m]) # [B, N, E_m]
            emb_list.append(emb)
            
        embeds = torch.cat(emb_list, dim=-1) # [B, N, Sum(E_m)]
        
        # 2. Concat with Continuous Params
        semantic_vec = torch.cat([embeds, cont_params], dim=-1) # [B, N, Input_Dim]
        
        # 3. MLP Processing
        feature = self.pre_mlp(semantic_vec) # [B, N, H]
        
        # 4. FiLM Modulation (Broadcasting)
        modulated_feature = gamma * feature + beta
        
        # 5. Map to Symbols
        sym_real_flat = self.symbol_mapper(modulated_feature) # [B, N, 2 * S]
        
        # Reshape to complex: [B, N, S, 2] -> [B, N, S] complex
        sym_real_flat = sym_real_flat.view(B, N, self.symbol_dim, 2)
        tx_symbols = torch.complex(sym_real_flat[..., 0], sym_real_flat[..., 1])
        
        # Power Normalization (Unit average power per symbol)
        # power = torch.mean(torch.abs(tx_symbols)**2, dim=-1, keepdim=True) # [B, N, 1]
        # New: Use dim=[1, 2] to allow FiLM to allocate power across points based on viewpoint
        power = torch.mean(torch.abs(tx_symbols)**2, dim=[1, 2], keepdim=True) # [B, 1, 1]
        tx_symbols = tx_symbols / (torch.sqrt(power) + 1e-8)
        
        return tx_symbols

# class WirelessChannel(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, x, snr_db):
#         """
#         Args:
#             x: [B, N, S] Complex symbols
#             snr_db: float or scalar tensor
#         Returns:
#             y: Received symbols [B, N, S]
#             h: Channel coefficients [B, N, S] (Rayleigh)
#         """
#         B, N, S = x.shape
#         device = x.device
        
#         # 1. Rayleigh Fading
#         h_real = torch.randn(B, N, S, device=device) / math.sqrt(2)
#         h_imag = torch.randn(B, N, S, device=device) / math.sqrt(2)
#         h = torch.complex(h_real, h_imag)
        
#         # 2. Add Noise
#         sig_power = torch.mean(torch.abs(h * x)**2)
        
#         snr_linear = 10 ** (snr_db / 10.0)
#         noise_power = sig_power / snr_linear
#         noise_std = torch.sqrt(noise_power / 2) # /2 for real/imag split
        
#         noise = torch.complex(
#             torch.randn_like(h_real) * noise_std,
#             torch.randn_like(h_imag) * noise_std
#         )
        
#         y = h * x + noise
        
#         return y, h

class WirelessChannel(nn.Module):
    def __init__(self, channel_type='rayleigh'):
        """
        Args:
            channel_type: 'rayleigh' or 'awgn'
                - 'rayleigh': Rayleigh fading + AWGN
                - 'awgn': 仅高斯白噪声（无衰落）
        """
        super().__init__()
        self.channel_type = channel_type
        print(f"🔧 Channel initialized: {channel_type.upper()}")
        
    def forward(self, x, snr_db):
        """
        Args:
            x: [B, N, S] Complex symbols
            snr_db: float or scalar tensor
        Returns:
            y: Received symbols [B, N, S]
            h: Channel coefficients [B, N, S] (Rayleigh) or None (AWGN)
        """
        B, N, S = x.shape
        device = x.device
        
        if self.channel_type == 'rayleigh':
            return self._rayleigh_channel(x, snr_db, device)
        elif self.channel_type == 'awgn':
            return self._awgn_channel(x, snr_db, device)
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")
    
    def _rayleigh_channel(self, x, snr_db, device):
        """Rayleigh Fading + AWGN"""
        B, N, S = x.shape
        
        # 1. Rayleigh Fading
        h_real = torch.randn(B, N, S, device=device) / math.sqrt(2)
        h_imag = torch.randn(B, N, S, device=device) / math.sqrt(2)
        h = torch.complex(h_real, h_imag)
        
        # 2. 信号功率（衰落后）
        sig_power = torch.mean(torch.abs(h * x)**2)
        
        # 3. 噪声功率
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = torch.sqrt(noise_power / 2)  # /2 for real/imag split
        
        # 4. 添加噪声
        noise = torch.complex(
            torch.randn_like(h_real) * noise_std,
            torch.randn_like(h_imag) * noise_std
        )
        
        y = h * x + noise
        
        return y, h
    
    def _awgn_channel(self, x, snr_db, device):
        """纯高斯白噪声信道（无衰落）"""
        B, N, S = x.shape
        
        # 1. 信号功率（假设输入已归一化）
        sig_power = torch.mean(torch.abs(x)**2)
        
        # 2. 噪声功率
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = torch.sqrt(noise_power / 2)
        
        # 3. 添加噪声
        noise_real = torch.randn(B, N, S, device=device) * noise_std
        noise_imag = torch.randn(B, N, S, device=device) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        
        y = x + noise
        
        # ✅ AWGN 信道没有衰落，h = 1（全1）
        h = torch.ones_like(x)
        
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
        h_denom = torch.where(torch.abs(h) < 1e-6, torch.ones_like(h)*1e-6, h)
        x_hat = y / h_denom
        
        # 2. Complex -> Real
        x_real = torch.cat([x_hat.real, x_hat.imag], dim=-1) # [B, N, 2*S]
        
        # 3. Decode MLP
        latent = self.dec_mlp(x_real) # [B, N, H]
        
        # 4. FiLM Demodulation (Disabled: Receiver-side FiLM removed to avoid noise amplification)
        # latent_demod = gamma * latent + beta
        # features = self.post_mlp(latent_demod)
        
        # 5. Post Processing (Directly use latent without FiLM)
        features = self.post_mlp(latent) 
        
        # 6. Predictions
        pred_indices_logits = []
        for head in self.index_heads:
            pred_indices_logits.append(head(features)) # [B, N, K_m]
            
        pred_cont_params = self.cont_head(features) # [B, N, D_c]
        
        return pred_indices_logits, pred_cont_params

class ViewConditionedChannelSystem(nn.Module):
    def __init__(self, 
                 num_vq_groups,
                 codebook_sizes,
                 embedding_dims=[32, 16, 16, 32],
                 cont_dim=4,
                 hidden_dim=128,
                 symbol_dim=16,
                 use_multihead=True,
                 channel_type='rayleigh',
                 enable_film=True):  # ✅ 新增参数
        super().__init__()
        
        assert len(codebook_sizes) == num_vq_groups
        assert len(embedding_dims) == num_vq_groups
        
        self.use_multihead = use_multihead
        self.channel_type = channel_type
        self.enable_film = enable_film  # ✅ 保存配置
        
        # ✅ 传递 enable_film 给 FiLMGenerator
        self.film_gen = FiLMGenerator(hidden_dim, enable_film=enable_film)
        
        self.encoder = ChannelEncoder(
            num_vq_groups, codebook_sizes, embedding_dims, 
            cont_dim, hidden_dim, symbol_dim
        )
        
        self.channel = WirelessChannel(channel_type=channel_type)
        
        if use_multihead:
            print("🔧 Using MultiHead Decoder")
            self.decoder = MultiHeadChannelDecoder(
                num_vq_groups, codebook_sizes, 
                cont_dim, hidden_dim, symbol_dim
            )
        else:
            print("🔧 Using Standard Decoder")
            self.decoder = ChannelDecoder(
                num_vq_groups, codebook_sizes, 
                cont_dim, hidden_dim, symbol_dim
            )
        
        # ✅ 打印配置
        print(f"\n{'='*60}")
        print(f"System Configuration:")
        print(f"  FiLM (View-Adaptive): {'Enabled' if enable_film else 'Disabled (Benchmark)'}")
        print(f"  Channel Type:         {channel_type.upper()}")
        print(f"  Decoder Type:         {'MultiHead' if use_multihead else 'Standard'}")
        print(f"{'='*60}\n")
        
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
        gamma, beta = self.film_gen(R, t, intrinsics) # [B, 1, H]
        
        # 2. Channel Encoding
        tx_symbols = self.encoder(indices, cont_params, gamma, beta)
        
        # 3. Wireless Channel
        rx_symbols, h = self.channel(tx_symbols, snr_db)
        
        # 4. Channel Decoding
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
            B, N, K = logits.shape
            loss_idx += F.cross_entropy(
                logits.reshape(-1, K), 
                target_indices[..., m].reshape(-1)
            )
            
        # 2. Continuous Params Loss (MSE)
        loss_cont = F.mse_loss(pred_cont, target_cont)
        
        total_loss = w_idx * loss_idx + w_cont * loss_cont
        
        return total_loss, {'loss_idx': loss_idx.item(), 'loss_cont': loss_cont.item()}

    def get_param_groups(self):
        """将参数分为索引相关和连续参数相关两组"""
        index_params = []
        cont_params = []
        
        # 索引相关：编码器、解码器、符号嵌入、注意力
        for name, param in self.named_parameters():
            if any(key in name for key in ['vq_encoders', 'index_decoders', 'symbol_embeddings', 'cross_attention']):
                index_params.append(param)
            else:
                cont_params.append(param)
        
        return {
            'index': index_params,
            'cont': cont_params
        }

# 在文件末尾（ViewConditionedChannelSystem 类之前）添加新的解码器类

class MultiHeadChannelDecoder(nn.Module):
    """
    多头解码器：分离索引和连续参数的特征提取路径
    """
    def __init__(self, num_vq_groups, codebook_sizes, cont_dim=4, 
                 hidden_dim=128, symbol_dim=16):
        super().__init__()
        
        self.num_vq_groups = num_vq_groups
        self.symbol_dim = symbol_dim
        
        input_dim = 2 * symbol_dim
        
        # ========== 索引分支（需要判别性）==========
        self.index_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ========== 连续参数分支（需要平滑性）==========
        self.cont_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # 限制输出范围，提高稳定性
        )
        
        # ========== 索引预测头 ==========
        self.index_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, k)
            ) for k in codebook_sizes
        ])
        
        # ========== 连续参数预测头 ==========
        self.cont_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, cont_dim)
        )
        
    def forward(self, y, h, gamma, beta):
        """
        Args:
            y: Received symbols [B, N, S]
            h: Channel state [B, N, S]
            gamma, beta: [B, 1, H] FiLM params
        Returns:
            pred_indices_logits: list of [B, N, K_m]
            pred_cont_params: [B, N, D_c]
        """
        B, N, S = y.shape
        
        # 1. 均衡
        h_denom = torch.where(torch.abs(h) < 1e-6, 
                             torch.ones_like(h) * 1e-6, h)
        x_hat = y / h_denom
        
        # 2. 复数 -> 实数
        x_real = torch.cat([x_hat.real, x_hat.imag], dim=-1)  # [B, N, 2*S]
        
        # ========== 3. 分支处理 ==========
        # 索引分支
        index_features = self.index_branch(x_real)  # [B, N, H]
        
        # 连续参数分支
        cont_features = self.cont_branch(x_real)    # [B, N, H]
        
        # ========== 4. FiLM 调制（已禁用：接收端不再使用视角自适应）==========
        # index_features = gamma * index_features + beta
        # cont_features = gamma * cont_features + beta
        
        # ========== 5. 预测 ==========
        pred_indices_logits = []
        for head in self.index_heads:
            pred_indices_logits.append(head(index_features))  # [B, N, K_m]
            
        pred_cont_params = self.cont_head(cont_features)  # [B, N, D_c]
        
        return pred_indices_logits, pred_cont_params