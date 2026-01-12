"""
Standard DeepJSCC System for 3D Gaussian Splatting Parameters
- Direct end-to-end learning (no VQ quantization)
- Encoder: Gaussian parameters -> Channel symbols
- Decoder: Noisy symbols -> Reconstructed parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepJSCCEncoder(nn.Module):
    """
    Encoder: Maps Gaussian parameters directly to channel symbols
    No VQ quantization - fully differentiable end-to-end
    """
    def __init__(self, 
                 input_dim=59,      # rotation(4) + scale(3) + dc(3) + sh(48) + xyz(3) + opacity(1) = 62
                 hidden_dim=256,
                 symbol_dim=64):    # Number of complex symbols
        super().__init__()
        
        self.symbol_dim = symbol_dim
        
        # Feature extraction network
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Map to channel symbols (real-valued, will convert to complex)
        self.symbol_mapper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * symbol_dim)  # Real and Imaginary parts
        )
        
    def forward(self, gauss_params):
        """
        Args:
            gauss_params: [B, N, D] Concatenated Gaussian parameters
        Returns:
            tx_symbols: [B, N, symbol_dim] Complex tensor
        """
        B, N, D = gauss_params.shape
        
        # 1. Feature extraction
        features = self.encoder_net(gauss_params)  # [B, N, H]
        
        # 2. Map to symbols
        sym_real_flat = self.symbol_mapper(features)  # [B, N, 2*S]
        
        # 3. Reshape to complex
        sym_real_flat = sym_real_flat.view(B, N, self.symbol_dim, 2)
        tx_symbols = torch.complex(sym_real_flat[..., 0], sym_real_flat[..., 1])
        
        # 4. Power normalization (unit average power)
        power = torch.mean(torch.abs(tx_symbols)**2, dim=-1, keepdim=True)  # [B, N, 1]
        tx_symbols = tx_symbols / (torch.sqrt(power) + 1e-8)
        
        return tx_symbols


class DeepJSCCDecoder(nn.Module):
    """
    Decoder: Maps received symbols to reconstructed Gaussian parameters
    """
    def __init__(self,
                 output_dim=59,
                 hidden_dim=256,
                 symbol_dim=64):
        super().__init__()
        
        self.symbol_dim = symbol_dim
        
        # Process received symbols
        input_dim = 2 * symbol_dim  # Real and Imaginary parts
        
        self.decoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Output heads for different parameter types
        # Rotation (4 params - quaternion)
        self.rot_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # Scale (3 params)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # DC color (3 params - RGB)
        self.dc_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # SH coefficients (48 params)
        self.sh_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 48)
        )
        
        # XYZ position (3 params) - normalized
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Opacity (1 param)
        self.opacity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, rx_symbols, h=None):
        """
        Args:
            rx_symbols: [B, N, S] Complex received symbols
            h: [B, N, S] Channel coefficients (optional, for equalization)
        Returns:
            reconstructed_params: Dict of reconstructed parameters
        """
        B, N, S = rx_symbols.shape
        
        # 1. Perfect equalization (if channel state available)
        if h is not None:
            h_denom = torch.where(torch.abs(h) < 1e-6, torch.ones_like(h)*1e-6, h)
            eq_symbols = rx_symbols / h_denom
        else:
            eq_symbols = rx_symbols
        
        # 2. Complex to real
        symbols_real = torch.cat([eq_symbols.real, eq_symbols.imag], dim=-1)  # [B, N, 2*S]
        
        # 3. Decode
        features = self.decoder_net(symbols_real)  # [B, N, H]
        
        # 4. Predict parameters
        pred_rotation = self.rot_head(features)      # [B, N, 4]
        pred_scale = self.scale_head(features)       # [B, N, 3]
        pred_dc = self.dc_head(features)             # [B, N, 3]
        pred_sh = self.sh_head(features)             # [B, N, 48]
        pred_xyz = self.xyz_head(features)           # [B, N, 3]
        pred_opacity = self.opacity_head(features)   # [B, N, 1]
        
        return {
            'rotation': pred_rotation,
            'scale': pred_scale,
            'dc': pred_dc,
            'sh': pred_sh,
            'xyz': pred_xyz,
            'opacity': pred_opacity
        }


class WirelessChannel(nn.Module):
    """
    Wireless channel with Rayleigh fading and AWGN
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x, snr_db):
        """
        Args:
            x: [B, N, S] Complex symbols
            snr_db: float or scalar tensor
        Returns:
            y: Received symbols [B, N, S]
            h: Channel coefficients [B, N, S]
        """
        B, N, S = x.shape
        device = x.device
        
        # 1. Rayleigh Fading
        h_real = torch.randn(B, N, S, device=device) / math.sqrt(2)
        h_imag = torch.randn(B, N, S, device=device) / math.sqrt(2)
        h = torch.complex(h_real, h_imag)
        
        # 2. Add Noise
        sig_power = torch.mean(torch.abs(h * x)**2)
        
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = torch.sqrt(noise_power / 2)
        
        noise = torch.complex(
            torch.randn_like(h_real) * noise_std,
            torch.randn_like(h_imag) * noise_std
        )
        
        y = h * x + noise
        
        return y, h


class DeepJSCCDecoder(nn.Module):
    """
    Decoder: Maps received symbols to reconstructed Gaussian parameters
    """
    def __init__(self,
                 output_dim=59,  # ✅ 改为 59（默认值）
                 hidden_dim=256,
                 symbol_dim=64):
        super().__init__()
        
        self.symbol_dim = symbol_dim
        self.output_dim = output_dim  # ✅ 保存输出维度
        
        # Process received symbols
        input_dim = 2 * symbol_dim  # Real and Imaginary parts
        
        self.decoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Output heads for different parameter types
        # Rotation (4 params - quaternion)
        self.rot_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # Scale (3 params)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # DC color (3 params - RGB)
        self.dc_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # ✅ SH coefficients (动态维度：45 或 48)
        # output_dim = 4(rot) + 3(scale) + 3(dc) + sh_dim + 3(xyz) + 1(opacity)
        # sh_dim = output_dim - 14
        sh_dim = output_dim - 14  # 59 - 14 = 45
        
        self.sh_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, sh_dim)  # ✅ 动态 SH 维度
        )
        
        # XYZ position (3 params) - normalized
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Opacity (1 param)
        self.opacity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, gauss_params, snr_db):
        """
        Args:
            gauss_params: [B, N, D] Concatenated Gaussian parameters
            snr_db: float
        Returns:
            reconstructed_params: Dict of reconstructed parameters
        """
        # 1. Encode
        tx_symbols = self.encoder(gauss_params)
        
        # 2. Channel
        rx_symbols, h = self.channel(tx_symbols, snr_db)
        
        # 3. Decode
        reconstructed = self.decoder(rx_symbols, h)
        
        return reconstructed
    
    def compute_loss(self, pred_params, target_params, weights=None):
        """
        Compute reconstruction loss for all parameters
        
        Args:
            pred_params: Dict of predicted parameters
            target_params: Dict of target parameters
            weights: Dict of loss weights for each parameter type
        """
        if weights is None:
            weights = {
                'rotation': 1.0,
                'scale': 1.0,
                'dc': 1.0,
                'sh': 1.0,
                'xyz': 1.0,
                'opacity': 1.0
            }
        
        losses = {}
        total_loss = 0.0
        
        # MSE loss for each parameter type
        for key in ['rotation', 'scale', 'dc', 'sh', 'xyz', 'opacity']:
            if key in pred_params and key in target_params:
                loss = F.mse_loss(pred_params[key], target_params[key])
                losses[f'loss_{key}'] = loss.item()
                total_loss += weights[key] * loss
        
        return total_loss, losses


# ========== Utility Functions ==========

# 找到 prepare_gaussian_params 函数（约第 150 行），确保它能正确处理：

def prepare_gaussian_params(gaussians, xyz_mean=None, xyz_std=None, 
                           opacity_mean=None, opacity_std=None):
    """
    Prepare and normalize Gaussian parameters for DeepJSCC
    
    Args:
        gaussians: GaussianModel object
        xyz_mean, xyz_std: Normalization stats for xyz (optional)
        opacity_mean, opacity_std: Normalization stats for opacity (optional)
    
    Returns:
        params_dict: Dict containing all parameters
        params_concat: [N, D] Concatenated parameters
    """
    N = gaussians._xyz.shape[0]
    
    # 1. Get all parameters
    rotation = gaussians._rotation.detach()      # [N, 4]
    scale = gaussians._scaling.detach()          # [N, 3]
    
    # ✅ DC features
    features_dc = gaussians._features_dc.detach()
    if features_dc.dim() == 3:
        dc = features_dc.squeeze(1)  # [N, 1, 3] -> [N, 3]
    else:
        dc = features_dc  # 已经是 [N, 3]
    
    # ✅ SH features (rest)
    features_rest = gaussians._features_rest.detach()
    if features_rest.dim() == 3:
        # [N, 15, 3] -> [N, 45] for degree 3
        sh = features_rest.reshape(N, -1)
    else:
        sh = features_rest  # 已经是 [N, 45] 或其他
    
    xyz = gaussians._xyz.detach()                # [N, 3]
    opacity = gaussians._opacity.detach()        # [N, 1]
    
    # 2. Normalize xyz and opacity
    if xyz_mean is not None and xyz_std is not None:
        xyz_normalized = (xyz - xyz_mean) / (xyz_std + 1e-8)
        xyz_normalized = torch.clamp(xyz_normalized, -5.0, 5.0)
    else:
        xyz_normalized = xyz
    
    if opacity_mean is not None and opacity_std is not None:
        opacity_normalized = (opacity - opacity_mean) / (opacity_std + 1e-8)
        opacity_normalized = torch.clamp(opacity_normalized, -5.0, 5.0)
    else:
        opacity_normalized = opacity
    
    # 3. Concatenate all parameters (flexible dimension)
    params_concat = torch.cat([
        rotation,           # 4
        scale,              # 3
        dc,                 # 3
        sh,                 # 45 (for degree 3) or 48 (for degree 4)
        xyz_normalized,     # 3
        opacity_normalized  # 1
    ], dim=-1)  # Total: 59 for degree 3
    
    # 4. Return both dict and concatenated version
    params_dict = {
        'rotation': rotation,
        'scale': scale,
        'dc': dc,
        'sh': sh,
        'xyz': xyz_normalized,
        'opacity': opacity_normalized
    }
    
    return params_dict, params_concat

def apply_reconstructed_params(gaussians, reconstructed_params, 
                               xyz_mean=None, xyz_std=None,
                               opacity_mean=None, opacity_std=None):
    """
    Apply reconstructed parameters back to GaussianModel
    
    Args:
        gaussians: GaussianModel object
        reconstructed_params: Dict of reconstructed parameters
        xyz_mean, xyz_std: Denormalization stats
        opacity_mean, opacity_std: Denormalization stats
    """
    # 1. Denormalize xyz and opacity
    xyz_reconstructed = reconstructed_params['xyz']
    opacity_reconstructed = reconstructed_params['opacity']
    
    if xyz_mean is not None and xyz_std is not None:
        xyz_reconstructed = xyz_reconstructed * (xyz_std + 1e-8) + xyz_mean
    
    if opacity_mean is not None and opacity_std is not None:
        opacity_reconstructed = opacity_reconstructed * (opacity_std + 1e-8) + opacity_mean
    
    # 2. Apply to gaussians (detach gradients)
    gaussians._rotation = reconstructed_params['rotation'].detach().requires_grad_(True)
    gaussians._scaling = reconstructed_params['scale'].detach().requires_grad_(True)
    gaussians._features_dc = reconstructed_params['dc'].unsqueeze(1).detach().requires_grad_(True)
    
    # ✅ 修复：动态 reshape SH features
    sh_flat = reconstructed_params['sh']
    N = sh_flat.shape[0]
    sh_dim = sh_flat.shape[1]
    
    # 根据维度判断如何 reshape
    if sh_dim == 45:  # degree 3: 15*3=45
        sh_reshaped = sh_flat.view(N, 15, 3)
    elif sh_dim == 48:  # degree 4: 16*3=48
        sh_reshaped = sh_flat.view(N, 16, 3)
    else:
        # 其他情况：尝试自动推断
        if sh_dim % 3 == 0:
            num_sh_coefs = sh_dim // 3
            sh_reshaped = sh_flat.view(N, num_sh_coefs, 3)
        else:
            raise ValueError(f"Invalid SH dimension: {sh_dim} (not divisible by 3)")
    
    gaussians._features_rest = sh_reshaped.detach().requires_grad_(True)
    gaussians._xyz = xyz_reconstructed.detach().requires_grad_(True)
    gaussians._opacity = opacity_reconstructed.detach().requires_grad_(True)


def save_original_params(gaussians):
    """Save original Gaussian parameters"""
    return {
        'xyz': gaussians._xyz.clone(),
        'rotation': gaussians._rotation.clone(),
        'scaling': gaussians._scaling.clone(),
        'features_dc': gaussians._features_dc.clone(),
        'features_rest': gaussians._features_rest.clone(),
        'opacity': gaussians._opacity.clone()
    }


def restore_original_params(gaussians, original_params):
    """Restore original Gaussian parameters"""
    gaussians._xyz = original_params['xyz']
    gaussians._rotation = original_params['rotation']
    gaussians._scaling = original_params['scaling']
    gaussians._features_dc = original_params['features_dc']
    gaussians._features_rest = original_params['features_rest']
    gaussians._opacity = original_params['opacity']