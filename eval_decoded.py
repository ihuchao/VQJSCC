"""
评估解码后的 3DGS 渲染质量
"""
import torch
import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_decoded_gaussians(model_path, iteration):
    """加载解码后的 Gaussians"""
    checkpoint_path = os.path.join(model_path, f"decoded_chkpnt{iteration}.pth")
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_path, f"decoded_params{iteration}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Decoded checkpoint not found: {checkpoint_path}")
        
        decoded_params = torch.load(checkpoint_path)
        gaussians = GaussianModel(sh_degree=decoded_params['max_sh_degree'])
        
        with torch.no_grad():
            gaussians._xyz = decoded_params['xyz'].cuda()
            gaussians._rotation = decoded_params['rotation'].cuda()
            gaussians._scaling = decoded_params['scale'].cuda()
            gaussians._features_dc = decoded_params['f_dc'].cuda()
            gaussians._features_rest = decoded_params['f_rest'].cuda()
            gaussians._opacity = decoded_params['opacity'].cuda()
            gaussians.active_sh_degree = decoded_params['active_sh_degree']
            gaussians.max_sh_degree = decoded_params['max_sh_degree']
    
    else:
        model_params, first_iter = torch.load(checkpoint_path)
        gaussians = GaussianModel(sh_degree=3)
        
        if isinstance(model_params, dict) and '_xyz' in model_params:
            with torch.no_grad():
                gaussians._xyz = model_params['_xyz'].cuda()
                gaussians._rotation = model_params['_rotation'].cuda()
                gaussians._scaling = model_params['_scaling'].cuda()
                gaussians._features_dc = model_params['_features_dc'].cuda()
                gaussians._features_rest = model_params['_features_rest'].cuda()
                gaussians._opacity = model_params['_opacity'].cuda()
                gaussians.active_sh_degree = model_params['active_sh_degree']
                gaussians.max_sh_degree = model_params['max_sh_degree']
        else:
            gaussians.restore(model_params, training_setup=False)
    
    return gaussians


def evaluate_decoded(model_path, iteration, source_path, white_background=False, 
                    use_train_cameras=False):
    """评估解码后的渲染质量"""
    print(f"\n{'='*80}")
    print(f"Loading Decoded Gaussians")
    print(f"{'='*80}")
    print(f"  Model path: {model_path}")
    print(f"  Iteration:  {iteration}")
    
    gaussians = load_decoded_gaussians(model_path, iteration)
    
    print(f"  Loaded {len(gaussians._xyz)} Gaussians")
    print(f"{'='*80}\n")
    
    # ✅ 修复：正确构造参数
    class DatasetParams:
        def __init__(self, source_path, model_path, white_background):
            self.source_path = source_path
            self.model_path = model_path
            self.white_background = white_background
            self.sh_degree = 3
            self.images = "images"
            self.resolution = -1
            self.data_device = "cuda"
            self.eval = False
    
    dataset_args = DatasetParams(source_path, model_path, white_background)
    
    # 加载场景
    print("Loading scene and cameras...")
    scene = Scene(dataset_args, gaussians, load_iteration=None, shuffle=False)
    
    # ✅ 获取相机（测试集或训练集）
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    
    print(f"  Train cameras: {len(train_cameras)}")
    print(f"  Test cameras:  {len(test_cameras)}")
    
    # ✅ 如果没有测试相机，使用训练相机
    if len(test_cameras) == 0:
        print("\n⚠️  Warning: No test cameras found! Using train cameras instead.")
        test_cameras = train_cameras
    
    if use_train_cameras:
        print("\n📌 Using train cameras for evaluation (as requested)")
        test_cameras = train_cameras
    
    if len(test_cameras) == 0:
        raise ValueError("❌ No cameras available for evaluation!")
    
    print(f"  Using {len(test_cameras)} cameras for evaluation\n")
    
    # 准备渲染
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    psnr_values = []
    ssim_values = []
    
    print(f"{'='*80}")
    print(f"Evaluating Decoded Gaussians (Iteration {iteration})")
    print(f"{'='*80}")
    print(f"  Model: {model_path}")
    print(f"  Cameras: {len(test_cameras)}")
    print(f"{'='*80}\n")
    
    # Pipeline 参数
    class PipeParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    
    pipe_args = PipeParams()
    
    for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Rendering")):
        try:
            # 渲染
            with torch.no_grad():
                render_pkg = render(viewpoint, gaussians, pipe_args, background)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # Ground truth
            gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
            
            # 计算指标
            psnr_val = psnr(image, gt_image).mean().item()
            ssim_val = ssim(image, gt_image).mean().item()
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            
        except Exception as e:
            print(f"\n⚠️  Error rendering camera {idx}: {e}")
            continue
    
    # ✅ 检查是否有有效结果
    if len(psnr_values) == 0:
        print("\n❌ No valid renderings produced!")
        return None
    
    # 统计结果
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    print(f"\n{'='*80}")
    print(f"Results:")
    print(f"{'='*80}")
    print(f"  Valid renderings: {len(psnr_values)} / {len(test_cameras)}")
    print(f"  Average PSNR:     {avg_psnr:.2f} dB")
    print(f"  Average SSIM:     {avg_ssim:.4f}")
    print(f"  PSNR range:       [{min(psnr_values):.2f}, {max(psnr_values):.2f}]")
    print(f"  SSIM range:       [{min(ssim_values):.4f}, {max(ssim_values):.4f}]")
    print(f"{'='*80}\n")
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'psnr_values': psnr_values,
        'ssim_values': ssim_values,
        'num_valid': len(psnr_values),
        'num_total': len(test_cameras)
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate decoded Gaussians")
    
    parser.add_argument('--model_path', '-m', required=True, type=str,
                       help='Path to model (parent directory of decoded_chkpnt)')
    parser.add_argument('--iteration', type=int, default=30000,
                       help='Iteration to evaluate')
    parser.add_argument('--source_path', '-s', required=True, type=str,
                       help='Path to source dataset')
    parser.add_argument('--white_background', action='store_true', default=False,
                       help='Use white background')
    parser.add_argument('--use_train', action='store_true', default=False,
                       help='Use train cameras instead of test cameras')
    
    args = parser.parse_args()
    
    result = evaluate_decoded(
        args.model_path, 
        args.iteration, 
        args.source_path,
        args.white_background,
        args.use_train
    )
    
    if result is None:
        print("\n❌ Evaluation failed!")
        sys.exit(1)