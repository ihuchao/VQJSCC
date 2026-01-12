import torch
from scene import Scene, GaussianModel  # 从 3DGS 项目中导入模型类

# 初始化高斯模型
gaussians = GaussianModel(sh_degree=3)  # sh_degree 需与训练时一致

# 加载压缩模型参数
# checkpoint = torch.load("compressed_gaussians.pth")
# gaussians.restore(checkpoint['model_params'], opt=None)  # 根据实际保存结构调整
checkpoint = torch.load("compressed_gaussians.pth")
gaussians.restore(checkpoint, {})

# 提取关键属性
positions = gaussians.get_xyz.detach().cpu().numpy()        # 位置 [N, 3]
colors = gaussians.get_features.detach().cpu().numpy()      # 颜色/SH系数 [N, K]
opacities = gaussians.get_opacity.detach().cpu().numpy()    # 透明度 [N, 1]
scales = gaussians.get_scaling.detach().cpu().numpy()       # 缩放 [N, 3]
rotations = gaussians.get_rotation.detach().cpu().numpy()   # 旋转四元数 [N, 4]

def save_gaussians_to_ply(positions, colors, opacities, scales, rotations, filename):
    header = f"""ply
format ascii 1.0
comment Generated from compressed_gaussians.pth
element vertex {len(positions)}
property float x
property float y
property float z
property float nx  
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property float scale_x
property float scale_y
property float scale_z
property float rot_w
property float rot_x
property float rot_y
property float rot_z
property float opacity
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(positions)):
            # 转换颜色：SH系数 → RGB（若颜色为SH需额外计算）
            r, g, b = colors[i][:3] * 255  # 简化处理，实际需根据SH计算
            line = f"{positions[i][0]} {positions[i][1]} {positions[i][2]} " \
                   f"0 0 0 " \
                   f"{int(r)} {int(g)} {int(b)} " \
                   f"{scales[i][0]} {scales[i][1]} {scales[i][2]} " \
                   f"{rotations[i][0]} {rotations[i][1]} {rotations[i][2]} {rotations[i][3]} " \
                   f"{opacities[i][0]}\n"
            f.write(line)

# 保存为PLY文件
save_gaussians_to_ply(positions, colors, opacities, scales, rotations, "decompressed_gaussians.ply")