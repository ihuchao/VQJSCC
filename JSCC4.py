# Modification of code from Original 3D Gaussian Splat repo

# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr


# Apply k-Means based vector quantization to color and covariance parameters

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import sys
import pdb
from os.path import join
import datetime
import json
import time
from bitarray import bitarray

import numpy as np
import torch
from random import randint

from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.kmeans_quantize import Quantize_kMeans

# import ViewConditionedChannelSystem
from channel_coding import ViewConditionedChannelSystem
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    training.xyz_mean = None
    training.xyz_std = None
    training.opacity_mean = None
    training.opacity_std = None    

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # ✅ 添加测试日志（确保 TensorBoard 正常工作）
    if tb_writer:
        print(f"✅ TensorBoard initialized: writing to {dataset.model_path}")
        tb_writer.add_scalar('test/init', 1.0, 0)
        tb_writer.flush()  # 立即刷新到磁盘
    else:
        print(f"❌ TensorBoard not initialized!")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    num_gaussians_per_iter = []

    # k-Means quantization
    quantized_params = args.quant_params
    n_cls = args.kmeans_ncls
    n_cls_sh = args.kmeans_ncls_sh
    n_cls_dc = args.kmeans_ncls_dc
    n_it = args.kmeans_iters
    kmeans_st_iter = args.kmeans_st_iter
    freq_cls_assn = args.kmeans_freq

    kmeans_w_iter = args.kmeans_w_iter

    if 'pos' in quantized_params:
        kmeans_pos_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'dc' in quantized_params:
        kmeans_dc_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'sh' in quantized_params:
        kmeans_sh_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)
    if 'scale' in quantized_params:
        kmeans_sc_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'rot' in quantized_params:
        kmeans_rot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'scale_rot' in quantized_params:
        kmeans_scrot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'sh_dc' in quantized_params:
        kmeans_shdc_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)

    # 初始化信道编解码系统
    if args.mock_channel:
        # 根据你的量化参数配置
        # 假设量化了4种参数：rotation, scale, dc, sh
        # channel_system = ViewConditionedChannelSystem(
        #     num_vq_groups=4,  # rotation, scale, dc, sh
        #     codebook_sizes=[args.kmeans_ncls, args.kmeans_ncls, args.kmeans_ncls_dc, args.kmeans_ncls_sh],  # 每种参数的码本大小
        #     embedding_dims=[64, 32, 64, 32],  # 嵌入维度
        #     cont_dim=4,  # xyz + opacity (连续参数)
        #     hidden_dim=32,
        #     symbol_dim=64
        # ).cuda()
        channel_system = ViewConditionedChannelSystem(
            num_vq_groups=4,
            codebook_sizes=[args.kmeans_ncls] * 4,
            embedding_dims=[32, 16, 16, 32],
            cont_dim=4,
            hidden_dim=128,
            symbol_dim=16,
            use_multihead=True,  # ✅ 启用多头解码器
            channel_type=args.channel_type,  # ✅ 传入信道类型
            enable_film=not args.disable_film  # ✅ 是否启用 FiLM
        ).cuda()        
        
        # 设置为训练模式
        channel_system.train()
        
        # # 添加到优化器
        # channel_optimizer = torch.optim.Adam(channel_system.parameters(), lr=1e-2) 
        # 分组参数并设置不同学习率
        param_groups = channel_system.get_param_groups()
        channel_optimizer = torch.optim.Adam([
            {'params': param_groups['index'], 'lr': args.lr_index},
            {'params': param_groups['cont'], 'lr': args.lr_cont}
        ])
        
        xyz_mean = None
        xyz_std = None
        opacity_mean = None
        opacity_std = None

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()

        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Set different learning rates for different parameters
        # if iteration <= 21000:
        #     channel_optimizer = torch.optim.Adam(channel_system.parameters(), lr=1e-2)    
        # if iteration > 21000 and iteration <= 22000:
        #     channel_optimizer = torch.optim.Adam(channel_system.parameters(), lr=1e-3)
        # if iteration > 22000 and iteration <= 25000:
        #     channel_optimizer = torch.optim.Adam(channel_system.parameters(), lr=1e-4)
        # if iteration > 25000 and iteration <= 28000:
        #     channel_optimizer = torch.optim.Adam(channel_system.parameters(), lr=1e-2)
        # if iteration > 28000 and iteration <= 30000:
        #     channel_optimizer = torch.optim.Adam(channel_system.parameters(), lr=1e-4)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration > 3100:
            freq_cls_assn = 100
            if iteration > (opt.iterations - 5000):
                freq_cls_assn = 5000

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Show the number of Gaussians in the current iteration
        with open("gaussian_count_log.csv", "a") as f:
            f.write(f"{iteration},{gaussians._xyz.shape[0]}\n")
        
        if iteration <= kmeans_w_iter:
            weights = None
        if iteration == kmeans_w_iter+1:
            weights = kmeans_dc_q.get_weights_near(gaussians)

        # quantize params
        if iteration > kmeans_st_iter:
            if iteration % freq_cls_assn == 1:
                assign = True
            else:
                assign = False
            if 'pos' in quantized_params:
                kmeans_pos_q.forward_pos(gaussians, assign=assign)
            if 'dc' in quantized_params:
                kmeans_dc_q.forward_dc(gaussians, assign=assign)
                kmeans_dc_q.cweights = True if iteration > kmeans_w_iter else False
                kmeans_dc_q.weights = weights if iteration > kmeans_w_iter else None
            if 'sh' in quantized_params:
                kmeans_sh_q.forward_frest(gaussians, assign=assign)
                kmeans_sh_q.cweights = True if iteration > kmeans_w_iter else False
                kmeans_sh_q.weights = kmeans_dc_q.weights if iteration > kmeans_w_iter else None
            if 'scale' in quantized_params:
                kmeans_sc_q.forward_scale(gaussians, assign=assign)
                kmeans_sc_q.cweights = True if iteration > kmeans_w_iter else False
                kmeans_sc_q.weights = kmeans_dc_q.weights if iteration > kmeans_w_iter else None
            if 'rot' in quantized_params:
                kmeans_rot_q.forward_rot(gaussians, assign=assign)
                kmeans_rot_q.cweights = True if iteration > kmeans_w_iter else False
                kmeans_rot_q.weights = kmeans_dc_q.weights if iteration > kmeans_w_iter else None
            if 'scale_rot' in quantized_params:
                kmeans_scrot_q.forward_scale_rot(gaussians, assign=assign)
            if 'sh_dc' in quantized_params:
                kmeans_shdc_q.forward_dcfrest(gaussians, assign=assign)

        # --- 方案B：分阶段训练（交替优化）---
        used_channel = False
        loss_for_log = 0.0  # 初始化

        Ll1 = torch.tensor(0.0, device="cuda")  # 在循环开始就初始化
        loss = torch.tensor(0.0, device="cuda")  # 在循环开始就初始化
        
        if args.mock_channel and iteration > (args.kmeans_st_iter + 0):
            #动态调整学习率 18dB
            if args.mock_channel and iteration > args.kmeans_st_iter:
                if iteration <= 21000:
                    lr_index, lr_cont = 1e-2, 1e-2
                elif iteration <= 22000:
                    lr_index, lr_cont = 1e-3, 1e-2
                elif iteration <= 25000:
                    lr_index, lr_cont = 1e-4, 1e-4
                elif iteration <= 28000:
                    lr_index, lr_cont = 1e-2, 1e-4  # index 快，cont 慢
                else:
                    lr_index, lr_cont = 1e-2, 1e-4

            # 13dB
            # if args.mock_channel and iteration > args.kmeans_st_iter:
            #     if iteration <= 21000:
            #         lr_index, lr_cont = 1e-2, 1e-2
            #     elif iteration <= 22000:
            #         lr_index, lr_cont = 1e-3, 1e-2
            #     elif iteration <= 25000:
            #         lr_index, lr_cont = 1e-4, 1e-5
            #     elif iteration <= 28000:
            #         lr_index, lr_cont = 1e-2, 1e-4  # index 快，cont 慢
            #     else:
            #         lr_index, lr_cont = 1e-5, 1e-2
                
                # 更新优化器学习率
                for param_group in channel_optimizer.param_groups:
                    if param_group is channel_optimizer.param_groups[0]:
                        param_group['lr'] = lr_index
                    else:
                        param_group['lr'] = lr_cont


            with torch.no_grad():
                # 计算全局统计信息
                xyz_mean = gaussians._xyz.mean(dim=0, keepdim=True)  # [1, 3]
                xyz_std = gaussians._xyz.std(dim=0, keepdim=True).clamp(min=1e-6)  # [1, 3]
                opacity_mean = gaussians._opacity.mean()  # scalar
                opacity_std = gaussians._opacity.std().clamp(min=1e-6)  # scalar
            # 阶段1：每10次迭代训练一次信道系统
            if iteration % 2 == 0:
                # 1. 冻结 gaussians 参数
                gaussians._xyz.requires_grad = False
                gaussians._features_dc.requires_grad = False
                gaussians._features_rest.requires_grad = False
                gaussians._opacity.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False
                
                # 2. 准备输入数据
                indices = prepare_indices(gaussians, kmeans_rot_q, kmeans_sc_q, 
                                        kmeans_dc_q, kmeans_sh_q)
                # 归一化连续参数
                cont_params = torch.cat([
                    (gaussians._xyz - xyz_mean) / xyz_std,
                    (gaussians._opacity - opacity_mean) / opacity_std
                ], dim=-1)
                
                indices = indices.unsqueeze(0)
                cont_params = cont_params.unsqueeze(0)
                
                # 3. 获取相机参数
                if isinstance(viewpoint_cam.R, torch.Tensor):
                    R = viewpoint_cam.R.unsqueeze(0)
                else:
                    R = torch.tensor(viewpoint_cam.R, dtype=torch.float32, device="cuda").unsqueeze(0)
                
                if isinstance(viewpoint_cam.T, torch.Tensor):
                    t = viewpoint_cam.T.unsqueeze(0)
                else:
                    t = torch.tensor(viewpoint_cam.T, dtype=torch.float32, device="cuda").unsqueeze(0)
                
                import math
                fx = viewpoint_cam.image_width / (2.0 * math.tan(viewpoint_cam.FoVx / 2.0))
                fy = viewpoint_cam.image_height / (2.0 * math.tan(viewpoint_cam.FoVy / 2.0))
                cx = viewpoint_cam.image_width / 2.0
                cy = viewpoint_cam.image_height / 2.0
                intrinsics = torch.tensor([fx, fy, cx, cy], 
                                        dtype=torch.float32, 
                                        device="cuda").unsqueeze(0)
                
                # 4. 保存原始参数
                original_params = save_original_params(gaussians)
                
                # 5. 前向传播（保持梯度）
                output = channel_system(
                    indices, 
                    cont_params, 
                    R, t, intrinsics, 
                    snr_db=args.snr_db
                )
                
                # 6. 解码（不使用 no_grad）
                corrupted_params = decode_output_to_gaussians_soft(
                    output, gaussians, kmeans_rot_q, kmeans_sc_q, 
                    kmeans_dc_q, kmeans_sh_q,
                    xyz_mean, xyz_std, opacity_mean, opacity_std,  # ← 传入归一化参数
                    temperature=0.5
                )
                
                # 7. 应用损坏参数
                apply_corrupted_params(gaussians, corrupted_params)
                
                # 8. 渲染损坏图像
                gt_image = viewpoint_cam.original_image.cuda(non_blocking=True)
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                corrupted_image = render_pkg["render"]  # ✅ 明确命名为 corrupted_image
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                
                # 9. 计算渲染损失（使用损坏图像）
                Ll1 = l1_loss(corrupted_image, gt_image)
                render_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(corrupted_image, gt_image))
                
                # # 10. 计算信道重建损失
                # channel_loss, loss_dict = channel_system.compute_loss(
                #     output, 
                #     indices,
                #     cont_params,
                #     w_idx=1.0,
                #     w_cont=1.0
                # )

                # 在第 340 行后添加 效果更好，重连续参数
                # ✅ 阶段性训练
                if iteration <= 25000:
                    # 阶段1：联合优化
                    w_idx, w_cont = 0.0, 1.0
                elif iteration < 28000:
                    # 阶段2：侧重连续参数
                    w_idx, w_cont = 0.01, 100.0
                else:
                    # 阶段3：只优化连续参数
                    w_idx, w_cont = 0.01, 100.0  # ← 完全忽略索引！

                # if iteration <= 25000:
                #     # 阶段1：联合优化
                #     w_idx, w_cont = 0.0, 2.0
                # elif iteration < 28000:
                #     # 阶段2：侧重连续参数
                #     w_idx, w_cont = 0.01, 2.0
                # else:
                #     # 阶段3：只优化连续参数
                #     w_idx, w_cont = 0.01, 4.0  # ← 完全忽略索引！

               # 在第 340 行后添加
                # ✅ 阶段性训练
                # if iteration <= 25000:
                #     # 阶段1：联合优化
                #     w_idx, w_cont = 0.1, 0.0
                # elif iteration < 28000:
                #     # 阶段2：侧重连续参数
                #     w_idx, w_cont = 0.1, 0.01
                # else:
                #     # 阶段3：只优化连续参数
                #     w_idx, w_cont = 0.0, 100.0  # ← 完全忽略索引！
                    
                #     # 冻结索引相关参数
                #     for name, param in channel_system.named_parameters():
                #         if any(key in name for key in ['vq_encoders', 'index_decoders', 'symbol_embeddings']):
                #             param.requires_grad = False

                # if iteration < args.kmeans_st_iter + 10000:
                #     w_idx = 0.1
                #     w_cont = 5.0
                #     stage = "Stage 1: Continuous Params Focus"   
                # # 阶段2：平衡优化（10k-20k 迭代）
                # elif iteration < args.kmeans_st_iter + 20000:
                #     w_idx = 0.5
                #     w_cont = 5.0
                #     stage = "Stage 2: Balanced Training"
                # # 阶段3：强化索引（20k+ 迭代）
                # else:
                #     # 使用余弦退火逐渐增大索引权重
                #     progress = (iteration - args.kmeans_st_iter - 20000) / 10000
                #     progress = min(progress, 1.0)
                #     # w_idx = 0.5 + 0.5 * (1 - math.cos(progress * math.pi)) / 2
                #     # w_cont = 0.5 - 0.2 * (1 - math.cos(progress * math.pi)) / 2
                #     w_cont = 0.5 + 0.5 * (1 - math.cos(progress * math.pi)) / 2
                #     w_idx = 0.5 - 0.2 * (1 - math.cos(progress * math.pi)) / 2
                #     stage = "Stage 3: Index Enhancement"
                # # 每 1000 次迭代打印一次当前阶段
                # if iteration % 1000 == 0:
                #     print(f"\n[{stage}] w_idx={w_idx:.3f}, w_cont={w_cont:.3f}")              

                channel_loss, loss_dict = channel_system.compute_loss(
                    output, indices, cont_params,
                    w_idx=w_idx, w_cont=w_cont
                )
                
                # 11. 联合损失
                total_loss = render_loss + 10 * channel_loss
                loss = total_loss
                loss_for_log = total_loss.item()
                
                # 12. 反向传播（只更新信道系统）
                channel_optimizer.zero_grad()
                total_loss.backward()
                channel_optimizer.step()
                channel_optimizer.zero_grad(set_to_none=True)
                
                # 13. 密集化统计
                with torch.no_grad():
                    if iteration < opt.densify_until_iter:
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter], 
                            radii[visibility_filter]
                        )
                        gaussians.add_densification_stats(
                            viewspace_point_tensor.detach(), 
                            visibility_filter
                        )
                
                # ✅ 14. 在恢复参数之前计算损坏图像的 PSNR
                with torch.no_grad():
                    # 此时 gaussians 还是损坏参数状态
                    # corrupted_image 是损坏参数渲染的结果
                    corrupted_psnr = psnr(corrupted_image, gt_image).mean().item()
                    corrupted_ssim = ssim(corrupted_image, gt_image).mean().item()
                
                # ✅ 15. 恢复原始参数
                restore_original_params(gaussians, original_params)
                
                # ✅ 16. 渲染干净图像
                with torch.no_grad():
                    # 此时 gaussians 是原始参数状态
                    clean_render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                    clean_image = clean_render_pkg["render"]
                    clean_psnr = psnr(clean_image, gt_image).mean().item()
                    clean_ssim = ssim(clean_image, gt_image).mean().item()
                    
                    # 立即清理
                    del clean_render_pkg, clean_image
                
                # 17. 解冻 gaussians 参数
                gaussians._xyz.requires_grad = True
                gaussians._features_dc.requires_grad = True
                gaussians._features_rest.requires_grad = True
                gaussians._opacity.requires_grad = True
                gaussians._scaling.requires_grad = True
                gaussians._rotation.requires_grad = True

                # 18. 同步 CUDA
                torch.cuda.synchronize()

                if iteration % 500 == 0 and 'accuracies' in locals():
                        print(f"  Index Accuracies:")
                        param_names = ['rot', 'scale', 'dc', 'sh']
                        for name, acc in zip(param_names, accuracies):
                            print(f"    - {name:6s}: {acc*100:.2f}%")
                        print(f"    - Average:  {avg_acc*100:.2f}%")
                
                # 19. 日志
                if iteration % 10 == 0:
                    print(f"\n[Channel Training] Iter {iteration}:")
                    print(f"  Render Loss:  {render_loss.item():.6f}")
                    print(f"  Channel Loss: {channel_loss.item():.6f}")
                    print(f"    - Index Loss:  {loss_dict['loss_idx']:.6f}")
                    print(f"    - Cont Loss:   {loss_dict['loss_cont']:.6f}")
                    print(f"  Total Loss:   {loss_for_log:.6f}")
                    print(f"  Clean PSNR:      {clean_psnr:.2f} dB")
                    print(f"  Corrupted PSNR:  {corrupted_psnr:.2f} dB")
                    print(f"  PSNR Drop:       {clean_psnr - corrupted_psnr:.2f} dB")
                    print(f"  Clean SSIM:      {clean_ssim:.4f}")
                    print(f"  Corrupted SSIM:  {corrupted_ssim:.4f}")

                    # 记录到 TensorBoard
                    if tb_writer:
                        tb_writer.add_scalar('channel/clean_psnr', clean_psnr, iteration)
                        tb_writer.add_scalar('channel/corrupted_psnr', corrupted_psnr, iteration)
                        tb_writer.add_scalar('channel/psnr_drop', clean_psnr - corrupted_psnr, iteration)
                        tb_writer.add_scalar('channel/clean_ssim', clean_ssim, iteration)
                        tb_writer.add_scalar('channel/corrupted_ssim', corrupted_ssim, iteration)
                        tb_writer.add_scalar('channel/render_loss', render_loss.item(), iteration)
                        tb_writer.add_scalar('channel/channel_loss', channel_loss.item(), iteration)
                        tb_writer.add_scalar('channel/index_loss', loss_dict['loss_idx'], iteration)
                        tb_writer.add_scalar('channel/cont_loss', loss_dict['loss_cont'], iteration)

                        # 权重
                        tb_writer.add_scalar('channel/w_idx', w_idx, iteration)
                        tb_writer.add_scalar('channel/w_cont', w_cont, iteration)
                
                # 20. 清理显存
                del output, corrupted_params, indices, cont_params, R, t, intrinsics
                del corrupted_image, gt_image, render_pkg, viewspace_point_tensor, visibility_filter, radii
                del original_params, render_loss, channel_loss, total_loss
                torch.cuda.empty_cache()
                
                used_channel = True
        
        # 阶段2：正常训练 gaussians
        if not used_channel:
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            gt_image = viewpoint_cam.original_image.cuda(non_blocking=True)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            Ll1 = l1_loss(image, gt_image)

            # 正则化逻辑
            if args.opacity_reg:
                if iteration > args.max_prune_iter or iteration < 15000:
                    lambda_reg = 0.
                else:
                    lambda_reg = args.lambda_reg
                L_reg_op = gaussians.get_opacity.sum()
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + (lambda_reg * L_reg_op)
                if args.scale_reg:
                    if iteration > args.max_prune_iter or iteration < 15000:
                        lambda_scale_reg = 0.
                    else:
                        lambda_scale_reg = args.lambda_scale_reg
                    L_reg_scale = gaussians.get_scaling.sum()
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + (lambda_reg * L_reg_op) + (lambda_scale_reg * L_reg_scale)
                else:
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + (lambda_reg * L_reg_op)
            else:
                if args.scale_reg:
                    if iteration > args.max_prune_iter or iteration < 15000:
                        lambda_scale_reg = 0.
                    else:
                        lambda_scale_reg = args.lambda_scale_reg
                    L_reg_scale = gaussians.get_scaling.sum()
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + (lambda_scale_reg * L_reg_scale)
                else:
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            loss_for_log = loss.item()
            loss.backward()
        
        # --- End 分阶段训练 ---
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_for_log + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ✅ 在调用 elapsed_time 前同步
            torch.cuda.synchronize()
            
            # ✅ 安全获取 elapsed_time
            try:
                elapsed_time = iter_start.elapsed_time(iter_end)
            except RuntimeError as e:
                print(f"[WARNING] CUDA timing error at iter {iteration}: {e}")
                elapsed_time = 0.0

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # if (iteration in saving_iterations):
            #     print(args.model_path)
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     all_attributes = {'xyz': 'xyz', 'dc': 'f_dc', 'sh': 'f_rest', 'opacities': 'opacities',
            #                       'scale': 'scale', 'rot': 'rotation'}
            #     save_attributes = [val for (key, val) in all_attributes.items() if key not in quantized_params]
            #     if iteration > kmeans_st_iter:
            #         scene.save(iteration, save_q=quantized_params, save_attributes=save_attributes)
                    
            #         kmeans_dict = {'rot': kmeans_rot_q, 'scale': kmeans_sc_q, 'sh': kmeans_sh_q, 'dc': kmeans_dc_q}
            #         kmeans_list = []
            #         for param in quantized_params:
            #             kmeans_list.append(kmeans_dict[param])
            #         out_dir = join(scene.model_path, 'point_cloud/iteration_%d' % iteration)
            #         save_kmeans(kmeans_list, quantized_params, out_dir)
            #     else:
            #         scene.save(iteration, save_q=[])
            # 在第 273 行左右，完整替换为：

            if (iteration in saving_iterations):
                print(args.model_path)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                all_attributes = {'xyz': 'xyz', 'dc': 'f_dc', 'sh': 'f_rest', 'opacities': 'opacities',
                                'scale': 'scale', 'rot': 'rotation'}
                save_attributes = [val for (key, val) in all_attributes.items() if key not in quantized_params]
                
                if iteration > kmeans_st_iter:
                    # ========== 保存原始量化数据 ==========
                    scene.save(iteration, save_q=quantized_params, save_attributes=save_attributes)
                    
                    kmeans_dict = {'rot': kmeans_rot_q, 'scale': kmeans_sc_q, 'sh': kmeans_sh_q, 'dc': kmeans_dc_q}
                    kmeans_list = []
                    for param in quantized_params:
                        kmeans_list.append(kmeans_dict[param])
                    out_dir = join(scene.model_path, 'point_cloud/iteration_%d' % iteration)
                    save_kmeans(kmeans_list, quantized_params, out_dir)
                    
                    # ========== ✅ 新增：保存解码后的数据 ==========
                    if args.mock_channel:
                        print("\n" + "="*80)
                        print("[Saving Decoded Data from Channel]")
                        print("="*80)
                        
                        # 1. 通过信道获取解码参数
                        with torch.no_grad():
                            # 准备输入：索引
                            indices_rot = kmeans_rot_q.cls_ids.unsqueeze(0)  # [1, N]
                            indices_scale = kmeans_sc_q.cls_ids.unsqueeze(0)
                            indices_dc = kmeans_dc_q.cls_ids.unsqueeze(0)
                            indices_sh = kmeans_sh_q.cls_ids.unsqueeze(0)
                            indices = torch.stack([indices_rot, indices_scale, indices_dc, indices_sh], dim=-1).squeeze(0)  # [N, 4]
                            indices = indices.unsqueeze(0)  # [1, N, 4]
                            
                            # 归一化连续参数
                            if not hasattr(training, 'xyz_mean') or training.xyz_mean is None:
                                training.xyz_mean = gaussians._xyz.mean(dim=0, keepdim=True)
                                training.xyz_std = gaussians._xyz.std(dim=0, keepdim=True).clamp(min=1e-6)
                                training.opacity_mean = gaussians._opacity.mean()
                                training.opacity_std = gaussians._opacity.std().clamp(min=1e-6)
                            
                            xyz_normalized = (gaussians._xyz - training.xyz_mean) / training.xyz_std
                            opacity_normalized = (gaussians._opacity - training.opacity_mean) / training.opacity_std
                            cont_params = torch.cat([xyz_normalized, opacity_normalized], dim=-1)
                            cont_params = cont_params.unsqueeze(0)  # [1, N, 4]
                            
                            # 获取相机参数（使用第一个训练相机）
                            test_cam = scene.getTrainCameras()[0]
                            
                            if isinstance(test_cam.R, torch.Tensor):
                                R = test_cam.R.unsqueeze(0)
                            else:
                                R = torch.tensor(test_cam.R, dtype=torch.float32, device="cuda").unsqueeze(0)
                            
                            if isinstance(test_cam.T, torch.Tensor):
                                t = test_cam.T.unsqueeze(0)
                            else:
                                t = torch.tensor(test_cam.T, dtype=torch.float32, device="cuda").unsqueeze(0)
                            
                            import math
                            fx = test_cam.image_width / (2.0 * math.tan(test_cam.FoVx / 2.0))
                            fy = test_cam.image_height / (2.0 * math.tan(test_cam.FoVy / 2.0))
                            cx = test_cam.image_width / 2.0
                            cy = test_cam.image_height / 2.0
                            intrinsics = torch.tensor([fx, fy, cx, cy], 
                                                    dtype=torch.float32, 
                                                    device="cuda").unsqueeze(0)
                            
                            # 通过信道系统
                            output = channel_system(
                                indices, 
                                cont_params, 
                                R, t, intrinsics, 
                                snr_db=args.snr_db
                            )
                            
                            # 解码（硬解码，用于保存）
                            decoded_params = decode_output_to_gaussians_hard(
                                output, gaussians, kmeans_rot_q, kmeans_sc_q, 
                                kmeans_dc_q, kmeans_sh_q,
                                training.xyz_mean, training.xyz_std, 
                                training.opacity_mean, training.opacity_std
                            )
                        
                        # 2. 保存解码后的数据
                        save_decoded_gaussians(
                            gaussians=gaussians,
                            corrupted_params=decoded_params,
                            kmeans_list=kmeans_list,
                            quantized_params=quantized_params,
                            output_dir=scene.model_path,
                            iteration=iteration,
                            save_ply=True
                        )
                        
                        # 3. 保存完整 checkpoint
                        save_decoded_checkpoint(
                            gaussians=gaussians,
                            corrupted_params=decoded_params,
                            iteration=iteration,
                            model_path=scene.model_path
                        )
                        
                        print("="*80 + "\n")
                else:
                    scene.save(iteration, save_q=[])

            # Densification
            if iteration < opt.densify_until_iter:
                # 只在正常训练分支才处理
                if not used_channel:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Prune
            if args.opacity_reg and iteration > 15000:
                if iteration <= args.max_prune_iter and iteration % 1000 == 0:
                    print('Num Gaussians: ', gaussians._xyz.shape[0])
                    size_threshold = None
                    gaussians.prune(0.005, scene.cameras_extent, size_threshold)
                    print('Num Gaussians after prune: ', gaussians._xyz.shape[0])

            # Optimizer step
            if iteration < opt.iterations:
                # 只在正常训练分支才更新 gaussians
                if not used_channel:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        num_gaussians_per_iter.append(gaussians.get_xyz.shape[0])

    print("Number of Gaussians at the end: ", gaussians._xyz.shape[0])
    np.save(f'{scene.model_path}/num_g_per_iters.npy', np.array(num_gaussians_per_iter))



def dec2binary(x, n_bits=None):
    """Convert decimal integer x to binary.

    Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    if n_bits is None:
        n_bits = torch.ceil(torch.log2(x)).type(torch.int64)
    mask = 2**torch.arange(n_bits-1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)


def save_kmeans(kmeans_list, quantized_params, out_dir):
    """Save the codebook and indices of KMeans.

    """
    # Convert to bitarray object to save compressed version
    # saving as npy or pth will use 8bits per digit (or boolean) for the indices
    # Convert to binary, concat the indices for all params and save.
    bitarray_all = bitarray([])
    for kmeans in kmeans_list:
        n_bits = int(np.ceil(np.log2(len(kmeans.cls_ids))))
        assignments = dec2binary(kmeans.cls_ids, n_bits)
        bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
        bitarray_all.extend(bitarr)
    with open(join(out_dir, 'kmeans_inds.bin'), 'wb') as file:
        bitarray_all.tofile(file)

    # Save details needed for loading
    args_dict = {}
    args_dict['params'] = quantized_params
    args_dict['n_bits'] = n_bits
    args_dict['total_len'] = len(bitarray_all)
    np.save(join(out_dir, 'kmeans_args.npy'), args_dict)
    centers_dict = {param: kmeans.centers for (kmeans, param) in zip(kmeans_list, quantized_params)}

    # Save codebook
    torch.save(centers_dict, join(out_dir, 'kmeans_centers.pth'))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # psnr_test = -1.
    psnr_out = {'train': -1., 'test': -1}
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                psnr_out[config['name']] = psnr_test
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return psnr_out

def prepare_indices(gaussians, kmeans_rot_q, kmeans_sc_q, kmeans_dc_q, kmeans_sh_q):
    """提取当前的量化索引"""
    indices = torch.stack([
    kmeans_rot_q.cls_ids.clone(),
    kmeans_sc_q.cls_ids.clone(),
    kmeans_dc_q.cls_ids.clone(),
    kmeans_sh_q.cls_ids.clone()
], dim=-1)
    return indices

def decode_output_to_gaussians_soft(output, gaussians, kmeans_rot_q, 
                                   kmeans_sc_q, kmeans_dc_q, kmeans_sh_q,
                                   xyz_mean, xyz_std, opacity_mean, opacity_std,  # ✅ 添加这 4 个参数
                                   temperature=0.1):
    """使用 Gumbel-Softmax 实现可微分解码"""
    pred_logits = output['pred_indices_logits']
    pred_cont = output['pred_cont_params'].squeeze(0)
    
    corrupted_params = {}
    
    # 使用 Gumbel-Softmax（训练时 hard=False）
    rot_probs = F.gumbel_softmax(pred_logits[0].squeeze(0), tau=temperature, hard=False, dim=-1)
    scale_probs = F.gumbel_softmax(pred_logits[1].squeeze(0), tau=temperature, hard=False, dim=-1)
    dc_probs = F.gumbel_softmax(pred_logits[2].squeeze(0), tau=temperature, hard=False, dim=-1)
    sh_probs = F.gumbel_softmax(pred_logits[3].squeeze(0), tau=temperature, hard=False, dim=-1)
    
    # 软查表（可微分）
    corrupted_params['rotation'] = torch.matmul(
        rot_probs, kmeans_rot_q.centers
    ).reshape(gaussians._rotation.shape)
    
    corrupted_params['scale'] = torch.matmul(
        scale_probs, kmeans_sc_q.centers
    ).reshape(gaussians._scaling.shape)
    
    corrupted_params['f_dc'] = torch.matmul(
        dc_probs, kmeans_dc_q.centers
    ).reshape(gaussians._features_dc.shape)
    
    corrupted_params['f_rest'] = torch.matmul(
        sh_probs, kmeans_sh_q.centers
    ).reshape(gaussians._features_rest.shape)
    
    corrupted_params['xyz'] = pred_cont[:, :3]
    corrupted_params['opacity'] = pred_cont[:, 3:4]

    # 2. ✅ 连续参数解码（反归一化）
    xyz_normalized = pred_cont[:, :3]  # [N, 3]
    opacity_normalized = pred_cont[:, 3:4]  # [N, 1]
    
    # 反归一化：denorm = norm * std + mean
    corrupted_params['xyz'] = xyz_normalized * (xyz_std + 1e-8) + xyz_mean  # [N, 3]
    corrupted_params['opacity'] = opacity_normalized * (opacity_std + 1e-8) + opacity_mean  # [N, 1]
    
    return corrupted_params

def decode_output_to_gaussians_hard(output, gaussians, kmeans_rot_q, 
                                    kmeans_sc_q, kmeans_dc_q, kmeans_sh_q,
                                    xyz_mean, xyz_std, opacity_mean, opacity_std):
    """
    硬解码：直接使用 argmax 选择最可能的索引（用于保存和评估）
    
    Args:
        output: 信道解码器输出
        gaussians: GaussianModel
        kmeans_*_q: K-Means 量化器
        xyz_mean, xyz_std, opacity_mean, opacity_std: 归一化参数
    
    Returns:
        decoded_params: 解码后的参数字典
    """
    pred_logits = output['pred_indices_logits']
    pred_cont = output['pred_cont_params'].squeeze(0)
    
    decoded_params = {}
    
    # 1. 索引解码（硬解码：argmax）
    rot_indices = torch.argmax(pred_logits[0].squeeze(0), dim=-1)  # [N]
    scale_indices = torch.argmax(pred_logits[1].squeeze(0), dim=-1)
    dc_indices = torch.argmax(pred_logits[2].squeeze(0), dim=-1)
    sh_indices = torch.argmax(pred_logits[3].squeeze(0), dim=-1)
    
    # 2. 查表获取量化值
    decoded_params['rotation'] = kmeans_rot_q.centers[rot_indices].reshape(
        gaussians._rotation.shape
    )
    decoded_params['scale'] = kmeans_sc_q.centers[scale_indices].reshape(
        gaussians._scaling.shape
    )
    decoded_params['f_dc'] = kmeans_dc_q.centers[dc_indices].reshape(
        gaussians._features_dc.shape
    )
    decoded_params['f_rest'] = kmeans_sh_q.centers[sh_indices].reshape(
        gaussians._features_rest.shape
    )
    
    # 3. 连续参数解码（反归一化）
    xyz_normalized = pred_cont[:, :3]
    opacity_normalized = pred_cont[:, 3:4]
    
    decoded_params['xyz'] = xyz_normalized * (xyz_std + 1e-8) + xyz_mean
    decoded_params['opacity'] = opacity_normalized * (opacity_std + 1e-8) + opacity_mean
    
    return decoded_params

def save_original_params(gaussians):
    """保存原始参数"""
    return {
        'xyz': gaussians._xyz.clone(),
        'rotation': gaussians._rotation.clone(),
        'scale': gaussians._scaling.clone(),
        'f_dc': gaussians._features_dc.clone(),
        'f_rest': gaussians._features_rest.clone(),
        'opacity': gaussians._opacity.clone()
    }

def apply_corrupted_params(gaussians, corrupted_params):
    """应用损坏的参数"""
    # gaussians._xyz = corrupted_params['xyz'].detach()
    # gaussians._rotation = corrupted_params['rotation'].detach()
    # gaussians._scaling = corrupted_params['scale'].detach()
    # gaussians._features_dc = corrupted_params['f_dc'].detach()
    # gaussians._features_rest = corrupted_params['f_rest'].detach()
    # gaussians._opacity = corrupted_params['opacity'].detach()
    gaussians._xyz.data = corrupted_params['xyz']  # ← 只修改数据，保持叶子节点
    gaussians._rotation.data = corrupted_params['rotation']
    gaussians._scaling.data = corrupted_params['scale']
    gaussians._features_dc.data = corrupted_params['f_dc']
    gaussians._features_rest.data = corrupted_params['f_rest']
    gaussians._opacity.data = corrupted_params['opacity']

def restore_original_params(gaussians, original_params):
    """恢复原始参数"""
    gaussians._xyz.data = original_params['xyz']
    gaussians._rotation.data = original_params['rotation']
    gaussians._scaling.data = original_params['scale']
    gaussians._features_dc.data = original_params['f_dc']
    gaussians._features_rest.data = original_params['f_rest']
    gaussians._opacity.data = original_params['opacity']

def save_decoded_gaussians(gaussians, corrupted_params, kmeans_list, quantized_params, 
                           output_dir, iteration, save_ply=True):
    """
    保存接收端解码后的 3DGS 数据
    
    Args:
        gaussians: 原始 GaussianModel (用于获取结构信息)
        corrupted_params: 解码后的参数字典
        kmeans_list: K-Means 量化器列表 [kmeans_rot_q, kmeans_sc_q, kmeans_dc_q, kmeans_sh_q]
        quantized_params: 量化参数名称列表 ['rot', 'scale', 'dc', 'sh']
        output_dir: 输出目录
        iteration: 当前迭代次数
        save_ply: 是否保存 PLY 文件
    """
    import os
    from pathlib import Path
    
    # 创建输出目录
    save_dir = Path(output_dir) / f"decoded_point_cloud/iteration_{iteration}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"[Saving Decoded Gaussians] Iteration {iteration}")
    print(f"{'='*80}")
    print(f"  Output directory: {save_dir}")
    
    # ========== 1. 保存 K-Means 索引（二进制格式）==========
    bitarray_all = bitarray([])
    n_bits_list = []
    
    for kmeans in kmeans_list:
        n_bits = int(np.ceil(np.log2(len(kmeans.cls_ids))))
        n_bits_list.append(n_bits)
        assignments = dec2binary(kmeans.cls_ids, n_bits)
        bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
        bitarray_all.extend(bitarr)
    
    with open(save_dir / 'kmeans_inds.bin', 'wb') as file:
        bitarray_all.tofile(file)
    
    print(f"  ✓ Saved kmeans_inds.bin ({len(bitarray_all)} bits)")
    
    # ========== 2. 保存 K-Means 参数 ==========
    args_dict = {
        'params': quantized_params,
        'n_bits': n_bits_list[0] if len(set(n_bits_list)) == 1 else n_bits_list,
        'total_len': len(bitarray_all),
        'num_gaussians': len(gaussians._xyz)
    }
    np.save(save_dir / 'kmeans_args.npy', args_dict)
    print(f"  ✓ Saved kmeans_args.npy")
    
    # ========== 3. 保存 K-Means 中心（码本）==========
    centers_dict = {
        param: kmeans.centers 
        for kmeans, param in zip(kmeans_list, quantized_params)
    }
    torch.save(centers_dict, save_dir / 'kmeans_centers.pth')
    print(f"  ✓ Saved kmeans_centers.pth")
    
    # ========== 4. 保存解码后的 PLY 文件 ==========
    if save_ply:
        # 创建临时 GaussianModel 来保存
        from scene.gaussian_model import GaussianModel
        from plyfile import PlyData, PlyElement
        
        # 准备数据
        xyz = corrupted_params['xyz'].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        
        # 处理其他参数
        f_dc = corrupted_params['f_dc'].detach().cpu().numpy()
        f_rest = corrupted_params['f_rest'].detach().cpu().numpy()
        opacities = corrupted_params['opacity'].detach().cpu().numpy()
        scale = corrupted_params['scale'].detach().cpu().numpy()
        rotation = corrupted_params['rotation'].detach().cpu().numpy()
        
        # 转换为适合保存的格式
        # DC: [N, 1, 3] -> [N, 3]
        if f_dc.ndim == 3:
            f_dc = f_dc.squeeze(1)
        
        # SH rest: [N, 15, 3] -> [N, 45]
        if f_rest.ndim == 3:
            f_rest = f_rest.reshape(f_rest.shape[0], -1)
        
        # 构建 PLY 数据
        def construct_list_of_attributes():
            """构建 PLY 属性列表"""
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            
            # DC colors
            for i in range(f_dc.shape[1]):
                l.append('f_dc_{}'.format(i))
            
            # SH features
            for i in range(f_rest.shape[1]):
                l.append('f_rest_{}'.format(i))
            
            l.append('opacity')
            
            # Scale
            for i in range(scale.shape[1]):
                l.append('scale_{}'.format(i))
            
            # Rotation
            for i in range(rotation.shape[1]):
                l.append('rot_{}'.format(i))
            
            return l
        
        # 构建数据列表
        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        # 保存 PLY
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_dir / 'point_cloud.ply')
        print(f"  ✓ Saved point_cloud.ply ({xyz.shape[0]} Gaussians)")
    
    # ========== 5. 保存额外的统计信息 ==========
    stats = {
        'iteration': iteration,
        'num_gaussians': len(gaussians._xyz),
        'quantized_params': quantized_params,
        'codebook_sizes': {
            param: len(kmeans.centers)
            for param, kmeans in zip(quantized_params, kmeans_list)
        },
        'total_bits': len(bitarray_all),
        'bits_per_gaussian': len(bitarray_all) / len(gaussians._xyz)
    }
    
    with open(save_dir / 'decode_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ Saved decode_stats.json")
    print(f"\n  Statistics:")
    print(f"    Num Gaussians:      {stats['num_gaussians']}")
    print(f"    Total bits:         {stats['total_bits']}")
    print(f"    Bits per Gaussian:  {stats['bits_per_gaussian']:.2f}")
    print(f"    Codebook sizes:     {stats['codebook_sizes']}")
    print(f"{'='*80}\n")

def save_decoded_checkpoint(gaussians, corrupted_params, iteration, model_path):
    """
    保存解码后的完整 checkpoint（用于后续加载和渲染）
    
    Args:
        gaussians: 原始 GaussianModel
        corrupted_params: 解码后的参数
        iteration: 迭代次数
        model_path: 模型路径
    """
    # 创建参数字典
    decoded_state = {
        # 核心参数
        '_xyz': corrupted_params['xyz'].detach().cpu(),
        '_rotation': corrupted_params['rotation'].detach().cpu(),
        '_scaling': corrupted_params['scale'].detach().cpu(),
        '_features_dc': corrupted_params['f_dc'].detach().cpu(),
        '_features_rest': corrupted_params['f_rest'].detach().cpu(),
        '_opacity': corrupted_params['opacity'].detach().cpu(),
        
        # SH degree
        'active_sh_degree': gaussians.active_sh_degree,
        'max_sh_degree': gaussians.max_sh_degree,
        
        # 其他属性（如果需要）
        'num_gaussians': len(gaussians._xyz),
    }
    
    # 保存 checkpoint
    checkpoint_path = os.path.join(model_path, f"decoded_chkpnt{iteration}.pth")
    torch.save((decoded_state, iteration), checkpoint_path)
    
    print(f"  ✓ Saved decoded checkpoint: {checkpoint_path}")
    print(f"    - Num Gaussians: {decoded_state['num_gaussians']}")
    print(f"    - SH degree: {decoded_state['active_sh_degree']}/{decoded_state['max_sh_degree']}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 15_000, 20_000,
                                                                           25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--total_iterations', type=int, default=30000,
                        help='Total iterations of training')

    # Compress3D parameters
    parser.add_argument('--kmeans_w_iter', type=int, default=30000,
                        help='Start weighted k-Means based vector quantization from this iteration')
    parser.add_argument('--kmeans_st_iter', type=int, default=30000,
                        help='Start k-Means based vector quantization from this iteration')
    parser.add_argument('--kmeans_ncls', type=int, default=64,
                        help='Number of clusters in k-Means quantization')
    # parser.add_argument('--kmeans_ncls', type=int, default=16384,
    #                     help='Number of clusters in k-Means quantization')
    # parser.add_argument('--kmeans_ncls', type=int, default=32768,
    #                     help='Number of clusters in k-Means quantization')
    parser.add_argument('--kmeans_ncls_sh', type=int, default=64,
                        help='Number of clusters in k-Means quantization of spherical harmonics')
    # parser.add_argument('--kmeans_ncls_sh', type=int, default=512,
    #                     help='Number of clusters in k-Means quantization of spherical harmonics')
    parser.add_argument('--kmeans_ncls_dc', type=int, default=64,
                        help='Number of clusters in k-Means quantization of DC component of color')
    parser.add_argument('--kmeans_iters', type=int, default=1,
                        help='Number of assignment and centroid calculation iterations in k-Means')
    parser.add_argument('--kmeans_freq', type=int, default=100,
                        help='Frequency of cluster assignment in k-Means')
    parser.add_argument('--grad_thresh', type=float, default=0.0002,
                        help='threshold on xyz gradients for densification')
    parser.add_argument("--quant_params", nargs="+", type=str, default=['sh', 'dc', 'scale', 'rot'])
    # parser.add_argument("--quant_params", nargs="+", type=str, default=[])

    # Opacity regularization parameters
    parser.add_argument('--max_prune_iter', type=int, default=20000,
                        help='Iteration till which pruning is done')
    parser.add_argument('--opacity_reg', action='store_true', default=False,
                        help='use opacity regularization during training')  
    parser.add_argument('--lambda_reg', type=float, default=0.,
                        help='Weight for opacity regularization in loss')

    # Scale regularization parameters
    parser.add_argument('--scale_reg', action='store_true', default=False,
                        help='use scale regularization during training')
    parser.add_argument('--lambda_scale_reg', type=float, default=0.,
                        help='Weight for scale regularization in loss')

    parser.add_argument('--mock_channel', action='store_true', default=False,
                        help='Enable mock wireless channel simulation (y = x + noise)')
    parser.add_argument('--snr_db', type=float, default=5.0,
                        help='SNR in dB for mock channel')
    parser.add_argument('--lr_index', type=float, default=1e-2,
                        help='Initial learning rate for index parameters')
    parser.add_argument('--lr_cont', type=float, default=1e-2,
                        help='Initial learning rate for continuous parameters')
        # 在第 840 行后添加
    parser.add_argument('--save_decoded_iters', nargs="+", type=int, 
                       default=[30000],
                       help='Iterations to save decoded Gaussians from channel')
        # 在 ArgumentParser 部分（约第 940 行）添加：
    parser.add_argument('--channel_type', type=str, default='rayleigh',
                       choices=['rayleigh', 'awgn'],
                       help='Wireless channel type: rayleigh (fading) or awgn (white noise only)')
        # 在第 940 行左右（参数定义部分）添加：
    parser.add_argument('--disable_film', action='store_true',
                       help='Disable FiLM (view-adaptive modulation) for benchmark')
    

    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.test_iterations = list(np.arange(0, args.total_iterations + 1, 100))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    outfile = join(args.model_path, 'train_args.json')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as fp:
        json.dump(vars(args), fp, indent=4, default=str)
    print('Quantized Params: ', args.quant_params)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
