#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

# import wandb
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def log_fps_to_file(model_path, dataset_type, fps, iteration):
    """FPS record"""
    log_file = "fps_results.txt"  
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{model_path}, {dataset_type}, iter_{iteration}, {fps:.2f}\n")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t_start = time.time() # syn
        rendering = render(view, gaussians, pipeline, background)["render"]
        torch.cuda.synchronize(); t_end = time.time()

        t_list.append(t_end - t_start)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    return t_list


# def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
#                 load_quant: bool, wandb=None, tb_writer=None, dataset_name=None, logger=None):
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                load_quant: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_quant=load_quant)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             t_train_list = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

             train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
             print(f"Train FPS: {train_fps:.2f}")
             log_fps_to_file(dataset.model_path, "train", train_fps, iteration)


            #  logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            #  if wandb is not None:
            #     wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
             t_test_list = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

             test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
             print(f"Test FPS: {test_fps:.2f}")
             log_fps_to_file(dataset.model_path, "test",  test_fps, iteration)

            #  logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            #  if tb_writer:
            #     tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            #  if wandb is not None:
            #     wandb.log({"test_fps":test_fps, })

# def get_logger(path):
#     import logging

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
#     fileinfo.setLevel(logging.INFO)
#     controlshow = logging.StreamHandler()
#     controlshow.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
#     fileinfo.setFormatter(formatter)
#     controlshow.setFormatter(formatter)

#     logger.addHandler(fileinfo)
#     logger.addHandler(controlshow)

#     return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_quant", action="store_true",
                        help='load quantized parameters')
    # parser.add_argument('--use_wandb', action='store_true', default=False)
    
    args = get_combined_args(parser)

    # model_path = args.model_path
    # logger = get_logger(model_path)
    # dataset = args.source_path.split('/')[-1]
    # exp_name = args.model_path.split('/')[-2]

    # if args.use_wandb:
    #     wandb.login()
    #     run = wandb.init(
    #         # Set the project where this run will be logged
    #         project=f"Scaffold-GS-{dataset}",
    #         name=exp_name,
    #         # Track hyperparameters and run metadata
    #         settings=wandb.Settings(start_method="fork"),
    #         config=vars(args)
    #     )
    # else:
    #     wandb = None

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
    #             args.load_quant, wandb=wandb, logger=logger)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
            args.load_quant)