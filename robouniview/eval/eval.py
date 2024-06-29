""" Main training script """
print('inininininii')
import argparse
import glob
import os
import sys
env = os.environ    
current_path = os.getcwd()  
robouniview_path =  current_path  
env['PATH'] = env['PATH'] + ':'+  robouniview_path
sys.path.append(robouniview_path)

sys.path.append('.../RoboUniView/open_flamingo')
sys.path.append('.../calvin/calvin_models')
sys.path.append('.../calvin/calvin_env')
sys.path.append('.../calvin/calvin_env/tacto_env')

from collections import OrderedDict
import copy
import random
import logging
from torch.distributed.elastic.multiprocessing.errors import record

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import torch
import wandb
import torch.distributed as dist
from tqdm.auto import tqdm
from pathlib import Path
from collections import Counter, defaultdict, namedtuple
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP

from robouniview.data.data import get_data, get_env
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from eval_utils import eval_one_epoch_calvin, eval_one_epoch_calvin_ddp, eval_one_epoch_calvin_with_dataloder
from robouniview.models.factory import create_model_and_transforms, mpt_dict
logger = logging.getLogger(__name__)
#from lff_lightning.machine_learning.global_config import GlobalConfig
import yaml

class GlobalConfig:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self._load_config()

    def _load_config(self):
        for key, val in self.config.items():
            setattr(self, key, val)


def load_global_config_yaml_only(config_path: str) -> GlobalConfig:
    with open(config_path, "r") as infile:
        config = yaml.safe_load(infile)
    global_config = GlobalConfig(config)
    return global_config

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate_from_checkpoint",
        type=str,
        help="path to checkpoint to evaluate , this should contain model",
        default=None,
    )
    _args = parser.parse_args()
    yaml_path = _args.evaluate_from_checkpoint.replace(os.path.basename(_args.evaluate_from_checkpoint),'config.yaml')
    args = load_global_config_yaml_only(yaml_path)
    args.evaluate_from_checkpoint = _args.evaluate_from_checkpoint

    if args.head_type == "diffusion":
        args.pad_length = args.n_obs_steps
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if 'sep' in args.evaluate_from_checkpoint:
        args.sep_resampler = True
    if 'lm_head' in args.evaluate_from_checkpoint:
        args.sep_lm_head = True
    if 'res_' in args.evaluate_from_checkpoint:
        args.residual = True
    if 'tcp' in args.evaluate_from_checkpoint:
        args.tcp_rel = True
    if 'fur' in args.evaluate_from_checkpoint.split('_'):
        name_attrs = args.evaluate_from_checkpoint.split('_')
        args.multi_step_action = int(name_attrs[name_attrs.index('fur')-1])
    if 'difws' in args.evaluate_from_checkpoint:
        args.dif_ws = True
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('difws')
        min_ws = int(name_attrs[ix+1])
        max_ws = int(name_attrs[ix+2])
        args.min_window_size = min_ws
        args.max_window_size = max_ws
        args.window_size = max_ws
    if 'latent' in args.evaluate_from_checkpoint:
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('latent')
        args.global_latent = int(name_attrs[ix+1])
    if 'no_image_patch' in args.evaluate_from_checkpoint:
        args.no_image_patch = True
    if 'gpt' in args.evaluate_from_checkpoint:
        args.decoder_type = 'gpt'
        name_attrs = args.evaluate_from_checkpoint.split('_')
        hidden_size = int(name_attrs[name_attrs.index('gpt')+1])
        args.hidden_size = hidden_size
    for name in ['mpt_3b', 'mpt_4b', 'mpt_9b', 'mpt_dolly_3b', 'mpt_base_4b']:
        if name in args.evaluate_from_checkpoint:
            args.llm_name = name
            break
    
    args.lm_path = mpt_dict[args.llm_name]["lang_encoder_path"]
    args.tokenizer_path = mpt_dict[args.llm_name]["tokenizer_path"]
    args.cross_attn_every_n_layers = mpt_dict[args.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.llm_name]["openflamingo_checkpoint"]
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    print("world_size: ", torch.distributed.get_world_size())
    random_seed(args.seed)

    model, image_processor, tokenizer = create_model_and_transforms(
        args,
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"),
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        fusion_mode=args.fusion_mode,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        pad_length=args.pad_length,
        debug=args.debug,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        sep_lm_head=args.sep_lm_head,
        return_feature=True,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
        replan=args.replan,
        decoder_type=args.decoder_type,
        hidden_size=args.hidden_size,
        freeze_sampler=args.freeze_sampler,
        fwd_pred=args.fwd_pred,
        fwd_pred_hand=args.fwd_pred_hand,
        no_image_patch=args.no_image_patch,
        global_latent=args.global_latent,
        # refresh=args.refresh,
        clip_cache_dir=args.clip_cache_dir,
    )
    # model, image_processor, tokenizer = create_model_and_transforms(
        
    #     args.vision_encoder_path,
    #     args.vision_encoder_pretrained,
    #     args.lm_path,
    #     args.tokenizer_path if args.tokenizer_path else args.lm_path,
    #     cross_attn_every_n_layers=args.cross_attn_every_n_layers,
    #     use_gripper=args.use_gripper,
    #     use_state=args.use_state,
    #     use_hist=args.use_hist,
    #     fusion_mode=args.fusion_mode,
    #     use_local_files=args.offline,
    #     use_media_placement_augmentation=args.use_media_placement_augmentation,
    #     window_size=args.eval_hist_size,
    #     freeze_embed=args.freeze_embed,
    #     train_params=args.train_params,
    #     sep_resampler=args.sep_resampler,
    #     last_action=args.last_action,
    #     use_diff=(args.head_type == "diffusion"), # Diff still have bugs of loaded data mismatch
    #     n_timesteps=args.n_timesteps,
    #     diff_horizon=args.diff_horizon,
    #     predict_epsilon=args.predict_epsilon,
    #     sep_lm_head=args.sep_lm_head,
    #     unfreeze_vit=args.unfreeze_vit,
    #     multi_step_action=args.multi_step_action,
    #     llm_name=args.llm_name,
    #     pooling=args.pooling,
    #     residual=args.residual,
    #     tcp_rel=args.tcp_rel,
    #     decoder_type=args.decoder_type,
    #     hidden_size=args.hidden_size,
    #     freeze_sampler=args.freeze_sampler,
    #     fwd_pred=args.fwd_pred,
    #     fwd_pred_hand=args.fwd_pred_hand,
    #     no_image_patch=args.no_image_patch,
    #     global_latent=args.global_latent,
    #     clip_cache_dir=args.clip_cache_dir
    # )
    checkpoint_path = args.openflamingo_checkpoint
    print("Loading origin flamingo checkpoint from ", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    if args.sep_lm_head:
        model.lm_head.requires_grad_(True)
    else:
        model.lang_encoder.lm_head.requires_grad_(True)

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    model = model.to(device_id)
    model.eval()

    ddp_model = DDP(model, device_ids=[device_id])
    if args.residual:
        model.lang_encoder.clone_parameters()
    # if args.evaluate_from_checkpoint is specified, load checkpoint
    assert args.evaluate_from_checkpoint is not None, "Please specify a checkpoint to evaluate."
    if args.rank == 0:
        print(f"Loading robot-flamingo checkpoint from {args.evaluate_from_checkpoint}")
    checkpoint = torch.load(args.evaluate_from_checkpoint, map_location="cpu")
    def filter_ckpt(checkpoint, flags=[]):
            new_state_dict = OrderedDict()
            for key, value in checkpoint["model_state_dict"].items():
                load_p = True
                for flag in flags:
                    if flag in key:
                        load_p = True
                
                if load_p:
                    if 'bevformer' in key:
                        key = key.replace('bevformer','uvformer')

                    if 'bev2vision' in key:
                        key = key.replace('bev2vision','alignment_layer')
                    new_state_dict[key] = value
            return new_state_dict
    flags = []

    checkpoint = filter_ckpt(checkpoint,flags)

    ddp_model.load_state_dict(checkpoint, False)

    #ddp_model.load_state_dict(checkpoint["model_state_dict"], False)  # 只保存了求梯度的部分

    ddp_model.eval()
    eval_log_dir = None
    # if args.debug:
    eval_log_dir = '{}'.format(args.evaluate_from_checkpoint.split('.')[0])
    if 0:
        calvin_sim = get_data(args, image_processor, tokenizer, "calvinSim")
        calvin_sim.set_epoch(0)
        calvin_loader = calvin_sim.dataloader
        eval_one_epoch_calvin_with_dataloder(
            args=args,
            model=ddp_model,
            image_processor=image_processor,
            tokenizer=tokenizer,
            calvin_loader=calvin_loader,
            future_act_len=args.future_act_len,
            eval_log_dir=eval_log_dir,
            debug=args.debug,
            reset=args.reset,
            diverse_inst=args.diverse_inst
        )

    else:
        eval_one_epoch_calvin_ddp(
            args=args,
            model=ddp_model,
            image_processor=image_processor,
            tokenizer=tokenizer,
            dataset_path=args.calvin_dataset,
            future_act_len=args.future_act_len,
            eval_log_dir=eval_log_dir,
            debug=args.visualize,
            reset=args.reset,
            diverse_inst=args.diverse_inst
        )





if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = '1'
    import os
    import torch
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['WORLD_SIZE'] = '1'
    main()









