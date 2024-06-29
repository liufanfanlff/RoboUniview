""" Main training script """
import sys
import os  
env = os.environ    
current_path = os.getcwd()  
robouniview_path =  current_path  
env['PATH'] = env['PATH'] + ':'+  robouniview_path
sys.path.append(robouniview_path)

sys.path.append('.../RoboUniView/open_flamingo')
sys.path.append('.../calvin/calvin_models')
sys.path.append('.../calvin/calvin_env')
sys.path.append('.../calvin/calvin_env/tacto_env')
import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
from huggingface_hub import hf_hub_download
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from robouniview.data.multi_cam_data import get_data
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from train_utils import get_checkpoint, train_one_epoch_calvin, train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
get_ckpt_name, get_ckpt_name_pattern,val_one_epoch_calvin
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from robouniview.models.factory import create_model_and_transforms, mpt_dict
from robouniview.eval.eval_utils import eval_one_epoch_calvin_with_dataloder
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
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

@record
def main():
    print('excutin')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    args = parser.parse_args()
    args = load_global_config_yaml_only(args.config)

    attributes_and_values = vars(args)
    attributes_and_values = dict(attributes_and_values)
    
    if not os.path.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir)
        except:
            pass
    with open(args.save_dir+'/config.yaml', 'w') as f:
        yaml.dump(attributes_and_values['config'], f, default_flow_style=None, sort_keys=False)

    # window_size:
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # args.rank = 0
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    random_seed(args.seed)
    args.lm_path = mpt_dict[args.llm_name]["lang_encoder_path"]
    args.tokenizer_path = mpt_dict[args.llm_name]["tokenizer_path"]
    args.cross_attn_every_n_layers = mpt_dict[args.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.llm_name]["openflamingo_checkpoint"]

    model, image_processor, tokenizer = create_model_and_transforms(
        args,
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        fusion_mode=args.fusion_mode,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"), # Diff still have bugs of loaded data mismatch
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        predict_epsilon=args.predict_epsilon,
        sep_lm_head=args.sep_lm_head,
        unfreeze_vit=args.unfreeze_vit,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        pooling=args.pooling,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
        decoder_type=args.decoder_type,
        hidden_size=args.hidden_size,
        freeze_sampler=args.freeze_sampler,
        fwd_pred=args.fwd_pred,
        fwd_pred_hand=args.fwd_pred_hand,
        no_image_patch=args.no_image_patch,
        global_latent=args.global_latent,
        clip_cache_dir=args.clip_cache_dir
    )

    checkpoint_path = args.openflamingo_checkpoint
    if not args.debug and not args.no_pretrain:
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        if args.residual:
            model.lang_encoder.clone_parameters()

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    if args.debug:
        calvin_dataset = get_data(args, image_processor, tokenizer, "debug")
    elif args.real_data:
        calvin_dataset = get_data(args, image_processor, tokenizer, "real")
    else:
        calvin_dataset = get_data(args, image_processor, tokenizer, "calvin")
        calvin_sim = get_data(args, image_processor, tokenizer, "calvinSim")
        calvin_validation = get_data(args, image_processor, tokenizer, "calvin_validation")

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    #if args.rank == 0 and args.report_to_wandb:
        # wandb.init(
        #     project=args.wandb_project,
        #     # entity=args.wandb_entity,
        #     name=args.run_name,
        #     config=vars(args),
        # )
    writer = SummaryWriter(os.path.join(args.save_dir ,'log'))

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    if args.head_type == "diffusion" and (not args.debug):
        normalizer = model.diffusion_model.normalizer
        all_actions = np.vstack([calvin_dataset.dataset.__getitem__((i,1),True)["actions"] for i in range(0,10000)])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')

    model = model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                ("gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x)
                or ("uvformer"in x)
                or ("Upsample2d_3d"in x)
                or ("alignment_layer" in x)
                or ("occ_decoder" in x)
            )

        for n, p in model.named_parameters():

            if apply_decay(n):
                params_with_wd.append(p)
                # print(n)
            else:
                params_without_wd.append(p)

        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": args.weight_decay},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]
    # args.learning_rate = args.learning_rate * args.batch_size_calvin / 6 # adaptive lr

    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)

    total_training_steps = (
        (args.train_num_samples_calvin) // (args.batch_size_calvin * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    use_diff = (args.head_type == "diffusion")
    # check if a checkpoint exists for this run

    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        ckpt_name = get_ckpt_name_pattern(args)
        checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
        print(ckpt_name)
        checkpoint_list = [_ for _ in checkpoint_list if "__sep" not in _ and 'iter' not in _ and 'weights' not in _]
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.load_from_checkpoint is not None:
        checkpoint = torch.load(args.load_from_checkpoint, map_location="cpu")
        if args.rank == 0:
            print(f"Loading checkpoint from {args.load_from_checkpoint}")

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

        flags =['bevformer','bevformer_gripper','Upsample2d_3d','linear']

        checkpoint = filter_ckpt(checkpoint,flags)

        ddp_model.load_state_dict(checkpoint, False)


    if args.resume_from_checkpoint is not None and args.from_scratch is False:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")

        def filter_ckpt(checkpoint, skip_keys=[]):
            new_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                flag = True
                for skip_key in skip_keys:
                    if skip_key in key:
                        flag = False
                        break
                if flag:
                    new_state_dict[key] = value
            return new_state_dict
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        if not args.real_data:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                resume_from_epoch = checkpoint["epoch"] + 1
            except:
                print("optimizer_state_dict_wining")
    ddp_model.train()
    if args.real_data:
        resume_from_epoch = 0
    for epoch in range(resume_from_epoch, args.num_epochs):
        ddp_model.train()
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader

        calvin_validation.set_epoch(epoch)
        calvin_loader_validation = calvin_validation.dataloader

        if args.head_type == "diffusion":
            train_one_epoch_calvin_diff(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif args.fusion_mode == 'two_way':
            train_one_epoch_calvin_two_way(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        else:
            train_one_epoch_calvin(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
                writer=writer,
            )

        if args.validation:
            print("++++++++++++++++validation++++++++++++++++++")
            ddp_model.eval()
            with torch.no_grad():
                val_one_epoch_calvin(
                    args=args,
                    model=ddp_model,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    calvin_loader=calvin_loader_validation,
                    device_id=device_id,
                    wandb=wandb,
                    writer=writer,
                )


        if args.rank == 0:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }

            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(args.save_dir, ckpt_name)

            print('args.delete_previous_checkpoint')
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint_dict, ckpt_path)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(ckpt_path)

        if args.eval:
            eval_log_dir = args.save_dir
            ddp_model.eval()
            calvin_sim.set_epoch(epoch)
            calvin_loader = calvin_sim.dataloader
            results = eval_one_epoch_calvin_with_dataloder(
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
            if torch.distributed.get_rank() == 0:
                print(f"agg_metric", np.array(results).mean())

    if args.rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        ckpt_name = get_ckpt_name(args,)
        torch.save(get_checkpoint(ddp_model), f"{args.save_dir}/{ckpt_name}")
        # if args.report_to_wandb and args.save_checkpoints_to_wandb:
            # wandb.save(f"{args.run_name}/{ckpt_name}")
    # 关闭wandb
    writer.close()



if __name__ == "__main__":
    
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['RANK'] = '0'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ['WORLD_SIZE'] = '1'
    main()
