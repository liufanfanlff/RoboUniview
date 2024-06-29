import os

ckpt_dirs = ['.../']
ckpt_names = ["checkpoint_gripper_petr_hist_1_state_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_7.pth7.pth",
              ]
              
for ckpt_name in ckpt_names:
    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        os.system('bash robot_flamingo/pt_eval_ckpts.bash {}'.format(ckpt_path))




