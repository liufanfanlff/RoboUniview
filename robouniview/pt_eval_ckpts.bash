# !!! Set for your own path

evaluate_from_checkpoint=$1

export MESA_GL_VERSION_OVERRIDE=4.1

node_num=8

torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6066 robouniview/eval/eval.py \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint}

