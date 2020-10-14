# M-CURL: Masked Contrastive Representation Learning for Reinforcement Learning

This repository contains the code for M-CURL.

## Installation 

All of the dependencies are in the `conda_env.yml` file.
They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train an M-CURL agent on the `cartpole swingup` task,  please use `script/train.sh` 
from the root of this directory. One example is as follows, 
and you can modify it to try different environments / hyperparamters.
```
#!/usr/bin/env bash
set -x
set -e
export PYTHONIOENCODING="UTF-8"
nvidia-smi
export MJKEY_PATH=/path/mjkey.txt
DOMAIN=cartpole
TASK=swingup
SEED=1
RATIO=0.5
LAYER=2
ANNEAL=true
NORMBEFO=false
DROPOOUT=0.1
BSZ=16
LEN=32

CUDA=${1:-0}

cd scripts
EXTRA=''
if [ "$ANNEAL" == 'true' ]
then
  EXTRA+=" --encoder_annealling "
fi
if [ "$NORMBEFO" == 'true' ]
then
  EXTRA+=" --normalize_before "
fi

export MUJOCO_GL=egl
bash train.sh $DOMAIN $TASK -s $SEED -c $CUDA  --mtm_ratio $RATIO $EXTRA --num_attn_layer $LAYER \
--adam_warmup_step 6e3 --dropout $DROPOOUT  --mtm_length $LEN --mtm_bsz $BSZ -d exp  \
--num_train_steps 1000000
```
