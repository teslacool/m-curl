#!/usr/bin/env bash
set -x
set -e
cd ..

DOMAIN=${1:-"cartpole"}
TASK=${2:-"swingup"}
shift 2
WORKDIR=run
AGENT=ctmr_sac
SEED=-1
CUDA=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
      -s|--seed)
        SEED=$2
        shift 2
        ;;
      -a|--agent)
        AGENT=$2
        shift 2
        ;;
      -d|--work_dir)
        WORKDIR=$2
        shift 2
        ;;
      -c|--cuda)
        CUDA=$2
        shift 2
        ;;
      *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done

SUFFIX=`echo ${POSITIONAL[*]} | sed -r 's/-//g' | sed -r 's/\s/-/g'`
if [ -n "$SUFFIX" ]
then
  more="--exp_suffix $SUFFIX"
else
  more=""
fi

CUDA_VISIBLE_DEVICES=$CUDA python train.py \
  --domain_name $DOMAIN --task_name $TASK \
  --save_tb --work_dir ./$WORKDIR --agent $AGENT \
  --seed $SEED ${POSITIONAL[@]} \
  $more