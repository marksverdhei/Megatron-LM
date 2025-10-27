docker run \
  --gpus all \
  -it \
  --rm \
  -v /home/me/Repos/Megatron-LM/megatron:/workspace/megatron \
  -v /home/me/Repos/Megatron-LM/examples:/workspace/examples \
  -v /home/me/Repos/Megatron-LM/data:/workspace/dataset \
  -v /media/me/storage/Models/MegatronLM:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.04-py3

docker run --gpus all -it --rm -v /home/me/Repos/Megatron-LM/megatron:/workspace/megatron -v /home/me/Repos/Megatron-LM/examples:/workspace/examples -v /home/me/Repos/Megatron-LM/data:/workspace/dataset -v /media/me/storage/Models/MegatronLM:/workspace/checkpoints -e PIP_CONSTRAINT= nvcr.io/nvidia/pytorch:25.04-py3

