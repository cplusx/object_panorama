# python tools/overfit_rectangular_conditional_jit.py \
#   configs/experiment/edge3d_flow_overfit.yaml \
#   --device cuda \
#   --num-samples 4 \
#   --max-steps 3000
CUDA_VISIBLE_DEVICES=0,1 \
python tools/train_lightning_rectangular_conditional_jit.py \
  configs/experiment/edge3d_flow_train.yaml \
  --device cuda \
  --precision 32-true