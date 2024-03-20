# Mask Ratio Study
#CUDA_VISIBLE_DEVICES=0,1 python3 apretrain.py --runname maskrat00 --mask 0
# CUDA_VISIBLE_DEVICES=0,1 python3 apretrain.py --runname maskrat05 --mask 0.05

# # CUDA_VISIBLE_DEVICES=0,1 python3 apretrain.py --runname maskrat25 --mask 0.25
# CUDA_VISIBLE_DEVICES=0,1 python3 apretrain.py --runname maskrat50 --mask 0.50

CUDA_VISIBLE_DEVICES=4,5 python3 apretrain.py --runname maskrat15 --mask 0.15