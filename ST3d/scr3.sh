# Augmentation study
# CUDA_VISIBLE_DEVICES=4,5 python3 apretrain.py --runname resizeremove --rrc True
# CUDA_VISIBLE_DEVICES=4,5 python3 apretrain.py --runname flipremove --rrc True
# CUDA_VISIBLE_DEVICES=4,5 python3 apretrain.py --runname normremove --nonorm True

CUDA_VISIBLE_DEVICES=6,7 python3 apretrain.py --runname maskrat20 --mask 0.20