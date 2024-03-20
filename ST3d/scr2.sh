# Optimizer study
# CUDA_VISIBLE_DEVICES=2,3 python3 apretrain.py --runname adam --opt adam
# CUDA_VISIBLE_DEVICES=2,3 python3 apretrain.py --runname adamw --opt adaw
# CUDA_VISIBLE_DEVICES=2,3 python3 apretrain.py --runname sgd --opt sgd

CUDA_VISIBLE_DEVICES=6,7 python3 apretrain.py --runname maskrat40 --mask 0.40
CUDA_VISIBLE_DEVICES=6,7 python3 apretrain.py --runname maskrat60 --mask 0.60