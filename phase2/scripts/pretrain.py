"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

def check_is_pretrain():
    if os.path.isdir("configs_train") and os.path.isdir("configs_pretrain"):
        os.rename("configs_pretrain", "configs")
        return
    elif os.path.isdir("configs_train") and os.path.isdir("configs"):
        return
    elif os.path.isdir("configs_pretrain") and os.path.isdir("configs"):
        os.rename("configs", "configs_train")
        os.rename("configs_pretrain", "configs")
        return
    else:
        raise Exception("Something is wrong with your configs folder")
check_is_pretrain()

from options import Options
from training.coach import Coach

def get_best_model(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "timestamp.txt"), "r") as file:
        timestamp = file.readlines()

    best_models = [line for line in timestamp if line.startswith("**Best saved at best_model")][-1]
    best_models = eval(best_models.split(", Loss - ")[1])

    checkpoint_path = os.path.join(checkpoint_dir, best_models[-1][0])
    return checkpoint_path

def main():
    opts = Options(is_train=True).parse()
    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir)

    opts.checkpoint_path = None
    if opts.checkpoint_dir is not None:
        opts.checkpoint_path = get_best_model(opts.checkpoint_dir)
    
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)
    
    coach = Coach(opts)
    coach.train()
    
    os.rename("configs", "configs_pretrain")

if __name__ == '__main__':
    main()
