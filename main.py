import argparse
from pyprnt import prnt
from utils.trainer_utils import set_trainer
from utils.utils import print_with_timestamp, set_seed

def main(args):
    set_seed(0)
    trainer = set_trainer(args)
    print_with_timestamp(f"Start training: {trainer.log_name}")
    
    args_dict = vars(args)
    # prnt(args_dict)
    print_with_timestamp(f"Hyperparameters:")
    for key, value in args_dict.items():
        print_with_timestamp(f"    {key}: {value}")

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Teacher - MRI Registration 
    parser.add_argument("--teacher_path", type=str, default='teacher_models/Mid_U_Net/VM-diff_NCC(tvl2_0.2)_epochs400_augTrue_geoFalse_12-01_16-25_bestDSC.pt')
    parser.add_argument("--teacher_model", type=str, default='Mid_U_Net', choices=['U_Net', 'Mid_U_Net', 'Big_U_Net'])
    parser.add_argument("--template_path", type=str, default="data/mni152_resample.npy")
    parser.add_argument("--saved_path", default=None)

    # Student - PET Spatial Normalization
    ## training options 
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--KD", type=str, choices=['None', 'field', 'feature'], default='field')
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--numpy", action='store_true', default=False)

    ## model
    parser.add_argument("--model", type=str, default='U_Net', choices=['U_Net', 'Mid_U_Net', 'Big_U_Net', 'Tiny_U_Net'])

    ## dataset
    parser.add_argument("--dataset", type=str, default='SNUH', choices=['DLBS', 'OASIS', 'LUMIR', 'SNUH'])

    ## loss
    parser.add_argument("--data_aug", action='store_true', default=False)
    parser.add_argument("--data_aug_geo", action='store_true', default=False)
    
    # for regularizer
    ## NON-KD
    parser.add_argument("--sim_loss", type=str, default='None', choices=['None', 'MSE', 'NCC'])
    parser.add_argument("--alpha_tv", type=float, default=1.0)
    parser.add_argument("--alpha_dice", type=float, default=1.0)
    parser.add_argument("--alpha_suvr", type=float, default=1.0)

    ## KD field loss
    parser.add_argument("--lamb", type=float, default=0.2)

    # for uncertainty
    # parser.add_argument("--image_sigma", type=float, default=0.02)
    # parser.add_argument("--prior_lambda", type=float, default=20.0)
    # parser.add_argument("--num_samples", type=int, default=1)

    # validation options
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--save_num", type=int, default=2)
    # parser.add_argument("--val_detail", default=False, action='store_true')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none', choices=['none', 'multistep'])
    parser.add_argument("--lr_milestones", type=str, default=None)

    # log options
    parser.add_argument("--log_method", type=str, default='tensorboard', choices=['tensorboard', 'wandb'])
    parser.add_argument("--wandb_name", type=str, default='None')
    parser.add_argument("--log_dir", type=str, default='logs')

    args = parser.parse_args()

    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    main(args)

