import torch
from Trainer.Trainer_base import Trainer_base
from utils.loss import Direct_loss
from utils.utils import apply_deformation_using_disp

class Trainer_direct(Trainer_base):
    def __init__(self, args):
        self.log_name = f'Direct_{args.sim_loss}_tv{args.alpha_tv}_dice{args.alpha_dice}_suvr{args.alpha_suvr}'
        self.Direct_loss_fn = Direct_loss(sim_loss=args.sim_loss, alpha_tv=args.alpha_tv, alpha_dice=args.alpha_dice, alpha_suvr=args.alpha_suvr)

        super().__init__(args)

    def forward(self, MRI, MRI_template, PET, img_segs, temp_segs, epoch):
        # Model inference
        s_displace, _ = self.S_model(PET)  # [B, 3, D, H, W], [B, 1, D, H, W], list of [B, C, D, H, W]
        s_displace = s_displace[-1]  # [B, 3, D, H, W]
        
        img_segs, temp_segs = [i.unsqueeze(1).cuda() for i in img_segs], [t.unsqueeze(1).cuda() for t in temp_segs]

        deformed_PET = apply_deformation_using_disp(PET, s_displace)
        deformed_MRI = apply_deformation_using_disp(MRI, s_displace)
        deformed_segs = [apply_deformation_using_disp(s, s_displace, mode='nearest') for s in img_segs]

        loss, sim_loss_value, tv_loss_value, dice_loss_value, SUVr_loss_value = self.Direct_loss_fn(
            s_displace, PET, deformed_PET,
            img_segs, deformed_segs, temp_segs,
            deformed_MRI, MRI_template
        )

        self.log_dict['Loss_tot'] += loss.item()
        self.log_dict['Loss_Sim'] += sim_loss_value
        self.log_dict['Loss_TV'] += tv_loss_value
        self.log_dict['Loss_Dice'] += dice_loss_value
        self.log_dict['Loss_SUVr'] += SUVr_loss_value

        self.disp = s_displace

        return loss, deformed_PET, deformed_segs

    def log(self, epoch, phase=None):
        if phase not in ['train', 'valid']:
            raise ValueError("Trainer's log function can only get phase ['train', 'valid'], but received", phase)

        if phase == 'train':
            num = len(self.train_loader)
            tag = 'Train'
        elif phase == 'valid':
            num = len(self.val_loader)
            tag = 'Val'
        
        for key, value in self.log_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", value/num, epoch)


    def reset_logs(self):
        # for single layer, deterministic version (VM)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_Sim':0.0,
            'Loss_TV':0.0,
            'Loss_Dice':0.0,
            'Loss_SUVr':0.0
        }
    
    def get_disp(self):
        # get displacement field from student model and teacher
        return self.disp