import os, torch
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.utils import print_with_timestamp, save_middle_slices_mfm, save_grid_spline, apply_deformation_using_disp
from utils.dataset import set_dataloader_usingcsv
from networks.network_utils import set_model

class Trainer_base:
    def __init__(self, args):
        self.args = args
        # other initializations
        self.epochs = args.epochs
        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.save_num = args.save_num

        if args.epochs != 400:
            self.log_name = f'{self.log_name}_epochs{args.epochs}'
        if args.lr_scheduler == 'multistep':
            self.log_name = f'{self.log_name}_sche(multi_{args.lr_milestones})'
        
        if args.transform:
            self.log_name = f'{self.log_name}_aug'

        # add start time
        now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m-%d_%H-%M")
        self.log_name = f'{self.log_name}_{now}'

        self.writer = SummaryWriter(log_dir=f'{args.log_dir}/{args.dataset}/{args.model}/{self.log_name}')
        
        # Setting Model
        self.S_model = set_model(args.model, in_channels=1, out_channels=3, out_layers=1) # basic model (VM)
        if args.pretrained_path is not None:
            self.S_model.load_state_dict(torch.load(args.pretrained_path, weights_only=True, map_location=torch.device('cpu')))
            self.start_epoch = args.start_epoch
        else:
            self.start_epoch = 0
        self.S_model = self.S_model.cuda()
        
        self.T_model = set_model(args.teacher_model, in_channels=2, out_channels=3, out_layers=1) # basic model (VM)
        self.T_model.load_state_dict(torch.load(args.teacher_path, weights_only=True,map_location=torch.device('cpu')))
        self.T_model = self.T_model.cuda()
        self.T_model.eval()

        self.train_loader, self.val_loader, self.save_loader = set_dataloader_usingcsv(args.dataset, 'data/data_list', ['data/FDG_MRI_numpy', 'data/FDG_PET_percent_numpy'], args.template_path, args.batch_size, numpy=args.numpy, transform=args.transform)
        self.save_dir = f'./results/saved_models/{args.dataset}/{args.model}'
        
        os.makedirs(f'{self.save_dir}/completed', exist_ok=True)
        os.makedirs(f'{self.save_dir}/not_finished', exist_ok=True)
        
        self.optimizer = optim.Adam(self.S_model.parameters(), lr=args.lr)
        # set learning rate scheduler 
        if args.lr_scheduler == 'none':
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        elif args.lr_scheduler == 'multistep':
            milestones = [int(i)*len(self.train_loader) for i in args.lr_milestones.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)


    def train(self):
        best_loss = 1e+9
        for epoch in range(self.start_epoch, self.epochs):
            self.train_1_epoch(epoch)
            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    self.valid(epoch)
                    cur_loss = self.log_dict['Loss_tot']
                    if best_loss > cur_loss:
                        best_loss = cur_loss
                        torch.save(self.S_model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_best.pt')

                    # if cnt >= 3:
                    #     # early stop
                    #     break
                torch.save(self.S_model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_last.pt')
            if epoch % self.save_interval == 0:
                with torch.no_grad():
                    self.save_imgs(epoch, self.save_num)

        # move trained model to complete folder 
        try:
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_best.pt', f'{self.save_dir}/completed/{self.log_name}_best.pt')
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_last.pt', f'{self.save_dir}/completed/{self.log_name}_last.pt')
        except Exception as e:
            print_with_timestamp(f"Failed to move {self.save_dir}/not_finished/{self.log_name}.pt: {e}")

    def train_1_epoch(self, epoch):
        self.reset_logs()
        self.S_model.train()
        tot_loss = 0.
        for (MRI, MRI_template, PET, img_segs, temp_segs) in self.train_loader:
            MRI, MRI_template, PET = MRI.unsqueeze(1).cuda(), MRI_template.unsqueeze(1).cuda(), PET.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            # forward & calculate loss in child trainer
            loss, _, _ = self.forward(MRI, MRI_template, PET, img_segs, temp_segs, epoch)
            tot_loss += loss.item()

            # backward & update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        print_with_timestamp(f'Epoch {epoch}: train loss {round(tot_loss/len(self.train_loader), 4)}')

        # log into wandb
        self.log(epoch, phase='train')

    def valid(self, epoch):
        self.reset_logs()
        self.S_model.eval()
        tot_loss = 0.
        for (MRI, MRI_template, PET, img_segs, temp_segs) in self.val_loader:
            MRI, MRI_template, PET = MRI.unsqueeze(1).cuda(), MRI_template.unsqueeze(1).cuda(), PET.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]

            # forward & calculate loss in child trainer
            loss, _, _ = self.forward(MRI, MRI_template, PET, img_segs, temp_segs, epoch)
            tot_loss += loss.item()

        print_with_timestamp(f'Epoch {epoch}: valid loss {round(tot_loss/len(self.val_loader), 4)}')

        self.log(epoch, phase='valid')

    def save_imgs(self, epoch, num):
        self.S_model.eval()
        for idx, (MRI, MRI_template, PET, img_segs, temp_segs) in enumerate(self.save_loader):
            if idx >= num:
                break
            MRI, MRI_template, PET = MRI.unsqueeze(1).cuda(), MRI_template.unsqueeze(1).cuda(), PET.unsqueeze(1).cuda()
            
            # forward & calculate loss in child trainer
            _, deformed_pet, deformed_segs = self.forward(MRI, MRI_template, PET, img_segs, temp_segs, epoch)
            t_disp, s_disp = self.get_disp()

            fig = save_middle_slices_mfm(MRI_template, PET, deformed_pet, epoch, idx)
            self.writer.add_figure(f'deformed_PET(s_disp)_img{idx}', fig, epoch)
            plt.close(fig)

            deformed_MRI = apply_deformation_using_disp(MRI, t_disp)
            fig = save_middle_slices_mfm(MRI_template, MRI, deformed_MRI, epoch, idx)
            self.writer.add_figure(f'deformed_MRI(t_disp)_img{idx}', fig, epoch)
            plt.close(fig)
            
            for name, (img_seg, temp_seg, deformed_seg) in enumerate(zip(img_segs, temp_segs, deformed_segs)):
                fig = save_middle_slices_mfm(img_seg, temp_seg, deformed_seg, epoch, idx)
                self.writer.add_figure(f'deformed_slices_img{idx}_seg{name}', fig, epoch)
                plt.close(fig)

            fig = save_grid_spline(s_disp)
            self.writer.add_figure(f's_disp_{idx}', fig, epoch)
            plt.close(fig)

            fig = save_grid_spline(t_disp)
            self.writer.add_figure(f't_disp_{idx}', fig, epoch)
            plt.close(fig)

        print_with_timestamp(f'Epoch {epoch}: Successfully saved {num} images')

