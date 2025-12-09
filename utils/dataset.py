import csv, random
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import numpy as np

def set_dataloader_usingcsv(dataset, csv_dir, dirs, template_path, batch_size, numpy=False, return_path=False, transform=False):
    train_file = f'{csv_dir}/{dataset}/{dataset}_train.csv'
    valid_file = f'{csv_dir}/{dataset}/{dataset}_valid.csv'

    train_dataset = MedicalImageDatasetCSV(csv_file=train_file, dirs=dirs, template_path=template_path, numpy=numpy, return_path=return_path, transform=transform)
    val_dataset = MedicalImageDatasetCSV(csv_file=valid_file, dirs=dirs, template_path=template_path, numpy=numpy, return_path=return_path, transform=False)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader

# Define dataset class
class MedicalImageDatasetCSV(Dataset):
    def __init__(self, csv_file, dirs, template_path, numpy=False, transform=False, return_path=False):
        ext = '.npy' if numpy else '.nii.gz'
        
        with open(csv_file, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))[1:]
        self.MRI_paths = [f"{dirs[0]}/{s[0]}_T1w_MRI{ext}" for s in rows]
        self.PET_paths = [f"{dirs[1]}/core_{s[0]}_FDG_PET{ext}" for s in rows]
        self.seg_dirs = [f"data/FDG_label_cortex_mask/{s[0]}_T1w_MRI" for s in rows]

        self.numpy = numpy
        if numpy:
            template = np.load(template_path)
        else:
            template = nib.load(template_path).get_fdata().astype(np.float32)
        
        # Template normalize - percentile
        t_data = template.flatten()
        p1_temp = np.percentile(t_data, 1)
        p99_temp = np.percentile(t_data, 99)
        template = np.clip(template, p1_temp, p99_temp)
        
        template_min, template_max = template.min(), template.max()
        self.template = (template - template_min) / (template_max - template_min)

        self.transform = transform
        self.return_path = return_path

        # self.augment = IntensityAug()

        # load segments
        self.temp_seg = []
        for i in range(6):
            seg_path = f"data/FDG_label_cortex_mask/template_T1w_MRI/mask{i+1}.nii.gz"
            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            self.temp_seg.append(torch.from_numpy(seg))

    def __len__(self):
        return len(self.MRI_paths)
    
    def __getitem__(self, idx):
        if self.numpy:
            MRI = np.load(self.MRI_paths[idx]).astype(np.float32)
            PET = np.load(self.PET_paths[idx]).astype(np.float32)
            affine = np.load('data/affine.npy')
        else:
            MRI = nib.load(self.MRI_paths[idx]).get_fdata().astype(np.float32)
            PET = nib.load(self.PET_paths[idx]).get_fdata().astype(np.float32)
            affine = nib.load(self.MRI_paths[idx]).affine

        MRI_min, MRI_max = MRI.min(), MRI.max()
        MRI = (MRI - MRI_min) / (MRI_max - MRI_min)  # Normalize to [0,1]#
        MRI = torch.from_numpy(MRI)

        PET_min, PET_max = PET.min(), PET.max()
        PET = (PET - PET_min) / (PET_max - PET_min)  # Normalize to [0,1]#
        PET = torch.from_numpy(PET)

        # Load segments
        img_seg = []
        for i in range(6):
            seg_path = f"{self.seg_dirs[idx]}/mask{i+1}.nii.gz"
            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            img_seg.append(torch.from_numpy(seg))

        # if self.transform:
        #     img = self.augment(img, geo=False)

        # return format
        if self.return_path:
            return MRI, torch.from_numpy(self.template), PET, img_seg, self.temp_seg, self.MRI_paths[idx]
        else:
            return MRI, torch.from_numpy(self.template), PET, img_seg, self.temp_seg