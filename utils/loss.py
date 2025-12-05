import torch
class KD_loss:
    def __init__(self, KD_mode='none', lamb=1.0):
        self.KD_mode = KD_mode
        self.lamb = lamb

    def __call__(self, s_displace, t_displace, moving_img, fixed_img, moved_seg, fixed_seg):
        if self.KD_mode == 'none':
            KD_loss = 0.0
        elif self.KD_mode == 'field':
            # Knowledge Distillation on deformation fields
            KD_loss = self.field_KD_loss(s_displace, t_displace)
            # KD_loss = loss
        # elif self.KD_mode == 'feature':
        #     # Knowledge Distillation on intermediate features
        #     loss = 0.0
        #     for s_feat, t_feat in zip(student_outputs['features'], teacher_outputs['features']):
        #         loss += ((s_feat - t_feat) ** 2).mean()
        #     return loss
        else:
            raise ValueError(f"Unknown KD mode: {self.KD_mode}")

        SUVr_loss = self.SUVr_consistency_loss(moving_img, fixed_img, moved_seg, fixed_seg)

        return KD_loss + self.lamb * SUVr_loss, KD_loss.item(), SUVr_loss.item()

    # Distillation loss
    def field_KD_loss(self, s_displace, t_displace):
        '''
        Calcuate MSE loss between student and teacher deformation fields
        - s_displace: [B, 3, D, H, W], predicted displacement field by student network
        - t_displace: [B, 3, D, H, W], predicted displacement field by teacher network
        '''
        return ((s_displace - t_displace) ** 2).mean()

    # def image_KD_loss(self, student_image, teacher_image):
    #     return ((student_image - teacher_image) ** 2).mean()

    # Student loss
    def SUVr_consistency_loss(self, moving_img, fixed_img, moved_seg, fixed_seg):
        '''
        Calcuate SUVr
        - moved_seg: [N, (B, C, D, H, W)], list of subject segmentations before spatial normalization
        - fixed_seg: [N, (B, C, D, H, W)], list of template segmentations
        '''
        m_list, f_list = [], []
        for m_seg, f_seg in zip(moved_seg[:-1], fixed_seg[:-1]):  # exclude reference region
            m_mean = moving_img[m_seg > 0].mean()
            f_mean = fixed_img[f_seg > 0].mean()
            m_list.append(m_mean)
            f_list.append(f_mean)

        # calculate reference SUV
        ref_m_mean = moving_img[moved_seg[-1] > 0].mean()
        ref_f_mean = fixed_img[fixed_seg[-1] > 0].mean()

        student_SUVr = torch.stack([m / ref_m_mean for m in m_list], dim=0).mean()
        teacher_SUVr = torch.stack([f / ref_f_mean for f in f_list], dim=0).mean()

        return ((student_SUVr - teacher_SUVr) ** 2).mean()

    # def deformation_smoothness_loss(self, displacement_field):
    #     dx = torch.abs(displacement_field[:, :, 1:, :, :] - displacement_field[:, :, :-1, :, :])
    #     dy = torch.abs(displacement_field[:, :, :, 1:, :] - displacement_field[:, :, :, :-1, :])
    #     dz = torch.abs(displacement_field[:, :, :, :, 1:] - displacement_field[:, :, :, :, :-1])
    #     return (dx.mean() + dy.mean() + dz.mean()) / 3.0