import torch
import torch.nn.functional as F

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


class Direct_loss:
    def __init__(self, sim_loss='None', alpha_tv=1.0, alpha_dice=1.0, alpha_suvr=1.0):
        # similarity = paired MRI
        if sim_loss == 'MSE':
            self.loss_fn_sim = self.MSE_loss
        elif sim_loss == 'NCC': 
            self.loss_fn_sim = self.NCC_loss
        elif sim_loss == 'None':
            self.loss_fn_sim = self.none_fn

        self.loss_fn_tv = self.tv_loss_l2
        self.loss_fn_dice = self.dice_loss
        self.loss_fn_suvr = self.suvr_ratio_consistency_loss
        self.alpha_tv = alpha_tv
        self.alpha_dice = alpha_dice
        self.alpha_suvr = alpha_suvr

    def MSE_loss(self, x, y):
        mse = torch.mean(torch.norm(x - y, dim=1, p=2) ** 2)
        return mse

    def NCC_loss(self, x, y, eps=1e-8):
        """
        Calculate the normalized cross correlation (NCC) between two tensors.
        Args:
        - x (torch.Tensor): Deformed image tensor (B, 1, D, H, W)
        - y (torch.Tensor): Template image tensor (B, 1, D, H, W)
        - eps (float): Small constant to avoid division by zero.

        Returns:
        - NCC loss value (torch.Tensor)
        """
        x_mean = torch.mean(x, dim=[2,3,4], keepdim=True).to(x.device)
        y_mean = torch.mean(y, dim=[2,3,4], keepdim=True).to(y.device)

        x = x - x_mean
        y = y - y_mean

        numerator = torch.sum(x * y, dim=[2,3,4])
        denominator = torch.sqrt(torch.sum(x ** 2, dim=[2,3,4]) * torch.sum(y ** 2, dim=[2,3,4]) + eps)

        ncc = numerator / denominator
        return -torch.mean(ncc)  # Maximize NCC by minimizing the negative value

    def none_fn(self, moved_pet, fixed_pet):
        return torch.tensor(0.0).to(moved_pet.device)

    def tv_loss_l2(self, displace):
        """
        displace: Tensor of shape [B, 3, D, H, W]
        TV loss는 인접 voxel 간의 L2 차이의 평균을 구하는 방식입니다.
        """
        # Depth 방향 차이 (D axis)
        dz = torch.mean((displace[:, :, 1:, :, :] - displace[:, :, :-1, :, :])**2)
        # Height 방향 차이 (H axis)
        dy = torch.mean((displace[:, :, :, 1:, :] - displace[:, :, :, :-1, :])**2)
        # Width 방향 차이 (W axis)
        dx = torch.mean((displace[:, :, :, :, 1:] - displace[:, :, :, :, :-1])**2)

        loss = (dx + dy + dz)/3.
        return loss.mean()

    def dice_loss(self, seg_after, seg_temp, eps=1e-6, normalize=True):
        """
        seg_after: # bilinear로 워핑된 연속값 텐서 (0~1 권장)
        seg_temp : # 템플릿 마스크 (하드 or 소프트 모두 OK)
        반환: 1 - 평균 Soft Dice (모든 region에 대한 평균)

        지원 shape: 각 텐서 [B,1,D,H,W] 또는 [B,D,H,W]
        """

        def to_float_wo_channel(x):
            # [B,1,D,H,W] -> [B,D,H,W]
            if x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            return x.to(torch.float32)

        # 1) 입력 정리
        sa_1, sa_2, sa_3, sa_4, sa_5, sa_6 = map(to_float_wo_channel, seg_after)  # 연속값 (soft)
        st_1, st_2, st_3, st_4, st_5, st_6 = map(to_float_wo_channel, seg_temp)   # 하드(0/1) 또는 연속값

        # 2) 채널 결합: [B,6,D,H,W]
        pred = torch.stack([sa_1, sa_2, sa_3, sa_4, sa_5, sa_6], dim=1)
        tgt  = torch.stack([st_1, st_2, st_3, st_4, st_5, st_6], dim=1)

        # # 3) (선택) 예측 확률 재정규화: 채널합이 1 되도록 (겹침/틈 방지용)
        # if normalize:
        #     pred = pred.clamp(0, 1)
        #     pred = pred / (pred.sum(dim=1, keepdim=True) + eps)

        # 4) Soft Dice (클래스별 평균 후 다시 평균) — BG 없음(GM/WM 두 채널만)
        dims = (2, 3, 4)
        inter = (pred * tgt).sum(dim=dims)                                   # [B,6]
        denom = (pred.pow(2).sum(dim=dims) + tgt.pow(2).sum(dim=dims))       # [B,6]
        dice  = (2.0 * inter + eps) / (denom + eps)                          # [B,6]
        dice_loss  = 1.0 - dice.mean()                                       # scalar

        return dice_loss

    def suvr_ratio_consistency_loss(self, seg_before, img_before, seg_temp, img_after, eps=1e-6, detach_ref=False):
        def _to_BDHW(x):
            # [B, 1, D, H, W] → [B, D, H, W], 이미 [B, D, H, W]면 그대로
            if x.ndim == 5 and x.shape[1] == 1:
                return x[:, 0]
            return x

        def mean_within(mask, img):
            img = _to_BDHW(img)
            mask = _to_BDHW(mask)
            # 마스크 이진화(마스크는 보통 레이블/확률이며, 여기서는 >0을 내부영역으로 사용)
            m = (mask > 0).to(img.dtype)
            # [B, ...] → [B, N]으로 펴서 합
            num = (img * m).flatten(1).sum(1)             # [B]
            den = m.flatten(1).sum(1)                     # [B]
            return num / (den + eps)                      # [B]

        # 참조 영역 평균 (리스트의 마지막 레벨 사용)
        ref_before = mean_within(seg_before[-1], img_before)   # [B]
        ref_after  = mean_within(seg_temp[-1],   img_after)    # [B]
        if detach_ref:
            ref_before = ref_before.detach()
            ref_after  = ref_after.detach()

        # 각 스케일(마지막 제외)에서 평균을 계산해 쌓음 → [S-1, B]
        b_means = torch.stack([mean_within(sb, img_before) for sb in seg_before[:-1]], dim=0)
        a_means = torch.stack([mean_within(st, img_after) for st in seg_temp[:-1]],  dim=0)

        # SUVR = 영역평균 / 참조영역평균
        ref_before = ref_before.unsqueeze(0)  # [1, B]로 브로드캐스트
        ref_after  = ref_after.unsqueeze(0)   # [1, B]
        r_b = b_means / (ref_before + eps)    # [S-1, B]
        r_a = a_means / (ref_after  + eps)    # [S-1, B]

        # L1 일관성 손실(스케일, 배치 평균)
        return F.l1_loss(r_b, r_a, reduction='mean')

    def __call__(self, disp, moving_pet, moved_pet, moving_seg, moved_seg, fixed_seg, moved_mri, fixed_mri):
        '''
        - seg shape: [2, [B, 1, H, W, D]]
        - moving_pet, moving_seg: original
        - moved_pet, moved_seg: deformed
        - fixed_pet, fixed_seg: PET Template
        - disp: displacement field (for regularizer)
        '''
        sim_loss = self.loss_fn_sim(moved_mri, fixed_mri)
        tv_loss = self.loss_fn_tv(disp)
        dice_loss = self.loss_fn_dice(moved_seg, fixed_seg)
        suvr_loss = self.loss_fn_suvr(moving_seg, moving_pet, fixed_seg, moved_pet)

        tot_loss = sim_loss + self.alpha_tv*tv_loss + self.alpha_dice*dice_loss + self.alpha_suvr*suvr_loss

        return tot_loss, sim_loss.item(), tv_loss.item(), dice_loss.item(), suvr_loss.item()