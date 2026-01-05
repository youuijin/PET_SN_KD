import torch
import torch.nn as nn
import torch.nn.functional as F


class Tiny_U_Net(nn.Module):
    """
    Tiny U-Net with Mid_U_Net-compatible output format
    - Single-scale internally
    - Multi-scale format preserved for future extension
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=3,
        out_layers=1,
    ):
        super().__init__()
        assert out_channels in [3, 6]
        assert out_layers >= 1

        self.out_channels = out_channels
        self.out_layers = out_layers

        base = 8  # tiny 핵심

        # ------------------
        # Encoder
        # ------------------
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(base, base, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv3d(base * 2, base * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # ------------------
        # Decoder
        # ------------------
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base * 4, base * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base * 3, base, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # ------------------
        # Flow head (single scale)
        # ------------------
        self.flow = nn.Conv3d(base, out_channels, 3, padding=1)

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x):

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Decoder
        u2 = self.up2(x3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        out = self.flow(d1)

        # -------------------------------------------------
        # Format compatibility
        # -------------------------------------------------
        if self.out_channels == 6:
            means = []
            stds = []

            mu = out[:, :3]
            sigma = torch.exp(0.5 * out[:, 3:])

            means.append(mu)
            stds.append(sigma)

            # format 유지용 slicing
            means = means[-self.out_layers:]
            stds = stds[-self.out_layers:]

            tot_means = self.combine_residuals(means)
            tot_stds = self.combine_residuals_std(stds)

            return tot_means, tot_stds, means, stds

        else:
            disp = []
            disp.append(out)

            disp = disp[-self.out_layers:]
            tot_disp = self.combine_residuals(disp)

            return tot_disp, disp

    # -------------------------------------------------
    # Residual combination (identical to Mid_U_Net)
    # -------------------------------------------------
    def combine_residuals(self, flows):
        tot_flows = [flows[0]]
        for f in flows[1:]:
            prev = F.interpolate(
                tot_flows[-1],
                size=f.shape[2:],
                mode="trilinear",
                align_corners=True,
            )
            prev *= 2
            tot_flows.append(prev + f)
        return tot_flows

    def combine_residuals_std(self, stds):
        tot_vars = [stds[0]]
        for s in stds[1:]:
            prev = F.interpolate(
                tot_vars[-1],
                size=s.shape[2:],
                mode="trilinear",
                align_corners=True,
            )
            tot_vars.append(torch.sqrt(prev ** 2 + s ** 2))
        return tot_vars


if __name__ == "__main__":
    from torchinfo import summary

    model = Tiny_U_Net(
        in_channels=2,
        out_channels=6,
        out_layers=1,
    ).cuda()

    summary(model, input_size=(1, 2, 160, 192, 160))
