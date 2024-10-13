import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    """ Build a correlation pyramid. Each layer is an all pairs correlation between 2 sets of features map. """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius  # radius to find correlation.
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        # maybe author does this to fit dimension of avg_pool2d
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor):  # coords: [B, 2, H1, W1], 2 are order of cols, rows
        """Pull out the correlation values for each position in the coords grid."""
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)    # coords: [B, H1, W1, 2]
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # [B*H1*W1, 1, H2, W2]
            # [-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.] for r=4
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)  # [R]
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl  # [B*H1*W1, R, R, 2]

            # TODO: read bilinear_sampler
            # [B*H1*W1, 1, H2, W2] and [B*H1*W1, R, R, 2] -> [B*H1*W1, 1, R, R]
            corr_samples = bilinear_sampler(corr, coords_lvl)  # [B*H1*W1, 1, R, R]
            corr_vw = corr_samples.view(batch, h1, w1, -1)     # [B, H1, W1, R*R]
            out_pyramid.append(corr_vw)

        # Combine the correlation for each pixel at each level.
        out = torch.cat(out_pyramid, dim=-1)  # [B, H1, W1, (R*R)*num_levels]
        # R*R*num_levels is the correlation windows per each abstraction pyramid level.
        return out.permute(0, 3, 1, 2).contiguous().float()  # [B, (R*R)*num_levels, H1, W1]

    @staticmethod
    def corr(fmap1, fmap2):  # fmap1, fmap2: [B, N-features, H, W] -> [B, N-features, H, W, 1, H, W]
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)  # fmap1: [B, N-features, M] M is H*W
        fmap2 = fmap2.view(batch, dim, ht*wd)
        
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)  # corr: B x [M, N] x [N, M] = [B, M, M]
        # Decompse the M dimension back into H, W
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords: torch.Tensor):  # coords: [B, 2, H, W]
        coords = coords.permute(0, 2, 3, 1)    # coords: [B, H, W, 2]
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
