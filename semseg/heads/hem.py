import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from fvcore.nn import flop_count_table, FlopCountAnalysis

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), norm_cfg=None):
        super().__init__()
        self.ham_in = ConvModule(ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = ConvModule(ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)
        return ham

class LightHamHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 320, 512], ham_channels=512, ham_kwargs=dict(), num_classes=25):
        super().__init__()
        self.in_channels = in_channels[0:]
        self.in_index = [0,1,2,3]
        self.ham_channels = self.channels = ham_channels
        self.conv_cfg = None
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.act_cfg = dict(type='ReLU')

        self.ham_channels = ham_channels
        self.squeeze = ConvModule(sum(self.in_channels), self.ham_channels, 1, conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.hamburger = Hamburger(ham_channels, ham_kwargs, self.norm_cfg)
        self.align = ConvModule(self.ham_channels, self.channels, 1, conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        inputs = [inputs[i] for i in self.in_index]

        inputs = [F.interpolate(level, size=inputs[0].shape[2:], mode='bilinear', align_corners=False) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.conv_seg(output)
        return output

if __name__ == '__main__':
    model = LightHamHead(num_classes=25)
    model = model.cuda()
    x = [torch.zeros(1, 64, 256, 256), torch.ones(1, 128, 128, 128), torch.ones(1, 320, 64, 64)*2, torch.ones(1, 512, 32, 32) *3]
    x = [xi.cuda() for xi in x]
    outs = model(x)
    print(model)
    for y in outs:
        print(y.shape)
    print(flop_count_table(FlopCountAnalysis(model, x)))        
