import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import LightHamHead
from semseg.models.heads import UPerHead
from semseg.models.heads import Turbo_decoder
from fvcore.nn import flop_count_table, FlopCountAnalysis


class FACENet(BaseModel):
    def __init__(self, backbone: str = 'FACENet-B4', num_classes: int = 9, modals: list = ['img', 'ther']) -> None:
        super().__init__(backbone, num_classes, modals)
        self.decode_head = LightHamHead(in_channels=self.backbone.channels, ham_channels=512, num_classes=num_classes)
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        return y

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if self.backbone.num_modals > 0:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                if 'model' in checkpoint.keys():
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)

def load_dualpath_model(model, model_file):
    extra_pretrained = None
    if isinstance(extra_pretrained, str):
        raw_state_dict_ext = torch.load(extra_pretrained, map_location=torch.device('cpu'))
        if 'state_dict' in raw_state_dict_ext.keys():
            raw_state_dict_ext = raw_state_dict_ext['state_dict']
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys(): 
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v

    if isinstance(extra_pretrained, str):
        for k, v in raw_state_dict_ext.items():
            if k.find('patch_embed1.proj') >= 0:
                state_dict[k.replace('patch_embed1.proj', 'extra_downsample_layers.0.proj.module')] = v 
            if k.find('patch_embed2.proj') >= 0:
                state_dict[k.replace('patch_embed2.proj', 'extra_downsample_layers.1.proj.module')] = v 
            if k.find('patch_embed3.proj') >= 0:
                state_dict[k.replace('patch_embed3.proj', 'extra_downsample_layers.2.proj.module')] = v 
            if k.find('patch_embed4.proj') >= 0:
                state_dict[k.replace('patch_embed4.proj', 'extra_downsample_layers.3.proj.module')] = v 
            
            if k.find('patch_embed1.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed1.norm', 'extra_downsample_layers.0.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed2.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed2.norm', 'extra_downsample_layers.1.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed3.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed3.norm', 'extra_downsample_layers.2.norm.ln_{}'.format(i))] = v 
            if k.find('patch_embed4.norm') >= 0:
                for i in range(model.num_modals):
                    state_dict[k.replace('patch_embed4.norm', 'extra_downsample_layers.3.norm.ln_{}'.format(i))] = v 
            elif k.find('block') >= 0:
                state_dict[k.replace('block', 'extra_block')] = v
            elif k.find('norm') >= 0:
                state_dict[k.replace('norm', 'extra_norm')] = v


    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict

if __name__ == '__main__':
    modals = ['img', 'ther']
    model = FACENet('FACENet-B4', 9, modals)
    model.init_pretrained('../../checkpoints/pretrained/segformer/mit_b4.pth')
    x = [torch.zeros(1, 3, 480, 640), torch.ones(1, 1, 480, 640)]
    y = model(x)
    print(y.shape)
