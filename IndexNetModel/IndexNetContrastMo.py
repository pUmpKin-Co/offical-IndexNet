import math
import torch
import torch.nn as nn
from backbone import resnet
from einops import rearrange
import torch.nn.functional as F


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


def conv_loss(pred, feat, mask1, mask2, spatial_dimention):
    C = pred.shape[1]
    pred = pred.contiguous().view(-1, spatial_dimention * spatial_dimention, C)
    feat = feat.contiguous().view(-1, spatial_dimention * spatial_dimention, C)
    B, grid_size = pred.shape[0], pred.shape[1]
    if pred.shape[2] != mask1.shape[1]:
        mask1 = F.interpolate(mask1.unsqueeze(1), size=spatial_dimention, mode="nearest").squeeze().view(B, -1)
        mask2 = F.interpolate(mask2.unsqueeze(1), size=spatial_dimention, mode="nearest").squeeze().view(B, -1)
    else:
        mask1 = mask1.view(B, -1)
        mask2 = mask2.view(B, -1)

    def make_same_obj(mask1, mask2):
        same_obj = torch.eq(mask1.contiguous().view(B, grid_size, 1),
                               mask2.contiguous().view(B, 1, grid_size))
        return same_obj

    same_obj = make_same_obj(mask1, mask2)

    pred = F.normalize(pred, dim=-1)
    feat = F.normalize(feat, dim=-1)
    logits = torch.bmm(pred, feat.permute(0, 2, 1))

    # BYOL Loss
    loss = torch.masked_select(logits, same_obj)
    loss = (2 - 2 * loss).mean()

    return loss


class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )


class PyramidConvMLP(nn.Module):
    def __init__(self, hidden_dim_list, out_dim, mode="projector"):
        super(PyramidConvMLP, self).__init__()
        assert mode in ["projector", "predictor"]
        self.head = nn.ModuleList()
        self.mode = mode

        if self.mode == "projector":
            for hidden_dim in hidden_dim_list:
                layer = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, out_dim)
                )
                self.head.append(layer)
        else:
            for _ in range(len(hidden_dim_list)):
                layer = nn.Sequential(
                    nn.Linear(out_dim, out_dim * 2),
                    nn.BatchNorm1d(out_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_dim * 2, out_dim)
                )
                self.head.append(layer)

    def forward(self, feature_list):
        out = []

        if not isinstance(feature_list, list):
            feature_list = [feature_list]

        for feature, head in zip(feature_list, self.head):
            out.append(head(feature))
        return out


class MLP(nn.Module):
    def __init__(self, hidden_dim, out_dim, mode="projector"):
        super(MLP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mode = mode
        if self.mode == "projector":
            self.net = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.BatchNorm1d(out_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim * 2, out_dim)
            )

    def forward(self, feature_list):
        if self.mode == "projector":
            feature = self.avgpool(feature_list[0]).squeeze()
        else:
            feature = feature_list
        return self.net(feature)


class NetWrapper(nn.Module):
    def __init__(self,
                 net,
                 hidden_dim_list,
                 out_dim,
                 backbone_stage=[1, 2, 3, 4],
                 no_index_head=False):
        super(NetWrapper, self).__init__()
        self.hidden_dim = hidden_dim_list
        self.out_dim = out_dim
        self.backbone_stage = backbone_stage
        self.no_index_head = no_index_head

        self.net = resnet.__dict__[net](outstride=32, pretrained=False)
        if no_index_head:
            self.projector = MLP(self.hidden_dim[0], self.out_dim, mode="projector")
        else:
            self.projector = PyramidConvMLP(hidden_dim_list=hidden_dim_list,
                                            out_dim=out_dim,
                                            mode="projector")

    def forward(self, img):
        feature_list = self.net(img)
        out_feature = []
        for i in self.backbone_stage:
            out_feature.append(feature_list[i - 1])

        if self.no_index_head:
            return self.projector(out_feature)

        for stage, feature in enumerate(out_feature):
            feature = rearrange(feature, "b c h w -> (b h w) c")
            out_feature[stage] = feature

        out_feature = self.projector(out_feature)
        return out_feature


class IndexNet(nn.Module):
    def __init__(self, config):
        super(IndexNet, self).__init__()
        self.no_pyd = config.network.no_pyd
        self.no_index = config.network.no_index

        self.lr = config.schedule.base_weight_lr
        self.base_momentum = config.network.base_momentum
        self.final_momentum = config.network.final_momentum

        if self.no_pyd:
            self.online_encoder = NetWrapper(net=config.network.backbone,
                                             hidden_dim_list=[config.nework.head_hidden_dim[-1]],
                                             out_dim=config.network.head_out_dim,
                                             backbone_stage=[4]
                                             )
            self.momentum_encoder =NetWrapper(net=config.network.backbone,
                                             hidden_dim_list=[config.nework.head_hidden_dim[-1]],
                                             out_dim=config.network.head_out_dim,
                                             backbone_stage=[4]
                                             )
            self.predictor = PyramidConvMLP(config.network.head_hidden_dim,
                                            config.network.head_out_dim,
                                            mode="predictor")
        elif self.no_index:
            self.online_encoder = NetWrapper(net=config.network.backbone,
                                             hidden_dim_list=[config.nework.head_hidden_dim[-1]],
                                             out_dim=config.network.head_out_dim,
                                             backbone_stage=[4],
                                             no_index_head=True
                                             )
            self.momentum_encoder =NetWrapper(net=config.network.backbone,
                                             hidden_dim_list=[config.nework.head_hidden_dim[-1]],
                                             out_dim=config.network.head_out_dim,
                                             backbone_stage=[4],
                                              no_index_head=True,
                                             )
            self.predictor = MLP(None, config.network.head_out_dim, mode="predictor")
        else:
            self.online_encoder = NetWrapper(net=config.network.backbone,
                                            hidden_dim_list=config.network.head_hidden_dim,
                                            out_dim=config.network.head_out_dim
                                             )
            self.momentum_encoder = NetWrapper(net=config.network.backbone,
                                            hidden_dim_list=config.network.head_hidden_dim,
                                            out_dim=config.network.head_out_dim
                                            )
            self.predictor = PyramidConvMLP(config.network.head_hidden_dim,
                                            config.network.head_out_dim,
                                            mode="predictor")

        initialize_momentum_params(self.online_encoder, self.momentum_encoder)
        self.ema = MomentumUpdater(self.base_momentum, self.final_momentum)

    @property
    def learnable_params(self):
        return [
            {"name": "online_encoder", "params": self.online_encoder.parameters(), "lr": self.lr},
            {"name": "predictor", "params": self.predictor.parameters(), "lr": self.lr * 10, "static_lr": True}
        ]

    @property
    def momentum_pairs(self):
        return [(self.online_encoder, self.momentum_encoder)]

    @torch.no_grad()
    def momentum_forward(self, img1, img2):
        feature_list_k1 = self.momentum_encoder(img1)
        feature_list_k2 = self.momentum_encoder(img2)

        return feature_list_k1, feature_list_k2

    def forward(self, img1, img2):
        feature_list_q1 = self.predictor(self.online_encoder(img1))
        feature_list_q2 = self.predictor(self.online_encoder(img2))

        feature_list_k1, feature_list_k2 = self.momentum_forward(img1, img2)
        return feature_list_q1, feature_list_q2, feature_list_k1, feature_list_k2

