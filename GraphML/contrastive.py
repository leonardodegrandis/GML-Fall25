import torch
import torch.nn as nn
from .encoder import Encoder



class Head(nn.Module):
    def __init__(self, in_dim=128, out_dim=64, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=64, bottleneck_dim=64, drop_rate=0.2):
        super().__init__()
        self.out_dim = out_dim
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        # the following five lines only work with recent versions of torch D:
        #self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        #with torch.no_grad():
        #    self.last_layer.parametrizations.weight.original0.fill_(1)
        #if norm_last_layer:
        #    self.last_layer.parametrizations.weight.original0.requires_grad = False

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    



class Network(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.config=config
        self.encoder = Encoder(model=config['model'], layer=config['encoder']['layer'], dim=config['encoder']['dim'], num_layers=config['encoder']['n_layers'], num_heads=config['encoder']['n_heads'], 
                               mlp_ratio=config['encoder']['mlp_ratio'], drop_rate=config['encoder']['drop_rate'], attn_drop_rate=config['encoder']['attn_drop_rate'], 
                               embedding = config['embedding'], norm_layer=config['encoder']['norm_layer'], act_layer=config['encoder']['act_layer'], batch_size=config['loader']['batch_size'], dataset=config['dataset'])
        
        self.head = Head(in_dim=config['encoder']['dim'], out_dim=config['head']['out_dim'], use_bn=config['head']['use_bn'], norm_last_layer=config['head']['norm_last_layer'], 
                         nlayers=config['head']['n_layers'], hidden_dim=config['head']['hidden_dim'], bottleneck_dim=config['head']['bottleneck_dim'], drop_rate=config['encoder']['drop_rate'])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr if isinstance(data.x,torch.Tensor) else None
        x = self.encoder(x, edge_index, edge_attr)
        x = self.head(x)
        return x
        

class GraphCL(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.branch1 = Network(self.config)
        self.branch2 = Network(self.config)
        

    def forward(self, data, mode=1):
        if mode == 1:
            branch1_out = self.branch1(data)
            return branch1_out
        elif mode == 2:
            branch2_out = self.branch2(data)
            return branch2_out 