import torch
import torch.nn as nn
from torch_geometric.nn.aggr import AttentionalAggregation, MeanAggregation
from torch_geometric.nn import TransformerConv



class Tokenizer(nn.Module):
    def __init__(self, dim, node_classes: list[int], edge_classes: list[int],  method: str = 'cat', dropout: float = 0.2):
        super().__init__()
        self.method = method
        self.dim = dim

        node_size = dim//len(node_classes) if method=='cat' else dim 
        edge_size = dim//len(node_classes) if method=='cat' else dim 

        if self.method == 'cat':
            # all embeddings have classes+1 to account for the augmentations where the features are masked with -1
            self.node_embeddings = nn.ModuleList([
                nn.Embedding(node_classes[i], node_size) 
                for i in range(len(node_classes))
            ])
            self.node_drop = nn.Dropout(dropout)

            self.edge_embeddings = nn.ModuleList([
                nn.Embedding(edge_classes[i], edge_size) 
                for i in range(len(edge_classes))
            ])
            if len(edge_classes) < len(node_classes):
                self.edge_embeddings.append(
                    Mlp(in_features=len(edge_classes), hidden_features=(len(node_classes)-len(edge_classes))*edge_size//2, out_features=(len(node_classes)-len(edge_classes))*edge_size)
                )


        elif self.method == 'sum':
            # all embeddings have classes+1 to account for the augmentations where the features are masked with -1
            self.node_embeddings = nn.ModuleList([
                nn.Embedding(node_classes[i], node_size) 
                for i in range(len(node_classes))           
            ])

            self.edge_embeddings = nn.ModuleList([
                nn.Embedding(edge_classes[i], edge_size) 
                for i in range(len(edge_classes)) 
            ])

        self.node_drop = nn.Dropout(dropout)
        self.edge_drop = nn.Dropout(dropout)


    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        if self.method == 'cat':
            x, edge_attr = self._concat_embedding(x, edge_index, edge_attr)
        elif self.method == 'sum':
            x, edge_attr = self._sum_embedding(x, edge_index, edge_attr)
        return x, edge_attr
    
    
    def _concat_embedding(self, x, edge_index, edge_attr)  -> torch.Tensor:
        x = torch.cat([
            embedding(
                torch.where(
                    x[:, i] == -1,
                    torch.tensor(0, device=x.device),  # Use 0 as padding index
                    (x[:, i] * embedding.num_embeddings).long() + 1
                )
            ) if embedding.num_embeddings > 3 else
            embedding(
                torch.where(
                    x[:, i] == -1,
                    torch.tensor(0, device=x.device),  # Use 0 as padding index
                    (x[:, i]).long() + 1
                )
            )
            for i, embedding in enumerate(self.node_embeddings)
        ], dim=-1)
        x = self.node_drop( x ) 

        edge_attr = torch.cat([
            (
                embedding(
                    torch.where(
                        edge_attr[:, i] == -1,
                        torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                        (edge_attr[:, i] * embedding.num_embeddings).long() +1
                    )
                ) if embedding.num_embeddings > 3
                else embedding(
                    torch.where(
                        edge_attr[:, i] == -1,
                        torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                        edge_attr[:, i].long() + 1
                    )
                )
            ) if isinstance(embedding, nn.Embedding)
            else embedding(edge_attr[:, i].view(-1, 1)) if isinstance(embedding, nn.Linear)
            else embedding(edge_attr[:]) if isinstance(embedding, Mlp)
            else embedding(
                torch.where(
                    edge_attr[:, i] == -1,
                    torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                    edge_attr[:, i].long() + 1
                )
            )
            for i, embedding in enumerate(self.edge_embeddings)
        ], dim=-1)
        edge_attr = self.edge_drop( edge_attr ) 
        return x, edge_attr
    

    def _sum_embedding(self, x, edge_index, edge_attr) -> torch.Tensor:
        x = torch.sum(
            torch.stack(
                [
                    embedding(
                        torch.where(
                            x[:, i] == -1,
                            torch.tensor(0, device=x.device),  # Use 0 as padding index
                            (x[:, i] * embedding.num_embeddings).long() + 1
                        )
                    ) if embedding.num_embeddings > 3 else
                    embedding(
                        torch.where(
                            x[:, i] == -1,
                            torch.tensor(0, device=x.device),  # Use 0 as padding index
                            (x[:, i]).long() + 1
                        )
                    )
                    for i, embedding in enumerate(self.node_embeddings)
                ], dim = 0
            ), dim = 0
        )
        x = self.node_drop( x ) 

        edge_attr = torch.sum(
            torch.stack(
                [
                    (
                        embedding(
                            torch.where(
                                edge_attr[:, i] == -1,
                                torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                                (edge_attr[:, i] * embedding.num_embeddings).long() +1
                            )
                        ) if embedding.num_embeddings > 3
                        else embedding(
                            torch.where(
                                edge_attr[:, i] == -1,
                                torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                                edge_attr[:, i].long() + 1
                            )
                        )
                    ) if isinstance(embedding, nn.Embedding)
                    else embedding(edge_attr[:, i].view(-1, 1)) if isinstance(embedding, nn.Linear)
                    else embedding(edge_attr[:]) if isinstance(embedding, Mlp)
                    else embedding(
                        torch.where(
                            edge_attr[:, i] == -1,
                            torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                            edge_attr[:, i].long() + 1
                        )
                    )
                    for i, embedding in enumerate(self.edge_embeddings)
                ], dim = 0
            ), dim = 0
        )
        edge_attr = self.edge_drop( edge_attr )
        return x, edge_attr



class Mlp(nn.Module):
    """
    Traditional MLP implementation
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='GELU', norm_layer=None, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if norm_layer != None:
            self.norm = getattr(nn, norm_layer)(hidden_features)
        self.act = getattr(nn, act_layer)()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, 'norm_layer'):
            x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



# TODO check if it is actually used, who is setting self.training?
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)




class Attention(nn.Module):
    """
    Attention block based on TransformerConv PyG implementation
    """
    def __init__(self, dim=128, n_heads=2, attn_drop=0.25, proj_drop=0.25, beta=False, concat=True):
        super().__init__()

        self.attention = TransformerConv(
            in_channels=dim, 
            out_channels = dim // n_heads,
            heads = n_heads, 
            dropout = attn_drop,
            edge_dim = dim, 
            beta = beta, 
            concat = concat
        )
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
            attn = self.attention(x, edge_index, edge_attr)
            x = self.drop(self.proj(attn[0]))
            return x
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config['dim']
        n_heads = config['n_heads']
        norm = config['norm']
        dropout = config['dropout']
        act = config['activation']

        # layers
        self.attn = Attention(dim, n_heads=n_heads, attn_drop=dropout, proj_drop=dropout)

        self.norm1 = getattr(nn, norm)(dim)
        self.norm2 = getattr(nn, norm)(dim)

        self.drop_path = DropPath(config['drop_path']) if config['drop_path'] > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * config['mlp_ratio'])
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act, drop=dropout)


    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        x = x + self.drop_path( self.attn( self.norm1(x), edge_index, edge_attr ) )
        x = x + self.drop_path( self.mlp( self.norm2(x) ) )
        return x
    


class Encoder(torch.nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.config = config

        self.dim = config['dim']
        self.n_layers = config['n_layer']
        self.dropout = config['dropout']
        self.drop_path = DropPath(drop_prob=config['drop_path']) if config['drop_path'] > 0. else nn.Identity()
        self.act = getattr(nn, config['act'])()
        self.mlp_ratio = config['mlp_ratio']
        self.use_global_bool = True if config['global_pool'] is not None else False

        # setup layers
        self.embedding = Tokenizer(config['tokenizer'])

        self.layers = nn.ModuleDict()
        for i in range(self.n_layers):
            self.layers[f'layer_{i+1}'] = TransformerBlock(config)

        if self.use_global_bool:
            self._setup_attentional_aggr()


        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


    def _setup_attentional_aggr(self):
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        gate = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, out_features=self.dim, act_layer=self.act, drop=self.dropout)
        nn = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, out_features=self.dim, act_layer=self.act, drop=self.dropout)

        self.global_pool = AttentionalAggregation(gate, nn)


    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        x, edge_index, edge_attr = self.tokenizer(x, edge_index, edge_attr) 

        for _, layer in self.layers:
            x, edge_index, edge_attr = layer(x, edge_index, edge_attr)

        if self.use_global_bool:
            x = self.global_pool(x, batch)

        return x