import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        return self.net(inputs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, inputs):
        batch_size, token_count, _ = inputs.shape
        inputs = self.norm(inputs)
        query, key, value = self.to_qkv(inputs).chunk(3, dim=-1)
        query, key, value = map(
            lambda tensor: tensor.view(
                batch_size, token_count, self.heads, -1
            ).transpose(1, 2),
            (query, key, value),
        )
        attention = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, token_count, -1)
        return self.to_out(output)


class BrainViT(nn.Module):
    def __init__(self, num_rois, dim, depth, heads, mlp_dim, dropout=0.5):
        super().__init__()
        self.num_rois = num_rois
        self.dim = dim
        self.to_embedding = nn.Sequential(
            nn.LayerNorm(num_rois), nn.Linear(num_rois, dim), nn.LayerNorm(dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_rois + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, inputs, batch):
        batch_size = int(batch.max().item() + 1)
        inputs = inputs.view(batch_size, self.num_rois, -1)
        inputs = self.to_embedding(inputs)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        inputs = torch.cat((cls_tokens, inputs), dim=1)
        inputs = self.dropout(inputs + self.pos_embedding)
        for attention, feed_forward in self.layers:
            inputs = attention(inputs) + inputs
            inputs = feed_forward(inputs) + inputs
        return inputs[:, 1:, :]


class OCRead(nn.Module):
    def __init__(self, d_model, num_clusters, dropout=0.1):
        super().__init__()
        self.scale = d_model**-0.5
        self.cluster_centers = nn.Parameter(torch.empty(num_clusters, d_model))
        nn.init.orthogonal_(self.cluster_centers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        centers = F.normalize(self.cluster_centers, dim=-1)
        assignment = F.softmax(
            torch.einsum("bnd,kd->bnk", inputs, centers) * self.scale, dim=-1
        )
        clusters = torch.einsum("bnk,bnd->bkd", assignment, inputs)
        return self.dropout(self.norm(clusters))


class TextQueryCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])

    def forward(self, query, graph_tokens):
        residual, _ = self.self_attention(query, query, query)
        query = self.norms[0](query + self.dropouts[0](residual))
        residual, _ = self.cross_attention(query, graph_tokens, graph_tokens)
        query = self.norms[1](query + self.dropouts[1](residual))
        residual = self.feed_forward(query)
        return self.norms[2](query + self.dropouts[2](residual))


class QueryReadout(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.query_score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query):
        global_query = query[:, 0, :]
        weights = torch.softmax(self.query_score(query).squeeze(-1), dim=1)
        pooled_query = torch.einsum("bs,bsd->bd", weights, query)
        gate = self.gate(torch.cat([global_query, pooled_query], dim=-1))
        return self.output_norm(gate * global_query + (1.0 - gate) * pooled_query)


class DiseaseClassifier(nn.Module):
    def __init__(self, dim, output_dim, dropout, hidden_mult):
        super().__init__()
        middle_dim = int(dim * hidden_mult)
        self.layers = nn.Sequential(
            nn.Linear(dim, middle_dim),
            nn.LayerNorm(middle_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(middle_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim, output_dim),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class CITCA(nn.Module):
    """Proposed text-query/fMRI-key-value model."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=1,
        dropout=0.5,
        nhead=8,
        clf_hidden_mult=2,
        vit_depth=1,
        vit_mlp_dim=64,
        qformer_depth=2,
        readout_type="query_readout",
        use_ocread=False,
        ocread_num_clusters=8,
        **_,
    ):
        super().__init__()
        self.fusion_dim = 768
        self.readout_type = readout_type
        self.graph_encoder = BrainViT(
            num_rois=input_dim,
            dim=hidden_dim,
            depth=vit_depth,
            heads=nhead,
            mlp_dim=vit_mlp_dim,
            dropout=dropout,
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, self.fusion_dim), nn.LayerNorm(self.fusion_dim)
        )
        self.feature_dropout = nn.Dropout(dropout)
        self.use_ocread = use_ocread
        if use_ocread:
            self.ocread = OCRead(self.fusion_dim, ocread_num_clusters, dropout)
        self.qformer_blocks = nn.ModuleList(
            [
                TextQueryCrossAttentionBlock(self.fusion_dim, nhead, dropout)
                for _ in range(qformer_depth)
            ]
        )
        if readout_type == "query_readout":
            self.query_readout = QueryReadout(self.fusion_dim, dropout)
        self.classifier = DiseaseClassifier(
            self.fusion_dim, output_dim, dropout, clf_hidden_mult
        )

    def _readout(self, query):
        if self.readout_type == "query_readout":
            return self.query_readout(query)
        if self.readout_type == "cls":
            return query[:, 0, :]
        return query.mean(dim=1)

    def forward(self, x, batch, text_tokens):
        if text_tokens is None:
            raise ValueError("text_tokens must not be None")
        batch_size = int(batch.max().item() + 1)
        text_tokens = text_tokens.view(batch_size, -1, self.fusion_dim)
        text_tokens = self.feature_dropout(text_tokens)

        graph_tokens = self.graph_projection(self.graph_encoder(x, batch))
        if self.use_ocread:
            graph_tokens = torch.cat(
                [self.ocread(graph_tokens), graph_tokens], dim=1
            )

        query = text_tokens
        for block in self.qformer_blocks:
            query = block(query, graph_tokens)
        return self.classifier(self._readout(query))
