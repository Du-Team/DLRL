import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.vit import ViT


class AdaptiveFeatureFusion(nn.Module):
    """SAMVGC-inspired feature-adaptive cross-view fusion."""

    def __init__(self, n_modalities, channels, hidden_dim=64,
                 min_weight=0.0, temperature=1.0,
                 preserve_scale=False, uniform_init=False):
        super().__init__()
        self.n_modalities = n_modalities
        self.min_weight = float(min_weight)
        self.temperature = float(temperature)
        self.preserve_scale = bool(preserve_scale)
        if not 0.0 <= self.min_weight < 1.0 / max(1, n_modalities):
            raise ValueError(
                "min_weight must be in [0, 1 / n_modalities)"
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.view_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(self._num_groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(self._num_groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.cross_view_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if uniform_init:
            # Start the Augsburg three-view run without an arbitrary modality
            # preference. The gate remains fully trainable after this neutral
            # initialization.
            nn.init.zeros_(self.cross_view_gate[-1].weight)
            nn.init.zeros_(self.cross_view_gate[-1].bias)
        self.last_reliability = None

    @staticmethod
    def _num_groups(channels):
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def forward(self, modal_features):
        b, m, c, h, w = modal_features.shape
        if m != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} modalities, got {m}.")

        view_tokens = self.view_encoder(modal_features.reshape(b * m, c, h, w)).reshape(b, m, -1)
        view_context = view_tokens.mean(dim=1, keepdim=True).expand_as(view_tokens)
        gate_input = torch.cat([
            view_tokens,
            view_context,
            view_tokens - view_context,
        ], dim=-1)
        logits = self.cross_view_gate(gate_input).squeeze(-1)
        if self.temperature == 1.0:
            # Preserve the original Trento/MUUFL computation exactly.
            weights = torch.softmax(logits, dim=1)
        else:
            weights = torch.softmax(logits / self.temperature, dim=1)
        if self.min_weight > 0:
            free_mass = 1.0 - self.n_modalities * self.min_weight
            weights = self.min_weight + free_mass * weights
        self.last_reliability = weights
        if self.preserve_scale:
            weights_for_features = float(self.n_modalities) * weights
        else:
            weights_for_features = weights
        return modal_features * weights_for_features.view(b, m, 1, 1, 1)


class MultiModalDecoupleLayer(nn.Module):
    """Extract common, private and mixed information from every view."""

    def __init__(self, n_modalities, channel_per_modality, group_channels):
        super().__init__()
        self.n_modalities = n_modalities
        common_channels, private_channels, mixed_channels = group_channels
        self.common_extractor = self._feature_block(channel_per_modality, common_channels, 3)
        self.private_extractors = nn.ModuleList([
            self._feature_block(channel_per_modality, private_channels, 3)
            for _ in range(n_modalities)
        ])
        self.mix_extractor = self._feature_block(channel_per_modality, mixed_channels, 5)
        total_channels = sum(group_channels) * n_modalities
        self.reproject = nn.Conv2d(total_channels, channel_per_modality, kernel_size=1)
        self.out_norm = nn.BatchNorm2d(channel_per_modality)

    @staticmethod
    def _num_groups(channels):
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    @classmethod
    def _feature_block(cls, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(cls._num_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(cls._num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        pieces = []
        for modality_idx in range(self.n_modalities):
            feat = x[:, modality_idx]
            pieces.append(torch.cat([
                self.common_extractor(feat),
                self.private_extractors[modality_idx](feat),
                self.mix_extractor(feat),
            ], dim=1))
        return self.out_norm(self.reproject(torch.cat(pieces, dim=1)))


class PatchEmbedding(nn.Module):
    """
    transform different modalities into the same dim
    """

    def __init__(self, n_modalities, in_channels, out_channel, image_size,
                 use_reliability_fusion=False,
                 reliability_min_weight=0.0,
                 reliability_temperature=1.0,
                 reliability_preserve_scale=False,
                 reliability_uniform_init=False):
        """
        :param n_modalities: number of modalities
        :param in_channels: tuple of input channels of multiple modalities, or a single modality
        :param out_channel:
        """
        super(PatchEmbedding, self).__init__()
        self.n_modalities = n_modalities
        self.out_channel = out_channel
        self.use_reliability_fusion = use_reliability_fusion
        self.last_modality_reliability = None
        self.in_channels = in_channels
        if not isinstance(self.in_channels, tuple):
            self.in_channels = (self.in_channels,)
        self.layers = nn.ModuleList([nn.Conv2d(self.in_channels[i], out_channel, (3, 3)) for i in range(n_modalities)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_channel) for i in range(n_modalities)])
        if use_reliability_fusion:
            self.reliability_fusion = AdaptiveFeatureFusion(
                n_modalities=n_modalities,
                channels=out_channel,
                hidden_dim=64,
                min_weight=reliability_min_weight,
                temperature=reliability_temperature,
                preserve_scale=reliability_preserve_scale,
                uniform_init=reliability_uniform_init,
            )

    def forward(self, x):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        modal_features = [F.relu(bn(layer(x_i))) for x_i, layer, bn in zip(x, self.layers, self.bn)]
        if not self.use_reliability_fusion:
            self.last_modality_reliability = None
            return torch.cat(modal_features, dim=-1)

        stacked = torch.stack(modal_features, dim=1)
        fused_modal = self.reliability_fusion(stacked)
        self.last_modality_reliability = self.reliability_fusion.last_reliability
        return torch.cat(list(fused_modal.unbind(dim=1)), dim=-1)


class ContrastiveHead(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ContrastiveHead, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp_head(x)
        return x


class ClusteringHead(nn.Module):
    def __init__(self, n_dim, n_class, alpha=1.):
        super(ClusteringHead, self).__init__()
        # Clustering head
        self.alpha = alpha
        # initial_cluster_centers = torch.tensor(torch.randn((n_class, n_dim), dtype=torch.float, requires_grad=True))
        self.cluster_centers = nn.Parameter(torch.Tensor(n_class, n_dim), requires_grad=True)
        # torch.nn.init.orthogonal_(self.cluster_centers.data, gain=1)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        """
        :param x: n_batch * n-dim
        :return:
        """
        pred_prob = self.get_cluster_prob(x)
        return pred_prob

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class Net(nn.Module):
    def __init__(self, n_modalities, in_channels, in_patch_size, common_channel, n_class, dim_emebeding,
                 use_reliability_fusion=False, projection_dim=128,
                 reliability_min_weight=0.0,
                 reliability_temperature=1.0,
                 reliability_preserve_scale=False,
                 reliability_uniform_init=False):
        super(Net, self).__init__()
        # Validate input dimensions
        if not isinstance(in_channels, tuple):
            in_channels = (in_channels,)
        assert len(in_channels) == n_modalities, \
            f"Number of channels ({len(in_channels)}) must match number of modalities ({n_modalities})"

        self.embedding_layer = PatchEmbedding(
            n_modalities, in_channels, common_channel,
            image_size=in_patch_size[0],
            use_reliability_fusion=use_reliability_fusion,
            reliability_min_weight=reliability_min_weight,
            reliability_temperature=reliability_temperature,
            reliability_preserve_scale=reliability_preserve_scale,
            reliability_uniform_init=reliability_uniform_init,
        )
        embedded_size = (in_patch_size[0] - 2, in_patch_size[1] - 2)
        embedded_size = (embedded_size[0], embedded_size[1] * n_modalities)

        # Validate embedded size for ViT
        assert embedded_size[0] > 0 and embedded_size[1] > 0, \
            f"Invalid embedded size {embedded_size} after patch embedding. Check image_size and n_modalities."

        self.vit = ViT(image_size=embedded_size,  # use 3*3 kernel in embedding layer
                       # image_size=(in_patch_size[0], in_patch_size[1] * 2),
                       patch_size=1,
                       # num_classes=n_class,
                       dim=512,
                       depth=4,
                       heads=8,
                       mlp_dim=1024,
                       pool='mean',
                       channels=common_channel,
                       dim_head=64,
                       dropout=0.1,
                       emb_dropout=0.1
                       )
        self.clustering_head = ClusteringHead(dim_emebeding, n_class, alpha=1) ## ContrastiveHead(512, 128)
        # Keep instance discrimination out of the low-dimensional cluster
        # probability simplex.  The projection is used by InfoNCE only.
        self.projection_head = ContrastiveHead(dim_emebeding, projection_dim)

    def encode(self, x):
        return self.vit(self.embedding_layer(x))

    def forward(self, x_1, x_2, return_features=False, return_embeddings=False):
        """
        :param x_1, x_2: tuple of modalities, e.g., [aug_1, aug_2]-->
        ([img_rgb, img_hsi, img_sar], [img_rgb, img_hsi, img_sar])
        :return:
        """
        x_1 = self.encode(x_1)
        x_2 = self.encode(x_2)

        y_1 = self.clustering_head(x_1)
        y_2 = self.clustering_head(x_2)

        if return_embeddings and not return_features:
            raise ValueError("return_embeddings requires return_features=True")

        if return_features:
            z_1 = self.projection_head(x_1)
            z_2 = self.projection_head(x_2)
            if return_embeddings:
                return y_1, y_2, z_1, z_2, x_1, x_2
            return y_1, y_2, z_1, z_2
        return y_1, y_2

    def forward_pretrain(self, x_1, x_2):
        """Return projection features for contrastive representation pretraining."""
        h_1 = self.encode(x_1)
        h_2 = self.encode(x_2)
        return self.projection_head(h_1), self.projection_head(h_2)

    def forward_embedding(self, x):
        # h = self.clustering_head(self.vit(self.embedding_layer(x)))
        h = self.encode(x)
        return h

    def forward_cluster(self, x, return_h=False):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        h = self.encode(x)
        pred = self.clustering_head(h)
        labels = torch.argmax(pred, dim=1)
        if return_h:
            return labels, h
        return labels
