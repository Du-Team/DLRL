import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants for numerical stability
EPSILON = 1e-8


def build_dual_view_target(q_aug1, q_aug2, eps=EPSILON):
    """Build the detached pseudo target shared by clustering and relation losses."""
    with torch.no_grad():
        q_aug1 = q_aug1.detach().clamp_min(eps)
        q_aug2 = q_aug2.detach().clamp_min(eps)
        q_aug1 = q_aug1 / q_aug1.sum(dim=1, keepdim=True).clamp_min(eps)
        q_aug2 = q_aug2 / q_aug2.sum(dim=1, keepdim=True).clamp_min(eps)
        support = torch.maximum(q_aug1, q_aug2)
        target = support.pow(2)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
        confidence, pseudo_label = target.max(dim=1)
        stable_mask = q_aug1.argmax(dim=1).eq(q_aug2.argmax(dim=1))
    return {
        "q_aug1": q_aug1,
        "q_aug2": q_aug2,
        "target": target,
        "confidence": confidence,
        "pseudo_label": pseudo_label,
        "stable_mask": stable_mask,
    }


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.register_buffer("mask", self.mask_correlated_samples(batch_size))
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        n_samples = 2 * batch_size
        mask = torch.ones((n_samples, n_samples))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask.bool()

    def forward(self, z_i, z_j, negative_exclusion=None):
        batch_size = z_i.size(0)
        n_samples = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n_samples, 1)
        mask = self.mask if batch_size == self.batch_size else self.mask_correlated_samples(batch_size).to(sim.device)
        if negative_exclusion is not None:
            if negative_exclusion.shape != (batch_size, batch_size):
                raise ValueError("negative_exclusion must have shape [batch_size, batch_size]")
            negative_exclusion = negative_exclusion.to(sim.device, dtype=torch.bool)
            # A relation selected in either direction is a pseudo-positive and
            # must not simultaneously act as an instance-level negative.
            negative_exclusion = negative_exclusion | negative_exclusion.t()
            mask = mask & ~negative_exclusion.repeat(2, 2)
        negative_samples = sim.masked_fill(~mask, torch.finfo(sim.dtype).min)

        labels = torch.zeros(n_samples).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= n_samples
        return loss


class ReliabilityGraphContrastiveLoss(nn.Module):
    """RAC-DMVC-inspired soft contrastive loss for FUATR's two augmentations.

    RAC-DMVC builds a reliability graph from a slowly moving target branch and
    uses every graph entry as a continuous positive/negative indicator.  FUATR
    has two augmented fused-modal samples rather than two independent vector
    views, so we retain only relations on which both augmentations agree.  This
    avoids cluster pseudo-labels, confidence thresholds and hard pseudo-class
    neighbour selection in the contrastive objective.  Optional Top-K pruning
    only sparsifies the continuous reliability graph.
    """

    def __init__(self, temperature=0.5, graph_sigma=0.1,
                 graph_alpha=0.5, graph_steps=1, graph_topk=16,
                 eps=EPSILON):
        super().__init__()
        self.temperature = float(temperature)
        self.graph_sigma = float(graph_sigma)
        self.graph_alpha = float(graph_alpha)
        self.graph_steps = max(1, int(graph_steps))
        self.graph_topk = max(0, int(graph_topk))
        self.eps = float(eps)
        self.last_stats = {}

    @torch.no_grad()
    def _affinity(self, features):
        features = F.normalize(features.detach(), p=2, dim=1)
        squared_distance = (2.0 - 2.0 * (features @ features.t())).clamp_min(0.0)
        affinity = torch.exp(-squared_distance / max(self.graph_sigma, self.eps))
        affinity = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(self.eps)
        if self.graph_steps > 1:
            affinity = torch.matrix_power(affinity, self.graph_steps)
        return affinity

    @torch.no_grad()
    def build_reliability_graph(self, teacher_z_1, teacher_z_2):
        graph_1 = self._affinity(teacher_z_1)
        graph_2 = self._affinity(teacher_z_2)

        # Geometric agreement suppresses a relation that is strong in only one
        # augmentation, which is the augmentation-specific adaptation of RAC.
        graph = torch.sqrt((graph_1 * graph_2).clamp_min(0.0))
        graph = 0.5 * (graph + graph.t())
        graph = graph / graph.sum(dim=1, keepdim=True).clamp_min(self.eps)

        if self.graph_topk > 0 and graph.size(0) > 1:
            diagonal = torch.eye(graph.size(0), device=graph.device, dtype=torch.bool)
            candidates = graph.masked_fill(diagonal, -1.0)
            k = min(self.graph_topk, graph.size(0) - 1)
            _, indices = torch.topk(candidates, k=k, dim=1)
            keep = diagonal.clone()
            keep.scatter_(1, indices, True)
            keep = keep | keep.t()
            graph = graph * keep.to(graph.dtype)
            graph = graph / graph.sum(dim=1, keepdim=True).clamp_min(self.eps)

        identity = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = self.graph_alpha * identity + (1.0 - self.graph_alpha) * graph
        # The paired augmentation of the same pixel is always fully positive.
        graph.fill_diagonal_(1.0)
        return graph.clamp(0.0, 1.0)

    def _soft_multi_positive_loss(self, feature_1, feature_2, graph):
        features = F.normalize(torch.cat([feature_1, feature_2], dim=0), p=2, dim=1)
        logits = (features @ features.t()) / self.temperature
        pair_probability = graph.repeat(2, 2).to(logits.dtype)
        self_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        valid = ~self_mask
        pair_probability = pair_probability.masked_fill(self_mask, 0.0)

        # Soft multi-positive InfoNCE. P_ij contributes to the numerator while
        # (1-P_ij) remains its negative probability. Since the two probabilities
        # sum to one, the denominator stays well calibrated and numerically stable.
        log_weight = pair_probability.clamp_min(self.eps).log()
        log_numerator = torch.logsumexp(
            (logits + log_weight).masked_fill(~valid, -torch.inf), dim=1
        )
        log_denominator = torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)
        return (log_denominator - log_numerator).mean()

    def forward(self, z_1, z_2, teacher_z_1=None, teacher_z_2=None,
                encoder_1=None, encoder_2=None, encoder_mix=0.0):
        if teacher_z_1 is None:
            teacher_z_1 = z_1.detach()
        if teacher_z_2 is None:
            teacher_z_2 = z_2.detach()
        graph = self.build_reliability_graph(teacher_z_1, teacher_z_2)
        encoder_mix = float(min(1.0, max(0.0, encoder_mix)))
        projection_loss = self._soft_multi_positive_loss(z_1, z_2, graph)
        encoder_loss = projection_loss.new_zeros(())
        if encoder_mix > 0.0:
            if encoder_1 is None or encoder_2 is None:
                raise ValueError(
                    "encoder features are required when RAC encoder_mix is positive"
                )
            encoder_loss = self._soft_multi_positive_loss(encoder_1, encoder_2, graph)
        if encoder_mix <= 0.0:
            # Preserve the legacy projection-only computation exactly.
            loss = projection_loss
        elif encoder_mix >= 1.0:
            loss = encoder_loss
        else:
            loss = (1.0 - encoder_mix) * projection_loss + encoder_mix * encoder_loss

        batch_size = graph.size(0)
        off_diagonal = ~torch.eye(batch_size, device=graph.device, dtype=torch.bool)
        off_values = graph[off_diagonal]
        offdiag_nonzero = (graph > 0.0) & off_diagonal
        self.last_stats = {
            "reliability_graph_mean": graph.mean().detach(),
            "reliability_graph_offdiag_mean": (
                off_values.mean().detach() if off_values.numel() else graph.new_tensor(0.0)
            ),
            "reliability_graph_offdiag_max": (
                off_values.max().detach() if off_values.numel() else graph.new_tensor(0.0)
            ),
            "reliability_graph_offdiag_density": (
                (
                    offdiag_nonzero.sum().to(graph.dtype)
                    / (batch_size * (batch_size - 1))
                ).detach()
                if batch_size > 1 else graph.new_tensor(0.0)
            ),
            "reliability_graph_neighbors_per_anchor": (
                offdiag_nonzero.sum(dim=1).to(graph.dtype).mean().detach()
                if batch_size > 1 else graph.new_tensor(0.0)
            ),
            "false_negative_mass_per_anchor": graph.masked_fill(~off_diagonal, 0.0).sum(1).mean().detach(),
            "reliability_contrastive_loss": loss.detach(),
            "rac_projection_loss": projection_loss.detach(),
            "rac_encoder_loss": encoder_loss.detach(),
            "rac_encoder_mix": graph.new_tensor(encoder_mix),
        }
        return loss


class CrossCorrelationLoss(nn.Module):
    def __init__(self, out_dim, lambd, device):
        super(CrossCorrelationLoss, self).__init__()
        self.lambd = lambd
        self.device = device
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, y_i, y_j):
        batch_size = y_i.size(0)
        c = self.bn(y_i).T @ self.bn(y_j)
        c.div_(batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ClusteringLoss(nn.Module):
    """DEC clustering objective without additional weighted regularizers."""

    def __init__(self, frequency_power=0.5,
                 use_high_confidence=False, high_confidence_ratio=1.0,
                 min_confidence=0.0, min_samples_per_cluster=1,
                 balance_selected_clusters=False,
                 balance_coef=0.0, center_orthogonal_coef=0.0,
                 small_cluster_protection_weight=0.0,
                 small_cluster_floor_factor=0.5,
                 small_cluster_min_ratio=0.0):
        super(ClusteringLoss, self).__init__()
        self.frequency_power = float(frequency_power)
        self.use_high_confidence = bool(use_high_confidence)
        self.high_confidence_ratio = float(high_confidence_ratio)
        self.min_confidence = float(min_confidence)
        self.min_samples_per_cluster = int(min_samples_per_cluster)
        self.balance_selected_clusters = bool(balance_selected_clusters)
        self.balance_coef = float(balance_coef)
        self.center_orthogonal_coef = float(center_orthogonal_coef)
        self.small_cluster_protection_weight = float(small_cluster_protection_weight)
        self.small_cluster_floor_factor = float(small_cluster_floor_factor)
        self.small_cluster_min_ratio = float(small_cluster_min_ratio)
        self.reference_cluster_ratio = None
        self.active_cluster_mask = None
        self.eps = EPSILON
        self.last_stats = {}

    def set_high_confidence(self, enabled, high_confidence_ratio=None):
        self.use_high_confidence = bool(enabled)
        if high_confidence_ratio is not None:
            self.high_confidence_ratio = float(high_confidence_ratio)

    def set_reference_cluster_freq(self, freq):
        if freq is None:
            self.reference_cluster_ratio = None
            return
        freq = freq.detach().float().cpu().clamp_min(0)
        self.reference_cluster_ratio = freq / freq.sum().clamp_min(self.eps)

    def set_active_clusters(self, active_cluster_mask):
        if active_cluster_mask is None:
            self.active_cluster_mask = None
            return
        mask = torch.as_tensor(
            active_cluster_mask, dtype=torch.bool
        ).detach().cpu().view(-1)
        if mask.numel() == 0:
            raise ValueError("active_cluster_mask must not be empty")
        self.active_cluster_mask = mask

    def get_active_cluster_mask(self, cluster_count, device):
        if self.active_cluster_mask is None:
            return torch.ones(cluster_count, dtype=torch.bool, device=device)
        if self.active_cluster_mask.numel() != cluster_count:
            raise ValueError(
                "active_cluster_mask size must match the number of clusters"
            )
        return self.active_cluster_mask.to(device=device)

    def build_high_confidence_sample_weight(self, q_aug1, q_aug2, shared_target=None):
        with torch.no_grad():
            if shared_target is None:
                shared_target = build_dual_view_target(q_aug1, q_aug2, self.eps)
            confidence = shared_target["confidence"]
            pseudo_label = shared_target["pseudo_label"]
            stable_mask = shared_target["stable_mask"]
            candidate_mask = stable_mask & (confidence >= self.min_confidence)
            active_clusters = self.get_active_cluster_mask(
                q_aug1.size(1), q_aug1.device
            )
            candidate_mask &= active_clusters[pseudo_label]

            selected = torch.zeros_like(candidate_mask)
            ratio = min(1.0, max(0.0, self.high_confidence_ratio))
            for cluster_id in range(q_aug1.size(1)):
                if not bool(active_clusters[cluster_id].item()):
                    continue
                cluster_index = torch.nonzero(
                    candidate_mask & pseudo_label.eq(cluster_id),
                    as_tuple=False,
                ).view(-1)
                if cluster_index.numel() == 0:
                    continue
                keep_count = int(math.ceil(cluster_index.numel() * ratio))
                keep_count = max(self.min_samples_per_cluster, keep_count)
                keep_count = min(cluster_index.numel(), keep_count)
                _, order = torch.topk(confidence[cluster_index], k=keep_count, largest=True)
                selected[cluster_index[order]] = True

            selected_cluster_count = pseudo_label[selected].unique().numel() if selected.any() else 0
            selected_weight = selected.float()
            if selected.any() and self.balance_selected_clusters:
                # Give every represented healthy cluster the same total weight.
                # This prevents large pseudo-clusters from dominating CLU while
                # preserving the original mean loss scale.
                selected_weight = confidence.new_zeros(confidence.shape)
                for cluster_id in pseudo_label[selected].unique().tolist():
                    cluster_selected = selected & pseudo_label.eq(cluster_id)
                    selected_weight[cluster_selected] = (
                        1.0 / cluster_selected.sum().to(confidence.dtype)
                    )
                selected_weight *= (
                    selected.sum().to(confidence.dtype)
                    / selected_weight.sum().clamp_min(self.eps)
                )
            stats = {
                "clu_high_confidence_selected_ratio": selected.float().mean(),
                "clu_high_confidence_stable_ratio": stable_mask.float().mean(),
                "clu_high_confidence_mean_confidence": confidence[selected].mean()
                if selected.any() else confidence.new_tensor(0.0),
                "clu_high_confidence_selected_clusters": confidence.new_tensor(
                    float(selected_cluster_count)
                ),
                "clu_active_clusters": active_clusters.float().sum(),
            }
            sample_weight = torch.cat([selected_weight, selected_weight], dim=0)
        return sample_weight, stats

    def _normalize_prob(self, prob):
        prob = prob.clamp_min(self.eps)
        return prob / prob.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def forward(self, y_prob, cluster_center=None,
                global_freq: torch.Tensor = None,
                sample_weight: torch.Tensor = None):
        target_prob = self.target_distribution(y_prob, global_freq=global_freq).detach()
        per_sample_loss = F.kl_div(
            y_prob.clamp_min(self.eps).log(),
            target_prob,
            reduction='none',
        ).sum(dim=1)
        active_clusters = self.get_active_cluster_mask(
            y_prob.size(1), y_prob.device
        )
        if sample_weight is None and not bool(active_clusters.all().item()):
            pseudo_label = y_prob.detach().argmax(dim=1)
            sample_weight = active_clusters[pseudo_label].to(per_sample_loss.dtype)
        if sample_weight is not None:
            sample_weight = sample_weight.to(y_prob.device, dtype=per_sample_loss.dtype).view(-1)
            selected_weight = sample_weight.sum()
            if selected_weight.item() > 0:
                loss = (per_sample_loss * sample_weight).sum() / selected_weight.clamp_min(self.eps)
            else:
                loss = per_sample_loss.sum() * 0.0
        else:
            selected_weight = y_prob.new_tensor(float(y_prob.shape[0]))
            loss = per_sample_loss.mean()

        marginal = self._normalize_prob(y_prob).mean(dim=0)
        balance_loss = (
            math.log(marginal.numel())
            + (marginal * marginal.clamp_min(self.eps).log()).sum()
        )
        center_orthogonal_loss = y_prob.new_tensor(0.0)
        if cluster_center is not None and cluster_center.size(0) > 1:
            normalized_center = F.normalize(cluster_center, p=2, dim=1)
            gram = normalized_center @ normalized_center.t()
            eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            center_orthogonal_loss = (gram - eye).pow(2).sum()
        small_cluster_floor_loss = y_prob.new_tensor(0.0)
        protected_count = y_prob.new_tensor(0.0)
        min_floor_ratio = y_prob.new_tensor(0.0)
        if (self.small_cluster_protection_weight > 0
                and self.reference_cluster_ratio is not None):
            reference_ratio = self.reference_cluster_ratio.to(
                device=y_prob.device, dtype=marginal.dtype
            )
            floor_ratio = torch.maximum(
                reference_ratio * self.small_cluster_floor_factor,
                torch.full_like(reference_ratio, self.small_cluster_min_ratio),
            )
            relative_deficit = (
                (floor_ratio - marginal).clamp_min(0)
                / floor_ratio.clamp_min(self.eps)
            )
            small_cluster_floor_loss = relative_deficit.square().mean()
            protected_count = relative_deficit.gt(0).float().sum()
            min_floor_ratio = floor_ratio.min()
        loss = (
            loss
            + self.balance_coef * balance_loss
            + self.center_orthogonal_coef * center_orthogonal_loss
            + self.small_cluster_protection_weight * small_cluster_floor_loss
        )

        self.last_stats = {
            "clu_effective_sample_ratio": (
                selected_weight.detach() / y_prob.new_tensor(float(y_prob.shape[0]))
            ),
            "balance_loss": balance_loss.detach(),
            "center_orthogonal_loss": center_orthogonal_loss.detach(),
            "small_cluster_floor_loss": small_cluster_floor_loss.detach(),
            "small_cluster_protected_count": protected_count.detach(),
            "small_cluster_min_batch_ratio": marginal.min().detach(),
            "small_cluster_min_floor_ratio": min_floor_ratio.detach(),
            "clu_active_clusters": active_clusters.float().sum().detach(),
        }
        return loss

    def target_distribution(self, batch: torch.Tensor, global_freq: torch.Tensor = None) -> torch.Tensor:
        if global_freq is not None:
            frequency = global_freq.to(batch.device).clamp_min(self.eps)
        else:
            frequency = torch.sum(batch, 0).clamp_min(self.eps)
        weight = (batch ** 2) / frequency.pow(self.frequency_power)
        return (weight.t() / torch.sum(weight, 1).clamp_min(self.eps)).t()


class PretrainLoss(nn.Module):
    """
    pretrain model for n epoch with a contrastive task, e.g., SimCLR/BarlowTwins
    """
    def __init__(self, batch_size, lambda_, device='cpu'):
        super(PretrainLoss, self).__init__()
        self.device = device
        self.criterion = InstanceLoss(batch_size, lambda_, device).to(device)

    def forward(self, x_1, x_2):
        return self.criterion(x_1, x_2)


class _LegacyHighConfidenceTargetRelationLoss(nn.Module):
    """Deprecated FUATR1.4 predecessor; retained only for old import compatibility."""

    def __init__(self, temperature=0.2, reliable_ratio=0.2, topk=5, eps=EPSILON):
        super().__init__()
        self.temperature = float(temperature)
        self.reliable_ratio = float(reliable_ratio)
        self.topk = int(topk)
        self.eps = eps
        self.last_stats = {}

    def set_reliable_ratio(self, ratio):
        self.reliable_ratio = float(min(1.0, max(0.0, ratio)))

    def _normalize_prob(self, prob):
        prob = prob.clamp_min(self.eps)
        return prob / prob.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def build_relation_targets(self, q_aug1, q_aug2, shared_target=None):
        with torch.no_grad():
            if shared_target is None:
                shared_target = build_dual_view_target(q_aug1, q_aug2, self.eps)
            p_target = shared_target["target"]
            confidence = shared_target["confidence"]
            pseudo_label = shared_target["pseudo_label"]
            dual_augmentation_cluster_stability = shared_target["stable_mask"]
            stable_confidence = confidence[dual_augmentation_cluster_stability]
            reliable_mask = torch.zeros_like(dual_augmentation_cluster_stability, dtype=torch.bool)
            tau = confidence.new_tensor(1.0)
            if stable_confidence.numel() > 0 and self.reliable_ratio > 0:
                quantile_level = 1.0 - self.reliable_ratio
                quantile_level = min(1.0, max(0.0, quantile_level))
                tau = torch.quantile(stable_confidence.float(), quantile_level).to(confidence.dtype)
                reliable_mask = dual_augmentation_cluster_stability & (confidence >= tau)

            p_norm = F.normalize(p_target, p=2, dim=1)
            soft_distribution_similarity = torch.matmul(p_norm, p_norm.t()).clamp_min(0.0)
            same_cluster = pseudo_label.view(-1, 1).eq(pseudo_label.view(1, -1))
            pair_reliable = reliable_mask.view(-1, 1) & reliable_mask.view(1, -1)
            pair_weight = (
                pair_reliable.float()
                * same_cluster.float()
                * soft_distribution_similarity
                * confidence.view(-1, 1)
                * confidence.view(1, -1)
            )
            pair_weight.fill_diagonal_(0.0)
            pair_weight = self._keep_topk_relations(pair_weight)

            valid_pair_mask = pair_weight > 0
            stable_mean = (
                stable_confidence.mean()
                if stable_confidence.numel() > 0
                else confidence.new_tensor(0.0)
            )
            mean_relation_weight = (
                pair_weight[valid_pair_mask].mean()
                if valid_pair_mask.any()
                else confidence.new_tensor(0.0)
            )
            stats = {
                "reliable_ratio_target": confidence.new_tensor(self.reliable_ratio),
                "dual_augmentation_cluster_stability_rate": dual_augmentation_cluster_stability.float().mean(),
                "actual_reliable_sample_ratio": reliable_mask.float().mean(),
                "confidence_threshold_tau": tau,
                "mean_confidence_of_stable_samples": stable_mean,
                "mean_relation_weight": mean_relation_weight,
                "num_valid_relation_pairs": valid_pair_mask.float().sum(),
            }
        return p_target, confidence, pseudo_label, reliable_mask, pair_weight, stats

    def _keep_topk_relations(self, pair_weight):
        if self.topk <= 0 or pair_weight.size(1) <= 1:
            return pair_weight.zero_()
        k = min(self.topk, pair_weight.size(1) - 1)
        values, indices = torch.topk(pair_weight, k=k, dim=1, largest=True)
        keep = torch.zeros_like(pair_weight, dtype=torch.bool)
        keep.scatter_(1, indices, values > 0)
        return pair_weight * keep.float()

    def forward(self, z_1, z_2, q_1, q_2, relation_targets=None):
        z_rel = F.normalize((z_1 + z_2).mul(0.5), p=2, dim=1)
        if relation_targets is None:
            relation_targets = self.build_relation_targets(q_1, q_2)
        _, _, _, _, pair_weight, stats = relation_targets
        valid_anchor = pair_weight.sum(dim=1) > 0

        if z_rel.size(0) <= 1 or not valid_anchor.any():
            loss_rel = z_rel.sum() * 0.0
            # Track how often relation loss is zero to help diagnose issues
            stats["relation_loss_zero"] = z_rel.new_tensor(1.0)
        else:
            logits = torch.matmul(z_rel, z_rel.t()) / self.temperature
            diag_mask = torch.eye(z_rel.size(0), device=z_rel.device, dtype=torch.bool)
            logits = logits.masked_fill(diag_mask, -1e9)
            log_prob = F.log_softmax(logits, dim=1)
            per_anchor_loss = -(
                pair_weight.to(log_prob.dtype) * log_prob
            ).sum(dim=1) / pair_weight.sum(dim=1).clamp_min(self.eps)
            loss_rel = per_anchor_loss[valid_anchor].mean()
            stats["relation_loss_zero"] = z_rel.new_tensor(0.0)

        self.last_stats = {name: value.detach() for name, value in stats.items()}
        self.last_stats["relation_loss"] = loss_rel.detach()
        return loss_rel


class JointLoss(nn.Module):
    """Weighted joint objective with an optional EMA-teacher anchor."""

    def __init__(self, batch_size, instance_temperature=0.5, lambda_clu=1.0, device='cpu',
                 use_reliability_contrastive=False, lambda_rel=0.0,
                 frequency_power=0.5,
                 use_high_confidence_clu=False, clu_reliable_ratio=1.0,
                 clu_min_confidence=0.0, clu_min_samples_per_cluster=1,
                 use_cluster_balanced_clu=False,
                 instance_weight=1.0, anchor_weight=0.0,
                 balance_coef=0.0, center_orthogonal_coef=0.0,
                 small_cluster_protection_weight=0.0,
                 small_cluster_floor_factor=0.5,
                 small_cluster_min_ratio=0.0,
                 reliability_graph_sigma=0.1,
                 reliability_graph_alpha=0.5,
                 reliability_graph_steps=1,
                 reliability_graph_topk=0,
                 rac_connection_mode="projection", rac_encoder_mix=0.0):
        super(JointLoss, self).__init__()
        self.device = device
        self.lambda_clu = float(lambda_clu)
        self.instance_weight = float(instance_weight)
        self.anchor_weight = float(anchor_weight)
        self.criterion_contrastive = InstanceLoss(
            batch_size, instance_temperature, device
        ).to(device)
        self.clustering_loss = ClusteringLoss(
            frequency_power=frequency_power,
            use_high_confidence=use_high_confidence_clu,
            high_confidence_ratio=clu_reliable_ratio,
            min_confidence=clu_min_confidence,
            min_samples_per_cluster=clu_min_samples_per_cluster,
            balance_selected_clusters=use_cluster_balanced_clu,
            balance_coef=balance_coef,
            center_orthogonal_coef=center_orthogonal_coef,
            small_cluster_protection_weight=small_cluster_protection_weight,
            small_cluster_floor_factor=small_cluster_floor_factor,
            small_cluster_min_ratio=small_cluster_min_ratio,
        )
        self.lambda_rel = float(lambda_rel)
        self.use_reliability_contrastive = bool(use_reliability_contrastive)
        self.uses_features = self.use_reliability_contrastive
        self.rac_connection_mode = str(rac_connection_mode).strip().lower()
        if self.rac_connection_mode not in {"projection", "encoder", "hybrid"}:
            raise ValueError(
                "rac_connection_mode must be one of: projection, encoder, hybrid"
            )
        configured_encoder_mix = float(rac_encoder_mix)
        if not 0.0 <= configured_encoder_mix <= 1.0:
            raise ValueError("rac_encoder_mix must be in [0, 1]")
        if self.rac_connection_mode == "projection":
            self.rac_encoder_mix = 0.0
        elif self.rac_connection_mode == "encoder":
            self.rac_encoder_mix = 1.0
        else:
            self.rac_encoder_mix = configured_encoder_mix
        self.requires_encoder_features = (
            self.use_reliability_contrastive and self.rac_encoder_mix > 0.0
        )
        self.reliability_contrastive = ReliabilityGraphContrastiveLoss(
            temperature=instance_temperature,
            graph_sigma=reliability_graph_sigma,
            graph_alpha=reliability_graph_alpha,
            graph_steps=reliability_graph_steps,
            graph_topk=reliability_graph_topk,
        ).to(device)
        self.last_relation_stats = {}
        self.global_freq: torch.Tensor = None

    def set_lambda_rel(self, value):
        self.lambda_rel = float(value)

    def set_lambda_clu(self, value):
        """Update the clustering weight without rebuilding the loss module."""
        self.lambda_clu = float(value)

    def set_instance_weight(self, value):
        self.instance_weight = float(value)

    def set_anchor_weight(self, value):
        self.anchor_weight = float(value)

    def set_global_freq(self, freq: torch.Tensor):
        self.global_freq = None if freq is None else freq.detach().cpu()

    def set_reference_cluster_freq(self, freq: torch.Tensor):
        self.clustering_loss.set_reference_cluster_freq(freq)

    def set_high_confidence_clu(self, enabled, reliable_ratio=None):
        self.clustering_loss.set_high_confidence(enabled, reliable_ratio)

    def set_active_clusters(self, active_cluster_mask):
        self.clustering_loss.set_active_clusters(active_cluster_mask)

    def forward(self, y_1, y_2, cluster_center=None, z_1=None, z_2=None,
                h_1=None, h_2=None,
                anchor_y_1=None, anchor_y_2=None,
                teacher_z_1=None, teacher_z_2=None):
        h = torch.cat([y_1, y_2], dim=0)
        if z_1 is None or z_2 is None:
            raise ValueError("JointLoss requires projection features z_1 and z_2 for instance contrastive learning.")

        relation_active = self.uses_features and self.lambda_rel > 0
        shared_target = None
        if self.clustering_loss.use_high_confidence:
            shared_target = build_dual_view_target(y_1, y_2)

        standard_instance_loss = self.criterion_contrastive(z_1, z_2)
        loss_ins = standard_instance_loss
        reliability_loss = standard_instance_loss.new_zeros(())
        if relation_active:
            reliability_loss = self.reliability_contrastive(
                z_1, z_2,
                teacher_z_1=teacher_z_1, teacher_z_2=teacher_z_2,
                encoder_1=h_1, encoder_2=h_2,
                encoder_mix=self.rac_encoder_mix,
            )
            graph_strength = min(1.0, max(0.0, self.lambda_rel))
            effective_graph_strength = graph_strength / (1.0 + graph_strength)
            loss_ins = (
                (1.0 - effective_graph_strength) * standard_instance_loss
                + effective_graph_strength * reliability_loss
            )
        sample_weight = None
        clu_stats = {}
        if self.clustering_loss.use_high_confidence:
            sample_weight, clu_stats = self.clustering_loss.build_high_confidence_sample_weight(
                y_1, y_2, shared_target=shared_target
            )
        raw_clu = self.clustering_loss(
            h,
            cluster_center=cluster_center,
            global_freq=self.global_freq,
            sample_weight=sample_weight,
        )

        if relation_active:
            self.last_relation_stats = dict(self.reliability_contrastive.last_stats)
        else:
            self.last_relation_stats = {}

        self.last_relation_stats.update(clu_stats)
        self.last_relation_stats.update(self.clustering_loss.last_stats)
        self.last_relation_stats["relation_lambda"] = torch.tensor(
            self.lambda_rel, device=y_1.device
        )
        self.last_relation_stats["rac_encoder_mix"] = torch.tensor(
            self.rac_encoder_mix, device=y_1.device
        )
        self.last_relation_stats["reliability_graph_strength"] = torch.tensor(
            effective_graph_strength if relation_active else 0.0,
            device=y_1.device
        )
        self.last_relation_stats["cluster_lambda"] = torch.tensor(
            self.lambda_clu, device=y_1.device
        )
        self.last_relation_stats["instance_loss"] = loss_ins.detach()
        self.last_relation_stats["standard_instance_loss"] = standard_instance_loss.detach()
        self.last_relation_stats["reliability_contrastive_loss"] = reliability_loss.detach()
        self.last_relation_stats["clustering_loss"] = raw_clu.detach()
        raw_anchor = torch.zeros((), device=y_1.device)
        if self.anchor_weight > 0 and anchor_y_1 is not None and anchor_y_2 is not None:
            # The EMA model supplies slowly moving targets initialized from the
            # KMeans solution. KL is evaluated in the teacher-to-student
            # direction so gradients only update the online model.
            target_1 = anchor_y_1.detach().clamp_min(EPSILON)
            target_2 = anchor_y_2.detach().clamp_min(EPSILON)
            raw_anchor = 0.5 * (
                F.kl_div(y_1.clamp_min(EPSILON).log(), target_1, reduction="batchmean")
                + F.kl_div(y_2.clamp_min(EPSILON).log(), target_2, reduction="batchmean")
            )
        self.last_relation_stats["anchor_weight"] = torch.tensor(
            self.anchor_weight, device=y_1.device
        )
        self.last_relation_stats["anchor_loss"] = raw_anchor.detach()
        weighted_ins = self.instance_weight * loss_ins
        weighted_clu = self.lambda_clu * raw_clu
        # Reliability is interpolated inside the sole instance objective; it is
        # not added as a second loss that would double-count contrastive learning.
        weighted_anchor = self.anchor_weight * raw_anchor
        loss = weighted_ins + weighted_clu + weighted_anchor
        return loss, weighted_ins, weighted_clu
