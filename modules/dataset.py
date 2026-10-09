import copy

import torch
from torch.utils.data import Dataset
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import StandardScaler
import numpy as np


class MultiModalDataset(Dataset):

    def __init__(self, gt_path, *src_path, patch_size=(7, 7), transform=None, is_labeled=True,
                 sampling_strategy='all', max_samples=None, sampling_seed=42,
                 scalers=None):
        self.transform = transform
        p = Processor()
        n_modality = len(src_path)
        modality_list = []
        in_channels = []
        sample_indices = None
        n_classes = None
        fitted_scalers = []
        if scalers is not None and len(scalers) != n_modality:
            raise ValueError("scalers must contain one fitted scaler per modality")
        for i in range(n_modality):
            try:
                if i == 0:
                    img, gt = p.prepare_data(src_path[i], gt_path)
                else:
                    img = p.prepare_data(src_path[i])
            except Exception as e:
                raise RuntimeError(f"Failed to load modality {i} from {src_path[i]}: {str(e)}") from e
            if i == 0:
                flat_labels = gt.reshape(-1)
                n_classes = (np.unique(flat_labels[flat_labels != 0]).shape[0]
                             if is_labeled else np.unique(flat_labels).shape[0] - 1)
                sample_indices = self._sample_indices(
                    flat_labels, sampling_strategy, max_samples, sampling_seed
                )
            x_patches, y_ = p.get_HSI_patches_rw(
                img, gt, (patch_size[0], patch_size[1]),
                is_indix=False, is_labeled=is_labeled,
                sample_indices=sample_indices if not is_labeled else None,
            )
            y_selected = y_
            n_samples, n_row, n_col, n_channel = x_patches.shape
            scaler = StandardScaler() if scalers is None else scalers[i]
            batch_size = 5000
            if scalers is None:
                # Fit once on the training population. Evaluation datasets must
                # reuse these statistics so train and inference stay in the same
                # feature space.
                for start_id in range(0, x_patches.shape[0], batch_size):
                    batch = x_patches[start_id:start_id+batch_size]
                    scaler.partial_fit(batch.reshape(batch.shape[0], -1))
            # Second pass: transform the data using the fitted scaler
            for start_id in range(0, x_patches.shape[0], batch_size):
                batch = x_patches[start_id: start_id+batch_size]
                shape = batch.shape
                x_temp = batch.reshape(shape[0], -1)
                x_patches[start_id: start_id+batch_size] = scaler.transform(x_temp).reshape(shape)
            x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
            x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)
            modality_list.append(x_tensor)
            in_channels.append(n_channel)
            fitted_scalers.append(scaler)
        y = p.standardize_label(y_selected)
        self.gt_shape = gt.shape
        self.data_size = len(y)
        self.n_classes = n_classes
        self.y_tensor = torch.from_numpy(y).type(torch.LongTensor)
        self.modality_list = tuple(modality_list)
        self.n_modality = n_modality
        self.in_channels = tuple(in_channels)
        self.scalers = tuple(fitted_scalers)
        self.sample_indices = sample_indices

    def with_transform(self, transform=None):
        """Return a lightweight dataset view that shares preprocessed tensors."""
        dataset_view = copy.copy(self)
        dataset_view.transform = transform
        return dataset_view

    @staticmethod
    def _sample_indices(labels, strategy, max_samples, seed):
        strategy = str(strategy).lower()
        if strategy == 'all' or max_samples is None:
            return None
        max_samples = int(max_samples)
        if max_samples <= 0:
            return None
        if strategy == 'labeled':
            candidates = np.flatnonzero(labels)
        elif strategy == 'random':
            candidates = np.arange(labels.shape[0])
        else:
            raise ValueError("sampling_strategy must be one of: all, random, labeled")
        if candidates.size <= max_samples:
            return candidates
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(candidates, size=max_samples, replace=False))

    def __getitem__(self, idx):
        x_list = [self.modality_list[i][idx] for i in range(self.n_modality)]
        if self.transform is not None:
            x_list = self.transform.apply_multimodal(x_list)
        y = self.y_tensor[idx]
        return x_list, y

    def __len__(self):
        return self.data_size
