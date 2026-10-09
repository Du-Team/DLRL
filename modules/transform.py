# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torchvision
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.decomposition import PCA
from sklearn import random_projection


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, img):
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            # torchvision preserves the channel-first C x H x W layout and
            # applies the blur independently to every spectral band.
            img = torchvision.transforms.functional.gaussian_blur(
                img, kernel_size=[self.kernel_size, self.kernel_size],
                sigma=[sigma, sigma]
            )
        return img


class Transforms:
    def __init__(self, size, mean=None, std=None, blur=False,
                 crop_scale=(0.7, 1.0), mask_pixel_prob=0.1,
                 mask_band_prob=0.1, modality_aware_augmentation=False,
                 use_group_permute_bands=True):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.crop_scale = tuple(crop_scale)
        self.crop_ratio = (0.9, 1.1)
        self.modality_aware_augmentation = bool(modality_aware_augmentation)
        # Four augmentation options; GroupPermuteBands adds spectral-order invariance
        # which helps the HSI encoder ignore arbitrary band ordering.
        # MaskPixels probability is reduced from 0.6 → 0.3 to avoid destroying
        # the narrow-channel LiDAR modality too aggressively.
        spectral_order_transform = (
            GroupPermuteBands(n_group=4)
            if use_group_permute_bands
            else torch.nn.Identity()
        )
        self.appearance_transform = torchvision.transforms.RandomChoice([
            GaussianBlur(3),
            MaskPixels(p=mask_pixel_prob),
            MaskBands(p=mask_band_prob),
            spectral_order_transform,
        ], p=[0.2, 0.3, 0.3, 0.2])
        self.low_channel_appearance_transform = torchvision.transforms.RandomChoice([
            GaussianBlur(3),
            MaskPixels(p=mask_pixel_prob),
            torch.nn.Identity(),
        ], p=[0.25, 0.35, 0.40])
        self.extra_blur = GaussianBlur(kernel_size=3) if blur else None
        self.normalize = None
        # self.train_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            # torchvision.transforms.Resize(size=(size, size)),
            # MaskBands(),
            # RandomProjectionBands(n_band=200)
            # torchvision.transforms.ToTensor(),
            # MaskBands(p=0.2),
            # RandomProjectionBands(n_band=32),
            # PermuteBands(10)
        ]
        if mean and std:
            self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        view_1, view_2 = self.apply_multimodal([x])
        return view_1[0], view_2[0]

    def _sample_geometry(self, reference):
        crop = torchvision.transforms.RandomResizedCrop.get_params(
            reference, scale=self.crop_scale, ratio=self.crop_ratio
        )
        return crop, bool(torch.rand(()) < 0.5), bool(torch.rand(()) < 0.5)

    def _apply_geometry(self, x, geometry):
        (top, left, height, width), flip_horizontal, flip_vertical = geometry
        x = torchvision.transforms.functional.resized_crop(
            x, top, left, height, width, self.size, antialias=True
        )
        if flip_horizontal:
            x = torchvision.transforms.functional.hflip(x)
        if flip_vertical:
            x = torchvision.transforms.functional.vflip(x)
        return x

    def _apply_appearance(self, x):
        if self.modality_aware_augmentation and x.size(0) <= 2:
            # MaskBands can erase an entire DSM/LiDAR view when it has only one
            # channel. Keep narrow modalities informative and use spatial
            # corruption instead.
            x = self.low_channel_appearance_transform(x)
        else:
            x = self.appearance_transform(x)
        if self.extra_blur is not None:
            x = self.extra_blur(x)
        if self.normalize is not None:
            x = self.normalize(x)
        return x

    def apply_multimodal(self, modalities):
        """Create two views while sharing geometric parameters across modalities."""
        geometry_1 = self._sample_geometry(modalities[0])
        geometry_2 = self._sample_geometry(modalities[0])
        view_1 = [self._apply_appearance(self._apply_geometry(x, geometry_1)) for x in modalities]
        view_2 = [self._apply_appearance(self._apply_geometry(x, geometry_2)) for x in modalities]
        return view_1, view_2


class GroupPermuteBands(object):
    """
    shuffle bands into n_groups
    """
    def __init__(self, n_group=3):
        self.n_group = n_group

    def __call__(self, img):
        n_channel = img.size(0)
        n_group_band = int(np.ceil(n_channel / self.n_group))
        for i in range(self.n_group):
            start = i * n_group_band
            end = start + n_group_band
            if end >= n_channel:
                indx = np.arange(start, n_channel)
                indx_ = np.arange(start, n_channel)
            else:
                indx = np.arange(start, end)
                indx_ = np.arange(start, end)
            np.random.shuffle(indx)
            img[indx_] = img[indx]
        # indx_selected = indx[:n_shuffle]
        # select_mask = np.zeros((n_channel, 1, 1))
        # select_mask[indx_selected] = 1
        # img_shuffled = img[indx]

        return img


class MaskPixels(object):
    def __init__(self, p=0.5):
        """
        :param p:  every pixel will be masked  with a probability of p
        """
        self.p = 1 - p

    def __call__(self, img):
        n_band, h, w = img.shape
        mask = np.random.binomial(1, self.p, size=(h, w))
        mask = torch.from_numpy(mask).float()
        mask = mask.expand((n_band, h, w))
        img = mask * img
        return img


class MaskBands(object):

    def __init__(self, p=0.5):
        """

        :param p: a band will be masked with probability of p
        """
        self.p = 1. - p

    def __call__(self, img):
        # indx = np.arange(img.shape[0])
        # indx_selected = np.random.choice(indx, self.n_band, replace=False)
        # img = img[indx_selected]

        prob = np.random.binomial(1, self.p, img.shape[0])
        prob = np.reshape(prob, (img.shape[0], 1, 1))
        prob = torch.from_numpy(prob).float()
        # img = img[np.where(prob == 1)]
        img = img * prob
        return img


class RandomProjectionBands(object):

    def __init__(self, n_band=None):
        """
        :param n_band: project to n_band
        """
        self.n_band = n_band

    def __call__(self, img):
        # # n_band * w * h
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        n_band, h, w = img.shape
        if self.n_band is None:
            # self.n_band = np.random.randint(3, n_band//2)
            transformer = random_projection.SparseRandomProjection(n_components='auto')
        else:
            transformer = random_projection.SparseRandomProjection(n_components=self.n_band)
        img_ = img.transpose((1, 2, 0))
        x_2d = img_.reshape((-1, n_band))
        x_2d_ = transformer.fit_transform(x_2d)
        img_new = x_2d_.reshape((h, w, -1)).transpose(2, 0, 1)
        img_new = torch.from_numpy(img_new).float()
        return img_new


class ShufflePixel(object):

    def __init__(self):
        pass

    def __call__(self, img):
        n_band, h, w = img.shape
        img_ = img.view(n_band, -1)
        img_ = img_[torch.randperm(n_band)]
        img = img_.view(n_band, h, w)
        return img
