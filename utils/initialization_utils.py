import torch
from sklearn.cluster import MiniBatchKMeans
from utils.metric import cluster_accuracy
import hashlib
import random
import numpy as np


def make_torch_generator(seed):
    """Create an isolated CPU generator for a reproducible DataLoader stage."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_dataloader_worker(worker_id):
    """Seed NumPy/Python augmentations from the worker seed assigned by PyTorch."""
    del worker_id  # The unique seed is already encoded in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def tensor_sha256(tensor):
    """Return a stable fingerprint used to compare initialized cluster centers."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def init_centers(model, dataloader, n_class, device, is_labeled_pixel, seed=42,
                 evaluate_labels=False, init_sample_size=8192):
    """Fit centers on unaugmented pretrained embeddings without retaining them all in RAM."""
    model.eval()
    init_sample_size = max(3 * n_class, min(int(init_sample_size), len(dataloader.dataset)))
    clustering_model = MiniBatchKMeans(
        n_clusters=n_class, init='k-means++', random_state=seed,
        batch_size=max(1024, dataloader.batch_size or 1024),
        init_size=init_sample_size, n_init=10,
    )
    initialization_features = []
    initialization_count = 0
    initialized = False
    for i, (x, y) in enumerate(dataloader):
        x_list = [x_i.to(device) for x_i in x]
        with torch.no_grad():
            h = model.forward_embedding(x_list)
        h_numpy = h.cpu().numpy()
        if not initialized:
            initialization_features.append(h_numpy)
            initialization_count += h_numpy.shape[0]
            if initialization_count >= init_sample_size or i == len(dataloader) - 1:
                initial_batch = np.concatenate(initialization_features, axis=0)
                clustering_model.partial_fit(initial_batch)
                initialization_features.clear()
                initialized = True
        else:
            clustering_model.partial_fit(h_numpy)

    acc = kappa = nmi = ari = pur = float("nan")
    ca = np.asarray([], dtype=float)
    if evaluate_labels:
        labels_vector, y_pred_vector = [], []
        for x, y in dataloader:
            x_list = [x_i.to(device) for x_i in x]
            with torch.no_grad():
                h = model.forward_embedding(x_list)
            y_pred_vector.extend(clustering_model.predict(h.cpu().numpy()))
            labels_vector.extend(y.numpy())
        labels_vector = np.asarray(labels_vector)
        y_pred_vector = np.asarray(y_pred_vector)
        if is_labeled_pixel:
            acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels_vector, y_pred_vector)
        else:
            indx_labeled = np.nonzero(labels_vector)[0]
            y = labels_vector[indx_labeled]
            y_pred = y_pred_vector[indx_labeled]
            acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y, y_pred)
        print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(
            clustering_model.n_steps_, acc, clustering_model.cluster_centers_.shape))
    else:
        print("Iterations:{}, centers:{} (label evaluation skipped)".format(
            clustering_model.n_steps_, clustering_model.cluster_centers_.shape))
    centers = torch.from_numpy(clustering_model.cluster_centers_)
    return centers, acc, kappa, nmi, ari, pur, ca


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
