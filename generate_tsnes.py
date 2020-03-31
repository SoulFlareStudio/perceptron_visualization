# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:56:42 2020

@author: rados
"""

import matplotlib
from matplotlib import cm as cmx
import numpy as np
import torch
import open3d as o3d
import pylab as pl
from sklearn.manifold import TSNE
from itertools import product
import h5py
import cloudpickle as cpkl

import argparse


# %%
def pickle4h5(obj):
    return np.frombuffer(cpkl.dumps(obj), dtype=np.uint8)


def unpickle_h5(h5_dset):
    return cpkl.loads(h5_dset[0].tobytes())


xx = [0, 1]
val_list = list(product(xx, xx))

RANGE = 30
range_min = -RANGE
range_max = RANGE
mark_size = 0.2
n_components = 2
cmap_name = "plasma"

useful_tsne_params = ["angle", "early_exaggeration", "init", "learning_rate", "method", "metric", "perplexity"]
tsne_params = {
    "angle": 0.5,
    "early_exaggeration": 12,
    "init": "random",
    "learning_rate": 200,
    "method": "barnes_hut",
    "metric": "euclidean",
    "perplexity": 30
}

# %% Compute
# for operation in ["xor"]:
for operation in ["and", "xor"]:
    print(f">>> Operation '{operation}' <<<")
    for samples_per_dim in [30]:
        print(f"\t=== Samples per dimension: {samples_per_dim} ===")
        device = torch.device(0)
        data = torch.functional.F.pad(torch.tensor(val_list, dtype=torch.float32, device=device), (0, 1), "constant", value=1)
        weights = torch.tensor(np.mgrid[range_min:range_max:samples_per_dim * 1j,
                                        range_min:range_max:samples_per_dim * 1j,
                                        range_min:range_max:samples_per_dim * 1j], dtype=torch.float32, device=device).view(3, -1).t()
        if operation == "or":
            targets = [0, 1, 1, 1]
        elif operation == "and":
            targets = [0, 0, 0, 1]
        elif operation == "xor":
            targets = [0, 1, 1, 0]
        targets = torch.tensor(targets, dtype=torch.float, device=device)

        responses = torch.mm(weights, data.t())
        activations = torch.sigmoid(responses)
        losses = torch.div(torch.pow(torch.sub(targets, activations), 2), 2).mean(1)
        error_space = torch.cat((weights, losses.view(-1, 1)), 1)

        weights_np = weights.cpu().numpy()
        losses_np = losses.cpu().numpy()
        error_space_np = error_space.cpu().numpy()

        cNorm = matplotlib.colors.Normalize(vmin=losses.min(), vmax=losses.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=pl.get_cmap(cmap_name))

        error_pcl = o3d.geometry.PointCloud()
        error_pcl.points = o3d.utility.Vector3dVector(weights_np)
        point_colors = scalarMap.to_rgba(losses_np)[:, :-1]
        pt_mean_dist = np.mean(error_pcl.compute_nearest_neighbor_distance())

        for n_components in [2]:
            for tsne_perplexity in np.r_[40:60:3j]:
                for tsne_early_exaggeration in np.r_[5:40:2j]:
                    for tsne_learning_rate in np.r_[10:400:3j]:
                        # for tsne_metric in ["euclidean", "cosine"]:
                        for tsne_init in ["random", "pca"]:
                            # for tsne_angle in [0.1, 0.2, 0.5]:
                            try:
                                tsne = TSNE(
                                    n_components,
                                    perplexity=tsne_perplexity,
                                    early_exaggeration=tsne_early_exaggeration,
                                    learning_rate=tsne_learning_rate,
                                    # metric=tsne_metric,
                                    init=tsne_init,
                                    # angle=tsne_angle,
                                    n_jobs=-1)
                                print(f"\t\t> Computing tSNE with params:\n\t\t{tsne.get_params()}")
                                embeded_points = tsne.fit_transform(error_space_np)

                                embeded_pcl = o3d.geometry.PointCloud()
                                embeded_pcl.points = o3d.utility.Vector3dVector(np.r_["1,2,0", embeded_points, np.zeros((embeded_points.shape[0], 3 - embeded_points.shape[1]))])
                                embeded_pt_mean_dist = np.mean(embeded_pcl.compute_nearest_neighbor_distance())

                                # %% Save
                                with h5py.File(f"test.hdf5", "a") as hfile:
                                    vlen_uint8 = h5py.vlen_dtype("uint8")

                                    op_group = hfile.require_group(f"{operation}")
                                    range_group = op_group.require_group(f"[{range_min}-{range_max}]")
                                    sample_group = range_group.require_group(f"{samples_per_dim}_samples")
                                    if "points" not in sample_group:
                                        sample_group.require_dataset("points", shape=weights_np.shape, dtype=weights_np.dtype)[...] = weights_np
                                        sample_group.require_dataset("losses", shape=losses_np.shape, dtype=losses_np.dtype)[...] = losses_np
                                        sample_group.require_dataset("error_space", shape=error_space_np.shape, dtype=error_space_np.dtype)[...] = error_space_np
                                        sample_group.attrs["pt_mean_dist"] = pt_mean_dist

                                    color_group = sample_group.require_group(f"c_{cmap_name}")
                                    if "scalarMap" not in color_group:
                                        color_group.require_dataset("scalarMap", shape=(1, ), dtype=vlen_uint8)[0] = pickle4h5(scalarMap)
                                        color_group.require_dataset("colors", shape=point_colors.shape, dtype=point_colors.dtype)[...] = point_colors

                                    ncomp_group = sample_group.require_group(f"{n_components}D")
                                    embed_group = ncomp_group
                                    for p_name, p_value in tsne.get_params().items():
                                        if p_name in useful_tsne_params:
                                            embed_group = embed_group.require_group(f"{p_name}: {p_value}")
                                    embed_group.require_dataset("tsne", shape=(1, ), dtype=vlen_uint8)[0] = pickle4h5(tsne)
                                    embed_group.require_dataset("embeded_points", shape=embeded_points.shape, dtype=embeded_points.dtype)[...] = embeded_points
                                    embed_group.attrs["embeded_pt_mean_dist"] = embeded_pt_mean_dist
                            except Exception as e:
                                print(f"Failed for some reason {e}")
