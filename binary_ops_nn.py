# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:59:05 2020

@author: rados
"""
# %% Imports
import torch
from torch.nn.parameter import Parameter
from utils import ORSet, ANDSet, XORSet
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from visualization import DataSpacePlotter, Error3DSpacePlotter, Error2DSpacePlotter
import h5py
import cloudpickle as cpkl


# %% Parameters
# learning_rate = 0.00001
learning_rate = 0.01
num_epochs = 1000
batch_size = 50
dataset_size = 500

dataset_type = "or"

stop_condition = 1e-5
normalize_factor = 0.0
# activation = "step"
activation = "sigma"

vis_type = "3D"

# for 2D visualization
if vis_type == "2D":
    n_components = 2
    samples_per_dim = 30
    RANGE = 30
    range_min = -RANGE
    range_max = RANGE
    cmap_name = "plasma"

    h5_filename = "src/precomp_error_space.hdf5"

    def unpickle_h5(h5_dset):
        return cpkl.loads(h5_dset[0].tobytes())

    useful_tsne_params = ["angle", "early_exaggeration", "init", "learning_rate", "method", "metric", "perplexity"]
    if dataset_type == "or":
        tsne_params = {
            "angle": 0.5,
            "early_exaggeration": 40,
            "init": "random",
            "learning_rate": 10,
            "method": "barnes_hut",
            "metric": "euclidean",
            "perplexity": 40
        }
    elif dataset_type == "and":
        tsne_params = {
            "angle": 0.5,
            "early_exaggeration": 40,
            "init": "random",
            "learning_rate": 10,
            "method": "barnes_hut",
            "metric": "euclidean",
            "perplexity": 40
        }
    elif dataset_type == "xor":
        tsne_params = {
            "angle": 0.5,
            "early_exaggeration": 40,
            "init": "random",
            "learning_rate": 10,
            "method": "barnes_hut",
            "metric": "euclidean",
            "perplexity": 40
        }

    hfile = h5py.File(h5_filename, "r+")
    sample_group = hfile[f"{dataset_type}"][f"[{range_min}-{range_max}]"][f"{samples_per_dim}_samples"]
    losses = sample_group["losses"][...]

    color_group = sample_group[f"c_{cmap_name}"]
    point_colors = color_group["colors"][...]

    pt_mean_dist = sample_group.attrs["pt_mean_dist"]

    ncomp_group = sample_group[f"{n_components}D"]

    embed_group = ncomp_group
    for p_name, p_value in tsne_params.items():
        if p_name in useful_tsne_params:
            if f"{p_name}: {p_value}" in embed_group:
                embed_group = embed_group[f"{p_name}: {p_value}"]
            elif f"{p_name}: {float(p_value)}" in embed_group:
                embed_group = embed_group[f"{p_name}: {float(p_value)}"]
            else:
                raise KeyError(f"The precomputed values do not contain the value {p_value} for the tSNE parameter {p_name}.")

    tsne = unpickle_h5(embed_group["tsne"])
    embeded_points = embed_group["embeded_points"][...]

    embeded_pt_mean_dist = embed_group.attrs["embeded_pt_mean_dist"]


# %% The old school (illustrative) way
class ManualPerceptron(torch.nn.Module):

    def __init__(self, n_dims, activation="sigma", index=-1, device=torch.device(0)):
        super().__init__()
        self.index = index
        self.n_dims = n_dims
        self.activation = activation
        self.device = device
        self.total_dims = self.n_dims + 1  # input dimensionality + 1 for bias
        self.weights = Parameter(torch.randn(self.total_dims, device=self.device), requires_grad=False)  # assign random weight + bias

    def forward(self, x):
        if x.size(-1) == self.n_dims:
            x_1 = torch.functional.F.pad(x, (0, 1), "constant", value=1).float()  # just append "1" to the input for the bias term
        else:
            x_1 = x
        x_1 = x_1.view(-1, self.total_dims)  # enables input of [batch x dim] or just [dim] (i.e. single input)

        # compute "response", i.e. the dot product between the input and weights (plus bias)
        response = torch.mm(self.weights.view(1, self.total_dims), x_1.t())  # a.k.a "sum(w_i * x_i) - bias"
        # compute the activation function
        if self.activation == "step":
            activation = torch.relu(torch.sign(response))  # 0-1 step activation function
        elif self.activation == "sigma":
            activation = torch.sigmoid(response)  # sigmoid activation function
        return activation


# %% Training
def train(net, dataset_loader, n_epochs=5, learning_rate=0.05, stop_condition=0.1, dataset_type="or", normalize_factor=0):
    if vis_type == "1D":
        search_plotter = DataSpacePlotter(net)
    elif vis_type == "2D":
        search_plotter = Error2DSpacePlotter(net, points_2d=embeded_points, losses=losses, colors=point_colors, tsne=tsne)
    elif vis_type == "3D":
        search_plotter = Error3DSpacePlotter(net, operation=dataset_type, activation=net.activation, samples_per_dim=80)
    abort = False

    for epoch in range(n_epochs):
        epoch_error = 0.0
        for batch_no, data in enumerate(dataset_loader, 0):
            X, y_target = data
            X_1 = torch.functional.F.pad(X, (0, 1), "constant", value=1).float()
            batch_size = X.size(0)

            # compute network prediction
            y_predict = net(X_1)
            # error = 1/2(Y_target - Y_predict) ^ 2
            error = (y_target - y_predict)**2 / 2

            # loss = (Y_target - Y_predict) * X
            loss = ((y_target - y_predict).view(1, batch_size) * X_1.t()).t().sum(0)
            # w(t+1) = w(t) + n * loss     (the "normalize_factor" bit is an optional primitive weight regularization)
            net.weights.mul_(1 - normalize_factor).add_(learning_rate * loss)

            epoch_error += error.sum().cpu()
            if vis_type == "1D":
                search_plotter.update(error)
            else:
                if search_plotter.update():
                    print("User requested termination.")
                    abort = True
                    break

        if abort:
            break
        print(f"Epoch {epoch} ended.\n  > epoch loss = {epoch_error:.3f}")
        if vis_type == "1D":
            search_plotter.update_epoch(epoch, epoch_error)

        if epoch_error <= stop_condition:
            print(f"Cumulative loss after epoch {epoch} was {epoch_error}, which satisfies the stopping condition -> terminating training.")
            break
    if vis_type == "1D":
        search_plotter.update(error)
    else:
        search_plotter.destroy()


# %% make a dataset

if dataset_type == "or":
    dataset = ORSet(dataset_size, generate_online=False)
elif dataset_type == "and":
    dataset = ANDSet(dataset_size, generate_online=False)
elif dataset_type == "xor":
    dataset = XORSet(dataset_size, generate_online=False)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# %% Run
perceptron = ManualPerceptron(2, activation=activation)

train(perceptron, train_loader, num_epochs, learning_rate, stop_condition=stop_condition, dataset_type=dataset_type, normalize_factor=normalize_factor)

print(perceptron.weights)

if vis_type == "1D":
    pl.waitforbuttonpress()
