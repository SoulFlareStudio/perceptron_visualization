import numpy as np
from itertools import product
import open3d as o3d
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm as cmx
from scipy.spatial.transform.rotation import Rotation
import torch


class DataSpacePlotter():

    def __init__(self, net, operation=""):
        self.net = net
        self.xx = [0, 1]
        self.xx_ext = [-10, 10]
        self.loss_plot = None
        self.loss_line = None
        self.loss_annot = None
        self.loss_index = 0
        self.operation = operation

        self.__drawSpace()

    def lineq(self):
        return (lambda a, b, c: lambda x: -(a / b) * x - c / b)(*self.net.weights.cpu().data.numpy())

    def __drawSpace(self):
        pts = list(product(self.xx, self.xx))

        fig, axes = plt.subplots(1, 2)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].axvline(x=0, color='k', linewidth=0.5)
        axes[0].set_title(self.operation)

        axes[0].scatter(*list(zip(*pts)))

        for pt in pts:
            axes[0].annotate(f"{pt}", pt, xytext=(5, 5), textcoords="offset points")

        yy = [self.lineq()(x).item() for x in self.xx]
        self.line = axes[0].plot(self.xx, yy)[0]
        self.scatter = axes[0].scatter(self.xx, yy)
        self.aw1 = axes[0].annotate(f"weights", (0, -0.5), xytext=(0, 0), textcoords="offset points")

        self.loss_ax = axes[1]
        self.loss_ax.autoscale(enable=True, axis="x", tight=True)
        self.loss_ax.axhline(y=0, linewidth=0.5, color="k")

        plt.ion()
        axes[0].set_xlim(-0.1, 1.2)
        axes[0].set_ylim(-2, 3)

        fig.show()

    def update(self, loss):
        self.line.set_data(*[
            self.xx_ext,
            [self.lineq()(x) for x in self.xx_ext]
        ])
        self.scatter.set_offsets(np.transpose([
            self.xx,
            [self.lineq()(x) for x in self.xx]
        ]))
        self.aw1.set_text("W1 = {:.3f}; W2 = {:.3f}; Th = {:.3f}".format(*self.net.weights))
        plt.draw()
        plt.pause(0.05)

        loss = loss.sum().item()
        if self.loss_plot is None:

            self.loss_plot = self.loss_ax.plot(loss)[0]
            self.loss_ax.set_ylim(-0.2, loss * 1.1)

            self.loss_line = self.loss_ax.axhline(y=loss, linewidth=0.5, color="g")
            self.loss_annot = self.loss_ax.annotate(f"total error: {loss:.4f}", (self.loss_index, loss), xytext=(3, 0), textcoords="offset points")
        else:
            index_list, loss_list = self.loss_plot.get_data()
            self.loss_index = index_list[-1] + 1
            index_list = np.r_[index_list, self.loss_index]
            loss_list = np.r_[loss_list, loss]
            self.loss_plot.set_data((index_list, loss_list))
            self.loss_ax.relim()
            self.loss_ax.autoscale_view()

            self.loss_line.set_ydata([loss] * 2)
            self.loss_annot.set_text(f"total error: {loss:.4f}")
            self.loss_annot.xy = (self.loss_index, loss)

    def update_epoch(self, epoch, loss):
        # self.update(loss)
        self.loss_ax.axvline(x=self.loss_index, linewidth=0.5, color="r")
        self.loss_ax.annotate(f"E {epoch}", (self.loss_index, 0), xytext=(-20, -10), textcoords="offset points")


class Error3DSpacePlotter():
    WIDTH = 1900
    HEIGHT = 980
    RANGE = 20
    MARK_SIZE = 0.2

    def __init__(self, net, fov=90, **kwargs):
        self.net = net
        self.terminate = False

        self.marker = o3d.geometry.TriangleMesh.create_octahedron(self.MARK_SIZE)
        self.marker.paint_uniform_color(np.r_[0, 1, 0.1])

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Perceptron Error Space", width=self.WIDTH, height=self.HEIGHT)
        self.vis.register_key_callback(ord(" "), self.esc_cb)
        ctr = self.vis.get_view_control()
        ctr.change_field_of_view(step=fov)

        self.vis.add_geometry(self.marker)
        self.vis.get_render_option().show_coordinate_frame = True
        self.vis.get_render_option().line_width = 10.0

        self.arrows = [o3d.geometry.TriangleMesh.create_arrow(self.MARK_SIZE * 0.6, self.MARK_SIZE * 1, self.MARK_SIZE * 4, self.MARK_SIZE * 2, 10, 2) for i in range(6)]
        rotations = [
            [0, 90, 0],
            [-90, 0, 0],
            [0, 0, 0],
            [0, -90, 0],
            [90, 0, 0],
            [-180, 0, 0],
        ]
        for i, arrow in enumerate(self.arrows):
            c = [0] * 3
            c[i % 3] = 1
            arrow.paint_uniform_color(np.r_[c])
            arrow.rotate(Rotation.from_euler("xyz", rotations[i], degrees=True).as_dcm())
            self.vis.add_geometry(arrow)

        points = np.r_["0,2,1",
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       ] * self.RANGE * 1.5
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
        ]
        colors = np.diag(np.ones(3))
        axis = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        axis.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(axis)

        self._init_error_space(**kwargs)

    def _init_error_space(self, **kwargs):
        xx = [0, 1]
        val_list = list(product(xx, xx))
        range_min = -self.RANGE
        range_max = self.RANGE

        data = torch.functional.F.pad(torch.tensor(val_list, dtype=torch.float32), (0, 1), "constant", value=1)
        weights = torch.tensor(np.mgrid[range_min:range_max:kwargs["samples_per_dim"] * 1j,
                                        range_min:range_max:kwargs["samples_per_dim"] * 1j,
                                        range_min:range_max:kwargs["samples_per_dim"] * 1j], dtype=torch.float32).view(3, -1).t()
        if kwargs["operation"] == "or":
            targets = [0, 1, 1, 1]
        elif kwargs["operation"] == "and":
            targets = [0, 0, 0, 1]
        elif kwargs["operation"] == "xor":
            targets = [0, 1, 1, 0]
        targets = torch.tensor(targets, dtype=torch.float)

        responses = torch.mm(weights, data.t())
        if kwargs["activation"] == "sigma":
            activations = torch.sigmoid(responses)
        elif kwargs["activation"] == "step":
            activations = torch.relu(torch.sign(responses))
        losses = torch.div(torch.pow(torch.sub(targets, activations), 2), 2).mean(1)

        cNorm = matplotlib.colors.Normalize(vmin=losses.min(), vmax=losses.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("plasma"))

        error_points = o3d.geometry.PointCloud()
        error_points.points = o3d.utility.Vector3dVector(weights)
        error_points.colors = o3d.utility.Vector3dVector(scalarMap.to_rgba(losses.data.numpy())[:, :-1])
        pt_dist = np.mean(error_points.compute_nearest_neighbor_distance())

        error_volume = o3d.geometry.VoxelGrid.create_from_point_cloud(error_points, pt_dist)

        self.vis.add_geometry(error_volume)

    def esc_cb(self, vis):
        self.terminate = True

    def _compute_marker_position(self):
        return self.net.weights.cpu().numpy()

    def update(self):
        xyz = self._compute_marker_position()
        self.marker.translate(xyz, relative=False)
        for arrow, coord in zip(self.arrows, np.concatenate([np.diag(np.ones(3)) * (-xyz + (self.RANGE + 1) * i) + xyz for i in [-1, 1]])):
            arrow.translate(coord, relative=False)
            self.vis.update_geometry(arrow)
        self.vis.update_geometry(self.marker)
        self.vis.poll_events()
        self.vis.update_renderer()
        return self.terminate

    def destroy(self):
        self.vis.destroy_window()


class Error2DSpacePlotter(Error3DSpacePlotter):

    def __init__(self, net, fov=90, **kwargs):
        # points_2d, losses, colors, tsne,
        self.tsne = kwargs["tsne"]

        super().__init__(net, fov=fov, **kwargs)

    def _init_error_space(self, **kwargs):
        points_3d = np.concatenate((kwargs["points_2d"], kwargs["losses"].reshape(-1, 1) * 80), axis=1)
        error_pcl = o3d.geometry.PointCloud()
        error_pcl.points = o3d.utility.Vector3dVector(points_3d)
        error_pcl.colors = o3d.utility.Vector3dVector(kwargs["colors"])
        pt_dist = np.mean(error_pcl.compute_nearest_neighbor_distance())

        error_volume = o3d.geometry.VoxelGrid.create_from_point_cloud(error_pcl, pt_dist * 2)

        self.vis.add_geometry(error_volume)

    def _compute_marker_position(self):
        # return self.tsne.transform(self.net.weights.cpu().numpy())
        return self.net.weights.cpu().numpy()
