"""
Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
"""

from __future__ import division
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy import sparse as sp
from matplotlib import cm
import numpy as np
import triutils as utils


__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2018, Gilbert Peralta"
__version__ = "1.0"
__maintainer__ = "The author"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "8 July 2021"


class Mesh():
    """
    Abstract finite element mesh classself.

    Credits: https://github.com/kinnala/scikit-fem
    """
    name = "Abstract"
    def __str__(self):
        string = self.name + " mesh with {} nodes and {} cells."
        return string.format(self.num_node, self.num_cell)

    def __repr__(self):
        return self.__str__()

    def dim(self):
        """
        Returns the spatial dimension of the mesh.
        """
        return int(self.node.shape[1])

    def remove_cell(self, cell_indices):
        """
    	Construct new mesh with elements removed.

	Parameter
	---------
	cell_indices : array_like
	    An array of indices of cells to be removed.

	Returns
	-------
	A mesh class with the same type as the object.
	"""
        keep_indices = np.setdiff1d(np.arange(self.cell.shape[0]), cell_indices)
        new_cell = self.cell[keep_indices, :]
        node_keep_indices = np.unique(new_cell)
        reverse = np.zeros(self.node.shape[0])
        reverse[node_keep_indices] = np.arange(len(node_keep_indices))
        new_cell = reverse[new_cell]
        new_node = self.node[node_keep_indices, :]
        if new_node.shape[0] == 0.0:
            raise UserWarning("Mesh contains no nodes!")
        MeshClass = type(self)
        return MeshClass(new_node, new_cell)

    def scale(self, factor):
        """
        Scales the mesh.

        Parameter
        ---------
        factor : float or tuple
            The scaling factor for which the node coordinates will be
            multiplied. When factor is a tuple, the corresponding components
            are multiplied to the associated components of node coordinates.
        """
        for itr in range(self.dim()):
            if isinstance(factor, tuple):
                self.node[:, itr] *= factor[itr]
            else:
                self.node[:, itr] *= factor

    def translate(self, vec):
        """
        Translates the mesh.

        Parameter
        ---------
        vec : array_like
            The direction for which the node coordinates will be translated.
        """
        for itr in range(self.dim()):
            self.node[:, itr] += vec[itr]

    def rotate(self, phi):
        """
        Rotates the mesh around the origin.

        Parameter
        ---------
        phi : float
            The angle (in radians) for which the node coordinates will
            be rotated.
        """
        A = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]])
        self.node = self.node.dot(A)


class MeshTri(Mesh):
    """
    Triangular mesh class.
    """
    name = "Triangular"
    def __init__(self, node, cell):
        """
        Class initialization.

        Attributes
        ----------
        node : numpy.array
            Array consisting of the node coordinates.
        cell : numpy.array
            Array consisting of node indices determined by the triangles.
        num_node : int
            Number of nodes.
        num_cell : int
            Number of cells.
        """
        self.node = node.astype(float)
        self.cell = cell.astype(int)
        self.num_node = node.shape[0]
        self.num_cell = cell.shape[0]

    @staticmethod
    def square_uni_trimesh(n):
        """
        Uniform triangular mesh on the unit square.

        Parameter
        ---------
        n : int
            Number of nodes in a side of the unit square.

        Returns
        -------
        MeshTri class
        """
        cell, node = utils.square_uni_trimesh(n)
        return MeshTri(cell, node)

    @staticmethod
    def sample(n=1):
        """
        Triangular mesh samples.

        Parameter
        ---------
        n : int
            Values are either 0, 1, 2, 3, 4, 5. Default value is 1.

        Returns
        -------
        MeshTri class
        """
        cell, node = utils.sample(n)
        #return MeshTri(cell, node).fix()
        return MeshTri(cell, node)

    def get_cell_center(self):
        """
        Determine triangle barycenters.

        Returns
        -------
        cell_center : numpy.array
            Array consisting of the coordinates of the barycenters of the
            triangles.
        """
        return utils.cell_center(self.node, self.cell)

    def set_cell_center(self):
        """
        Set 'cell_center' attribute to mesh. Refer to instance function
        'MeshTri.get_cell_center()' fo details.
        """
        self.cell_center = self.get_cell_center()

    def get_edge(self, return_cell_to_edge=False):
        """
        Determine edge connectivities.

        Parameters
        ----------
        return_cell_to_edge : bool, optional
            If True, also returns the indices of the edges containg
            each triangles.

        Returns
        -------
        edge : numpy.ndarray
            Array consisting of node indices describing the edges of the
            mesh. Shape is equal to (Ne, 2) where Ne is the number of edges.
        num_edge : int
            Total number of edges, equal to edge.shape[0].
        cell_to_edge : numpy.ndarray
            Array for which if cell_to_edge[i, :] = [j, k, l] then j, k and l
            are the indices of the edges in the ith triangle. Shape is equal
            to (Nt, 3) where Nt = cell.shape[0].
        """
        return utils.edge(self.cell, return_cell_to_edge=return_cell_to_edge)

    def set_edge(self, return_cell_to_edge=False):
        """
        Set 'edge' attribute to mesh. Refer to instance function
        'MeshTri.edge()' for details.
        """
        if return_cell_to_edge:
            self.edge, self.num_edge, self.cell_to_edge \
            = utils.edge(self.cell, return_cell_to_edge=True)
        else:
            self.edge, self.num_edge = utils.edge(self.cell)

    def get_cell_to_edge(self):
        """
        Determine cell to edge mapping. Refer to instance function
        'MeshTri.edge()' for details.
        """
        if hasattr(self, "cell_to_edge"):
            return self.cell_to_edge
        else:
            return self.get_edge(return_cell_to_edge=True)[2]

    def set_cell_to_edge(self):
        """
        Set 'cell_to_edge' attribute to mesh. Refer to instance function
        'MeshTri.edge()' for details.
        """
        self.cell_to_edge = self.get_cell_to_edge()

    def get_edge_coo(self):
        """
        Determine edge coordinates.

        Returns
        -------
        edge_coo : numpy.ndarray
            Array consisting of the node coordinates of the edges.
        """
        if hasattr(self, "edge"):
            return utils.edge_coo(self.node, self.edge)
        else:
            return utils.edge_coo(self.node, self.get_edge()[0])

    def set_edge_coo(self):
        """
        Set 'edge_coo' attribute to mesh. Refer to instance function
        'MeshTri.get_edge_coo()' for details.
        """
        self.edge_coo = self.get_edge_coo()

    def get_edge_vec(self):
        """
        Determine edge vectors.

        Returns
        -------
        edge_vec : numpy.ndarray
            Array consisting of the components of the vectors associated
            with the edges.
        """
        if hasattr(self, "edge_coo"):
            return utils.edge_vec(self.edge_coo)
        else:
            return utils.edge_vec(self.get_edge_coo())

    def set_edge_vec(self):
        """
        Set 'edge_vec' attribute to mesh. Refer to instance function
        'MeshTri.get_edge_vec()' for details.
        """
        self.edge_vec = self.get_edge_vec()

    def get_edge_size(self):
        """
        Determine edge lengths.

        Returns
        -------
        edge_size : numpy.ndarray
            Array consisting of the lengths of the edges in the mesh
        """
        if hasattr(self, "edge_vec"):
            return utils.edge_size(self.edge_vec)
        else:
            return utils.edge_size(self.get_edge_vec())

    def set_edge_size(self):
        """
        Set 'edge_size' attribute to mesh. Refer to instance function
        'MeshTri.get_edge_size()' for details.
        """
        if hasattr(self, "edge_vec"):
            self.edge_size = utils.edge_size(self.edge_vec)
        else:
            self.edge_size = utils.edge_size(self.get_edge_vec())

    def get_size(self):
        """
        Return largest edge length.
        """
        if hasattr(self, "edge_size"):
            return max(self.edge_size)
        else:
            return max(self.get_edge_size())

    def set_size(self):
        """
        Set 'size' attribute to mesh. Refer to instance function
        'MeshTri.get_size()' for details.
        """
        self.size = self.get_size()

    def get_edge_center(self):
        """
        Determine edge midpoints.

        Returns
        -------
        edge_center : numpy.ndarray
            Array consisting of edge midpoint coordinates
        """
        if hasattr(self, "edge"):
            return utils.edge_center(self.node, self.edge)
        else:
            return utils.edge_center(self.node, self.get_edge()[0])

    def set_edge_center(self):
        """
        Set 'edge_center' attribute to mesh. Refer to instance function
        'MeshTri.get_edge_center()' for details.
        """
        self.edge_center = self.get_edge_center()

    def get_edge_to_cell(self):
        """
        Determine edge to cell mapping.

        Returns
        -------
        cell_to_edge : numpy.ndarray
            Array consisting of edge to cell mapping.
        """
        if (hasattr(self, "num_edge") and hasattr(self, "cell_to_edge")):
            return utils.edge_to_cell(self.num_cell, self.num_edge,
                self.cell_to_edge)
        else:
            num_edge, cell_to_edge \
                = self.get_edge(return_cell_to_edge=True)[1:]
            return utils.edge_to_cell(self.num_cell, num_edge, cell_to_edge)

    def set_edge_to_cell(self):
        """
        Set 'edge_to_cell' attribute to mesh. Refer to instance function
        'MeshTri.get_edge_to_cell()' for details.
        """
        self.edge_to_cell = self.get_edge_to_cell()

    def get_bdy_edge_indices(self):
        """
        Determine indices of boundary edges.

        Returns
        -------
        bdy_edge_indices : numpy.ndarray
            Array of indices corresponding to edges on the boundary.
        """
        if hasattr(self, "edge_to_cell"):
            return utils.bdy_edge_indices(self.edge_to_cell)
        else:
            return utils.bdy_edge_indices(self.get_edge_to_cell())

    def set_bdy_edge_indices(self):
        """
        Set 'bdy_edge_indices' attribute to mesh. Refer to instance function
        'MeshTri.get_bdy_edge_indices()' for details.
        """
        self.bdy_edge_indices = self.get_bdy_edge_indices()

    def get_bdy_node_indices(self):
        """
        Determine indices of boundary nodes.

        Returns
        -------
        bdy_node_indices : numpy.ndarray
            Array of indices corresponding to nodes on the boundary.
        """
        if hasattr(self, "bdy_edge_indices"):
            if hasattr(self, "edge"):
                return utils.bdy_node_indices(self.edge, self.bdy_edge_indices)
            else:
                return utils.bdy_node_indices(self.get_edge()[0], self.bdy_edge_indices)
        else:
            return utils.bdy_node_indices(self.get_edge()[0],
                self.get_bdy_edge_indices())

    def set_bdy_node_indices(self):
        """
        Set 'bdy_node_indices' attribute to mesh. Refer to instance function
        'MeshTri.get_node_edge_indices()' for details.
        """
        self.bdy_node_indices = self.get_bdy_node_indices()

    def get_int_node_indices(self):
        """
        Deterime indices of interior nodes.

        Returns
        -------
        int_node_indices : numpy.ndarray
            Array of indices corresponding to nodes in the interior.
        """
        if hasattr(self, "bdy_node_indices"):
            return utils.int_node_indices(self.num_node, self.bdy_node_indices)
        else:
            return utils.int_node_indices(self.num_node, self.get_bdy_node_indices())

    def set_int_node_indices(self):
        """
        Set 'int_node_indices' attribute to mesh. Refer to instance function
        'MeshTri.int_node_edge_indices()' for details.
        """
        self.int_node_indices = self.get_int_node_indices()

    def get_int_edge_indices(self):
        """
        Deterime indices of interior edges.

        Returns
        -------
        bdy_edge_indices : numpy.ndarray
            Array of indices corresponding to edges in the interior.
        """
        if hasattr(self, "num_edge"):
            if hasattr(self, "bdy_edge_indices"):
                return utils.int_edge_indices(self.num_edge, self.bdy_edge_indices)
            else:
                return utils.int_edge_indices(self.get_edge()[1], self.bdy_edge_indices)
        else:
            return utils.int_edge_indices(self.get_edge()[1],
                self.get_bdy_edge_indices())

    def set_int_edge_indices(self):
        """
        Determine node to cell mapping.

        Returns
        -------
        int_edge_indices : numpy.ndarray
            Array of indices corresponding to edges in the interior.
        """
        self.int_edge_indices = self.get_int_edge_indices()

    def get_node_to_cell(self):
        """
        Determine node to cell mapping.

        Returns
        -------
        node_to_cell : scipy.csr_matrix
            Sparse matrix such that the data corresponding to ith row and
            column are the indices of triangles having node with index i.
        """
        return utils.node_to_cell(self.num_node, self.cell)

    def set_node_to_cell(self):
        """
        Set 'node_to_cell' attribute to mesh. Refer to instance function
        'MeshTri.get_node_to_cell()' for details.
        """
        self.node_to_cell = self.get_node_to_cell()

    def get_node_to_edge(self):
        """
        Determine node to edge mapping.

        Returns
        -------
        node_to_edge : scipy.csr_matrix
            Sparse matrix such that the node_to_edge[k, l] = j if the (j-1)st
            edge belongs to the kth and lth nodes.
        """
        if hasattr(self, 'node_to_cell'):
            return utils.node_to_edge(self.num_node, self.node_to_cell)
        else:
            return utils.node_to_edge(self.num_node,
                self.get_node_to_cell())

    def set_node_to_edge(self):
        """
        Set 'node_to_edge' attribute to mesh. Refer to instance function
        'MeshTri.get_node_to_edge()' for details.
        """
        self.node_to_edge = self.get_node_to_edge()

    def plot(self, fignum=1, show=True, node_numbering=False, cell_numbering=False,
        edge_numbering=False, white_background=True, title=None, **kwargs):
        """
        Plots the mesh.

        Parameters
        ----------
        fignum : int, optional
            Figure number window. Default value is 1.
        show : bool, optional
            If True, invoke 'plt.show()'.
        node_numbering : bool, optional
            If True, display node indices on the plot.
        cell_numbering : bool, optional
            If True, display cell indices on the plot.
        edge_numbering : bool, optional
            If True, display edge indices on the plot.
        white_background : bool, optional
            Plots mesh on a white canvas.
        title : str, optional
            Title of the plot.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        utils.assert_positive_integer(fignum)
        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)
        if not white_background:
            ax.tripcolor(self.node[:, 0], self.node[:, 1], self.cell,
                np.ones(self.num_node), vmin=0.5, alpha=0.4, cmap=cm.RdBu)
        ax.triplot(self.node[:, 0], self.node[:, 1], self.cell, **kwargs)
        if node_numbering:
            for itr in range(self.num_node):
                ax.text(self.node[itr, 0], self.node[itr, 1], str(itr),
                bbox=dict(facecolor='blue', alpha=0.25))
        if edge_numbering:
            edge_center = self.get_edge_center()
            for itr in range(edge_center.shape[0]):
                ax.text(edge_center[itr, 0], edge_center[itr, 1], str(itr),
                bbox=dict(facecolor='green', alpha=0.25))
        if cell_numbering:
            cell_center = self.get_cell_center()
            for itr in range(self.num_cell):
                ax.text(cell_center[itr, 0], cell_center[itr, 1], str(itr),
                bbox=dict(facecolor='red', alpha=0.25))
        plt.gca().set_aspect('equal')
        plt.box(False)
        plt.axis('off')
        if title == None:
            title = self.__str__()
        plt.title(title, fontsize=11)
        if show:
            plt.show()
        return ax, fig

    def set_attribute(self, *args):
        """
        Set attributes in the list of names in args. For more details,
        do help(utils.help_attribute_to_instance_linker()).
        """
        for name in args:
            try:
                utils.attribute_to_instance_linker(self)[name]()
            except KeyError:
                msg = "Got an invalid attribute name '{}'.".format(name)
                raise UserWarning(msg)

    def min_angle(self):
        """
        Returns the smallest interior angle (in degrees) of the triangles.

        Returns
        -------
        angle : float
            The minimum interior angle among all triangles.
        """
        theta = np.pi
        edge1 = self.node[self.cell[:, 0], :] - self.node[self.cell[:, 1], :]
        edge2 = self.node[self.cell[:, 1], :] - self.node[self.cell[:, 2], :]
        edge3 = self.node[self.cell[:, 2], :] - self.node[self.cell[:, 0], :]
        e1_norm = np.linalg.norm(edge1, axis=1)
        e2_norm = np.linalg.norm(edge2, axis=1)
        e3_norm = np.linalg.norm(edge3, axis=1)
        theta1 = np.arccos(- np.sum(edge1 * edge2, axis=1) / (e1_norm * e2_norm))
        theta2 = np.arccos(- np.sum(edge2 * edge3, axis=1) / (e2_norm * e3_norm))
        theta3 = np.arccos(- np.sum(edge3 * edge1, axis=1) / (e3_norm * e1_norm))
        theta = min(np.array([theta1, theta2, theta3]).flatten())
        return theta * 180. / np.pi

    def _refine(self):
        """
        Main function for the uniform mesh refinement via bisection.
        Refer to instance 'MeshTri.refine()' for details.
        """
        # update node
        new_node = np.append(self.node, self.get_edge_center(), axis=0)
        if hasattr(self, 'cell_to_edge'):
            pass
        else:
            self.set_attribute('cell_to_edge')
        # update triangle list
        new_cell = np.zeros((4*self.num_cell, 3), dtype=np.int)
        new_cell[:self.num_cell, :] \
            = np.vstack([self.cell[:, 0],
            self.num_node + self.cell_to_edge[:, 2],
            self.num_node + self.cell_to_edge[:, 1]]).T
        new_cell[self.num_cell:2*self.num_cell, :] \
            = np.vstack([self.cell[:, 1],
            self.num_node + self.cell_to_edge[:, 0],
            self.num_node + self.cell_to_edge[:, 2]]).T
        new_cell[2*self.num_cell:3*self.num_cell, :] \
            = np.vstack([self.cell[:, 2],
            self.num_node + self.cell_to_edge[:, 1],
            self.num_node + self.cell_to_edge[:, 0]]).T
        new_cell[3*self.num_cell:, :] \
            = np.vstack([self.num_node + self.cell_to_edge[:, 0],
            self.num_node + self.cell_to_edge[:, 1],
            self.num_node + self.cell_to_edge[:, 2]]).T
        return MeshTri(new_node, new_cell)

    def refine(self, level=1):
        """
        Refines the triangulation by creating four triangles on each
        triangle connecting the edge midpoints.

        Parameter
        ---------
        level : int, optional
            Number of refinements to be done. Default values is 1, that is,
            single refinement.

        Returns
        -------
        MeshTri class
        """
        # utils.assert_positive_integer(level)
        ref_mesh = self
        for it in range(level):
            ref_mesh = ref_mesh._refine()
        return MeshTri(ref_mesh.node, ref_mesh.cell)

    def regularize(self, info=False):
        """
        Barycentric regularization of the triangulation.

        Returns
        -------
        MeshTri class

        Reference
        ---------
        Quarteroni, Numerical Mathematics, Springer.

        To Do
        -----
        Include fixed interior nodes.

        Notes
        -----
        Modify to next neighborhood barycentric regularization.
        """
        self.set_attribute('bdy_node_indices', 'int_node_indices')
        node_to_edge_indptr = self.get_node_to_edge().indptr
        row_indices = self.cell.flat
        col_indices = self.cell[:, [1, 2, 0]].flat
        data = np.array([-1]*(3*self.num_cell))
        A = sp.coo_matrix((data, (row_indices, col_indices)),
            shape=(self.num_node, self.num_node), dtype=np.float).tocsr()
        # number of edges in a given node
        data = node_to_edge_indptr[1:] - node_to_edge_indptr[:-1]
        A = A + sp.coo_matrix((data[self.int_node_indices],
            (self.int_node_indices, self.int_node_indices)),
            shape=(self.num_node, self.num_node), dtype=np.float).tocsr()
        Mat = A[self.int_node_indices, :][:, self.bdy_node_indices]
        A = A[self.int_node_indices, :][:, self.int_node_indices]
        new_node = self.node.copy()
        new_node[self.int_node_indices, 0] \
            = spsolve(A, - Mat * self.node[:, 0][self.bdy_node_indices])
        new_node[self.int_node_indices, 1] \
            = spsolve(A, - Mat * self.node[:, 1][self.bdy_node_indices])
        if info:
            print("\nBARYCENTRIC REGULARIZATION\n")
            print("Sparse Matrix Dimension: {}".format(A.shape))
            print("Sparse Matrix Density:   {:.12e}\n".format(A.nnz / (A.shape[0]**2)))
        return MeshTri(new_node, self.cell).fix()

    def adaptive_refine(self, marked_cell=None):
        """
        Refine the mesh according to the set of cell indices using the
        Red-Blue-Green Refinement Method.

        Credits: https://github.com/kinnala/scikit-fem

        Parameters
        ----------
        marked_cell : array_like, optional
            Array of cell indices to be refined.

        Returns
        -------
        MeshTri class
        """
        def sort_mesh(node, cell):
            """
            Make (0, 2) the longest edge in cell.
            """
            l01 = np.sqrt(np.sum((
                node[cell[:, 0], :] - node[cell[:, 1], :])**2, axis=1))
            l12 = np.sqrt(np.sum((
                node[cell[:, 1], :] - node[cell[:, 2], :])**2, axis=1))
            l02 = np.sqrt(np.sum((
                node[cell[:, 0], :] - node[cell[:, 2], :])**2, axis=1))
            ix01 = (l01 > l02) * (l01 > l12)
            ix12 = (l12 > l01) * (l12 > l02)
            # row swaps
            tmp = cell[ix01, 2]
            cell[ix01, 2] = cell[ix01, 1]
            cell[ix01, 1] = tmp
            tmp = cell[ix12, 0]
            cell[ix12, 0] = cell[ix12, 1]
            cell[ix12, 1] = tmp
            return cell

        def find_edges(mesh, marked_cell):
            """
            Find the edges to split.
            """
            try:
                edges = np.zeros(mesh.num_edge, dtype=np.int64)
            except AttributeError:
                mesh.set_attribute('edge', 'cell_to_edge')
                # TO FIX
                mesh.cell_to_edge = mesh.cell_to_edge[:, [2, 0, 1]]
                edges = np.zeros(mesh.num_edge, dtype=np.int64)
            edges[mesh.cell_to_edge[marked_cell, :].flatten()] = 1
            prev_nnz = -1e10

            while np.count_nonzero(edges) - prev_nnz > 0:
                prev_nnz = np.count_nonzero(edges)
                cell_to_edges = edges[mesh.cell_to_edge]
                cell_to_edges[cell_to_edges[:, 0]
                    + cell_to_edges[:, 1] > 0, 2] = 1
                edges[mesh.cell_to_edge[cell_to_edges == 1]] = 1
            return edges

        def split_cells(mesh, edges):
            """ Define new elements. """
            ix = (-1) * np.ones(mesh.num_edge, dtype=np.int64)
            ix[edges == 1] = np.arange(np.count_nonzero(edges)) \
                + mesh.num_node
            # (0, 1) (1, 2) (0, 2)
            ix = ix[mesh.cell_to_edge]
            red =   (ix[:, 0] >= 0) * (ix[:, 1] >= 0) * (ix[:, 2] >= 0)
            blue1 = (ix[:, 0] ==-1) * (ix[:, 1] >= 0) * (ix[:, 2] >= 0)
            blue2 = (ix[:, 0] >= 0) * (ix[:, 1] ==-1) * (ix[:, 2] >= 0)
            green = (ix[:, 0] ==-1) * (ix[:, 1] ==-1) * (ix[:, 2] >= 0)
            rest =  (ix[:, 0] ==-1) * (ix[:, 1] ==-1) * (ix[:, 2] ==-1)
            # new red elements
            cell_red \
            = np.hstack([
            np.vstack([mesh.cell[red, 0], ix[red, 0], ix[red, 2]]),
            np.vstack([mesh.cell[red, 1], ix[red, 0], ix[red, 1]]),
            np.vstack([mesh.cell[red, 2], ix[red, 1], ix[red, 2]]),
            np.vstack([       ix[red, 1], ix[red, 2], ix[red, 0]]),
            ]).T
            # new blue elements
            cell_blue1 \
            = np.hstack([
            np.vstack([mesh.cell[blue1, 1], mesh.cell[blue1, 0], ix[blue1, 2]]),
            np.vstack([mesh.cell[blue1, 1], ix[blue1, 1], ix[blue1, 2]]),
            np.vstack([mesh.cell[blue1, 2], ix[blue1, 2], ix[blue1, 1]]),
            ]).T
            cell_blue2 \
            = np.hstack([
            np.vstack([mesh.cell[blue2, 0], ix[blue2, 0], ix[blue2, 2]]),
            np.vstack([       ix[blue2, 2], ix[blue2, 0], mesh.cell[blue2, 1]]),
            np.vstack([mesh.cell[blue2, 2], ix[blue2, 2], mesh.cell[blue2, 1]]),
            ]).T
            # new green elements
            cell_green \
            = np.hstack([
            np.vstack([mesh.cell[green, 1], ix[green, 2], mesh.cell[green, 0]]),
            np.vstack([mesh.cell[green, 2], ix[green, 2], mesh.cell[green, 1]]),
            ]).T
            refined_edges = mesh.edge[edges == 1, :]
            # new nodes
            node = 0.5 * (mesh.node[refined_edges[:, 0], :]
                  + mesh.node[refined_edges[:, 1], :])
            return np.vstack([mesh.node, node]),\
                   np.vstack([mesh.cell[rest, :], cell_red, cell_blue1,
                        cell_blue2, cell_green]), refined_edges
        sorted_mesh = MeshTri(self.node, sort_mesh(self.node, self.cell))
        edges = find_edges(sorted_mesh, marked_cell)
        node, cell, self.refined_edges = split_cells(sorted_mesh, edges)
        return MeshTri(node, cell).fix()

    def cell_refine(self, marked_cell):
        """
        Same as the instance 'MeshTri.adaptive_refine()'.
        """
        return self.adaptive_refine(marked_cell)

    def node_refine(self, marked_node, level=1):
        """
        Refine the mesh according to the set of cells containing the
        indices of the input nodes.

        Parameters
        ----------
        marked_node : array_like, optional
            Array of node indices to be refined.
        level : int, optional
            Number of refinements to be done. Default values is 1, that is,
            single refinement.

        Returns
        -------
        MeshTri class
        """
        ref_mesh = self
        for itr in range(level):
            if not hasattr(ref_mesh, "node_to_cell"):
                ref_mesh.set_attribute("node_to_cell")
            marked_cell = ref_mesh.node_to_cell[marked_node, :].data
            marked_cell = np.append(marked_cell,
                ref_mesh.node_to_cell[:, marked_node].data) - 1
            marked_cell = np.unique(marked_cell)
            ref_mesh = ref_mesh.adaptive_refine(marked_cell)
        return ref_mesh

    def edge_refine(self, marked_edge):
        """
        Refine the mesh according to the set of cells containing the
        indices of the input edges.

        Parameters
        ----------
        marked_edge : array_like, optional
            Array of edge indices to be refined.

        Returns
        -------
        MeshTri class
        """
        if not hasattr(self, "edge_to_cell"):
            self.set_attribute("edge_to_cell")
        marked_cell = self.edge_to_cell[marked_edge, :].flatten()
        marked_cell = marked_cell[marked_cell >= 0]
        return self.adaptive_refine(marked_cell)

    def signed_area(self):
        """
        Determine the array of signed cell areas.

        Returns
        -------
        signed_area : numpy.ndarray
            Array consisting of signed areas of the triangles.
        """
        return utils.signed_area(self.node, self.cell)

    def fix(self, tol=2e-13):
        """
        Remove duplicated or unused nodes and fix element orientation.

        Credits: https://github.com/bfroehle/pydistmesh

        Parameter
        ---------
        tol : float, optional
            Tolerance value for fixing.

        Returns
        -------
        MeshTri class
        """
        snap = tol * (self.node.max(0) - self.node.min(0)).max()
        _, index, inv_index = np.unique(snap * np.vectorize(round)(self.node / snap),
            axis=0, return_index=True, return_inverse=True)
        self.node = self.node[index, :]
        self.cell = inv_index[self.cell]
        flip = (utils.signed_area(self.node, self.cell) < 0)
        self.cell[flip, :2] = self.cell[flip, 1::-1]
        return MeshTri(self.node, self.cell)

    def boundary_projection(self, dist_function, eps=None):
        """
        Boundary projection of mesh.

        Credits: https://github.com/bfroehle/pydistmesh

        Parameters
        ----------
        dist_function : callable
            Distance function determining the interior of the mesh.
        eps : float, optional
            The value of numerical gradient step size. If None, will
            return the the product of the square root of machine epsilon
            and the diameter of the region enclosing the mesh.

        Returns
        -------
        MeshTri class
        """
        if eps == None:
            eps = np.sqrt(np.finfo(np.double).eps)*max(self.node.max(0)
                - self.node.min(0))
        bdy_node_indices = self.get_bdy_node_indices()
        node_dist_bdy = dist_function(self.node[bdy_node_indices, :])
        dgradx = (dist_function(self.node[bdy_node_indices, :] + [eps, 0])
            - node_dist_bdy) / eps
        dgrady = (dist_function(self.node[bdy_node_indices, :] + [0, eps])
            - node_dist_bdy) / eps
        dgrad2 = dgradx**2 + dgrady**2
        dgrad2[dgrad2 == 0] = 1.
        self.node[bdy_node_indices, :] \
            -= np.vstack([node_dist_bdy * dgradx / dgrad2,
            node_dist_bdy * dgrady / dgrad2]).T
        return MeshTri(self.node, self.cell).fix()

    def quality(self):
        """
        Deterime the array of triangle qualities.

        Returns
        -------
        quality : numpy.ndarray
            Array consisting of ratios between radii of circumcircle to
            incircle inside the triangle.
        """
        if hasattr(self, "edge_size"):
            if hasattr(self, "cell_to_edge"):
                mesh_quality = utils.quality(self.edge_size, self.cell_to_edge)
            else:
                mesh_quality = utils.quality(self.edge_size, self.get_cell_to_edge())
        else:
            mesh_quality = utils.quality(self.get_edge_size(), self.get_cell_to_edge())
        return mesh_quality

    def plot_quality(self, fignum=2, show=True):
        """
        Plots the mesh qualities in a histogram.

        Parameters
        ----------
        fignum : int, optional
            Figure number window. Default value is 2.
        show : bool, optional
            If True, invoke 'plt.show()'.

        To Fix
        ------
        Proper plot when mesh quality has only one unique value.
        """
        utils.assert_positive_integer(fignum)
        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)
        ax.hist(self.quality(), facecolor='blue', alpha=0.7)
        ax.set(xlim=(0, 1))
        plt.title(self)
        plt.xlabel("Quality")
        plt.ylabel("Number")
        if show:
            plt.show()
        return ax
        