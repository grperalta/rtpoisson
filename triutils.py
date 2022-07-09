"""
Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
"""

from __future__ import division
from scipy import sparse as sp
import numpy as np


__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2018, Gilbert Peralta"
__version__ = "1.0"
__maintainer__ = "The author"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "8 July 2021"


def assert_positive_integer(n):
    correct_input = (n > 0 and type(n) == int)
    if not correct_input:
        raise ValueError("Input must be a positive integer.")


def square_uni_trimesh(n):
    """
    Generates a uniform triangulation of the unit square with
    vertices at (0, 0), (1, 0), (0, 1) and (1, 1).

    Parameter
    ---------
    n : int
        Number of nodes in a side of the unit square.

    Returns
    -------
    node : numpy.ndarray
        Array consisting of the node coordinates with shape (Np, 2),
        where Np = 2n^2.
    cell : numpy.ndarray
        Array consisting of node indices describing the triangle
        connectivities with shape (Nt, 3), where Nt = 2(n-1)^2.
    """
    assert_positive_integer(n)
    # number of elements
    numelem = 2*(n-1)**2
    # pre-allocation of node array
    node = np.zeros((n**2, 2)).astype(float)
    # generation of node list
    for i in range(1, n+1):
        for j in range(1, n+1):
            # node index
            index = (i-1)*n + j - 1
            # x-coordinates of a node
            node[index, 0] = (j-1) / (n-1)
            # y-coordinate of a node
            node[index, 1] = (i-1) / (n-1)
    # pre-allocation of node connectivity
    cell = np.zeros((numelem, 3)).astype(int)
    ctr = 0
    for i in range(n-1):
        for j in range(n-1):
            # lower right node of square determined by two triangles
            lr_node = i*n + j + 1
            # lower left triangle
            cell[ctr, :] = [lr_node, lr_node+n, lr_node-1]
            #upper right triangle
            cell[ctr+1, :] = [lr_node+n-1, lr_node-1, lr_node+n]
            ctr += 2
    return node, cell


def sample(n=1):
    """
    Examples of coarse triangulations for the unit square.

    Parameter
    ---------
    n : int
        Values are either 0, 1, 2, 3, 4, 5.

    Returns
    -------
    node : numpy.ndarray
        Array consisting of the node coordinates.
    cell : numpy.ndarray
        Array consisting of node indices describing the triangles.
    """
    correct_input = (n in [0, 1, 2, 3, 4, 5])
    if not correct_input:
        raise ValueError("Input must be in the list 0, 1, 2, 3, 4, 5.")
    if n == 0:
        node = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        cell = [[0, 1, 3], [1, 2, 3]]
    if n == 1:
        node = [[0., 0.], [1., 0.], [0.5, 0.5], [0., 1.], [1., 1.]]
        cell = [[0, 1, 2], [0, 2, 3], [2, 4, 3], [1, 4, 2]]
    if n == 2:
        node = [[0., 0.], [0.5, 0.], [1., 0.], [0, 0.5], [0.5, 0.5],
                [1., 0.5], [0., 1.], [0.5, 1.], [1., 1.]]
        cell = [[0, 1, 4], [1, 2, 4], [2, 4, 5], [0, 3, 4],
                [3, 4, 6], [4, 6, 7], [4, 7, 8], [4, 5, 8]]
    if n == 3:
        node = [[0., 0.], [1., 0.], [0.25, 0.5], [0., 1.], [1., 1.]]
        cell = [[0, 1, 2], [0, 2, 3], [2, 4, 3], [1, 4, 2]]
    if n == 4:
        node = [[0.,  0. ], [0.5, 0. ], [1.,  0. ], [0.,  0.5],
                [0.6, 0.4], [1.,  0.5], [0.,  1. ], [0.5, 1. ],
                [1.,  1. ]]
        cell = [[1, 3, 0], [3, 1, 4], [2, 5, 1], [4, 1, 5],
                [4, 6, 3], [6, 4, 7], [5, 8, 4], [7, 4, 8]]
    if n == 5:
        node = [[0., 0.], [1., 0.], [0.2, 0.4], [0.6, 0.4],
                [0., 1.], [1., 1.]]
        cell = [[0, 3, 2], [0, 1, 3], [0, 2, 4], [3, 5, 2],
                [1, 5, 3], [5, 4, 2]]
    return np.array(node).astype(float), np.array(cell).astype(int)


def unique_rows(array):
    """
    Binding to numpy.unique instance with parameters 'return_index' and
    'return_inverse' set to True and 'axis' set to 0.  More precisely:

        unique_rows(array) = np.unique(array, axis=0, return_index=True,
            return_inverse=True)

    Parameter
    ---------
    array : array_like
        Input array.

    Returns
    -------
    unique : numpy.ndarray
        The sorted unique rows of the array.
    unique_indices : numpy.ndarray
        The indices of the first occurences of the unique rows in the
        original array.
    unique_inverse: numpy.ndarray
        The indices to reconstruct original array for the unique array.
    """
    return np.unique(array, axis=0, return_index=True, return_inverse=True)


def help_attribute_to_instance_linker():
    """
    Prints documentation of instance 'attribute_to_instance_linker'.
    """
    print(attribute_to_instance_linker.__doc__)


def attribute_to_instance_linker(MeshTriObj):
    """
    Dictionary of attribute names to class setter instances for MeshTri.

    Parameter
    ---------
    MeshTriObj : MeshTri class

    Returns
    -------
    linker : dict
        Dictionary with keys and values corresponding to the attribute
        names and class instances, resectively.

    ================        =============================
          Keys                       Description
    ================        =============================
    size                    Mesh size
    edge                    Edge connectivities
    edge_coo                Edge coordinates
    edge_vec                Edge vectors
    edge_size               Edge lengths
    edge_center             Edge midpoints
    cell_center             Cell barycenters
    node_to_edge            Node to edge mapping
    cell_to_edge            Cell to edge mapping
    edge_to_cell            Edge to cell mapping
    node_to_cell            Node to cell mapping
    bdy_node_indices        Boundary node indices
    bdy_edge_indices        Boundary edge indices
    int_node_indices        Interior node indices
    int_edge_indices        Interior node indices
    """
    linker = dict()
    linker['size'] = MeshTriObj.set_size
    linker['edge'] = MeshTriObj.set_edge
    linker['edge_coo'] = MeshTriObj.set_edge_coo
    linker['edge_vec'] = MeshTriObj.set_edge_vec
    linker['edge_size'] = MeshTriObj.set_edge_size
    linker['edge_center'] = MeshTriObj.set_edge_center
    linker['cell_center'] = MeshTriObj.set_cell_center
    linker['node_to_edge'] = MeshTriObj.set_node_to_edge
    linker['cell_to_edge'] = MeshTriObj.set_cell_to_edge
    linker['edge_to_cell'] = MeshTriObj.set_edge_to_cell
    linker['node_to_cell'] = MeshTriObj.set_node_to_cell
    linker['bdy_edge_indices'] = MeshTriObj.set_bdy_edge_indices
    linker['bdy_node_indices'] = MeshTriObj.set_bdy_node_indices
    linker['int_edge_indices'] = MeshTriObj.set_int_edge_indices
    linker['int_node_indices'] = MeshTriObj.set_int_node_indices
    return linker


def cell_center(node, cell):
    """
    Determine triangle barycenters.

    Parameters
    ----------
    node : numpy.ndarray
        Array consisting of the node coordinates.
    cell : numpy.ndarray
        Array consisting of node indices describing the triangles.

    Returns
    -------
    cell_center : numpy.array
        Array consisting of the coordinates of the barycenters of the
        triangles, with shape (Nt, 2) where Nt = cell.shape[0].
    """
    cell_center = (node[cell[:, 0], :] + node[cell[:, 1], :] + node[cell[:, 2], :]) / 3
    return cell_center.astype(float)


def edge(cell, return_cell_to_edge=False):
    """
    Determine edge connectivities.

    Parameters
    ----------
    cell : numpy.ndarray
        Array consisting of node indices describing the triangles.
    return_cell_to_edge : bool, optional
        If True, also returns the indices of the edges containg
        each triangles.

    Returns
    -------
    edge : numpy.ndarray
        Array consisting of node indices describing the edges of the mesh.
        Shape is equal to (Ne, 2) where Ne is the number of edges.
    num_edge : int
        Total number of edges, equal to edge.shape[0].
    cell_to_edge : numpy.ndarray
        Array for which if cell_to_edge[i, :] = [j, k, l] then j, k and l
        are the indices of the edges in the ith triangle. Shape is equal
        to (Nt, 3) where Nt = cell.shape[0].
    """
    edge_temp = np.hstack([cell[:, [1, 2]], cell[:, [2, 0]],
        cell[:, [0, 1]]]).reshape(3*cell.shape[0], 2)
    edge_temp = np.sort(edge_temp)
    edge, index, inv_index = unique_rows(edge_temp)
    if return_cell_to_edge:
        return edge, edge.shape[0], inv_index.reshape((cell.shape[0], 3))
    else:
        return edge, edge.shape[0]


def edge_coo(node, edge):
    """
    Determine edge coordinates.

    Parameters
    ----------
    node : numpy.ndarray
        Array consisting of the node coordinates.
    edge : numpy.ndarray
        Array consisting of node indices describing the edges of the mesh.

    Returns
    -------
    edge_coo : numpy.ndarray
        Array consisting of the node coordinates of the edges, with
        shape equal to (Ne, 2), where Ne = edge.shape[0].

    See Also
    --------
    edge : Determine edge connectivities.
    """
    edge_coo = np.array([node[edge[:, 0], :], node[edge[:, 1], :]])
    return edge_coo.astype(float)


def edge_vec(edge_coo):
    """
    Determine edge vectors.

    Parameter
    ---------
    edge_coo : numpy.ndarray
        Array consisting of the node coordinates of the edges.

    Returns
    -------
    edge_vec : numpy.ndarray
        Array consisting of the components of the vectors associated
        with the edges. Shape is equal to (2, Ne, 2) where
        Ne = edge_coo.shape[0].

    See Also
    --------
    edge_coo : Determine edge coordinates.
    """
    edge_vec = edge_coo[0, :, :] - edge_coo[1, :, :]
    return edge_vec.astype(float)


def edge_size(edge_vec):
    """
    Determine edge lengths.

    Parameter
    ---------
    edge_vec : numpy.ndarray
        Array consisting of the components of the vectors associated
        with the edges.

    Returns
    -------
    edge_size : numpy.ndarray
        Array consisting of the lengths of the edges in the mesh, with shape
        equal to (Ne, 2), where Ne = edge.shape[0].

    See Also
    --------
    edge_vec : Determine edge vectors.
    """
    edge_size = np.linalg.norm(edge_vec, axis=1)
    return edge_size


def edge_center(node, edge):
    """
    Determine edge midpoints.

    Parameters
    ----------
    node : numpy.ndarray
        Array consisting of the node coordinates.
    edge : numpy.ndarray
        Array consisting of node indices describing the edges of the mesh.

    Returns
    -------
    edge_center : numpy.ndarray
        Array consisting of edge midpoint coordinates, with shape equal to
        (Ne, 2), where Ne = edge.shape[0].
    """
    edge_center = 0.5 * (node[edge[:, 0], :] + node[edge[:, 1], :])
    return edge_center.astype(float)


def edge_to_cell(num_cell, num_edge, cell_to_edge):
    """ Edge to cell mapping.

    Parameters
    ----------
    num_cell : int
        Number of cells.
    num_edge: int
        Number of edges.
    cell_to_edge : numpy.ndarray
        Array for which if cell_to_edge[i, :] = [j, k, l] then j, k and l
        are the indices of the edges in the ith triangle.

    Returns
    -------
    edge_to_cell : numpy.ndarray
        Array consisting of edges to cells of the mesh, with shape equal to
        (Ne, 2) where Ne = num_edge.

    See Also
    --------
    edge : Determine edge connectivities.
    """
    edge_temp = np.hstack([cell_to_edge[:, 0], cell_to_edge[:, 1], cell_to_edge[:, 2]])
    cell_temp = np.tile(np.arange(num_cell), (1, 3))[0]
    edge_frst, index_edge_frst = np.unique(edge_temp, return_index=True)
    edge_last, index_edge_last = np.unique(edge_temp[::-1], return_index=True)
    index_edge_last = edge_temp.shape[0] - index_edge_last - 1
    edge_to_cell = np.zeros((num_edge, 2), dtype=np.int64)
    edge_to_cell[edge_frst, 0] = cell_temp[index_edge_frst]
    edge_to_cell[edge_last, 1] = cell_temp[index_edge_last]
    edge_to_cell[np.nonzero(edge_to_cell[:, 0] == edge_to_cell[:, 1])[0], 1] = -1
    return edge_to_cell


def bdy_edge_indices(edge_to_cell):
    """
    Determine indices of boundary edges.

    Parameter
    ---------
    edge_to_cell : numpy.ndarray
        Array consisting of edges to cells of the mesh.

    Returns
    -------
    bdy_edge_indices : numpy.ndarray
        Array of indices corresponding to edges on the boundary.

    See Also
    --------
    edge_to_cell : Edge to cell mapping.
    """
    bdy_edge_indices = np.nonzero(edge_to_cell[:, 1] == -1)[0]
    return bdy_edge_indices


def bdy_node_indices(edge, bdy_edge_indices):
    """
    Determine indices of boundary nodes.

    Parameters
    ----------
    edge : numpy.ndarray
        Array consisting of node indices describing the edges of the mesh.
    bdy_edge_indices : numpy.ndarray
        Array of indices corresponding to edges on the boundary.

    Returns
    -------
    bdy_node_indices : numpy.ndarray
        Array of indices corresponding to nodes on the boundary.

    See Also
    --------
    edge : Determine edge connectivities.
    bdy_edge_indices : Determine indices of boundary edges.
    """
    bdy_node_indices = np.unique(edge[bdy_edge_indices])
    return bdy_node_indices


def int_node_indices(num_node, bdy_node_indices):
    """
    Deterime indices of interior nodes.

    Parameters
    ----------
    num_node : int
        Number of nodes.
    bdy_node_indices : numpy.ndarray
        Array of indices corresponding to nodes on the boundary.

    Returns
    -------
    int_node_indices : numpy.ndarray
        Array of indices corresponding to nodes in the interior.

    See Also
    --------
    bdy_node_indices : Determine indices of boundary nodes.
    """
    int_node_indices = set(range(num_node)).difference(set(bdy_node_indices))
    int_node_indices = np.asarray(list(int_node_indices))
    return int_node_indices


def int_edge_indices(num_edge, bdy_edge_indices):
    """
    Deterime indices of interior edges.

    Parameters
    ----------
    num_edge : int
        Number of edges.
    bdy_edge_indices : numpy.ndarray
        Array of indices corresponding to edges on the boundary.

    Returns
    -------
    int_edge_indices : numpy.ndarray
        Array of indices corresponding to edges in the interior.

    See Also
    --------
    bdy_edge_indices : Determine indices of boundary edges.
    """
    int_edge_indices = set(range(num_edge)).difference(set(bdy_edge_indices))
    int_edge_indices = np.asarray(list(int_edge_indices))
    return int_edge_indices


def node_to_cell(num_node, cell):
    """
    Determine node to cell mapping.

    Parameters
    ----------
    num_node : int
        Number of nodes.
    cell : numpy.ndarray
        Array consisting of node indices describing the triangles.

    Returns
    -------
    node_to_cell : scipy.csr_matrix
        Sparse matrix such that the data corresponding to ith row and
        column are the indices of triangles having node with index i.
    """
    data = np.hstack([cell[:, [1, 2]], cell[:, [2, 0]],
        cell[:, [0, 1]]]).reshape(3*cell.shape[0], 2)
    ent = np.kron(range(cell.shape[0]), [1]*3) + 1
    node_to_cell = sp.coo_matrix((ent, (data[:, 0], data[:, 1])),
        shape=(num_node, num_node), dtype=np.int).tocsr()
    return node_to_cell


def node_to_edge(num_node, node_to_cell):
    """
    Determine node to edge mapping.

    Parameters
    ----------
    num_node : int
        Number of nodes.
    node_to_cell :
        Sparse matrix such that the data corresponding to ith row and
        column are the indices of triangles having node with index i.

    Returns
    -------
    node_to_edge : scipy.csr_matrix
        Sparse matrix such that the node_to_edge[k, l] = j if the (j-1)st
        edge belongs to the kth and lth nodes.

    See Also
    --------
    node_to_cell : Determine node to cell mapping.
    """
    nz = sp.find(sp.tril(node_to_cell + node_to_cell.T))
    num_edge = len(nz[1])
    node_to_edge = sp.coo_matrix((range(1, num_edge+1), (nz[1], nz[0])),
        shape=(num_node, num_node), dtype=np.int).tocsr()
    node_to_edge = node_to_edge + node_to_edge.T
    return node_to_edge


def signed_area(node, cell):
    """
    Determine the array of signed cell areas.

    Parameters
    ----------
    node : numpy.ndarray
        Array consisting of the node coordinates.
    cell : numpy.ndarray
        Array consisting of node indices describing the triangles.

    Returns
    -------
    signed_area : numpy.ndarray
        Array consisting of signed areas of the triangles.
    """
    vec1 = node[cell[:, 1], :] - node[cell[:, 0], :]
    vec2 = node[cell[:, 2], :] - node[cell[:, 0], :]
    return 0.5 * (vec1[:, 0] * vec2[:,1] - vec1[:, 1] * vec2[:,0])


def quality(edge_size, cell_to_edge):
    """
    Deterime the array of triangle qualities.

    Parameters
    ----------
    edge_size : numpy.ndarray
        Array consisting of the lengths of the edges in the mesh.
    cell_to_edge : numpy.ndarray
        Edge to cell mapping.

    Returns
    -------
    quality : numpy.ndarray
        Array consisting of ratios between radii of circumcircle to
        incircle inside the triangle.

    See Also
    --------
    edge_size : Determine edge lengths.
    edge : Determine edge connectivities.
    """
    lens = edge_size[cell_to_edge]
    a = lens[:, 1] + lens[:, 2] - lens[:, 0]
    b = lens[:, 2] + lens[:, 0] - lens[:, 1]
    c = lens[:, 0] + lens[:, 1] - lens[:, 2]
    rad_in = a * b *c
    rad_out = lens[:, 0] * lens[:, 1] * lens[:, 2]
    mesh_quality = rad_in / rad_out
    return mesh_quality
