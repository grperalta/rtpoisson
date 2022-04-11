"""
==============================================================================
            MFEM AND HFEM FOR OPTIMAL CONTROL OF POISSON EQUATION
==============================================================================

This Python module approximates the optimal control of the Poisson equation
written in displacement-pressure (or temperature-heat flux) formulation:

    min (1/2){alpha |u - ud|^2 + beta |p - pd|^2 + gamma |q|^2}
    subject to          p + nabla u = 0,        in Omega,
                              div p = - q,      in Omega,
                                  u = 0,        on Gamma,
    over all controls q in L^2(Omega).

Here, u and p are the displacement and pressure (or temperature of heat flux,
respectively). Mixed and Hybrid finite element methods using the lowest order
Raviart-Thomas finite elements are utilized. The computational domain is the
unit square (0, 1)^2.

For more details, refer to the manuscript:
    G. Peralta, Error Estimates for Mixed and Hybrid FEM for Elliptic
    Optimal Control Problems with Penalizations, preprint.


Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph

"""

from __future__ import division
from numpy import linalg as la
from scipy import sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import tri
import numpy as np
import matplotlib
import warnings
import datetime
import platform
import time
import sys


__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2018, Gilbert Peralta"
__version__ = "1.0"
__maintainer__ = "The author"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "11 February 2021"


def mesh_repr(obj):
    """
    Returns the string representation of <Mesh> class.
    """

    np.set_printoptions(threshold=np.inf)
    string = ""

    for key, val in obj.__dict__.items():
        txt_attr = "Attribute: {}\n"
        txt_dscr = "Description: {}"
        txt_remk = "Remark: {}"
        if type(val) in [sp.csr.csr_matrix, sp.csc.csc_matrix,
                         sp.bsr.bsr_matrix]:
            val = val.tocoo()
            string +=  txt_attr.format(str(key))
            if key == 'node_to_edge':
                string += txt_dscr.format('Node to edge data structure.')
                string += '\n' + txt_remk.format('Indeing starts at 1.') + '\n'
            for row, col, value in zip(val.row, val.col, val.data):
                string += "({}, {}) {}".format(row, col, value) + '\n'
            string += '\n'
        else:
            string += txt_attr.format(str(key))
            if key == 'node':
                string += txt_dscr.format('Coordinates of the nodes.')
            elif key == 'tri':
                string += txt_dscr.format('Node connectivity'
                    + ' defining the triangles.')
                string += '\n' + txt_remk.format('Indexing starts at 1.')
            elif key == 'num_elems':
                string += txt_dscr.format('Number of elements.')
            elif key == 'num_edges':
                string += txt_dscr.format('Number of edges.')
            elif key == 'edge':
                string += txt_dscr.format('Node connectivity'
                    + ' defining the edges.')
            elif key == 'elem_center':
                string += txt_dscr.format('Coordinates of the centers'
                    + ' of the triangles.')
            elif key == 'edge_to_elem':
                string += txt_dscr.format('Edge to element data structure.')
            elif key == 'elem_to_edge':
                string += txt_dscr.format('Element to edge data structure.')
                string += '\n' + txt_remk.format('Indexing starts at 1.')
            elif key == 'elem_to_edge_sgn':
                string += txt_dscr.format('Global orientation of the edges'
                    + ' in the triangulation.')
            elif key == 'all_edge':
                string += txt_dscr.format('Local to global index map'
                    + ' for the dof associated to the edges.')
                string += '\n' + txt_remk.format('Indexing starts at 1.')
            elif key == 'int_edge':
                string += txt_dscr.format('Index of interior nodes.')
                string += '\n' + txt_remk.format('Indexing starts at 1.')
            elif key == 'num_nodes':
                string += txt_dscr.format('Number of nodes.')
            string += '\n' + str(val) + '\n\n'

    return string


class Mesh:
    """
    The mesh class for the triangulation of the domain in the finite
    element method.
    """

    def __init__(self, node, tri):
        """
        Class initialization/construction.

        ------------------
        Keyword arguments:
            - node              array of coordinates of the nodes in the mesh
            - tri               array of geometric connectivity of the
                                elements/triangles with respect to the
                                global indices in the array <node>

        -----------
        Attributes:
            - node              array of nodes (see key argument node)
            - tri               array of triangles (see key argument tri)
            - edge              array of edges with respect to the ordering
                                of the indices in <node>
            - num_nodes         number of nodes
            - num_elems         number of triangles
            - num_edges         number of edges
            - all_edge          array of all edges counting multiplicity for
                                each elements
            - int_edge          array of interior edges with respect to the
                                ordering in <edge>
            - elem_center       array of barycentes of the elements in the
                                mesh
            - node_to_edge      The node to edge data structure. The matrix
                                with entries such that <node_to_edge(k, l)=j>
                                if the jth edge belongs to the kth and lth
                                nodes and <node_to_edge(k, l)=0> otherwise.
            - edge_to_elem      The edge to element data structure. The matrix
                                such that the jth row is [k, l, m, n] where
                                k is the initial node, l is the terminal node,
                                and m and n are the indices of the elements
                                sharing a common edge, where n = 0 if there
                                is only one triangle containing the edge.
            - elem_to_edge      The element to edge data structure. The matrix
                                such that the jth row is [k, l, m] where k,
                                l, and m are the indices of the edge of the
                                jth element.
            - elem_to_edge_sgn  The sign of the edges with respect to a global
                                fixed orientation.
        """

        self.node = node
        self.tri = tri
        self.num_elems = self.tri.shape[0]
        self.num_nodes = self.node.shape[0]

        # get mesh data structure
        meshinfo = get_mesh_info(node, tri)

        self.edge = meshinfo['edge']
        self.num_edges = meshinfo['num_edges']
        self.elem_center = meshinfo['elem_center']
        self.node_to_edge = meshinfo['node_to_edge']
        self.edge_to_elem = meshinfo['edge_to_elem']
        self.elem_to_edge = meshinfo['elem_to_edge']
        self.elem_to_edge_sgn = meshinfo['elem_to_edge_sgn']
        self.all_edge = self.elem_to_edge.reshape(3*self.num_elems,)
        self.int_edge = sp.find(self.edge_to_elem[:, 3] != 0)[1]

    def __repr__(self):
        """
        Class string representation.
        """

        txt = '='*78 + '\n'
        txt += '\t\t\tMESH DATA STRUCTURE\n' + '='*78 + '\n\n'
        txt += mesh_repr(self)  + '='*78
        return txt

    def size(self):
        """
        Returns the smallest interior angles (in degrees) of the triangles.
        """

        h = 0.0
        for elem in range(self.num_elems):
            edge1 = (self.node[self.tri[elem, 1]-1, :]
                - self.node[self.tri[elem, 0]-1, :])
            edge2 = (self.node[self.tri[elem, 2]-1, :]
                - self.node[self.tri[elem, 1]-1, :])
            edge3 = (self.node[self.tri[elem, 0]-1, :]
                - self.node[self.tri[elem, 2]-1, :])
            h = max(h, la.norm(edge1), la.norm(edge2), la.norm(edge3))

        return h

    def plot(self, **kwargs):
        """
        Plots the mesh.
        """

        import matplotlib.pyplot as plt

        plt.figure()
        plt.triplot(self.node[:, 0], self.node[:, 1],
                    self.tri - 1, 'b-', lw=1.0, **kwargs)
        plt.show()


def square_uni_trimesh(n):
    """
    Generates a uniform triangular mesh of the unit square.

    -----------------
    Keyword argument:
        - n         number of nodes on one side of the unit square

    -------
    Return:
        A <Mesh> class of the uniform triangulation.
    """

    # number of elements
    num_elems = 2 * (n - 1) ** 2

    # pre-allocation of node list
    node = np.zeros((n**2, 2)).astype(float)

    # generation of node list
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            # node index
            index = (i - 1)*n + j - 1
            # x-coordinate of a node
            node[index, 0] = float((j-1)) / (n-1)
            # y-coordinate of a node
            node[index, 1] = float((i-1)) / (n-1)

    # pre-allocation of triangular elements
    tri = np.zeros((num_elems, 3)).astype(int)
    ctr = 0;

    # generation of triangular elements
    for i in range(1, n):
        for j in range(1, n):
            # lower right node of the square determined by two intersecting
            # triangles
            lr_node = (i-1) * n + j + 1
            # lower left triangle
            tri[ctr, :] = [lr_node, lr_node + n, lr_node - 1]
            # upper right triangle
            tri[ctr+1, :] = [lr_node + n - 1, lr_node - 1, lr_node + n]
            # increment counter
            ctr = ctr + 2

    return Mesh(node, tri)


def get_mesh_info(node, tri):
    """
    Returns the mesh data structure, to be called by the <Mesh> class.

    ------------------
    Keyword arguments:
        - node      array of coordinates of the nodes
        - tri       array of triangles

    ----------
    Reference:
        C. Bahriawati and C. Carstensen, Three Matlab implementation of the
        lowest-order Raviart-Thomas MFEM with a posteriori error control,
        Comp. Meth. App. Math. 5, pp. 333-361, 2005.
    """

    num_nodes = node.shape[0]
    num_elems = tri.shape[0]

    # pre-allocation of and column indices, and entries for node_to_elem
    row = np.zeros((3*num_elems,)).astype(int)
    col = np.zeros((3*num_elems,)).astype(int)
    ent = np.zeros((3*num_elems,)).astype(int)

    # generation row and column indices, and entries for node to element
    # data structure
    for i in range(num_elems):
        row[3*i : 3*i+3] = tri[i, :]-1
        col[3*i : 3*i+3] = tri[i, [1, 2, 0]]-1
        ent[3*i : 3*i+3] = [i+1, i+1, i+1]

    # node to element data stucture
    node_to_elem = sp.coo_matrix((ent, (row, col)),
        shape=(num_nodes, num_nodes)).tocsr()

    # nonzero entries in the lower triangular part of node_to_elem
    nz = sp.find(sp.tril(node_to_elem + node_to_elem.T))

    # number of edges
    num_edges = len(nz[1])

    # node to edge data data structure
    node_to_edge = sp.coo_matrix((range(1, num_edges+1), (nz[1], nz[0])),
        shape=(num_nodes, num_nodes)).tocsr()
    node_to_edge = node_to_edge + node_to_edge.T

    # pre-allocation of edge to element data stucture
    edge_to_elem = np.zeros((num_edges, 4)).astype(int)

    # assembly of edge to element data structure
    for i in range(num_elems):
        for j in range(3):
            initial_node = tri[i, j]
            terminal_node = tri[i, (j%3 + 1) % 3]
            index = node_to_edge[initial_node-1, terminal_node-1]
            if edge_to_elem[index-1, 0] == 0:
                edge_to_elem[index-1, :] = \
                    [initial_node, terminal_node,
                     node_to_elem[initial_node-1, terminal_node-1],
                     node_to_elem[terminal_node-1, initial_node-1]]

    # edges of the mesh
    edge = edge_to_elem[:, 0 : 2]

    # element to edge data stucture
    elem_to_edge = np.zeros((num_elems, 3)).astype(int)
    elem_to_edge[:, 0] = node_to_edge[tri[:, 1]-1, tri[:, 2]-1]
    elem_to_edge[:, 1] = node_to_edge[tri[:, 2]-1, tri[:, 0]-1]
    elem_to_edge[:, 2] = node_to_edge[tri[:, 0]-1, tri[:, 1]-1]

    # signs for the element to edge
    elem_to_edge_sgn = np.ones((num_elems, 3)).astype(int)

    # pre-allocation centers of each elements
    elem_center = np.zeros((num_elems, 2)).astype(float)

    # generates the barycenters of the triangles
    for i in range(num_elems):
        find_index = sp.find(i + 1 == edge_to_elem[elem_to_edge[i, :]-1, 3])
        elem_to_edge_sgn[i, find_index[1]] = - 1
        elem_center[i, 0] = np.sum(node[tri[i, :]-1, 0]) / 3
        elem_center[i, 1] = np.sum(node[tri[i, :]-1, 1]) / 3

    return {'edge':edge, 'node_to_edge':node_to_edge,
            'num_edges':num_edges, 'edge_to_elem':edge_to_elem,
            'elem_to_edge':elem_to_edge, 'elem_center':elem_center,
            'elem_to_edge_sgn':elem_to_edge_sgn}


def gauss1D_quad(l, a, b):
    """
    One-dimensional Gauss integration.

    ------------------
    Keyword arguments:
        - l     number of quadrature nodes
        - a     left endpoint of the interval of integration
        - b     right endpoint of the interval of integration

    -------
    Return:
        A dictionary with the keys <'nodes'>, <'weights'>, <'order'>, <'dim'>
        corresponding to the quadrature nodes, quadrature weights, order
        and dimension of the numerical quadrature.
    """

    dim = 1
    m = float((a + b)) / 2
    delta = float(b - a)

    if l == 1:
        nodes = np.array([m])
        weights = np.array([delta])
        order = 1
    elif l == 2:
        const = delta * np.sqrt(3) / 6
        nodes = np.array([m-const, m+const])
        weights = delta * np.array([0.5, 0.5])
        order = 3
    elif l == 3:
        const = delta * np.sqrt(3./5) / 2
        nodes = np.array([m-const, m, m+const])
        weights = delta * np.array([5./18, 8./18, 5./18])
        order = 5
    else:
        nodes = None
        weights = None
        order = None
        dim = None
        print('Gauss quadrature only available up to order 5 only.')

    return {'nodes':nodes, 'weights':weights, 'order':order, 'dim':dim}


def tri_quad(num_quad):
    """
    Gauss integration on the unit triangle with vertices at
    (0,0), (0,1) and (1,0).

    -----------------
    Keyword argument:
        - num_quad      number of quadrature nodes

    -------
    Return:
        A dictionary with the keys <'nodes'>, <'weights'>, <'order'>, <'dim'>
        corresponding to the quadrature nodes, quadrature weights, order
        and dimension of the numerical quadrature.

    ------
    To do:
        * Include quadrature nodes higher than 6.
    """

    dim = 2

    # change the number of quadrature nodes to the next number
    if num_quad == 2:
        num_quad = 3
        print('Number of quadrature nodes changed from 2 to 3.')
    elif num_quad == 5:
        num_quad = 6
        print('Number of quadrature nodes changed from 5 to 6.')

    if num_quad == 1:
        nodes = np.matrix([1./3, 1./3])
        weights = np.matrix([1./2])
        order = 1
    elif num_quad == 3:
        nodes = np.array([[2./3, 1./6],
                          [1./6, 2./3],
                          [1./6, 1./6]])
        weights = (1./2) * np.array([1, 1, 1]) / 3
        order = 2
    elif num_quad == 4:
        nodes = np.array([[1./3, 1./3],
                          [1./5, 1./5],
                          [3./5, 1./5],
                          [1./5, 3./5]])
        weights = (1./2) * np.array([-27, 25, 25, 25]) / 48
        order = 3
    elif num_quad == 6:
        nodes = np.array([[0.816847572980459, 0.091576213509771],
                          [0.091576213509771, 0.816847572980459],
                          [0.091576213509771, 0.091576213509771],
                          [0.108103018168070, 0.445948490915965],
                          [0.445948490915965, 0.108103018168070],
                          [0.445948490915965, 0.445948490915965]])
        weights = np.array([0.109951743655322,
                            0.109951743655322,
                            0.109951743655322,
                            0.223381589678011,
                            0.223381589678011,
                            0.223381589678011]) * (1./2)
        order = 4
    else:
        nodes = None
        weights = None
        order = None
        dim = None
        print('Number of quadrature nodes available up to 6 only.')

    return {'nodes':nodes, 'weights':weights, 'order':order, 'dim':dim}


def RT0_basis(p):
    """
    Generates the function values and divergence of the lowest order
    Raviart-Thomas finite element at the array of points p.

    -----------------
    Keyword argument:
        - p     array of points in the two-dimesional space

    -------
    Return:
        A dictionary with keys <'val'> and <'div'> corresponding to the
        function and divergence values at p, with shapes (3, N, 1) and
        (3, N, 1) where N is the number of points in p.
    """

    x = p[:, 0]
    y = p[:, 1]
    val = np.zeros((3, p.shape[0], 2)).astype(float)
    div = np.zeros((3, p.shape[0], 1)).astype(float) + 2

    val[0, :, :] = np.array([x, y]).T
    val[1, :, :] = np.array([x-1, y]).T
    val[2, :, :] = np.array([x, y-1]).T

    return {'val':val, 'div':div}


def affine_transform(mesh):
    """
    Generates the transformations from the reference triangle with
    vertices at (0,0), (0,1) and (1,0) to each element of the mesh.

    -----------------
    Keyword argument:
        - mesh      the domain triangulation (a class <Mesh>)

    -------
    Return:
        A dictionary with keys <'mat'>, <'vec'>, <'det'> corresponding to
        A, b and det(A), where Tx = Ax + b is the linear transformation
        from the reference element to an element in the mesh.
    """

    num_elems = mesh.num_elems
    B_K = np.zeros((num_elems, 2, 2)).astype(float)

    # coordinates of the triangles with local indices 0, 1, 2, respectively
    A = mesh.node[mesh.tri[:, 0]-1, :]
    B = mesh.node[mesh.tri[:, 1]-1, :]
    C = mesh.node[mesh.tri[:, 2]-1, :]

    a = B - A
    b = C - A
    B_K[:, :, 0] = a
    B_K[:, :, 1] = b
    B_K_det = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]

    return {'mat':B_K, 'vec':A, 'det':B_K_det}


def RT0_assemble(mesh, transformations, num_quad=6):
    """
    Assembles the mass and stiffness matrices for the lowest order
    Raviart-Thomas finite element using the edge basis formulation.

    ------------------
    Keyword arguments:
        - mesh              the domain triangulation (a class <Mesh>)
        - transformations   dictionary of transformations between each
                            triangle in the mesh and the reference triangle
        - num_quad          (optional) number of quadrature nodes with
                            default value 6

    -------
    Return:
        Returns a tuple corresponding to the mass and stiffness matrices.

    To do
    -----
        * Faster assembly, e.g. vectorization.
    """

    # number of elements and edges
    num_edges = mesh.num_edges
    num_elems = mesh.num_elems

    # absolute value for the Jacobian of the matrices of transformations
    B_K_detA = np.abs(transformations['det'])

    # set-up quadrature class
    quad = tri_quad(num_quad)

    # get the number of integration points in the numerical quadrature
    num_int_pts	= quad['nodes'].shape[0]

    # compute the values and divergence values of the basis functions
    # at the quadrature nodes
    rt0 = RT0_basis(quad['nodes'])

    # pre-allocation of row and column indices, and entries of the mass
    # matrix
    row = np.zeros((9*num_elems,)).astype(int)
    col = np.zeros((9*num_elems,)).astype(int)
    ent = np.zeros((9*num_elems,)).astype(float)

    for i in range(num_elems):
        # indices of the edges of the ith element
        ind = mesh.elem_to_edge[i, :] - 1
        ctri = 9 * i
        for j in range(3):
            # sign of the jth edge of the ith element
            sgnj = mesh.elem_to_edge_sgn[i, j]
            ctrj = 3 * j
            for k in range(3):
                ctrk = ctri + ctrj + k
                row[ctrk] = ind[j]
                col[ctrk] = ind[k]
                # sign of the kth edge of the ith element
                sgnk = mesh.elem_to_edge_sgn[i, k]
                local_ent = 0
                for m in range(num_int_pts):
                    local_ent = local_ent + quad['weights'][m] \
                        * np.dot(np.dot(transformations['mat'][i, :, :],
                        rt0['val'][j, m, :]),
                        np.dot(transformations['mat'][i, :, :],
                        rt0['val'][k, m, :]))
                    ent[ctrk] = sgnj * sgnk * (1./B_K_detA[i]) * local_ent

    # assembly of the mass matrix
    A = sp.coo_matrix((ent, (row, col)),
        shape=(num_edges, num_edges)).tocsr()

    # pre-allocation of row and column indices, and entries of the stiffness
    # matrix
    row = np.zeros((3*num_elems,)).astype(int)
    col = np.zeros((3*num_elems,)).astype(int)
    ent = np.zeros((3*num_elems,)).astype(float)

    for i in range(num_elems):
        # indices of the edges of the ith element
        ind = mesh.elem_to_edge[i, :] - 1
        ctri = 3 * i
        for j in range(3):
            ctrj = ctri + j
            row[ctrj] = ind[j]
            col[ctrj] = i
            # sign of the jth edge of the ith element
            sgnj = mesh.elem_to_edge_sgn[i, j]
            local_ent = 0
            for m in range(num_int_pts):
                local_ent = local_ent \
                    + np.dot(quad['weights'][m], rt0['div'][j, m, :])
                ent[ctrj] = sgnj * local_ent

    # assembly of the stiffness matrix
    B = sp.coo_matrix((ent, (row, col)),
        shape=(num_edges, num_elems)).tocsr()

    return A, B


def RT0_hybrid_assemble(mesh, transformations):
    """
    Assembles the mass, stiffness and Langrane matrices for the lowest
    order Raviart-Thomas finite element using hybridixation.

    -----------------
    Keyword argument:
        - mesh              the domain triangulation (a class mesh)
        - transformations   dictionary of transformations between each
                            triangle in the mesh and the reference triangle

    -------
    Return:
        A tuple corresponding to the mass, stiffness and Lagrange matrices.

    ------
    To do:
        * Faster assembly, e.g. vectorization.
    """

    # number of edges and elements
    num_edges = mesh.num_edges
    num_elems = mesh.num_elems

    # absolute value for the Jacobian of the matrices of transformations
    B_K_detA = np.abs(transformations['det'])

    # pre-allocation of the row/column indices and entries of the mass
    # matrix
    ind = np.zeros((3*num_elems,)).astype(int)
    ent = np.zeros((3*num_elems,)).astype(int)

    # all nodes counting multiplicity in each triangles of the mesh
    pts = mesh.node[mesh.tri[0, :]-1, :]

    # sum of squares of the edge lengths for each triangle
    s = np.linalg.norm(pts[0, :] - pts[1, :])**2 \
        + np.linalg.norm(pts[1, :] - pts[2, :])**2 \
        + np.linalg.norm(pts[2, :] - pts[0, :])**2

    ind = np.array(range(3*num_elems))
    ent = 0.5 * (np.kron(B_K_detA, [1, 1, 1])
                 * np.array([1, 1, s/36] * num_elems))

    # assembly of mass matrix
    A = sp.coo_matrix((ent, (ind, ind)), shape=(3*num_elems, 3*num_elems)).tocsr()

    # row and column indices, and entries of the stiffness matrix
    row = 3 * np.array(range(num_elems)).astype(int) + 2
    col = np.array(range(num_elems)).astype(int)
    ent = B_K_detA

    # assembly of the stiffness matrix
    B = sp.coo_matrix((ent, (row, col)),
        shape=(3*num_elems, num_elems)).tocsr()

    # number of interior edges
    num_int_edges = len(mesh.int_edge)

    # pre-allocation of row, column indices, and entries of the
    # matrix associated with the lagrange multiplier
    row = np.zeros((6*num_int_edges,)).astype(int)
    col = np.zeros((6*num_int_edges,)).astype(int)
    ent = np.zeros((6*num_int_edges,)).astype(float)

    for i in range(num_int_edges):
        initial_node = mesh.node[mesh.edge[mesh.int_edge[i], 0]-1, :]
        terminal_node = mesh.node[mesh.edge[mesh.int_edge[i], 1]-1, :]
        edge = terminal_node - initial_node
        elem_plus = mesh.edge_to_elem[mesh.int_edge[i], 2] - 1
        elem_minus = mesh.edge_to_elem[mesh.int_edge[i], 3] - 1
        mp_plus = mesh.elem_center[elem_plus, :]
        mp_minus = mesh.elem_center[elem_minus, :]
        ent_plus = np.matrix([mp_plus[1] - initial_node[1],
            initial_node[0] - mp_plus[0]])*np.matrix(edge).T
        ent_minus = np.matrix([mp_minus[1] - initial_node[1],
            initial_node[0] - mp_minus[0]])*np.matrix(edge).T
        row[6*i : 6*i+3] = range(3*elem_plus, 3*elem_plus+3)
        row[6*i+3: 6*i+6] = range(3*elem_minus, 3*elem_minus+3)
        col[6*i : 6*i+6] = [i] * 6
        ent[6*i : 6*i+6] = [-edge[1], edge[0], -ent_plus[0, 0],
            edge[1], -edge[0], ent_minus[0, 0]]

    # assembly of the matrix associated with the Lagrange multiplier
    C = sp.coo_matrix((ent, (row, col)),
        shape=(3*num_elems, num_int_edges)).tocsr()

    return A, B, C


def mat_dg_P0P1(num_elems):
    """
    Mass matrix corresponding to the inner product of discontinous P0
    and discontinuous P1 Lagrange elements divided by the area of the
    triangles.

    -----------------
    Keyword argument:
        - num_elems         number of elements in the mesh
    """

    return sp.kron(sp.eye(num_elems), [1./3, 1./3, 1./3])


def mat_dg_P1(num_elems):
    """
    Mass matrix corresponding to the discontinuous P1 Lagrange element
    divided by the area of the triangles.

    -----------------
    Keyword argument:
        - num_elems         number of elements in the mesh
    """

    local_ent = np.array([[2.0, 1.0, 1.0],
                          [1.0, 2.0, 1.0],
                          [1.0, 1.0, 2.0]]) * (1./12)

    return sp.kron(sp.eye(num_elems), local_ent)


def mat_postproc(num_elems):
    """
    Matrix required in the post-processing the Lagrange multipliers.

    -----------------
    Keyword argument:
        - num_elems         number of elements in the mesh
    """

    local_ent = np.array([[-1.0, 1.0, 1.0],
                          [1.0, -1.0, 1.0],
                          [1.0, 1.0, -1.0]])

    return sp.kron(sp.eye(num_elems), local_ent)


def mat_extend(mesh):
    """
    Matrix required in extending the Lagrange multipliers.

    -----------------
    Keyword argument:
        - mesh         the domain triangulation (a class <Mesh>)
    """

    row = np.array(range(3*mesh.num_elems)).astype(int)
    col = mesh.elem_to_edge-1
    col = col.flatten()

    M = sp.coo_matrix((np.ones(3*mesh.num_elems), (row, col)),
        shape=(3*mesh.num_elems, mesh.num_edges)).tocsc()

    return M


def default_parameters():
    """
    Returns a dictionary for the default parameters of the optimal
    control problems.
    """

    return dict(pen=1e-10, pen1=1e-10, pen2=1e-10, alpha=1.0, beta=1.0,
                gamma=1e-6, cgmaxit=None, cgtol=1e-12, ocptol=1e-6, ocpmaxit=100,
                hybrid=False, alt_adj=False)


class Matrices:
    """
    The class for the matrices in the lowest order Raviart-Thomas FE.
    """

    def __init__(self, mesh, prm, transformations=None):
        """
        Class initialization/construction.

        -----------
        Attributes:
            - mss           mass matrix
            - stf           stiffness matrix
            - total         total matrix for the linear systems
            - lag           Lagrange matrix in the hybrid formulation
            - postproc      post-processing matrix in the hybrid formulation
            - extend        matrix used in extending Lagrange multipliers
            - extint        matrix used in extending the interior Lagrange
                            multipliers
            - P1            mass matrix for the discontinuous P1 elements
            - P0P1          mass matrix for the inner product between
                            P0 and P1 elements
        """

        if transformations == None:
            transformations = affine_transform(mesh)

        if not prm['hybrid']:
            self.mss, self.stf = RT0_assemble(mesh, transformations)
            self.total = self.mss + (self.stf * self.stf.T) / prm['pen']
            self.lag = None
        else:
            self.mss, self.stf, self.lag \
                = RT0_hybrid_assemble(mesh, transformations)
            self.total = self.mss + (self.stf * self.stf.T / prm['pen1']
                + self.lag * self.lag.T / prm['pen2'])
            self.postproc = mat_postproc(mesh.num_elems)
            self.extend = mat_extend(mesh)
            self.extint = self.extend[:, mesh.int_edge]
        self.P1 = mat_dg_P1(mesh.num_elems)
        self.P0P1 = mat_dg_P0P1(mesh.num_elems)


def sparse_matrix_density(sp_matrix):
    """
    Calculates the density of the sparse matrix sp_matrix, that is,
    the ratio between the number of nonzero entries and the size of the
    matrix.
    """

    nnz = len(sp.find(sp_matrix)[2])
    return nnz / (sp_matrix.shape[0] * sp_matrix.shape[1])


def plot_sparse_matrix(sp_matrix, fn=1, info=True, ms=1):
    """
    Plot the sparse matrix.

    ------------------
    Keyword arguments:
        - sp_matrix     the sparse matrix (type <scipy.sparse.coo_matrix>)
        - fn            figure number window (default 1)
        - info          boolean variable if to print the size and
                        density of the matrix (default <True>)
        - ms            markersize (default 1)
    """

    fig = plt.figure(fn)
    plt.spy(sp_matrix, markersize=ms)
    plt.xticks([])
    plt.yticks([])
    if info == True:
        density = sparse_matrix_density(sp_matrix)
        string = ("Size of Matrix : {}-by-{}\nDensity : {:.4f}")
        plt.xlabel(string.format(sp_matrix.shape[0],
            sp_matrix.shape[1], density))
    plt.show()


def flux_interp(mesh):
    """ Interpolation of flux in the lowest order Raviart-Thomas FE.

    Returns the coefficients of the edge basis functions for the RT0-FE.
    The coefficients are the integrals of the normal component of the
    flux along the edges. This line integral is approximated using
    one-dimensional Gaussian quadrature. The default number of nodes
    for the quadrature is 3.

    ------------------
    Keyword Arguments:
        - mesh              the domain triangulation

    --------
    Returns:
        The global interpolation of the flux provided by the function
        grad_p in functions.py.
    """

    # number of elements and edges
    num_elems = mesh.num_elems
    num_edges = mesh.num_edges

    # pre-allocation of the interpolated flux
    p = np.zeros((num_edges,)).astype(float)

    # set-up one dimensional gaussian quadrature
    quad = gauss1D_quad(3, 0, 1)

    for i in range(num_edges):
        # initial and terminal nodes of the edge
        initial_node = mesh.node[mesh.edge[i, 0]-1, :]
        terminal_node = mesh.node[mesh.edge[i, 1]-1, :]
        # vector corresponding to the edge
        edge = terminal_node - initial_node
        edgelength = np.linalg.norm(edge)
        # normal component of the edge
        normal = [edge[1], - edge[0]] / edgelength
        # coordinates of the gauss nodes on the edge
        x = quad['nodes'] * initial_node[0] \
            + (1 - quad['nodes']) * terminal_node[0]
        y = quad['nodes'] * initial_node[1] \
            + (1 - quad['nodes']) * terminal_node[1]
        # function values of the flux at the gauss nodes
        (px, py) = grad_p(x, y)
        # coefficient of the interpolated flux on the edge
        p[i] = (normal[0] * np.dot(quad['weights'], px)
            + normal[1] * np.dot(quad['weights'], py)) * edgelength

    return p


def flux_convert(mesh, transformations, p):
    """
    Conversion of flux in terms of the coefficients of local
    barycentric coordinates.

    ------------------
    Keyword Arguments:
        - mesh              the domain triangulation
        - transformations   the affine transformations from the reference
                            triangle to each triangle of the mesh
        - p                 coefficients of the flux as a linear combination
                            of the edge basis functions

    -------
    Return:
        Coefficients of the local barycentric coordinates of each triangles.
    """

    # number of elements
    num_elems = mesh.num_elems

    # pre-allocation of coefficient vector
    cnvrt_p = np.zeros((3*num_elems, 2)).astype(float)

    for i in range(num_elems):
        # ith element
        elem = mesh.tri[i, :] - 1
        # sign of the edges in the element
        sgn = mesh.elem_to_edge_sgn[i, :]
        # coefficients of p with respect to the edges of the element
        c = p[mesh.elem_to_edge[i, :]-1]
        pt1 = mesh.node[elem[0], :] / transformations['det'][i]
        pt2 = mesh.node[elem[1], :] / transformations['det'][i]
        pt3 = mesh.node[elem[2], :] / transformations['det'][i]
        coeff1 = sgn[1]*c[1]*(pt1 - pt2) + sgn[2]*c[2]*(pt1 - pt3);
        coeff2 = sgn[0]*c[0]*(pt2 - pt1) + sgn[2]*c[2]*(pt2 - pt3);
        coeff3 = sgn[1]*c[1]*(pt3 - pt2) + sgn[0]*c[0]*(pt3 - pt1);
        # x and y coordinates of the coefficient vector
        cnvrt_p[3*i : 3*i+3, 0] = [coeff1[0], coeff2[0], coeff3[0]]
        cnvrt_p[3*i : 3*i+3, 1] = [coeff1[1], coeff2[1], coeff3[1]]

    return cnvrt_p


def flux_interp_hybrid(mesh, p):
    """
    Interpolation for the hybridized lowest Raviart-Thomas finite elements.

    ------------------
    Keyword arguments:
        - mesh      the domain triangulation
        - p         coefficients corresponding to the global interpolation
                    of the flux using edge basis functions

    -------
    Return:
        Coefficients of the flux in each element corresponding to the
        local basis functions (1,0), (0,1) and (x - x_b, y - y_b) where
        (x_b, y_b) is the barycenter of the triangle.
    """

    # number of elements
    num_elems = mesh.num_elems

    # pre-allocation of the coefficient vector
    interp_p = np.zeros((3*num_elems,)).astype(float)

    for i in range(num_elems):
        # ith element
        elem = mesh.tri[i, :] - 1
        # coordinates of the current element
        x_coord = mesh.node[elem, 0]
        y_coord = mesh.node[elem, 1]
        if x_coord[1] == x_coord[0]:
            index = [3*i, 3*i+2]
            coord = [x_coord[0], x_coord[2]]
        else:
            index = [3*i, 3*i+1]
            coord = [x_coord[0], x_coord[1]]
        coeff1 = (p[index[0], 0]*coord[1] - p[index[1], 0]*coord[0]) \
            / (coord[1] - coord[0])
        coeff3 = (p[index[1], 0] - p[index[0], 0]) / (coord[1] - coord[0])
        coeff2 = p[3*i, 1] - coeff3*y_coord[0]
        # coefficients corresponding to the current element
        interp_p[3*i : 3*i+3] = [coeff1 + coeff3*mesh.elem_center[i, 0],
            coeff2 + coeff3*mesh.elem_center[i, 1], coeff3]

    return interp_p


def flux_convert_hybrid(mesh, p):
    """
    Conversion of flux in terms of the coefficients of local
    barycentric coordinates.

    ------------------
    Keyword arguments:
        - mesh      the domain triangulation
        - p         coefficients corresponding to the global interpolation
                    of the flux using edge basis functions

    ------
    Return:
        Coefficients of the local barycentric coordinates of each triangles.
    """

    # number of elements
    num_elems = mesh.num_elems

    # pre-allocation of the coefficient vector
    cnvrt_p = np.zeros((3*num_elems, 2)).astype(float)

    for i in range(num_elems):
        # current element
        elem = mesh.tri[i, :] - 1
        # coefficients of p at the current element
        c = p[3*i : 3*i+3]
        # nodes of the the element
        pt1 = mesh.node[elem[0], :]
        pt2 = mesh.node[elem[1], :]
        pt3 = mesh.node[elem[2], :]
        # x components of the coefficient vector at the element
        cnvrt_p[3*i : 3*i+3, 0] \
            = [c[0] + c[2]*(pt1[0] - mesh.elem_center[i, 0]),
            c[0] + c[2]*(pt2[0] - mesh.elem_center[i, 0]),
            c[0] + c[2]*(pt3[0] - mesh.elem_center[i, 0])]
        # y components of the coefficient vector at the element
        cnvrt_p[3*i : 3*i+3, 1] \
            = [c[1] + c[2]*(pt1[1] - mesh.elem_center[i, 1]),
            c[1] + c[2]*(pt2[1] - mesh.elem_center[i, 1]),
            c[1] + c[2]*(pt3[1] - mesh.elem_center[i, 1])]

    return cnvrt_p


def cg(A, b, prm, x=None):
    """
    Solve the linear system Ax = b, where A is a symmetric
    positive definite matrix using conjugate gradient method.

    ------------------
    Keyword arguments:
        - A     symmetric positive definite matrix
        - b     vector for the right hand side of the linear system
        - prm   parameters for the conjugate gradient method
        - x     initial point

    -------
    Return:
        A tuple corresponding to the approximate solution of the linear
        system Ax = b and the number of iterations.
    """

    if x == None:
        x = np.zeros((len(b),)).astype(float)
        r = b
    else:
        r = b - A * x

    if prm['cgmaxit'] == None:
        prm['cgmaxit'] = 3 * len(x)

    p = r
    rho = np.dot(r, r)
    rtol = (prm['cgtol'] * np.linalg.norm(b)) ** 2
    it = 0
    while rho > rtol and it < prm['cgmaxit']:
        it = it + 1
        if it > 1:
            beta = rho / rho_old
            p = r + beta * p
        q = A * p
        alpha = rho / np.dot(p, q)
        x = x + alpha*p
        r = r - alpha*q
        rho_old = rho
        rho = np.dot(r,r)

    if it == prm['cgmaxit'] and rho > rtol:
        print("CG WARNING: Maximum number of iterations reached"
              +"  without satisfying tolerance.")

    return dict(sol=x, nit=it)


def init_p(x, y):
    """
    Initial pressure in the wave equation.
    """

    PI2 = 2 * np.math.pi

    return np.sin(PI2 * x) * np.sin(PI2 * y)


def grad_p(x, y):
    """
    Gradient of the initial pressure.
    """

    PI2 = 2 * np.math.pi

    px = PI2 * np.cos(PI2 * x) * np.sin(PI2 * y)
    py = PI2 * np.sin(PI2 * x) * np.cos(PI2 * y)

    return px, py


def delta_p(x, y):
    """
    Negative Laplacian of the initial pressure.
    """

    PI2 = 2 * np.math.pi

    return - 2 * (PI2 ** 2) * init_p(x,y)


class Poisson_Vars:
    """
    The variables of the Poisson equation.
    """

    def __init__(self, dsp, prs, lag=None):
        """
        Class initialization/construction.

        -----------
        Attributes:
            - disp      displacement
            - prs       pressure
            - lag       Lagrange multipliers
        """

        self.dsp = dsp
        self.prs = prs
        self.lag = lag

    def __sub__(self, other):
        if self.lag is not None and other.lag is not None:
            self.lag = self.lag - other.lag
        else:
            pass
        return Poisson_Vars(self.dsp - other.dsp,
                            self.prs - other.prs, self.lag)

    def __mul__(self, c):
        if self.lag is not None:
            self.lag = c * self.lag
        else:
            pass
        return Poisson_Vars(c * self.dsp, c * self.prs, self.lag)


def dsp_norm_P0(dsp, area):
    """
    L2-norm with respect to piecewise constant basis functions.
    """

    return np.sqrt(area * np.dot(dsp, dsp))


def dsp_norm_P1(dsp, area, Mat):
    """
    L2-norm with respect to piecewise linear basis functions.
    """

    return np.sqrt(area * np.dot(dsp, Mat.P1 * dsp))


def prs_norm(prs, Mat):
    """
    L2-norm with respect to the lowest order Raviart-Thomas finite
    elements.
    """

    return np.sqrt(np.dot(prs, Mat.mss * prs))


def MFEM_State_Solver(Mat, control, area, prm):
    """ Solves the state equation. """

    b = - area * control
    p = cg(Mat.total, (1 / prm['pen']) * Mat.stf * b, prm)['sol']
    u = (1 / prm['pen']) * (Mat.stf.T * p - b)

    return Poisson_Vars(u, p)


def MFEM_Residual(state, desired):
    """
    Computes the difference between the state and the desired state.
    """

    return Poisson_Vars(np.kron(state.dsp, [1, 1, 1]) - desired.dsp,
                        state.prs - desired.prs)


def MFEM_Adjoint_Solver(Mat, residual, area, prm):
    """ Solves the adjoint equation. """

    bf = - prm['beta'] * Mat.mss * residual.prs
    bg = - prm['alpha'] * area * (Mat.P0P1 * residual.dsp)
    zp = cg(Mat.total, bf + (1 / prm['pen']) * Mat.stf * bg, prm)['sol']
    zu = (1 / prm['pen']) * (Mat.stf.T * zp - bg)

    return Poisson_Vars(zu, zp)


def MFEM_Adjoint_to_Control(adjoint):
    """ Maps the adjoint to the control. """

    return adjoint.dsp


def MFEM_Cost(Mat, residual, control, area, prm):
    """ Computes the cost. """

    J = 0.5 * area * prm['alpha'] * np.dot(residual.dsp, residual.dsp) \
        + 0.5 * prm['beta'] * np.dot(residual.prs, Mat.mss * residual.prs) \
        + 0.5 * area * prm['gamma'] * np.dot(control, control)

    return J


def MFEM_Cost_Derivative(control, adj_to_ctrl, prm):
    """ Computes the derivative of the cost functional. """

    return prm['gamma'] * control + adj_to_ctrl


def MFEM_Optimality_Residual(Mat, residual, area):
    """
    Calculates the residual norm of the optimality condition in the
    mixed formulation.
    """

    return np.sqrt(area * np.dot(residual, residual))


def HFEM_PostProcess(state, Mat, mesh):
    """ Post-process the lagrange multiplier to pressure. """

    lag_extend = np.zeros((mesh.num_edges,)).astype(float)
    lag_extend[mesh.int_edge] = state.lag

    return Mat.postproc * (Mat.extend * lag_extend)


def HFEM_State_Solver(Mat, control, area, prm):
    """ Solves the state equation in the hybrid formulation. """

    sol = MFEM_State_Solver(Mat, Mat.P0P1 * control, area, prm)
    lag = Mat.lag.T * sol.prs / prm['pen2']

    return Poisson_Vars(sol.dsp, sol.prs, lag)


def HFEM_Residual(state, desired, Mat, mesh):
    """
    Computes the difference between the state and the desired
    state in the hybrid formulation.
    """

    dsp_postproc = HFEM_PostProcess(state, Mat, mesh)

    return Poisson_Vars(dsp_postproc - desired.dsp, state.prs - desired.prs)


def HFEM_Adjoint_Solver(Mat, residual, area, prm):
    """ Solves the adjoint equation in the hybrid formulation. """

    bf = - prm['beta'] * Mat.mss * residual.prs
    bg = - prm['alpha'] * area * Mat.P0P1 * residual.dsp
    zp = cg(Mat.total, bf + (1 / prm['pen1']) * Mat.stf * bg, prm)['sol']
    zu = (1 / prm['pen1']) * (Mat.stf.T * zp - bg)
    zlag = Mat.lag.T * zp / prm['pen2']

    return Poisson_Vars(zu, zp, zlag)


def HFEM_Adjoint_Solver_Alternative(Mat, residual, area, prm):
    """ Solves the adjoint equation in the hybrid formulation. """

    bf = - prm['beta'] * Mat.mss * residual.prs
    bg = - prm['alpha'] * area * Mat.P1 * residual.dsp
    bg = Mat.extint.T * Mat.postproc.T * bg
    zp = cg(Mat.total, bf + (1 / prm['pen2']) * Mat.lag * bg, prm)['sol']
    zu = (1 / prm['pen1']) * Mat.stf.T * zp
    zlag = (Mat.lag.T * zp - bg) / prm['pen2']

    return Poisson_Vars(zu, zp, zlag)


def HFEM_Adjoint_to_Control(adjoint, Mat, mesh):
    """ Maps the adjoint to control in the hybrid formulation. """

    return HFEM_PostProcess(adjoint, Mat, mesh)


def HFEM_Cost(Mat, residual, control, area, prm):
    """ Computes the cost in the hybrid formulation. """

    J = 0.5 * area * prm['alpha'] \
        * np.dot(residual.dsp, Mat.P1 * residual.dsp) \
        + 0.5 * prm['beta'] * np.dot(residual.prs, Mat.mss * residual.prs) \
        + 0.5 * area * prm['gamma'] * np.dot(control, Mat.P1 * control)

    return J


def HFEM_Optimality_Residual(Mat, residual, area):
    """
    Calculates the residual norm of the optimality condition in the
    hybrid formulation.
    """

    return np.sqrt(area * np.dot(residual, Mat.P1 * residual))


def build_desired(mesh, transformations, prm):
    """
    Assembles the desired states for the optimal control problem.
    """

    x = mesh.node[mesh.tri-1, 0].reshape(3*mesh.num_elems,)
    y = mesh.node[mesh.tri-1, 1].reshape(3*mesh.num_elems,)
    ud = init_p(x, y)
    if not prm['hybrid']:
        pd = flux_interp(mesh)
    else:
        pd = flux_interp(mesh)
        pd = flux_convert(mesh, transformations, pd)
        pd = flux_interp_hybrid(mesh, pd)
    return Poisson_Vars(ud, pd)


class OCP:
    """
    The optimal control problem class.
    """

    def __init__(self, prm=None, desired=None):
        """
        Class initialization.
        """

        self.mesh = square_uni_trimesh(prm['n'])
        self.transformations = affine_transform(self.mesh)
        self.area = self.transformations['det'][0] / 2
        self.prm = default_parameters()
        if prm != None:
            self.prm.update(prm)
        self.Mat = Matrices(self.mesh, self.prm, self.transformations)
        self.desired = build_desired(
            self.mesh, self.transformations, self.prm)
        if not self.prm['hybrid']:
            self.init_control \
                = np.zeros((self.mesh.num_elems,)).astype(float)
        else:
            self.init_control \
                = np.zeros((3*self.mesh.num_elems,)).astype(float)
        self.rhs = None

    def state_solver(self, control):
        if self.rhs is not None:
            if not self.prm['hybrid']:
                control = control + self.Mat.P0P1 * self.rhs
            else:
                control = control + self.rhs
        if not self.prm['hybrid']:
            return MFEM_State_Solver(self.Mat, control, self.area, self.prm)
        else:
            return HFEM_State_Solver(self.Mat, control, self.area, self.prm)

    def residual(self, state):
        if not self.prm['hybrid']:
            return MFEM_Residual(state, self.desired)
        else:
            return HFEM_Residual(state, self.desired, self.Mat, self.mesh)

    def adjoint_solver(self, residual):
        if not self.prm['hybrid']:
            return MFEM_Adjoint_Solver(self.Mat, residual, self.area, self.prm)
        else:
            if not self.prm['alt_adj']:
                return HFEM_Adjoint_Solver(
                        self.Mat, residual, self.area, self.prm)
            else:
                return HFEM_Adjoint_Solver_Alternative(
                        self.Mat, residual, self.area, self.prm)

    def der_cost(self, control, adj_to_ctrl):
        return MFEM_Cost_Derivative(control, adj_to_ctrl, self.prm)

    def cost(self, residual, control):
        if not self.prm['hybrid']:
            return MFEM_Cost(self.Mat, residual, control, self.area, self.prm)
        else:
            return HFEM_Cost(self.Mat, residual, control, self.area, self.prm)

    def adjoint_to_control(self, adjoint):
        if not self.prm['hybrid']:
            return MFEM_Adjoint_to_Control(adjoint)
        else:
            return HFEM_Adjoint_to_Control(adjoint, self.Mat, self.mesh)

    def denom_init_step(self, state, control):
        if not self.prm['hybrid']:
            return MFEM_Cost(self.Mat, state, control, self.area, self.prm)
        else:
            state.dsp = np.kron(state.dsp, [1, 1, 1])
            return HFEM_Cost(self.Mat, state, control, self.area, self.prm)

    def optimality_residual(self, residual):
        if not self.prm['hybrid']:
            return MFEM_Optimality_Residual(self.Mat, residual, self.area)
        else:
            return HFEM_Optimality_Residual(self.Mat, residual, self.area)


def Barzilai_Borwein(ocp, SecondPoint=None, info=True, version=1):
    """
    Barzilai-Borwein version of the gradient method.

    The algorithm stops if the consecutive cost function values have
    relative error less than the pescribed tolerance or the maximum
    number of iterations is reached.

    ------------------
    Keyword arguments:
        - ocp           a class for the optimal control problem
        - info          Prints the iteration number, cost value and relative
                        error of consecutive cost values. (default True).
        - version       Either 1, 2, or 3. Method of getting the steplength.
                        Let dc and dj be the residues of the control and the
                        derivative of the cost functional, and s be the
                        steplength. The following are implemented depending
                        on the value of version:
                        If <version==1> then
                            s = (dc,dj) / (dj,dj).
                        If <version==2> then
                            s = (dc,dc) / (dc,dj).
                        If <version==3> then
                            s = (dc,dj) / (dj,dj) if the iteration number is
                            odd and s = (dc,dc) / (dc,dj) otherwise.
                        Here, (,) represents the inner product in Rn.
                        The default value of version is set to 1.
        - SecondPoint   The second point of the gradient method. If value is
                        <None> then the second point is given by
                        x = x - g/|g| where x is the initial point and g is
                        its gradient value. If value is <'LS'> then the
                        second point is calculated via inexact line search
                        with Armijo steplenght criterion.
    --------
    Returns:
        The list of state, control, adjoint and residual variables
        of the optimal control problem.

    ------
    Notes:
        The ocp class should have at least the following methods:
        <state_solver>
            A function that solves the state equation.
        <adjoint_solver>
            A function that solves the adjoint equation.
        <residual>
            A function that computes the difference between
            the state and the desired state.
        <cost>
            The cost functional.
        <der_cost>
            Derivative of the cost functional.
        <adjoint_to_control>
            A function that maps the adjoint to the control.
        <optimality_residual>
            A function that computes the measure on which the
            necessary condition is satisfied, that is, norm of
            gradient is small.
        <denom_init_step>
            A function that calculates the denominator in the
            steepest descent steplength.
    """

    if info:
        string = ("BARZILAI-BORWEIN GRADIENT METHOD:"
                  + "\t Tolerance = {:.1e}\t Version {}\n")
        print(string.format(ocp.prm['ocptol'], version))

    # main algorithm
    start_time = time.time()
    for i in range(ocp.prm['ocpmaxit']):
        if i == 0:
            if info:
                print("Iteration: 1")
            state = ocp.state_solver(ocp.init_control)
            residue = ocp.residual(state)
            cost_old = ocp.cost(residue, ocp.init_control)
            adjoint = ocp.adjoint_solver(residue)
            control_old = ocp.init_control
            control = ocp.init_control \
                - ocp.der_cost(ocp.init_control,
                ocp.adjoint_to_control(adjoint))
            if SecondPoint == 'LS':
                num = np.sum(control * control)
                steplength = num / (2 * ocp.denom_init_step(state, control))
                control = steplength * control
                state = ocp.state_solver(control)
                residue = ocp.residual(state)
                cost = ocp.cost(residue, control)
                alpha = 1
                iters = 0
                while cost > cost_old - (1e-4) * alpha * num:
                    alpha = alpha * 0.5
                    control = alpha * control
                    state = state * alpha
                    residue = ocp.residual(state)
                    cost = ocp.cost(residue, control)
                    iters = iters + 1
                if info:
                    print("Number of Backtracking Iterations: " + str(iters))
            elif SecondPoint == None:
                state = ocp.state_solver(control)
                residue = ocp.residual(state)
                cost = ocp.cost(residue, control)
                steplength = 1.0
            try:
                cost
            except UnboundLocalError:
                message = ("Undefined option: Either of the following:"
                           + " <None> or 'LS' is implemented.")
                warnings.warn(message, UserWarning)
                break
            if info:
                string = "\t Cost Value = {:.6e}"
                print(string.format(cost))
                print("\t Steplength = {:.6e}".format(steplength))
        else:
            if info:
                print("Iteration: {}".format(i+1))
            adjoint_old = ocp.adjoint_to_control(adjoint)
            adjoint = ocp.adjoint_solver(residue)
            control_residue = control - control_old
            adjoint_residue = ocp.adjoint_to_control(adjoint) - adjoint_old
            res_dercost = ocp.der_cost(control_residue, adjoint_residue)
            if version == 1:
                steplength = np.sum(control_residue * res_dercost) \
                    / np.sum(res_dercost * res_dercost)
            elif version == 2:
                steplength = np.sum(control_residue * control_residue) \
                    / np.sum(control_residue * res_dercost)
            elif version == 3:
                if (i % 2) == 1:
                    steplength = np.sum(control_residue * res_dercost) \
                        / np.sum(res_dercost * res_dercost)
                else:
                    steplength = np.sum(control_residue * control_residue) \
                        / np.sum(control_residue * res_dercost)
            control_old = control
            control = control \
                - steplength \
                * ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
            state = ocp.state_solver(control)
            cost_old = cost
            residue = ocp.residual(state)
            cost = ocp.cost(residue, control)
            rel_error = np.abs(cost - cost_old) / cost
            if info:
                string = ("\t Cost Value = {:.6e}"
                    + "\t Relative Error = {:.6e}")
                print(string.format(cost, rel_error))
                string = ("\t Steplength = {:.6e}"
                    + "\t Optimality Res = {:.6e}")
                res = ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
                opt_res = ocp.optimality_residual(res)
                print(string.format(steplength, opt_res))
            if rel_error < ocp.prm['ocptol']:
                if info:
                    print("Optimal solution found.")
                break
    if i == ocp.prm['ocpmaxit'] and rel_error > ocp.prm['ocptol']:
        print("BB Warning: Maximum number of iterations reached"
            + " without satisfying the tolerance.")
    end_time = time.time()
    if info:
        print("\t Elapsed time is " + "{:.8f}".format(end_time-start_time)
            + " seconds.\n")

    return {'state':state, 'adjoint':adjoint,
            'control':control, 'residue':residue}


def print_line():
    """
    Prints a double line.
    """

    print('='*80)


def print_start():
    """
    Prints machine platform and python version.
    """

    print('*'*78 + '\n')
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start of Run: " + start + '\n')

    string = ("PYTHON VERSION: {} \nPLATFORM: {} \nPROCESSOR: {}"
        + "\nVERSION: {} \nMAC VERSION: {}")
    print(string.format(sys.version, platform.platform(),
        platform.uname()[5], platform.version()[:60]
        + '\n' + platform.version()[60:], platform.mac_ver()) + '\n')


def print_end():
    """
    Prints end datetime of execution.
    """

    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("End of Run: " + end + '\n')
    print('*'*78 + '\n')


def print_details(mesh, Mat, prm):
    """
    Prints the details on the mesh, matrices and parameter.
    """

    print_line()
    if prm['hybrid']:
        print("\t\t Hybrid FEM for Optimal Control of Poisson Equation")
    else:
        print("\t\t Mixed FEM for Optimal Control of Poisson Equation")
    print_line()
    print("MESH DETAILS")
    print("\t Number of Nodes: {}".format(mesh.num_nodes))
    print("\t Number of Elements: {}".format(mesh.num_elems))
    print("\t Number of Edges: {}".format(mesh.num_edges))
    print("\t Spatial Meshsize: {:.4f}".format(mesh.size()))

    print_line()
    print("MATRIX DETAILS")
    print("\t Size of Mass Matrix: {}-by-{}"
        .format(Mat.mss.shape[0], Mat.mss.shape[1]))
    print("\t Size of Stiffness Matrix: {}-by-{}"
        .format(Mat.stf.shape[0], Mat.stf.shape[1]))
    if Mat.lag != None:
        print("\t Size of Lagrange Matrix: {}-by-{}"
            .format(Mat.lag.shape[0], Mat.lag.shape[1]))
    print("\t Total Matrix Density: {:.6f}"
        .format(sparse_matrix_density(Mat.total)))

    print_line()
    print("COEFFICIENTS IN THE COST FUNCTIONAL")
    print("\t alpha = {}".format(prm['alpha']))
    print("\t beta = {}".format(prm['beta']))
    print("\t gamma = {:.1e}".format(prm['gamma']))
    print_line()


def error_analysis(_type, alter_adj=False, fullinfo=False,
                   penalty_range=10, spatial_range=9):
    """
    Plots the error between the exact state, adjoint state and control
    with their computed numerical counterparts.

    ----------
    Arguments:
        - _type             either "spatial" or "penalty"
        - alter_adj         boolean variable for the alternative adjoint
                            formulation of the optimal control problem
        - fullinfo          boolean variable to print full details
                            of the algorithm
        - penalty_range     last power of the penalty parameter
        - spatial_range     last power of the spatial meshsize
    """

    if _type == "penalty":
        data = [10**k for k in range(2, penalty_range+1)]
    elif _type == "spatial":
        data = [2**k + 1 for k in range(2, spatial_range+1)]

    error = np.zeros((len(data), 6)).astype(float)
    order = np.zeros((len(data)-1, 5)).astype(float)

    print_line()
    if _type == "penalty":
        print("\t\t\tPENALIZATION ERRORS")
    elif _type == "spatial":
        print("\t\t\tDISCRETIZATION ERRORS")
    print_line()
    txt = "Control\t\tFlux\t\tTemp\t\tAdjoint Flux\tAdjoint Temp"
    row_data = "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}"
    if not fullinfo:
        print(txt)

    i = 0
    for num in data:
        if _type == "penalty":
            ocp = OCP(prm=dict(n=65, hybrid=True, gamma=1e-1,
                alt_adj=alter_adj, pen1=1./num, pen2=1./num))
        elif _type == "spatial":
            ocp = OCP(prm=dict(n=num, hybrid=True, gamma=1e-1,
                alt_adj=alter_adj))

        # coordinates of nodes
        x = ocp.mesh.node[ocp.mesh.tri-1, 0].reshape(3*ocp.mesh.num_elems,)
        y = ocp.mesh.node[ocp.mesh.tri-1, 1].reshape(3*ocp.mesh.num_elems,)

        # exact adjoint, state, and control
        adjoint_exact = build_desired(ocp.mesh, ocp.transformations, ocp.prm)
        state_exact = build_desired(ocp.mesh, ocp.transformations, ocp.prm)
        control_exact = - adjoint_exact.dsp / ocp.prm['gamma']

        # source term
        ocp.rhs = - delta_p(x, y) - control_exact

        # desired state
        ocp.desired = Poisson_Vars(state_exact.dsp + delta_p(x, y),
            state_exact.prs)

        # solve optimal control problem
        if fullinfo:
            print_details(ocp.mesh, ocp.Mat, ocp.prm)
        sol = Barzilai_Borwein(ocp, SecondPoint='LS', info=fullinfo, version=3)

        # post-process optimal control, state and adjoint state
        sol['control'] = - HFEM_PostProcess(sol['adjoint'], ocp.Mat,
            ocp.mesh) / ocp.prm['gamma']
        sol['state'].dsp = HFEM_PostProcess(sol['state'], ocp.Mat,
            ocp.mesh)
        sol['adjoint'].dsp = HFEM_PostProcess(sol['adjoint'], ocp.Mat,
            ocp.mesh)

        # error in control
        error[i, 0] \
            = dsp_norm_P1(sol['control'] - control_exact, ocp.area, ocp.Mat)
        # error in flux
        error[i, 1] \
            = prs_norm(sol['state'].prs - state_exact.prs, ocp.Mat)
        # error in temperature
        error[i, 2] \
            = dsp_norm_P1(sol['state'].dsp - state_exact.dsp, ocp.area,
            ocp.Mat)
        # error in adjoint flux
        error[i, 3] \
            = prs_norm(sol['adjoint'].prs - adjoint_exact.prs, ocp.Mat)
        # error in adjoint temperature
        error[i, 4] \
            = dsp_norm_P1(sol['adjoint'].dsp - adjoint_exact.dsp, ocp.area,
            ocp.Mat)
        if fullinfo:
            if _type == "penalty":
                print("Penalty Parameter = {:.2e}".format(ocp.prm['pen1']))
            print(txt)
        print(row_data.format(error[i, 0], error[i, 1], error[i, 2],
            error[i, 3], error[i, 4]))

        if _type == "penalty":
            error[i, 5] = ocp.prm['pen1']
        elif _type == "spatial":
            error[i, 5] = ocp.mesh.size()

        if i > 0:
            order[i-1, :] = np.log(error[i-1, :5] / error[i, :5]) \
                / np.log(error[i-1, 5] / error[i, 5])
        i += 1

    print_line()
    print("\t\t\tORDER OF CONVERGENCE")
    print_line()
    print(txt)
    for i in range(len(data)-1):
        print(row_data.format(order[i, 0], order[i, 1], order[i, 2],
            order[i, 3], order[i, 4]))
    print_line()

    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.unicode'] = True

    list1 = [0, 3, 4]
    list2 = [1, 2]
    markers = ['^', 'o', 's', 'o', 's']
    linestyles = ['-', '-', '-', '-', '-']
    colors = ['red', 'black', 'blue', 'black', 'blue']
    labels = [r'$\|\bar{q}_h - \bar{q}_{h\varepsilon}\|$',
        r'$\|\bar{\sigma}_h - \bar{\sigma}_{h\varepsilon}\|$',
        r'$\|\bar{u}_h - R_h^1\bar{\lambda}_{h\varepsilon}\|$',
        r'$\|\bar{\varphi}_h - \bar{\varphi}_{h\varepsilon}\|$',
        r'$\|\bar{w}_h - R_h^1\bar{\mu}_{h\varepsilon}\|$']

    if _type == 'penalty':
        rescale = 1e7
        order = 1
        order_label = r'$O(\varepsilon)$'
        filename = 'penalty_error.pdf'
    elif _type == 'spatial':
        rescale = 1./5
        order = 2
        order_label = r'$O(h^2)$'
        filename = 'spatial_error.pdf'

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    for index in list2:
        ax1.loglog(error[:, 5], error[:, index], color=colors[index],
            marker=markers[index], ls=linestyles[index],
            label=labels[index], ms=4, lw=1)
    ax1.loglog(error[:, 5], error[:, 5]**order*rescale, color='black',
        ls='-.', label=order_label, ms=4, lw=1)
    ax1.legend(loc='best')
    plt.grid(True)

    ax2 = fig.add_subplot(122)
    for index in list1:
        ax2.loglog(error[:, 5], error[:, index], color=colors[index],
            marker=markers[index], ls=linestyles[index],
            label=labels[index], ms=4, lw=1)
    ax2.loglog(error[:, 5], error[:, 5]**order*rescale, color='black',
        ls='-.', label=order_label, ms=4, lw=1)
    ax2.legend(loc='best')
    plt.grid(True)

    fig.savefig(filename, bbox_inches='tight', format='pdf', dpi=900)

    # save error data
    if _type == "penalty":
        np.save('penalty_error.npy', error)
    elif _type == "spatial":
        np.save('spatial_error.npy', error)

    #plt.show()


def mixed_vs_hybrid_main(num):
    """
    Main function for calculating the norm of difference between the numerical
    optimal controls, states and adjoint states in the mixed and hybrid
    methods. The argument <num> is the number of subdivisions in an axis.
    """

    error = np.zeros(5).astype(float)

    for hybrid_flag in [True, False]:
        ocp = OCP(prm=dict(n=num, hybrid=hybrid_flag, gamma=1e-1))

        # coordinates of nodes
        x = ocp.mesh.node[ocp.mesh.tri-1, 0].reshape(3*ocp.mesh.num_elems,)
        y = ocp.mesh.node[ocp.mesh.tri-1, 1].reshape(3*ocp.mesh.num_elems,)

        # exact adjoint, state, and control
        adjoint_exact = build_desired(ocp.mesh, ocp.transformations, ocp.prm)
        state_exact = build_desired(ocp.mesh, ocp.transformations, ocp.prm)
        control_exact = - adjoint_exact.dsp / ocp.prm['gamma']

        # source term
        ocp.rhs = - delta_p(x, y) - control_exact

        # desired state
        ocp.desired \
            = Poisson_Vars(state_exact.dsp + delta_p(x, y), state_exact.prs)

        # solve optimal control problem
        if hybrid_flag == False:
            sol_mixed = Barzilai_Borwein(ocp, info=False, SecondPoint='LS',
                version=3)
            sol_mixed['control'] = np.kron(sol_mixed['control'], [1, 1, 1])
            for var in ['state', 'adjoint']:
                sol_mixed[var].prs \
                    = flux_convert(ocp.mesh, ocp.transformations,
                    sol_mixed[var].prs)

        else:
            sol_hybrid = Barzilai_Borwein(ocp, info=False, SecondPoint='LS',
                version=3)
            for var in ['state', 'adjoint']:
                sol_hybrid[var].prs \
                    = flux_convert_hybrid(ocp.mesh, sol_hybrid[var].prs)

    # error in control
    error[0] \
        = dsp_norm_P1(sol_mixed['control'] - sol_hybrid['control'],
        ocp.area, ocp.Mat)
    # error in flux
    error[1] \
        = (dsp_norm_P1(sol_mixed['state'].prs[:, 0]
        - sol_hybrid['state'].prs[:, 0], ocp.area, ocp.Mat)**2
        + dsp_norm_P1(sol_mixed['state'].prs[:, 1]
        - sol_hybrid['state'].prs[:, 1], ocp.area, ocp.Mat)**2) ** 0.5
    # error in temperature
    error[2] \
        = dsp_norm_P0(sol_mixed['state'].dsp - sol_hybrid['state'].dsp,
        ocp.area)
    # error in flux
    error[3] \
        = (dsp_norm_P1(sol_mixed['adjoint'].prs[:, 0]
        - sol_hybrid['adjoint'].prs[:, 0], ocp.area, ocp.Mat)**2
        + dsp_norm_P1(sol_mixed['adjoint'].prs[:, 1]
        - sol_hybrid['adjoint'].prs[:, 1], ocp.area, ocp.Mat)**2) ** 0.5
    # error in temperature
    error[4] \
        = dsp_norm_P0(sol_mixed['adjoint'].dsp - sol_hybrid['adjoint'].dsp,
        ocp.area)

    return error


def mixed_vs_hybrid(spatial_range=9):
    """
    Prints the norms of difference between the computed optimal controls,
    states and adjoint states in the mixed and hybrid formulations.
    """

    data = [2**k + 1 for k in range(2, spatial_range+1)]
    error_data = np.zeros((len(data), 5)).astype(float)
    print_line()
    print("\t\tDIFFERENCE BETWEEN MIXED AND HYBRID FORMULATIONS")
    print_line()
    txt = "Control\t\tFlux\t\tTemp\t\tAdjoint Flux\tAdjoint Temp"
    print(txt)
    row_data = "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}"
    i = 0
    for num in data:
        error_data[i, :] = mixed_vs_hybrid_main(num)
        print(row_data.format(error_data[i, 0], error_data[i, 1],
            error_data[i, 2], error_data[i, 3], error_data[i, 4]))
        i += 1
    np.save('mixed_vs_hybrid.npy', error_data)
    print_line()
    print('\n')


def reduction():
    """
    Prints the number of elements, number of edges, degrees of freedom,
    reduced number of degrees of freedom and the reduction rate.
    """

    data = [2**k + 1 for k in range(2, 10)]

    print_line()
    print("\t\t\tREDUCTION OF DEGREES OF FREEDOM")
    print_line()
    txt = "k = {}\nN_Kh = {}\nN_eh = {}\ndof = {}\n"
    txt += "reduced dof = {}\nreduction = {:2.2f}%\n"
    k = 2

    for num in data:
        mesh = square_uni_trimesh(num)
        N_Kh = mesh.num_elems
        N_eh = len(mesh.int_edge)
        print(txt.format(k, N_Kh, N_eh, 4*N_Kh + N_eh, 3*N_Kh,
            100 * (N_Kh + N_eh) / (4*N_Kh + N_eh)))
        k += 1

    print_line()


if __name__ == '__main__':
    print_start()
    error_analysis("penalty")
    error_analysis("spatial")
    mixed_vs_hybrid()
    reduction()
    print_end()
