"""
Simple python package for the finite element method of the Poisson equation
on a two dimensional mesh. Implementation is based on the H1-conforming 
finite element method on triangles.

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
"""

from __future__ import division
from matplotlib import pyplot as plt
from scipy import sparse as sp
from meshtri import MeshTri
import numpy as np

__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2022, Gilbert Peralta"
__version__ = "1.0"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "9 July 2021"


class Basis():
    """
    Class for finite element basis.
    """
    def __init__(self, val, grad, dof, dim):
        """
        Class initialization.

        Attributes
        ----------
        val : array
            function values at quadrature nodes of basis functions with
            shape = (no. of basis elements) x (no. quadtrature points)
        grad : array
            gradient values at quadrature nodes of basis functions with
            shape = (no. of basis elements) x (no. quadrature points) x (dim)
        dof : int
            number of local nodal degrees of freedom
        dim : int
            dimension of element
        """
        self.val = val
        self.grad = grad
        self.dof = dof
        self.dim = dim


def p1basis(p):
    """
    P1 Lagrange finite element bases at the points p.

    Returns
    -------
    h1poisson.Basis class
    """
    dof, dim = 3, 2
    x, y = p[:, 0], p[:, 1]
    numnode = p.shape[0]
    # initialization
    val = np.zeros((3, numnode)).astype(float)
    grad = np.zeros((3, numnode, 2)).astype(float)
    one = np.ones(numnode).astype(float)
    zero = np.zeros(numnode).astype(float)
    # basis function values
    val[0, :] = 1 - x - y
    val[1, :] = x
    val[2, :] = y
    # gradient values of basis functions
    grad[0, :, :] = np.array([-one, -one]).T
    grad[1, :, :] = np.array([ one, zero]).T
    grad[2, :, :] = np.array([zero,  one]).T
    return Basis(val, grad, dof, dim)


class Quadrature():
    """
    Class for numerical quadrature formulas.
    """

    def __init__(self, node, weight, dim, order):
        """
        Class initialization.

        Attributes
        ----------
        node : array
            quadrature nodes
        weight : array
            quadrature weights
        dim : int
            dimension of integration
        order : int
            order of quadrature rule
        """
        self.node = node
        self.weight = weight
        self.order = order
        self.dim = dim


def quad_gauss_tri(order):
    """
    Gauss integration on the reference triangle with vertices at
    (0, 0), (0, 1), and (1, 0).

    Parameters
    ----------
    order : int
        order of Gaussian quadrature

    Returns
    -------
    h1poisson.Quadrature class

    To do
    ------
    Include quadrature order higher than 6.
    """
    dim = 2
    if order == 1:
        node = np.array([1./3, 1./3])
        weight = np.array([1./2])
    elif order == 2:
        node = np.array([
            [0.1666666666666666666666, 0.6666666666666666666666,
             0.1666666666666666666666],
            [0.1666666666666666666666, 0.1666666666666666666666,
             0.6666666666666666666666]]).T
        weight = np.array(
            [0.1666666666666666666666,
             0.1666666666666666666666,
             0.1666666666666666666666])
    elif order == 3:
        node = np.array([
            [0.333333333333333, 0.200000000000000,
             0.600000000000000, 0.200000000000000],
            [0.333333333333333, 0.600000000000000,
             0.200000000000000, 0.200000000000000]]).T
        weight = np.array(
            [-0.28125000000000, 0.260416666666667,
             0.260416666666667, 0.260416666666667])
    elif order == 4:
        node = np.array([
            [0.4459484909159650, 0.0915762135097699, 0.1081030181680700,
             0.8168475729804590, 0.4459484909159650, 0.0915762135097710],
            [0.1081030181680700, 0.8168475729804590, 0.4459484909159650,
             0.0915762135097710, 0.4459484909159650, 0.0915762135097699]]).T
        weight = np.array(
            [0.111690794839006, 0.054975871827661, 0.111690794839006,
             0.054975871827661, 0.111690794839006, 0.054975871827661])
    elif order == 5:
        node = np.array([
            [0.333333333333333, 0.470142064105115, 0.101286507323457,
             0.059715871789770, 0.797426985353087, 0.470142064105115,
             0.101286507323456],
            [0.333333333333333, 0.059715871789770, 0.797426985353087,
             0.470142064105115, 0.101286507323456, 0.470142064105115,
             0.101286507323457]]).T
        weight = np.array(
            [0.1125000000000000, 0.0661970763942530, 0.0629695902724135,
             0.0661970763942530, 0.0629695902724135, 0.0661970763942530,
             0.0629695902724135])
    elif order == 6:
        node = np.array([
            [0.2492867451709110, 0.0630890144915021, 0.5014265096581790,
             0.8738219710169960, 0.2492867451709100, 0.0630890144915020,
             0.6365024991213990, 0.3103524510337840, 0.0531450498448170,
             0.0531450498448170, 0.3103524510337840, 0.6365024991213990],
            [0.5014265096581790, 0.8738219710169960, 0.2492867451709100,
             0.0630890144915020, 0.2492867451709110, 0.0630890144915021,
             0.0531450498448170, 0.0531450498448170, 0.3103524510337840,
             0.6365024991213990, 0.6365024991213990, 0.3103524510337840]]).T
        weight = np.array(
            [0.0583931378631895, 0.0254224531851035, 0.0583931378631895,
             0.0254224531851035, 0.0583931378631895, 0.0254224531851035,
             0.0414255378091870, 0.0414255378091870, 0.0414255378091870,
             0.0414255378091870, 0.0414255378091870, 0.0414255378091870])
    else:
        node = None
        weight = None
        order = None
        dim = None
        message = 'Number of quadrature order available up to 6 only.'
        raise UserWarning(message)
    return Quadrature(node, weight, dim, order)


class Transform():
    """
    Class for transformations from the reference element to the physical element.
    """

    def __init__(self):
        """
        Class initialization.

        Parameters
        ----------
        invmatT : numpy.ndarray
            Inverse transpose of the transformation matrices with
            shape = (no. of cell) x 2 x 2.
        det : numpy.ndarray
            Absolute value of determinants of the transformation matrices with
            length = (no. of cell).
        """
        self.invmatT = None
        self.det = None


def affine_transform(mesh):
    """
    Generates the affine transformations Tx = Ax + b from the
    reference triangle with vertices at (0, 0), (0, 1) and (1, 0)
    to each element of the mesh.

    Parameter
    ---------
    mesh : h1poisson.TriMesh class
        the domain triangulation

    Returns
    -------
    h1poisson.Transform class
    """
    transform = Transform()
    transform.invmatT = np.zeros((mesh.num_cell, 2, 2)).astype(float)
    # coordinates of the triangles with local indices 0, 1, 2
    A = mesh.node[mesh.cell[:, 0], :]
    B = mesh.node[mesh.cell[:, 1], :]
    C = mesh.node[mesh.cell[:, 2], :]
    a = B - A
    b = C - A
    transform.invmatT[:, 0, :] = a
    transform.invmatT[:, 1, :] = b
    transform.det = np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])
    transform.invmatT = np.linalg.inv(transform.invmatT)
    return transform


class FEMDataStruct():
    """
    Class for finite element data structure.
    """

    def __init__(self, quad, vbasis, dxvbasis, dyvbasis, transform):
        """
        Class initialization.

        Parameters
        ----------
        quad : h1poisson.Quadrature class
            Numerical quadrature data structure.
        vbasis : numpy.ndarray
            Velocity basis functions at quadrature nodes.
        dxvbasis : numpy.ndarray
            Derivative wrt x of basis functions at quadrature nodes.
        dyvbasis : numpy.ndarray
            Derivative wrt y of basis functions at quadrature nodes.
        transform : h1poisson.Transform class
            Affine transformations data structure.
        """
        self.quad = quad
        self.vbasis = vbasis
        self.dxvbasis = dxvbasis
        self.dyvbasis = dyvbasis
        self.transform = transform

    def __str__(self):
        return "{} FEM".format(self.name.upper())


def get_fem_data_struct(mesh, quad_order=6):
    """
    Returns the finite element data structure for matrix assembly.

    Parameters
    ----------
    mesh : h1poisson.Mesh class
        Triangulation of the domain.
    quad_order : int
        Order of numerical quadrature.
    Returns
    -------
    h1poisson.FEMDataStruct class
    """
    quad = quad_gauss_tri(quad_order)
    vbasis = p1basis(quad.node)
    transform = affine_transform(mesh)
    gradvbasis = np.zeros((mesh.num_cell, vbasis.dim, vbasis.dof,
        len(quad.node))).astype(float)
    # loop over all Gauss points
    for gpt in range(len(quad.node)):
        gradvbasis_temp = vbasis.grad[:, gpt, :]
        gradvbasis[:, :, :, gpt] = \
            np.dot(transform.invmatT.reshape(vbasis.dim*mesh.num_cell, vbasis.dim),
            gradvbasis_temp.T).reshape(mesh.num_cell, vbasis.dim,
            vbasis.dof)
    return FEMDataStruct(quad, vbasis.val, gradvbasis[:, 0, :, :],
        gradvbasis[:, 1, :, :], transform)


def assemble(mesh, femdatastruct):
    """
    Matrix assembly.

    Parameters
    ----------
    mesh : h1poisson.Mesh class
        Triangulation of the domain.
    femdatastruct : h1poisson.FEMDataStruct class
        Finite element data structure.

    Returns
    -------
    A, M : tuple of scipy.sparse.csr_matrix
        stiffness matrix A and mass matrix M
    """
    # pre-allocation of arrays for the matrix entries
    Ae = np.zeros((mesh.num_cell, 3, 3)).astype(float)
    Me = np.zeros((mesh.num_cell, 3, 3)).astype(float)
    # element contributions
    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        for j in range(3):
            for k in range(3):
                Me[:, j, k] += wgpt * femdatastruct.transform.det \
                    * phi[j] * phi[k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * dphidx[:, k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * dphidy[:, k]
    # pre-allocation of sparse matrices
    M = sp.csr_matrix((mesh.num_node, mesh.num_node)).astype(float)
    A = sp.csr_matrix((mesh.num_node, mesh.num_node)).astype(float)
    # sparse matrix assembly
    for j in range(3):
        row = mesh.cell[:, j]
        for k in range(3):
            col = mesh.cell[:, k]
            M += sp.coo_matrix((Me[:, j, k], (row, col)),
                    shape=(mesh.num_node, mesh.num_node)).tocsr()
            A += sp.coo_matrix((Ae[:, j, k], (row, col)),
                    shape=(mesh.num_node, mesh.num_node)).tocsr()
    return A, M


def apply_noslip_bc(A, index):
    """
    Apply no-slip boundary conditions to the sparse matrix A and convert it to
    csc format.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    A = A.tolil()
    if type(index) != list:
        index = list(index)
    A[index, :] = 0.0
    A[:, index] = 0.0
    for k in list(index):
        A[k, k] = 1.0
    return A.tocsc()
