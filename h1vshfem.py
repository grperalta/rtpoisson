from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from scipy import sparse as sp
import numpy as np
import h1poisson
import matplotlib

def generate_data():
    """
    Generate the errors for the primal, dual and control variables using the H1-conforming FEM.
    """
    gamma = 0.1
    subdivision = [2**k + 1 for k in range(2, 10)]
    error = np.zeros((len(subdivision), 6)).astype(float)
    i = 0
    for N in subdivision:
        mesh = h1poisson.MeshTri.square_uni_trimesh(N)                   # triangulatiom
        mesh.set_bdy_node_indices()                                      # boundary nodes
        femstruct = h1poisson.get_fem_data_struct(mesh)                  # generate fem data structure
        A, M = h1poisson.assemble(mesh, femstruct)                       # stiffness and mass matrices
        A = h1poisson.apply_noslip_bc(A, mesh.bdy_node_indices)          # apply Dirichlet conditions
        x, y = mesh.node[:, 0], mesh.node[:, 1]                          # coordinates of nodes
        e = np.sin(2.0 * np.math.pi * x) * np.sin(2.0 * np.math.pi * y)  # eigenfunction
        ud = (1.0 - 8.0 * np.math.pi**2) * e                             # desired temperature
        f = (8.0 * np.math.pi**2 + (1.0 / gamma)) * e                    # source function
        qe = - (1.0 / gamma) * e                                         # exact optimal control
        K = sp.bmat([[A, (1/gamma)*M], [-A-M, A]], format="csr")         # primal-dual matrix
        F = np.hstack([M*f, -M*ud - A*e])                                # right-hand side vector
        F[mesh.bdy_node_indices] = 0.0
        F[mesh.bdy_node_indices + mesh.num_node] = 0.0
        u, w = np.split(spsolve(K, F), 2)                                # temperature and its adjoint
        q = -(1/gamma)*w                                                 # numerical optimal control
        error[i, 0] = np.sqrt(np.dot(M*(q - qe), q - qe))                # error in control
        error[i, 1] = np.sqrt(np.dot(A*(u - e), u - e))                  # error in flux
        error[i, 2] = np.sqrt(np.dot(M*(u - e), u - e))                  # error in temperature
        error[i, 3] = np.sqrt(np.dot(A*(w - e), w - e))                  # error in adjoint flux
        error[i, 4] = np.sqrt(np.dot(M*(w - e), w - e))                  # error in adjoint temperature
        i = i + 1
    np.save('h1fem_error.npy', error) 
   
    
def plot_errors():
    """
    Log-log plots of the errors obtained from the H1-conforming FEM and penalized hybrid FEM with
    post-processing.
    """
    matplotlib.rcParams['text.usetex'] = True
    error = np.load('spatial_error.npy')
    h1_error = np.load('h1fem_error.npy')
    markers = ['^', 'o', 's', 'o', 's']
    linestyles = ['-', '-', '-', '-', '-']
    linestyles_h1 = ['--', '--', '--', '--', '--']
    colors = ['red', 'black', 'blue', 'black', 'blue']
    labels = [r'$\|\bar{q}_h - \bar{q}_{h\varepsilon}\|$',
          r'$\|\bar{\sigma}_h - \bar{\sigma}_{h\varepsilon}\|$',
          r'$\|\bar{u}_h - R_h^1\bar{\lambda}_{h\varepsilon}\|$',
          r'$\|\bar{\varphi}_h - \bar{\varphi}_{h\varepsilon}\|$',
          r'$\|\bar{w}_h - R_h^1\bar{\mu}_{h\varepsilon}\|$']
    labels_h1 = [r'$\|\widetilde{q}_h - \widehat{q}_{h}\|$',
          r'$\|\nabla\widetilde{u}_h - \nabla\widehat{u}_h\|$',
          r'$\|\widetilde{u}_h - \widehat{u}_h\|$',
          r'$\|\nabla\widetilde{w}_h - \nabla\widehat{w}_h\|$',
          r'$\|\widetilde{w}_h - \widehat{w}_h\|$',]
    fig = plt.figure(figsize=(10,4))
    # plots for the temperature and flux
    ax1 = fig.add_subplot(131)
    for index in [1, 2]:
        ax1.loglog(error[:, 5], h1_error[:, index], color=colors[index], marker=markers[index],
            ls=linestyles_h1[index], label=labels_h1[index], ms=4, lw=1)
        ax1.loglog(error[:, 5], error[:, index], color=colors[index], marker=markers[index], 
            ls=linestyles[index], label=labels[index], ms=4, lw=1)
    ax1.legend(loc='best')
    plt.grid(True)
    # plots for the dual temperature and dual flux
    ax2 = fig.add_subplot(132)
    for index in [3, 4]:
        ax2.loglog(error[:, 5], h1_error[:, index], color=colors[index], marker=markers[index], 
            ls=linestyles_h1[index], label=labels_h1[index], ms=4, lw=1)
        ax2.loglog(error[:, 5], error[:, index], color=colors[index], marker=markers[index], 
            ls=linestyles[index], label=labels[index], ms=4, lw=1)
    ax2.legend(loc='best')
    plt.grid(True)
    # plots for the control
    ax3 = fig.add_subplot(133)
    ax3.loglog(error[:, 5], h1_error[:, 0], color=colors[0], marker=markers[0], ls=linestyles_h1[0], 
        label=labels_h1[0], ms=4, lw=1)
    ax3.loglog(error[:, 5], error[:, 0], color=colors[0], marker=markers[0], ls=linestyles[0],
        label=labels[0], ms=4, lw=1)
    ax3.legend(loc='best')
    plt.grid(True)
    fig.savefig("h1vshfem.pdf", bbox_inches='tight', format='pdf', dpi=900)   
    plt.show()
    
    
if __name__ == "__main__":
    generate_data()
    plot_errors()
