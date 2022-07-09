# rtpoisson

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

Performance with the usual H1-conforming FEM was compared based on an 
eigenfunction of the Dirichlet Laplacian.

For more details, refer to the manuscript:
    G. Peralta, Error Estimates for Mixed and Hybrid FEM for Elliptic
    Optimal Control Problems with Penalizations, preprint.


Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
