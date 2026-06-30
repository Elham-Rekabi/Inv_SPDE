import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint


# 1D: 

def First_order_derivative(f, dx):

    dpdx = torch.zeros_like(f)
    dpdx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    dpdx[0] = (f[1] - f[0]) / dx
    dpdx[-1] = (f[-1] - f[-2]) / dx

    return dpdx
    
def Second_order_derivative(f, dx):

    d2pdx2 = torch.zeros_like(f)
    d2pdx2[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dx ** 2)
    d2pdx2[0] = 2 * (f[1] - f[0]) / (dx ** 2)  # Neumann BC 
    d2pdx2[-1] = 2 * (-f[-1] + f[-2]) / (dx ** 2)  # Neumann BC

    return d2pdx2

# ND

def gradient(u, spacing):

    # u: (N1, N2, ..., Nd)
    # spacing: list of length d, each element is the grid spacing in that dimension

    dim = len(spacing)

    grad = torch.zeros((*u.shape, dim), device=u.device)

    for idx, dxi in enumerate(spacing):

        # interior points: central difference

        center = [slice(None)] * u.ndim
        left   = [slice(None)] * u.ndim
        right  = [slice(None)] * u.ndim
        center[idx] = slice(1, -1)
        left[idx]   = slice(0, -2)
        right[idx]  = slice(2, None)
        grad[(*center, idx)] = (
            u[tuple(right)] - u[tuple(left)]
        ) / (2 * dxi)

        # boundary points: forward/backward difference

        # left boundary
        left_boundary = [slice(None)] * u.ndim
        left_boundary_side = left_boundary.copy()
        left_boundary[idx] = 0
        left_boundary_side[idx] = 1
        grad[(*left_boundary, idx)] = (u[tuple(left_boundary_side)] - u[tuple(left_boundary)])/dxi

        # right boundary
        
        right_boundary = [slice(None)] * u.ndim
        right_boundary_side = right_boundary.copy()
        right_boundary[idx] = -1
        right_boundary_side[idx] = -2
        grad[(*right_boundary, idx)] = (u[tuple(right_boundary)]-u[tuple(right_boundary_side)])/dxi


    return grad


def hessian(u, spacing): 

    # u: (N1, N2, ..., Nd)
    # spacing: list of length d, each element is the grid spacing in that dimension

    dim = len(spacing)

    hessian = torch.zeros((*u.shape, dim, dim), device=u.device, dtype=u.dtype)

    for idx, dxi in enumerate(spacing):

        # interior points: central difference

        center = [slice(None)] * u.ndim
        left   = [slice(None)] * u.ndim
        right  = [slice(None)] * u.ndim
        center[idx] = slice(1, -1)
        left[idx]   = slice(0, -2)
        right[idx]  = slice(2, None)

        hessian[(*center, idx, idx)] = (u[tuple(left)] - 2*u[tuple(center)] 
                                                    + u[tuple(right)] )/dxi**2
        
        # Boundaries

        # right Boundary

        rb0 = [slice(None)] * u.ndim 
        rb1 = [slice(None)] * u.ndim 
        rb2 = [slice(None)] * u.ndim 
        rb3 = [slice(None)] * u.ndim 
        
        rb0[idx] = -1
        rb1[idx] = -2
        rb2[idx] = -3
        rb3[idx] = -4

        hessian[(*rb0, idx, idx)] = (2*u[tuple(rb0)]-5*u[tuple(rb1)]
                                                +4*u[tuple(rb2)]-u[tuple(rb3)])/dxi**2

        # Left Boundary

        lb0 = [slice(None)] * u.ndim 
        lb1 = [slice(None)] * u.ndim 
        lb2 = [slice(None)] * u.ndim 
        lb3 = [slice(None)] * u.ndim 
        
        lb0[idx] = 0
        lb1[idx] = 1
        lb2[idx] = 2
        lb3[idx] = 3

        hessian[(*lb0, idx, idx)] = (2*u[tuple(lb0)]-5*u[tuple(lb1)]
                                                +4*u[tuple(lb2)]-u[tuple(lb3)])/dxi**2



        for  idx2, dxi2 in enumerate(spacing):

            if idx != idx2:

                # interior points

                '''center2 = [slice(None)] * u.ndim
                center2[idx] = slice(1, -1)
                center2[idx2] = slice(1, -1)

                NE = [slice(None)] * u.ndim
                NW = [slice(None)] * u.ndim

                SE = [slice(None)] * u.ndim
                SW = [slice(None)] * u.ndim


                NE[idx] = slice(2, None)
                NE[idx2] = slice(2, None)
                NW[idx] = slice(0, -2)
                NW[idx2] = slice(2, None)

                SE[idx] = slice(2, None)
                SE[idx2] = slice(0, -2)
                SW[idx] = slice(0, -2)
                SW[idx2] = slice(0, -2)

                hessian[(*center2, idx, idx2)] = (u[tuple(NE)]-u[tuple(SE)]
                                                    -u[tuple(NW)]+u[tuple(SW)])/(4.0 * dxi * dxi2)'''
                
                # Interior and Boundary mixed derivatives
                
                left_i = [
                    (0, -3.0 / (2.0 * dxi)),
                    (1, 4.0 / (2.0 * dxi)),
                    (2, -1.0 / (2.0 * dxi)),]
                
                right_i = [
                    (-1, 3.0 / (2.0 * dxi)),
                    (-2, -4.0 / (2.0 * dxi)),
                    (-3, 1.0 / (2.0 * dxi)),]
                
                interior_i = [
                    (slice(0, -2), -1.0 / (2.0 * dxi)),
                    (slice(2, None), 1.0 / (2.0 * dxi)),]
                

                left_j = [
                    (0, -3.0 / (2.0 * dxi2)),
                    (1, 4.0 / (2.0 * dxi2)),
                    (2, -1.0 / (2.0 * dxi2)),]
                
                right_j = [
                    (-1, 3.0 / (2.0 * dxi2)),
                    (-2, -4.0 / (2.0 * dxi2)),
                    (-3, 1.0 / (2.0 * dxi2)),]
                
                interior_j = [
                    (slice(0, -2), -1.0 / (2.0 * dxi2)),
                    (slice(2, None), 1.0 / (2.0 * dxi2)),]
                

                mixed_cases = [
                (slice(1, -1), interior_i, slice(1, -1), interior_j),
                (0, left_i, slice(1, -1), interior_j),
                (-1, right_i, slice(1, -1), interior_j),
                (slice(1, -1), interior_i, 0, left_j),
                (slice(1, -1), interior_i, -1, right_j),
                (0, left_i, 0, left_j),
                (0, left_i, -1, right_j),
                (-1, right_i, 0, left_j),
                (-1, right_i, -1, right_j),]
            

                for target_i, stencil_i, target_j, stencil_j in mixed_cases:

                    target = [slice(None)] * u.ndim
                    target[idx] = target_i
                    target[idx2] = target_j

                    mixed_value = 0.0

                    for source_i, coeff_i in stencil_i:
                        for source_j, coeff_j in stencil_j:

                            source = [slice(None)] * u.ndim
                            source[idx] = source_i
                            source[idx2] = source_j

                            mixed_value = (
                                mixed_value
                                + coeff_i * coeff_j * u[tuple(source)]
                            )

                    hessian[(*target, idx, idx2)] = mixed_value
                

    return hessian
