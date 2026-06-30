import numpy as np
import torch

spacing = [1.0, 1.0, 1.0]
u = torch.rand(5, 5, 5) # 
print(u.shape)
grad = torch.zeros((*u.shape, len(spacing)), device=u.device)
print(grad.shape)
print(u.ndim)


for idx, dxi in enumerate(spacing):

    center = [slice(None)] * u.ndim
    print(center)
    left   = [slice(None)] * u.ndim
    print(left)
    right  = [slice(None)] * u.ndim
    print(right)

    center[idx] = slice(1, -1)
    print(center)
    left[idx]   = slice(0, -2)
    print(left)
    right[idx]  = slice(2, None)
    print(right)

    print(tuple(center))
    print(u[tuple(center)].shape)
    print(tuple(left))
    print(u[tuple(left)].shape)
    print(tuple(right))
    print(u[tuple(right)].shape)

    

    grad[(*center, idx)] = (
        u[tuple(right)] - u[tuple(left)]
    ) / (2 * dxi)  

    # left boundary
    left_boundary = [slice(None)] * u.ndim
    print(left_boundary)
    left_boundary[idx] = 0
    print(left_boundary)
    print((*left_boundary, idx)), print(grad[(*left_boundary, idx)].shape)


left_boundary = [slice(None)] * 3
print(left_boundary)
left_boundary = [left_boundary] * 4
print(left_boundary)
