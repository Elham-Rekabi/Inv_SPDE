import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

class FokkerPlanckND(nn.Module):
    def __init__(self, drift, diffusion, range_x =[[-10, 10], [-10, 10]],  num_points=[100, 100], T = 10.0, dt=0.01, mean_0=[0.0, 0.0], cov_0=[[1.0, 0.0], [0.0, 1.0]]):
        super(FokkerPlanckND, self).__init__()

        self.drift_fun = drift["function"]
        self.drift_params = drift["params"]
        self.diffusion_fun = diffusion["function"]
        self.diffusion_params = diffusion["params"]
        self.range_x = range_x
        self.num_points = num_points
        self.T = T
        self.dt = dt
        if isinstance(mean_0, torch.Tensor) is False:
            mean_0 = torch.tensor(mean_0)
        if isinstance(cov_0, torch.Tensor) is False:
            cov_0 = torch.tensor(cov_0)
       

    def Gaussian_Initialization(self, x, mean=None, cov=None):

        dim = x.shape[-1]

        if mean is None:
            mean = torch.zeros(dim, device=x.device, dtype=x.dtype)
        if cov is None:
            cov = torch.eye(dim, device=x.device, dtype=x.dtype)

        inv_cov = torch.linalg.inv(cov)
        det_cov = torch.linalg.det(cov)

        diff = x - mean

        norm = 1.0 / torch.sqrt(((2*np.pi) ** dim) * det_cov)

        exponent = -0.5 * torch.einsum('...i,ij,...j->...', diff, inv_cov, diff)
        
        p0 = norm * torch.exp(exponent)

        return p0

    
    
    
    

    def forward(self, t, p):

        x = self.x # N dimensional 

        b = self.drift_fun(t, x, self.drift_params) # N dimensional
        sigma = self.diffusion_fun(t, x, self.diffusion_params) # N by m dimensional

        dx = self.dx # spacing


        bp = b * p
        Dp = 0.5* sigma * sigma * p

        dpdt = -self.First_order_derivative(bp, dx ) + self.Second_order_derivative(Dp, dx)
        dpdt[0] = 0.0
        dpdt[-1] = 0.0

        return dpdt

                    









        



            



            





            

    




        