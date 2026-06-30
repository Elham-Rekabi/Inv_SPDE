import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from SPDE.finite_differences import First_order_derivative,  Second_order_derivative

class FokkerPlanck1D(nn.Module):
    def __init__(self, drift, diffusion, range_x =(-10, 10), num_points=1000, T = 10.0, dt=0.01, mean_0=0.0, std_0=1.0):
        super(FokkerPlanck1D, self).__init__()
        self.drift_fun = drift["function"]
        self.drift_params = drift["params"]
        self.diffusion_fun = diffusion["function"]
        self.diffusion_params = diffusion["params"]
        self.range_x = range_x
        self.num_points = num_points
        self.T = T
        self.dt = dt
        self.mean_0 = mean_0
        self.std_0 = std_0

        # spatial grid
        x = torch.linspace(range_x[0], range_x[1], num_points)
        dx = x[1] - x[0]

        self.register_buffer("x", x)
        self.register_buffer("dx", dx)


    def solve(self):

        x = self.x
        device = x.device

        t = torch.arange(0, self.T + self.dt, self.dt, device=device)
      

        p0 = self.Gaussian_Initialization(self.x, mean=self.mean_0, std=self.std_0)
        p = odeint(self.forward, p0, t, method='dopri5', rtol=1e-6,atol=1e-8)
        print("ODE solution completed.")
        #p = p/torch.sum(p * self.dx, dim=1, keepdim=True) # Normalize at each time step
        self.p = p


        return t, x, p


    def Gaussian_Initialization(self, x, mean=0.0, std=1.0):

        dx = self.dx
        p0 = torch.exp(-0.5 * ((x - mean) / std) ** 2)
        p0 = p0/torch.sum(p0 * dx) # Normalize the distribution so that it integrates to 1

        return p0


    
    def forward(self, t, p):

        x = self.x

        b = self.drift_fun(t, x, self.drift_params)
        sigma = self.diffusion_fun(t, x, self.diffusion_params)

        dx = self.dx
        bp = b * p
        Dp = 0.5* sigma * sigma * p

        dpdt = -First_order_derivative(bp, dx ) + Second_order_derivative(Dp, dx)
        dpdt[0] = 0.0
        dpdt[-1] = 0.0

        return dpdt
    
    
    


    

    

    




    