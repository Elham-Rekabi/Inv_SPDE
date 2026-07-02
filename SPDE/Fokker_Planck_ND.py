import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from SPDE.finite_differences import hessian, gradient

class FokkerPlanckND(nn.Module):

    def __init__(self, drift, diffusion, range_x , spacing, T = 10.0, dt=0.001, mean_0 = None, cov_0 = None, device = None, dtype = None):
        super(FokkerPlanckND, self).__init__()

        self.drift_fun = drift["function"]
        self.drift_params = drift["params"]
        self.diffusion_fun = diffusion["function"]
        self.diffusion_params = diffusion["params"]
        self.range_x = range_x
        self.spacing = spacing

        self.T = T
        self.dt = dt

        dim = len(spacing)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        self.device = device
        self.dtype = dtype


        if mean_0 is None:
            mean_0 = [(0.5*(rng[1]+rng[0])) for rng in range_x]
        if cov_0 is None:
            cov_0 = torch.eye(dim, device=self.device, dtype=self.dtype)
        
        self.mean_0 = torch.as_tensor(mean_0, device=self.device, dtype=self.dtype)
        self.cov_0 = torch.as_tensor(cov_0, device=self.device, dtype=self.dtype)


    def solve(self, store_every_n_steps = 100, method = 'dopri5'):

        T = self.T
        dt = self.dt
        num_steps = int(T / dt)
        t = torch.linspace(0, T, num_steps + 1, device=self.device, dtype=self.dtype)

        x = self.discretize_grid()
        x = x.to(self.device, self.dtype)
        self.x = x
        p0 = self.Gaussian_Initialization(x, mean=self.mean_0, cov=self.cov_0)
        p0 = self.normalize_distribution(p0)
        p = p0.clone()
        p_trajectory = [p.clone()]

        if method == 'RK4':
            for n in range(num_steps):
                p = self.rk4_step(t[n], p, dt)
                if (n % store_every_n_steps == 0) or (n == num_steps - 1):
                    print(f"Step {n}/{num_steps} completed.")
                    p_trajectory.append(p.clone())

            p_trajectory = torch.stack(p_trajectory, dim=0)

        elif method == 'dopri5':
            p_trajectory = odeint(self.forward, p0, t, method='dopri5', rtol=1e-6, atol=1e-8)
            p = p_trajectory[-1]
            print("ODE solution completed.")

        else:
            raise ValueError(f"Unknown method: {method}")


        
        self.p_trajectory = p_trajectory

        return t, x, p, p_trajectory



    def forward(self, t, p):
  
        # p : (*grid)
        x = self.x # (*grid)
        spacing = self.spacing # list with length d
        dim = len(spacing)

        
        # Drift term: 
        # \nabla.(bp) = (\nabla.b)p + b.\nabla p
        # -------------------------
        # drift =  - div(bp)
        # -------------------------

        b = self.drift_fun(t, x, self.drift_params) # (*grid x dim)
        div_bp = torch.zeros_like(p)

        for i in range(0, dim):
            dbi_p_dx = gradient(b[...,i]*p, spacing)[...,i]
            div_bp += dbi_p_dx

        drift = -div_bp
        
        # -------------------------
        # Diffusion term
        # sum_ij d²/dx_i dx_j (D_ij p)
        # where D = 0.5 sigma sigma^T
        # -------------------------

        sigma = self.diffusion_fun(t, x, self.diffusion_params)

        D = 0.5 * torch.einsum('...ik,...jk->...ij', sigma, sigma)

        diffusion = torch.zeros_like(p)

        for i in range(0, dim):
            for j in range(0, dim):
                Dij_p = D[...,i,j] * p
                d2p_dxidxj = hessian(Dij_p, spacing)[...,i,j]
                diffusion += d2p_dxidxj

        # Final Fokker-Planck Equation

        dpdt = drift + diffusion

        return dpdt

    def rk4_step(self, t, p, dt):

        k1 = self.forward(t, p)
        k2 = self.forward(t + dt / 2, p + dt * k1 / 2)
        k3 = self.forward(t + dt / 2, p + dt * k2 / 2)
        k4 = self.forward(t + dt, p + dt * k3)

        return p + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    

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
    
    def normalize_distribution(self, p):

        # Compute the volume element based on the grid spacing
        volume_element = torch.prod(torch.tensor(self.spacing, device=p.device, dtype=p.dtype))

        # Normalize the distribution
        p_normalized = p / (torch.sum(p) * volume_element)

        return p_normalized
    

    
    def discretize_grid(self):

        # Create a grid of points in the specified range and spacing

        grid_list = []

        for rng, dx in zip(self.range_x, self.spacing):

            num_points = int((rng[1] - rng[0]) / dx) + 1
            grid_points = torch.linspace(rng[0], rng[1], num_points, device=self.device, dtype=self.dtype)
            grid_list.append(grid_points)

        mesh = torch.meshgrid(*grid_list, indexing='ij')

        x = torch.stack(mesh, dim=-1)

        return x



        





        






        


            



    










       


        

        

                    









        



            



            





            

    




        
