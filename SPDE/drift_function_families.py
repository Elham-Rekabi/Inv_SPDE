import torch
import numpy as np
import os



class drift_function_families:
    def __init__(self, fun_type):

        self.fun_type = fun_type

    def forward(self, t, x, params):

        if self.fun_type == 'linear':
            a = params[0]
            b = params[1]
