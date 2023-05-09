import cupy as cp
import numpy as np


class SchroModel:
    def __init__(self, model, dt:float, dau:float):
        self.model = model
        self.dt = dt
        self.dau = dau
        self.e_field = lambda x, y: x
        self.sim_steps = cp.empty([])
    
    def add_step(self, phi, ev, V):
        step = cp.stack([
                cp.sum(phi, axis=0),
                V.reshape(phi.shape[1:]) 
            ], axis=-1)
        self.sim_steps = cp.append(self.sim_steps, step, axis=0)

    def clear_steps(self):
        self.sim_steps = cp.empty([0, *self.sim_steps.shape[1:]])

    def train(self, epochs):
        self.model.train_batch(self.sim_steps[:-1, :, :, :], self.sim_steps[1:, :, :, :1], epochs=epochs)
        self.clear_steps()
        return np.mean(self.model.history[-epochs:])
    
    
    def predict(self, phi, n_samp):
        ev, V = self.e_field(phi, n_samp)
        self.add_step(phi, ev, V)

        pred = self.model.predict(self.sim_steps)[cp.newaxis, :, :, :, 0] 
        self.clear_steps()
        return pred
