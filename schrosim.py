import numpy as np
import cupy as cp
from npy_append_array import NpyAppendArray as npy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nn_lib import Sequential

import os


class SchroSim:

    h = 1   # Planck's constant divided by 2*pi (J*s)
    qe = -1 # electron charge
    qp = 1 # proton charge
    ep = cp.pi * 4 # vaccum permittivity 
    me = 1    # mass of electron in atomic mass units
    c = qe**2 / (ep * h * (1/137)) # speed of light
    pw = 0.04

    dims = (0,0)
    ext_potential_funcs = []

    electrons = []
    protons = np.empty(shape=[0, 2, 2])
    V = np.empty([])
    ev = np.empty([])
    pev = np.empty([])

    dau = 0
    dt = 0
    n_dim = 0

    simulation_frames = []
    simulation_frames_ev = []

    # Calculate the partial derivative of the wave function with respect to time
    def d_dt(self, phi):
        
        self.e_field(phi)
        d_dxdx = cp.diff(phi, axis=2, n=2, prepend=0, append=0) / self.dau
        d_dxdx += (cp.diff(phi, axis=3, n=2, prepend=0, append=0)  / self.dau)

        return (d_dxdx * self.h * 1j / (2 * self.me)) + ((1j * self.qe * (self.ev + self.pev) * phi) / self.h) + ((1j * phi * self.V) / self.h)

    # Calculate the electric field potentials produced by the charged particles in the system
    def e_field(self, phi):

        if self.protons.shape[0] > 0:
            p_rad = cp.sqrt(cp.sum(cp.square(self.coor - self.protons[:,0].reshape([-1, 2, 1, 1])), axis=1, keepdims=True))
            phi = cp.where(cp.min(p_rad, axis=0) <= self.pw, 0, phi)
            self.pev = cp.nansum(self.qp / (self.ep * p_rad), axis=0, keepdims=True)
        else:
            self.pev = cp.zeros([1])
        
        positions = cp.empty([0, 2])
        for el in phi:
            pos = self.coor.reshape([2, -1]).T[cp.random.choice(el.size, size=1, p=(cp.abs(el.reshape([-1])) / cp.sum(abs(el))))]
            positions = cp.append(positions, pos, axis=0)

        self.ev = self.qe / (self.ep * cp.sqrt(cp.sum(cp.square(self.coor - positions.reshape([-1,2,1,1])), axis=1, keepdims=True)))
        self.ev = cp.where(cp.isfinite(self.ev), self.ev, 0)

        self.ev = cp.sum(self.ev, axis=0, keepdims=True) - self.ev


    # Calculates the integral over time using the Rk4 method
    def rk4(self, phi, **kwargs):
        ka = cp.zeros(phi.shape, dtype=phi.dtype)

        kn = self.d_dt(phi, **kwargs)
        ka += kn

        kn = self.d_dt(phi+(self.dt * (kn/2)), **kwargs)
        ka += (kn * 2)

        kn = self.d_dt(phi+(self.dt*(kn/2)), **kwargs)
        ka += (kn * 2)

        ka += self.d_dt(phi+(self.dt*kn), **kwargs)

        return phi + ((ka/6) * self.dt)
    
    # Calculates the integral over time using the euler method
    def euler(self, phi, **kwargs):
        return phi + (self.d_dt(phi, **kwargs) * self.dt)
    

    # convienance function for adding a constant potential to the environment via a lambda function
    def add_potential(self, func):
        self.ext_potential_funcs.append(func)

    # Normalize the wave function 
    def norm(self, phi):
        if self.protons.shape[0] > 0:
            p_rad = cp.sqrt(cp.sum(cp.square(self.coor - self.protons[:,0].reshape([-1, 2, 1, 1])), axis=1, keepdims=True))
            phi = cp.where(cp.min(p_rad, axis=0) <= self.pw, 0, phi)

        return phi / cp.sqrt(cp.nansum(cp.square(cp.abs(phi)), axis=(1,2,3), keepdims=True) * self.dau**self.n_dim)
    
    # convienance function for adding an electron to the environment
    def add_electron(self, p=[0.0, 0.0], pos=[0.0, 0.0], sig=0.1):
        pos = cp.array(pos).reshape([2, 1, 1])
        p = cp.array(p).reshape([2, 1, 1])

        self.electrons.append(lambda x: cp.exp(1j * cp.sum(x[0] * p, axis=0, keepdims=True) - (cp.sum(cp.square(x[0] - pos), axis=0, keepdims=True) / sig**2)))

    # convienance function for adding a point like proton to the environment 
    def add_proton(self, vel=[0.0, 0.0], pos=[0.0, 0.0]):
        self.protons = np.append(self.protons, np.array([[pos, vel]]), axis=0)
            
    # Simulates the currently specified environment.
    def simulate(
            self, 
            dims, dau, steps=10000, time_arrow=1,
            frame_rate=False, 
            model = Sequential(), train_model = False
        ):
        """
        dims:   An int/float/tuple describing how large each spacial dimention is. If a single value is given then only 1 dimention is simulated
        dau:    step size in atomic units. Step size is applied to spacial and temporal steps
        steps:  number of time steps to simulate
        time_arrow: This product of time_arrow and dau will be used as the time step value. If set to 1j, then the simulation will step towards the ground state of the system (theoretically)
        save:   a bool to determine weather to save the simulation steps. The combined electron wave functions and the combined non electron electric potentials will be saved
        checkpoint: how often to save the simulated steps to the save file. After saved to file the held states are cleared to save memory
        sim_name:   The name that will be used as the folder name to hold the saved files
        frame_rate: how often to record a frame for animation. If None then no frames will be availible for animation
        """
        if isinstance(dims, (int, float)):
            dims = (dims, )

        self.dt = dau * time_arrow

        self.n_dim = len(dims)
        if self.n_dim >= 3:
            raise Exception('Error: 3D simulations are not currently supported')

        # set the dimentions of the environment matrix and define coordinates
        self.dau = dau
        self.dims = [cp.arange(-(dim/2), (dim/2), dau) for dim in dims] + [cp.array([0])]*(2 - len(dims))
        self.coor = cp.stack(cp.meshgrid(*self.dims, indexing='ij'), axis=0)[cp.newaxis]
        
        self.protons = cp.array(self.protons)
        
        # initialize environment with specified electron locations and momentums
        phi = self.norm(cp.stack([el(self.coor) for el in self.electrons], axis=0))
        
        # create static potential matrix with specified lambda functions
        if len(self.ext_potential_funcs) > 0:
            self.V = cp.sum(cp.stack([v_func(self.coor) for v_func in self.ext_potential_funcs], axis=0), axis=0).astype(cp.float16)
        else:
            self.V = cp.zeros(self.coor.shape[2:], dtype=cp.float16)


        self.dims = [d.get() for d in self.dims]
        cp._default_memory_pool.free_all_blocks()

        self.e_field(phi)

        self.simulation_frames = []
        self.simulation_frames_ev = []
        
        simulation_steps = cp.empty([0, phi.shape[2], phi.shape[3], 4])
        simulation_steps = cp.append(
            simulation_steps,
            cp.stack([
                cp.squeeze(cp.real(cp.sum(phi, axis=0)))*(self.dau**self.n_dim),
                cp.squeeze(cp.imag(cp.sum(phi, axis=0)))*(self.dau**self.n_dim),
                cp.abs(cp.squeeze(cp.sum(self.ev, axis=0))) / 8,
                cp.abs(cp.squeeze(self.pev + self.V)) / 8
            ], axis=-1)[cp.newaxis],
            axis=0
        )


        if frame_rate:
            self.simulation_frames.append(cp.sum(phi, axis=0).get())
            self.simulation_frames_ev.append((cp.sum(self.ev, axis=0) + self.V + self.pev).get())


        for i in range(1, steps+1):
            if train_model:
                simulation_steps = cp.append(
                    simulation_steps,
                    cp.stack([
                        cp.squeeze(cp.real(cp.sum(phi, axis=0)))*(self.dau**self.n_dim),
                        cp.squeeze(cp.imag(cp.sum(phi, axis=0)))*(self.dau**self.n_dim),
                        cp.abs(cp.squeeze(cp.sum(self.ev, axis=0))) / 8,
                        cp.abs(cp.squeeze(self.pev + self.V)) / 8
                    ], axis=-1)[cp.newaxis],
                    axis=0
                )

                if (i % train_model == 0) or (i == steps):
                    model.train_batch(simulation_steps[:-1], simulation_steps[1:], epochs=1)
                    simulation_steps = cp.empty([0, phi.shape[2], phi.shape[3], 4])
                    print(f'Model Trained at Step {i}')
            

            if frame_rate and (i % frame_rate == 0):
                print(f'Step: {i}')
                self.simulation_frames.append(cp.sum(phi, axis=0).get())
                self.simulation_frames_ev.append((cp.sum(self.ev, axis=0) + self.V + self.pev).get())

            phi = self.norm(self.rk4(phi))


        # free up vram and return saved time steps
        self.V = self.V.get()
        self.coor = self.coor.get()
        self.protons = self.protons.get()
        self.ev = self.ev.get()
        self.pev = self.pev.get()
        cp._default_memory_pool.free_all_blocks()  

        
    
    # Animates the saved time steps in 1 or 2 dimentions. 
    # 3D visualization isn't currently working.
    def animate(self, interval=1):

        fig = plt.figure(figsize=(8,8))

        if self.n_dim == 1:
            ax = fig.add_subplot()
            x = np.squeeze(self.coor[0][0])

            plot = [ax.plot(x, np.squeeze(np.real(self.simulation_frames[0])), label='real')[0],
                    ax.plot(x, np.squeeze(np.imag(self.simulation_frames[0])), label='imag')[0],
                    ax.plot(x, np.squeeze(np.abs(self.simulation_frames[0])), label='P')[0],
                    ax.plot(x, np.squeeze(self.simulation_frames_ev[0])*-1)[0]]
            
            def animate(frame):
                plot[0].set_data((x, np.squeeze(np.real(self.simulation_frames[frame]))))
                plot[1].set_data((x, np.squeeze(np.imag(self.simulation_frames[frame]))))
                plot[2].set_data((x, np.squeeze(np.abs(self.simulation_frames[frame]))))
                plot[3].set_data((x, np.squeeze(self.simulation_frames_ev[frame])*-1))
                
                return plot

            ax.set_ylim(-3, 3)

        elif self.n_dim == 2:
            x = np.squeeze(self.coor[0][0])
            y = np.squeeze(self.coor[0][1])

            ax = fig.add_subplot(111, projection='3d')

            processed_frames = []
            for phi, V in zip(self.simulation_frames, self.simulation_frames_ev):
                f = [
                    np.squeeze(np.real(phi)),
                    np.squeeze(np.imag(phi))-0.5,
                    np.squeeze(np.abs(phi))+0.5,
                    (np.squeeze(V)*-1)-1
                ]
                processed_frames.append(f)

            plot = [ax.plot_surface(x, y, Z=processed_frames[0][0], cmap='magma', rcount=10, ccount=10, vmin=-0.005, vmax=0.005),
                    ax.plot_surface(x, y, Z=processed_frames[0][1], cmap='jet', rcount=10, ccount=10, vmin=-0.505, vmax=-0.495),
                    ax.plot_surface(x, y, Z=processed_frames[0][2], cmap='plasma', rcount=10, ccount=10),
                    ax.plot_surface(x, y, Z=processed_frames[0][3], cmap='gray', alpha=0.4, rcount=10, ccount=10)]

            def animate(frame):
                plot[0].remove()
                plot[1].remove()
                plot[2].remove()
                plot[3].remove()

                plot[0] = ax.plot_surface(x, y, Z=processed_frames[frame][0], cmap='magma', rcount=10, ccount=10, vmin=-0.005, vmax=0.005)
                plot[1] = ax.plot_surface(x, y, Z=processed_frames[frame][1], cmap='jet', rcount=10, ccount=10, vmin=-0.505, vmax=-0.495)
                plot[2] = ax.plot_surface(x, y, Z=processed_frames[frame][2], cmap='plasma', rcount=10, ccount=10)
                plot[3] = ax.plot_surface(x, y, Z=processed_frames[frame][3], cmap='gray', alpha=0.4, rcount=10, ccount=10)

            ax.set_zlim(-3, 3)
        
        elif self.n_dim >= 3:
            raise Exception('Error: There is no animation method for 3d simulations at this time')

        anim = FuncAnimation(fig, animate, frames=int(len(self.simulation_frames)), interval=interval)
        return anim
    

    




