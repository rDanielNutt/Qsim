import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
import nvidia_smi


class SchroSim:

    h = 1   # Planck's constant divided by 2*pi (J*s)
    qe = -1 # electron charge
    qp = 1 # proton charge
    ep = 1 # vaccum permittivity 
    me = 1    # mass of electron in atomic mass units
    c = qe**2 / (ep * h * (1/137)) # speed of light

    dims = (0,0,0)
    ext_potential_funcs = []

    electrons = []
    protons = np.empty(shape=[0, 2, 3])
    V = np.empty([])
    ev = np.empty([])

    dau = 0
    dt = 0
    n_dim = 0
    sim_batches = 1

    kernel = cp.array([1, -2, 1])

    # Calculate the partial derivative of the wave function with respect to time
    def d_dt(self, phi):
        
        self.e_field(phi)
        d_dxdx = cp.diff(phi, axis=1, n=2, prepend=0, append=0) / self.dau
        d_dxdx += (cp.diff(phi, axis=2, n=2, prepend=0, append=0)  / self.dau)

        return (d_dxdx * self.h * 1j / (2 * self.me)) + ((1j * self.qe * self.ev * phi) / self.h)

    # Calculate the electric field vector potentials produced by the charged particles in the system
    def e_field(self, phi):
        
        self.ev = cp.empty([0, *phi.shape[1:]])
        positions = cp.empty([0, 3])
        for el in self.phi:
            pos = self.coor.reshape([3, -1])[:,cp.random.choice(el.size, p=el.reshape([-1]))]
            positions = cp.append(positions, pos, axis=0)

        self.ev = self.qe / (cp.pi * 4 * self.ep * cp.sqrt(cp.sum(cp.square(self.coor - positions.reshape([-1,3,1,1,1]), axis=1))))
        pev = self.qp / (cp.pi * 4 * self.ep * cp.sqrt(cp.sum(cp.square(self.coor - self.protons[:,0].reshape([-1,3,1,1,1])), axis=1)))
        pev = cp.sum(pev, axis=0, keepdims=True)

        self.ev = (cp.sum(self.ev, axis=0, keepdims=True) - self.ev)
        self.ev += pev


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
        phi = cp.where((cp.abs(self.coor[1]) <= 0.04) & (cp.abs(self.coor[2]) <= 0.04), 0, phi)
        norm = cp.sum(cp.square(cp.abs(phi))) * self.dau
        return phi / cp.sqrt(norm)
    
    # convienance function for adding an electron to the environment
    def add_electron(self, p=[0.0, 0.0, 0.0], pos=[0.0, 0.0, 0.0], sig=0.1):
        pos = cp.array(pos + [0]*(3-len(pos))).reshape([3, 1, 1, 1])
        p = cp.array(p + [0]*(3-len(p))).reshape([3, 1, 1, 1])

        self.electrons.append(lambda x: cp.exp(1j * cp.sum(x * p, axis=0) - (cp.sum(cp.square(x - pos), axis=0) / sig**2)))

    # convienance function for adding a point like proton to the environment 
    def add_proton(self, vel=[0.0, 0.0, 0.0], pos=[0.0, 0.0, 0.0]):
        self.protons = np.append(self.protons, np.array([[pos, vel]]), axis=0)
            
    # Simulates the currently specified environment.
    def simulate(self, dims, dau, steps=10000, 
                 save_rate=None, check_point=None, path=None, sim_name=None, 
                 frame_rate=None, imag_time=False,
                 batch_calc=True):
        """
        dims:   An int/float/tuple describing how large each spacial dimention is. If a single value is given then only 1 dimention is simulated
        dau:    step size in atomic units. Step size is applied to spacial and temporal steps
        steps:  number of time steps to simulate
        save_rate: how often to record the state of the simulation
        checkpoint: how often to save the simulation to the save file
        file_name: name of the save file
        frame_rate: how often to record a frame for animation. If None then no animation will be generated
        save_rate: The interval of time steps between saved states
        imag_time: If True then the time steps will be taken in the -1j * dau direction, otherwise time steps through real values    
        """
        if isinstance(dims, (int, float)):
            dims = (dims, )

        if imag_time:
            self.dt = dau * 1j
        else:
            self.dt = dau

        self.n_dim = len(dims)

        # set the dimentions of the environment matrix and define coordinates
        self.dau = dau
        self.dims = [cp.arange(-(dim/2), (dim/2), dau) for dim in dims] + [cp.array([0])]*(3 - len(dims))
        self.coor = cp.stack(cp.meshgrid(*self.dims, indexing='ij'), axis=0)[cp.newaxis]
        self.protons = cp.array(self.protons)

        # initialize environment with specified electron locations and momentums
        phi = self.norm(cp.stack([el(self.coor) for el in self.electrons]), axis=0)

        # create static potential matrix with specified lambda functions
        if len(self.ext_potential_funcs) > 0:
            self.V = cp.sum(cp.stack([v_func(self.coor) for v_func in self.ext_potential_funcs], axis=0), axis=0).astype(cp.float16)
        else:
            self.V = cp.zeros(self.coor.shape[1:], dtype=cp.float16)


        self.dims = [d.get() for d in self.dims]
        cp._default_memory_pool.free_all_blocks()

        if batch_calc:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            batch_lim = info.free // (4 * self.coor.nbytes)
            nvidia_smi.nvmlShutdown()
        else:
            batch_lim = phi.size // 16


        self.sim_batches = [range(i, i+batch_lim) for i in range(0, phi.size-batch_lim, batch_lim)]
        self.sim_batches += [range(len(self.sim_batches)*batch_lim, phi.size)]
        
        self.e_field(phi)

        simulation_steps_phi = []
        simulation_steps_ef = []
        simulation_steps_protons = []
        simulation_frames = []

        if path:
            path = f'{path}{sim_name}/'
            if not os.path.exists(path):
                os.mkdir(path)

        simulation_steps_phi.append(cp.sum(phi, axis=0).get())
        simulation_steps_protons.append(self.protons.get())

        simulation_frames.append(phi.get())

        # Save the initial state and iterate through time steps
        for i in range(1, steps+1):
            if (save_rate is not None) and (i % save_rate == 0):
                simulation_steps_phi.append(cp.sum(phi, axis=0).get())
                simulation_steps_protons.append(self.protons.get())

                if i % check_point == 0:
                    with open(f'{path}{sim_name}_phi.npy', 'ab') as f:
                        np.save(f, np.stack(simulation_steps_phi, axis=0))
                    simulation_steps_phi = []
                    with open(f'{path}{sim_name}_ef.npy', 'ab') as f:
                        np.save(f, np.stack(simulation_steps_ef, axis=0))
                    simulation_steps_ef = []
                    with open(f'{path}{sim_name}_proton.npy', 'ab') as f:
                        np.save(f, np.stack(simulation_steps_protons, axis=0))
                    simulation_steps_protons = []


            if i % frame_rate == 0:
                print(f'Step: {i}')
                simulation_frames.append(phi.get())

            phi = self.norm(self.rk4(phi))
            
        # free up vram and return saved time steps
        self.V = self.V.get()
        self.coor = self.coor.get()
        self.protons = self.protons.get()
        cp._default_memory_pool.free_all_blocks()  

        if len(simulation_frames) > 1:
            return self.animate(simulation_frames)
        
    
    # Animates the saved time steps in 1 or 2 dimentions. 
    # 3D visualization isn't currently working.
    def animate(self, frames, interval=1):

        fig = plt.figure(figsize=(8,8))

        if self.n_dim == 1:
            ax = fig.add_subplot()
            x = np.squeeze(self.coor[0])

            plot = [ax.plot(x, np.squeeze(np.real(frames[0])), label='real')[0],
                    ax.plot(x, np.squeeze(np.imag(frames[0])), label='imag')[0],
                    ax.plot(x, np.squeeze(np.abs(frames[0])), label='P')[0],
                    ax.plot(x, np.squeeze(self.V))[0]]
            
            def animate(frame):
                plot[0].set_data((x, np.squeeze(np.real(frames[frame]))))
                plot[1].set_data((x, np.squeeze(np.imag(frames[frame]))))
                plot[2].set_data((x, np.squeeze(np.abs(frames[frame]))))
                
                return plot

            ax.set_ylim(-3, 3)

        elif self.n_dim == 2:
            x = np.squeeze(self.coor[0])
            y = np.squeeze(self.coor[1])

            ax = fig.add_subplot(111, projection='3d')

            processed_frames = []
            for frame in frames:
                f = [
                    np.squeeze(np.real(frame)),
                    np.squeeze(np.imag(frame))-0.5,
                    np.squeeze(np.abs(frame))+0.5,
                ]
                processed_frames.append(f)

            plot = [ax.plot_surface(x, y, Z=processed_frames[0][0], cmap='magma', rcount=10, ccount=10, vmin=-0.005, vmax=0.005),
                    ax.plot_surface(x, y, Z=processed_frames[0][1], cmap='jet', rcount=10, ccount=10, vmin=-0.505, vmax=-0.495),
                    ax.plot_surface(x, y, Z=processed_frames[0][2], cmap='plasma', rcount=10, ccount=10),
                    ax.plot_surface(x, y, np.squeeze(self.V)-1.5, cmap='gray', alpha=0.4, rcount=5, ccount=5)]

            def animate(frame):
                plot[0].remove()
                plot[1].remove()
                plot[2].remove()

                plot[0] = ax.plot_surface(x, y, Z=processed_frames[frame][0], cmap='magma', rcount=10, ccount=10, vmin=-0.005, vmax=0.005)
                plot[1] = ax.plot_surface(x, y, Z=processed_frames[frame][1], cmap='jet', rcount=10, ccount=10, vmin=-0.505, vmax=-0.495)
                plot[2] = ax.plot_surface(x, y, Z=processed_frames[frame][2], cmap='plasma', rcount=10, ccount=10)

            ax.set_zlim(-1.25, 1.25)
        
        elif self.n_dim >= 3:
            raise Exception('Error: There is no animation method for 3d simulations at this time')

        anim = FuncAnimation(fig, animate, frames=int(len(frames)), interval=interval)
        return anim
    

    




