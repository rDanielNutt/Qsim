import numpy as np
import cupy as cp
from cupyx.scipy import fft
from cupyx.scipy.signal import fftconvolve
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from qsim.models import SchroModel

class SchroSim:

    def __init__(self, *, reduced_h=1, q_electron=-1, q_proton=1, m_electron=1, proton_width=0.04, vacuum_perm=1):
        self.h = reduced_h
        self.qe = q_electron
        self.qp = q_proton
        self.me = m_electron
        self.pw = proton_width

        self.dims = (0,0)
        self.ext_potential_funcs = []
        
        self.electrons = []
        self.protons = np.empty([0, 2])
        self.V = np.empty([])
        self.ev = np.empty([])
        self.dists = np.empty([])

        self.dau = 1e-2
        self.dt = 5e-3
        self.n_dim = 0

        self.ep = vacuum_perm
        self.c = self.qe**2 / (self.ep * self.h * (1/137))

        self.simulation_frames = []
        self.simulation_frames_ev = []

    # Calculate the partial derivative of the wave function with respect to time
    def d_dt(self, phi):
        
        d_dxdx = cp.diff(phi, axis=1, n=2, prepend=phi[:, -1:, :], append=phi[:, :1, :]) / self.dau
        d_dxdx += (cp.diff(phi, axis=2, n=2, prepend=phi[:, :, -1:], append=phi[:, :, :1])  / self.dau)

        return (d_dxdx * self.h * 1j / (2 * self.me)) + ((1j * phi * ((cp.sum(self.ev, axis=0, keepdims=True) - self.ev)  + self.V)) / self.h)

    # Calculate the electric field potentials produced by the charged particles in the system
    def e_field(self, phi, n_samp):

        if n_samp == 'full':

            pad = (phi.shape[1] / 2, phi.shape[2] / 2)
            pad = ((0,0), (math.floor(pad[0]), math.ceil(pad[0]) - 1), (math.floor(pad[1]), math.ceil(pad[1]) - 1))

            self.ev = self.qe / (self.ep * cp.clip(self.dists, self.dau, None))
            self.ev = fftconvolve(cp.pad(cp.square(cp.abs(phi)) * self.dau**self.n_dim, pad, 'wrap'), self.ev, 'valid', axes=(1, 2))

        elif n_samp > 0:
            
            coord = lambda p: ((p // phi.shape[2]) - (phi.shape[1] // 2), (p % phi.shape[2]) - (phi.shape[2] // 2))
            self.ev = cp.empty([0, *phi.shape[1:]])
            ev = self.qe / (self.ep * cp.clip(self.dists, self.dau, None))

            for el in phi:
                pos = cp.random.choice(el.size, size=n_samp, p=( cp.square(cp.square(cp.abs(el.reshape[-1])) * self.dau**self.n_dim ))).get()
                temp_ev = cp.nanmean(cp.vstack([cp.roll(ev, coord(p), axis=(1, 2)) for p in pos]), axis=0, keepdims=True)
                self.ev = cp.append(self.ev, temp_ev, axis=0)

        else:
            self.ev = cp.zeros([1, *phi.shape[1:]])

    # Calculates the integral over time using the Rk4 method
    def rk4(self, phi, n_samp):
        self.e_field(phi, n_samp)

        ka = cp.zeros(phi.shape, dtype=phi.dtype)

        kn = self.d_dt(phi)
        ka += kn

        kn = self.d_dt(phi+(self.dt * (kn/2)))
        ka += (kn * 2)

        kn = self.d_dt(phi+(self.dt*(kn/2)))
        ka += (kn * 2)

        ka += self.d_dt(phi+(self.dt*kn))

        return self.norm(phi + ((ka/6) * self.dt))
    
    # Calculates the integral over time using the euler method
    def euler(self, phi, n_samp):
        self.e_field(phi, n_samp)
        return self.norm(phi + (self.d_dt(phi) * self.dt))
    
    
    def fft(self, phi, n_samp):
        V = (cp.sum(self.ev, axis=0, keepdims=True) - self.ev) + self.V
        phi *= cp.exp(1j * V * self.dt/2.0)

        phihat = fft.fftn(phi)
        phihat = cp.exp(self.dt * (-1j * fft.ifftshift(cp.square(self.dists))/(2.0 * self.me)))  * phihat
        phi = fft.ifftn(phihat)

        self.e_field(phi, n_samp)
        
        V = (cp.sum(self.ev, axis=0, keepdims=True) - self.ev) + self.V
        phi *= cp.exp(1j * V * self.dt/2.0)

        return self.norm(phi)


    # convienance function for adding a constant potential to the environment via a lambda function
    def add_potential(self, func):
        self.ext_potential_funcs.append(func)

    # Normalize the wave function 
    def norm(self, phi):
        if self.protons.shape[0] > 0:
            p_rad = cp.vstack([cp.roll(self.dists, (p[0]//self.dau, p[1]//self.dau), axis=(1, 2)) for p in self.protons])
            phi = cp.where(cp.min(p_rad, axis=0) <= self.pw, 0, phi)

        return phi / cp.sqrt(cp.nansum(cp.square(cp.abs(phi)), axis=(1,2), keepdims=True) * self.dau**self.n_dim)
    
    
    # convienance function for adding an electron to the environment
    def add_electron(self, p=[0.0, 0.0], pos=[0.0, 0.0], sig=0.1):
        pos = cp.array(pos).reshape([2, 1, 1])
        p = cp.array(p).reshape([2, 1, 1])

        self.electrons.append(lambda x: cp.exp(1j * cp.sum(x * p, axis=0) - (cp.sum(cp.square(x - pos), axis=0) / sig**2)))

    # convienance function for adding a point like proton to the environment 
    def add_proton(self, pos=[0.0, 0.0]):
        self.protons = np.append(self.protons, np.array([pos]), axis=0)

    
    def plot_1D(self, ax1, ax2, phi, ev, V):
        ax1.cla()
        ax2.cla()

        ax1.plot(self.dims[0], np.squeeze(np.real(phi)), label='real')
        ax1.plot(self.dims[0], np.squeeze(np.imag(phi)), label='imag')
        ax1.plot(self.dims[0], np.squeeze(np.abs(phi)), label=r'$|\psi|$')

        ax2.plot(self.dims[0], np.squeeze(ev) * -1, label='Electron Potential')
        ax2.plot(self.dims[0], np.squeeze(V) * -1, label='Proton/External Potential')

        ax1.set_ylim(-3, 3)
        ax2.set_ylim(-75, 75)
        ax2.yaxis.tick_right()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax1.set_title(r'$\psi$')
        ax2.set_title('Potential Energy')

        ax1.set_xlabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax1.set_ylabel(r'$\psi$')

        ax2.set_xlabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax2.set_ylabel('eV')
        ax2.yaxis.set_label_position('right')

        return ax1, ax2
    
    def plot_2D(self, ax1, ax2, phi, ev, V):
        ax1.cla()
        ax2.cla()

        ax1.imshow(np.squeeze(np.square(np.abs(phi))), cmap = 'inferno', vmin=0, vmax=0.5)
        ax1.set_aspect('equal')	

        ax2.imshow(np.squeeze(V + ev), cmap = 'bwr', vmin=-30, vmax=30)	
        ax2.set_aspect('equal')	

        ax1.set_title(r'$|\psi|^2$')
        ax2.set_title('Potential Energy (eV)')

        ax1.set_xlabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax1.set_ylabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax1.set_xticks(ax1.get_xticks()[2:-1], ax1.get_xticks()[2:-1] * self.dau - np.round(np.max(self.dims[0]), 1))
        ax1.set_yticks(ax1.get_yticks()[2:-1], ax1.get_yticks()[2:-1] * self.dau - np.round(np.max(self.dims[1]), 1))

        ax2.set_xlabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax2.set_ylabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax2.set_xticks(ax1.get_xticks(), ax1.get_xticks() * self.dau - np.round(np.max(self.dims[0]), 1))
        ax2.set_yticks(ax1.get_yticks(), ax1.get_yticks() * self.dau - np.round(np.max(self.dims[1]), 1))
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()

        return ax1, ax2
    
    def plot_surface(self, ax, x, y, phi, ev, V):
        ax.cla()

        ax.plot_surface(x, y, Z=np.squeeze(np.real(phi)), cmap='magma', rcount=10, ccount=10, vmin=-0.005, vmax=0.005)
        ax.plot_surface(x, y, Z=np.squeeze(np.imag(phi))-1, cmap='jet', rcount=10, ccount=10, vmin=-1.005, vmax=-0.995)
        ax.plot_surface(x, y, Z=np.squeeze(np.abs(phi))+1, cmap='plasma', rcount=10, ccount=10)
        ax.plot_surface(x, y, Z=np.squeeze((ev + V)/50)-1.5, cmap='gray', alpha=0.6, rcount=10, ccount=10, vmax=4)

        ax.set_zlim(-4, 3)
        ax.set_zticks([])

        ax.set_xlabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax.set_ylabel(r'$a_{0}$ (5.292 x $10^{-11}$ m)')
        ax.set_zlabel(r'$\psi$')
        return ax


    # Simulates the currently specified environment.
    def simulate(
            self, 
            dims, dau, steps=10000, time_step=1,
            save_rate=False, 
            model: SchroModel = None, train_model = False,
            ev_samp_rate='full',
            method='fft',
            real_time=False
        ):
        """
        dims:   An int/float/tuple describing how large each spacial dimention is. If a single value is given then only 1 dimention is simulated
        dau:    step size in atomic units. Step size is applied to spacial and temporal steps
        steps:  number of time steps to simulate
        time_step: The product of time_step and dau will be used as the time step value. If set to 1j, then the simulation will step towards the ground state of the system (theoretically)
        save:   a bool to determine weather to save the simulation steps. The combined electron wave functions and the combined non electron electric potentials will be saved
        checkpoint: how often to save the simulated steps to the save file. After saved to file the held states are cleared to save memory
        sim_name:   The name that will be used as the folder name to hold the saved files
        frame_rate: how often to record a frame for animation. If None then no frames will be availible for animation
        """

        try:
            self.n_dim = len(dims)
        except TypeError:
            dims = (dims, )
            self.n_dim = 1

        if self.n_dim >= 3:
            raise Exception('Error: 3D simulations are not currently supported')

        # set the dimentions of the environment matrix and define coordinates
        self.dau = dau
        self.dt = time_step
        self.dims = [cp.arange(-(dim/2), (dim/2), dau) for dim in dims] + [cp.array([0])]*(2 - len(dims))

        coords = cp.stack(cp.meshgrid(*self.dims, indexing='ij'), axis=0)
        self.dists = cp.sqrt(cp.sum(cp.square(coords), axis=0, keepdims=True))
        self.V = cp.zeros(self.dists.shape, dtype=cp.float16)

        if self.protons.shape[0] > 0:
            p_rad = cp.vstack([cp.roll(self.dists, (p[0]//self.dau, p[1]//self.dau), axis=(1, 2)) for p in self.protons])
            self.V += cp.clip(cp.nansum(self.qp / (self.ep * cp.clip(p_rad, self.dau, None)), axis=0, keepdims=True), None, self.qp/self.dau)

        # create static potential matrix with specified lambda functions
        if len(self.ext_potential_funcs) > 0:
            self.V += cp.sum(cp.stack([v_func(coords) for v_func in self.ext_potential_funcs], axis=0), axis=0).astype(cp.float16)
                

        if method == 'euler':
            method = self.euler
        elif method == 'rk4':
            method = self.rk4
        elif method == 'fft':
            method = self.fft
        elif method == 'model':
            method = model.predict
        else:
            raise Exception(f"SchroSim: '{method}' is not a recognized method")

        # initialize environment with specified electron locations and momentums
        phi = self.norm(cp.stack([el(coords) for el in self.electrons], axis=0))
        self.e_field(phi, ev_samp_rate)

        self.dims = [d.get() for d in self.dims]
        coords = coords.get()
        cp._default_memory_pool.free_all_blocks()

        self.simulation_frames = []
        self.simulation_frames_ev = []

        if save_rate:
            self.simulation_frames.append(cp.sum(phi, axis=0).get())
            self.simulation_frames_ev.append(cp.sum(self.ev, axis=0).get())

        if real_time:
            fig = plt.figure(figsize=(8,5), dpi=200)
            grid = plt.GridSpec(1, 2, wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(grid[0,0])
            ax2 = plt.subplot(grid[0,1])
         

        for i in range(1, steps+1):
            if train_model:
                model.add_step(phi)

                if (i % train_model == 0) or (i == steps):
                    loss = model.train(epochs=5)
                    print(f'Model Trained at Step {i}: loss [{loss}]')
            

            if save_rate and (i % save_rate == 0):
                print(f'Step: {i}')
                self.simulation_frames.append(cp.sum(phi, axis=0).get())
                self.simulation_frames_ev.append(cp.sum(self.ev, axis=0).get())

            if real_time:
                if self.n_dim == 2:
                    self.plot_2D(ax1, ax2, cp.sum(phi, axis=0).get(), cp.sum(self.ev, axis=0).get(), self.V.get())
                elif self.n_dim == 1:
                    self.plot_1D(ax1, ax2, cp.sum(phi, axis=0).get(), cp.sum(self.ev, axis=0).get(), self.V.get())

                plt.pause(0.0001)

            phi = method(phi, ev_samp_rate)

        # free up vram
        self.V = self.V.get()
        self.dists = self.dists.get()
        self.ev = self.ev.get()
        cp._default_memory_pool.free_all_blocks()  

        
    
    # Animates the saved time steps in 1 or 2 dimentions. 
    # 3D visualization isn't currently working.
    def animate(self, interval=1, surface_plot=False, save_file=None, fps=30):

        fig = plt.figure(figsize=(8,5), dpi=200)

        if self.n_dim == 1:
            print('animating 1D plot...')

            grid = plt.GridSpec(1, 2, wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(grid[0,0])
            ax2 = plt.subplot(grid[0,1])

            self.plot_1D(ax1, ax2, self.simulation_frames[0], self.simulation_frames_ev[0], self.V)
            
            def animate(frame):
                self.plot_1D(ax1, ax2, self.simulation_frames[frame], self.simulation_frames_ev[frame], self.V)

        elif self.n_dim == 2:
            if surface_plot:
                print('animating 2D surface plot...')
                ax = fig.add_subplot(111, projection='3d')
                
                x, y = np.meshgrid(self.dims[0], self.dims[1], indexing='ij')
                self.plot_surface(ax, x, y, self.simulation_frames[0], self.simulation_frames_ev[0], self.V)

                def animate(frame):
                    self.plot_surface(ax, x, y, self.simulation_frames[frame], self.simulation_frames_ev[frame], self.V)

            else:
                print('animating 2D plot...')
                grid = plt.GridSpec(1, 2, wspace=0.0, hspace=0.0)
                ax1 = fig.add_subplot(grid[0,0])
                ax2 = fig.add_subplot(grid[0,1])

                self.plot_2D(ax1, ax2, self.simulation_frames[0], self.simulation_frames_ev[0], self.V)

                def animate(frame):
                    self.plot_2D(ax1, ax2, self.simulation_frames[frame], self.simulation_frames_ev[frame], self.V)
        
        elif self.n_dim >= 3:
            raise Exception('Error: There is no animation method for 3d simulations at this time')

        anim = FuncAnimation(fig, animate, frames=int(len(self.simulation_frames)), interval=interval)
        
        if save_file is not None:
            print('saving animation...')
            anim.save(save_file, writer=PillowWriter(fps=fps))
            print(f'animation saved as {save_file}')
        
        plt.show()
        plt.close()
    

    




