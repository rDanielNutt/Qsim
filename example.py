from schrosim import SchroSim
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np



box_2d = lambda x: cp.where((cp.abs(x[0]) < 0.5) & (cp.abs(x[1]) < 0.5), 0, 1)
proton_2d = lambda x: (-1 / cp.clip(cp.sqrt(cp.sum(cp.square(x), axis=0)), 1e-2)) 
parab = lambda x: cp.sum((3*x)**2, axis=0)

box_mask_2d = lambda x: cp.where((cp.abs(x[0]) < 0.5) & (cp.abs(x[1]) < 0.5), 1, 0)
proton_mask_2d = lambda x: cp.where(cp.sqrt(cp.sum(cp.square(x), axis=0)) < 0.01, 0, 1)


sim = SchroSim()
# sim.add_potential(parab)
sim.add_electron(pos=[0.25, 0.25], p=[0, 0], sig=0.25)

anim = sim.simulate(dims=(1, 1), dau=1e-2, steps=100, imag_time=False,
                    save_rate=None, check_point=1000, path=None, sim_name='test',
                    frame_rate=1, batch_calc=False)

plt.show()
plt.close()