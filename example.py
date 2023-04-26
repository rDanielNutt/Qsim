from schrosim import SchroSim
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np



box_2d = lambda x: cp.where((cp.abs(x[0]) < 0.5) & (cp.abs(x[1]) < 0.5), 0, 1)
proton_2d = lambda x: (-1 / cp.clip(cp.sqrt(cp.sum(cp.square(x), axis=0)), 1e-2, None)) 
parab = lambda x: cp.sum((x)**2, axis=1, keepdims=True)

box_mask_2d = lambda x: cp.where((cp.abs(x[0]) < 0.5) & (cp.abs(x[1]) < 0.5), 1, 0)
proton_mask_2d = lambda x: cp.where(cp.sqrt(cp.sum(cp.square(x), axis=0)) < 0.01, 0, 1)


sim = SchroSim()
# sim.add_potential(proton_2d)
# sim.add_proton(pos=[1, 1])
sim.add_proton(pos=[0, 0])
sim.add_electron(pos=[1], p=[0], sig=0.1)
sim.add_electron(pos=[-1], p=[0], sig=0.1)

anim = sim.simulate(dims=(5,5), dau=1e-2, steps=5000,
                    save_rate=None, check_point=1000, path=None, sim_name='test',
                    frame_rate=50)

plt.show()
plt.close()