from schrosim import SchroSim
import matplotlib.pyplot as plt
import cupy as cp
from nn_lib import Sequential, Conv2D



box_2d = lambda x: cp.where((cp.abs(x[0]) < 0.5) & (cp.abs(x[1]) < 0.5), 0, 1)
electron = lambda x: (-1 / cp.clip(cp.sqrt(cp.sum(cp.square(x + cp.array([4, 0]).reshape([1, 2, 1, 1])), axis=1)), 1e-2, None)) 
parab = lambda x: cp.sum(-(x*0.5)**2, axis=1, keepdims=True)
slope = lambda x: (cp.sum(x, axis=1, keepdims=True) + 5) * -1

box_mask_2d = lambda x: cp.where((cp.abs(x[0]) < 0.5) & (cp.abs(x[1]) < 0.5), 1, 0)
proton_mask_2d = lambda x: cp.where(cp.sqrt(cp.sum(cp.square(x), axis=0)) < 0.01, 0, 1)

model = Sequential([
    Conv2D(n_kernels=1, kernel_size=3, padding='same', activation='norm', input_size=(500,500,2))
])
model.load_weights(path='./q_models/', name='1e_parab_mod')

sim = SchroSim()
sim.add_potential(parab)
# sim.add_proton(pos=[-5, 0])
# sim.add_proton(pos=[-4, 0])
# sim.add_electron(pos=[-1, 0], p=[5, 0], sig=0.25)
sim.add_electron(pos=[0, 0], p=[0, 0], sig=0.25)

sim.simulate(dims=(5,5), dau=1e-2, steps=5000, time_arrow=0.5,
            frame_rate=50, ev_samp_rate=0, model=model, sim_with_model=True)

anim = sim.animate(interval=1)
plt.show()
plt.close()