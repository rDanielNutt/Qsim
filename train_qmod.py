from schrosim import SchroSim
import cupy as cp
from numpy.random import randint
from nn_lib import Sequential, Conv2D
import matplotlib.pyplot as plt

model = Sequential([
    Conv2D(n_kernels=1, kernel_size=3, padding='same', activation='norm', input_size=(500,500,2)),
    ], lr=1e-7, loss='wave', dtype=cp.complex128)
model.layers[0].weights *= 1j

parab = lambda x: cp.sum(-(x)**2, axis=1, keepdims=True)

# for i in range(5):
#     sim = SchroSim()
#     sim.add_electron(pos=randint(-10,10,2)/10, sig=0.25)
#     sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=10)
#     model.save(path='./q_models/', name='1e_mod')
#     print(f'\nCompleted 1e sim {i}\n')

# for i in range(5):
#     sim = SchroSim()
#     sim.add_electron(p=randint(-5, 5, 2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=10, ev_samp_rate=32)
#     model.save(path='./q_models/', name='1e_mo_mod')
#     print(f'\nCompleted 1e mo sim {i}\n')


for i in range(10):
    sim = SchroSim()
    sim.add_electron(p=randint(-5,6,2), pos=randint(-3,4,2)/10, sig=0.25)
    sim.add_potential(parab)
    sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=20, ev_samp_rate=0)
    model.save(path='./q_models/', name='1e_parab_mod')
    print(f'\nCompleted ev training sim {i}\n')
    

# for i in range(5):
#     sim = SchroSim()
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.add_proton(pos=randint(-20, 20, 2)/10)
#     sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=10, ev_samp_rate=32)
#     model.save(path='./q_models/', name='1e_1p_mod')
#     print(f'\nCompleted 1e 1p sim {i}\n')

# for i in range(5):
#     sim = SchroSim()
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=10, ev_samp_rate=32)
#     model.save(path='./q_models/', name='2e_mod')
#     print(f'\nCompleted 2e sim {i}\n')

# for i in range(5):
#     sim = SchroSim()
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.add_proton(pos=randint(-20, 20, 2)/10)
#     sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=10, ev_samp_rate=32)
#     model.save(path='./q_models/', name='2e_1p_mod')
#     print(f'\nCompleted 2e 1p sim {i}\n')

# for i in range(5):
#     sim = SchroSim()
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.add_proton(pos=randint(-20, 20, 2)/10)
#     sim.add_proton(pos=randint(-20, 20, 2)/10)
#     sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=10, ev_samp_rate=32)
#     model.save(path='./q_models/', name='1e_2p_mod')
#     print(f'\nCompleted 1e 2p sim {i}\n')

# for i in range(5):
#     sim = SchroSim()
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.add_electron(p=randint(-5,5,2), pos=randint(-10,10,2)/10, sig=0.25)
#     sim.add_proton(pos=randint(-20, 20, 2)/10)
#     sim.add_proton(pos=randint(-20, 20, 2)/10)
#     sim.simulate(dims=(5,5), dau=1e-2, steps=1000, model=model, train_model=10, ev_samp_rate=32)
#     model.save(path='./q_models/', name='2e_2p_mod')
#     print(f'\nCompleted 2e 2p sim {i}\n')

# print('\nDone!\n')
