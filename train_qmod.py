from schrosim import SchroSim
import cupy as cp
from numpy.random import randint
from nn_lib import Sequential, Conv2D

model = Sequential([
    Conv2D(n_kernels=4, kernel_size=5, padding='same', activation='tanh', input_size=(500,500,4)),
    ], lr=1e-6)

parab = lambda x: -cp.sum((x)**2, axis=1, keepdims=True)

for i in range(5):
    sim = SchroSim()
    sim.add_electron(p=randint(-20, 20, 2), pos=randint(-20,20,2)/10, sig=randint(5, 20)/100)
    sim.simulate(dims=(5,5), dau=1e-2, steps=250, model=model, train_model=15)
    model.save(path='./q_models/', name='1e_mod')
    print(f'\nCompleted 1e sim {i}\n')


for i in range(5):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_potential(parab)
    sim.simulate(dims=(5,5), dau=1e-2, steps=250, model=model, train_model=10)
    model.save(path='./q_models/', name='1e_parab_mod')
    print(f'\nCompleted parab 1e sim {i}\n')
    

for i in range(5):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=250, model=model, train_model=10)
    model.save(path='./q_models/', name='1e_1p_mod')
    print(f'\nCompleted 1e 1p sim {i}\n')

for i in range(5):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.simulate(dims=(5,5), dau=1e-2, steps=250, model=model, train_model=10)
    model.save(path='./q_models/', name='2e_mod')
    print(f'\nCompleted 2e sim {i}\n')

for i in range(5):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=250, model=model, train_model=10)
    model.save(path='./q_models/', name='2e_1p_mod')
    print(f'\nCompleted 2e 1p sim {i}\n')

for i in range(5):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=250, model=model, train_model=10)
    model.save(path='./q_models/', name='1e_2p_mod')
    print(f'\nCompleted 1e 2p sim {i}\n')

for i in range(5):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=250, model=model, train_model=10)
    model.save(path='./q_models/', name='2e_2p_mod')
    print(f'\nCompleted 2e 2p sim {i}\n')

print('\nDone!\n')
