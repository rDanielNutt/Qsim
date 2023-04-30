from schrosim import SchroSim
import cupy as cp
from numpy.random import randint

parab = lambda x: -cp.sum((x)**2, axis=1, keepdims=True)

for i in range(14, 25):
    sim = SchroSim()
    sim.add_electron(p=randint(-20, 20, 2), pos=randint(-20,20,2)/10, sig=randint(5, 20)/100)
    sim.simulate(dims=(5,5), dau=1e-2, steps=5000, 
                 save=True, check_point=1000, path='./simulations/', sim_name=f'1e_sim{i}')


for i in range(25):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_potential(parab)
    sim.simulate(dims=(5,5), dau=1e-2, steps=5000, 
                 save=True, check_point=1000, path='./simulations/', sim_name=f'parab_1e_sim{i}')
    

for i in range(50):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=5000, 
                 save=True, check_point=1000, path='./simulations/', sim_name=f'1p_1e_sim{i}')


for i in range(50):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.simulate(dims=(5,5), dau=1e-2, steps=5000, 
                 save=True, check_point=1000, path='./simulations/', sim_name=f'2e_sim{i}')
    

for i in range(50):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=5000, 
                 save=True, check_point=1000, path='./simulations/', sim_name=f'1p_2e_sim{i}')


for i in range(50):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=5000, 
                 save=True, check_point=1000, path='./simulations/', sim_name=f'2p_1e_sim{i}')
    

for i in range(50):
    sim = SchroSim()
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_electron(p=randint(-20,20,2), pos=randint(-20,20,2)/10, sig=randint(5,20)/100)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.add_proton(pos=randint(-20, 20, 2)/10)
    sim.simulate(dims=(5,5), dau=1e-2, steps=5000, 
                 save=True, check_point=1000, path='./simulations/', sim_name=f'2p_2e_sim{i}')

print('\nDone!\n')
