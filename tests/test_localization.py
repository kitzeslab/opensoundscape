import numpy as np
from opensoundscape import localization

def close(x,y,tol):
    return (x<y+tol) and (x > y-tol)

def test_cal_speed_of_sound():
    assert(close(localization.calc_speed_of_sound(20),343,1))
    
def test_lorentz_ip_3():
    assert(localization.lorentz_ip([1,1,2],[1,1,2])==-2)
    
def test_lorentz_ip_4():
    assert(localization.lorentz_ip([1,1,1,2],[1,1,1,2])==-1)
    
def test_lorentz_ip_self():
    assert(localization.lorentz_ip([1,1,1,2])==-1)
    
def test_travel_time():
    source = [0,0,0]
    receiver = [0,0,1]
    assert(close(localization.travel_time(source, receiver, 343),1/343, .0001))
    
def test_localize_2d():
    reciever_positions = [[0,0],[0,20],[20,20],[20,0]]
    arrival_times = [1,1,1,1]
    temperature=20.0,  # celcius
    invert_alg="gps",  # options: 'lstsq', 'gps'
    center=True,  # True for original Sound Finder behavior
    pseudo=True,  # False for original Sound Finder
    estimate = localization.localize(
                        reciever_positions,
                        arrival_times,
                        temperature=20.0,  # celcius
                        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
                        center=center,  # True for original Sound Finder behavior
                        pseudo=pseudo,  # False for original Sound Finder
    )
    assert(close(np.linalg.norm(np.array(estimate[0:2])-np.array([10,10])),0,.01))
    
def test_localize_3d():
    reciever_positions = [[0,0,0],[0,20,1],[20,20,-1],[20,0,.1]]
    arrival_times = [1,1,1,1]
    temperature=20.0,  # celcius
    invert_alg="gps",  # options: 'lstsq', 'gps'
    center=True,  # True for original Sound Finder behavior
    pseudo=True,  # False for original Sound Finder
    estimate = localization.localize(
                        reciever_positions,
                        arrival_times,
                        temperature=20.0,  # celcius
                        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
                        center=center,  # True for original Sound Finder behavior
                        pseudo=pseudo,  # False for original Sound Finder
    )
    assert(close(np.linalg.norm(np.array(estimate[0:3])-np.array([10,10,0])),0,.1))
    
    
def test_localize_lstsq():
    reciever_positions = [[0,0,0],[0,20,1],[20,20,-1],[20,0,.1]]
    arrival_times = [1,1,1,1]
    temperature=20.0,  # celcius
    invert_alg="lstsq",  # options: 'lstsq', 'gps'
    center=True,  # True for original Sound Finder behavior
    pseudo=True,  # False for original Sound Finder
    estimate = localization.localize(
                        reciever_positions,
                        arrival_times,
                        temperature=20.0,  # celcius
                        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
                        center=center,  # True for original Sound Finder behavior
                        pseudo=pseudo,  # False for original Sound Finder
    )
    assert(close(np.linalg.norm(np.array(estimate[0:3])-np.array([10,10,0])),0,.1))
    
    
def test_localize_nocenter():
    reciever_positions = [[100,0,0],[100,20,1],[120,20,-1],[120,0,.1]]
    arrival_times = [1,1,1,1]
    temperature=20.0,  # celcius
    invert_alg="lstsq",  # options: 'lstsq', 'gps'
    center=False,  # True for original Sound Finder behavior
    pseudo=True,  # False for original Sound Finder
    estimate = localization.localize(
                        reciever_positions,
                        arrival_times,
                        temperature=20.0,  # celcius
                        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
                        center=center,  # True for original Sound Finder behavior
                        pseudo=pseudo,  # False for original Sound Finder
    )
    assert(close(np.linalg.norm(np.array(estimate[0:3])-np.array([110,10,0])),0,.1))
    
def test_localize_nopseudo():
    reciever_positions = [[0,0,0],[0,20,1],[20,20,-1],[20,0,.1]]
    arrival_times = [1,1,1,1]
    temperature=20.0,  # celcius
    invert_alg="lstsq",  # options: 'lstsq', 'gps'
    center=True,  # True for original Sound Finder behavior
    pseudo=False,  # False for original Sound Finder
    estimate = localization.localize(
                        reciever_positions,
                        arrival_times,
                        temperature=20.0,  # celcius
                        invert_alg=invert_alg,  # options: 'lstsq', 'gps'
                        center=center,  # True for original Sound Finder behavior
                        pseudo=pseudo,  # False for original Sound Finder
    )
    assert(close(np.linalg.norm(np.array(estimate[0:3])-np.array([10,10,0])),0,.1))