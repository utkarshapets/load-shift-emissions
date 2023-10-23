import numpy as np
import json

def get_time_series_differential(emissions, shiftable_window_length):
    '''
    Compute MOER differential time series. 

    Args:
        emissions (list or numpy array): time series (length T) of MOER readings
        shiftable_window_length (int): the amount of time by which it can be shifted (in number of timesteps)
    
    Returns:
        moer_differential (numpy array): max(MOER) - min(MOER) within the shiftable window for each timestep
    '''
    T = len(emissions)
    moer_differential = np.zeros(T)

    emissions = np.array(emissions)
    for i in range(T):
        window = emissions[max(0, i - shiftable_window_length) : min(T, i + shiftable_window_length)]
        moer_differential[i] = emissions[i] - min(window)
    return moer_differential

def read_assumptions(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)