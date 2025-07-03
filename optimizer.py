import cvxpy as cp
import numpy as np
import utils


def load_shift(emissions, load, time_series_differential, proportion_shiftable, shiftable_window_length):
    """
    Compute the optimal load curve for a load shift event. 

    Args:
        emissions (list or numpy array): time series (length T) of MOER readings
        load (list or numpy array): time series (length T) of baseline energy consumption for a specific end use
        time_series_differential (list or numpy array): max(MOER) - min(MOER) within the shiftable window
        proportion_shiftable (float): the proportion of load (at peak MOER time) that can be shifted
        shiftable_window_length (int): the amount of time by which it can be shifted (in number of timesteps)

    Returns:
        new_load: new load curve as time series (length T)
    """
    assert len(emissions) == len(load) == len(time_series_differential), "time series must all be the same length"
    assert 0 <= proportion_shiftable <= 1, "proportion_shiftable must be between 0 and 1"

    T = len(load)
    
    peak_MOER_timestep = np.argmax(time_series_differential)
    peak_MOER_load = load[peak_MOER_timestep]
    
    shiftable_load = proportion_shiftable * peak_MOER_load

    new_load_lower = np.array(load) - \
                        shiftable_load * np.array([1 if i == peak_MOER_timestep else 0 for i in range(T)])
    
    load_delta_times = [1 if ((i <= peak_MOER_timestep + shiftable_window_length) \
                        and (i >= peak_MOER_timestep - shiftable_window_length)) \
                  else 0 for i in range(T)]
    load_delta = np.array(load_delta_times) * shiftable_load
    new_load_upper = np.array(load) + np.array(load_delta)

    new_load = cp.Variable(T)
    
    constraints = [new_load_lower <= new_load, new_load <= new_load_upper, cp.sum(new_load) == cp.sum(load)]
    prob = cp.Problem(cp.Minimize(emissions@new_load), constraints)
    prob.solve()
    return new_load.value

def load_shift_penalty(emissions, load, time_series_differential, proportion_shiftable, shiftable_window_length):
    """
    Compute the optimal load curve for a load shift event. Imposes a 5% energy penalty per hour of shift. 

    Args:
        emissions (list or numpy array): time series (length T) of MOER readings
        load (list or numpy array): time series (length T) of baseline energy consumption for a specific end use
        time_series_differential (list of numpy array): max(MOER) - min(MOER) within the shiftable window
        proportion_shiftable (float): the proportion of load (at peak MOER time) that can be shifted
        shiftable_window_length (int): the amount of time by which it can be shifted (in number of timesteps)
    
    Returns:
        new_load: new load curve as time series (length T)
    """
    assert len(emissions) == len(load) == len(time_series_differential), "time series must all be the same length"
    assert 0 <= proportion_shiftable <= 1, "proportion_shiftable must be between 0 and 1"

    T = len(load)
    
    peak_MOER_timestep = np.argmax(time_series_differential)
    peak_MOER_load = load[peak_MOER_timestep]
    
    shiftable_load = proportion_shiftable * peak_MOER_load

    new_load_lower = np.array(load) - shiftable_load * np.array([1 if i == peak_MOER_timestep else 0 for i in range(T)])

    load_delta_times = [1 + 0.05*(i - peak_MOER_timestep)/4 if ((i <= peak_MOER_timestep + shiftable_window_length) \
                          and (i >= peak_MOER_timestep)) \
                          else(
                              1 + 0.05*(peak_MOER_timestep - i)/4 if ((i <= peak_MOER_timestep) \
                          and (i >= peak_MOER_timestep - shiftable_window_length)) \
                          else 0) for i in range(T)
                        ]

    load_delta = np.array(load_delta_times) * shiftable_load
    new_load_upper = np.array(load) + np.array(load_delta)

    new_load = cp.Variable(T)

    shift_penalty_times = [1 + 0.05*(i - peak_MOER_timestep)/4 if i > peak_MOER_timestep else (1 + 0.05*(peak_MOER_timestep - i)/4) for i in range(T)]
    penalty = [1/shift_penalty_times[i] for i in range(T)]    

    A = [np.identity(len(load)), - np.identity(len(load))]
    A = np.reshape(A, (2*len(load), len(load)))

    load = np.array(load)
    constraints = [A @ new_load <= np.reshape([new_load_upper, -new_load_lower], (2*len(load),)), penalty @ new_load == penalty @ load]
    prob = cp.Problem(cp.Minimize(np.array(emissions)@new_load), constraints)
    prob.solve()
    return new_load.value

def load_shift_refrigerator(emissions, load, time_series_differential, proportion_shiftable, shiftable_window_length):
    """
    Compute the optimal load curve for a load shift event. 
    Imposes a 5% energy penalty per hour of shift. 
    Imposes the additional constrant that the refrigerator load can only be shifted forward in time.

    Args:
        emissions (list or numpy array): time series (length T) of MOER readings
        load (list or numpy array): time series (length T) of baseline energy consumption for a specific end use
        time_series_differential (list of numpy array): max(MOER) - min(MOER) within the shiftable window
        proportion_shiftable (float): the proportion of load (at peak MOER time) that can be shifted
        shiftable_window_length (int): the amount of time by which it can be shifted (in number of timesteps)
    
    Returns:
        new_load: new load curve as time series (length T)
    """
    assert len(emissions) == len(load) == len(time_series_differential), "time series must all be the same length"
    assert 0 <= proportion_shiftable <= 1, "proportion_shiftable must be between 0 and 1"
    
    T = len(load)
    time_series_differential = utils.get_time_series_differential(emissions, shiftable_window_length)
    
    peak_MOER_timestep = np.argmax(time_series_differential)
    peak_MOER_load = load[peak_MOER_timestep]
    
    shiftable_load = proportion_shiftable * peak_MOER_load

    new_load_lower = np.array(load) - shiftable_load * np.array([1 if i == peak_MOER_timestep else 0 for i in range(T)])

    load_delta_times = [1 + 0.05*(i - peak_MOER_timestep)/4 if ((i <= peak_MOER_timestep + shiftable_window_length) \
                          and (i >= peak_MOER_timestep)) \
                          else 0 for i in range(T)
                        ]

    load_delta = np.array(load_delta_times) * shiftable_load
    new_load_upper = np.array(load) + np.array(load_delta)

    new_load = cp.Variable(T)

    shift_penalty_times = [1 + 0.05*(i - peak_MOER_timestep)/4 if i > peak_MOER_timestep else (1 + 0.05*(peak_MOER_timestep - i)/4) for i in range(T)]
    penalty = [1/shift_penalty_times[i] for i in range(T)]    

    A = [np.identity(len(load)), - np.identity(len(load))]
    A = np.reshape(A, (2*len(load), len(load)))

    load = np.array(load)
    constraints = [A @ new_load <= np.reshape([new_load_upper, -new_load_lower], (2*len(load),)), penalty @ new_load == penalty @ load]
    prob = cp.Problem(cp.Minimize(emissions@new_load), constraints)
    prob.solve()
    return new_load.value

def load_shed(emissions, load, proportion_shedable, shedable_window_length):
    """
    Use this function to get the optimal load curve for a load shed event.

    Input:
        emissions (list or numpy array): time series of MOER readings
        load (list or numpy array): time series of baseline energy consumption for a specific end use
        proportion_shedable (float): the proportion of load that can be shedded per timestep
        shiftable_window_length (int): the size of the time window in which load shed can be applied
    
    Returns:
        new_load: new load curve as time series (length T)
    """
    assert len(emissions) == len(load), "time series must all be the same length"
    assert 0 <= proportion_shedable <= 1, "proportion_shedable must be between 0 and 1"

    rolling_sum = [sum(emissions[i : i+shedable_window_length]) for i in range(len(emissions) - shedable_window_length)]
    start_of_shed = np.argmax(rolling_sum)
    new_load = list(load[:start_of_shed]) + list((1 - proportion_shedable) * np.array(load[start_of_shed : start_of_shed + shedable_window_length])) + list(load[start_of_shed + shedable_window_length:])
    return np.array(new_load)

