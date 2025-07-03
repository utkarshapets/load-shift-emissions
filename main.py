from utils import *
from optimizer import *
import pandas as pd
import numpy as np


TIMESTEPS_PER_DAY = 96  # 24 hours * 4 per hour

SMALL_SHIFTABLE_WINDOW_LENGTH = 1

LOW_SHIFTABLE_WINDOW_LENGTH = 4 # 4 timesteps = 1 hour
MED_SHIFTABLE_WINDOW_LENGTH = 8
HIGH_SHIFTABLE_WINDOW_LENGTH = 16

def get_daily_shift_savings(load_type='residential', write=True):
    """
    Simulates load shift actions for each day of the year and each end use for 4 sets of flexibility assumptions:
        1. Small - 1 timestep (15 min)
        2. Low Flexibility - 4 timesteps (1 hour)
        3. Medium Flexibility - 8 timesteps (2 hours)
        4. High flexibility - 16 timesteps (4 hours)

    - We penalize HVAC, refrigeration, and water heating with a 5% energy penalty of shift.
    - We assume that refrigerator loads can only be shifted forward in time. 
    - We assume that lighting can only be shifted in the scenario with the smallest time flexibility.
    
    Args:
        load (str, optional): Specifies which set of loads, either 'residential' or 'commercial', to simulate. Defaults to 'residential'.
        write (bool, optional): Specifies whether to save the results of the simulation as a csv. Defaults to True.

    Returns:
        (df, df, df, df): 4-tuple where each element is a pandas dataframe containing results from the simulation for a given set of assumptions (small, low, medium, and high respectively). Each row of the dataframe represents a day and each column corresponds to an end use. Each entry specifies the lbs. of CO2 savings from implementing the optimal load shift action on a specific day. 
    """
    assert load_type == 'residential' or load_type == 'commercial', f"Load must be either either 'residential' or 'commercial', got '{load_type}'."

    assumptions = read_assumptions(f'data/{load_type}_assumptions.json')
    data = pd.read_csv(f'data/{load_type}_data.csv')

    daily_savings_small = pd.DataFrame()
    daily_savings_low = pd.DataFrame()
    daily_savings_med = pd.DataFrame()
    daily_savings_high = pd.DataFrame()

    for cat in assumptions.keys():
        for end_use in assumptions[cat].keys():
            load_savings_small_values = []
            load_savings_low_values = []
            load_savings_med_values = []
            load_savings_high_values = []

            for date in np.unique(data["date"]):
                emissions = np.array(data[data["date"] == date]["MOER"])
                load = np.array(data[data["date"] == date][end_use])

                time_series_differential_small = get_time_series_differential(emissions, SMALL_SHIFTABLE_WINDOW_LENGTH)
                time_series_differential_low = get_time_series_differential(emissions, LOW_SHIFTABLE_WINDOW_LENGTH)
                time_series_differential_med = get_time_series_differential(emissions, MED_SHIFTABLE_WINDOW_LENGTH)
                time_series_differential_high = get_time_series_differential(emissions, HIGH_SHIFTABLE_WINDOW_LENGTH)

                if cat == 'HVAC' or cat == 'Hot Water':
                    new_load_small = load_shift_penalty(emissions, load, time_series_differential_small, assumptions[cat][end_use][0], SMALL_SHIFTABLE_WINDOW_LENGTH)
                    new_load_low = load_shift_penalty(emissions, load, time_series_differential_low, assumptions[cat][end_use][1], LOW_SHIFTABLE_WINDOW_LENGTH)
                    new_load_med = load_shift_penalty(emissions, load, time_series_differential_med, assumptions[cat][end_use][2], MED_SHIFTABLE_WINDOW_LENGTH)
                    new_load_high = load_shift_penalty(emissions, load, time_series_differential_high, assumptions[cat][end_use][3], HIGH_SHIFTABLE_WINDOW_LENGTH)
                elif cat == 'Refrigeration':
                    new_load_small = load_shift_refrigerator(emissions, load, time_series_differential_low, assumptions[cat][end_use][0], SMALL_SHIFTABLE_WINDOW_LENGTH)
                    new_load_low = load_shift_refrigerator(emissions, load, time_series_differential_low, assumptions[cat][end_use][1], LOW_SHIFTABLE_WINDOW_LENGTH)
                    new_load_med = load_shift_refrigerator(emissions, load, time_series_differential_med, assumptions[cat][end_use][2], MED_SHIFTABLE_WINDOW_LENGTH)
                    new_load_high = load_shift_refrigerator(emissions, load, time_series_differential_high, assumptions[cat][end_use][3], HIGH_SHIFTABLE_WINDOW_LENGTH)
                else:
                    new_load_small = load_shift(emissions, load, time_series_differential_low, assumptions[cat][end_use][0], SMALL_SHIFTABLE_WINDOW_LENGTH)
                    new_load_low = load_shift(emissions, load, time_series_differential_low, assumptions[cat][end_use][1], LOW_SHIFTABLE_WINDOW_LENGTH)
                    new_load_med = load_shift(emissions, load, time_series_differential_med, assumptions[cat][end_use][2], MED_SHIFTABLE_WINDOW_LENGTH)
                    new_load_high = load_shift(emissions, load, time_series_differential_high, assumptions[cat][end_use][3], HIGH_SHIFTABLE_WINDOW_LENGTH)

                load_savings_small = sum(emissions * (load - new_load_small)) / 1000  # Division by 1000 to cancel units. MOER is in (lb. CO2 / MWh) and load is in kWH
                load_savings_low = sum(emissions * (load - new_load_low)) / 1000  
                load_savings_med = sum(emissions * (load - new_load_med)) / 1000
                load_savings_high = sum(emissions * (load - new_load_high)) / 1000

                load_savings_small_values.append(load_savings_small)
                load_savings_low_values.append(load_savings_low)
                load_savings_med_values.append(load_savings_med)
                load_savings_high_values.append(load_savings_high)

            daily_savings_small[end_use] = load_savings_small_values
            if cat != 'Lighting': # We assume that lighting is not shiftable except in the case of a small shift
                daily_savings_low[end_use] = load_savings_low_values
                daily_savings_med[end_use] = load_savings_med_values
                daily_savings_high[end_use] = load_savings_high_values

    if write:
        daily_savings_small.to_csv(f'out/{load_type}/shift_daily_savings_small.csv')
        daily_savings_low.to_csv(f'out/{load_type}/shift_daily_savings_low.csv')
        daily_savings_med.to_csv(f'out/{load_type}/shift_daily_savings_med.csv')
        daily_savings_high.to_csv(f'out/{load_type}/shift_daily_savings_high.csv')

    return daily_savings_small, daily_savings_low, daily_savings_med, daily_savings_high


def get_daily_shed_savings(load_type='residential', write=True):

    """
    Simulates load shed actions for each day of the year and each end use for 3 sets of flexibility assumptions:
        1. Low Flexibility - 4 timesteps (1 hour)
        2. Medium Flexibility - 8 timesteps (2 hours)
        3. High flexibility - 16 timesteps (4 hours)
          
    Args:
        load (str, optional): Specifies which set of loads, either 'residential' or 'commercial', to simulate. Defaults to 'residential'.
        write (bool, optional): Specifies whether to save the results of the simulation as a csv. Defaults to True.

    Returns:
        (df, df, df): 3-tuple where each element is a pandas dataframe containing results from the simulation for a given set of assumptions (low, medium, and high respectively). Each row of the dataframe represents a day and each column corresponds to an end use. Each entry specifies the lbs. of CO2 savings from implementing the optimal load shift action on a specific day. 
    """
    assert load_type == 'residential' or load_type == 'commercial', f"Load must be either either 'residential' or 'commercial', got {load_type}."

    assumptions = read_assumptions(f'data/{load_type}_assumptions.json')
    data = pd.read_csv(f'data/{load_type}_data.csv')

    daily_savings_low = pd.DataFrame()
    daily_savings_med = pd.DataFrame()
    daily_savings_high = pd.DataFrame()

    for cat in assumptions.keys():
        for end_use in assumptions[cat].keys():
            load_savings_low_values = []
            load_savings_med_values = []
            load_savings_high_values = []

            for date in np.unique(data["date"]):
                emissions = np.array(data[data["date"] == date]["MOER"])
                load = np.array(data[data["date"] == date][end_use])

                new_load_low = load_shed(emissions, load, assumptions[cat][end_use][1], LOW_SHIFTABLE_WINDOW_LENGTH)
                new_load_med = load_shed(emissions, load, assumptions[cat][end_use][2], MED_SHIFTABLE_WINDOW_LENGTH)
                new_load_high = load_shed(emissions, load, assumptions[cat][end_use][3], HIGH_SHIFTABLE_WINDOW_LENGTH)
                
                load_savings_low = sum(emissions * (load - new_load_low)) / 1000  # Division by 1000 to cancel units. MOER is in (lb. CO2 / MWh) and load is in kWH
                load_savings_med = sum(emissions * (load - new_load_med)) / 1000
                load_savings_high = sum(emissions * (load - new_load_high)) / 1000

                load_savings_low_values.append(load_savings_low)
                load_savings_med_values.append(load_savings_med)
                load_savings_high_values.append(load_savings_high)

            daily_savings_low[end_use] = load_savings_low_values
            daily_savings_med[end_use] = load_savings_med_values
            daily_savings_high[end_use] = load_savings_high_values


    if write:
        daily_savings_low.to_csv(f'out/{load_type}/shed_daily_savings_low.csv')
        daily_savings_med.to_csv(f'out/{load_type}/shed_daily_savings_med.csv')
        daily_savings_high.to_csv(f'out/{load_type}/shed_daily_savings_high.csv')

    return daily_savings_low, daily_savings_med, daily_savings_high


def main():
    get_daily_shift_savings(load_type='commercial')
    get_daily_shift_savings(load_type='residential')
    get_daily_shed_savings(load_type='commercial')
    get_daily_shed_savings(load_type='residential')

if __name__ == "__main__":
    main()
