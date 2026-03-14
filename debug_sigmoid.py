import numpy as np

MU_TIME  = 11.211313
STD_TIME = 1.129861
MU_CO2   = 0.483506
STD_CO2  = 0.121195

def test_val(t, c):
    z_time = (np.log(t + 1e-5) - MU_TIME) / (STD_TIME + 1e-9)
    z_co2  = (np.log(c + 1e-5) - MU_CO2)  / (STD_CO2  + 1e-9)
    
    score_time = 1 / (1 + np.exp(-z_time))
    score_co2  = 1 / (1 + np.exp(-z_co2))
    
    r_time = 1.0 - score_time
    r_co2  = 1.0 - score_co2
    
    reward = r_time + r_co2
    return reward

print(f"Empty road (0.0): {test_val(0.0, 0.0)}")
print(f"Tiny traffic (10, 0.05): {test_val(10.0, 0.05)}")
