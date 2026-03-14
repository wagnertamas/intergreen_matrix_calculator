import numpy as np

# A konverzációból ismert konstansok
MU_TIME = 11.21
STD_TIME = 1.05
MU_CO2 = 0.48
STD_CO2 = 1.33
traffic_duration = 3600

# Tegyük fel, hogy egy autó 20 másodpercet tölt átlagosan dugóban:
avg_travel_time = 20.0  
total_co2 = 10000.0

projected_tts = avg_travel_time * traffic_duration
projected_co2 = total_co2 * traffic_duration

print(f"Projected TTS: {projected_tts}")

# Z-score számítás (np.log)
z_time = (np.log(projected_tts + 1e-5) - MU_TIME) / (STD_TIME + 1e-9)
print(f"Z_time logaritmikusan: {z_time}")
score_time = 1 / (1 + np.exp(-z_time))
print(f"Score time (szigmoid): {score_time}")

# Mivel az eddigi reward számítás 1 - score:
reward = 1 - score_time
print(f"Final Reward Time component: {reward}")
