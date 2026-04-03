import grid_model as gm
import numpy as np
import matplotlib.pyplot as plt

profile = gm.LOAD_PROFILE * 2.0
avg = profile.mean()

a = 0.5
s = -np.cos((np.linspace(0, 1, 24) -1/24)*2*np.pi)
s = np.clip(s - 0.1, 0, .6) * avg
print(s)
p2 = np.roll(profile, 2)
plt.plot(profile)
plt.plot(p2)
plt.plot(profile - p2)
plt.plot(p2 + profile)
plt.axhline(avg)
plt.show()
