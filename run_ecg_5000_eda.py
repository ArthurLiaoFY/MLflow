# %%
from configparser import ConfigParser

import numpy as np


# %%

config = ConfigParser()
config.read("projects/ecg_5000/ecg_5000.ini")
# %%
data = np.loadtxt(
    './data/ECG5000/ECG5000_TRAIN.txt'
)
# %%
import matplotlib.pyplot as plt
plt.plot(data[0,:],'o-')
plt.show()
# %%
