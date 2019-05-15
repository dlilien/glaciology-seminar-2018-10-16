import numpy as np
import numpy.random as random

import matplotlib.pyplot as plt

import scipy.interpolate
from scipy.signal import filtfilt, butter

import model


def load():
    random_state = random.RandomState(100)
    b, a = butter(1, 0.15)
    b2, a2 = butter(1, 0.01)
    times = np.linspace(0, 5000, 1001)
    timesteps = np.diff(times)

    accumulation = random_state.rand(1000)
    accumulation = filtfilt(b, a, accumulation)
    
    # lets have trends and variability
    accumulation_scale = random_state.rand(times.shape[0] * 2) * 2
    accumulation_scale = filtfilt(b2, a2, accumulation_scale)[times.shape[0] // 2: times.shape[0] // 2 + times.shape[0]]
    accumulation_scale = accumulation_scale + np.linspace(-0.1, 0.1, times.shape[0])

    velocity_scale = random_state.rand(times.shape[0] * 2) * 2
    velocity_scale = filtfilt(b2, a2, velocity_scale)[times.shape[0] // 2: times.shape[0] // 2 + times.shape[0]]
    velocity_scale = velocity_scale + np.linspace(0.1, -0.1, times.shape[0])

    velocity = np.linspace(10, 5, accumulation.shape[0])

    x = np.linspace(0, 50000, accumulation.shape[0])
    dx = np.diff(x)

    years_layers = [50, 150, 250, 350, 1000, 1500, 2500, 5000]

    years_out = [np.where(np.abs(year - times) < 1)[0] for year in years_layers]

    z = model.layer_depth(x, accumulation_scale, velocity_scale, accumulation, velocity, timesteps, mask=True)

    plt.figure()
    plt.gca().invert_yaxis()
    for year in years_layers:
        ind = np.where(year == times)[0]
        plt.plot(x, z[ind, :].flatten(), label=str(year))
    plt.legend(loc='best')

    plt.figure()
    plt.plot(times, accumulation_scale, label='acc')
    plt.plot(times, velocity_scale, label='vel')
    plt.legend(loc='best')

    return x, dx, z[years_out, :], years_layers, accumulation, velocity, accumulation_scale, velocity_scale


if __name__ == '__main__':
    load()
