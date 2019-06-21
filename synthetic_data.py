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
    
    targ_variability = 0.5
    accumulation = random_state.rand(1000)
    accumulation = filtfilt(b, a, accumulation)
    accumulation = accumulation / np.std(accumulation) * targ_variability
    
    # lets have trends and variability
    # also need to match the modern at present, otherwise we are hosed
    accumulation_scale = random_state.rand(times.shape[0] * 2) * 2.0
    accumulation_scale = filtfilt(b2, a2, accumulation_scale)[times.shape[0] // 2: times.shape[0] // 2 + times.shape[0]]
    accumulation_scale = accumulation_scale + np.linspace(-0.1, 0.1, times.shape[0])
    accumulation_scale += (1. - accumulation_scale[-1])

    velocity_scale = random_state.rand(times.shape[0] * 2) * 2.0
    velocity_scale = filtfilt(b2, a2, velocity_scale)[times.shape[0] // 2: times.shape[0] // 2 + times.shape[0]]
    velocity_scale = velocity_scale + np.linspace(0.1, -0.1, times.shape[0])
    velocity_scale += (1. - velocity_scale[-1])

    velocity = np.linspace(10, 5, accumulation.shape[0])

    x = np.linspace(0, 50000, accumulation.shape[0])
    dx = np.diff(x)

    years_layers = [50, 150, 250, 350, 1000, 1500, 2500, 5000]
    uncertainty = np.atleast_2d(np.array([0.2, 0.2, 0.2, 0.3, 20, 20, 20, 20])).transpose()
    uncertainty = np.ones_like(uncertainty)

    years_out = [np.where(np.abs(year - times) < 1)[0] for year in years_layers]

    z = model.layer_depth(x, accumulation_scale, velocity_scale, accumulation, velocity, timesteps)  # , mask=True)

    plt.figure(figsize=(12, 8))
    plt.gca().invert_yaxis()
    for i, year in enumerate(years_layers):
        ind = np.where(year == times)[0]
        ln = plt.plot(x, z[ind, :].flatten(), label=str(year))
        plt.fill_between(x, z[ind, :].flatten() - uncertainty[i], z[ind, :].flatten() + uncertainty[i], color=ln[0].get_color(), alpha=0.5)
    plt.legend(loc='best')

    plt.figure()
    plt.plot(times, accumulation_scale, label='acc')
    plt.plot(times, velocity_scale, label='vel')
    plt.legend(loc='best')

    return x, dx, z[years_out, :], years_layers, accumulation, velocity, accumulation_scale, velocity_scale, uncertainty


if __name__ == '__main__':
    load()
    plt.show()
