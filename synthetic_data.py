#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.random as random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.interpolate
from scipy.signal import filtfilt, butter
from scipy.io import loadmat, savemat

import model

res = 5001


def create(noisy_scaling=False, delta_a=0.2, delta_v=0.2):
    print('Uh oh, we have to do the slow bit')
    random_state = random.RandomState(100)
    b, a = butter(1, 0.02)
    b2, a2 = butter(1, 0.002)
    times = np.linspace(0, 5000, res)
    timesteps = np.diff(times)

    targ_value = 0.1
    targ_variability = 0.02
    accumulation = random_state.rand(1001)
    # careful here because edge effects are really annoying
    accumulation = filtfilt(b, a, np.hstack((accumulation, np.flip(accumulation, axis=-1))))[10: len(accumulation) + 10]
    accumulation = (accumulation - np.mean(accumulation)) / np.std(accumulation) * targ_variability + targ_value

    # lets have trends and variability
    # also need to match the modern at present, otherwise we are hosed
    if noisy_scaling:
        accumulation_scale = random_state.rand(times.shape[0] * 2) * 2.0
        accumulation_scale = filtfilt(b2, a2, accumulation_scale)[times.shape[0] // 2: times.shape[0] // 2 + times.shape[0]]
        accumulation_scale = accumulation_scale + np.linspace(-0.1, 0.1, times.shape[0])
        accumulation_scale += (1. - accumulation_scale[-1])

        velocity_scale = random_state.rand(times.shape[0] * 2) * 2.0
        velocity_scale = filtfilt(b2, a2, velocity_scale)[times.shape[0] // 2: times.shape[0] // 2 + times.shape[0]]
        velocity_scale = velocity_scale + np.linspace(0.1, -0.1, times.shape[0])
        velocity_scale += (1. - velocity_scale[-1])
    else:
        accumulation_scale = np.linspace(0., delta_a, times.shape[0])
        accumulation_scale += (1. - accumulation_scale[-1])

        velocity_scale = np.linspace(delta_v, 0., times.shape[0])
        velocity_scale += (1. - velocity_scale[-1])

    velocity = np.linspace(10, 5, accumulation.shape[0])

    x = np.linspace(0, 50000, accumulation.shape[0])
    dx = np.diff(x)

    years_layers = [150, 200, 245, 283, 315, 350, 1233, 1666, 1875, 2500, 5000]
    uncertainty = np.atleast_2d(np.array([1.0 if year <= 1000. else 10.0 for year in years_layers])).transpose()

    years_out = [np.argmin(np.abs(year - times)) for year in years_layers]

    z = model.layer_depth(x, accumulation_scale, velocity_scale, accumulation, velocity, timesteps, printout=True)  # , mask=True)
    target_layers = z[years_out]

    if not noisy_scaling:
        for i in range(target_layers.shape[0]):
            # Let's assume that the noise is correlated
            noise = random_state.randn(target_layers.shape[1]) * uncertainty[i]
            target_layers[i, :] = target_layers[i, :] + filtfilt(b2, a2, np.hstack((noise, np.flip(noise, axis=-1))))[10:10 + len(noise)]

    mat = {'x': x, 'dx': dx, 'z': z, 'years_out': years_out, 'years_layers': years_layers, 'accumulation': accumulation, 'velocity': velocity, 'accumulation_scale': accumulation_scale, 'velocity_scale': velocity_scale, 'uncertainty': uncertainty, 'target_layers': target_layers}
    savemat('cached_da{:06.3f}_dv{:06.3f}.mat'.format(delta_a, delta_v), mat)


def compare(x, dx, zyo, years_layers, accumulation, velocity, accumulation_scale, velocity_scale, uncertainty, z, delta_a=0.2, delta_v=0.2):
    times = np.linspace(0, 5000, res)
    plt.figure(figsize=(12, 8))
    plt.gca().invert_yaxis()
    for i, year in enumerate(years_layers):
        ind = np.where(year == times)[0]
        ln = plt.plot(x, zyo[i, :].flatten(), label=str(year))
        plt.fill_between(x, zyo[i, :].flatten() - uncertainty[i], zyo[i, :].flatten() + uncertainty[i], color=ln[0].get_color(), alpha=0.5)
    plt.legend(loc='best')

    plt.figure()
    plt.plot(times, accumulation_scale, label='acc')
    plt.plot(times, velocity_scale, label='vel')
    plt.legend(loc='best')

    times_fives = np.linspace(0, 5000, 1001)
    accumulation_scale_fives = [np.mean(accumulation_scale[:3])] + [np.mean(accumulation_scale[5 * i - 2: 5 * i + 3]) for i in range(1, len(times_fives) - 1)] + [np.mean(accumulation_scale[-3:])]
    velocity_scale_fives = [np.mean(velocity_scale[:3])] + [np.mean(velocity_scale[5 * i - 2: 5 * i + 3]) for i in range(1, len(times_fives) - 1)] + [np.mean(velocity_scale[-3:])]

    times_tens = np.linspace(0, 5000, 501)
    accumulation_scale_tens = [np.mean(accumulation_scale[:5])] + [np.mean(accumulation_scale[10 * i - 5: 10 * i + 5]) for i in range(1, len(times_tens) - 1)] + [np.mean(accumulation_scale[-5:])]
    velocity_scale_tens = [np.mean(velocity_scale[:5])] + [np.mean(velocity_scale[10 * i - 5: 10 * i + 5]) for i in range(1, len(times_tens) - 1)] + [np.mean(velocity_scale[-5:])]

    times_twentyfives = np.linspace(0, 5000, 201)
    accumulation_scale_twentyfives = [np.mean(accumulation_scale[:13])] + [np.mean(accumulation_scale[25 * i - 12: 25 * i + 13]) for i in range(1, len(times_twentyfives) - 1)] + [np.mean(accumulation_scale[-13:])]
    velocity_scale_twentyfives = [np.mean(velocity_scale[:13])] + [np.mean(velocity_scale[25 * i - 12: 25 * i + 13]) for i in range(1, len(times_twentyfives) - 1)] + [np.mean(velocity_scale[-13:])]

    times_fifties = np.linspace(0, 5000, 101)
    accumulation_scale_fifties = [np.mean(accumulation_scale[:25])] + [np.mean(accumulation_scale[50 * i - 25: 50 * i + 25]) for i in range(1, len(times_fifties) - 1)] + [np.mean(accumulation_scale[-25:])]
    velocity_scale_fifties = [np.mean(velocity_scale[:25])] + [np.mean(velocity_scale[50 * i - 25: 50 * i + 25]) for i in range(1, len(times_fifties) - 1)] + [np.mean(velocity_scale[-25:])]

    plt.figure()
    plt.plot(times, accumulation_scale)
    for time, scale in zip(times_fives, accumulation_scale_fives):
        plt.plot([time - 2.5, time + 2.5], [scale, scale], color='C1')
    for time, scale in zip(times_tens, accumulation_scale_tens):
        plt.plot([time - 5, time + 5], [scale, scale], color='C2')
    for time, scale in zip(times_twentyfives, accumulation_scale_twentyfives):
        plt.plot([time - 12, time + 13], [scale, scale], color='C3')
    for time, scale in zip(times_fifties, accumulation_scale_fifties):
        plt.plot([time - 25, time + 25], [scale, scale], color='C4')

    gs = gridspec.GridSpec(4, 1, hspace=0.03, wspace=0, left=0.075, right=0.99, top=0.99, bottom=0, height_ratios=[1, 1, 1, 0.5])
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)
    ax3 = plt.subplot(gs[2, 0], sharex=ax1)

    layer_age = 150.

    ax1.invert_yaxis()
    # ax2.invert_yaxis()
    # use rougher timesteps
    print('Calc 5s')
    z_fives = model.layer_depth(x, accumulation_scale_fives, velocity_scale_fives, accumulation, velocity, np.diff(times_fives))
    print('Calc 10s')
    z_tens = model.layer_depth(x, accumulation_scale_tens, velocity_scale_tens, accumulation, velocity, np.diff(times_tens))
    print('Calc 25s')
    z_twentyfives = model.layer_depth(x, accumulation_scale_twentyfives, velocity_scale_twentyfives, accumulation, velocity, np.diff(times_twentyfives))
    print('Calc 50s')
    z_fifties = model.layer_depth(x, accumulation_scale_fifties, velocity_scale_fifties, accumulation, velocity, np.diff(times_fifties))
    # use inferred accumulation
    z2fn = 'z2_da{:06.3f}_dv{:06.3f}'.format(delta_a, delta_v)
    if not os.path.exists(z2fn):
        print('Calc 1s...this is slow')
        ind = np.where(layer_age == times)[0][0]
        z2 = model.layer_depth(x, accumulation_scale[-ind:], velocity_scale[-ind:], z[ind, :] / 150., velocity, np.diff(times)[-ind:], printout=True)
        x_unoffset = x + layer_age * velocity / 2.
        accinterp = scipy.interpolate.interp1d(x_unoffset, z[ind, :].flatten() / 150., fill_value='extrapolate')
        z2better = model.layer_depth(x, accumulation_scale[-ind:], velocity_scale[-ind:], accinterp(x), velocity, np.diff(times)[-ind:], printout=True)
        savemat(z2fn, {'z2': z2, 'z2better': z2better})

    else:
        mat = loadmat(z2fn)
        z2 = mat['z2']
        z2better = mat['z2better']
    ind = np.where(layer_age == times)[0]
    ind_fives = np.where(layer_age == times_fives)[0]
    ind_tens = np.where(layer_age == times_tens)[0]
    ind_twentyfives = np.where(layer_age == times_twentyfives)[0]
    ind_fifties = np.where(layer_age == times_fifties)[0]

    markevery = 20

    ax1.plot(x / 1000., accumulation, label='Target', color='C0', marker='.', markevery=markevery)
    ax1.plot(x / 1000., z[ind, :].flatten() / layer_age, label='Inferred $a$, no advection', color='C1', marker='s', markevery=markevery)
    x_unoffset = x + layer_age * velocity / 2.
    accinterp = scipy.interpolate.interp1d(x_unoffset, z[ind, :].flatten() / 150., fill_value='extrapolate')
    ax1.plot(x / 1000., accinterp(x), label='Radar-inferred $a$', color='C2', marker='x', markevery=markevery)

    ax2.plot(x / 1000., z[ind, :].flatten(), label='Target', color='C0', marker='.', markevery=markevery)
    # ax2.plot(x / 1000., z2[-1, :].flatten(), label='Inferred $a$, no advection', color='C1', marker='s', markevery=markevery)
    ax2.plot(x / 1000., z2better[-1, :], label='Radar-inferred $a$', color='C2', marker='x', markevery=markevery)

    # just do this for easy legend creation (it is not shown)
    ax3.plot([-100, -200], [0, 0], color='C1', label='Inferred $a$, no advection', marker='s')
    ax3.plot([-100, -200], [0, 0], color='C2', label='Inferred $a$', marker='x')

    ax3.plot(x / 1000., 100 * (z[ind, :].flatten() - z_fives[ind_fives, :].flatten()) / z[ind, :].flatten(), label='5-y timesteps', marker='d', color='C5', markevery=markevery)
    ax3.plot(x / 1000., 100 * (z[ind, :].flatten() - z_tens[ind_tens, :].flatten()) / z[ind, :].flatten(), label='10-y timesteps', marker='+', color='C3', markevery=markevery)
    ax3.plot(x / 1000., 100 * (z[ind, :].flatten() - z_twentyfives[ind_twentyfives, :].flatten()) / z[ind, :].flatten(), label='25-y timesteps', marker='^', color='C4', markevery=markevery)
    ax3.plot(x / 1000., 100 * (z[ind, :].flatten() - z_fifties[ind_fifties, :].flatten()) / z[ind, :].flatten(), label='25-y timesteps', marker='>', color='C6', markevery=markevery)

    ax3.legend(loc='upper left', bbox_to_anchor=(0.05, -0.25), frameon=False, ncol=3)

    ax1.set_xlim(0, 50)
    ax3.set_xlabel('Distance (km)')

    ax1.set_ylabel('Depth (m)')
    ax2.set_ylabel('Depth (m)')
    ax3.set_ylabel(r'Error (\%)')

    ax1.get_xaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])

    fig.savefig('/Users/dlilien/cloud/Dropbox/Apps/Overleaf/HMC/depth_accuracy.png', dpi=400)


def load(delta_a=0.2, delta_v=0.2):
    if delta_a is not None:
        fn = 'cached_da{:06.3f}_dv{:06.3f}.mat'.format(delta_a, delta_v)
    else:
        fn = 'cached.mat'

    if not os.path.exists(fn):
        create(delta_a=delta_a, delta_v=delta_v)
    mat = loadmat(fn)

    return mat['x'].flatten(), mat['dx'].flatten(), mat['target_layers'], mat['years_layers'].flatten(), mat['accumulation'].flatten(), mat['velocity'].flatten(), mat['accumulation_scale'].flatten(), mat['velocity_scale'].flatten(), mat['uncertainty'], mat['z']


if __name__ == '__main__':
    x, dx, zyo, years_layers, accumulation, velocity, accumulation_scale, velocity_scale, uncertainty, zfull = load()
    compare(x, dx, zyo, years_layers, accumulation, velocity, accumulation_scale, velocity_scale, uncertainty, zfull)
    plt.show()
