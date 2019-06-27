#!/usr/bin/env python
# coding: utf-8


import os.path
import numpy as np
import numpy.random as random
import scipy.io
import scipy.interpolate
import scipy.sparse
import scipy.sparse.linalg
import numpy.linalg
import matplotlib.pyplot as plt
import load_data
import model
import pickle

x, dx, target_layers, target_ages, a, u = load_data.load()
target_layer = target_layers[-1, :]
timestep_target = 17  # This is our target year spacing
times = np.hstack([np.linspace(0, target_ages['Age'][0], int(np.round(target_ages['Age'][0] / timestep_target)) + 1)[:-1]] + [np.linspace(target_ages['Age'][i], target_ages['Age'][i + 1], int(np.round((target_ages['Age'][i + 1] - target_ages['Age'][i]) / timestep_target)) + 1)[:-1] for i in range(len(target_ages) - 1)] + [np.array(target_ages['Age'][-1])])
compare_indices = np.array([i for i in range(len(times)) if np.any(times[i] == target_ages['Age'])])
timesteps = np.diff(times)
num_steps = len(timesteps)
accumulation_scale = 1.5 * np.ones(num_steps)
velocity_scale = 1.05 * np.ones(num_steps)

z = model.layer_depth(x, accumulation_scale, velocity_scale,
                      a, u, timesteps)

seed = 1
random_state = random.RandomState(seed=seed)

r = 20
two_step = np.hstack((timesteps, timesteps))
Id = scipy.sparse.diags([np.ones(num_steps * 2)], [0])
D = scipy.sparse.diags([1. / two_step, -1. / two_step], [0, 1], shape=(len(two_step) - 1, len(two_step)))
L = r**2 * D.T * D * np.mean(timesteps)
D2 = scipy.sparse.diags([1. / timesteps, -1. / timesteps], [0, 1], shape=(len(timesteps) - 1, len(timesteps)))
L2 = r**2 * D2.T * D2 * np.mean(timesteps)
M = Id + L

λs, vs = numpy.linalg.eigh(M.toarray())


def generate_momentum(random_state, σ):
    ζ = random_state.normal(size=num_steps * 2)
    return sum(((ζ[k] * np.sqrt(λs[k] / (num_steps * 2)) * σ) * vs[:, k] for k in range(2 * num_steps)))


def potential_energy(θ_a, θ_v):
    z_θ = model.layer_depth(x, θ_a, θ_v, a, u, timesteps)
    return np.sum(model.msm(x, z_θ[compare_indices, :], target_layers) + 0.5 * np.dot(θ_a, L2 * θ_a) + 0.5 * np.dot(θ_v, L2 * θ_v))


def kinetic_energy(ϕ):
    return 0.5 * np.dot(ϕ.flatten(), scipy.sparse.linalg.spsolve(M, ϕ))


def force(θ_a, θ_v):
    z_θ = model.layer_depth(x, θ_a, θ_v, a, u, timesteps)
    return -model.derivative_scales(x, θ_a, θ_v, a, u, z_θ, target_layers, compare_indices, timesteps) - np.hstack((L2 * θ_a, L2 * θ_v))


def velocity(ϕ):
    return scipy.sparse.linalg.spsolve(M, ϕ)


def hamiltonian_update(δτ, θ_a, θ_v, ϕ):
    vel = velocity(ϕ)
    θ_a_τ = θ_a + 0.5 * δτ * vel[:len(θ_a)]
    θ_v_τ = θ_v + 0.5 * δτ * vel[len(θ_a):]

    ϕ_τ = ϕ + δτ * (force(θ_a_τ, θ_v_τ))
    vel = velocity(ϕ_τ)
    θ_a_τ += 0.5 * δτ * vel[:len(θ_a)]
    θ_v_τ += 0.5 * δτ * vel[len(θ_a):]
    return θ_a_τ, θ_v_τ, ϕ_τ


θ_a = accumulation_scale.copy()
θ_v = velocity_scale.copy()

ϕ = generate_momentum(random_state, 1)

# Solve the system using a fictitious timestep of 1/32 for 2/3 of a fictitious time unit.
# I had to hand-tune both of these parameters to get something sensible.


δτ = 1.0 / 32
num_hamiltonian_steps = int(3.0 / 4 / δτ)

num_samples = 3
θs = np.zeros((num_samples, num_steps))
θs[0, :] = accumulation_scale.copy()
θsv = np.zeros((num_samples, num_steps))
θsv[0, :] = velocity_scale.copy()
for sample in range(1, num_samples):
    print('Sample', sample, '...', end='')
    ϕ = generate_momentum(random_state, 1)
    θ_a = θs[sample - 1, :].copy()
    θ_v = θsv[sample - 1, :].copy()

    for k in range(num_hamiltonian_steps):
        θ_a, θ_v, ϕ = hamiltonian_update(δτ, θ_a, θ_v, ϕ)
    print(kinetic_energy(ϕ), potential_energy(θ_a, θ_v) / (x[-1] * len(compare_indices)))
    θs[sample, :] = θ_a.copy()
    θsv[sample, :] = θ_v.copy()

pickle_fn = 'packaged_out_{:d}samp_{:d}seed.pickle'
pickle.dump((θ_a, θ_v), open(pickle_fn.format(num_samples, seed), 'wb'))

cm1 = plt.get_cmap('autumn')
cm2 = plt.get_cmap('winter')
fig, ax = plt.subplots(figsize=(12, 8))
for sample in range(num_samples):
    ax.plot(times[1:], θs[sample, :], color=cm1((sample + 1) / num_samples))
    ax.plot(times[1:], θsv[sample, :], color=cm2((sample + 1) / num_samples))

ax.set_xlabel('t (years)')
fig.savefig('lots_of_lines.png')
