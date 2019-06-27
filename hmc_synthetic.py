#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np
import numpy.random as random
import scipy.io
import scipy.interpolate
import scipy.sparse
import scipy.sparse.linalg
import numpy.linalg
import matplotlib.pyplot as plt
import synthetic_data
import model
import pickle


# Load some synthetic data produced in synthetic_data.py
def main(num_samples, da, dv):
    x, dx, target_layers, target_ages, a, u, a_scale_data, u_scale_data, uncertainty, z = synthetic_data.load(delta_a=da, delta_v=dv)
    timestep_target = 20  # This is our target year spacing
    target_ages = np.array(target_ages)
    uncertainty = np.ones_like(uncertainty)

    times = np.flip(-np.hstack([np.linspace(0, target_ages[0], int(np.round(target_ages[0] / timestep_target)) + 1)[:-1]] + [np.linspace(target_ages[i], target_ages[i + 1], int(np.round((target_ages[i + 1] - target_ages[i]) / timestep_target)) + 1)[:-1] for i in range(len(target_ages) - 1)] + [np.array(target_ages[-1])]))
    compare_indices = np.flip(len(times) - 1 - np.array([i for i in range(len(times)) if np.any(times[i] == -target_ages)]))
    timesteps = np.diff(times)
    num_steps = len(timesteps)

    accumulation_scale = np.ones(num_steps)
    velocity_scale = np.ones(num_steps)

    z = model.layer_depth(x, accumulation_scale, velocity_scale, a, u, timesteps)

    # We are going to do this to see how we are doing at every layer
    # It should give us a good idea of whether dJ_da should be positive
    plt.figure(figsize=(12, 8))
    for i, index in enumerate(compare_indices):
        ln = plt.plot(x, z[index], linestyle='dotted')
        plt.plot(x, target_layers[i, :].flatten(), color=ln[0].get_color(), linestyle='solid', label=str(i))
    plt.legend(loc='upper left', bbox_to_anchor=(0.1, -0.1), ncol=6)
    plt.gca().invert_yaxis()

    random_state = random.RandomState()

    r = 40

    Id_single = scipy.sparse.diags([np.ones(num_steps)], [0])
    diag_single = np.ones(num_steps)
    D_single = scipy.sparse.diags([diag_single, -diag_single], [0, 1], shape=(len(timesteps) - 1, len(timesteps))) / np.mean(timesteps)
    L2 = r**2 * D_single.T * D_single * np.mean(timesteps)
    M_single = Id_single + L2

    M = scipy.sparse.csr_matrix(np.hstack((np.vstack((M_single.toarray(), np.zeros_like(M_single.toarray()))), np.vstack((np.zeros_like(M_single.toarray()), M_single.toarray())))))
    assert np.all(M[:num_steps, :num_steps].toarray() == M_single.toarray())
    assert np.all(M[num_steps:, num_steps:].toarray() == M_single.toarray())
    # M = M_single

    λs, vs = numpy.linalg.eigh(M.toarray())

    def generate_momentum(random_state, σ):
        ζ = random_state.normal(size=num_steps * 2)
        return sum(((ζ[k] * np.sqrt(λs[k] / (num_steps * 2)) * σ) * vs[:, k] for k in range(2 * num_steps)))
        # ζ = random_state.normal(size=num_steps)
        # return sum(((ζ[k] * np.sqrt(λs[k] / num_steps) * σ) * vs[:, k] for k in range(num_steps)))

    def potential_energy(θ_a, θ_v):
        z_θ = model.layer_depth(x, θ_a, θ_v, a, u, timesteps)
        return np.nansum(model.msm(x, z_θ[compare_indices, :], target_layers, uncertainty=uncertainty) + 0.5 * np.dot(θ_a, L2 * θ_a) + 0.5 * np.dot(θ_v, L2 * θ_v))

    def kinetic_energy(ϕ):
        return 0.5 * np.dot(ϕ.flatten(), scipy.sparse.linalg.spsolve(M, ϕ))

    def hamiltonian(ϕ, θ_a, θ_v):
        return potential_energy(θ_a, θ_v) + kinetic_energy(ϕ)

    def force(θ_a, θ_v):
        z_θ = model.layer_depth(x, θ_a, θ_v, a, u, timesteps)
        return -model.derivative_scales(x, θ_a, θ_v, a, u, z_θ, target_layers, compare_indices, timesteps, uncertainty=uncertainty) - np.hstack((L2 * θ_a, L2 * θ_v))
        # return -model.derivative_ascale(x, θ_a, θ_v, a, u, z_θ, target_layers, compare_indices, timesteps, uncertainty=uncertainty) - L2 * θ_a

    def velocity(ϕ):
        return scipy.sparse.linalg.spsolve(M, ϕ)

    def accept_or_reject(KE_0, PE_0, KE_1, PE_1, rescale=1.0):
        """Return true if you should accept, false if you should reject. Reject if we have NaNs in the new Hamiltonian calculation"""
        H_0 = KE_0 + PE_0
        H_1 = KE_1 + PE_1
        if np.isnan(H_1):
            return False
        else:
            return (np.random.rand(1) < min(1., np.exp((H_0 - H_1) / rescale)))[0]

    def hamiltonian_update(δτ, θ_a, θ_v, ϕ):
        vel = velocity(ϕ)
        θ_a_τ = θ_a + 0.5 * δτ * vel[:len(θ_a)]
        θ_v_τ = θ_v
        θ_v_τ = θ_v + 0.5 * δτ * vel[len(θ_a):]

        ϕ_τ = ϕ + δτ * (force(θ_a_τ, θ_v_τ))
        vel = velocity(ϕ_τ)
        θ_a_τ += 0.5 * δτ * vel[:len(θ_a)]
        θ_v_τ += 0.5 * δτ * vel[len(θ_a):]
        return θ_a_τ, θ_v_τ, ϕ_τ

    def integrate_phase_space(δτ, num_hamiltonian_steps, θ_a, θ_v, ϕ):
        print('Going along phase space')
        for k in range(num_hamiltonian_steps):
            print('..{:d}'.format(k), end='', flush=True)
            θ_a, θ_v, ϕ = hamiltonian_update(δτ, θ_a, θ_v, ϕ)
        return θ_a, θ_v, ϕ

    θ_a = accumulation_scale.copy()
    θ_v = velocity_scale.copy()
    ϕ = generate_momentum(random_state, 1)

    δτ = 0.02
    num_hamiltonian_steps = 10

    print('Running one step with loud printing so you know if this works')
    for k in range(num_hamiltonian_steps):
        θ_a, θ_v, ϕ = hamiltonian_update(δτ, θ_a, θ_v, ϕ)
        print('{:d}: {:E} {:E}'.format(k, kinetic_energy(ϕ), potential_energy(θ_a, θ_v)))

    θs = np.zeros((num_samples, num_steps))
    θs[0, :] = θ_a.copy()
    θsv = np.zeros((num_samples, num_steps))
    θsv[0, :] = θ_v.copy()
    KEs = np.zeros((num_samples,))
    PEs = np.zeros((num_samples,))
    PEs[0] = potential_energy(θ_a, θ_v)
    KEs[0] = kinetic_energy(ϕ)

    print('Beginning actual HMC')
    sample = 1
    num_accepted = 0
    num_rejected = 0
    while sample < num_samples:
        print(sample, ':   ', end='')
        ϕ_0 = generate_momentum(random_state, 1)
        ϕ = ϕ_0.copy()
        θ_a = θs[sample - 1, :].copy()
        θ_v = θsv[sample - 1, :].copy()
        θ_a, θ_v, ϕ = integrate_phase_space(δτ, num_hamiltonian_steps, θ_a, θ_v, ϕ)
        KEs[sample] = kinetic_energy(ϕ)
        PEs[sample] = potential_energy(θ_a, θ_v)
        print('')
        print('{:E} {:E}'.format(KEs[sample], PEs[sample]))
        print('')
        if accept_or_reject(KEs[sample - 1], PEs[sample - 1], KEs[sample], PEs[sample]):
            θs[sample, :] = θ_a.copy()
            θsv[sample, :] = θ_v.copy()
            sample += 1
            num_accepted += 1
        else:
            num_rejected += 1
            print('Redoing iteration: H_0={:E} while H_1={:E}'.format(KEs[sample - 1] + PEs[sample - 1], KEs[sample] + PEs[sample]))
    print('Rejected {:d} samples of {:d}'.format(num_rejected, num_accepted + num_rejected))
    print('Acceptance percentage {:f}'.format(num_accepted / (num_accepted + num_rejected)))

    pickle.dump([θs, θsv, KEs, PEs], open('hmc_output_da{:06.3f}_dv{:06.3f}_n{:d}.pickle'.format(da, dv, num_samples), 'wb'))

    min_PE = np.min(PEs)
    pre_samples = np.min(np.where(PEs < 1.1 * min_PE))
    cm1 = plt.get_cmap('autumn')
    cm2 = plt.get_cmap('winter')
    fig, ax = plt.subplots()
    for sample in range(pre_samples, num_samples):
        ax.plot(times[1:], θs[sample, :], color=cm1((sample + 1) / (num_samples - pre_samples)))
        ax.plot(times[1:], np.flip(θsv[sample, :]), color=cm2((sample + 1) / (num_samples - pre_samples)))
    ax.plot(np.linspace(-5000, 0, len(u_scale_data)), a_scale_data, color='k')
    ax.plot(np.linspace(-5000, 0, len(u_scale_data)), np.flip(u_scale_data), color='purple')
    ax.set_xlabel('t (years)')

    cm1 = plt.get_cmap('autumn')
    cm2 = plt.get_cmap('winter')

    PE_normed = (PEs - np.min(PEs[pre_samples:])) / (np.max(PEs[pre_samples:]) - np.min(PEs[pre_samples:]))

    fig, ax = plt.subplots()
    for sample in range(pre_samples, num_samples):
        ax.plot(times[1:], θs[sample, :], color=cm1(PE_normed[sample]))
        ax.plot(times[1:], np.flip(θsv[sample, :]), color=cm2(PE_normed[sample]))

    ax.plot(np.linspace(-5000, 0, len(u_scale_data)), a_scale_data, color='k')
    ax.plot(np.linspace(-5000, 0, len(u_scale_data)), u_scale_data, color='purple')

    ax.set_xlabel('t (years)')

    std = np.std(θs[pre_samples:, :], axis=0)
    plt.figure()
    plt.plot(times[1:], std)
    colormap = plt.get_cmap('binary')

    fig, ax = plt.subplots(figsize=(12, 8))
    for sample in range(pre_samples, num_samples):
        θ = θs[sample, :]
        θv = θsv[sample, :]
        z_θ = model.layer_depth(x, θ, θv, a, u, timesteps)
        for i, index in enumerate(compare_indices):
            ax.plot(x / 1000., z_θ[index, :], color=colormap((sample + 1) / (num_samples - pre_samples)))
    for i, index in enumerate(compare_indices):
        ax.plot(x / 1000., target_layers[i], color='r')

    ax.invert_yaxis()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = sys.argv[1]
    else:
        n = 3
    if len(sys.argv) > 2:
        da = sys.argv[2]
        dv = sys.argv[3]
    else:
        da = 0.2
        dv = 0.2

    main(int(n), float(da), float(dv))
