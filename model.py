import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def vertical_velocity(a, z, kink_height=0.2, h=2700.0):
    """Return the vertical velocities at the given depths from the Dansgaard-
    Johnsen 1969 model

    Parameters
    ----------
    a : np.ndarray(n x 1)
        Accumulation rates
    z : np.ndarray(n x 1)
        Depths within the ice sheet at which to evaluate the velocity
    kink_height: float
        Parameter in the DJ model
    h : float
        Total ice sheet thickness

    Returns
    -------
    w : np.ndarray(n x 1)
    """
    return (2 * (1 - z / h) - kink_height) / (2 - kink_height) * a


def vertical_strain_rate(a, kink_height=0.2, h=2700.0):
    return -2 * a / h / (2 - kink_height)


def layer_depth(x, a_scale, u_scale, a, u, time, num_steps):
    """Compute the depth of an advected surface ice layer

    Parameters
    ----------
    x : np.ndarray(num_gridpoints x 1)
        Numerical grid
    a_scale : np.ndarray(num_steps x 1)
        Factor to scale the accumulation at each time step
    u_scale : np.ndarray(num_steps x 1)
        Factor to scale the horizontal velocity at each time step
    a : np.ndarray(num_gridpoints x 1)
        Accumulation rates along the transect
    u : np.ndarray(num_gridpoints x 1)
        Horizontal velocities along the transect
    time : float
        Final time of the simulation in years
    num_steps : int
        Number of timesteps to use

    Returns
    -------
    z : np.ndarray(num_gridpoints x 1)
        The depth of the advected surface layer
    """
    dx = np.diff(x)
    dt = time / num_steps

    I = scipy.sparse.eye(len(x))
    D = scipy.sparse.diags([np.hstack((-1/dx, [0])), 1/dx], [0, 1])

    z = np.zeros((num_steps + 1, len(x)))
    for step in range(num_steps):
        L = I - dt * u_scale[step] * scipy.sparse.diags([u], [0]) * D
        f = z[step, :] + dt * vertical_velocity(a_scale[step] * a, z[step, :])
        z[step + 1, :] = scipy.sparse.linalg.spsolve(L, f)

    return z


def adjoint_solve(x, a_scale, u_scale, a, u, z, λ_T, time, num_steps):
    """Solve the advection equation backwards in time from `λ_T` to compute the
    derivative of some functional w.r.t. the parameters"""
    dx = np.diff(x)
    dt = time / num_steps

    I = scipy.sparse.eye(len(x))
    D = scipy.sparse.diags([np.hstack(([0], -1/dx)), 1/dx], [0, 1])

    λ = np.zeros((num_steps + 1, len(x)))
    λ[num_steps, :] = λ_T
    for step in range(num_steps - 1, -1, -1):
        L = I - dt * u_scale[step] * D.T * scipy.sparse.diags([u], [0])
        f = (1 + dt * vertical_strain_rate(a_scale[step] * a)) * λ[step + 1, :]
        λ[step, :] = scipy.sparse.linalg.spsolve(L, f)

    return λ


def mean_square_misfit(x, z, zo):
    mse = 0.0
    for n in range(len(x) - 1):
        dx = x[n + 1] - x[n]
        mse += 0.5 * ((z[n] + z[n+1])/2 - (zo[n] + zo[n+1])/2)**2 * dx

    return mse


def adjoint_sensitivity_ascale(x, z, λ, a):
    num_steps = z.shape[0] - 1
    num_points = z.shape[1]
    dJ_da = np.zeros(num_steps)

    for k in range(num_steps):
        w = vertical_velocity(a[k], z[k + 1, :])
        for n in range(num_points - 1):
            dx = x[n + 1] - x[n]
            dJ_da[k] -= (λ[k, n + 1] * w[n + 1] + λ[k, n] * w[n]) / 2 * dx

    return dJ_da


def derivative_ascale(x, a_scale, u_scale, a, u,
                      z, target_layer, time, num_steps):
    λ_final = target_layer - z[-1, :]
    λ = adjoint_solve(x, a_scale, u_scale, a, u, z, λ_final, time, num_steps)
    return adjoint_sensitivity_ascale(x, z, λ, a)
