import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def vertical_velocity(a, z, kink_height=0.2, h=2500.0):
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

    return (2. * (-z / h + 1.) - kink_height) / (2. - kink_height) * a


def vertical_strain_rate(a, kink_height=0.2, h=2500.0):
    """Return the vertical strain rate, which by DJ is independent of depth above the kink.

    .. math::
        $\dot{\epsilon_{zz}} = \frac{\partial v_z}{\partial z} =  \frac{\partial}{\partial z} \left(a\frac{2(1-z/h) - k_h}{2 - k_h}\right)=-\frac{2a}{(2-k_h)h}$
    """
    return -2 * a / h / (2 - kink_height)


def layer_depth(x, a_scale, u_scale, a, u, timesteps, mask=False, printout=False, h=2500.):
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
    z : np.ndarray(num_gridpoints x num_tsteps)
        The depth of the advected surface layer
    """
    num_steps = len(timesteps)
    dx = np.diff(x)

    I = scipy.sparse.eye(len(x))
    D = scipy.sparse.diags([np.hstack((-1. / dx, [0])), 1. / dx], [0, 1])
    UD = scipy.sparse.diags([u], [0]) * D

    z = np.zeros((num_steps + 1, len(x)))
    if printout:
        print('Running through {:d} timesteps'.format(num_steps))
    for step in range(num_steps):
        if printout:
            if step % (num_steps // 10) == 0:
                print(str(step), end='', flush=True)
            else:
                print('.', end='', flush=True)
        L = I - timesteps[step] * u_scale[step] * UD
        f = z[:step + 1, :] + timesteps[step] * vertical_velocity(a_scale[step] * a, z[:step + 1, :], h=h)
        z[1:step + 2, :] = scipy.sparse.linalg.spsolve(L, f.T).T
    if printout:
        print('done')

    if mask:
        z[valid_data(x, u_scale, u, timesteps)] = np.nan
    return z


def valid_data(x, u_scale, u, timesteps):
    """Make a mask of where our output from layer_depth will actually be valid"""
    num_steps = len(timesteps)
    dx = np.diff(x)

    mask = np.zeros((num_steps + 1, len(x)), dtype=bool)
    right_edge = x[-1]
    for step in range(num_steps):
        ind_right_edge0 = np.where(x < right_edge)[0][-1]
        if ind_right_edge0 == len(x) - 1:
            vel_right_edge = u_scale[step] * u[ind_right_edge0]
        else:
            vel_right_edge = (u_scale[step] * u[ind_right_edge0] * (x[ind_right_edge0 + 1] - right_edge) + u_scale[step] * u[ind_right_edge0 + 1] * (right_edge - x[ind_right_edge0])) / (x[ind_right_edge0 + 1] - x[ind_right_edge0])
        right_edge -= vel_right_edge * timesteps[step]
        mask[step + 1, :] = x > right_edge
    return mask


def adjoint_solve(x, a_scale, u_scale, a, u, z, λ_T, timesteps):
    """Solve the advection equation backwards in time from `λ_T` to compute the
    derivative of some functional w.r.t. the parameters
    
    Forward equation is $\frac{\partial z}{\partial t} = s_u(t)u\frac{\partial z}{\partial x} + s_a(t)v(z)
    
    adjoint is
    
    -\frac{\partial z}{\partial t} = -s_u(t)u\frac{\partial z}{\partail z} + s_a(t)v(z)
    
    """
    dx = np.diff(x)
    num_steps = len(timesteps)

    I = scipy.sparse.eye(len(x))
    D = scipy.sparse.diags([np.hstack(([0], -1. / dx)), 1. / dx], [0, 1])
    DU = D.T * scipy.sparse.diags([u], [0])

    # Pull a out of the vertical strain rate
    VSR = vertical_strain_rate(1.)

    λ = np.zeros((num_steps + 1, len(x)))
    λ[num_steps, :] = λ_T
    ##!! NEED TO CHECK IF SHOULD BE STEP OR STEP - 1 below...
    for step in range(num_steps - 1, -1, -1):
        L = I - timesteps[step] * u_scale[step] * DU
        f = (1 + timesteps[step] * a_scale[step] * a * VSR) * λ[step + 1:, :]
        λ[step:-1, :] = scipy.sparse.linalg.spsolve(L, f.T).T
    return λ


def msm(x, z, targ, uncertainty=None):
    if uncertainty is not None:
        return 0.5 * np.nansum(((z[:, 1:] + z[:, :-1]) / uncertainty**2.0 / 2. - (targ[:, 1:] + targ[:, :-1]) / uncertainty**2.0 / 2.)**2.0 * np.diff(x))
    else:
        return 0.5 * np.nansum(((z[:, 1:] + z[:, :-1]) / 2. - (targ[:, 1:] + targ[:, :-1]) / 2.)**2.0 * np.diff(x))


def adjoint_sensitivity_ascale(dx, z, λ, a):
    num_steps = z.shape[0] - 1
    dJ_da = np.zeros(num_steps)
    for k in range(num_steps):
        w = vertical_velocity(a[k], z[k + 1, :])
        dJ_da[k] = np.sum(-(λ[k, 1:] * w[1:] + λ[k, :-1] * w[:-1]) / 2 * dx)
    return dJ_da


def adjoint_sensitivity_a(dt, z, λ, s_a):
    num_points = z.shape[1]
    dJ_da = np.zeros(num_points)
    for k in range(num_points):
        w = vertical_velocity(s_a[k], z[k + 1, :])
        dJ_da[k] = np.sum((λ[k, 1:] * v[1:] + λ[k, :-1] * v[:-1]) * dzdx / 2 * dx)

    return dJ_da


def adjoint_sensitivity_vscale(dx, z, λ, v):
    num_steps = z.shape[0] - 1
    dJ_du = np.zeros(num_steps)

    for k in range(num_steps):
        dzdx = -np.diff(z[k, :]) / dx
        # we have the dzdx on the intervals
        # and we are calculating with trapezoids,
        # so we can just keep dzdx outside the expression
        dJ_du[k] = np.sum((λ[k, 1:] * v[1:] + λ[k, :-1] * v[:-1]) * dzdx / 2 * dx)
    return dJ_du


def derivative_ascale(x, a_scale, u_scale, a, u, z, target_layers, target_indices, timesteps, uncertainty=None):
    dx = np.diff(x)
    dJ_da = np.zeros_like(a_scale)
    for i, index in enumerate(target_indices):
        if uncertainty is not None:
            unc = uncertainty[i] ** 2.0
        else:
            unc = 1.0
        λ_i = (target_layers[i] - z[index, :]) / x[-1] / unc
        λ_i[np.isnan(λ_i)] = 0.0
        λ_new = adjoint_solve(x, a_scale[:index + 1], u_scale[:index + 1], a, u, z[:index + 1, :], λ_i, timesteps[:index + 1])
        dJ_da[-index:] += np.flip(adjoint_sensitivity_ascale(dx, z[:index + 1], λ_new, a), axis=-1)

    # Match present values at modern
    dJ_da[-1] = 0.
    return dJ_da


def derivative_vscale(x, a_scale, u_scale, a, u, z, target_layers, target_indices, timesteps, uncertainty=None):
    dx = np.diff(x)
    dJ_du = np.zeros_like(u_scale)
    for i, index in enumerate(target_indices):
        if uncertainty is not None:
            unc = uncertainty[i] ** 2.0
        else:
            unc = 1.0
        λ_i = (target_layers[i] - z[index, :]) / x[-1] / unc
        λ_i[np.isnan(λ_i)] = 0.0

        λ_new = adjoint_solve(x, a_scale[:index + 1], u_scale[:index + 1], a, u, z[:index + 1, :], λ_i, timesteps[:index + 1])
        dJ_du[-index:] += np.flip(adjoint_sensitivity_vscale(dx, z[:index + 1], λ_new, u), axis=-1)

    # We are assuming we know present day values, so
    dJ_du[-1] = 0.
    return dJ_du


def derivative_scales(x, a_scale, u_scale, a, u, z, target_layers, target_indices, timesteps, uncertainty=None):
    dx = np.diff(x)
    dJ_da = np.zeros_like(a_scale)
    dJ_du = np.zeros_like(u_scale)
    for i, index in enumerate(target_indices):
        if uncertainty is not None:
            unc = uncertainty[i] ** 2.0
        else:
            unc = 1.0

        λ_i = (target_layers[i] - z[index, :]) / x[-1] / unc
        λ_i[np.isnan(λ_i)] = 0.0
        λ_new = adjoint_solve(x, a_scale[:index + 1], u_scale[:index + 1], a, u, z[:index + 1, :], λ_i, timesteps[:index + 1])
        dJ_da[-index:] += np.flip(adjoint_sensitivity_ascale(dx, z[:index + 1], λ_new, a), axis=-1)
        dJ_du[-index:] += np.flip(adjoint_sensitivity_vscale(dx, z[:index + 1], λ_new, u), axis=-1)

    dJ_da[-1] = 0.
    dJ_du[-1] = 0.
    return np.hstack((dJ_da, dJ_du))
