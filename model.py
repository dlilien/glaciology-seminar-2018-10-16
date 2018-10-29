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
    """Return the vertical strain rate, which by DJ is independent of depth above the kink.

    .. math::
        \dot{\epsilon_{zz}} = \frac{\partial v_z}{\partial z} =  \frac{\partial}{\partial z} \left(a\frac{2(1-z/h) - k_h}{2 - k_h}\right)=-\frac{2a}{(2-k_h)h}
    """
    return -2 * a / h / (2 - kink_height)


def layer_depth(x, a_scale, u_scale, a, u, timesteps):
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
    num_steps = len(timesteps)
    dx = np.diff(x)

    I = scipy.sparse.eye(len(x))
    D = scipy.sparse.diags([np.hstack((-1/dx, [0])), 1/dx], [0, 1])

    z = np.zeros((num_steps + 1, len(x)))
    for step in range(num_steps):
        L = I - timesteps[step] * u_scale[step] * scipy.sparse.diags([u], [0]) * D
        f = z[step, :] + timesteps[step] * vertical_velocity(a_scale[step] * a, z[step, :])
        z[step + 1, :] = scipy.sparse.linalg.spsolve(L, f)

    return z


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
    D = scipy.sparse.diags([np.hstack(([0], -1/dx)), 1/dx], [0, 1])

    λ = np.zeros((num_steps + 1, len(x)))
    λ[num_steps, :] = λ_T
    ##!! NEED TO CHECK IF SHOULD BE STEP OR STEP - 1 below...
    for step in range(num_steps - 1, -1, -1):
        L = I - timesteps[step] * u_scale[step] * D.T * scipy.sparse.diags([u], [0])
        f = (1 + timesteps[step] * vertical_strain_rate(a_scale[step] * a)) * λ[step + 1, :]
        λ[step, :] = scipy.sparse.linalg.spsolve(L, f)

    return λ


def msm(x, z, targ):
    return 0.5 * np.nansum(((z[:, 1:] + z[:, :-1]) / 2. - (targ[:, 1:] + targ[:, :-1]) / 2.)**2.0 * np.diff(x))


def adjoint_sensitivity_ascale(x, z, λ, a):
    num_steps = z.shape[0] - 1
    num_points = z.shape[1]
    dJ_da = np.zeros(num_steps)
    dx = np.diff(x)
    for k in range(num_steps):
        w = vertical_velocity(a[k], z[k + 1, :])
        dJ_da[k] = np.sum(-(λ[k, 1:] * w[1:] + λ[k, :-1] * w[:-1]) / 2 * dx)
    return dJ_da


def adjoint_sensitivity_vscale(x, z, λ, v):
    num_steps = z.shape[0] - 1
    num_points = z.shape[1]
    dJ_du = np.zeros(num_steps)
    dx = np.diff(x)

    for k in range(num_steps):
            dzdx = -np.diff(z[k, :]) / dx
            # we have the dzdx on the intervals
            # and we are calculating with trapezoids,
            # so we can just keep dzdx outside the expression
            dJ_du[k] = np.sum((λ[k, 1:] * v[1:] + λ[k, :-1] * v[:-1]) * dzdx / 2 * dx)
    return dJ_du


def derivative_ascale(x, a_scale, u_scale, a, u,
                      z, target_layers, target_indices, timesteps):
    dJ_da = np.zeros_like(a_scale)
    for i, index in enumerate(target_indices):

        λ_i = (target_layers[i] - z[index, :]) / x[-1]
        λ_i[np.isnan(λ_i)] = 0.0
        λ_new = adjoint_solve(x, a_scale[:index + 1], u_scale[:index + 1], a, u, z[:index + 1, :], λ_i, timesteps[:index + 1])
        dJ_da[:index] += adjoint_sensitivity_ascale(x, z[:index + 1], λ_new, a)
    return dJ_da
        
                           
def derivative_vscale(x, a_scale, u_scale, a, u,
                      z, target_layers, target_indices, timesteps):
    dJ_du = np.zeros_like(u_scale)
    for i, index in enumerate(target_indices):
        λ_i = (target_layers[i] - z[index, :]) / x[-1]
        λ_i[np.isnan(λ_i)] = 0.0

        λ_new = adjoint_solve(x, a_scale[:index + 1], u_scale[:index + 1], a, u, z[:index + 1, :], λ_i, timesteps[:index + 1])
        dJ_du[:index] += adjoint_sensitivity_vscale(x, z[:index + 1], λ_new, u)
    return dJ_du


def derivative_scales(x, a_scale, u_scale, a, u,
                      z, target_layers, target_indices, timesteps):
    num_tsteps = u_scale.shape[0]
    dJ_dau = np.zeros((u_scale.shape[0] * 2,))
    λ_new = np.zeros((1, x.shape[0]))
    for i in range(len(target_indices) - 1, 0, -1):
        indend = target_indices[i]
        indstart = target_indices[i - 1]

        λ_mis = (target_layers[i] - z[indend, :]) / x[-1]
        λ_mis[np.isnan(λ_mis)] = 0.0

        λ_i = λ_mis + λ_new[0, :]
        λ_new = adjoint_solve(x, a_scale[indstart:indend + 1], u_scale[:indend + 1], a, u, z[indstart:indend + 1, :], λ_i, timesteps[indstart:indend + 1])

        dJ_dau[indstart:indend] = adjoint_sensitivity_ascale(x, z[indstart:indend + 1], λ_new, a)
        dJ_dau[u_scale.shape[0] + indstart:u_scale.shape[0] + indend] += adjoint_sensitivity_vscale(x, z[indstart:indend + 1], λ_new, u)

    # and the fencepost
    if target_indices[0] != 0:
        indend = target_indices[0]
        indstart = 0

        λ_i = (target_layers[0] - z[indend, :]) / x[-1] + λ_new[0, :]
        λ_new = adjoint_solve(x, a_scale[indstart:indend + 1], u_scale[:indend + 1], a, u, z[indstart:indend + 1, :], λ_i, timesteps[indstart:indend + 1])

        dJ_dau[indstart:indend] = adjoint_sensitivity_ascale(x, z[indstart:indend + 1], λ_new, a)
        dJ_dau[u_scale.shape[0] + indstart:u_scale.shape[0] + indend] += adjoint_sensitivity_vscale(x, z[indstart:indend + 1], λ_new, u)
    return dJ_dau


def derivative_ascale_onelayer(x, a_scale, u_scale, a, u, z, target_layer, timesteps):
    λ_final = (target_layer - z[-1, :]) / x[-1]
    λ = adjoint_solve(x, a_scale, u_scale, a, u, z, λ_final, timesteps)
    return adjoint_sensitivity_ascale(x, z, λ, a)
