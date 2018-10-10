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
