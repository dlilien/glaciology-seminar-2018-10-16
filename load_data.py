import numpy as np
from scipy.io import loadmat
import scipy.interpolate

def load():
    layers_ie = loadmat('ie_layers.mat')
    sx, sy = layers_ie['psx_layers'][0], layers_ie['psy_layers'][0]
    dx = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
    x = np.hstack(([0], np.cumsum(dx)))[:5000]
    target_layers = {}
    target_ages = {}
    for i in range(24):
        if 'layer_{:d}'.format(i) in layers_ie:
            target_layers['layer_{:d}'.format(i)] = layers_ie['layer_{:d}'.format(i)][0][:5000]
            target_ages['layer_{:d}'.format(i)] = layers_ie['layer_{:d}_age'.format(i)][0]
    target_ages = np.array([(tn, ta) for tn, ta in target_ages.items()], dtype=[('Name', 'S12'), ('Age', float)])
    
    # target_ages = np.sort(target_ages, order='Age')
    ta_ind = target_ages.sort(order='Age')       
    target_layer_stack = np.vstack([target_layers[tn.decode('utf-8')] for tn in target_ages['Name']])

    # Find any points that are duplicated or where there's data missing and remove them
    repeat_point_indices = np.where(dx < 1.0)[0] + 1
    no_data_indices = np.where(np.isnan(np.sum(target_layer_stack, axis=0)))[0]
    good_indices = set(range(5000)) - set(repeat_point_indices) - set(no_data_indices)
    indices = np.array(list(good_indices))
    indices.sort()

    x = x[indices]
    for i in range(24):
        if 'layer_{:d}'.format(i) in target_layers:
            target_layers['layer_{:d}'.format(i)] = target_layers['layer_{:d}'.format(i)][indices]
            
    more_data = loadmat('vels.mat')
    vel_interpolater = scipy.interpolate.interp1d(more_data['dists'][0], more_data['vels'][0])
    u = vel_interpolater(x)
    acc_interpolater = scipy.interpolate.interp1d(more_data['acc_dists'][0], more_data['acc'][0])
    a = acc_interpolater(x)
    return x, dx, target_layer_stack[:, indices], target_ages, a, u