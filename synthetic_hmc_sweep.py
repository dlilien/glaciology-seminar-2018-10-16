#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 dlilien <dlilien@berens>
#
# Distributed under terms of the MIT license.

"""
Run through a number of changes in parameters
"""
import os
import numpy as np
import hmc_synthetic
from joblib import Parallel, delayed


n_samples = 1000
dv_range = np.arange(0, 0.25, 0.05)


def run_if_not_done(n_samples, da, dv):
    if not os.path.exists('hmc_output_da{:06.3f}_dv{:06.3f}_n{:d}.pickle'.format(da, dv, n_samples)):
        print('Doing da={:f}, dv={:f}'.format(da, dv))
        hmc_synthetic.main(n_samples, da, dv)


for da in np.arange(0, 0.25, 0.05):
    Parallel(n_jobs=len(dv_range))(delayed(run_if_not_done)(n_samples, da, dv) for dv in dv_range)
