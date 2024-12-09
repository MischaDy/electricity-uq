from pprint import pprint

import numpy as np


def f(a):
    mid = len(a) // 2
    first, second = a[:mid], a[mid:]
    ci_limits = zip(first, reversed(second))
    cis = [high - low for low, high in ci_limits]
    return cis


quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
# a2 = [0.1, 0.25, 0.75, 0.9]
#
# for a in [a1, a2]:
#     res = f(a)
#     print(res)

"""
(n_samples, 2, n_alpha).
[:, 0, :]: Lower bound of the prediction interval.
[:, 1, :]: Upper bound of the prediction interval.
"""

N = 10
mean = np.log(1 + np.arange(N))

alphas = f(quantiles)
cis = [
    ([val - 2 * alpha
      for alpha in alphas],
     [val + 2 * alpha
      for alpha in alphas])
    for val in mean
]
cis_np = np.array(cis).reshape(N, 2, -1)

# print("shape correct?", cis_np.shape == (N, 2, len(alphas)))
# pprint(cis_np)

# [[-1.8, -1.6, -1. ],
#  [ 1.8,  1.6,  1. ]]

y_quantiles = [
    sorted(ci.flatten())
    for ci in cis_np
]
pprint(y_quantiles)
