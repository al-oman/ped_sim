"""Mirrors diffuser/datasets/normalization.py. GaussianNormalizer matches
their class of the same name ((x - mean) / std per dimension, used for the
conditioning vectors). Their trajectory default, LimitsNormalizer (per-dim
min/max to [-1, 1]), cannot be used here: the ACI keep-out disks are circles,
so trajectory space must be scaled by a single scalar or the disks would
become ellipses — IsotropicNormalizer is that scalar replacement.
"""

import numpy as np


class Normalizer:
    """Fit on data X (n, dim) at construction; normalize/unnormalize after."""

    def __init__(self, X):
        self.X = X.astype(np.float32)

    def normalize(self, x):
        raise NotImplementedError()

    def unnormalize(self, x):
        raise NotImplementedError()


class GaussianNormalizer(Normalizer):
    """(x - mean) / std, per dimension."""

    def __init__(self, X):
        super().__init__(X)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0) + 1e-6

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means


class IsotropicNormalizer(Normalizer):
    """x / std with one scalar std over all dimensions, so circles stay
    circles (see module docstring)."""

    def __init__(self, X):
        super().__init__(X)
        self.std = self.X.std()

    def normalize(self, x):
        return x / self.std

    def unnormalize(self, x):
        return x * self.std
