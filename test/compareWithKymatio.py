import torch
import h5py
import numpy as np
from kymatio import Scattering2D
import torchvision


from kymatio.scattering1d.filter_bank import scattering_filter_factory

import numpy as np
import matplotlib.pyplot as plt


T = 2**9
J = 4
Q = 8

phi_f, psi1_f, psi2_f, tmp = scattering_filter_factory(np.log2(T), J, Q)


plt.xlim(0, 0.5)
plt.figure()
plt.plot(np.arange(T), phi_f[0], 'r')

for psi_f in psi1_f:
    plt.plot(np.arange(T), psi_f[0], 'b')



plt.xlabel(r'$\omega$', fontsize=18)
plt.ylabel(r'$\hat\psi_j(\omega)$', fontsize=18)
plt.title('First-order filters (Q = {})'.format(Q), fontsize=18)
plt.show()
np.linalg.norm()
