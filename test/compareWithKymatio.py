import torch
import h5py
import numpy as np
from kymatio import Scattering2D
import torchvision

from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio.scattering1d.filter_bank import compute_params_filterbank

import numpy as np
import matplotlib.pyplot as plt


T = 2**9
J = 4
Q = 1

phi_f = []
psi1_f = []
for Q in [1, 2, 4, 8]:
    phi_tmp, psi1_tmp, psi2_tmp, tmp = scattering_filter_factory(np.log2(T), J, Q)
    phi_f.append(phi_tmp)
    psi1_f.append(psi1_tmp)

norms = [np.linalg.norm(psi_f[0]) for psi_f in psi1_f]
norms.append(np.linalg.norm(phi_f[0]))
#plt.xlim(0, 0.5)
Tp = np.arange(T)[0:300]
plt.figure()
fig, axes = plt.subplots(4,1)
fig.subplots_adjust(hspace=1)
axes[0].plot(Tp, phi_f[0][0][0:300], 'k')
axes[1].plot(Tp, phi_f[1][0][0:300], 'b')
axes[2].plot(Tp, phi_f[2][0][0:300], 'r')
axes[3].plot(Tp, phi_f[3][0][0:300], 'g')


for psi_f in psi1_f[0]:
    axes[0].plot(Tp, psi_f[0][0:300], 'k')
for psi_f in psi1_f[1]:
    axes[1].plot(Tp, psi_f[0][0:300], 'b')
for psi_f in psi1_f[2]:
    axes[2].plot(Tp, psi_f[0][0:300], 'r')
for psi_f in psi1_f[3]:
    axes[3].plot(Tp, psi_f[0][0:300], 'g')



plt.xlabel(r'$\xi$', fontsize=18)
plt.ylabel(r'$\hat\psi_j(\xi)$', fontsize=18)
fig.suptitle(f'Comparison as Q increases', fontsize=18)
axes[0].set_title("Q = 1")
axes[1].set_title("Q = 2")
axes[2].set_title("Q = 4")
axes[3].set_title("Q = 8")
plt.savefig("Q_vaguely_num_per_octave.pdf")



Q=8
J=4
r_psi=math.sqrt(0.5)
sigma0=0.1
sigma_low = sigma0 / math.pow(2, J)
xi1, sigma1, j1 = compute_params_filterbank(sigma_low, Q, r_psi=math.sqrt(0.5), alpha=5.)
plt.figure()
plt.plot(np.diff([math.log(x)/math.log(2) for x in reversed(xi1)]))
plt.title(f'Scaling Ratio Kymatio (logarithmic is flat)')
plt.xlabel('wavelet index (increasing frequency)')
plt.ylabel('scale ratio, log(2) scaling')
plt.show()
