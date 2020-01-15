import matplotlib.pyplot as plt
import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from kymatio.scattering2d.utils import fft2


def rot(θ):
    return np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])
x=1; y=0; θ=np.pi/3; sigX = .4; sigY = .1
def rotated(θ, x, y):
    return rot(θ) @ np.array([x,y])

def f(xloc,yloc,sigX,sigY,θ):
    return np.array([[np.exp(-1/2 * (rotated(θ,x,y).transpose() @
                                     np.array([[1/sigX**2, 0], [0, 1/sigY**2]]) @
                                     rotated(θ,x,y))) for x in xloc] for y in yloc ])
rot(0.0)
xlocs = np.linspace(-1, 1, num=512)
gaussianExample = f(xlocs, xlocs, .6, .3, np.pi/6)
mutilated = np.copy(gaussianExample)
nZeros = 12
mutilated[0:nZeros, :] = np.zeros((nZeros, mutilated.shape[1]))
plt.imshow(gaussianExample)
plt.show()
plt.imshow(mutilated)
plt.show()
M = 512
J = 2
L = 7
filters_set = filter_bank(M, M, J, L=L)
transformed = np.zeros((512,512,len(filters_set['psi'])))
transMut = np.zeros((512,512,len(filters_set['psi'])))
for i,filter in enumerate(filters_set['psi']):
    f_r = filter[0][..., 0].numpy()
    f_i = filter[0][..., 1].numpy()
    f = f_r + 1j*f_i
    transformed[:,:,i] = np.fft.ifft2(fft2(gaussianExample) * f)
    transMut[:,:,i] = np.fft.ifft2(fft2(mutilated) * f)
    
filter0 = filters_set['psi'][2]
filter0 = filter0[0][..., 0].numpy() + 1j*filter0[0][..., 1].numpy()
plt.imshow(abs(filter0))
plt.show()


plt.imshow(transformed[:,:,4])
plt.show()
transMut.shape
fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
fig.set_figheight(6)
fig.set_figwidth(6)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
for i in range(transMut.shape[2]):
    axs[i // L, i% L].imshow(transMut[:,:,i])
    axs[i // L, i% L].axis('off')
    axs[i // L, i% L].set_title("$j = {}$ \n $\\theta={}$".format(i // L, i % L))
fig.show()


reordered = np.reshape(transformed, (-1,))
reordered = np.sort(np.abs(reordered))[::-1]
reorderedMut = np.reshape(transMut,(-1,))
reorderedMut = np.sort(np.abs(reorderedMut))[::-1]
plt.loglog(range(1,1+len(reordered)), reorderedMut, basex=np.exp(1),
           basey=np.exp(1),label = 'mutilated')
plt.loglog(range(1,1+len(reordered)), reordered, basex=np.exp(1),
           basey=np.exp(1),label = 'gaussian')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('log position')
plt.ylabel('log value')
plt.show()
plt.savefig("gaussiansKymats_Wavelets.pdf")
