import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('/mnt/scratch-lustre/sfzuo/programming/workspace/analysis_IM/maps/pol_freq_meanqu.txt')
freq = data[:, 0]
Q = data[:, 1]
U = data[:, 2]

plt.figure()
plt.plot(freq, Q, label='Q')
plt.plot(freq, U, label='U')
plt.xlabel(r'$\nu$ / MHz')
plt.ylim(-0.5, 2.5)
plt.legend(frameon=False, loc=1)
plt.show()