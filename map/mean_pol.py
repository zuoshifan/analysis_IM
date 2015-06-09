import numpy as np
import scipy.linalg as alg

# rotation matrix
rot_ang1 = 45.0 # degrees
rot_ang2 = -45.0  # degrees
# rot_ang2 = 45.0  # degrees

rot_ang1 = np.radians(rot_ang1) # radians
rot_ang2 = np.radians(rot_ang2) # radians

c1 = np.cos(rot_ang1)
s1 = np.sin(rot_ang1)
c2 = np.cos(rot_ang2)
s2 = np.sin(rot_ang2)
rot_mat = np.array([[c1, s1],
                    [-s1, c1],
                    [c2, s2],
                    [-s2, c2]])

# signal
sig = np.array([2.0, 1.5]).reshape(2, 1)
# sig = 10.0 * np.array([2.0, 1.5]).reshape(2, 1)
# sig = 0.1 * np.array([2.0, 1.5]).reshape(2, 1)

# mean noise
nbarQ = 2.1
nbarU = 1.3
nbar = np.array([nbarQ, nbarU, nbarQ, nbarU]).reshape(4, 1)
Nbar = np.array([[nbarQ**2, 0.0, nbarQ**2, 0.0],
                 [0.0, nbarU**2, 0.0, nbarU**2],
                 [nbarQ**2, 0.0, nbarQ**2, 0.0],
                 [0.0, nbarU**2, 0.0, nbarU**2]])


# random noise
sigma = 0.05
sigma2 = sigma**2
n = np.random.normal(0.0, sigma, 4).reshape(4, 1)
N = np.diag([sigma2, sigma2, sigma2, sigma2])

# get the simulated data
d = np.dot(rot_mat, sig) + n + nbar

# solve for signal
Ninv = alg.inv(N + Nbar)
dirty = np.dot(rot_mat.T, np.dot(Ninv, d))
inv_cov = alg.inv(np.dot(np.dot(rot_mat.T, Ninv), rot_mat))
s_est = np.dot(inv_cov, dirty)
print 's_est: ', s_est.flatten()
print 'sig: ', sig.flatten()
print 'diff: %s, %s%%' % ((s_est - sig).flatten(), (100.0 * (s_est - sig) / sig).flatten())