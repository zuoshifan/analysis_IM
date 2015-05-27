import numpy as np
import numpy.linalg as alg
import scipy as sp

import core.algebra as al
# from utils import mpiutil


Q_dmap_fname = '/mnt/scratch-lustre/sfzuo/programming/workspace/analysis_IM/maps_old_bak/tut_dirty_map_Q_762.npy'
U_dmap_fname = '/mnt/scratch-lustre/sfzuo/programming/workspace/analysis_IM/maps_old_bak/tut_dirty_map_U_762.npy'
Q_cmap_fname = '/mnt/scratch-lustre/sfzuo/programming/workspace/analysis_IM/maps_old_bak/tut_clean_map_Q_762_coupled.npy'
U_cmap_fname = '/mnt/scratch-lustre/sfzuo/programming/workspace/analysis_IM/maps_old_bak/tut_clean_map_U_762_coupled.npy'
Q_noise_inv_fname = '/mnt/scratch-lustre/sfzuo/programming/workspace/analysis_IM/maps_old_bak/tut_noise_inv_Q_762.npy'
U_noise_inv_fname = '/mnt/scratch-lustre/sfzuo/programming/workspace/analysis_IM/maps_old_bak/tut_noise_inv_U_762.npy'

# Load the dirty maps
Q_dirty_map = al.load(Q_dmap_fname)
Q_dirty_map = al.make_vect(Q_dirty_map)
if Q_dirty_map.axes != ('freq', 'ra', 'dec') :
    msg = ("Expeced dirty map to have axes ('freq',"
           "'ra', 'dec'), but it has axes: "
           + str(Q_dirty_map.axes))
    raise ce.DataError(msg)

U_dirty_map = al.load(U_dmap_fname)
U_dirty_map = al.make_vect(U_dirty_map)
if U_dirty_map.axes != ('freq', 'ra', 'dec') :
    msg = ("Expeced dirty map to have axes ('freq',"
           "'ra', 'dec'), but it has axes: "
           + str(U_dirty_map.axes))
    raise ce.DataError(msg)

# Initialize the clean map
Q_clean_map = al.info_array(sp.zeros(Q_dirty_map.shape))
Q_clean_map.info = dict(Q_dirty_map.info)
Q_clean_map = al.make_vect(Q_clean_map)
U_clean_map = al.info_array(sp.zeros(U_dirty_map.shape))
U_clean_map.info = dict(U_dirty_map.info)
U_clean_map = al.make_vect(U_clean_map)


# shape = dirty_map.shape

# Load the noise inv matrices
Q_noise_inv = al.open_memmap(Q_noise_inv_fname, 'r')
Q_noise_inv = al.make_mat(Q_noise_inv)

if Q_noise_inv.axes != ('freq', 'ra', 'dec', 'ra', 'dec'):
    msg = ("Expeced noise matrix to have axes "
           "('freq', 'ra', 'dec', 'ra', 'dec'), "
           "but it has: " + str(Q_noise_inv.axes))
    raise ce.DataError(msg)

U_noise_inv = al.open_memmap(U_noise_inv_fname, 'r')
U_noise_inv = al.make_mat(U_noise_inv)

if U_noise_inv.axes != ('freq', 'ra', 'dec', 'ra', 'dec'):
    msg = ("Expeced noise matrix to have axes "
           "('freq', 'ra', 'dec', 'ra', 'dec'), "
           "but it has: " + str(U_noise_inv.axes))
    raise ce.DataError(msg)

# construct Q, U dirty map
QU_dirty_map = sp.zeros((2,)+Q_dirty_map.shape, dtype=Q_dirty_map.dtype)
QU_dirty_map = al.make_vect(QU_dirty_map,
                            axis_names=('pol', 'freq', 'ra', 'dec'))
QU_dirty_map[0] = Q_dirty_map
QU_dirty_map[1] = U_dirty_map
del Q_dirty_map
del U_dirty_map

##################  simple inv in diagonal ##########################
# # construct inverse of the Q, U noise inv matrix
# inv_QU_noise_inv = sp.zeros((2, 2)+Q_noise_inv.row_shape()+Q_noise_inv.col_shape()[1:], dtype=Q_noise_inv.dtype)
# inv_QU_noise_inv = al.make_mat(inv_QU_noise_inv, axis_names=('pol', 'pol', 'freq', 'ra', 'dec', 'ra', 'dec'), row_axes=(0, 2, 3, 4), col_axes=(1, 2, 5, 6))

# shape = Q_noise_inv.row_shape()[1:] + Q_noise_inv.col_shape()[1:]
# mat_shape = (np.prod(Q_noise_inv.row_shape()[1:]), np.prod(Q_noise_inv.col_shape()[1:]))
# for ii in range(Q_noise_inv.shape[0]):
#     # inv_QU_noise_inv[0, 0, ii] = alg.pinv(np.array(Q_noise_inv[ii]).reshape(shape))
#     Qninv = alg.pinv(Q_noise_inv[ii].reshape(mat_shape)).reshape(shape)
#     Uninv = alg.pinv(U_noise_inv[ii].reshape(mat_shape)).reshape(shape)
#     inv_QU_noise_inv[0, 0, ii] = Qninv # al.make_mat(Qninv, axis_names=('ra', 'dec', 'ra', 'dec'), row_axes=(0, 1), col_axes=(2, 3)).inv()
#     inv_QU_noise_inv[1, 1, ii] = Uninv # al.make_mat(Uninv, axis_names=('ra', 'dec', 'ra', 'dec'), row_axes=(0, 1), col_axes=(2, 3)).inv()
# del Q_noise_inv
# del U_noise_inv
##################  simple inv in diagonal ##########################


##################  simple diagonal inv #############################
# construct Q, U noise inv matrix
row_shape = Q_noise_inv.row_shape()
row_shape = (row_shape[0],) + (2,) + row_shape[1:]
col_shape = row_shape[1:]
QU_noise_inv = sp.zeros(row_shape+col_shape, dtype=Q_noise_inv.dtype)
QU_noise_inv = al.make_mat(QU_noise_inv, axis_names=('freq', 'pol', 'ra', 'dec', 'pol', 'ra', 'dec'), row_axes=(0, 1, 2, 3), col_axes=(0, 4, 5, 6))
QU_noise_inv[:, 0, :, :, 0] = Q_noise_inv # Q noise matrix
del Q_noise_inv
QU_noise_inv[:, 1, :, :, 1] = U_noise_inv # U noise matrix
del U_noise_inv

# construct inverse of the Q, U noise inv matrix
inv_QU_noise_inv = sp.zeros(row_shape+col_shape, dtype=QU_noise_inv.dtype)
inv_QU_noise_inv = al.make_mat(inv_QU_noise_inv, axis_names=('freq', 'pol', 'ra', 'dec', 'pol', 'ra', 'dec'), row_axes=(0, 1, 2, 3), col_axes=(0, 4, 5, 6))
shape = col_shape + col_shape
mat_shape = (np.prod(col_shape), np.prod(col_shape))
for ii in range(inv_QU_noise_inv.shape[3]):
    inv_QU_noise_inv[ii] = alg.pinv(QU_noise_inv[ii].reshape(mat_shape)).reshape(shape)
# inv_QU_noise_inv = QU_noise_inv.inv()
del QU_noise_inv
print inv_QU_noise_inv.shape
print inv_QU_noise_inv.info
##################  simple diagonal inv #############################

QU_clean_map = al.partial_dot(inv_QU_noise_inv, QU_dirty_map)
del QU_dirty_map
print QU_clean_map.shape
print QU_clean_map.info

Q_clean_map[:] = QU_clean_map[:, 0]
U_clean_map[:] = QU_clean_map[:, 1]

al.save(Q_cmap_fname, Q_clean_map)
al.save(U_cmap_fname, U_clean_map)


print 'success'
