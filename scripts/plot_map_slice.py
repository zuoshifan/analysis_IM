# This script generates a set of maps in RA and DEC for selected frequencies in the map data. Can be used to make plots of maps.

import sys
import argparse
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import pylab
import pyfits
import core.algebra as al


parser = argparse.ArgumentParser(description="This script generates a set of maps in RA and DEC for selected frequencies in the map data. Can be used to make plots of maps.")
parser.add_argument("input_file", type=str, help="The name of the input file containing the maps.")
parser.add_argument('-f', '--figfmt', default='png', help='Output image format.')
parser.add_argument('-i', '--ifreqs', type=int, nargs='*', default=[0], help='Frequency channels to plot (start from 0). Negative integer N means the last Nth channel.')
parser.add_argument('--min', type=float, help='The min value of the visualize range in the output image.')
parser.add_argument('--max', type=float, help='The max value of the visualize range in the output image.')
# parser.add_argument('-l', '--figlength', type=float, default=13, help='Output figure length.')
# parser.add_argument('-w', '--figwidth', type=float, default=5, help='Output figure width.')

args = parser.parse_args()


filename = args.input_file

array = al.load(filename)
array = al.make_vect(array)

ras = array.get_axis('ra')
decs = array.get_axis('dec')
freqs = array.get_axis('freq')
freqs = freqs/1e6


print 'Total %d frequencies in the input map:' % len(freqs)
for slice, freq in enumerate(freqs):
    print slice, ": ", freq

# plot the maps
for slice in args.ifreqs:
   nancut = (array[slice] < 10e10) & ( array[slice] != np.NaN )
   cut = ( array[slice] > 3.0*array[slice][nancut].std() )
   array[slice][cut] = 3.0*array[slice][nancut].std()
   cut = ( array[slice] < -3.0*array[slice][nancut].std() )
   array[slice][cut] = -3.0*array[slice][nancut].std()

#   Need to rotate array[slice] because axes were flipped
   new_array = scipy.transpose(array[slice])

   pylab.imshow(new_array, cmap='hot', vmin=args.min, vmax=args.max, extent=(ras.max(),ras.min(),decs.min(),decs.max()), origin='lower')
#   pylab.imshow(new_array, interpolation='gaussian', cmap='hot', extent=(ras.max(),ras.min(),decs.min(),decs.max()), origin='lower')
   pylab.colorbar() #For some reason this isn't working, fixed...
   pylab.savefig(filename.replace('.npy', '_' + str(freqs[slice])[:3] + '.png'))
   pylab.clf()
