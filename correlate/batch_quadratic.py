r"""Code to run large batches of quadratic estimators on combinations of
data/sims
"""
import numpy as np
import math
from core import algebra
from utils import data_paths
from simulations import corr21cm
import multiprocessing
from map import physical_gridding
from utils import binning
from correlate import map_pair as mp
from correlate import pwrspec_estimation as pe
import sys
from utils import batch_handler


def call_xspec_run(map1_key, map2_key,
                   noiseinv1_key, noiseinv2_key,
                   subtract_mean=False, degrade_resolution=False,
                   unitless=True, return_3d=False):
    r"""a free-standing function which calls the xspec analysis
    batteries included
    """

    # define the bad frequency list
    cutlist = [6, 7, 8, 15, 16, 18, 19, 20, 21, 22, 37, 103, 104, 105, 106,
               107, 108, 130, 131, 132, 133, 134, 237, 244, 254, 255]

    augmented = [177, 194, 195, 196, 197, 198, 201, 204, 209, 213, 229]
    for entry in augmented:
        cutlist.append(entry)

    badweights = [133, 189, 192, 193, 194, 195, 196, 197, 198, 208, 209, 213,
                  233]
    for entry in badweights:
        cutlist.append(entry)

    freq = range(256)
    for entry in cutlist:
        try:
            freq.remove(entry)
        except ValueError:
            print "can't cut %d" % entry

    print "%d of %d freq slices removed" % (len(cutlist), len(freq))
    print freq

    # define the 1D spectral bins
    nbins=40
    bins = np.logspace(math.log10(0.00702349679605685),
                       math.log10(2.81187396154818),
                       num=(nbins + 1), endpoint=True)

    # initialize and calculate the xspec
    simpair = mp.MapPair(map1_key, map2_key,
                         noiseinv1_key, noiseinv2_key,
                         freq)

    if subtract_mean:
        simpair.subtract_weighted_mean()

    if degrade_resolution:
        simpair.degrade_resolution()

    retval = simpair.pwrspec_summary(bins=bins, unitless=unitless,
                                     return_3d=return_3d)
    return retval


def test_batch_sim_run():
    datapath_db = data_paths.DataPath()

    outpath = datapath_db.fetch("quadratic_batch_simulations")
    print "writing to: " + outpath

    funcname = "correlate.batch_quadratic.call_xspec_run"
    caller = batch_handler.MemoizeBatch(funcname, outpath,
                                        generate=True, verbose=True)
    for index in range(100):
        sim_key = "simideal_15hr_beam"
        map1_key = "db:%s:%s" % (sim_key, index)
        map2_key = "db:%s:%s" % (sim_key, index)
        noiseinv1_key = "db:GBT_15hr_map_cleaned_0mode:A_with_B;noise_inv"
        noiseinv2_key = "db:GBT_15hr_map_cleaned_0mode:A_with_B;noise_inv"
        # could also use ./observed_window.npy as the weight

        caller.execute(map1_key, map2_key,
                       noiseinv1_key, noiseinv2_key)

    caller.multiprocess_stack()
    #for specdata in zip(bin_left, bin_center,
    #                    bin_right, counts_histo, binavg):
    #    print ("%10.15g " * 5) % specdata

if __name__ == '__main__':
    test_batch_sim_run()

