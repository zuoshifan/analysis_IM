"""
A set of functions to bin optical catalog data into cubes
http://astronomy.swin.edu.au/~cblake/tzuching.html
    In the 15h field,
    ra range is 214. to 223.
    dec range is 0. to 4.
    freq range is 676. to 947.

    map_root = '/mnt/raid-project/gmrt/tcv/maps/'
    template_15hr = map_root + 'sec_A_15hr_41-90_clean_map_I.npy'
    template_22hr = map_root + 'sec_A_22hr_41-90_clean_map_I.npy'
    template_1hr = map_root + 'sec_A_1hr_41-16_clean_map_I.npy'
    template_15hr = 'templates/wigglez_15hr_complete.npy'
    template_22hr = 'templates/wigglez_22hr_complete.npy'
    template_1hr = 'templates/wigglez_1hr_complete.npy'
"""
import numpy as np
import shelve
import random
import copy
from core import algebra
from core import constants as cc
from utils import binning
from utils import data_paths
from kiyopy import parse_ini
# TODO: make better parameter passing for catalog binning


def bin_catalog_data(catalog, freq_axis, ra_axis,
                     dec_axis, debug=False, use_histogramdd=False):
    """
    bin catalog data onto a grid in RA, Dec, and frequency
    This currently assumes that all of the axes are uniformly spaced
    """
    catalog_frequencies = cc.freq_21cm_MHz * 1.e6 / (1 + catalog['z'])
    num_catalog = catalog.size
    sample = np.zeros((num_catalog, 3))
    sample[:, 0] = catalog_frequencies
    sample[:, 1] = catalog['RA']
    sample[:, 2] = catalog['Dec']

    freq_edges = binning.find_edges(freq_axis)
    ra_edges = binning.find_edges(ra_axis)
    dec_edges = binning.find_edges(dec_axis)

    if debug:
        binning.print_edges(sample[:, 0], freq_edges, "frequency")
        binning.print_edges(sample[:, 1], ra_edges, "RA")
        binning.print_edges(sample[:, 2], dec_edges, "Dec")
        print sample, freq_edges, ra_edges, dec_edges
        print np.max(sample, axis=0)
        print np.min(sample, axis=0)

    if use_histogramdd:
        count_cube, edges = np.histogramdd(sample, bins=[freq_edges,
                                                         ra_edges, dec_edges])
        print edges
    else:
        count_cube = binning.histogram3d(sample, freq_edges, ra_edges, dec_edges)

    return count_cube


def bin_catalog_file(filename, freq_axis, ra_axis,
                     dec_axis, skip_header=None, debug=False, mock=False):
    """Bin the catalog given in `filename` using the given freq, ra, and dec
    axes.
    """

    # read the WiggleZ catalog and convert redshift axis to frequency
    if mock:
        ndtype = [('RA', float), ('Dec', float), ('z', float),
                  ('mag', float), ('wsel', float), ('compl', float),
                  ('selz', float), ('selz_mu', float), ('bjlim', float)]
    else:
        ndtype = [('RA', float), ('Dec', float), ('z', float),
                  ('mag', float), ('wsel', float), ('compl', float),
                  ('selz', float), ('selz_mu', float), ('bjlim', float), 
                  ('serial', int)]

    # TODO: numpy seems to be an old version that does not have the skip_header
    # argument here! skiprows is identical
    catalog = np.genfromtxt(filename, dtype=ndtype, skiprows=skip_header)

    catalog['RA'] = catalog['RA']*180./np.pi
    catalog['Dec'] = catalog['Dec']*180./np.pi

    if debug:
        print filename + ": " + repr(catalog.dtype.names) + \
              ", n_records = " + repr(catalog.size)

    return bin_catalog_data(catalog, freq_axis, ra_axis, dec_axis,
                            debug=debug)

def convert_B1950_to_J2000(ra, dec, degree_in=False, degree_out=True):
    if degree_in:
        ra = ra*np.pi/180.
        dec = dec*np.pi/180.
    ra2000 = ra + 0.640265 + 0.27836 * np.sin(ra) * np.tan(dec)
    dec2000 = dec + 0.27836 * np.cos(ra)

    if degree_out:
        ra2000 = ra2000*180./np.pi
        dec2000 = dec2000*180./np.pi

    return ra2000, dec2000

bin2dfparams_init = {
        "infile_data": "/Users/ycli/DATA/2df/catalogue/real_catalogue_2df.out",
        "infile_mock": "/Users/ycli/DATA/2df/catalogue/mock_catalogue_2df_%03d.out",
        "outfile_data": "/Users/ycli/DATA/2df/map/real_map_2df.npy",
        "outfile_mock": "/Users/ycli/DATA/2df/map/mock_map_2df_%03d.npy",
        "outfile_deltadata": "/Users/ycli/DATA/2df/map/real_map_2df_delta.npy",
        "outfile_deltamock": "/Users/ycli/DATA/2df/map/mock_map_2df_delta_%03d.npy",
        "outfile_selection": "/Users/ycli/DATA/2df/map/sele_map_2df.npy",
        "outfile_separable": "/Users/ycli/DATA/2df/map/sele_map_2df_separable.npy",
        "template_file": "/Users/ycli/DATA/2df/tempfile",
        "mock_number": 10,
        }
bin2dfprefix = 'bin2df_'


class Bin2dF(object):
    def __init__(self, parameter_file=None, params_dict=None, feedback=0):
        self.params = params_dict
        #self.datapath_db = data_paths.DataPath()

        if parameter_file:
            self.params = parse_ini.parse(parameter_file, bin2dfparams_init,
                                          prefix=bin2dfprefix)

        # gather names of the input catalogs
        self.infile_data = self.params['infile_data']

        self.infile_mock = self.params['infile_mock']

        # gather names of all the output files
        self.outfile_data = self.params['outfile_data']

        self.outfile_delta_data = self.params['outfile_deltadata']

        self.outfile_selection  = self.params['outfile_selection']

        self.outfile_separable  = self.params['outfile_separable']

        self.outfile_mock       = self.params['outfile_mock']

        self.outfile_delta_mock = self.params['outfile_deltamock']

        # gather axis information from the template file
        self.template_map = \
            algebra.make_vect(algebra.load(self.params['template_file']))

        self.freq_axis = self.template_map.get_axis('freq')
        self.ra_axis = self.template_map.get_axis('ra')
        self.dec_axis = self.template_map.get_axis('dec')

        # placeholders for data products
        self.realmap_binning = None
        self.selection_function = None
        self.separable_selection = None

    def execute(self, processes):
        pass
        #np.set_printoptions(threshold=np.nan)
        print "finding the binned data"
        self.realmap()
        print "finding the binned mock and selection function"
        self.selection()
        #print "finding the separable form of the selection"
        #self.separable()
        #print "finding the optical overdensity"
        #self.delta()

    def realmap(self):
        """bin the real WiggleZ catalog"""
        self.realmap_binning = bin_catalog_file(self.infile_data,
                                                self.freq_axis,
                                                self.ra_axis, self.dec_axis,
                                                skip_header=1,
                                                debug=False)

        map_2df = algebra.make_vect(self.realmap_binning,
                                        axis_names=('freq', 'ra', 'dec'))

        map_2df.copy_axis_info(self.template_map)
        algebra.save(self.outfile_data, map_2df)

        return

    def selection(self):
        """bin the mock catalogs"""
        self.selection_function = np.zeros(self.template_map.shape)

        for mockindex in range(self.params['mock_number']):
            print mockindex
            mockfile = self.infile_mock%mockindex
            mock_binning = bin_catalog_file(mockfile, self.freq_axis,
                                            self.ra_axis, self.dec_axis,
                                            skip_header=1, mock=True)

            self.selection_function += mock_binning

            # if this binned mock catalog should be saved
            #if mockindex in self.outfile_mock[0]:

            print "mock", self.outfile_mock%mockindex
            map_2df_mock = algebra.make_vect(mock_binning,
                                             axis_names=('freq', 'ra', 'dec'))

            map_2df_mock.copy_axis_info(self.template_map)

            algebra.save(self.outfile_mock%mockindex, map_2df_mock)

        # adding the real map back to the selection function is a kludge which
        # ensures the selection function is not zero where there is real data
        # (limit of insufficient mocks)
        self.selection_function += self.realmap_binning
        self.selection_function /= float(self.params['mock_number'] + 1)
        print np.mean(self.selection_function)

        map_2df_selection = algebra.make_vect(self.selection_function,
                                              axis_names=('freq', 'ra', 'dec'))

        map_2df_selection.copy_axis_info(self.template_map)

        algebra.save(self.outfile_selection, map_2df_selection)

        return

    def separable(self):
        # now assume separability of the selection function
        spatial_selection = np.sum(self.selection_function, axis=0)

        freq_selection = np.apply_over_axes(np.sum, self.selection_function, [1, 2])

        self.separable_selection = (freq_selection * spatial_selection)

        self.separable_selection /= np.sum(freq_selection.flatten())

        map_2df_separable = algebra.make_vect(self.separable_selection,
                                                  axis_names=('freq', 'ra', 'dec'))

        map_2df_separable.copy_axis_info(self.template_map)

        algebra.save(self.outfile_separable, map_2df_separable)

        return


    def produce_delta_map(self, optical_file, optical_selection_file):
        map_optical = algebra.make_vect(algebra.load(optical_file))
        map_nbar = algebra.make_vect(algebra.load(optical_selection_file))

        old_settings = np.seterr(invalid="ignore", under="ignore")
        map_delta = map_optical / map_nbar - 1.
        np.seterr(**old_settings)

        # TODO: also consider setting the nbar to zero outside of galaxies?
        map_delta[np.isinf(map_delta)] = 0.
        map_delta[np.isnan(map_delta)] = 0.
        # if e.g. nbar is zero, then set the point as if there were no galaxies
        # downstream, nbar=0 should coincide with zero weight anyway
        #map_delta[np.isinf(map_delta)] = -1.
        #map_delta[np.isnan(map_delta)] = -1.

        return map_delta

    def delta(self):
        """find the overdensity using a separable selection function"""
        delta_data = self.produce_delta_map(self.outfile_data,
                                            self.outfile_separable)

        algebra.save(self.outfile_delta_data, delta_data)

        for mockindex in range(self.params['mock_number']):
            print "mock delta", mockindex
            mockinfile = self.outfile_mock%mockindex
            mockoutfile = self.outfile_delta_mock%mockindex

            delta_mock = self.produce_delta_map(mockinfile,
                                                self.outfile_separable)

            algebra.save(mockoutfile, delta_mock)

if __name__=="__main__":
    
    import os

    tempfile = algebra.make_vect(np.ones(shape=(64,200,140)), 
                                 axis_names=('freq', 'ra', 'dec'))
    tempfile.info['ra_delta']  = -0.05744777707
    tempfile.info['dec_delta'] = 0.05
    tempfile.info['ra_centre'] = 29.0
    tempfile.info['dec_centre'] = -29.5
    tempfile.info['freq_delta'] = -1000000.0
    tempfile.info['freq_centre'] = 1314500000.0

    algebra.save('/mnt/scratch-gl/ycli/2df_catalog/temp/tempfile', tempfile)

    map_dir = '/mnt/scratch-gl/ycli/2df_catalog/map/map_2929.5/'
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    bin2dfparams_init = {
        "infile_data": "/mnt/scratch-gl/ycli/2df_catalog/catalog/real_catalogue_2df.out",
        "infile_mock": "/mnt/scratch-gl/ycli/2df_catalog/catalog/mock_catalogue_2df_%03d.out",
        "outfile_data": map_dir + "real_map_2df.npy",
        "outfile_mock": map_dir + "mock_map_2df_%03d.npy",
        "outfile_deltadata": map_dir + "real_map_2df_delta.npy",
        "outfile_deltamock": map_dir + "mock_map_2df_delta_%03d.npy",
        "outfile_selection": map_dir + "sele_map_2df.npy",
        "outfile_separable": map_dir + "sele_map_2df_separable.npy",
        "template_file": "/mnt/scratch-gl/ycli/2df_catalog/temp/tempfile",
        "mock_number": 100,
        }
    
    Bin2dF(params_dict=bin2dfparams_init).execute(2)
    