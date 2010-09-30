"""Unit tests for the fitsGBT module and it's main class fitsGBT.Processor."""

import unittest
import copy
import os

import scipy as sp
import numpy.ma as ma

import fitsGBT
import data_block as db
import kiyopy.custom_exceptions as ce

# This fits file generated by the script make_test_GBT_fits_file.py
fits_test_file_name = 'testfile_GBTfits.fits'
# This file has known properties:
IF_set = (695, 725)
scan_set = (113, 114)
pol_set = (-5, -7, -8, -6)
cal_set = ('T', 'F')
nIFs = len(IF_set)
nscans = len(scan_set)
ntimes_scan = 10
npol = len(pol_set)
ncal = len(cal_set)
nfreq = 2048

# The name FileProcessor was inherited from an earlier version of this class.
# It is now not a processor but simple a Reader.
class TestReaderInit(unittest.TestCase) :
    
    def setUp(self) :
        self.FileProcessor = fitsGBT.Reader(fits_test_file_name)

    def test_gets_IFs(self) :
        for ii in range(len(IF_set)) :
            self.assertAlmostEqual(self.FileProcessor.IF_set[ii], IF_set[ii])

    def test_gets_scans(self) :
        for ii in range(len(scan_set)) :
            self.assertEqual(self.FileProcessor.scan_set[ii], scan_set[ii])

    def tearDown(self) :
        del self.FileProcessor

class TestReaderGetIFScanInds(unittest.TestCase) :
    
    def setUp(self) : 
        self.FileProcessor = fitsGBT.Reader(fits_test_file_name)
        self.IFs_all = sp.array(self.FileProcessor.fitsdata.field('CRVAL1')/1E6)
        self.IFs_all  = self.IFs_all.round(0)
        self.IFs_all  = sp.array(self.IFs_all, int)
        self.scans_all = sp.array(self.FileProcessor.fitsdata.field('SCAN'))

    def test_gets_records(self) :
        for scan_ind in range(2) :
            for IF_ind in range(2) :
                inds = self.FileProcessor.get_scan_IF_inds(scan_ind, IF_ind)
                IFs = self.IFs_all[inds]
                scans = self.scans_all[inds]
                # Verify we got all of them.
                self.assertEqual(sp.size(inds), npol*ncal*ntimes_scan)
                # Verify they are all unique.
                self.assertEqual(len(sp.unique(inds)), npol*ncal*ntimes_scan)
                # Check that they are all right.
                wrong_scan = sp.where(sp.not_equal(scans, scan_set[scan_ind]))
                wrong_IF = sp.where(sp.not_equal(IFs, IF_set[IF_ind]))
                self.assertEqual(len(wrong_scan[0]), 0)
                self.assertEqual(len(wrong_IF[0]), 0)

    def test_reforms_records(self) :
        """Test reshaping of indicies to time x pol x cal."""
        
        # Get the inds of a scan and IF and use them to get some data.
        inds = self.FileProcessor.get_scan_IF_inds(1, 1)
        LST = sp.array(self.FileProcessor.fitsdata.field('LST')[inds])
        pol = sp.array(self.FileProcessor.fitsdata.field('CRVAL4')[inds])
        cal = sp.array(self.FileProcessor.fitsdata.field('CAL')[inds])
        # Test that the indicies have the proper shape
        shape_expected = (ntimes_scan, npol, ncal)
        self.assertEqual(shape_expected, sp.shape(inds))
        # Make sure that LST is constant of indicies 1,2.  Etc. for pol, cal.
        aLST = sp.unique(LST[0,:,:])
        self.assertEqual(len(aLST), 1)
        apol = sp.unique(pol[:,0,:])
        self.assertEqual(len(apol), 1)
        acal = sp.unique(cal[:,:,0])
        self.assertEqual(len(acal), 1)

    def test_checks_data_order(self) :
        """Puts pols out of order and check if exception is raised."""

        # Mess up the cals in one of the scans, IFs
        inds = self.FileProcessor.get_scan_IF_inds(1, 1)
        self.FileProcessor.fitsdata.field('CAL')[inds[1,1,1]] = 'T'
        self.FileProcessor.fitsdata.field('CAL')[inds[1,1,0]] = 'T'
        # See if an error is raised when we try to re-get the inds.
        self.assertRaises(ce.DataError, self.FileProcessor.get_scan_IF_inds,
                          1, 1)
        # Mess up the pols in another of the scans, IFs
        inds = self.FileProcessor.get_scan_IF_inds(0, 0)
        self.FileProcessor.fitsdata.field('CRVAL4')[inds[1,1,1]] = '-8'
        self.FileProcessor.fitsdata.field('CRVAL4')[inds[1,2,1]] = '-8'
        self.assertRaises(ce.DataError, self.FileProcessor.get_scan_IF_inds,
                          0, 0)
        # Mess up time ordering in yet another scan, IF.
        inds = self.FileProcessor.get_scan_IF_inds(1, 0)
        self.FileProcessor.fitsdata.field('LST')[inds[4,:,:]] = '100'
        self.FileProcessor.fitsdata.field('LST')[inds[6,:,:]] = '100'
        self.assertRaises(ce.DataError, self.FileProcessor.get_scan_IF_inds,
                          1, 0)
        # Keep times in order but make one them slightly off.
        inds = self.FileProcessor.get_scan_IF_inds(0, 1)
        self.FileProcessor.fitsdata.field('LST')[inds[4,0,0]] = \
            self.FileProcessor.fitsdata.field('LST')[inds[4,0,0]] + 0.01
        self.assertRaises(ce.DataError, self.FileProcessor.get_scan_IF_inds,
                          0, 1)


    def tearDown(self) :
        del self.FileProcessor

class TestReads(unittest.TestCase) :
    """Some basic test for some know properties of the data in the test fits
    file."""
    
    def setUp(self) :
        self.Reader = fitsGBT.Reader(fits_test_file_name)
        self.datashape = (ntimes_scan, npol, ncal, nfreq)
        self.DBlock = self.Reader.read(0, 0)
        self.DBlock.verify()

    def test_reads_valid_data(self) :
        for ii in range(4) :
            self.assertEqual(self.datashape[ii], self.DBlock.dims[ii])
        
    def test_feilds(self) :
        self.assertEqual(self.DBlock.field['SCAN'], 113)
        self.assertEqual(round(self.DBlock.field['CRVAL1']/1e6), 695)
        for ii in range(npol) :
            self.assertEqual(pol_set[ii], self.DBlock.field['CRVAL4'][ii])
        for ii in range(ncal) :
            self.assertEqual(cal_set[ii], self.DBlock.field['CAL'][ii])
        self.assertEqual(self.DBlock.field_formats['CRVAL1'], '1D')
        self.assertEqual(self.DBlock.field_formats['CRVAL4'], '1I')

    def tearDown(self) :
        del self.Reader
        del self.DBlock

class TestMultiRead(unittest.TestCase) :
    """Test each scan and IF is read exactly 1 time be default."""
    
    def setUp(self) :
        self.nscans = len(scan_set)
        self.nIFs = len(IF_set)
        self.Reader = fitsGBT.Reader(fits_test_file_name)
        
    def test_multiple_reads(self) :
        Blocks = self.Reader.read()
        self.assertEqual(len(Blocks), nscans*nIFs)
        # Lists multiplied by two because each scan shows up in 2 IFs.
        scan_list = 2*list(scan_set)
        IF_list = 2*list(IF_set)
        for DB in Blocks :
            DB.verify()
            the_scan = DB.field['SCAN']
            the_IF = int(round(DB.field['CRVAL1']/1e6))
            self.assertTrue(scan_list.count(the_scan))
            scan_list.remove(the_scan)
            self.assertTrue(IF_list.count(the_IF))
            IF_list.remove(the_IF)

class TestWriter(unittest.TestCase) :
    """Unit tests for fits file writer.
    """

    def setUp(self) :
        self.Writer = fitsGBT.Writer()
        self.Reader = fitsGBT.Reader(fits_test_file_name)
        Block = self.Reader.read(0, 0)
        self.Writer.add_data(Block)

    def test_add_data(self) :
        for field_name in fitsGBT.fields_and_axes.iterkeys() :
            self.assertEqual(len(self.Writer.field[field_name]),
                             ntimes_scan*npol*ncal)
        Block = self.Reader.read(1, 0)
        self.Writer.add_data(Block)
        for field_name in fitsGBT.fields_and_axes.iterkeys() :
            self.assertEqual(len(self.Writer.field[field_name]),
                             2*ntimes_scan*npol*ncal)

    def test_error_on_bad_format(self) :
        Block = self.Reader.read(1, 0)
        Block.field_formats['CRVAL1'] = '1I'
        self.assertRaises(ce.DataError, self.Writer.add_data, Block)

    def tearDown(self) :
        del self.Writer
        del self.Reader

class TestCircle(unittest.TestCase) :
    """Circle tests for the reader and writer.

    I'm sure there is a word for it, but I've dubbed a circle test when you
    read some data, do something to it, then write it and read it again.  Then
    check it element by element that it hasn't changed.
    """

    def setUp(self) :
        self.Reader = fitsGBT.Reader(fits_test_file_name)
        self.Blocks = list(self.Reader.read([], []))

    def circle(self) :
        self.BlocksToWrite = copy.deepcopy(self.Blocks)
        self.Writer = fitsGBT.Writer(self.BlocksToWrite)
        self.Writer.write('temp.fits')
        self.newReader = fitsGBT.Reader('temp.fits')
        self.newBlocks = self.newReader.read()

        self.assertEqual(len(self.Blocks), len(self.newBlocks))
        for ii in range(len(self.newBlocks)) :
            OldDB = self.Blocks[ii]
            NewDB = self.newBlocks[ii]
            for jj in range(4) :
                self.assertEqual(OldDB.dims[ii], NewDB.dims[ii])
            self.assertTrue(ma.allclose(OldDB.data, NewDB.data))
            for field, axis in fitsGBT.fields_and_axes.iteritems() :
                self.assertEqual(axis, OldDB.field_axes[field])
                self.assertEqual(axis, NewDB.field_axes[field])
            for field in ['SCAN', 'DATE-OBS', 'OBJECT', 'TIMESTAMP',
                          'OBSERVER'] :
                self.assertEqual(OldDB.field[field], NewDB.field[field])
            for field in ['CRVAL1', 'BANDWID', 'RESTFREQ', 'DURATION',
                          'EXPOSURE'] :
                self.assertAlmostEqual(OldDB.field[field], NewDB.field[field])
            for field in ['LST', 'ELEVATIO', 'AZIMUTH', 'OBSFREQ'] :
                self.assertTrue(sp.allclose(OldDB.field[field], 
                                            NewDB.field[field]))
            for field in ['CRVAL4', 'CAL'] :
                self.assertTrue(all(OldDB.field[field] == NewDB.field[field]))

    def test_basic(self) :
        self.circle()

    def test_masking(self) :
        self.Blocks[1].data[3,2,1,30] = ma.masked
        self.circle()
        self.assertTrue(sp.all(self.Blocks[1].data.mask == 
                            self.newBlocks[1].data.mask))

    def tearDown(self) :
        del self.Reader
        del self.Writer
        del self.newReader
        os.remove('temp.fits')

class TestHistory(unittest.TestCase) :
    """Tests that histories are read, added and written."""

    def setUp(self) :
        # fits_test_file_name has no history.
        self.Reader = fitsGBT.Reader(fits_test_file_name)

    def test_reads_history(self) :
        self.Reader.hdulist[0].header.update('DB-HIST', 'First History')
        self.Reader.hdulist[0].header.update('DB-DET', 'A Detail')
        Block1 = self.Reader.read(0, 0)
        Block = self.Reader.read(0, 1)
        self.assertTrue(Block.history.has_key('000: First History'))
        self.assertEqual(Block.history['000: First History'][0], 'A Detail')
        self.assertTrue(Block.history.has_key('001: Read from file.'))
        self.assertEqual(Block.history['001: Read from file.'][0], 
                        'File name: ' + fits_test_file_name)

    def test_writes_history(self) :
        # Mock up a work flow history
        Block1 = self.Reader.read(0, 0)
        Block2 = self.Reader.read(0, 1)
        Block1.add_history('Processed.', ('Processing detail 1',))
        Block2.add_history('Processed.', ('Processing detail 2',))
        Writer = fitsGBT.Writer((Block1, Block2))
        Writer.write('temp2.fits')
        newReader = fitsGBT.Reader('temp2.fits')
        newBlock = newReader.read(0,0)
        # These two line need to come before anything likly to fail.
        del newReader
        os.remove('temp2.fits')
        # See that we have all the history we expect.
        hist = newBlock.history
        self.assertTrue(hist.has_key('000: Read from file.'))
        self.assertEqual(len(hist['000: Read from file.']), 1)
        self.assertTrue(hist.has_key('001: Processed.'))
        self.assertEqual(len(hist['001: Processed.']), 2)
        self.assertEqual(hist['001: Processed.'][0], 'Processing detail 1')
        self.assertEqual(hist['001: Processed.'][1], 'Processing detail 2')
        self.assertTrue(hist.has_key('002: Written to file.'))
        self.assertEqual(len(hist['002: Written to file.']), 1)
        self.assertEqual(hist['002: Written to file.'][0], 'File name: ' + 
                         'temp2.fits')
        self.assertTrue(hist.has_key('003: Read from file.'))


    def tearDown(self) :
        del self.Reader

        
                

if __name__ == '__main__' :
    unittest.main()

