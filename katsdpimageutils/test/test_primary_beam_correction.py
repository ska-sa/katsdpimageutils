"""Tests for :mod:`katsdpimageutils.primary_beam_correction`."""
import shutil
import os

import numpy as np
from astropy.io import fits
from nose.tools import assert_equal

from ..primary_beam_correction import _get_value_from_history, weighted_average, inverse_variance, standard_deviation, write_new_fits # noqa E501 


class TestStandardDeviation():
    def setup(self):
        self.arr = np.array([2, 4, 6, 8, 10]).astype(float)

    def test_nonans(self):
        data = self.arr
        med, sd = standard_deviation(data)
        assert_equal([med, sd], [6.0, 2.9652])

    def test_nans(self):
        data = self.arr
        data[3] = np.nan
        med, sd = standard_deviation(data)
        assert_equal([med, sd], [5.0, 2.9652])

    def test_allnan(self):
        data = self.arr
        data[:] = np.nan
        with np.testing.assert_warns(RuntimeWarning):
            med, sd = standard_deviation(data)


class TestInverseVariance():
    def setup(self):
        self.data = np.array([2, 4, 6, 8, 10]).astype(float)

    def test_all_zero(self):
        data = self.data
        data[:] = 0.0
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 0.0)

    def test_all_nans(self):
        data = self.data
        data[:] = np.nan
        with np.testing.assert_warns(RuntimeWarning):
            inverse_variance(data)

    def test_nonans(self):
        data = self.data
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 1/(2.9652)**2)

    def test_cut_5sd(self):
        data = self.data
        data[-1] = 30
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 1/(2.9652)**2)


class TestGetValueFromHistory:
    def setup(self):
        hdr = fits.Header()
        hdr['history'] = 'AIPS   CLEAN BMAJ=  5e-03 BMIN=  1e-03 BPA=  42.11'
        self.header = hdr

    def test_returnskey(self):
        hdr = self.header
        bpa = _get_value_from_history('BPA', hdr)
        assert_equal(bpa, '42.11')

    def test_nokey(self):
        with np.testing.assert_raises(KeyError):
            _get_value_from_history('CLEANMJ', self.header)


class TestWeightedAverage:
    def setup(self):
        self.data = np.ones([2, 3])
        self.weights = 0.2 * np.arange(1, 3)

    def test_nonans(self):
        result = weighted_average(self.data, self.weights)
        np.testing.assert_array_equal(result, np.array([1, 1, 1]))

    def test_nans(self):
        data = self.data
        data[0, 2] = np.nan
        result = weighted_average(self.data, self.weights)
        np.testing.assert_array_equal(result, np.array([1, 1, np.nan]))


class TestWriteNewFits:
    def setup(self):
        in_array = np.ones([1, 10, 20, 20])
        hdu = fits.PrimaryHDU(in_array)
        hdu.header['history'] = 'AIPS   CLEAN BMAJ=  5e-03 BMIN=  1e-03 BPA=  42.11'
        if os.path.isfile('/tmp/testdir/new.fits'):
            pass
        else:
            os.mkdir("/tmp/testdir")
            hdu.writeto('/tmp/testdir/new.fits')

    def tearDown(self):
        shutil.rmtree("/tmp/testdir")

    def test_case(self):
        out_array = 2 * np.ones([1, 1, 20, 20])
        write_new_fits(out_array, '/tmp/testdir/new.fits', '/tmp/testdir/new_pbc.fits')
        assert os.path.isfile('/tmp/testdir/new_pbc.fits')
