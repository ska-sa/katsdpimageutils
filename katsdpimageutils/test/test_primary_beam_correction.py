"""Tests for :mod:`katsdpimageutils.primary_beam_correction`."""
import os

import numpy as np
import tempfile
from astropy.io import fits
from nose.tools import assert_equal

from ..primary_beam_correction import (_get_value_from_history,
                                       inverse_variance,
                                       standard_deviation,
                                       write_new_fits,
                                       _weighted_accum)


class TestStandardDeviation:
    def setup(self):
        self.arr = np.array([2, 4, 6, 8, 10]).astype(float)
        self.MAD_TO_SD = 1.4826

    def test_nonans(self):
        data = self.arr
        MAD_TO_SD = self.MAD_TO_SD
        med, sd = standard_deviation(data)
        assert_equal([med, sd], [6.0, 2.0 * MAD_TO_SD])

    def test_nans(self):
        data = self.arr
        data[3] = np.nan
        MAD_TO_SD = self.MAD_TO_SD
        med, sd = standard_deviation(data)
        assert_equal([med, sd], [5.0, 2.0 * MAD_TO_SD])


class TestInverseVariance:
    def setup(self):
        self.data = np.array([2, 4, 6, 8, 10]).astype(float)
        self.MAD_TO_SD = 1.4826

    def test_all_zero(self):
        data = self.data
        data[:] = 0.0
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 0.0)

    def test_all_nans(self):
        data = self.data
        data[:] = np.nan
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 0.0)

    def test_mix_nans_zeros(self):
        data = np.zeros([5])
        data[0:2] = np.nan
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 0.0)

    def test_nonans(self):
        data = self.data
        MAD_TO_SD = self.MAD_TO_SD
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 1/(2.0 * MAD_TO_SD)**2)

    def test_cut_5sd(self):
        data = self.data
        MAD_TO_SD = self.MAD_TO_SD
        data[-1] = 30
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 1/(2.0 * MAD_TO_SD)**2)

    def test_reject_zeros(self):
        data = self.data
        MAD_TO_SD = self.MAD_TO_SD
        data[2:4] = 0.0
        inv_var = inverse_variance(data)
        assert_equal(inv_var, 1/(2.0 * MAD_TO_SD)**2)


class TestGetValueFromHistory:
    def setup(self):
        hdr = fits.Header()
        hdr['history'] = 'AIPS   CLEAN BMAJ=  5e-03 BMIN=  1e-03 BPA=  42.11'
        self.header = hdr

    def test_returnskey(self):
        hdr = self.header
        bpa = _get_value_from_history('BPA', hdr)
        bmin = _get_value_from_history('BMIN', hdr)
        bmaj = _get_value_from_history('BMAJ', hdr)
        assert_equal(bpa, '42.11')
        assert_equal(bmin, '1e-03')
        assert_equal(bmaj, '5e-03')

    def test_nokey(self):
        with np.testing.assert_raises(KeyError):
            _get_value_from_history('CLEANMJ', self.header)


class TestWeightedAverage:
    def setup(self):
        self.data = np.array([10., 25., 30., 60., 100.])
        self.beam = np.ones(5, dtype=float)
        self.out = np.zeros(5, dtype=float)
        self.MAD_TO_SD = 1.4826

    def test_nonans(self):
        data = self.data.copy()
        result = _weighted_accum(data, self.out, self.beam)
        expected_result = 1/(20.0 * self.MAD_TO_SD)**2
        assert_equal(result, expected_result)
        np.testing.assert_array_equal(self.out, self.data*expected_result)

    def test_accumulation(self):
        data = self.data.copy()
        result_one = _weighted_accum(data, self.out, self.beam)
        first_out = self.out.copy()
        data = self.data.copy()
        result_two = _weighted_accum(data, self.out, self.beam)
        assert_equal(result_one, result_two)
        np.testing.assert_array_equal(self.out, 2.0*first_out)

    def test_nans(self):
        data = self.data.copy()
        data[0] = np.nan
        data[1] = np.nan
        in_data = data.copy()
        result = _weighted_accum(data, self.out, self.beam)
        expected_result = 1/(30.0 * self.MAD_TO_SD)**2
        assert_equal(result, expected_result)
        np.testing.assert_array_equal(self.out, in_data*expected_result)

    def test_mask(self):
        data = self.data.copy()
        in_data = data.copy()
        in_data[3] = np.nan
        in_data[4] = np.nan
        mask = np.array([False, False, False, True, True])
        result = _weighted_accum(data, self.out, self.beam, mask=mask)
        expected_result = 1/(5.0 * self.MAD_TO_SD)**2
        assert_equal(result, expected_result)
        np.testing.assert_array_equal(self.out, in_data*expected_result)

    def test_allmasked(self):
        data = self.data.copy()
        mask = np.array([True, True, True, True, True])
        result = _weighted_accum(data, self.out, self.beam, mask=mask)
        assert_equal(result, 0.0)
        np.testing.assert_array_equal(self.out, np.nan)


class TestWriteNewFits:
    def setup(self):
        # create temporary directory
        self.tmpdir = tempfile.TemporaryDirectory()
        in_array = np.ones([1, 10, 20, 20])
        hdu = fits.PrimaryHDU(in_array)
        hdu.header['history'] = 'AIPS   CLEAN BMAJ=  5e-03 BMIN=  1e-03 BPA=  42.11'
        self.inFileName = os.path.join(self.tmpdir.name, 'image.fits')
        hdu.writeto(self.inFileName)

    def teardown(self):
        self.tmpdir.cleanup()

    def test_case(self):
        pbc_array = 2 * np.ones([1, 1, 20, 20])
        tmpdir = self.tmpdir
        inFileName = self.inFileName
        outFileName = os.path.join(tmpdir.name, 'image_pbc.fits')
        write_new_fits(pbc_array, inFileName, outFileName)
        assert os.path.isfile(outFileName)
        PBC_hdu = fits.open(outFileName)[0]
        # Check if the data in the PBC HDU is the same as the input pbc array.
        np.testing.assert_array_equal(pbc_array, PBC_hdu.data)
        # Check if the keywords update are stored correctly.
        hdr = PBC_hdu.header
        assert_equal(hdr['CTYPE3'], 'FREQ')
        assert_equal(hdr['NAXIS3'], 1)
        assert_equal(hdr['CDELT3'], 1.0)
