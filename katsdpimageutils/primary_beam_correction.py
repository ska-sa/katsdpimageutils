import logging

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy import wcs
from astropy import coordinates

import katbeam


_cosine_taper = katbeam.jimbeam._cosine_taper

# Mapping from 'BANDCODE' to katbeam model name
BAND_MAP = {'L': 'MKAT-AA-L-JIM-2020',
            'UHF': 'MKAT-AA-UHF-JIM-2020',
            'S': 'MKAT-AA-S-JIM-2020'}


def _circular_pattern(x, y, fwhm_x, fwhm_y):
    """Make the beam circular.

    Parameters
    ----------
    x, y : arrays of float of the same shape
       Coordinates where beam is sampled, in degrees
    """
    theta_b = np.sqrt(fwhm_x * fwhm_y)
    r = np.sqrt(x**2 + y**2)
    rr = r/theta_b
    return _cosine_taper(rr)


def read_fits(path):
    """Read in the FITS file.

    Parameters
    ----------
    path : str
        FITS file

    Returns
    -------
    output_file : astropy.io.fits.hdu.image.PrimaryHDU
        First element of the HDU list
    """
    fl = fits.open(path)
    images = fl[0]
    return images


def get_position(header):
    """Determine the sky coordinate of the pointing centre.

    This implementation assumes that the pointing centre is the
    same as the crval.

    Parameters
    ----------
    header : :class:astropy.io.fits.header.Header
        FITS header.

    Returns
    -------
    position : astropy.coordinations.SkyCoord
        Sky coordinate of the phase centre
    image_wcs : astropy.wcs.wcs.WCS
        WCS keywords in the primary HDU
    """
    # Parse the WCS keywords in the primary HDU (only celestial coordinates)
    image_wcs = wcs.WCS(header).celestial
    # Get pointing centre of the observation
    phase_centre_ra = image_wcs.wcs.crval[0]
    phase_centre_dec = image_wcs.wcs.crval[1]
    # Convert to astropy.coordinates.SkyCoord object
    phase_center = coordinates.SkyCoord(phase_centre_ra*u.deg, phase_centre_dec*u.deg)
    return phase_center, image_wcs


def radial_offset(phase_center, image_wcs):
    """Compute radial offset of pixels from the phase centre in degrees.

    Parameters
    ----------
    phase_center : astropy.coordinations.SkyCoord
        phase center position
    image_wcs : astropy.wcs.wcs.WCS
        WCS keywords in the primary HDU
    """
    # Get pixel coordinates
    pixcrd = np.indices(image_wcs.array_shape)
    row = np.ravel(pixcrd[0])
    col = np.ravel(pixcrd[1])
    # Convert pixel coordinates to world coordinates
    p2w = wcs.utils.pixel_to_skycoord(col, row, image_wcs)
    # Compute a separation vector between phase centre and source positions in degrees.
    separation_deg = p2w.separation(phase_center).deg
    return separation_deg


def central_freq(header):
    """Determine central frequency of each frequency plane.

    Parameters
    ----------
    header : :class:`astropy.io.fits.header.Header`
        FITS header

    Returns
    -------
    output : list
        A list of central frequencies in MHz of each frequency plane.
    """
    c_freq_plane = []
    # NSPEC -  number of frequency planes.
    for i in range(1, header['NSPEC']+1):
        # FREQXXXX is the central frequency for each plane XXXX.
        c_freq_plane.append(header['FREQ{0:04}'.format(i)])
    return c_freq_plane


def get_beam_model(header):
    """Get the appropriate beam model for the band determined from the FITS header.

    This uses the katbeam module.
    https://github.com/ska-sa/katbeam.git

    Parameters
    ----------
    header : :class:`astropy.io.fits.header.Header`
        FITS header

    Returns
    -------
    :class:`CircularBeam`
        katbeam model
    """
    band = header.get('BANDCODE')
    if band is None:
        logging.warning('BANDCODE not found in the FITS header. Therefore, frequency ranges'
                        ' are used to determine the band.')
        freqs = central_freq(header)
        start_freq = freqs[0]/1e6
        end_freq = freqs[-1]/1e6
        if start_freq >= 856 and end_freq <= 1712:  # L-band
            band = 'L'
        elif start_freq >= 544 and end_freq <= 1087:  # UHF-band
            band = 'UHF'
        elif start_freq >= 1750 and end_freq <= 3500:  # S-band
            band = 'S'
        # If BANDCODE and frequency ranges fails, the L-band model is returned by default.
        else:
            logging.warning('Frequency ranges do not match. Defalting to L-band frequency range.')
            band = 'L'
    model = BAND_MAP.get(band)
    logging.warning('The {} katbeam model for the {}-band is used.'.format(model, band))
    beam = CircularBeam(model)
    return beam


def power_pattern(x, y, beam, c_freq):
    """Compute Power pattern for a given frequency.

    This uses the katbeam module.
    https://github.com/ska-sa/katbeam.git

    Parameters
    ----------
    x, y : float (or array of float)
       Offsets where beam is sampled, in degrees
    beam : :class:`CircularBeam`
        The beam pattern to sample
    c_freq : list
        List of central frequencies in GHz for each frequency plane
    """
    nu = c_freq/1.e6  # GHz to MHz
    a_b = beam.I(x, y, nu)
    return a_b


def get_offsets(header):
    """Get the offsets of the image pixels from the ref position specified in header.

    Parameters
    ----------
    header : :class:`astropy.io.fits.header.Header`
        FITS header.
    Returns
    -------
    ndarray
        A 2d array of pixel offsets in degrees
    """
    phase_center, image_wcs = get_position(header)
    # Get radial separation between sources and the phase centre as well as make y=0
    # since we are circularising the beam.
    x = radial_offset(phase_center, image_wcs).reshape(image_wcs.array_shape)
    return x


def standard_deviation(data):
    """Compute the median and the estimate of the standard deviation.

    This is based on the median absolute deviation (MAD).
    """
    MAD_TO_SD = 1.4826
    med = np.nanmedian(data)
    dev = np.abs(data - med)
    return med, MAD_TO_SD * np.nanmedian(dev)


def inverse_variance(data):
    """Calculate the inverse variance.

    Reject pixels more than 5 sigma from the median and reject all zeros
    until either no more pixels are rejected or a maximum of 50 iterations
    is reached.
    """
    data = data[(data != 0.0) & (np.isfinite(data))]
    if len(data) == 0:
        return 0.0
    med, sd = standard_deviation(data)
    for i in range(50):
        old_sd = sd
        keep = np.abs(data - med) < 5.0 * sd
        if np.all(keep):
            return 1/(sd)**2
        data = data[keep]
        med, sd = standard_deviation(data)
        if sd == 0.0:
            return 1/(old_sd)**2
    return 1/(sd)**2


def _weighted_accum(data, accum, beam, mask=None):
    """Accumulate the beam corrected data multiplied by its inverse variance

    Parameters
    ----------
    data : 2d ndarray
        The input image data for a single plane
    accum : 2d ndarray
        The output accumulator to sum the weighted and PB correcetd data
    beam : 2d ndarray
        The beam model
    mask : 2d ndarray of boolean (optional)
        A mask to apply to the data array (masked values are set to np.nan)
        Note: nans will propogate into the sum
    Returns
    -------
    weight : float
        The inverse variance of the masked data array
    """
    if mask is not None:
        data[mask] = np.nan
    weight = inverse_variance(data)
    data *= weight
    data /= beam
    accum[:] += data
    return weight


def primary_beam_correction(beam_model, raw_image, px_cut=0.1):
    """Correct the effects of primary beam.

    Parameters
    ----------
    beam_model : :class:`CircularBeam`
        The beam model
    raw_image : astropy.io.fits.hdu.image.PrimaryHDU
        First element of the HDU list
    px_cut : float
       Threshold to cut off all the pixels with attenuated flux less than
       the value.
    """
    nterm = raw_image.header['NTERM']
    nspec = raw_image.header['NSPEC']
    image_data = raw_image.data
    weights = np.empty(nspec, dtype=float)
    # Frequencies are in increasing order (from MGImage)
    freqs = central_freq(raw_image.header)
    # Get an array of pixel offsets in degrees
    x = get_offsets(raw_image.header)
    # Start with the highest frequency to get bounds greater than px_cut
    accum_image = np.zeros(x.shape, dtype=image_data.dtype)
    beam_pattern = power_pattern(x, 0, beam_model, freqs[-1])
    beam_mask = beam_pattern <= px_cut
    weights[-1] = _weighted_accum(image_data[0, -1], accum_image,
                                  beam_pattern, mask=beam_mask)
    # Skip the highest frequency in the loop since we just did it
    for i, this_freq in enumerate(freqs[:-1]):
        # Get the beam pattern for this plane
        beam_pattern = power_pattern(x, 0, beam_model, this_freq)
        # Accumulate the weighted and beam-crrected data
        weights[i] = _weighted_accum(image_data[0, nterm+i], accum_image,
                                     beam_pattern)
    # Normalise the accumulated image
    accum_image[:] /= weights.sum()
    # Add new axes of dimension 1 for FREQ and Stokes
    corr_image = np.expand_dims(accum_image, axis=(0, 1))
    return corr_image


def _get_value_from_history(keyword, header):
    """
    Return the value of a keyword from the FITS HISTORY in header.

    Assumes keyword is found in a line of the HISTORY with format: 'keyword = value'.

    Parameters
    ----------
    keyword : str
          keyword to search for such as BMAJ, CLEANBMJ and BMIN
    header : astropy header
          Image header
    """
    for history in header['HISTORY']:
        line = history.replace('=', ' ').split()
        try:
            ind = line.index(keyword)
        except ValueError:
            continue
        return line[ind + 1]
    raise KeyError(f'{keyword} not found in HISTORY')


def write_new_fits(pbc_image, path, outputFilename):
    """
    Write out a new FITS image with primary beam corrected continuum.
    """
    images = read_fits(path)
    hdr = images.header
    newhdr = hdr.copy()
    # change the frequency plane keywords, we don't want multiple frequency axes
    newhdr['CTYPE3'] = 'FREQ'
    newhdr['NAXIS3'] = 1
    newhdr['CDELT3'] = 1.0
    try:
        if 'CLEANBMJ' in newhdr and newhdr['CLEANBMJ'] > 0:
            # add in required beam keywords
            newhdr['BMAJ'] = newhdr['CLEANBMJ']
            newhdr['BMIN'] = newhdr['CLEANBMN']
            newhdr['BPA'] = newhdr['CLEANBPA']
        else:
            # Check CLEANBMAJ in the history
            newhdr['BMAJ'] = float(_get_value_from_history('BMAJ', newhdr))
            newhdr['BMIN'] = float(_get_value_from_history('BMIN', newhdr))
            newhdr['BPA'] = float(_get_value_from_history('BPA', newhdr))
    except KeyError:
        logging.error('Exception occurred, keywords not found', exc_info=True)
    new_hdu = fits.PrimaryHDU(header=newhdr, data=pbc_image)
    return new_hdu.writeto(outputFilename, overwrite=True)


class CircularBeam(katbeam.JimBeam):
    def HH(self, x, y, freqMHz):
        """Calculate the H co-polarised beam at the provided coordinates.

        Parameters
        ----------
        x, y : arrays of float of the same shape
            Coordinates where beam is sampled, in degrees
        freqMHz : float
            Frequency, in MHz

        Returns
        -------
        HH : array of float, same shape as `x` and `y`
            The H co-polarised beam
        """
        squint, fwhm = self._interp_squint_fwhm(freqMHz)
        return _circular_pattern(x, y, fwhm[0], fwhm[1])

    def VV(self, x, y, freqMHz):
        """Calculate the V co-polarised beam at the provided coordinates.

        Parameters
        ----------
        x, y : arrays of float of the same shape
            Coordinates where beam is sampled, in degrees
        freqMHz : float
            Frequency, in MHz

        Returns
        -------
        VV : array of float, same shape as `x` and `y`
            The V co-polarised beam
        """
        squint, fwhm = self._interp_squint_fwhm(freqMHz)
        return _circular_pattern(x, y, fwhm[2], fwhm[3])
