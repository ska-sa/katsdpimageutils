import logging

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy import wcs
from astropy import coordinates


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


def get_position(path):
    """Determine the sky coordinate of the pointing centre.

    This implementation assumes that the pointing centre is the
    same as the crval.

    Parameters
    ----------
    path : str
        FITS file

    Returns
    -------
    position : astropy.coordinations.SkyCoord
        Sky coordinate of the phase centre
    image_wcs : astropy.wcs.wcs.WCS
        WCS keywords in the primary HDU
    """
    # Parse the WCS keywords in the primary HDU
    image_wcs = wcs.WCS(path)
    # Get pointing centre of the observation
    phase_centre_ra = image_wcs.celestial.wcs.crval[0]
    phase_centre_dec = image_wcs.celestial.wcs.crval[1]
    # Convert to astropy.coordinates.SkyCoord object
    phase_center = coordinates.SkyCoord(phase_centre_ra, phase_centre_dec, unit=(u.deg, u.deg))
    return phase_center, image_wcs


def radial_offset(phase_center, image_wcs):
    """Compute radial offset of pixels from the phase centre in radians.

    Parameters
    ----------
    phase_center : astropy.coordinations.SkyCoord
        phase center position
    image_wcs : astropy.wcs.wcs.WCS
        WCS keywords in the primary HDU
    """
    # Get pixel coordinates
    pixcrd = np.indices((image_wcs.array_shape[2], image_wcs.array_shape[3]))
    row = np.ravel(pixcrd[0])
    col = np.ravel(pixcrd[1])
    # Convert pixel coordinates to world coordinates
    p2w = image_wcs.pixel_to_world(col, row, 0, 0)[0]
    # Compute a separation vector between phase centre and source positions in radians.
    separation_rad = p2w.separation(phase_center).rad
    return separation_rad


def central_freq(path):
    """Determine central frequency of each frequency plane.

    Parameters
    ----------
    path : str
        FITS file

    Returns
    -------
    output : numpy array
        An array of central frequencie in MHz of each frequency plane.
    """
    images = read_fits(path)
    c_freq_plane = []
    # NSPEC -  number of frequency planes.
    for i in range(1, images.header['NSPEC']+1):
        # FREQ00X is the central frequency for each plane X.
        c_freq_plane.append(images.header['FREQ{0:04}'.format(i)])
    return np.array(c_freq_plane)


def cosine_power_pattern(separation_rad, c_freq):
    """Compute Power patterns for a given frequency.

    This uses the Cosine-squared power approximation from
    Mauch et al. (2020).

    Parameters
    ----------
    separation_rad : numpy array
        Radial separation array
    c_freq : numpy array
        An array of central frequencies for each frequency plane
    """
    rho = separation_rad
    # convert degrees to radians
    v_beam_rad = np.deg2rad(89.5 / 60.)
    h_beam_rad = np.deg2rad(86.2 / 60.)
    # Take the Geometric mean for the vertical and horizontal cut through the beam.
    vh_beam_mean = np.sqrt(v_beam_rad * h_beam_rad)
    flux_density = []
    for nu in c_freq:
        # Convert GHz to Hz
        nu = nu/1.e9
        theta_b = vh_beam_mean / nu
        ratio = rho/theta_b
        num = np.cos(1.189 * np.pi * ratio)
        den = 1-4 * (1.189 * ratio)**2
        a_b = (num/den)**2
        flux_density.append(a_b)
    return flux_density


def beam_pattern(path):
    """Make beam pattern.

    Parameters
    ----------
    path : str
        FITS file
    """
    # Read the fits file
    data = read_fits(path)
    # Get the central frequency of the image header given.
    c_freq = central_freq(path)
    # Get radial separation between sources and the phase centre
    phase_center, image_wcs = get_position(path)
    separation_rad = radial_offset(phase_center, image_wcs)
    beam_list = cosine_power_pattern(separation_rad, c_freq)
    return beam_list, data


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
    data = data[data != 0.0]
    if len(data) == 0:
        return 0.0
    med, sd = standard_deviation(data)
    for i in range(50):
        old_sd = sd
        cut = np.abs(data - med) < 5.0 * sd
        if np.all(~cut):
            return 1/(sd)**2
        data = data[cut]
        med, sd = standard_deviation(data)
        if sd == 0.0:
            return 1/(old_sd)**2
    return 1/(sd)**2


def weighted_average(arr, weights):
    """Compute weighted average of all the frequency planes.
    """
    wt_average = np.average(arr, weights=weights, axis=0)
    return wt_average


def primary_beam_correction(beam_pattern, raw_image, px_cut=0.1):
    """Correct the effects of primary beam.

    Parameters
    ----------
    beam_pattern : numpy array
        Array of beam pattern
    raw_image : astropy.wcs.wcs.WCS
        WCS keywords in the primary HDU
    px_cut : float
       Threshold to cut off all the pixels with attenuated flux less than
       the vulue.
    """
    nterm = raw_image.header['NTERM']
    weight = []
    pbc_image = []
    # Get all the pixels with attenuated flux of less than 10% of the peak
    beam_mask = beam_pattern[-1] <= px_cut
    for i in range(len(beam_pattern)):
        # Blank all the pixels with attenuated flux of less than 10% of the peak
        beam_pattern[i][beam_mask] = np.nan
        # Get the inverse variance (weight) in each frequency plane
        # (before primary beam correction)
        weight.append(inverse_variance(np.ravel(raw_image.data[0, i+nterm, :, :])))
        # correct the effect of the beam by dividing with the beam pattern.
        ratio = np.ravel(raw_image.data[0, i + nterm, :, :]) / beam_pattern[i]
        pbc_image.append(ratio)
    # Convert primary beam corrected (pbc) and weight list into numpy array
    pbc_image = np.array(pbc_image)
    weight = np.array(weight)
    # Calculate a weighted average from the frequency plane images
    corr_image = weighted_average(pbc_image, weight)
    # Add new axis
    corr_image = corr_image.reshape(1, 1, raw_image.data.shape[2], raw_image.data.shape[3])
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
    Write out a new FITS image with primary beam corrected continuum in its first plane.
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
