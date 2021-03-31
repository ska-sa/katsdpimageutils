#!/usr/bin/env python3
import argparse
import logging
import os

from primary_beam_correction import Circular_Beam


def initialize_logs():
    """
    Initialize the log settings
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser("Input a MeerKAT SDP pipeline continuum image and produce "
                                     "primary beam corrected image.")
    parser.add_argument('input',
                        help='MeerKAT continuum uncorrected primary beam fits file')
    parser.add_argument('output', nargs='?',
                        help="The output full path plus file name with '.fits'  extension. "
                        "e.g. /home/name/primary_beam_image.fits. Default: same directory as "
                        "the input image.")
    return parser


def main():
    # Initializing the log settings
    initialize_logs()
    logging.info('MeerKAT SDP continuum image primary beam correction.')
    parser = create_parser()
    args = parser.parse_args()
    path = os.path.abspath(args.input)
    outpath = args.output
    logging.info('----------------------------------------')
    logging.info('Getting the beam pattern for each frequency plane based on the '
                 'katbeam module (https://github.com/ska-sa/katbeam.git).')
    circbeam = Circular_Beam()
    # read in fits image and check the band type
    raw_image = circbeam.read_fits(path)
    if round(raw_image.header['CRVAL3']/1e6) == 1284:  # L-band
        circbeam = Circular_Beam('MKAT-AA-L-JIM-2020')
    if round(raw_image.header['CRVAL3']/1e6) == 816:  # UHF-band
        circbeam = Circular_Beam('MKAT-AA-UHF-JIM-2020')
    bp = circbeam.beam_pattern(path)
    # pbc - primary beam corrected
    logging.info('----------------------------------------')
    logging.info('Doing the primary beam correction in each frequency plane and averaging')
    pbc_image = circbeam.primary_beam_correction(bp, raw_image, px_cut=0.1)
    logging.info('----------------------------------------')
    logging.info('Saving the primary beam corrected image')
    fpath, fext = os.path.splitext(path)
    if outpath is None:
        output_path = fpath + '_PB' + fext
    else:
        output_path = outpath
    circbeam.write_new_fits(pbc_image, path, outputFilename=output_path)
    logging.info('------------------DONE-------------------')


if __name__ == "__main__":
    main()
