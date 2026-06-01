#!/usr/bin/env python3
import argparse
import logging
import os

import katsdpimageutils.primary_beam_correction as pbc


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
    raw_image = pbc.read_fits(path)
    beam_model = pbc.get_beam_model(raw_image.header)
    # pbc - primary beam corrected
    logging.info('----------------------------------------')
    logging.info('Doing the primary beam correction in each frequency plane and averaging')
    pbc_image = pbc.primary_beam_correction(beam_model, raw_image, px_cut=0.1)
    logging.info('----------------------------------------')
    logging.info('Saving the primary beam corrected image')
    fpath, fext = os.path.splitext(path)
    if outpath is None:
        output_path = fpath + '_PB' + fext
    else:
        output_path = outpath
    pbc.write_new_fits(pbc_image, path, outputFilename=output_path)
    logging.info('------------------DONE-------------------')


if __name__ == "__main__":
    main()
