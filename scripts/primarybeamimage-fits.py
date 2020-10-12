import argparse
import os
import logging
import numpy as np
import primary_beam_correction as pbc

def main():
    # Initializing the log settings
    pbc.intialize_logs()
    logging.info('MeerKAT SDP continuum image primary beam correction.')
    parser =  pbc.create_parser()
    args = parser.parse_args()
    path = os.path.abspath(args.input)
    logging.info('----------------------------------------')
    logging.info('Getting the beam pattern for each frequecy plane based on the '
                 'Cosine-squared power approximation from Mauch et al. (2020).')
    bp = pbc.beam_pattern(path)
    # pbc - primary beam corrected
    logging.info('----------------------------------------')
    #logging.info('Doing the primary beam correction in each frequency plane and averaging')
    pbc_image = pbc.primary_beam_correction(bp, path)
    # Saving the primary beam corrected image
    logging.info('----------------------------------------')
    logging.info('Saving the primary beam corrected image')
    head, tail = os.path.split(path)
    fname, fext = tail.split('.')
    outputpath = (head+fname+'_PB'+fext)
    pbc.write_new_fits(pbc_image, path, outputFilename=outputpath)
    logging.info('------------------DONE-------------------')


if __name__ == "__main__":
    main()     
