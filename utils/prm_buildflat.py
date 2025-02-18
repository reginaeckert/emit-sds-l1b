#!/usr/bin/env python
# David R Thompson
# Spring 2015
# Jet Propulsion Laboratory, California Institute of Technology

import os, sys, argparse, time
import spectral
from scipy.optimize import minimize
import pylab as plt
from scipy.interpolate import splrep, splev
from scipy import ndimage
from skimage import filters
import numpy as np
import spectral.io.envi as envi
from numba import jit
from fpa import FPA
from scipy.ndimage import gaussian_filter

def find_header(infile):
  if os.path.exists(infile+'.hdr'):
    return infile+'.hdr'
  elif os.path.exists('.'.join(infile.split('.')[:-1])+'.hdr'):
    return '.'.join(infile.split('.')[:-1])+'.hdr'
  else:
    raise FileNotFoundError('Did not find header file')


def main():

    description = 'Apply a multiplicative flat field (and optionally, an additive offset)';
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('input', type=str, help='Flat field')
    parser.add_argument('--config',type=str)
    parser.add_argument('--max_rdn',default=35.0)
    parser.add_argument('output', help='Output final flat field')
    args = parser.parse_args()
    fpa = FPA(args.config)

    reference_cols = []
    for extrema in fpa.reference_cols:
      # Need to adjust reference columns because code is run on radiance subframe
      reference_cols.extend(np.arange(extrema[0]-fpa.first_distributed_column,
                                      extrema[1]-fpa.first_distributed_column))
    reference_cols = np.array(reference_cols, dtype=int)

    # Define local variables
    print(args.input)
    inhdr  = find_header(args.input)
    img = np.copy(envi.open(inhdr).asarray(writable=False))
    # if np.any(img<-9990):
    #    return 1

    band = np.squeeze(img[:,:,50])
    nbands = img.shape[2]
    nsamps = img.shape[1]

    # Remove edges
    edges = abs(ndimage.sobel(ndimage.gaussian_filter(band, 3)))
    thresh = filters.threshold_otsu(edges)
    thresh = np.percentile(edges,70)
    edge = edges>thresh
    edge = ndimage.binary_dilation(edge)

    # Remove bright pixels (clouds, etc.)
    bright = np.any(img>args.max_rdn,axis=2)
    use = np.logical_or(edge, bright)
    for i in range(img.shape[0]):
        img[i,use[i,:],:] = np.nan

    flat = np.nanmedian(img,axis=0).T # convert to row, column
    new = flat.copy()

    # High pass filter
    blur = gaussian_filter(new,(0.4,2))

    # Remove edge effect
    blur[:,:4] = new[:,:4]
    blur[:,-4:] = new[:,-4:]

    new = new / blur
    flat = new
    for row in range(flat.shape[0]):
       ref = np.nanmedian(flat[row, reference_cols])
       flat[row,:] = ref / flat[row,:]
    flat[np.logical_not(np.isfinite(flat))]= 1

    I = envi.open(inhdr)
    meta = I.metadata.copy()
    outhdr = args.output + '.hdr'
    meta['lines'] = nbands
    meta['samples'] = nsamps
    meta['bands'] = 1
    meta['interleave'] = 'bsq'
    Icorr = envi.create_image(outhdr, meta, force=True, ext="")
    with open(args.output,'wb') as fout:
        np.array(flat, dtype=np.float32).tofile(fout)

if __name__ == '__main__':
    main()
