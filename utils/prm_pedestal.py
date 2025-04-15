#! /usr/bin/env python
#
#  Copyright 2020 California Institute of Technology
#
# EMIT Radiometric Calibration code
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov

import scipy.linalg
import os, sys
import numpy as np
from spectral.io import envi
import json
import logging
import argparse
from fpa import FPA, frame_embed, frame_extract



def find_header(infile):
  if os.path.exists(infile+'.hdr'):
    return infile+'.hdr'
  elif os.path.exists('.'.join(infile.split('.')[:-1])+'.hdr'):
    return '.'.join(infile.split('.')[:-1])+'.hdr'
  else:
    raise FileNotFoundError('Did not find header file')


def fix_pedestal(frame, fpa):


    # Measured properties of the dark pedestal shift curve
    curve = np.array([[41.8456, 35.3747, 30.5247, 26.6662, 23.5052, 20.9135, 18.7087, 16.8540, 15.1186, 13.7943]]).T
    curve_ind = np.arange(275,285,dtype='int')+1 - 1 # zero-indexed
    ncurve = len(curve)

    npan = int(2*np.ceil(fpa.native_columns/2/160)) # four panels in standard PRISM image

    X = np.concatenate((np.ones((ncurve,1)), -1/curve), axis=1)
    Xt = np.linalg.pinv(X)  # calculate matrix inverse
    di = 1 # index of d_sh term


    R_n = np.zeros(frame.T.shape, dtype=np.float32)

    for i in np.arange(np.floor(npan/2.0)):

        # indices of the two panels we will average over (one above center, the other below)
        # code below avoids using top and bottom 30 field rows,
        # since they are affected by edge effects
        ctr = fpa.native_columns/2

        i1 = np.arange(max(ctr-160*(i+1), 30), ctr-160*i, dtype='int')
        print('panel 1: samples %i->%i inclusive'%(min(i1),max(i1)))
        np_pan1 = len(i1)

        i2 = np.arange(ctr+160*i, min(ctr+160*(i+1), fpa.native_columns-30), dtype='int')
        print('panel 2: samples %i->%i inclusive'%(min(i2),max(i2)))
        np_pan2 = len(i2)

        # even indices
        i1e= (i1%2 == 0)
        i2e= (i2%2 == 0)

        # panel above center calculation
        y1 = frame.T[np.ix_(i1, curve_ind)]
        Y1 = y1.reshape(np_pan1, ncurve) / curve.T
        d1 = np.dot(Y1,Xt[di,:].T).reshape(1, np_pan1)
        d1e = d1[:,i1e].mean(axis=1)
        d1o = d1[:,np.logical_not(i1e)].mean(axis=1)

        # panel below center calculation
        y2 = frame.T[np.ix_(i2, curve_ind)]
        Y2 = y2.reshape(np_pan2, ncurve) / curve.T
        d2 = np.dot(Y2,Xt[di,:].T).reshape(1, np_pan2)
        d2e = d2[:,i2e].mean(axis=1)
        d2o = d2[:,np.logical_not(i2e)].mean(axis=1)

        # indexes of odd/even d_sh for panels calculated
        i1=np.arange(max(ctr-160*(i+1), 0), fpa.native_columns/2-160*i, dtype='int')
        i2=np.arange(ctr+160*i, min(ctr+160*(i+1), fpa.native_columns), dtype='int')
        i1e= (i1%2 == 0)
        i1o= np.logical_not(i1e)
        i2e= (i2%2 == 0)
        i2o= np.logical_not(i2e)

        # populate d_sh matrix
        R_n[i1[i1e],:] = np.tile(d1e.reshape(1,1), (len(i1[i1e]),fpa.native_rows))
        R_n[i1[i1o],:] = np.tile(d1o.reshape(1,1), (len(i1[i1o]),fpa.native_rows))
        R_n[i2[i2e],:] = np.tile(d2e.reshape(1,1), (len(i2[i2e]),fpa.native_rows))
        R_n[i2[i2o],:] = np.tile(d2o.reshape(1,1), (len(i2[i2o]),fpa.native_rows))


    return frame+R_n.T


def main():

    description = "Fix pedestal shift for a data cube"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input')
    parser.add_argument('config')
    parser.add_argument('output')
    args = parser.parse_args()

    fpa = FPA(args.config)

    infile = envi.open(find_header(args.input))

    if int(infile.metadata['data type']) == 2:
        dtype = np.int16
    elif int(infile.metadata['data type']) == 12:
        dtype = np.uint16
    elif int(infile.metadata['data type']) == 4:
        dtype = np.float32
    else:
        raise ValueError('Unsupported data type')
    if infile.metadata['interleave'] != 'bil':
        raise ValueError('Unsupported interleave')


    rows = int(infile.metadata['bands'])
    columns = int(infile.metadata['samples'])
    lines = int(infile.metadata['lines'])
    nframe = rows * columns


    metadata = infile.metadata.copy()
    metadata['data type'] = 4
    envi.write_envi_header(args.output+'.hdr', metadata)

    with open(args.input,'rb') as fin:
      with open(args.output,'wb') as fout:

        for line in range(lines):

            # Read a frame of data
            if line%10==0:
                logging.info('Line '+str(line))
            frame = np.fromfile(fin, count=nframe, dtype=dtype)
            frame = np.array(frame.reshape((rows, columns)),dtype=np.float32)
            fixed = fnp.ix_pedestal(frame, fpa)
            np.array(fixed, dtype=np.float32).tofile(fout)

    print('done')

if __name__ == '__main__':

    main()
