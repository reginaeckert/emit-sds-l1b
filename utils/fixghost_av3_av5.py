#! /usr/bin/env python
#
#  Copyright 2020 California Institute of Technology
#
# EMIT Radiometric Calibration code
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov

#import scipy.linalg
import os, sys
import numpy as np
from spectral.io import envi
import json
import logging
import argparse
from scipy.ndimage import gaussian_filter
from math import pow
from fpa import FPA
import ray
import pylab as plt
from scipy.stats import norm
import scipy.io as scio

def find_header(infile):
  if os.path.exists(infile+'.hdr'):
    return infile+'.hdr'
  elif os.path.exists('.'.join(infile.split('.')[:-1])+'.hdr'):
    return '.'.join(infile.split('.')[:-1])+'.hdr'
  else:
    raise FileNotFoundError('Did not find header file')

def box_filter(x,loc,half_width):
    y = np.zeros(len(x))
    dist = np.abs(x-loc)
    y[dist<=half_width] = 1
    return y
    
def build_psf_kernels(psf_params, cols = np.arange(-150,150)):

    #Peak controls relative energy, all 1's is equal
    kernels = np.zeros((len(psf_params['type']),len(cols)))
    cen_col = int(np.floor(len(cols)/2))
    for ii in np.arange(len(psf_params['type'])):

        if psf_params['type'][ii] == 'box':
            lineshape = box_filter(cols,cols[cen_col],psf_params['half_width'][ii])
            lineshape = gaussian_filter(lineshape,psf_params['blur_sigma'][ii])
            lineshape = psf_params['peak'][ii]*lineshape/max(lineshape)
        elif psf_params['type'][ii] == 'norm':
            lineshape = norm.pdf(cols,cols[cen_col],psf_params['blur_sigma'][ii])
            lineshape = lineshape/max(lineshape) * psf_params['peak'][ii]
        kernels[ii,:] = lineshape
    return kernels
    
def build_spatial_psfs(psf_params, fpa, cols=None):
    # weights is 480 x 3
    if cols is None:
        cols=np.arange(fpa.native_columns).astype(int)
    
    spatial_kernels = build_psf_kernels(psf_params=psf_params, cols=cols) # 3xlen(col)
    spatial_psf = psf_params['weights'] @ spatial_kernels
    cen_col = int(np.floor(len(cols)/2))
    spatial_psf[:,cen_col] += 1 #Add the identity #Weight of 1 for the identity

    spatial_psf = spatial_psf/np.sum(spatial_psf,axis=1)[:,np.newaxis] #Each psf energy = 1
    return spatial_psf #applied through 1D fourier transform

def build_spectral_psfs(psf_params, fpa):
    cen_col = int(np.floor(fpa.native_rows/2))
    
    kernels = build_psf_kernels(psf_params=psf_params, cols=np.arange(fpa.native_rows).astype(int)) # 3xlen(col)
    spectral_psf = psf_params['weights'] @ kernels #480 x 480
    spectral_psf[:,cen_col] += 1 #Add the identity #Weight of 1 for the identity
    
    for ii in np.arange(fpa.native_rows):
        spectral_psf[ii,:] = np.roll(spectral_psf[ii,:],ii-cen_col)
    #PSF is no longer symmetric!! Need to apply as spectral_psf @ ghost
    
    #Implement OSF zones
    osf_zones = psf_params['osf_zones']
    spectral_psf[:np.min(osf_zones),:np.min(osf_zones)] = np.eye(np.min(osf_zones)) # Masked regions
    spectral_psf[np.max(osf_zones)+1:,np.max(osf_zones)+1:] = np.eye(fpa.native_rows-np.max(osf_zones)-1) # Masked regions
    spectral_psf[:np.min(osf_zones),np.max(osf_zones)+1:] = 0 # Masked regions
    spectral_psf[np.max(osf_zones)+1:,:np.min(osf_zones)] = 0 # Masked regions
    
    for zone in osf_zones:
        spectral_psf[zone[0]:zone[1]+1,:zone[0]] = 0
        spectral_psf[zone[0]:zone[1]+1,zone[1]+1:] = 0
        spectral_psf[:zone[0],zone[0]:zone[1]+1] = 0
        spectral_psf[zone[1]+1:,zone[0]:zone[1]+1] = 0
    spectral_psf = spectral_psf/np.sum(spectral_psf,axis=1)[:,np.newaxis] #PSF energy = 1
    return spectral_psf #applied through a right-hand matmul (right onto left)
    
# Fourier transform helper functions
def F_1d(x,axis=1):
    return np.fft.ifftshift(np.fft.fft(np.fft.fftshift(x,axes=axis),axis=axis),axes=axis)
    
def iF_1d(x,axis=1):
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(x,axes=axis),axis=axis),axes=axis) 

@ray.remote
def fix_ghost_parallel(frame, fpa, ghostmap, spectral_psf, spatial_psf, center, plot):
  return fix_ghost(frame, fpa, ghostmap, spectral_psf = spectral_psf, spatial_psf = spatial_psf, center=center, plot=plot)


def fix_ghost(frame, fpa, ghostmap, spectral_psf, spatial_psf, center, plot=False):
  ghost = np.zeros(frame.shape)
  rows, cols = frame.shape
  if rows>cols:
      raise IndexError('Misformed frame')

  for col in range(cols):
     tcol = int(center*2 - col)
     if tcol<0 or tcol>=fpa.native_columns:
         continue
     source = frame[:,col]
     target = source[np.newaxis,:] @ ghostmap
     ghost[:, tcol] = target 

  ghost = spectral_psf@ghost
  ghost = np.abs(iF_1d(F_1d(ghost,axis=1)*F_1d(spatial_psf,axis=1),axis=1))
  
  new = frame - ghost

  if plot:
      plt.imshow(frame,vmin=-10,vmax=50)
      plt.figure()
      plt.imshow(ghost,vmin=-10,vmax=50)
      plt.figure()
      plt.imshow(new,vmin=-10,vmax=50)
      plt.show()

  return new


def main():

    description = "Fix spatial and spectral scatter"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--ncpus',default=30)
    parser.add_argument('config')
    parser.add_argument('output')
    args = parser.parse_args()

    fpa = FPA(args.config)

    ray.init()

    infile = envi.open(find_header(args.input))

    if int(infile.metadata['data type']) == 2:
        dtype = np.uint16
    elif int(infile.metadata['data type']) == 4:
        dtype = np.float32
    else:
        raise ValueError('Unsupported data type')

    rows = int(infile.metadata['bands'])
    columns = int(infile.metadata['samples'])
    lines = int(infile.metadata['lines'])
    nframe = rows * columns

    envi.write_envi_header(args.output+'.hdr',infile.metadata)


    with open(fpa.ghost_file,'r') as fin:
        ghost_params = scio.loadmat(fin,squeeze_me=True)
    ghostmap = ghost_params['ghostmap']
    ghost_spatial_blur = build_spatial_psfs(ghost_params,fpa)
    ghost_spectral_blur = build_spectral_psfs(ghost_params,fpa)
    ghost_center = ghost_params['center']

    with open(args.input,'rb') as fin:
      with open(args.output,'wb') as fout:

        frames = []
        for line in range(lines):

            # Read a frame of data
            if line%10==0:
                print('Line '+str(line))

            frame = np.fromfile(fin, count=nframe, dtype=dtype)
            if infile.metadata['interleave'] == 'bil':
                frame = np.array(frame.reshape((rows, columns)),dtype=np.float32)
            elif infile.metadata['interleave'] == 'bip':
                frame = np.array(frame.reshape((columns,rows)),dtype=np.float32).T
            else:
                raise ValueError('unsupported interleave')

            # embed subframe if needed
            frames.append(frame)

            if len(frames) == args.ncpus or line == (lines-1):
                jobs = [fix_ghost_parallel.remote(f, fpa, ghostmap, 
                                     ghost_spectral_blur, ghost_spatial_blur, center=center, plot=args.plot) for f in frames]
                fixed_all = ray.get(jobs)
                for fixed in fixed_all:

                   # remove embedding if needed
                   if infile.metadata['interleave'] == 'bil':
                       np.array(fixed, dtype=np.float32).tofile(fout)
                   elif infile.metadata['interleave'] == 'bip':
                       np.array(fixed.T, dtype=np.float32).tofile(fout)
                   else:
                       raise ValueError('unsupported interleave')
                frames = []

    print('done') 

if __name__ == '__main__':

    main()
