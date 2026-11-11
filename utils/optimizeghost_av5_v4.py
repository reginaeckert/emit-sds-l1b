# David R Thompson
import argparse, sys, os
import numpy as np
import pylab as plt
from copy import deepcopy
from glob import glob
from spectral.io import envi
from scipy.stats import norm
from scipy.linalg import solve, inv
from astropy import modeling
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import minimize
from scipy.interpolate import BSpline,interp1d
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import json
from scipy.optimize import minimize
from numba import jit
# from fixghost import fix_ghost
# from fixghostraster import build_ghost_matrix, build_ghost_blur
from fixghostraster_av5 import build_ghost_matrix_v2, build_ghost_blur_v2, fix_ghost_v2
from fpa import FPA
import ray

rayargs={'num_cpus':40}
ray.init(**rayargs)


def find_header(infile):
  if os.path.exists(infile+'.hdr'):
    return infile+'.hdr'
  elif os.path.exists('.'.join(infile.split('.')[:-1])+'.hdr'):
    return '.'.join(infile.split('.')[:-1])+'.hdr'
  else:
    raise FileNotFoundError('Did not find header file')


def randomize_ghost_config(config, seed):
  new_config = deepcopy(config)
  if seed is None:
      return config
  rng = np.random.default_rng(int(seed))
  slopes = np.array([f['intensity_slope'] for f in config['orders']])
  offsets = np.array([f['intensity_offset'] for f in config['orders']])
  scalings = np.array([f['scaling'] for f in config['orders']])
  for i in range(len(config['orders'])):
     new_config['orders'][i]['scaling'] = \
       rng.random(None,float) * (scalings.max()-scalings.min()) + scalings.min()
     new_config['orders'][i]['intensity_offset'] = \
       rng.random(None,float) * (offsets.max()-offsets.min()) + offsets.min()
     new_config['orders'][i]['intensity_slope'] = \
       rng.random(None,float) * (slopes.max()-slopes.min()) + slopes.min()
  return new_config


def serialize_ghost_config(config, coarse):
  x, bounds = [],[]
  if coarse==1:
      for i in range(len(config['orders'])):
          x.append(config['orders'][i]['scaling'])
          bounds.append((0,9999))
  elif coarse==2:
      for i in range(len(config['orders'])):
          i0 = config['orders'][i]['extent'][0] * config['orders'][i]['intensity_slope']) + config['orders'][i]['intensity_offset']
          i1 = config['orders'][i]['extent'][1] * config['orders'][i]['intensity_slope']) + config['orders'][i]['intensity_offset']
          x.append(i0)
          bounds.append((0,0.1))
          x.append(i1)
          bounds.append((0,0.1))
  elif coarse==3:
      for i in range(len(config['orders'])):
          x.append(config['orders'][i]['extent'][0])
          bounds.append((x[-1]-2,x[-1]+2))
          x.append(config['orders'][i]['extent'][1])
          bounds.append((x[-1]-2,x[-1]+2))
  else:
      
      for psf_type in ['spectral_psf_zones','spatial_psf_zones']:
          zone_type = config[psf_type]
          for zone in zone_type:
              for psf in zone['psfs']:
                  if psf['type'] == 'norm':
                      x.append(psf['sigma'])
                      bounds.append((0,100))
                      x.append(psf['peak'])
                      bounds.append((0,5))
                  elif psf['type'] == 'box':
                      x.append(psf['half_width'])
                      bounds.append((0,100))
                      x.append(psf['blur_sigma'])
                      bounds.append((0,100))
                      x.append(psf['peak'])
                      bounds.append((0,5))

  return x,bounds    


def deserialize_ghost_config(x, config, coarse):
  ghost_config = deepcopy(config) 
  if coarse==1: 
      for i in range(len(ghost_config['orders'])):
        ghost_config['orders'][i]['scaling'] = x[i]
  elif coarse==2: 
      for i in range(len(ghost_config['orders'])):
        ghost_config['orders'][i]['intensity_slope'] = x[i*2]
        ghost_config['orders'][i]['intensity_offset'] = x[i*2+1]
  elif coarse==3:
      for i in range(len(config['orders'])):
          ghost_config['orders'][i]['extent'][0] = x[i*2]
          ghost_config['orders'][i]['extent'][1] = x[i*2+1]
  else:
      ind = 0
      for psf_zone in ['spectral_psf_zones','spatial_psf_zones']:
          for zone in range(len(ghost_config[psf_zone])):
              for psf in range(len(ghost_config[psf_zone][zone]['psfs'])):
                  if ghost_config[psf_zone][zone]['psfs'][psf]['type'] == 'norm':
                      ghost_config[psf_zone][zone]['psfs'][psf]['sigma'] = x[ind]
                      ghost_config[psf_zone][zone]['psfs'][psf]['peak'] = x[ind+1]
                      ind = ind+2
                  elif ghost_config[psf_zone][zone]['psfs'][psf]['type'] == 'box':
                      ghost_config[psf_zone][zone]['psfs'][psf]['half_width'] = x[ind]
                      ghost_config[psf_zone][zone]['psfs'][psf]['blur_sigma'] = x[ind+1]
                      ghost_config[psf_zone][zone]['psfs'][psf]['peak'] = x[ind+2]
                      ind = ind+3

  return ghost_config   


def frame_error(frame, fpa, ghostmap, spectral_blur,spatial_blur, center):
    try:
        fixed = fix_ghost_v2(frame, fpa, ghostmap, blur_spectral = spectral_blur,spatial_blur=spatial_blur, center=center, plot=False) 
    except IndexError:
         # Something is out of bounds
         return 9e99
    half = int(round(fpa.native_columns/2))
    max_left = np.percentile(frame[:,:half],99)
    max_right = np.percentile(frame[:,half:],99)
    if max_left>max_right:
        return np.mean(pow(fixed[:,half:],2))# / np.mean(pow(frame[:,half:],2)) 
    else:
        return np.mean(pow(fixed[:,:half],2))# / np.mean(pow(frame[:,:half],2))


def err(x, fpa, frames, ghost_config, coarse):
    new_config = deserialize_ghost_config(x, ghost_config, coarse)
    ghostmap = build_ghost_matrix_v2(new_config, fpa)
    spectral_blur,spatial_blur = build_ghost_blur_v2(new_config, fpa)
    center = new_config['center']
    jobs = [frame_error(frame, fpa, ghostmap, spectral_blur,spatial_blur,
          center) for frame in frames]
    errs = np.array(jobs)
    #print(sum(errs))
   #for i,err in enumerate(jobs):
   #    print('frame %i error %10.2f'%(i,err))
    return sum(errs)
 

@ray.remote
def partial(x, i, fpa, frames, ghost_config, base_cost, coarse):
    x_perturb = x.copy()
    if coarse:
        eps = 1e-7
    else:
        eps = 0.001
    x_perturb[i] = x[i] + eps
    perturb_cost = err(x_perturb, fpa, frames, ghost_config, coarse)
    return (perturb_cost - base_cost)/eps


def jac(x, fpa, frames, ghost_config, coarse):
    base_cost = err(x,fpa, frames,ghost_config, coarse)
    jobs  = [partial.remote(x,i,fpa,frames,ghost_config,base_cost, coarse) for i in range(len(x))]
    derivs = ray.get(jobs)
    return np.array(derivs)


def main():

    description = "Optimize ghost model"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('ghost_config')
    parser.add_argument('input_file')
    parser.add_argument('output_config')
    parser.add_argument('--config', default=None)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--method', default='TNC')
    parser.add_argument('--test_file',default=None)
    parser.add_argument('--intermediate_file',default=None)
    # parser.add_argument('input',nargs='+')
    # parser.add_argument('output')
    args = parser.parse_args()

    fpa = FPA(args.config)

    # frames = []
    # for infile in args.input:
    #     I = envi.open(find_header(infile))
    #     frame = np.squeeze(I.load())
    #     if frame.shape[0] > frame.shape[1]:
    #         frame = frame.T
    #     frames.append(frame)
    # frames = np.array(frames)

    frames = envi.open(args.input_file).open_memmap()

    if args.test_file is not None:
        test_frames = envi.open(args.test_file).open_memmap()

    with open(args.ghost_config,'r') as fin:
        ghost_config = json.load(fin)

    #ghost_config = randomize_ghost_config(ghost_config, args.seed)

    coarse_list = ['psf sigma and scaling','overall scaling','intensity parameters','extent of lines']
    # We perform coordinate descent on different state vector subspaces
    for ii,coarse in enumerate([0,2,0,2,1,0,2]):#([3,2,0,3,2]):#[0,3,2,0,3,2,1]): #[3,2,0,1,2,0]):#[0,2,3,0,2]): #[3,1,2,0,1,2,0]:#[2,2]:#[1,0,2,1,0,2,1,0,2]:
        #"Coarse"
        #0 = psf sigma, peak
        #1 = scaling
        #2 = intensity slope, offset
        #3 = extent 
        print(f'iteration {ii}, coarse {coarse} ({coarse_list[coarse]})',flush=True)
        
        # nonlinear solution
        x0, bounds = serialize_ghost_config(ghost_config, coarse=coarse)
        best = minimize(err, x0, args=(fpa, frames, ghost_config, coarse), \
            jac=jac, bounds=bounds, method=args.method)
        best_config = deserialize_ghost_config(best.x, ghost_config, \
            coarse=coarse)
        
        # Print the result to screen
        print(best.nit,'iterations')
        final_error = np.sqrt(err(best.x, fpa, frames, ghost_config, coarse=coarse))/frames.shape[0]
        print('final train error:',final_error)
        if args.test_file is not None:
            test_error = np.sqrt(err(best.x, fpa, test_frames, ghost_config, coarse=coarse))/test_frames.shape[0]
            print('test error:',test_error)
        print(best.message)

        # Record final error
        #xbest, bounds = serialize_ghost_config(best_config, coarse=2)
        best_config['final_error'] = final_error
        if args.test_file is not None:
            best_config['test_error'] = test_error
        
        # Write provisional configuration to the output file
        with open(args.output_config,'w') as fout:
            fout.write(json.dumps(best_config,indent=2))
        if args.intermediate_file is not None:
            intermed_out = f'{args.intermediate_file}_coarse{coarse}_iter{ii}.json'
            with open(intermed_out,'w') as fout:
                fout.write(json.dumps(best_config,indent=2))
            

        # Initialize for the next round
        ghost_config = best_config

if __name__ == '__main__':
    main()
