#!/Users/drt/anaconda/bin/python
# David R Thompson
# Panel Ghosting correction

import os
import sys
import spectral
import json
import argparse
from numpy import *
from numpy import linalg

# Return the header associated with an image file
def find_header(imgfile):
    if os.path.exists(imgfile+'.hdr'):
      return imgfile+'.hdr'
    ind = imgfile.rfind('.raw')
    if ind >= 0:
      return imgfile[0:ind]+'.hdr'
    ind = imgfile.rfind('.img')
    if ind >= 0:
      return imgfile[0:ind]+'.hdr'
    raise IOError('No header found for file {0}'.format(imgfile));



# Measured properties of the panel ghost process
def panel_ghost_corr_prism(R_n, params,debug=False):

    # dimensions
    nsamp, nchan, nlines = R_n.shape
    npan = 2*int(ceil(nsamp/2/160))
    if remainder(nsamp,160) != 0:
      print('I need samples to be a multiple of 160')
      sys.exit(1)

    # Electronic panel ghost correction
    # fac_m is size [samples x panels] and represents the contribution of
    # each panel at each pixel

    a,b,c,d,e = [-0.005989099543320,-0.000326169909178,-0.001673673648179,
                  1.527258016207224,-0.000000050214933]
    fac_m=array([[0.00,  0.87,  0.98,  1.01],
                 [0.85,  0.00,  0.98,  1.04],
                 [0.94,  1.01,  0.00,  0.95],
                 [0.97,  1.07,  1.03,  0.00]])
    fac_m = reshape(fac_m.T,[fac_m.size,1])
    fac_m = reshape(tile(fac_m, [1,160]), [npan,nsamp]).T

    if debug:
      print('checkpoint 2:',fac_m[1:10,0:3])

    # Model of EPG row factors depedence on DN
    # f(x) = -(a*exp(b*x)+c)*d+e*x

    alpha_fun= lambda R: (R>0)*(-(a*exp(b*R)+c)*d+e*R)

    if debug:
      print('checkpoint 3',R_n[9,9,9:20])

    R_tmp = R_n
    for i  in range(npan):
      cols = arange(i*160,(i+1)*160)
      this_Rn = R_n[cols,:,:]
      a1 = tile(reshape(fac_m[:,i],[nsamp,1,1]),[1,nchan,nlines])
      a2 = tile(alpha_fun(this_Rn)*this_Rn, [npan,1,1])
      R_tmp = R_tmp + a1 * a2
    R_n = R_tmp

    if debug:
      print('checkpoint 3',R_n[9,9,9:20])

    # Electronic panel shift correction (end of tap spatial boundary column shift)
    # Model of contribution versus pixel # of tap rows 1:160,
    # f(x) = p1*x^6 + p2*x^5 + p3*x^4 + p4*x^3 + p5*x^2 + p6*x + p7

    p1,p2,p3,p4,p5,p6,p7 = [1.30459671940393e-006, -1.14184003323367e-003,
        416.269201225249e-003, -80.9071800991502e+000, 8.84223713360006e+003,
        -515.194624156203e+003, 12.5026157276662e+006];
    zr_i=137.35;   zr_sh=2;
    x = column_stack(linspace(zr_i,159,int(160-round(zr_i)-zr_sh+1)));
    beta_val = p1*power(x,6) + p2*power(x,5) + \
               p3*power(x,4) + p4*power(x,3) + \
               p5*power(x,2) + p6*x + p7;
    beta_val=concatenate((zeros((int(round(zr_i)-1+zr_sh),1)),
                         beta_val.T-beta_val[0,0]), axis=0);
    beta_val=beta_val-beta_val[int(round(zr_i))];
    beta_val=beta_val/sum(beta_val);

    if debug:
      print('checkpoint 4',beta_val[-1,-1])

    # Model of tap boundary shift dependence on DN,
    # f(x) = a*(x/b).^(k/2-1).*exp(-x/b/2)+c-d*x

    a,b,c,d,k=array([0.000004874499497,1.350376781282247,0.000001587131931,
                0.000000000039901,0.002145073519241])*1e3;
    tbfun = lambda R: (R>0)*(a*power(R/b,k/2-1)*exp(-R/b/2)+c-d*R)

    fac_m=array([[ 1.00,  0.98,  0.98,  1.15],
                 [ 0.78,  0.95,  0.93,  1.15],
                 [ 0.65,  0.72,  0.92,  1.10],
                 [ 0.52,  0.68,  0.82,  1.05]])*1.05;
    fac_m = reshape(fac_m.T,[fac_m.size,1])
    fac_m = reshape(tile(fac_m, [1,160]), [npan,nsamp]).T

    if debug:
      print('checkpoint 4.5',R_n[9,9,9:20])

    R_tmp = R_n
    for i  in range(npan):
      cols = arange(i*160,(i+1)*160)
      chans = arange(nchan-1)
      beta_cum = tile(reshape(beta_val,[160,1,1]), [1,nchan-1,nlines])
      beta_cum = beta_cum * tbfun(R_n[cols,:-1,:])
      beta_cum = beta_cum * R_n[cols,:-1,:]
      beta_cum = tile(concatenate((zeros((1,1,nlines)),
                        reshape(sum(beta_cum,axis=0),[1,nchan-1,nlines])),axis=1),
                      [nsamp,1,1])
      R_tmp = R_tmp - tile(reshape(fac_m[:,i],[nsamp,1,1]),[1,nchan,nlines]) * beta_cum
    R_n = R_tmp

    if debug:
      print('checkpoint 5',R_n[9,9,9:20])

    # ELECTRONIC PANEL DISCONTINUITY CORRECTION
    # Model of shape of edge discontinuity
    # f(x) = (p1*x^2 + p2*x + p3) / (x^2 + q1*x + q2)

    # form beta array one block at a time
    beta = {}
    for k,v in params.items():
      p1,p2,p3,q1,q2 = v['coeff']
      zr_i  = v['zr_i']
      zr_sh = v['zr_sh']
      st_sh = v['st_sh']
      alpha_1 = v['alpha_1']
      ind_1 = v['ind_1']
      alpha_2 = v['alpha_2']
      ind_2 = v['ind_2']

      linspace_start = v['linspace_start']
      x = c_[linspace(linspace_start,float(zr_i),int(round(zr_i)+zr_sh-st_sh))];
      beta_val=(p1*power(x,2) + p2*x + p3)/(power(x,2) + q1*x + q2);

      # conditional concatenation
      q = reshape(alpha_2*beta_val[ind_2], [1,1])
      if st_sh>0:
        q = r_[ones((st_sh,st_sh))*beta_val[ind_1]*alpha_1, q]

      beta[k] = concatenate((q, reshape(beta_val[1:],[size(beta_val)-1,1]),
                             zeros((int(160-round(zr_i)-zr_sh),1))), axis=0);

    beta_m=r_[c_[beta['11'], beta['21'], beta['31'], beta['41']],
              c_[beta['12'], beta['22'], beta['32'], beta['41']],
              c_[beta['12'], beta['23'], beta['33'], beta['41']],
              c_[beta['14'], beta['21'], beta['34'], beta['41']]]

    if debug:
      print('checkpoint 6')
      for i in beta_m:
        print(i)

    beta_Rfun = lambda R: sum((R>0)*R,axis=0)

    fac_m=array([[0.80, 0.89, 0.90, 1.03],
                 [0.70, 0.88, 0.85, 0.94],
                 [0.71, 0.78, 0.90, 0.93],
                 [0.71, 0.82, 0.80, 0.92]])
    fac_m = reshape(fac_m.T,[fac_m.size,1])
    fac_m = reshape(tile(fac_m, [1,160]), [npan,nsamp]).T
    R_tmp = R_n
    for i  in range(npan):
      cols = arange(i*160,(i+1)*160)
      cumul =  tile(reshape(fac_m[:,i],[nsamp,1,1]),[1,nchan,nlines])
      cumul = cumul * (tile(reshape(beta_m[:,i],[nsamp,1,1]),[1,nchan,nlines])*
                       tile(beta_Rfun(R_n[cols,:,:]),[nsamp,1,1]))
      R_tmp = R_tmp + cumul
    R_n = R_tmp

    if debug:
      print('checkpoint 7',R_n[9,9,9:20])

    return R_n


# parse the command line (perform the correction on all command line arguments)
def main():


  parser = argparse.ArgumentParser(description='Panel ghost correction.')
  parser.add_argument('input',  type=str, nargs='+',
                   help='files to correct (output will append _pgcor)')
  parser.add_argument('-c', '--config', action='store', default=default_config,
                   help='JSON configuration file')
  args = parser.parse_args()

  for infile in args.input:

    # input checking
    if not os.path.exists(infile):
      print('I could not find '+infile)
      parser.print_help()
      sys.exit(-1)
    if not os.path.exists(args.config):
      print('I could not find '+args.config)
      parser.print_help()
      sys.exit(-1)

    # load configuration file
    with open(args.config,'r') as f:
      params=json.loads(f.read())

    # output filenames
    inhdr = find_header(infile)
    outfile = infile+'_pgcor'
    outhdr = infile+'_pgcor.hdr'
    print('writing to '+outfile)

    # process the image in chunks
    img = spectral.io.envi.open(inhdr, infile)
    out = spectral.io.envi.create_image(outhdr, img.metadata,
                ext='', force=True)
    chunksize = 50
    intervals = [(i, min(i+chunksize, img.nrows)) \
                for i in range(0, img.nrows, chunksize)]
    if img.interleave != 1:
        raise IOError('bad image interleave: %i'%img.interleave)

    for istart, iend in intervals:

      print('Processing interval from row %i to %i'%(istart, iend))

      # we delete the old objects to flush everything to disk, empty cache
      del img
      img = spectral.io.envi.open(inhdr, infile)
      inmm = img.open_memmap(interleave='source', writable=False)
      bil = array(inmm[istart:iend, :, :], dtype=img.dtype)
      trp = transpose(bil,[2,1,0])
      trp = panel_ghost_corr_prism(trp, params)
      bil = transpose(trp,[2,1,0])

      del out
      out = spectral.io.envi.open(outhdr, outfile)
      outmm = out.open_memmap(interleave='source', writable=True)
      outmm[istart:iend, :, :] = bil[:]

if __name__ == "__main__":
  main()
