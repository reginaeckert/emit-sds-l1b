#! /usr/bin/env python
#
# AVIRIS-3 Radiometric Calibration code

import os, sys, os.path
import logging
import argparse
import json
os.environ['RAY_worker_register_timeout_seconds'] = '600'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import struct

import numpy as np
from spectral.io import envi
import ray
import sys

# Import some AVIRIS-3-specific functions
my_directory, my_executable = os.path.split(os.path.abspath(__file__))
sys.path.append(my_directory + '/utils/')
os.environ['PYTHONPATH'] = my_directory + '/utils/'

from fpa import FPA
from av3_shutter_position import get_shutter_states
from leftshift import left_shift_twice
from fixghostraster import build_ghost_matrix
from fixghostraster import build_ghost_blur
from fixbad import fix_bad
from fixosf import fix_osf
from fixlinearity import fix_linearity
from fixscatter import fix_scatter
from fixghost import fix_ghost
from pedestal import fix_pedestal
from darksubtract import subtract_dark

header_template = """ENVI
description = {{{instrument} L1B calibrated spectral radiance (units: uW nm-1 cm-2 sr-1)}}
samples = {ncolumns}
lines = {lines}
bands = {nchannels}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bil
byte order = 0
wavelength units = nanometers
wavelength = {{{wavelength_string}}}
fwhm = {{{fwhm_string}}}
band names = {{{band_names_string}}}
masked pixel noise = {masked_pixel_noise}
integration time = {integration_time}
bin factor = {bin_factor}
"""

replaced_header_template = """ENVI
description = {{AVIRIS-3 replaced channels}}
samples = {ncolumns}
lines = {lines}
bands = {nreplacedchannels}
header offset = 0
file type = ENVI Standard
data type = 1
interleave = bil
byte order = 0
"""


def find_header(infile):
    if os.path.exists(infile+'.hdr'):
        return infile+'.hdr'
    elif os.path.exists('.'.join(infile.split('.')[:-1])+'.hdr'):
        return '.'.join(infile.split('.')[:-1])+'.hdr'
    else:
        raise FileNotFoundError('Did not find header file')


def parse_integration_time(rawf, instrument = 'AVIRIS-3', rows = 328):

    integration_time  = None

    if instrument == 'AVIRIS-3':
        int_scale = {'013f': 1.,
                     '00a0': .5,
                     '0060': .3,
                     '0020': .1}

    if instrument == 'AVIRIS-5':
        int_scale = {'01d9': 1.}

    line = 0

    with open(rawf,'rb') as f:

        for line in range(100):

            f.seek(line * 1280 * rows * 2 + 324) 	#Byte 324 valid: 0xBABE invalid: 0xDEAD
            data1=f.read(2)
            pp_FIFO_flag = hex(struct.unpack('<H', data1)[0])

            data2=f.read(2)
            pp_FIFO_word_count = struct.unpack('<H', data2)[0] #Byte 326 valid: 36

            if (pp_FIFO_flag == '0xbabe') & (pp_FIFO_word_count == 36):

                f.seek(line * 1280 * rows * 2+359)
                lsb=f"0x{f.read(1)[0]:02x}"[-2:]
                msb=f"0x{f.read(1)[0]:02x}"[-2:]

                tint = f'{msb}{lsb}'

                if tint not in int_scale.keys():
                    logging.warn(f'Integration bytes not recognized in line {line}: {tint}')
                else:
                    integration_time  = int_scale[tint]
                    logging.info(f'Integration time: {integration_time}')
                    break

            else:
                logging.warning(f'Line {line}: Valid ROIC parameters not found')
                logging.warning(f'\tpp_FIFO_flag {pp_FIFO_flag}')
                logging.warning(f'\tpp_FIFO_word_count {pp_FIFO_word_count}' )

    return integration_time

class Config:

    def __init__(self, fpa, mode,integration_time):

        # Load calibration file data
        current_mode   = fpa.modes[mode]

        # Move this outside, to the main function
        if hasattr(fpa,'left_shift_twice') and fpa.left_shift_twice:
            # left shift, returning to the 16 bit range.
            self.dark = left_shift_twice(self.dark)

        if hasattr(fpa,'spectral_calibration_file'):
            _, self.wl_full, self.fwhm_full = \
                 np.loadtxt(fpa.spectral_calibration_file).T * 1000
        else:
            self.wl_full, self.fwhm_full = None, None

        if hasattr(fpa,'srf_correction_file'):
            self.srf_correction = np.fromfile(fpa.srf_correction_file,
                 dtype = np.float32).reshape((fpa.native_rows, fpa.native_rows))
            self.crf_correction = np.fromfile(fpa.crf_correction_file,
                 dtype = np.float32).reshape((fpa.native_columns, fpa.native_columns))
        else:
            self.srf_correction = None
            self.crf_correction = None

        if hasattr(fpa,'bad_element_file'):
            self.bad = np.fromfile(fpa.bad_element_file,
                 dtype = np.int16).reshape((fpa.native_rows, fpa.native_columns))
        else:
            self.bad = np.zeros((fpa.native_rows, fpa.native_columns))

        if 'flat_field_file' in current_mode.keys():
            self.flat_field_file = current_mode['flat_field_file']

            self.flat_field = np.fromfile(self.flat_field_file,
                 dtype = np.float32).reshape((1, fpa.native_rows, fpa.native_columns))
            self.flat_field = self.flat_field[0,:,:]
            self.flat_field[np.logical_not(np.isfinite(self.flat_field))] = 0
        else:
            self.flat_field = None

        if 'radiometric_coefficient_file' in current_mode.keys():
            self.radiometric_coefficient_file = current_mode['radiometric_coefficient_file']
            _, self.radiometric_calibration, self.radiometric_uncert = \
                 np.loadtxt(self.radiometric_coefficient_file).T

            self.radiometric_calibration /= integration_time
        else:
            self.radiometric_calibration, self.radiometric_uncert = None, None

        # zero offset perturbation
        if hasattr(fpa, 'zero_offset_file'):
            self.zero_offset = np.fromfile(fpa.zero_offset_file,
                dtype=np.float32).reshape((1, fpa.native_rows, fpa.native_columns))
        else:
            self.zero_offset = np.zeros((fpa.native_rows, fpa.native_columns))

        # Load ghost configuration and construct the matrix
        if hasattr(fpa,'ghost_map_file'):
            with open(fpa.ghost_map_file,'r') as fin:
                ghost_config = json.load(fin)
                self.ghost_matrix = build_ghost_matrix(ghost_config, fpa)
                self.ghost_blur = build_ghost_blur(ghost_config, fpa)
                self.ghost_center = ghost_config['center']
        else:
            self.ghost_matrix = None
            self.ghost_blur = None
            self.ghost_center = None

        if 'linearity_file' in current_mode.keys():
            self.linearity_file = current_mode['linearity_file']
            self.linearity_map_file = current_mode['linearity_map_file']
            basis = envi.open(self.linearity_file+'.hdr').load()
            self.linearity_mu = np.copy(np.squeeze(basis[0,:]))
            self.linearity_mu[np.isnan(self.linearity_mu)] = 0
            self.linearity_evec = np.copy(np.squeeze(basis[1:,:].T))
            self.linearity_evec[np.isnan(self.linearity_evec)] = 0
            self.linearity_coeffs = envi.open(self.linearity_map_file+'.hdr').load()
        else:
            self.linearity_file = None
            self.linearity_map_file = None
            self.linearity_mu = None
            self.linearity_evec =None
            self.linearity_coeffs = None


BAD_FLAG = -9000

@ray.remote(num_cpus=1)
def calibrate_raw(frames, fpa, config):

    if len(frames.shape) == 2:
        frames = np.reshape(frames,(1,frames.shape[0],frames.shape[1]))

    noises = []
    output_frames = []
    for _f in range(frames.shape[0]):
        frame = frames[_f,...]
        frame = np.copy(frame)
        saturated = np.ones(frame.shape)<0 # False

        # Don't calibrate a bad frame
        if not np.all(frame <= BAD_FLAG):

            # Left shift, returning to the 16 bit range.
            if hasattr(fpa,'left_shift_twice') and fpa.left_shift_twice:
                frame = left_shift_twice(frame)

            # Test for saturation
            if hasattr(fpa,'saturation_DN'):
                saturated = frame>fpa.saturation_DN

            # Dark state subtraction
            frame = subtract_dark(frame, config.dark)
            frame = frame - config.zero_offset

            # Delete telemetry
            if hasattr(fpa,'ignore_first_row') and fpa.ignore_first_row:
                frame[0,:] = frame[1,:]

            # Raw noise calculation
            if hasattr(fpa,'masked_columns'):
                noise = np.nanmedian(np.std(frame[:,fpa.masked_columns],axis=0))
            elif hasattr(fpa,'masked_rows'):
                noise = np.nanmedian(np.std(frame[fpa.masked_rows,:],axis=1))
            else:
                noise = -1

            # Detector corrections
            frame = fix_pedestal(frame, fpa)

            if config.linearity_mu is not None:
                frame = fix_linearity(frame, config.linearity_mu,
                    config.linearity_evec, config.linearity_coeffs)

            if config.flat_field is not None:
                frame = frame * config.flat_field

            # Fix bad pixels, saturated pixels, and any nonfinite
            # results from the previous operations
            flagged = np.logical_not(np.isfinite(frame))

            if fpa.replace_saturation:
                flagged = np.logical_or(saturated,flagged)
            frame[flagged] = 0

            if hasattr(fpa,'bad_element_file'):
                bad = config.bad.copy()
            else:
                bad = np.zeros(frame.shape).astype(int)

            bad[flagged] = -1

            frame = fix_bad(frame, bad, fpa)

            # Optical corrections
            if config.srf_correction is not None:
                frame = fix_scatter(frame, config.srf_correction, config.crf_correction)

            if config.ghost_matrix is not None:
                frame = fix_ghost(frame, fpa, config.ghost_matrix,
                      blur = config.ghost_blur, center = config.ghost_center)

            # # Absolute radiometry
            if config.radiometric_calibration is not None:
                frame = (frame.T * config.radiometric_calibration).T

            # # Fix OSF
            if hasattr(fpa,'osf_seam_positions'):
                frame = fix_osf(frame, fpa)

            # # Catch NaNs
            frame[np.logical_not(np.isfinite(frame))]=0

        if fpa.extract_subframe:

            # Clip the radiance data to the appropriate size
            frame = frame[:,fpa.first_distributed_column:(fpa.last_distributed_column + 1)]
            frame = frame[fpa.first_distributed_row:(fpa.last_distributed_row + 1),:]
            frame = np.flip(frame,axis = (0,1))

            # Clip the replaced channel mask
            bad = bad[:,fpa.first_distributed_column:(fpa.last_distributed_column + 1)]
            bad = bad[fpa.first_distributed_row:(fpa.last_distributed_row + 1),:]
            bad = np.flip(bad,axis = (0,1))

        # Replace all bad data flags with -9999
        cleanframe = frame.copy()
        cleanframe[frame<=(BAD_FLAG+1e-6)] = -9999
        output_frames.append(cleanframe)
        noises.append(noise)

    output_frames = np.stack(output_frames)
    output_frames[output_frames<=(BAD_FLAG+1e-6)] = np.nan

    noises = np.array(noises)
    if np.sum(noises != -9999) > 0:
        noises = np.nanmedian(noises[noises != -9999])
    else:
        noises = -9999

    # Co-add
    output_frames = np.nanmean(output_frames,axis=0)
    output_frames[np.isnan(output_frames)] = -9999

    return output_frames, noises, np.packbits(bad, axis=0)

def main():

    description = "AVIRIS-3/5 Spectroradiometric Calibration"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_file', default='')
    parser.add_argument('config_file', default='')
    parser.add_argument('output_file', default='')
    parser.add_argument('output_replaced', default='')
    parser.add_argument('instrument', choices=['AVIRIS-3', 'AVIRIS-5'])
    parser.add_argument('--binfac', type=str, default=None)
    parser.add_argument('--mode', default = 'default')
    parser.add_argument('--level', default='DEBUG',
                        help='verbosity level: INFO, ERROR, or DEBUG')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--max_jobs', type=int, default=40)
    parser.add_argument('--integration_time', type=float, default=None)
    parser.add_argument('--dark_science_indices', nargs='*', type=int, help='List of starting and ending indices of dark and science lines')

    args = parser.parse_args()

    # Set up logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file is None:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=args.level)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=args.level,
                            filename=args.log_file)

    #Find binfac file if not provided
    if args.binfac is None:
        args.binfac = args.input_file + '.binfac'

        if os.path.isfile(args.binfac) is False:
            logging.error(f'binfac file not found at expected location: {args.binfac}')
            raise ValueError('Binfac file not found - see log for details')

    try:
        binfac = int(args.binfac)
    except:
        binfac = int(np.genfromtxt(args.binfac))

    infile = envi.open(find_header(args.input_file))

    fpa = FPA(args.config_file)

    if args.integration_time is None:

        integration_time = infile.metadata.get('integration time')

        if integration_time:
            integration_time = float(integration_time)
        else:
            logging.info('Integration time not found in raw ENVI header. Checking frame header')
            integration_time = parse_integration_time(args.input_file,
                                                      instrument = args.instrument,
                                                      rows = fpa.native_rows)
            if integration_time is None:
                logging.error(f"Integration time not found in frame header.")
                sys.exit(1)
    else:
        integration_time = args.integration_time

    config = Config(fpa, args.mode, integration_time)

    logging.info('Initializing ray')
    ray.init(num_cpus=args.max_jobs,ignore_reinit_error=True)
    logging.info('Initialization complete, starting calibration')

    raw = 'Start'

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
    nframe = rows * columns * binfac
    noises = []

    if args.dark_science_indices and len(args.dark_science_indices) == 4:
        logging.debug('Using provided science and dark indices')
        dark_start,dark_end,sci_start,sci_end = args.dark_science_indices
        science_frame_idxs = np.arange(sci_start,sci_end)
        dark_frame_idxs = np.arange(dark_start,dark_end)

    elif not args.dark_science_indices:
        logging.debug('Detecting shutter position')
        shutter_pos =  get_shutter_states(args.input_file, rows = fpa.native_rows)
        dark_frame_idxs = np.argwhere(shutter_pos == 0).flatten()
        science_frame_idxs =  np.argwhere(shutter_pos == 2).flatten()
    else:
        logging.error(f"{len(args.dark_science_indices)} indices provided, expecting 4")
        sys.exit(1)

    science_lines = len(science_frame_idxs)

    logging.debug(f'Found {len(dark_frame_idxs)} dark frames and {len(science_frame_idxs)} science frames')
    logging.debug(f'Starting science frame {science_frame_idxs[0]} ')
    logging.debug(f'Ending science frame   {science_frame_idxs[-1]} ')
    binned_lines=  science_lines//binfac
    logging.debug(f'Output binned lines: {binned_lines}')

    if np.all(science_frame_idxs - science_frame_idxs[0] == np.arange(len(science_frame_idxs))) is False:
        logging.error('Science frames are not contiguous, cannot proceed')
        raise AttributeError('Science frames are not contiguous')

    # Read dark
    with open(args.input_file,'rb') as fin:
        fin.seek(dark_frame_idxs[10]*columns*rows*2) # Skip 10 dark frames
        dark_frames = np.fromfile(fin,count=fpa.num_dark_frames_use* fpa.native_columns*fpa.native_rows, dtype=dtype)
        dark_frames = dark_frames.reshape((fpa.num_dark_frames_use,fpa.native_rows, fpa.native_columns))

    config.dark = np.mean(dark_frames,axis=0)
    config.dark_std = np.std(dark_frames,axis=0)

    current_frame = science_frame_idxs[0]
    lines_processed = 0
    noises = []

    with open(args.input_file,'rb') as fin:
        fin.seek(science_frame_idxs[0]*columns*rows*2)

        with open(args.output_file,'wb') as fout:
            with open(args.output_replaced,'wb') as foutreplace:
                raw = np.fromfile(fin, count=nframe, dtype=dtype)
                jobs = []

                while lines_processed < binned_lines:
                    current_frame+=binfac
                    # Read frames of data
                    raw = np.array(raw, dtype=np.float32)
                    frames = raw.reshape((binfac,rows,columns))

                    if lines_processed%10==0:
                        logging.info(f'Calibrating lines {current_frame} - {current_frame+binfac}')

                    jobs.append(calibrate_raw.remote(frames, fpa, config))
                    lines_processed += 1

                    if len(jobs) == args.max_jobs:
                        # Write to file
                        result = ray.get(jobs)
                        for frame, noise,bad in result:
                            np.asarray(frame, dtype=np.float32).tofile(fout)
                            np.asarray(bad, dtype=np.uint8).tofile(foutreplace)
                            noises.append(noise)
                        jobs = []

                    # Read next chunk
                    raw = np.fromfile(fin, count=nframe, dtype=dtype)

                # Do any final jobs
                result = ray.get(jobs)
                for frame, noise,bad in result:
                    np.asarray(frame, dtype=np.float32).tofile(fout)
                    np.asarray(bad, dtype=np.uint8).tofile(foutreplace)
                    noises.append(noise)

    # Form output metadata strings
    wl = config.wl_full.copy()
    fwhm = config.fwhm_full.copy()

    if fpa.extract_subframe:
        ncolumns = fpa.last_distributed_column - fpa.first_distributed_column + 1
        nchannels = fpa.last_distributed_row - fpa.first_distributed_row + 1
        clip_rows = np.arange(fpa.last_distributed_row, fpa.first_distributed_row-1,-1,dtype=int)
        wl = wl[clip_rows]
        fwhm = fwhm[clip_rows]
    else:
        nchannels, ncolumns = fpa.native_rows, fpa.native_columns

    band_names_string = ','.join(['channel_'+str(i) \
        for i in range(len(wl))])
    fwhm_string =  ','.join([str(w) for w in fwhm])
    wavelength_string = ','.join([str(w) for w in wl])

    # Place all calibration parameters in header metadata
    params = {}
    params['masked_pixel_noise'] = np.nanmedian(np.array(noises))
    params['integration_time'] = integration_time
    params['bin_factor'] = binfac
    params['instrument'] = args.instrument

    # Write the header
    params.update(**locals())
    params['lines'] = binned_lines
    with open(args.output_file+'.hdr','w') as fout:
        fout.write(header_template.format(**params))

    # Output the header file for the replaced pixel image
    nreplacedchannels = bad.shape[0]
    params.update(**locals())
    params['lines'] = binned_lines
    with open(args.output_replaced+'.hdr','w') as fout:
        fout.write(replaced_header_template.format(**params))
    logging.info('Done')


if __name__ == '__main__':

    main()
