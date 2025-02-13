
import matplotlib.pyplot as plt

import scipy.io as scio
import numpy as np
import os
import fnmatch
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import medfilt
from spectral.io import envi

import astropy.modeling as modeling


# Fit a gaussian to a single peak in an ordered series of data
def find_peak(x, plot=False):
    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude=np.max(x),
                                       mean=np.argmax(x),
                                       stddev=1.0/2.35)   # depending on the data you need to give some initial values
    fitted_model = fitter(model, np.arange(len(x)), x)
    if plot:
        print(fitted_model.mean[0], fitted_model.amplitude[0], fitted_model.stddev[0])
        plt.plot(x)
        plt.plot(fitted_model(np.arange(len(x))))
        plt.show()
    return fitted_model.mean[0], fitted_model.amplitude[0], fitted_model.stddev[0]


def main():
    data_dir = '/beegfs/scratch/drt/20230608_AVIRIS3_LaserSphere/'
    data_filename = '20221212_Laser_Sphere_darksub_pedestal.hdr'
    
    out_dir = '/beegfs/scratch/reckert/develop/aviris3/data/'
    metadata_dir = out_dir
    output_file = '20221212_Laser_Sphere_spectralCal_allCols'
    
    
    # Read in data
    I = envi.open(data_dir + data_filename)
    I = I.load()
    print(I.shape)

    # # Load in modeled wavelength map
    # model_df = pd.read_csv(metadata_dir + 'aviris3_dispersion_map.csv')
    # zemax_pixel     = np.array(model_df['Pixel #'])-1 #Subtract 1 because these are 1-indexed, not 0-indexed
    # zemax_wvl_ideal = np.array(model_df['Desired Wave. (nm)'])
    # zemax_wvl_fit   = np.array(model_df['Wave Fit [6th Poly]'])
    # zemax_spectral_smpl = np.array(model_df['Spectral Sampling'])
    
    # AVIRIS-3
    # Laser info from E:\CWIS-2\TV5\20211210 Laser Sphere\CWIS2-spectral cal -- Dec-10-2021.xlsx on 137.79.132.212
    # wavelengths = np.array([2064.350098, 1550.599976, 1064, 632.8300171, 532, 406.7])
    # Merging Christine's measurements with the CWIS-2 metadata:
    laser_wvl_mean = np.array([2064.350098, 1550.599976, 1064.00, 632.8300171, 532.19, 406.7])
    laser_wvl_std  = np.array([1, 0.1, 0.007, 0.026, 0.079, 0.077]) 
    laser_pixel    = np.array([83,152,218,275,289,306]) 
    nlasers   = len(laser_wvl_mean)
    
    longest_wvl_guess = 2500

    fit_wvl_idx = [np.arange(nlasers).tolist()]
    #fit_wvl_idx.extend([[a for a in np.arange(nlasers) if a != y] for y in np.arange(nlasers)])
    output_tag_list = ['allwavelengths']
    #output_tag_list.extend([f'exclude{x:.0f}nm' for x in laser_wvl_mean])
    
    # Find the spatial location of the lasers, and fit each laser peak
    # put them as column, row, wavelength triplets into the "observed" list
    margin = 5
    if True:
        # our list of laser fits, one sublist per laser
        observed = [[] for c in laser_pixel]
        for line in range(I.shape[0]):
            frame = np.squeeze(I[line,:,:]).T 
            col,amp,_ = find_peak(medfilt(frame.mean(axis=0),5)) #Column where the peak occurs (?)
            #print(col,amp)
            if amp<100:
                continue
            for i, w, chn in zip(range(nlasers), laser_wvl_mean, laser_pixel):
                idx = np.arange(int(chn-margin),int(chn+margin+1), dtype=int)
                row,_,_ = find_peak(frame[idx,int(round(col))])
                row = row + idx[0] 
                observed[i].append([col,row,w]) #column, row, wavelength
    
        mdict={'observed': observed}

        scio.savemat(out_dir + output_file + '_observed.mat',mdict)
    else:
        mdict = scio.loadmat(out_dir + '20221212_Laser_Sphere_spectralCal_observed_mat')
        observed  = mdict['observed']
    
    ncols     = I.shape[1]
    nrows     = I.shape[2]  
    

    fit_coeffs = np.zeros((nlasers,3))
    spectral_wvl_mat = np.broadcast_to(np.expand_dims(laser_wvl_mean,1),(nlasers,ncols))

    # #Way to use the same set up for variation testing:
    # spectral_variation = np.random.default_rng().normal(0.0,1.0,(nlasers,ncols))
    # spectral_variation = spectral_variation*np.expand_dims(laser_wvl_std,1)
    # spectral_wvl_mat = spectral_wvl_mat + spectral_variation

    # Now plot the result, and save the laser fits to a file for later plotting
    plt.figure()
    spectral_y_fit = np.zeros((nlasers,ncols))
    spatial_x_fit = np.arange(ncols)
    legend_list = []
    for ii in range(nlasers):
        # map row to column for a single laser
        observed_i = np.array(observed[ii])
        spatial_x = observed_i[:,0] #Column for each observation
        spectral_y = observed_i[:,1]+6 #Row that the laser was observed at
        p = np.polyfit(spatial_x,spectral_y,2)
        spectral_y_fit[ii,:] = np.polyval(p,spatial_x_fit) #Fit of spectral_y pixel for wavelength nlasers across the cross-track spatial direction (spatial_x)

        fit_coeffs[ii,:] = p

        plt.plot(spatial_x,spectral_y-p[-1],'.') #Points, with row intercept subtracted
        plt.plot(spatial_x_fit,spectral_y_fit[ii,:]-p[-1],f'C{ii}') #Linear fit with row intercept subtracted
        #np.savetxt('../../data/plots/EMIT_Laser_%i_ColRow.txt'%wavelengths[i],D,fmt='%8.6f')
        legend_list.extend(['{:.1f}nm raw'.format(laser_wvl_mean[ii]),'{:.1f}nm fit'.format(laser_wvl_mean[ii])])
    plt.title(f'LaS Centroids, poly fit\n{output_file}')
    plt.xlabel('spatial pixel #')
    plt.ylabel('spectral pixel # - intercept')
    plt.legend(legend_list)
    ax = plt.gca()
    ax.set_ylim((-0.2,0.2))
    plt.savefig(out_dir + output_file + '_polyfit_data.png')
    plt.close('all')

    # # Prepare functions for final fitting    
    # model_wavelengths = zemax_wvl_ideal
    # model_dispersions = -1*zemax_spectral_smpl #Flip sign due to directionality of FPA
    # dispersion_function = interp1d(model_wavelengths, model_dispersions, bounds_error=False, fill_value='extrapolate')
    
    # now we fit the nonuniform dispersion curve
    model_wavelengths=[380.00000,439.00000,498.00000,557.00000,616.00000,675.00000,800.00000,925.00000,1050.00000,1175.00000,1300.00000,1542.00000,1784.00000,2026.00000,2268.00000,2510.00000]
    model_dispersions=[7.39600,7.41900,7.43300,7.44200,7.44800,7.45200,7.45700,7.45800,7.45600,7.45300,7.45000,7.43900,7.42600,7.41100,7.39300,7.37300]
    # FPA is flipped
    model_dispersions = -np.array(model_dispersions)

    dispersion_function = interp1d(model_wavelengths, model_dispersions, bounds_error=False, fill_value='extrapolate')


    def gen_wavelengths(x, model_wavelengths=model_wavelengths, model_dispersions=model_dispersions):
        wl_intercept = x[0]
        dispersion_scale = x[1]
        wls = [wl_intercept]
        #print(start)
        #print(stretch)
        for i in range(1,nrows):
            wls.append(wls[-1]+dispersion_function(wls[-1])*dispersion_scale) #Cumulative sum to get wavelength from sampling
        #print(wls)
        return np.array(wls)

    def errs(x,actual_wavelengths,actual_columns,offset):
        wl = gen_wavelengths(x)
        predicted_wl = interp1d(offset+np.arange(len(wl)), wl, bounds_error=False, fill_value='extrapolate')(actual_columns)
        err = np.sum((actual_wavelengths - predicted_wl)**2)
        return err 

    fit_columns = range(ncols) #[int(ncols/2)] #range(ncols)
    
    wvl_centers_all = np.zeros((len(fit_columns),nrows,len(output_tag_list)))
    params_all = np.zeros((len(fit_columns),2,len(output_tag_list)))
    
    #Loop through leave-one-out list
    for kk in range(len(output_tag_list)):

        # Perform the fit for each random permutation of laser wavelengths
        offset = 0 #Pixel 14 (0-indexing) --> wvl 5000
        pixel_index = offset+np.arange(nrows)
        wvl_centers = np.zeros((len(fit_columns),nrows))
        params = np.zeros((len(fit_columns),2))
        print('column: ',end='')
        for ii in range(len(fit_columns)): #range(ncols): #range(ncols): #[150]:#range(ncols):
            if ii%20 == 0:
                print(fit_columns[ii],end=',',flush=True)
            x0 = np.array([longest_wvl_guess,1])
            best = minimize(errs,x0,args=(spectral_wvl_mat[fit_wvl_idx[kk],fit_columns[ii]],spectral_y_fit[fit_wvl_idx[kk],fit_columns[ii]],offset))
            #best = minimize(errs,x0,args=(spectral_wvl_mat[:,ii],spectral_y[:,ii],offset))
            params[ii,:] = best.x
            #p = np.polyfit(x[:,i],y[:,i],1) #Not used for anything right now
        
            wvl_centers[ii,:] = gen_wavelengths(best.x)
            
        wvl_centers_all[:,:,kk] = wvl_centers
        params_all[:,:,kk] = params
        
    mdict = {'pixel_index': pixel_index,
        'wvl_centers': wvl_centers_all,
        'laser_wvl_mean': laser_wvl_mean,
        'laser_wvl_std': laser_wvl_std,
        'params':      params_all,
        'fit_columns': fit_columns,
        'fit_coeffs':  fit_coeffs,
        'fit_wvl_idx': fit_wvl_idx,
        'spatial_x_fit':  spatial_x_fit,
        'observed': observed,
        'output_tag_list': output_tag_list,
        'spectral_y_fit': spectral_y_fit}
    scio.savemat(out_dir + output_file + '_poly_fits_shift6.mat',mdict)
        
        


if __name__ == '__main__':
    main()