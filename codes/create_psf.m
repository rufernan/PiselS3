 
function W = create_psf(psf_size)

    % Creating a Gaussian PSF by adjunsting the kernel size to the FWHM

    % For a Gaussian function --> FWHM=2.355*SIGMA (see https://es.mathworks.com/matlabcentral/answers/304493-how-to-calculate-fwhm-for-multiple-gaussian-curves-in-a-plot)   
    
    % In the case of simulating the PSF of S3, we set the FWHM to 300m (number of S2 values in a S3 pixel) likewise the following work
    % Brown, Luke A., et al. "Synergetic exploitation of the Sentinel-2 missions for validating the Sentinel-3 ocean and land color instrument terrestrial chlorophyll index over a vineyard dominated mediterranean environment." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12.7 (2019): 2244-2251.
        
    sigma = psf_size/2.355;   
    W = fspecial('gaussian',psf_size,sigma);
            
end
