 
function OUT = apply_psf(IN,PSF)
    
    IN = double(IN);
    PSF = double(PSF);                        
    [~,~,bands] = size(IN);
    patch_size = size(PSF);
    
    fun = @(block_struct) sum(block_struct.data(:) .* PSF(:));   
    for k=1:bands        
        OUT(:,:,k) = blockproc(IN(:,:,k), patch_size, fun);
    end    
                                  
end
