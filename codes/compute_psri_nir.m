
function out_map = compute_psri_nir(img,sensor,normalize)

    if nargin<3
        normalize = false; % to normalize the input image before computing the index
    end
    
    if normalize
        img = normalize_image(img);
    end
    
    img = double(img);
        
    if strcmpi(sensor,'s2')                    
        out_map = ( ( img(:,:,4) - img(:,:,2) ) ./ img(:,:,8) );                        
    elseif strcmpi(sensor,'s3')
        out_map = ( ( img(:,:,8) - img(:,:,4) ) ./ img(:,:,16) );                
    else
        error(['Unknown sensor "',sensor,'"']);        
    end
    
    out_map(isnan(out_map)) = 0; out_map(isinf(out_map)) = 0; % post-process  
    out_map(out_map>1) = 1; out_map(out_map<-1) = -1; % index valid range
    
end

