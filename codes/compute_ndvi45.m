
function out_map = compute_ndvi(img,sensor,normalize)

    if nargin<3
        normalize = false; % to normalize the input image before computing the index
    end
    
    if normalize
        img = normalize_image(img);
    end
    
    img = double(img);
        
    if strcmpi(sensor,'s2')                    
        out_map = ( ( img(:,:,5) - img(:,:,4) ) ./ ( img(:,:,5) + img(:,:,4) ) );                        
    elseif strcmpi(sensor,'s3')
        out_map = ( ( img(:,:,11) - img(:,:,8) ) ./ ( img(:,:,11) + img(:,:,8) ) );                
    else
        error(['Unknown sensor "',sensor,'"']);        
    end
    
    out_map(isnan(out_map)) = 0; out_map(isinf(out_map)) = 0; % post-process
    out_map(out_map>1) = 1; out_map(out_map<-1) = -1; % index valid range

end
