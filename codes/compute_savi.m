
function out_map = compute_savi(img,sensor)

    if nargin<3
        normalize = false; % to normalize the input image before computing the index
    end
    
    if normalize
        img = normalize_image(img);
    end
    
    img = double(img);
        
    if strcmpi(sensor,'s2')                    
        out_map = ( ( img(:,:,8) - img(:,:,4) ) ./ ( img(:,:,8) + img(:,:,4) + 0.428 ) * 1.428 );                        
    elseif strcmpi(sensor,'s3')
        out_map = ( ( img(:,:,16) - img(:,:,8) ) ./ ( img(:,:,16) + img(:,:,8) + 0.428 ) * 1.428 );                        
    else
        error(['Unknown sensor "',sensor,'"']);        
    end
    
    out_map(isnan(out_map)) = 0; out_map(isinf(out_map)) = 0; % post-process
    out_map(out_map>10) = 10; out_map(out_map<-10) = -10; % index valid range

end
