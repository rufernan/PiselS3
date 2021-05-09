
function out_map = compute_psri(img,sensor)

    max_value = max(img(:));

    if max_value>1 % the values need to be in reflectance format
        img = normalize_image(img);
    end
    img = double(img);

    if strcmpi(sensor,'s2')
        out_map = ( ( img(:,:,4) - img(:,:,2) ) ./ img(:,:,6) );                        
    elseif strcmpi(sensor,'s3')
        out_map = ( ( img(:,:,8) - img(:,:,4) ) ./ img(:,:,13) );
    else
        error(['Unknown sensor "',sensor,'"']);
    end

    out_map(isnan(out_map)) = 0; out_map(isinf(out_map)) = 0; % post-process

end
