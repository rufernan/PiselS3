 
function OUT = normalize_image(IN, output_dynamic_range, contrast_adjust)
 
    if nargin<2
        output_dynamic_range = 0; % OUT --> [0,1]
    end
    if nargin<3
        contrast_adjust = false;
    end

    IN_vec = IN(:);
    amin = double(min(IN_vec));
    amax = double(max(IN_vec));
    IN_vec_norm = mat2gray(IN_vec,[amin,amax]);  
    
    if contrast_adjust
        IN_vec_norm = imadjust(IN_vec_norm);
    end
        
    if output_dynamic_range==8 % OUT --> 8 bits [0,255]
        IN_vec_norm = im2uint8(IN_vec_norm);
    elseif output_dynamic_range==16 % OUT --> 16 bits [0,65536]
        IN_vec_norm = im2uint16(IN_vec_norm);
    end
    
    OUT = reshape(IN_vec_norm, size(IN));

end
