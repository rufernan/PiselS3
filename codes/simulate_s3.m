 
function s3 = simulate_s3(s2,sim_type,ratio)

    if nargin<3
        ratio = 15; % by default s2(20m) s3(300m)
    end
        
    if strcmpi(sim_type,'mean')        
        W = fspecial('average',ratio);
        s3 = apply_psf(s2,W);
        %s3 = imresize(s2,1/ratio,'box','Antialiasing',false); % patch mean
    elseif strcmpi(sim_type,'bicubic')
        s3 = imresize(s2,1/ratio,'bicubic','Antialiasing',false); % 'bicubic'
    elseif strcmpi(sim_type,'lanczos3')
        s3 = imresize(s2,1/ratio,'lanczos3','Antialiasing',false); % 'lanczos3'        
    elseif strcmpi(sim_type,'psf')
        psf_s3 = create_psf(ratio);
        s3 = apply_psf(s2,psf_s3);   
    else
        error(['Unknown simulation "',sim_type,'"']);
    end
    
end
