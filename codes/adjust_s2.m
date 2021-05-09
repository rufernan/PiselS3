 
function s2_adj = adjust_s2(s2)
 
    % Adjusting the dynamic range of a S2 image likewise in SNAP visualization (https://gis.stackexchange.com/questions/259907/constrast-stretching-sentinel-2-l1c)
    
    adj_limits = stretchlim(s2(:),[0.025,0.975]);    
    s2_adj = cast(zeros(size(s2)),'like',s2);           
    for i=1:size(s2,3)
        s2_adj(:,:,i) = imadjust(s2(:,:,i),adj_limits);
    end
    
end
