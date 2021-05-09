 
function [IMG_DATA, selected_band] = extract_spatial_data(IMG, SELECTED_DATA, SENSOR, dynamic_range, contrast_adjust)
    
    if nargin<2
        SELECTED_DATA = 'ndvi';
    end
    if nargin<3
        SENSOR = 's2';                
    end
    if nargin<4
        dynamic_range = 0; % by default we use values normalized in [0,1] for the clustering                
    end
    if nargin<5
        contrast_adjust = false;
    end
    
    
    if strcmpi(SELECTED_DATA,'pca1')
        
        IMG_DATA = normalize_image(impca(IMG), dynamic_range, contrast_adjust); 
        selected_band = SELECTED_DATA;

    
    elseif strcmpi(SELECTED_DATA,'ndvi')
                
        IMG_DATA = normalize_image(compute_ndvi(IMG,SENSOR), normalize_image, contrast_adjust);
        selected_band = SELECTED_DATA;
            
    
    elseif strcmpi(SELECTED_DATA,'maxentropy') % just used when we want to select the maximum entropy band per image (normaly we fix this according to the training set)
        
        max_entropy = 0;
        selected_band = 0;
        for b=1:size(IMG,3)            
            band_entropy = entropy(normalize_image(IMG(:,:,b)));
            if band_entropy>max_entropy
                selected_band = b;
                max_entropy = band_entropy;
            end
        end
                        
        IMG_DATA = normalize_image(IMG(:,:,selected_band), dynamic_range, contrast_adjust);
            
    
    else % providing a number or string with the selected band number
        
        string_contains_numeric = @(S) ~isnan(str2double(S));
        if string_contains_numeric(SELECTED_DATA)
            selected_band = str2num(SELECTED_DATA);
        elseif isnumeric(SELECTED_DATA)
            selected_band = SELECTED_DATA;
        else
            error('Wrong selected band!');
        end
        IMG_DATA = normalize_image(IMG(:,:,selected_band), dynamic_range, contrast_adjust);
    
    
    end   
    
end


%{

% BANDS
for i=1:size(IMG,3)
    IMG_DATA = normalize_image(IMG(:,:,i)); 
    fig=figure;
    set(fig,'visible','off');  
    imagesc(IMG_DATA); 
    colorbar;
    title(['Band=',num2str(i,'%02d'),', Entropy=',num2str(entropy(IMG_DATA),'%.02f')]);
    export_fig(['B',num2str(i,'%02d'),'_E',num2str(entropy(IMG_DATA),'%.02f'),'.png'],'-png'); 
    close(fig);
end

% PCA1
IMG_DATA = normalize_image(impca(IMG)); 
fig=figure;
set(fig,'visible','off');  
imagesc(IMG_DATA); 
colorbar;
title(['PCA1, Entropy=',num2str(entropy(IMG_DATA),'%.02f')]);
export_fig(['PCA1_E',num2str(entropy(IMG_DATA),'%.02f'),'.png'],'-png'); 
close(fig);

% NDVI
IMG_DATA = normalize_image(compute_ndvi(IMG,SENSOR)); 
fig=figure;
set(fig,'visible','off');  
imagesc(IMG_DATA); 
colorbar;
title(['NDVI, Entropy=',num2str(entropy(IMG_DATA),'%.02f')]);
export_fig(['NDVI_E',num2str(entropy(IMG_DATA),'%.02f'),'.png'],'-png'); 
close(fig);

%}


