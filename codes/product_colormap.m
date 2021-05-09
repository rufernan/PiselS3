 
function cmap = product_colormap(type)

    if strcmpi(type,'psri-nir')
        cmap = 'parula';
    else
        cmap = 'parula';
    end                
        
end


%{
    % Colormap for the product
    %ranges = [ 0, 1, 1.8, 2.5, 4, 4.5, 5];
    ranges = [ -1, -0.7, -0.4, -0.1, 0.2, 0.5, 1]; % PSRI-NIR in [-1,1] (-0,1 to 0.2 green vegetation) see https://www.harrisgeospatial.com/Learn/Whitepapers/Whitepaper-Detail/ArtMID/17811/ArticleID/16162/Vegetation-Analysis-Using-Vegetation-Indices-in-ENVI
    base_colors= [
        0, 0,   0.5; ...
        0, 0.3, 0.8; ...
        1, 0.2, 0.2; ...
        1, 0.9, 0;   ...
        0, 0.8, 0.1; ...
        0, 0.6, 0.2; ...
        %1, 1,   1
        0, 0.25,0.25
        ];    
    prod_cmap = [];
    NUM_COLORS = 20;
    for c=1:(size(base_colors,1)-1)
        cols = colorGradient(base_colors(c,:),base_colors(c+1,:),NUM_COLORS);
        prod_cmap = [prod_cmap; cols];
    end
    clims = [ranges(1), ranges(end)];
%}
