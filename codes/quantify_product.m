
% En Sentinel-2 (tanto L1C como L2A) los productos de reflectividad (generalmente en el rango [0,1] salvo superficies tipo cristal, hielo...) pese a ser valores flotantes se almacenan en disco (dentro de los respectivos productos) como enteros por tema de eficiencia. Para ello, los valores de reflectividad se multiplican por la constante QUANTIFICATION_VALUE (para productos "nuevos" es 10^4) para almacenar los valores como uint16 en jpeg2000. Este valor esta definido dentro del XML de los productos. Ver para mas detalles: https://gis.stackexchange.com/questions/233874/what-is-the-range-of-values-of-sentinel-2-level-2a-images
 
function OUT = quantify_product(IN, QUANTIFICATION_VALUE)
 
    if nargin<2
        QUANTIFICATION_VALUE = 10^4; 
    end

    OUT = double(IN)/QUANTIFICATION_VALUE;
    OUT(OUT<0) = 0; OUT(OUT>1) = 1; % reflectance range between [0,1]

end
