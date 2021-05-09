 
function error_map = compute_errors(x,y,error_type)
    
    if nargin<3
        error_type = 'ae';    
    end
    
    x = double(x); y = double(y);
    
    if strcmpi(error_type,'ae') % absolute error
        error_map = abs(x-y);  
    elseif strcmpi(error_type,'se') % squared error
        error_map = (x-y).^2;
    else
        error(['Unknown error "',error_type,'"']);
    end      
    
end
