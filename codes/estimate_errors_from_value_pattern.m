 
function [exp_err_map, var_err_map, max_err_map] = estimate_errors_from_value_pattern(tra_prob_err_cube, tra_mean_err_cube, lr_chl, labels, edg_lr, new_class_position_by_index)   

    % tra_prob_err_cube --> (lr,err,class)
    % tra_err_cube --> (lr,err,class)
    % lr_chl --> (366,366)
    % labels --> hard-clustering (366,366), soft-clustering (366,366,class)
    % edg_lr --> intervals used in traing for discretization
    % new_class_position_by_index --> vector containing the classes in each slide of tra_prob_err_cube and tra_err_cube
    
    if ndims(labels)==3
        FUZZY=true;
    else
        FUZZY=false;
    end    
        
    [rows,cols] = size(lr_chl);
    [~, num_err, num_cla] = size(tra_prob_err_cube);
    
    exp_err_map = zeros([rows,cols]);        
    var_err_map = zeros([rows,cols]);        
    max_err_map = zeros([rows,cols]);        
    
    % discretizing test data according to training intervals
    thresholded_lr_chl = lr_chl;
    thresholded_lr_chl(lr_chl<edg_lr(1)) = edg_lr(1);
    thresholded_lr_chl(lr_chl>edg_lr(end)) = edg_lr(end);
    dis_lr_map = discretize(thresholded_lr_chl, edg_lr);
    
    for i=1:rows
        for j=1:cols
            
            if not(FUZZY)                
                                
                tra_err_probs = squeeze(tra_prob_err_cube( dis_lr_map(i,j), :, new_class_position_by_index(labels(i,j)) )); % Note that 'tra_prob_err_cube' according to new_class_position_by_index  
                tra_err_vals = squeeze(tra_mean_err_cube( dis_lr_map(i,j), :, new_class_position_by_index(labels(i,j)) ));                
                
            else % FUZZY
                
                tra_err_probs = zeros([1,num_err]);
                tra_err_vals = zeros([1,num_err]);
                
                for k=1:num_cla                    
                    tra_err_probs = tra_err_probs + squeeze(tra_prob_err_cube( dis_lr_map(i,j), :, new_class_position_by_index(k) )) * labels(i,j,k); % Note that 'labels' sorted according to centroids and 'tra_prob_err_cube' according to new_class_position_by_index  
                    tra_err_vals = tra_err_vals + squeeze(tra_mean_err_cube( dis_lr_map(i,j), :, new_class_position_by_index(k) )) * labels(i,j,k);                    
                end
                
            end
                                    
            exp_err_map(i,j) = sum( tra_err_probs .* tra_err_vals );
            var_err_map(i,j) = sum( tra_err_probs .* (tra_err_vals - mean(tra_err_vals)).^2 );
            
            [~,pos_max_prob] = max(tra_err_probs);
            max_err_map(i,j) = tra_err_vals(pos_max_prob);
                        
        end
    end
    
end
    
    