 
function mat_OUT = sort_rows(mat_IN,new_row_order)
         
    for i=1:numel(new_row_order)
        mat_OUT(i,:) = mat_IN(new_row_order(i),:);
    end
 
end
