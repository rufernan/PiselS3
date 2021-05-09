
function bool_indexes = find_any_in_cell(cell_str, cell_patterns)
    
    bool_indexes = zeros(1,numel(cell_str));
    
    for i=1:numel(cell_str)
       for j=1:numel(cell_patterns)
           positions = strfind(cell_str{i},cell_patterns{j});
           if numel(positions)>0
               bool_indexes(i) = 1;
           end        
       end
    end
    bool_indexes = logical(bool_indexes);
end
