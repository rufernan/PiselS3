 
function NEW_CLUST_ORDER = sort_clusters_by_area(lr_map, err_map, class_map, labs)

    num_labs = numel(labs);
    area_per_class = zeros(num_labs,1);

    for i=1:num_labs
    
        x = lr_map(class_map==labs(i));
        y = err_map(class_map==labs(i));
        
        if numel(x)>3 % for computing the area
            dt = delaunayTriangulation(x,y);
            k = convexHull(dt);
            Perimeter = sqrt(diff(x(k))*diff(x(k))'+ diff(y(k))*diff(y(k))');
            Area = abs(trapz(x(k),y(k)));
        else
            Area = 0;
        end
    
        area_per_class(i) = Area;
    
    end

    [~,I] = sort(area_per_class);
    
    NEW_CLUST_ORDER = I;

end




%% OLD CODES
%{
    % sorting clusters per standard deviation
    str_per_class = zeros(num_labs,1);
    %glob_std_lr = std(lr_map); glob_std_err = std(err_map); % for 'normalization'
    for i=1:num_labs
        str_per_class(i) = std(lr_map(class_map==labs(i)));
        %str_per_class(i) = std(lr_map(class_map==labs(i)))/glob_std_lr + std(err_map(class_map==labs(i)))/glob_std_err;
    end
    [~,I] = sort(str_per_class);
    
    % sorting clusters per standard deviations of 'lr', 'err' and also the number of cluster samples
    str_per_class = zeros(num_labs,3); % --> (std_lr,std_err,num_samples)
    for i=1:num_labs
        str_per_class(i,1) = std(lr_map(class_map==labs(i)));
        str_per_class(i,2) = std(err_map(class_map==labs(i)));
        str_per_class(i,3) = samples_per_class(labs(i));
    end    
    str_per_class = str_per_class ./ repmat(sum(str_per_class),[size(str_per_class,1),1]); % normalization (to be equally important)
    weights = [1,1,1]; % to ponderate the relevance of (std_lr,std_err,num_samples)
    str_per_class = sum( str_per_class ./ repmat(weights,[size(str_per_class,1),1]), 2);
    [~,I] = sort(str_per_class);
%}
