
function save_plots_errors_by_value_class(cell_lr_maps, cell_err_maps, cell_class_maps, FILE_RESULTS_2D, FILE_RESULTS_3D, LABELS, REMOVE_OUTLIERS, SELECTED_INDEXES, CLEAN_FILES, OPACITY)
        
    if nargin<6
        LABELS = [];
    end
    if nargin<7
        REMOVE_OUTLIERS = false;
    end
    if nargin<8
        SELECTED_INDEXES = []; % for reducing the number of considered instances ([]->all)                
    end   
    if nargin<9
        CLEAN_FILES = false;
    end
    if nargin<10
        OPACITY = 1;
    end

    if CLEAN_FILES
        delete(FILE_RESULTS_2D);
        delete(FILE_RESULTS_3D);
    end
    
    % Putting all maps together in a 2D plane (concatenating vertically)
    lr_map = cat(1,cell_lr_maps{:});
    err_map = cat(1,cell_err_maps{:});
    class_map = cat(1,cell_class_maps{:});

    % Vectorizing and sub-sampling (to avoid too many data points)
    lr_map = lr_map(:); 
    err_map = err_map(:); 

    if not(isempty(SELECTED_INDEXES)) && numel(SELECTED_INDEXES)<length(lr_map)
        lr_map = lr_map(SELECTED_INDEXES,:);
        err_map = err_map(SELECTED_INDEXES,:);
    end  

    % Removing outliers from error values
    if REMOVE_OUTLIERS
        % [err_map,TF] = rmoutliers(err_map,'percentiles',[0,99]); % this removes too much samples
        %B = maxk(err_map,num_labs*10); TF = err_map<min(B); err_map = err_map(logical(TF)); % remove the k-max errors
        B = std(err_map)*10; TF = err_map<min(B); err_map = err_map(logical(TF)); % remove the error higher than the std multiplied by a factor
        lr_map = lr_map(logical(TF));        
    end
    
    if ismatrix(class_map) % 2D matrix --> hard clustering
        FUZZY = false;
        class_map = class_map(:);
        if not(isempty(SELECTED_INDEXES)) && numel(SELECTED_INDEXES)<length(class_map)
            class_map = class_map(SELECTED_INDEXES,:);
        end
        if REMOVE_OUTLIERS
            class_map = class_map(logical(TF));
        end
        labs = unique(class_map);
        %labs = 1:max(class_map(:)); % problem if the last label is missing in test        
    else % 3D matrix --> fuzzy clustering
        FUZZY = true;                
        fuzzy_class_map = reshape( class_map, [size(class_map,1)*size(class_map,2),size(class_map,3)] );
        [~, class_map] = max(fuzzy_class_map,[],2); % we simulate a hard-clustering class map selecting the maximum activation        
        if not(isempty(SELECTED_INDEXES)) && numel(SELECTED_INDEXES)<length(class_map)
            class_map = class_map(SELECTED_INDEXES,:);
            fuzzy_class_map = fuzzy_class_map(SELECTED_INDEXES,:);
        end
        if REMOVE_OUTLIERS
            class_map = class_map(logical(TF));
            fuzzy_class_map = fuzzy_class_map(logical(TF));
        end
        labs = 1:size(fuzzy_class_map,2);
        
        % discretized fuzzy class map to set the transparency
        NUM_BINS_TRANSPARENCY = 5; THRESHOLD = 0.0;
        vec_fuzzy_class_map = fuzzy_class_map(:);
        [Y,E] = discretize(vec_fuzzy_class_map,NUM_BINS_TRANSPARENCY);
        if E(1)==0
            E = E(2:end); % we remove the 0 value of the intervals
        end
        if E(end)>1
            E(end) = 1; % we set the maximum value to 1 (in case the number of bins do not perfectly fit between 0-1)
        end
        discretized_fuzzy_class_map = E(Y)';        
        discretized_fuzzy_class_map(vec_fuzzy_class_map<=THRESHOLD) = 0; % we reset again 0 values
        diff_values = unique(discretized_fuzzy_class_map);
        discretized_fuzzy_class_map = reshape(discretized_fuzzy_class_map,size(fuzzy_class_map)); % recovering the original size
        
    end         
    
    if not(isempty(LABELS)) % labels provided as input argument
        labs = LABELS;
    end
    num_labs = numel(labs);
        
    %colors = vivid(num_labs);
    colors = distinguishable_colors(num_labs);
    
    % counting the number of samples per cluster
    samples_per_class = histcounts(class_map,num_labs);    
    
    % sorting clusters per plot area
    I = sort_clusters_by_area(lr_map, err_map, class_map, labs);    
    
    new_labs_order = labs(I); % to plot the clusters from the most compact ones to the more expanded
        
    
    if not(isfile(FILE_RESULTS_3D))
    
        disp('--building 3D plot');
        fig=figure;
        %set(fig,'visible','on');
        set(fig,'visible','off');
        pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
        set(gcf,'units','points','position',[100,100,width*2,height*2]);
                        
        % in the FUZZY case, we need to first paint a point of each class for the colors in the legend
        if FUZZY            
            for i=1:num_labs                
                xi = lr_map(class_map==labs(new_labs_order(i)));
                yi = err_map(class_map==labs(new_labs_order(i)));
                zi = repmat(i,[numel(yi),1]);                
                
                % selecting the point with maximum error (to avoid floating points in the 2D projection)
                [~,ind] = max(yi);                
                xi = xi(ind);
                yi = yi(ind);
                zi = zi(ind);   
            
                h = scatter3(zi,xi,yi,25);
                h.MarkerFaceColor = colors(i,:)/2;
                h.MarkerEdgeColor = colors(i,:);
                h.LineWidth = 0.2;                
                OPACITY = 1;
                h.MarkerFaceAlpha = OPACITY;
                h.MarkerEdgeAlpha = OPACITY;                
                hold on;                
            end            
        end
        
                
        for i=1:num_labs

            if not(FUZZY)            
                
                xi = lr_map(class_map==labs(new_labs_order(i)));
                yi = err_map(class_map==labs(new_labs_order(i)));
                zi = repmat(i,[numel(yi),1]);
                
                h = scatter3(zi,xi,yi,25);
                h.MarkerFaceColor = colors(i,:)/2;
                h.MarkerEdgeColor = colors(i,:);
                h.LineWidth = 0.2;
                if OPACITY>0 && OPACITY<1 % control the transparency
                    h.MarkerFaceAlpha = OPACITY;
                    h.MarkerEdgeAlpha = OPACITY;
                end
                hold on;

                
            else % FUZZY
                                                                
                discretized_fuzzy_class_map_i = discretized_fuzzy_class_map(:,new_labs_order(i));
                
                for v=1:numel(diff_values)
                    
                    if diff_values(v)~=0 % for efficiency we do not paint 100% transparent points                        
                        xi = lr_map(discretized_fuzzy_class_map_i==diff_values(v));
                        yi = err_map(discretized_fuzzy_class_map_i==diff_values(v));
                        zi = repmat(i,[numel(yi),1]);
                        
                        h = scatter3(zi,xi,yi,25);
                        h.MarkerFaceColor = colors(i,:)/2;
                        h.MarkerEdgeColor = colors(i,:);
                        h.LineWidth = 0.2;
                        
                        OPACITY = diff_values(v);
                        h.MarkerFaceAlpha = OPACITY;
                        h.MarkerEdgeAlpha = OPACITY;
                        
                        hold on;                        
                    end
                
                end
                                                                
            end
            
        end
               
        grid on; grid minor;
        set(gca,'fontsize',12);
        title('S3 Deviations','FontSize',30)
        xlabel('S2 Patterns','FontSize',20); xticks(1:num_labs); set(gca, 'xticklabel', new_labs_order);
        ylabel('S3 Values','FontSize',20);
        zlabel('Error','FontSize',20);
        
        L={};
        for i=1:num_labs
            ni = new_labs_order(i);
            L{i} = ['P=',num2str(labs(ni)),' (#',num2str(samples_per_class(ni)),')'];
        end
        legend(L,'FontSize',15);
        hold off;
        
        export_fig(FILE_RESULTS_3D,'-png','-transparent');
        close(fig);        
    end
    
        
    
    if not(isfile(FILE_RESULTS_2D))
    
        disp('--building 2D plot');
        fig=figure;
        %set(fig,'visible','on');
        set(fig,'visible','off');
        pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
        set(gcf,'units','points','position',[100,100,width*2,height*2]);
        
        
        % in the FUZZY case, we need to first paint a point of each class for the colors in the legend
        if FUZZY            
            for i=1:num_labs                
                xi = lr_map(class_map==labs(new_labs_order(i)));
                yi = err_map(class_map==labs(new_labs_order(i)));               

                % selecting the point with maximum error (to avoid floating points in the 2D projection)
                [~,ind] = max(yi);                
                xi = xi(ind);
                yi = yi(ind);
            
                h = scatter(xi,yi,25);
                h.MarkerFaceColor = colors(i,:)/2;
                h.MarkerEdgeColor = colors(i,:);
                h.LineWidth = 0.2;                
                OPACITY = 1;
                h.MarkerFaceAlpha = OPACITY;
                h.MarkerEdgeAlpha = OPACITY;                
                hold on;                
            end            
        end
        
        
        for i=1:num_labs
            
            if not(FUZZY)
                
                xi = lr_map(class_map==labs(new_labs_order(i)));
                yi = err_map(class_map==labs(new_labs_order(i)));
                
                h = scatter(xi,yi,25);
                h.MarkerFaceColor = colors(i,:)/2;
                h.MarkerEdgeColor = colors(i,:);
                h.LineWidth = 0.2;
                if OPACITY>0 && OPACITY<1 % control the transparency
                    h.MarkerFaceAlpha = OPACITY;
                    h.MarkerEdgeAlpha = OPACITY;
                end
                hold on;
                
            else  % FUZZY
                
                discretized_fuzzy_class_map_i = discretized_fuzzy_class_map(:,new_labs_order(i));
                
                for v=1:numel(diff_values)
                    
                    if diff_values(v)~=0 % for efficiency we do not paint 100% transparent points                        
                        xi = lr_map(discretized_fuzzy_class_map_i==diff_values(v));
                        yi = err_map(discretized_fuzzy_class_map_i==diff_values(v));                        
                        
                        h = scatter(xi,yi,25);
                        h.MarkerFaceColor = colors(i,:)/2;
                        h.MarkerEdgeColor = colors(i,:);
                        h.LineWidth = 0.2;
                        
                        OPACITY = diff_values(v);
                        h.MarkerFaceAlpha = OPACITY;
                        h.MarkerEdgeAlpha = OPACITY;
                        
                        hold on;                        
                    end
                end
                
            end
        end
        
        grid on; grid minor;
        set(gca,'fontsize',12);
        title('S3 Deviations','FontSize',30)
        xlabel('S3 Values','FontSize',20);
        ylabel('Error','FontSize',20);
        
        L={};
        for i=1:num_labs
            ni = new_labs_order(i);
            L{i} = ['P=',num2str(labs(ni)),' (#',num2str(samples_per_class(ni)),')'];
        end
        legend(L,'FontSize',15);
        
        % to print first the bigger clusters (overlapping generated in the 2D plot)
        P = get(gca,'Children');
        set(gca,'Children',flip(P));
        
        hold off;
        
        export_fig(FILE_RESULTS_2D,'-png','-transparent');
        close(fig);        
    end
end
