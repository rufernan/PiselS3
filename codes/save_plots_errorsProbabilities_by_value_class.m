
function [prob_err_cube, err_cube, count_cube, edg_err, edg_lr, mean_err_cube, new_class_position_by_index, colors] = save_plots_errorsProbabilities_by_value_class(cell_lr_maps, cell_err_maps, cell_class_maps, FILE_RESULTS_1D, FILE_RESULTS_2D, FILE_RESULTS_3D, LR_BINS, ERR_BINS, LABELS, REMOVE_OUTLIERS, SELECTED_INDEXES, CLEAN_FILES, OPACITY)

    if nargin<7
        LR_BINS = 2^6;
    end
    if nargin<8
        ERR_BINS = 2^6;
    end

    if nargin<9
        LABELS = [];
    end
    if nargin<10
        REMOVE_OUTLIERS = false;
    end
    if nargin<11
        SELECTED_INDEXES = []; % for reducing the number of considered instances ([]->all) 
    end    
    if nargin<12
        CLEAN_FILES = false;        
    end   
    if nargin<13
        OPACITY = 1;
    end

    if CLEAN_FILES
        delete(FILE_RESULTS_1D);
        delete(FILE_RESULTS_2D);
        delete(FILE_RESULTS_3D);
    end
    
    % Putting all maps together in a 2D plane
    lr_map = cat(1,cell_lr_maps{:}); lr_map = lr_map(:);
    err_map = cat(1,cell_err_maps{:}); err_map = err_map(:);
    
    if not(isempty(SELECTED_INDEXES)) && numel(SELECTED_INDEXES)<length(err_map)
        err_map = err_map(SELECTED_INDEXES);
        lr_map = lr_map(SELECTED_INDEXES);
    end  
    
    % Removing outliers from error values
    if REMOVE_OUTLIERS        
        % [err_map,TF] = rmoutliers(err_map,'percentiles',[0,99]); % this removes too much samples
        %B = maxk(err_map,num_labs*10); TF = err_map<min(B); err_map = err_map(logical(TF)); % remove the k-max errors
        B = std(err_map)*10; TF = err_map<min(B); err_map = err_map(logical(TF)); % remove the error higher than the std multiplied by a factor
        lr_map = lr_map(logical(TF));        
    end

    
    class_map = cat(1,cell_class_maps{:});
            
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
        labs = 1:size(class_map,2);
        
        % discretized fuzzy class map to set the transparency
        NUM_BINS_TRANSPARENCY = 4; THRESHOLD = 0.00;
        vec_fuzzy_class_map = fuzzy_class_map(:);
        [Y,E] = discretize(vec_fuzzy_class_map,NUM_BINS_TRANSPARENCY);
        if E(1)==0
            E = E(2:end); % we remove the 0 value of the intervals
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

    % re-naming class names 
    %F = @(x) new_labs_order(x);
    %class_map = arrayfun(F, class_map);
    
    new_class_position_by_index = zeros(size(new_labs_order)); % the vector position is the old class and the content is the new position of the class
    for i=1:numel(new_class_position_by_index)
        new_class_position_by_index(i) = find(new_labs_order==i);
    end
    
    if FUZZY
        cl_ma = fuzzy_class_map; % each element is a probability vector for classes
    else
        cl_ma = class_map; % each element is just a label
    end
    
    [prob_err_cube, err_cube, count_cube, edg_err, edg_lr] = compute_errors_probabilities(err_map, lr_map, cl_ma, ERR_BINS, LR_BINS, new_class_position_by_index);        
    [N_LR,N_ERR,N_CLA] = size(prob_err_cube);

    % converting to string the label values    
    str_edg_err = compose("%.2f",edg_err);
    str_edg_lr = compose("%.2f",edg_lr);
    
    mean_err_cube = err_cube ./ single(count_cube);    
    mean_err_cube(isnan(mean_err_cube)) = 0; % setting Nan's to 0
    mean_err_cube(isinf(mean_err_cube)) = 0; % setting Inf's to 0
    
    if not(isfile(FILE_RESULTS_3D))
    
        disp('--building 3D-probability plot');
        fig=figure;
        set(fig,'visible','off');
        pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
        set(gcf,'units','points','position',[100,100,width*2,height*2]);
        
        [X,Y,Z] = ndgrid(1:N_LR, 1:N_ERR, 1:N_CLA);
        
        % removing 0 probability values (for viasualization)
        vec_X = X(:); vec_Y = Y(:); vec_Z = Z(:);
        vec_prob_err_cube = prob_err_cube(:);
        no_zero = vec_prob_err_cube~=0;
        vec_X = vec_X(no_zero); vec_Y = vec_Y(no_zero); vec_Z = vec_Z(no_zero);
        vec_prob_err_cube = vec_prob_err_cube(no_zero);
                
        h = scatter3(vec_Z, vec_X, vec_Y, 25, vec_prob_err_cube,'filled');
        h.LineWidth = 0.2;
        if OPACITY>0 && OPACITY<1 % control the transparency
            h.MarkerFaceAlpha = OPACITY;
            h.MarkerEdgeAlpha = OPACITY;
        end
        hold on;
        
        colorbar;
        grid on; grid minor;
        set(gca,'fontsize',12);
        title('S3 Deviation Probabilities','FontSize',30)
        xlabel('S2 Patterns','FontSize',20); xticks(1:num_labs); set(gca, 'xticklabel', new_labs_order);
        ylabel(['S3 Value Bins (',num2str(LR_BINS),')'],'FontSize',20);
        zlabel(['Error Bins (',num2str(ERR_BINS),')'],'FontSize',20);
        
        NUM_TICKS = 5;
        LR_STEP = floor(LR_BINS/NUM_TICKS);
        LR_TICKS = 1:LR_STEP:(LR_BINS+1);
        ERR_STEP = floor(LR_BINS/NUM_TICKS);
        ERR_TICKS = 1:ERR_STEP:(ERR_BINS+1);
        
        set(gca, 'ytick', LR_TICKS, 'yticklabel', str_edg_lr(LR_TICKS));
        set(gca, 'ztick', ERR_TICKS, 'zticklabel', str_edg_err(ERR_TICKS));
        
        %%{
        % displaying the grid of the class distributions
        STEP = 1;
        X1 = X(1:STEP:end,1:STEP:end,1);
        X2 = Y(1:STEP:end,1:STEP:end,1);
        Z1 = ones(size(X1));
        
        for i=1:num_labs
            caxis('manual'); % to avoid affecting the colorbar of the scatter            
            h = mesh(Z1*i,X1,X2,'FaceColor','none', 'EdgeColor','k','EdgeAlpha',0.1);
            h.EdgeColor = colors(i,:);
        end        
        %%}
        
        hold off;
        
        export_fig(FILE_RESULTS_3D,'-png','-transparent');
        close(fig);
    end
    
    
    
    if not(isfile(FILE_RESULTS_2D))
    
        disp('--building 3D-cluster histograms');
        
        F1 = ceil(sqrt(num_labs));
        F2 = ceil(sqrt(num_labs));        
        fig=figure;
        set(fig,'visible','off');
        pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
        set(gcf,'units','points','position',[100,100,width*2,height*2]);
        sgtitle('Cluster Bivariate Histograms','FontSize',30);
        for i=1:num_labs
            subplot(F1,F2,i);            
            D1 = lr_map(class_map==labs(i));
            D2 = err_map(class_map==labs(i));                        
            histogram2(D1, D2, [LR_BINS, ERR_BINS], 'DisplayStyle','tile');
            title(['P=',num2str(i),' (#',num2str(numel(D1)),')'],'FontSize',20);
            xlabel(['S3 Value bins (',num2str(LR_BINS),')'],'FontSize',15);
            ylabel(['Error bins (',num2str(ERR_BINS),')'],'FontSize',15);
            grid on; grid minor;
            colorbar;
            hold on;
        end       
        
        hold off;
        export_fig(FILE_RESULTS_2D,'-png','-transparent');
        close(fig);
    end
    
    
    
    
    if not(isfile(FILE_RESULTS_1D))
    
        disp('--building mean probability plot');
        
        NUM_TICKS = 5;
        LR_STEP = floor(LR_BINS/NUM_TICKS);        
        LR_TICKS = 1:LR_STEP:(LR_BINS+1);
        
        mean_mean_err_cube = squeeze(mean(mean_err_cube,2)); % (lr,err,class) averaging 'err'
        MIN_ERR = min(mean_mean_err_cube(:));
        MAX_ERR = max(mean_mean_err_cube(:));
        
        F1 = ceil(sqrt(num_labs));
        F2 = ceil(sqrt(num_labs));        
        fig=figure;
        set(fig,'visible','off');
        pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
        set(gcf,'units','points','position',[100,100,width*2,height*2]);
        sgtitle('Clusters mean error per value bin','FontSize',30);
        for i=1:num_labs
            subplot(F1,F2,new_labs_order(i));                                    
            data = mean_mean_err_cube(:,i);
            h = plot(data);
            h.Color = colors(i,:);
            h.LineWidth = 2;
            
            ylim([MIN_ERR MAX_ERR]);
            %xlim('auto');
            
            title(['P=',num2str(new_labs_order(i)),' (#',num2str(samples_per_class(new_labs_order(i))),')'],'FontSize',20);
            xlabel(['S3 Value bins (',num2str(LR_BINS),')'],'FontSize',15);
            ylabel(['Error bins (',num2str(ERR_BINS),')'],'FontSize',15);
            set(gca, 'xtick', LR_TICKS, 'xticklabel', str_edg_lr(LR_TICKS));
            grid on; grid minor;            
            hold on;
        end       
        
        hold off;
        export_fig(FILE_RESULTS_1D,'-png','-transparent');
        close(fig);
    end
    
    
    
    
end
