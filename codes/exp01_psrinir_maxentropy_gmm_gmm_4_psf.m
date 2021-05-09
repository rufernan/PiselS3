close all; clearvars;

addpath(genpath('.'));
delete([mfilename,'.txt']);
diary([mfilename,'.txt']); diary on;

DATA_FOLDER = '../data2';
ALL_NAMES_struct = dir(fullfile(DATA_FOLDER,'S2*.mat'));
ALL_NAMES = extractfield(ALL_NAMES_struct,'name');


OUT = fullfile(pwd,mfilename);
makefolder(OUT);

mfilename_parts = strsplit(mfilename,'_');
simulation_type = mfilename_parts{end}; % simulation type from ths script name ('mean', 'bicubic','psf')
k = str2num(mfilename_parts{end-1}); % number of patterns from ths script name
assig_alg = mfilename_parts{end-2}; % algorithm to assing spatial patterns to images ('knn','gauss','gmm')
clust_alg = mfilename_parts{end-3}; % clustering algorithm to extract spatial patterns from tra ('kmeans','kmedoids','gmm')
selected_data = mfilename_parts{end-4}; % data used to compute the spatial patterns in S2 ('maxentropy','ndvi', 'pca1' or a band number e.g. 2,3,4,8)
bio_prod = mfilename_parts{end-5}; % biophysical product to compute the deviations ('psrinir','ndvi45')

DIST_CLUST = 'sqeuclidean'; % 'kmeans/kmedoids' and 'knnsearch' can have different distance functions
DIST_ASSIG = 'euclidean';
SN = 's2'; % sensor to compute the ndvi index in the spatial data
DR = 0; % dynamic range for clustering (0->[0,1], 8->[0,255], 16->[0,65535])
AD = false; % adjusting contrast of S2 images before clustering

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GENERATING TRAINING/TEST INDICES %
TRA_TILES = {'30STG', '30TVK', '29TPF', '30TYM', '30TXQ', '31TFN', '31UDQ', '32UNG', '33UXU', '32UQU', '32TQM'};
ID_TRA = find_any_in_cell(ALL_NAMES,TRA_TILES);
ID_TST = not(ID_TRA);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%
% TRAINING SAMPLES %
NUM_CLUST_SAMPLES = 500000; % for clustering (0->all)
NUM_VIS_SAMPLES = 50000; % for visualization (0->all)
NUM_PROB_SAMPLES = 0; % for computing probabilities (0->all)
NUM_REG_SAMPLES = 50000; % for training regressors (0->all)
seed = 1; % random seed for reproducibility 
NUM_MAX = max([NUM_CLUST_SAMPLES,NUM_VIS_SAMPLES,NUM_PROB_SAMPLES,NUM_REG_SAMPLES]);
selected_indexes = []; % to globally store the indexes of the selected samples (initialized to []->all)
%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NUMBER OF BINS TO TEST WITH THE PROPOSED APPROACH %
LIST_BINS = [64,128,258,512];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
disp('-------------------');
disp('-- TRAINIG PHASE --');
disp('-------------------');
disp('1-Computing spatial patterns from s2...');
NAMES = ALL_NAMES(ID_TRA); % training
PATCH_SIZE = [15,15]; % S2 patch size in a S3 pixel
FILE_CLUST = fullfile(OUT,'tra_Clusters.mat');

DATA2CLUSTER = [];

if isfile(FILE_CLUST)            
    disp('--pre-loading cluster file');
    load(FILE_CLUST); % 'clust_data','C','E','SYS','selected_data','DATA2CLUSTER','selected_indexes'

else            
    
    if strcmpi(selected_data,'maxentropy') % we overwrite 'selected_data' with the band with the maximum average entropy in training         
        disp('--finding the maximum entropy band for training images');        
        for i=1:numel(NAMES)
            disp(['----loading s2 image ',NAMES{i}]);
            s2_uint16 = loadfield(fullfile(DATA_FOLDER,NAMES{i}));
            s2 = quantify_product(s2_uint16); % reflectance [0,1]
            disp('----computing entropy per band');
            for b=1:size(s2,3)            
                tra_images_entropy_per_band(i,b) = entropy(normalize_image(s2(:,:,b)));
            end
        end
        mean_entropy_per_band = mean(tra_images_entropy_per_band,1);
        [~,selected_data] = max(mean_entropy_per_band);
    end
        
    
    for i=1:numel(NAMES)        
        disp(['--loading s2 image ',NAMES{i}]);        
        s2 = loadfield(fullfile(DATA_FOLDER,NAMES{i})); % uint16
        disp('--generating spatial data');        
        IMG_DATA = extract_spatial_data(s2, selected_data, SN, DR, AD);                   
        disp('--extracting patches');        
        PATCHES = im2col(IMG_DATA,PATCH_SIZE,'distinct')'; % patches in rows
        DATA2CLUSTER = [DATA2CLUSTER; PATCHES];
    end    
    
    disp('--clustering');
    selected_indexes = randsample(1:length(DATA2CLUSTER), NUM_MAX); % here we generate all the posible (maximun) indexes
    if NUM_CLUST_SAMPLES>0 && NUM_CLUST_SAMPLES<size(DATA2CLUSTER,1)        
        DATA2CLUSTER = DATA2CLUSTER(selected_indexes(1:NUM_CLUST_SAMPLES),:); % here we pick the first NUM_CLUST_SAMPLES
    end
    DATA2CLUSTER = rmoutliers(DATA2CLUSTER,'percentiles',[1,99]); % removing outliers
    [clust_data, C, E, SYS] = cluster_spatial_data(DATA2CLUSTER, k, clust_alg, DIST_CLUST); % clustered_data, centroids, exemplar_samples, inference_system            
    disp('--saving clustering');  
    save(FILE_CLUST,'clust_data','C','E','SYS','selected_data','DATA2CLUSTER','selected_indexes','-v7.3');

end


disp('2-Generating spatial patterns plot...');

FILE_PATTERNS = fullfile(OUT,'01_tra_patterns.png');

if not(isfile(FILE_PATTERNS))                    
    %MYCOLORMAP = vivid(256);        
    IM_CENT = reshape(C', [PATCH_SIZE(1) PATCH_SIZE(1) k]);    
    IM_EXEM = reshape(E, [size(E,1) size(E,2) PATCH_SIZE(1) PATCH_SIZE(1)]);
    num_exemplars_per_cluster = size(E,2);
    F1 = num_exemplars_per_cluster+1; % one centroid and some exemplars per rows
    F2 = k; % number of clusters per columns    
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    sgtitle('Spatial Patterns in S2','FontSize',30);        
    
    %clims = [min(C(:)), max(C(:))];
    for i=1:k % centroids        
        subplot(F1,F2,i);                
        imagesc(IM_CENT(:,:,i));        
        %imagesc(IM_CENT(:,:,i),clims); % all the same colorbar
        title(['P=',num2str(i)],'FontSize',20);        
        axis image;
        axis off;
        colorbar;                       
    end
    
    %cents_clims = [min(C,[],2), max(C,[],2)]; % to put the exemplars within the centroids value range
    %clims = [min(E(:)), max(E(:))];
    for c=1:k % clusters
        for e=1:num_exemplars_per_cluster % exemplars
            row=e+1; col=c;
            all_ind = reshape(1:F1*F2,[F2,F1])';
            i = all_ind(row,col);
            subplot(F1,F2,i);                
            imagesc(squeeze(IM_EXEM(c,e,:,:)));
            %imagesc(squeeze(IM_EXEM(c,e,:,:)),cents_clims(c,:));
            %imagesc(squeeze(IM_EXEM(c,e,:,:)),clims); % all the same colorbar            
            title(['#',num2str(e)],'FontSize',20);        
            axis image;
            axis off;
            colorbar;
        end
    end
    
    %colormap(MYCOLORMAP);
    export_fig(FILE_PATTERNS,'-png','-transparent');
    close(fig);
end
        

disp('3-Calculating product errors...');
NAMES = ALL_NAMES(ID_TRA); % training
S3_SIZE = [366,366]; 

for i=1:numel(NAMES)
           
    FILE_CHL = fullfile(OUT,['tra_CHL_',NAMES{i}]);
    
    if isfile(FILE_CHL)        
        disp(['--pre-loading ',NAMES{i}]);
        load(FILE_CHL); % 'lr_chl','err','labels'
        
    else
    
        disp(['--loading s2 image ',NAMES{i}]); 
        s2_uint16 = loadfield(fullfile(DATA_FOLDER,NAMES{i}));
        s2 = quantify_product(s2_uint16); % reflectance [0,1]
        disp('--simulating s3');
        ratio = size(s2,1)/S3_SIZE(1);       
        s3_uint16 = simulate_s3(s2_uint16,simulation_type,ratio); % simulation_type --> 'mean','bicubic','psf'
        s3 = quantify_product(s3_uint16); % reflectance [0,1]
        
        disp('--computing indexes');
        if strcmpi(bio_prod,'psrinir')        
            s2_index = compute_psri_nir(s2,'s2');
            s3_index = compute_psri_nir(s3,'s2'); % the simulated s3 contains the s2 bands        
        elseif strcmpi(bio_prod,'ndvi45')
            s2_index = compute_ndvi45(s2,'s2');
            s3_index = compute_ndvi45(s3,'s2'); % the simulated s3 contains the s2 bands
        else
            ERROR_STR = ['Unrecognized biophysical product --> "', bio_prod, '"'];
            error(ERROR_STR);
        end
            
        gt_chl = s2_index;
        lr_chl = s3_index;               
        
        disp('--averaging s2 index');
        mean_gt_chl = simulate_s3(gt_chl,'mean',ratio); % patch mean

        disp('--computing s2 patterns for each s3 pixel');                            
        disp('----generating spatial information');        
        IMG_DATA = extract_spatial_data(s2, selected_data, SN, DR, AD);        
        disp('----extracting patches');        
        PATCHES = im2col(IMG_DATA,PATCH_SIZE,'distinct')'; % patches in rows
        disp('----assigning clusters');
        [Idx, labels] = assign_spatial_data_to_clusters(PATCHES, C, SYS, assig_alg, size(s3), DIST_ASSIG);                                  
        disp('--computing index errors');    
        err = compute_errors(mean_gt_chl,lr_chl);
        disp('--saving results');    
        save(FILE_CHL,'lr_chl','err','labels','-v7.3');
         
    end
    
    CELL_LR_MAPS{i} = lr_chl;
    CELL_ERR_MAPS{i} = err;
    CELL_CLASS_MAPS{i} = labels;               
    
end



disp('4-Training regressors...'); 
% for the regression, we concatenate the index value in the 1st column of 'x_tra' and the cluster id (or probs) from the 2nd on

FILE_REG = fullfile(OUT,'tra_Regressors.mat');

if isfile(FILE_REG)
    disp(['--pre-loading ',NAMES{i}]);
    load(FILE_REG); % 'mdl_lin','mdl_rid','mdl_svr','mdl_gpr','mdl_btree'
    
else

    all_lr = [CELL_LR_MAPS{:}];
    all_lab = [CELL_CLASS_MAPS{:}];
    all_err = [CELL_ERR_MAPS{:}];
    
    x_tra1 = all_lr(:); % biophysical values
    [al1,al2,al3] = size(all_lab); % if the clustering is fuzzy we punt the whole vector
    x_tra2 = reshape(all_lab,[al1*al2,al3]); % cluster ids (or probs)
    y_tra = all_err(:); % gt errors
    
    x_tra = zeros([size(x_tra1,1),1+al3]);
    x_tra(:,1) = x_tra1;
    x_tra(:,2:end) = x_tra2;
    
    % Subset of data for training the regressors
    if NUM_REG_SAMPLES>0 && NUM_REG_SAMPLES<size(x_tra,1)
        x_tra = x_tra(selected_indexes(1:NUM_REG_SAMPLES),:);
        y_tra = y_tra(selected_indexes(1:NUM_REG_SAMPLES),:);
    end
    % remmoving outliers
    [x_tra,TF] = rmoutliers(x_tra,'percentiles',[1,99]);
    y_tra = y_tra(not(TF));
    
    
    disp('--training LINEAR regressor');
    tic;
    mdl_lin = fitlm(x_tra,y_tra,'linear','RobustOpts','on');
    toc;
    disp('--training RIDGE regressor');
    tic;
    mdl_rid = fitrlinear(x_tra, y_tra,'Learner','leastsquares','Regularization','ridge');    
    toc;  
    disp('--training SVR regressor');
    tic;
    mdl_svr = fitrsvm(x_tra,y_tra,'KernelFunction','rbf','KernelScale','auto','Standardize',true,'Verbose',1,'NumPrint',10);
    %mdl_svr = fitrsvm(x_tra,y_tra,'KernelFunction','rbf','KernelScale','auto','Standardize',true,'IterationLimit',100000, 'Verbose',1,'NumPrint',10);
    toc;
    disp('--training GPR regressor');
    tic;
    mdl_gpr = fitrgp(x_tra,y_tra,'KernelFunction','squaredexponential','Standardize',true,'Verbose',1);
    %mdl_gpr = fitrgp(x_tra,y_tra,'KernelFunction','squaredexponential','Standardize',true,'IterationLimitBCD',10000, 'Verbose',1);
    toc;            
    disp('--training BinTree regressor');
    tic;
    mdl_btree = fitrtree(x_tra,y_tra);    
    toc;
    
    disp('--saving results');
    save(FILE_REG, 'mdl_lin', 'mdl_rid', 'mdl_svr', 'mdl_gpr', 'mdl_btree', '-v7.3');    
            
end



disp('5-Generating training data plots...');
FILE_RESULTS_2D = fullfile(OUT,'02_tra_errors_2D.png');
FILE_RESULTS_3D = fullfile(OUT,'03_tra_errors_3D.png');
LABELS = 1:k;
REMOVE_OUTLIERS = false;
SELECTED_INDEXES = selected_indexes(1:NUM_VIS_SAMPLES); % samples selected for the plots
%delete(FILE_RESULTS_2D);
%delete(FILE_RESULTS_3D);
if not(isfile(FILE_RESULTS_2D)) || not(isfile(FILE_RESULTS_3D))
    save_plots_errors_by_value_class(CELL_LR_MAPS,CELL_ERR_MAPS,CELL_CLASS_MAPS,FILE_RESULTS_2D,FILE_RESULTS_3D,LABELS,REMOVE_OUTLIERS,SELECTED_INDEXES);
end



disp('6-Training proposed approach...');

FILE_RES = fullfile(OUT,'tra_Results.mat');

if isfile(FILE_RES)
    disp('--pre-loading training');
    load(FILE_RES); %  'list_prob_err_cube', 'list_err_cube', 'list_count_cube', 'list_edg_err', 'list_edg_lr', 'list_mean_err_cube', 'list_new_class_position_by_index', 'colors'
    
else

    for i =1:numel(LIST_BINS) % Iterating over LIST_BINS
        
        num_lr_bins = LIST_BINS(i);
        num_err_bins = LIST_BINS(i);
        NB = num2str(LIST_BINS(i));
        disp(['--using ',NB,' bins...']);
        
        FILE_RESULTS_2D = fullfile(OUT,['04_tra_errors_hist_density_2D_',NB,'.png']);
        FILE_RESULTS_1D = fullfile(OUT,['05_tra_errors_hist_mean_2D_',NB,'.png']);
        FILE_RESULTS_3D = fullfile(OUT,['06_tra_errors_prob_dists_3D_',NB,'.png']);
        LR_BINS = num_lr_bins;
        ERR_BINS = num_err_bins;
        SELECTED_INDEXES = selected_indexes(1:NUM_PROB_SAMPLES); % samples selected for the plots
        %delete(FILE_RESULTS_1D);
        %delete(FILE_RESULTS_2D);
        %delete(FILE_RESULTS_3D_prob);
        tic;
        [prob_err_cube, err_cube, count_cube, edg_err, edg_lr, mean_err_cube, new_class_position_by_index, colors] = save_plots_errorsProbabilities_by_value_class(CELL_LR_MAPS,CELL_ERR_MAPS,CELL_CLASS_MAPS,FILE_RESULTS_1D,FILE_RESULTS_2D,FILE_RESULTS_3D,LR_BINS,ERR_BINS,LABELS,REMOVE_OUTLIERS,SELECTED_INDEXES);
        toc;
        
        list_prob_err_cube{i} = prob_err_cube;
        list_prob_err_cube{i} = prob_err_cube;
        list_err_cube{i} = err_cube;
        list_count_cube{i} = count_cube;
        list_edg_err{i} = edg_err;
        list_edg_lr{i} = edg_lr;
        list_mean_err_cube{i} = mean_err_cube;
        
    end

    disp('--saving results');
    save(FILE_RES, 'list_prob_err_cube', 'list_err_cube', 'list_count_cube', 'list_edg_err', 'list_edg_lr', 'list_mean_err_cube', 'new_class_position_by_index', 'colors', '-v7.3');

end



disp('----------------');
disp('-- TEST PHASE --');
disp('----------------');
disp('6-Estimating test errors...');
NAMES = ALL_NAMES(ID_TST); % test
S3_SIZE = [366,366];

MSE_TABLE_RESULTS = zeros(numel(NAMES),5+numel(LIST_BINS)); % rows --> (tst images) x (cols --> lin, ridge, svr, gpr, btree, proposed_BIN_1, proposed_BIN_2, ...)

for i=1:numel(NAMES)
           
    FILE_CHL = fullfile(OUT,['tst_CHL_',NAMES{i}]);
    FILE_MSE = fullfile(OUT,'tst_MSE.txt');
    
    if isfile(FILE_CHL) && isfile(FILE_MSE)   
        disp(['--pre-loading ',NAMES{i}]);
        load(FILE_CHL); % 'lr_chl','list_exp_err_map','gt_err','list_diff_exp_gt_err_map','labels','PATCHES','lin_err_map','rid_err_map','svr_err_map','gpr_err_map','btree_err_map'
        
    else
    
        disp(['--loading s2 image ',NAMES{i}]); 
        s2_uint16 = loadfield(fullfile(DATA_FOLDER,NAMES{i}));
        s2 = quantify_product(s2_uint16); % reflectance [0,1]
        disp('--simulating s3');
        ratio = size(s2,1)/S3_SIZE(1);
        % simulation_type = {'mean', 'bicubic','psf'};
        s3_uint16 = simulate_s3(s2_uint16,simulation_type,ratio);
        s3 = quantify_product(s3_uint16); % reflectance [0,1]
        
        disp('--computing indexes');
        if strcmpi(bio_prod,'psrinir')        
            s2_index = compute_psri_nir(s2,'s2');
            s3_index = compute_psri_nir(s3,'s2'); % the simulated s3 contains the s2 bands        
        elseif strcmpi(bio_prod,'ndvi45')
            s2_index = compute_ndvi45(s2,'s2');
            s3_index = compute_ndvi45(s3,'s2'); % the simulated s3 contains the s2 bands
        else
            ERROR_STR = ['Unrecognized biophysical product --> "', bio_prod, '"'];
            error(ERROR_STR);
        end

        gt_chl = s2_index;
        lr_chl = s3_index;               
        
        disp('--averaging s2 index');
        mean_gt_chl = simulate_s3(gt_chl,'mean',ratio); % patch mean
        
        disp('--computing s2 patterns for each s3 pixel');                            
        disp('----generating spatial information');        
        IMG_DATA = extract_spatial_data(s2, selected_data, SN, DR, AD);        
        disp('----extracting patches');        
        PATCHES = im2col(IMG_DATA,PATCH_SIZE,'distinct')'; % patches in rows
        disp('----assigning clusters');
        [Idx, labels] = assign_spatial_data_to_clusters(PATCHES, C, SYS, assig_alg, size(s3));                                  
        disp('--computing ground-truth index errors');    
        gt_err = compute_errors(mean_gt_chl,lr_chl);
                       
        disp('--estimating regression test errors');                              
        y_tst_gt = gt_err(:);        
        x_tst1 = lr_chl(:); % biophysical values
        [al1,al2,al3] = size(labels); % if the clustering is fuzzy we punt the whole vector
        x_tst2 = reshape(labels,[al1*al2,al3]); % cluster ids (or probs)                                 
        x_tst = zeros([size(x_tst1,1),1+al3]);
        x_tst(:,1) = x_tst1;
        x_tst(:,2:end) = x_tst2;        
        lin_err = predict(mdl_lin, x_tst);
        rid_err = predict(mdl_rid, x_tst);
        svr_err = predict(mdl_svr, x_tst);
        gpr_err = predict(mdl_gpr, x_tst);        
        btree_err = predict(mdl_btree, x_tst);                
        lin_err_map = reshape(lin_err,size(gt_err));
        rid_err_map = reshape(rid_err,size(gt_err)); 
        svr_err_map = reshape(svr_err,size(gt_err));                
        gpr_err_map = reshape(gpr_err,size(gt_err));                
        btree_err_map = reshape(btree_err,size(gt_err));        
        MSE_TABLE_RESULTS(i,1) = immse(lin_err_map,gt_err);
        MSE_TABLE_RESULTS(i,2) = immse(rid_err_map,gt_err);
        MSE_TABLE_RESULTS(i,3) = immse(svr_err_map,gt_err);
        MSE_TABLE_RESULTS(i,4) = immse(gpr_err_map,gt_err);
        MSE_TABLE_RESULTS(i,5) = immse(btree_err_map,gt_err);
        
        for j=1:numel(list_prob_err_cube)
                
            disp(['--estimating proposed test errors (',num2str(j),'/',num2str(numel(list_prob_err_cube)),')']);
            
            prob_err_cube = list_prob_err_cube{j};
            mean_err_cube = list_mean_err_cube{j};
            edg_lr = list_edg_lr{j};
            
            [exp_err_map, ~, ~] = estimate_errors_from_value_pattern(prob_err_cube, mean_err_cube, lr_chl, labels, edg_lr, new_class_position_by_index);
            diff_exp_gt_err_map = compute_errors(gt_err,exp_err_map);
            
            list_exp_err_map{j} = exp_err_map;
            list_diff_exp_gt_err_map{j} = diff_exp_gt_err_map;
            
            MSE_TABLE_RESULTS(i,5+j) = immse(exp_err_map,gt_err); 
            
        end
        
        disp('--saving results');
        save(FILE_CHL,'lr_chl','list_exp_err_map','gt_err','list_diff_exp_gt_err_map','labels','PATCHES','lin_err_map','rid_err_map','svr_err_map','gpr_err_map','btree_err_map','-v7.3');        
        disp('--saving table MSE');        
        dlmwrite(FILE_MSE, MSE_TABLE_RESULTS, ' ');
        
    end
       
         
    disp('--saving output images');
        
    % colormaps and ranges
    prod_cmap = product_colormap('psri-nir');
    err_cmap = 'jet';
    class_cmap = sort_rows(colors,new_class_position_by_index);    
    product_clims = [-1,1]; % PSRI-NIR in [-1,1] (green vegetation --> [-0,1,0.2])        
    all_errs = cat(1,list_exp_err_map{:},gt_err);
    err_clims = [min(all_errs(:)), max(all_errs(:))];
      
    
    FILE_TEST = fullfile(OUT,['07_tst_map_', NAMES{i},'.png']);        
    fig=figure;       
    set(fig,'visible','off');
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Product Value Map','FontSize',30);
    imagesc(lr_chl, product_clims);
    axis image;
    axis off;    
    colormap(prod_cmap);
    colorbar;
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
    
    FILE_TEST = fullfile(OUT,['08_tst_map_err_gt_', NAMES{i},'.png']);
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Estimated Expected Error Map','FontSize',30);
    imagesc(gt_err, err_clims);
    axis image;
    axis off;
    colormap(err_cmap);
    colorbar;        
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
    
    
    if ndims(labels)==3 % in the fuzzy case we show a 2D map with the maximum activation cluster
        [~,labels ]= max(labels,[],3);
    end
    FILE_TEST = fullfile(OUT,['09_tst_map_cla_', NAMES{i},'.png']);
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Product Class Map','FontSize',30);
    imagesc(labels);
    axis image;
    axis off;    
    colormap(class_cmap); % to use the same colors for clusters than the previous ones    
    colorbar;       
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
        
        
    FILE_TEST = fullfile(OUT,['10_tst_map_err_lin_', NAMES{i},'.png']);
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Estimated Expected Error Map','FontSize',30);
    imagesc(lin_err_map, err_clims);
    axis image;
    axis off;
    colormap(err_cmap);
    colorbar;        
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
    
    
    FILE_TEST = fullfile(OUT,['11_tst_map_err_rid_', NAMES{i},'.png']);
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Estimated Expected Error Map','FontSize',30);
    imagesc(rid_err_map, err_clims);
    axis image;
    axis off;
    colormap(err_cmap);
    colorbar;        
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
    
    
    FILE_TEST = fullfile(OUT,['12_tst_map_err_svr_', NAMES{i},'.png']);
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Estimated Expected Error Map','FontSize',30);
    imagesc(svr_err_map, err_clims);
    axis image;
    axis off;
    colormap(err_cmap);
    colorbar;        
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
    
    
    FILE_TEST = fullfile(OUT,['13_tst_map_err_gpr_', NAMES{i},'.png']);
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Estimated Expected Error Map','FontSize',30);
    imagesc(gpr_err_map, err_clims);
    axis image;
    axis off;
    colormap(err_cmap);
    colorbar;        
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
    
    
    FILE_TEST = fullfile(OUT,['10_tst_map_err_btree_', NAMES{i},'.png']);
    fig=figure;
    set(fig,'visible','off');    
    pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
    set(gcf,'units','points','position',[100,100,width*2,height*2]);
    %sgtitle('Estimated Expected Error Map','FontSize',30);
    imagesc(btree_err_map, err_clims);
    axis image;
    axis off;
    colormap(err_cmap);
    colorbar;        
    export_fig(FILE_TEST,'-png','-transparent');
    close(fig);
    
    
    for j=1:numel(list_exp_err_map)
        exp_err_map = list_exp_err_map{j};
        FILE_TEST = fullfile(OUT,['14_tst_map_err_exp_',num2str(j),'_',NAMES{i},'.png']);
        fig=figure;
        set(fig,'visible','off');
        pos = get(gcf, 'Position'); width = pos(3); height = pos(4);
        set(gcf,'units','points','position',[100,100,width*2,height*2]);
        %sgtitle('Estimated Expected Error Map','FontSize',30);
        imagesc(exp_err_map, err_clims);
        axis image;
        axis off;
        colormap(err_cmap);
        colorbar;
        export_fig(FILE_TEST,'-png','-transparent');
        close(fig);        
    end
    
end

diary off;
