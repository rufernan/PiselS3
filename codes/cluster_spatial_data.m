 
function [clustered_data, centroids, some_exemplars, inference_system] = cluster_spatial_data(DATA2CLUSTER, K, METHOD, DISTANCE, NUM_EXEMPLARS)
    
    ERROR_STR = ['Unrecognized clustering method --> "', METHOD, '"'];

    if nargin<3
        METHOD = 'kmeans'; % 'kmeans','kmedoids','gmm'
    end    
    if nargin<4
        DISTANCE = 'sqeuclidean'; % only valid for 'kmeans','kmedoids'
    end    
    if nargin<5
        NUM_EXEMPLARS = 4; % num exemplars for each cluster
    end
    
    DATA2CLUSTER = double(DATA2CLUSTER);
    
    if (strcmpi(METHOD,'kmeans') || strcmpi(METHOD,'kmedoids'))
       
        if strcmpi(METHOD,'kmeans')
            [idx,C] = kmeans(DATA2CLUSTER, K, 'Distance', DISTANCE); % samples in rows
        else % 'kmedoids'
            [idx,C] = kmedoids(DATA2CLUSTER, K, 'Distance', DISTANCE); % samples in rows
        end
        
        clustered_data = idx;        
        centroids = C;        
        hard_clustered_data = idx; % for extracting exemplars        
        
        % for a possible fuzzy inference in tst, we compute a gaussian distribution for each cluster
        inference_system = [];
        
        [num_obs, ~] = size(DATA2CLUSTER);        
        for i=1:K
            DATA = DATA2CLUSTER(idx==i,:);
            inference_system.mu(i,:) = mean(DATA); % mean per cluster
            inference_system.sigma(i,:,:) = cov(DATA); % covariance matrix per cluster
            inference_system.priors(i) = size(DATA,1)/num_obs; % prior per cluster
        end
                            
        
        
    elseif strcmpi(METHOD,'gmm')
                
        options = statset('Display','final','MaxIter',100);
        gm = fitgmdist(DATA2CLUSTER, K, 'Options', options, 'Replicates',3);
        
        % computing centroids
        P = posterior(gm,DATA2CLUSTER);
        C = zeros(K,size(DATA2CLUSTER,2));
        for i=1:K
            C(i,:) = sum( DATA2CLUSTER .* repmat(P(:,i),[1,size(DATA2CLUSTER,2)]) ) / sum(P(:,i));
        end
        
        clustered_data = P;
        centroids = C;        
        [~, hard_clustered_data] = max(clustered_data,[],2); % for extracting exemplars
        inference_system = gm;
                
    else
        
        error(ERROR_STR);
    
    end
    
    
    % Extracting some exemplar samples for each cluster
    if NUM_EXEMPLARS>0
        some_exemplars = zeros(K, NUM_EXEMPLARS, size(DATA2CLUSTER,2));
        labels = unique(hard_clustered_data);    
        for i=1:numel(labels)
            data_i = DATA2CLUSTER(hard_clustered_data==labels(i),:);            
            % pick some random samples close to the centroid
            Idx = knnsearch(data_i,centroids(i,:),'K', NUM_EXEMPLARS*2, 'Distance','euclidean');            
            rand_ind = randsample(1:numel(Idx), NUM_EXEMPLARS);
            some_exemplars(i,:,:) = data_i(Idx(rand_ind),:);            
        end
    else
        some_exemplars = [];
    end
    
    
end
