 
function [clustered_data, img_labels] = assign_spatial_data_to_clusters(PATCHES, CENTROIDS, INFERENCE_SYSTEM, METHOD, OUT_SIZE, DISTANCE)
    
    ERROR_STR = ['Unrecognized clustering method --> "', METHOD, '"'];
  
    if nargin<5
        OUT_SIZE = [floor(sqrt(size(PATCHES,1))),floor(sqrt(size(PATCHES,1)))];
    end  
    if nargin<6
        DISTANCE = 'euclidean';
    end    
    
    PATCHES = double(PATCHES);
    NUM_CENT = size(CENTROIDS,1);
        
    
    if strcmpi(METHOD,'knn') % hard-assignment
        
        clustered_data = knnsearch(CENTROIDS, PATCHES, 'Distance', DISTANCE); % hard-clustering
        img_labels = reshape(clustered_data,[OUT_SIZE(1), OUT_SIZE(2)]); 
        
        
    elseif strcmpi(METHOD,'gauss') % soft-assignment
                               
        clustered_data = EstimatePosteriorsG(PATCHES, NUM_CENT, INFERENCE_SYSTEM.priors, INFERENCE_SYSTEM.mu, INFERENCE_SYSTEM.sigma);            
        img_labels = reshape(clustered_data',[OUT_SIZE(1), OUT_SIZE(2), NUM_CENT]);                                                
        
        
    elseif strcmpi(METHOD,'gmm')  % soft-assignment               
        
        clustered_data = posterior(INFERENCE_SYSTEM, PATCHES); % soft-clustering
        img_labels = reshape(clustered_data,[OUT_SIZE(1), OUT_SIZE(2), NUM_CENT]);
        
                
    else
        
        error(ERROR_STR);
    
    end        
    
    
end
