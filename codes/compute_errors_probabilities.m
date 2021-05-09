 
function [prob_err_cube, err_cube, count_cube, edg_err, edg_lr] = compute_errors_probabilities(err_map, lr_map, class_map, err_bins, lr_bins, NEW_CLASS_ORDER, VERBOSE)
    
    if nargin<6
        NEW_CLASS_ORDER = [];
    end
    if nargin<7
        VERBOSE = false;
    end
    
    if VERBOSE
        disp('Discretizing data...');
    end
    [dis_err_map,edg_err] = discretize(err_map, err_bins);
    [dis_lr_map,edg_lr] = discretize(lr_map, lr_bins);
    
    if VERBOSE
        disp('Creating co-ocurrence matrix...'); % H(error,S3value,patron)
    end
    N_err = err_bins;
    N_lr = lr_bins;
    
    if size(class_map,2)>1
        FUZZY = true;
        N_cla = size(class_map,2);
    else
        FUZZY = false;
        N_cla = numel(unique(class_map));
    end
        
    N_samples = size(class_map,1);
    if isempty(NEW_CLASS_ORDER)
        NEW_CLASS_ORDER = 1:N_cla;
    end
    comat = uint32(zeros(N_err, N_lr, N_cla));    
    norm_comat = single(zeros(N_err, N_lr, N_cla));
    err_comat = single(zeros(N_err, N_lr, N_cla));
    
    if FUZZY
        for i=1:N_samples            
            for c=1:N_cla
                comat(dis_err_map(i),dis_lr_map(i), NEW_CLASS_ORDER(c)) = comat(dis_err_map(i),dis_lr_map(i),NEW_CLASS_ORDER(c)) + class_map(i,c);
                err_comat(dis_err_map(i),dis_lr_map(i), NEW_CLASS_ORDER(c)) = err_comat(dis_err_map(i),dis_lr_map(i),NEW_CLASS_ORDER(c)) + class_map(i,c)*err_map(i);
            end
        end
        
    else
        for i=1:N_samples
            comat(dis_err_map(i),dis_lr_map(i), NEW_CLASS_ORDER(class_map(i))) = comat(dis_err_map(i),dis_lr_map(i),NEW_CLASS_ORDER(class_map(i))) + 1;
            err_comat(dis_err_map(i),dis_lr_map(i), NEW_CLASS_ORDER(class_map(i))) = err_comat(dis_err_map(i),dis_lr_map(i),NEW_CLASS_ORDER(class_map(i))) + err_map(i);
        end
    end
        
    if VERBOSE
        disp('Normalizing conditional distributions...'); % sum_error p(error/S3value=v1,patron=p1) = 1
    end
    for i=1:N_lr
        for j=1:N_cla
            err_lrcla = single(comat(:,i,j));
            sum_err_lrcla = sum(err_lrcla);
            if (sum_err_lrcla>0)
                norm_comat(:,i,j) = err_lrcla ./ repmat(sum_err_lrcla,[N_err,1]);
            end
        end
    end
    
    prob_err_cube = permute(norm_comat,[2,1,3]); % returning prob_err_cube(lr,err,class)
    err_cube = permute(err_comat,[2,1,3]);
    count_cube = permute(comat,[2,1,3]);
end
