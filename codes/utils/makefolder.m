function makefolder(folder_name, truncate, cell_subfolder_names)

%% Default parameters
if nargin<1
	error('ERROR ''makefolder'': too few parameters! Usage: makefolder(folder_name[,truncate][,cell_subfolder_names])');
end
if nargin<2
	truncate = false; % by default we do not remove the folders'
end
if nargin<3
	cell_subfolder_names = {};
end
if nargin>4
	error('ERROR ''makefolder'': too many parameters! Usage: makefolder(folder_name[,truncate][,cell_subfolder_names])');
end

% Creating the folder
if exist(folder_name, 'file') == 7
	if truncate
		rmdir(folder_name,'s'); % remove the folder and its content
		mkdir(folder_name);
	end
else
	mkdir(folder_name);
end

% ... and subfolers
NUM_SUBFOLDERS = numel(cell_subfolder_names);
for i=1:NUM_SUBFOLDERS
	sub_folder_name = fullfile(folder_name,cell_subfolder_names{i});
	if exist(folder_name, 'file') == 7
		if truncate
			rmdir(sub_folder_name,'s');
			mkdir(sub_folder_name);
		end
	else
		mkdir(sub_folder_name);
	end
end

end