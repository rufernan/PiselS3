function output_data = loadfield(input_struct_file_name,num_field)

%% Default parameters
if nargin<1
	error('ERROR ''loadfield'': too few parameters! Usage: loadfield(input_struct_file_name[,num_field])');
end
if nargin<2
	num_field = 1; % by default we retrieve the first field of the struct
end
if nargin>2
	error('ERROR ''loadfield'': too many parameters! Usage: loadfield(input_struct_file_name[,num_field])');
end

DATA_STRUCT = load(input_struct_file_name); % loading the input struct data
if not(isstruct(DATA_STRUCT))
	error('ERROR ''loadfield'': the input file does not contain a struct data type!');
end

FIELD_NAMES = fieldnames(DATA_STRUCT); % extracting all the field names associated to the input struct
if num_field<1 || num_field>numel(FIELD_NAMES)
	error('ERROR ''loadfield'': problem with the requested field!');
end

SELECTED_FIELD_NAME = FIELD_NAMES{num_field}; % retrieving only the name of the selected one
output_data = DATA_STRUCT.(SELECTED_FIELD_NAME); % output data

end
