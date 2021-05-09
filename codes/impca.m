function [output_image_projection, matrix_coeff, scores] = impca(input_image,num_components)

%% Default parameters
if nargin<1
	error('ERROR ''impca'': too few parameters! Usage: impca(input_image[,num_components])');
end
if nargin<2
	num_components = 1;
end
if nargin>2
	error('ERROR ''impca'': too many parameters! Usage: impca(input_image[,num_components])');
end

%% Checking some required conditions
if not(isnumeric(input_image))
	error('ERROR ''impca'': the input image is not of numerical type!');
end
if ndims(input_image)~=3
	error('ERROR ''impca'': the input image is not a 3D array!');
end
[x,y,z] = size(input_image); % input image size
if not(z>=num_components)
	error('ERROR ''impca'': the number of PCA components has to be smaller than the number of input bands!');
end

input_image = double(input_image); % double input format required
input_matrix = reshape(input_image,[x*y,z]); % reshaping the input image bands
[matrix_coeff,scores]  = pca(input_matrix); % computing the pca coefficients
projection  = input_matrix  * matrix_coeff(:,1:num_components); % projecting the data to the desired number of components
output_image_projection = reshape(projection,[x,y,num_components]); % reshaping the the projection to the output size

end