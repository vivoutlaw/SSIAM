function setup_siamese_metriclearning()

run /home/vsharma/Codes/matconvnet_12_2017/matlab/vl_setupnn;
vl_contrib setup siamese-mnist
%SETUP_SIAMESE_MNIST Sets up siamese-metriclearning, by adding its folders to the Matlab path
root = fileparts(mfilename('fullpath')) ;
addpath(root, fullfile(root, 'matlab')) ;
end

