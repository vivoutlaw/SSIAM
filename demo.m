function [] = demo()
% export CUDA_VISIBLE_DEVICES=4
% LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab

% Add path to matconvnet 
path2MatconNet = '/home/vsharma/Codes/matconvnet_12_2017/matlab';
run sprintf('%s/vl_setupnn.m',path2MatconNet);
addpath(fullfile(vl_rootnn, 'examples', 'mnist'));

%Add path for Siamese model and Contrastive loss 
addpath(fullfile('matlab')) ;

% Evaluation metrics
addpath('metrics')

%% Making directory for input data, saving models, and output data

path2InputFeats = 'experiment/input_data';
path2SaveModels = 'experiment/models';
path2SaveOutputFeats = 'experiment/output_data';

if ~exist(path2InputFeats,'dir')
    mkdir(path2InputFeats);
end

if ~exist(path2SaveModels,'dir')
    mkdir(path2SaveModels);
end

if ~exist(path2SaveOutputFeats,'dir')
    mkdir(path2SaveOutputFeats);
end

%% Parameters
layers = {512 0}; % MLP Models
series = 'bbt';   % Series name
reqEpochs = 15;   % Maximum Number of Epochs

if strcmp(series,'bbt')
    extract_feats_for = '*e01_features_VGG2.mat'; eval_feats_for  = '*e01_features_VGG2.mat';
else
    display('Define variables for your datasets!');
end

%% Model Training
siamese_mlp_model_training(layers,reqEpochs,series,path2InputFeats,path2SaveModels,path2SaveOutputFeats);

%% Feature Evaluation and Evaluation

% Feature_Extraction
siamese_mlp_model_feature_extraction(layers,reqEpochs,series,extract_feats_for,path2InputFeats,path2SaveModels,path2SaveOutputFeats);

% Evaluation
eval_type = 'track_level'; % 
temp(i,2) = siamese_mlp_model_feature_evaluation(layers, reqEpochs, eval_type, series,  eval_feats_for,path2InputFeats,path2SaveOutputFeats);

eval_type = 'frame_level'; %  
temp(i,3) = siamese_mlp_model_feature_evaluation(layers, reqEpochs, eval_type, series,  eval_feats_for,path2InputFeats,path2SaveOutputFeats);

end