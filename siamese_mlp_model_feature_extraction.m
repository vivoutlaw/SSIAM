function siamese_mlp_model_feature_extraction(layers,reqEpochs,series,extract_feats_for,path2InputFeats,path2SaveModels,path2SaveOutputFeats)

addSpecialChar2Folder = sprintf('epoch%d',reqEpochs);



for k=1:size(layers,1)
    intermediateLayers = ([layers{k,1}, layers{k,2}]); intermediateLayers(intermediateLayers==0) = [];
    % Load the network

    load(sprintf('%s/%s/%s_%d/net-epoch-%d.mat',path2SaveModels,series,series,intermediateLayers,reqEpochs));

    if isstruct(net)
        net  = dagnn.DagNN.loadobj(net);
    end
    

    % find the variables containing the features of interest
    % fc features (fc1), without relu
    featureVar = 'x1_a';

    % Extract Features of all the episodes for the given series
    Feature_Extraction(series, intermediateLayers, featureVar, net, addSpecialChar2Folder, extract_feats_for, path2InputFeats, path2SaveOutputFeats)


end

end

function [] = Feature_Extraction(series, intermediateLayers, featureVar, net, addSpecialChar2Folder, extract_feats_for, path2InputFeats, path2SaveOutputFeats)

% fc_dimensions for featureVar layer
fc_dimensions = intermediateLayers;
outputDest = sprintf('%s/%s_%s/%d',path2SaveOutputFeats,series,addSpecialChar2Folder,fc_dimensions);

if ~exist(outputDest,'dir')
    mkdir(outputDest)
end

netEval = copy(net);
layerNames = {netEval.layers.name};
netEval.removeLayer([layerNames(~cellfun(@isempty, strfind(layerNames, '_b'))), 'loss']);


netEval = dagnn.DagNN.loadobj(netEval);
move(netEval, 'gpu');

% Path to the list whose data needs to be extracted
dirName = sprintf('%s/%s',path2InputFeats,series);
appendFullPath =  strcat(dirName, '/');
fileExtension = extract_feats_for;

fileList = getAllFiles(dirName, fileExtension, appendFullPath);

for f=1:size(fileList,1)
    
    [PATHSTR,NAME,~] = fileparts(fileList{f,1});
%     fileList{f,1}
    
    series = NAME(1:strfind(NAME,'_')-1);
    season = str2num(NAME(strfind(NAME,'s')+1:strfind(NAME,'s')+2));
    episode = str2num(NAME(strfind(NAME,'e')+1:strfind(NAME,'e')+2));
    
    load(sprintf('%s/%s_s%02de%02d_features_VGG2.mat',PATHSTR,series,season,episode));
    temp_features_vgg = features_vgg; clear features_vgg;
    
    numFrames= {Tracks.numframes};
    numFrames = ([numFrames{:}]);
    max_frames = max(numFrames);
    
    trackId= {Tracks.trackerId};
    trackId = ([trackId{:}]);
    
    
    % Storing features
    features_vgg = zeros([fc_dimensions max_frames size(Tracks,2)], 'single');
    
    
    for i=1:size(Tracks,2)
        
        % Track number and Track length
        tracklen = numFrames(1,i);
        trackNo = trackId(1,i);
        
%         i
        k=1;
        for j=1:tracklen
            

            im_(1,1,1:2048,1) = temp_features_vgg(:,j,i);

            im_gpu_ = gpuArray(im_);
            
            % tell the network to preserve these variables
            netEval.vars(netEval.getVarIndex(featureVar)).precious = true;
            
            % run the DagNN
            netEval.eval({'input_a',im_gpu_}) ;
            
            % retrieve the features you are interested in
            features = netEval.vars(netEval.getVarIndex(featureVar)).value;
            
            % remove singletons (dimensions fc_dimensions x 1)
            features_vgg(:,k,i) = squeeze(gather(features));
            k= k+1;
        end
        
    end
    
    %Saving Mat files
    save(sprintf('%s/%s_s%02de%02d_features_VGG2.mat',outputDest,series,season,episode),'features_vgg','Tracks','-v7.3');
end


end
