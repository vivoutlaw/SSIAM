function [WCP] = siamese_mlp_model_feature_evaluation(layers,reqEpochs, eval_type,  series, eval_feats_for,path2InputFeats,path2SaveOutputFeats)

% Series name
% series = 'bbt';
% eval_type = 'track_level'; % frame_level track_level
% labels_to_remove = 2;

% ,path2InputFeats,path2SaveOutputFeats

if strcmp(series,'bbt')
    labels_to_remove = 2;
end


addSpecialChar2Folder = sprintf('epoch%d',reqEpochs);

for k=1:size(layers,1)
    intermediateLayers = ([layers{k,1}, layers{k,2}]); intermediateLayers(intermediateLayers==0) = [];
    outputFeatureDimensions = intermediateLayers(1,end);
    
    mlp_model_name = sprintf('%d',intermediateLayers(1,1));
    dirName = sprintf('%s/%s_%s/%s',path2SaveOutputFeats,series,addSpecialChar2Folder,mlp_model_name);

%     fprintf('----------------------------------------------------\n MLP Model: %s. Eval Type: %s \n ----------------------------------------------------\n',mlp_model_name,eval_type);
    
    [WCP] = Feature_Evaluation(dirName, outputFeatureDimensions, labels_to_remove, eval_type, mlp_model_name, eval_feats_for, path2InputFeats);
    
    
end

end

function [WCP] = Feature_Evaluation(dirName, outputFeatureDimensions, labels_to_remove, eval_type, mlp_model_name, eval_feats_for, path2InputFeats)

% Path to the list whose data needs to be extracted
appendFullPath =  strcat(dirName, '/');
fileExtension = eval_feats_for;

fileList = getAllFiles(dirName, fileExtension, appendFullPath);

% eval_type = 'frame_level'; % frame_level track_level
NC=outputFeatureDimensions; 

clustering_type = {'hac'}; % hac and kmeans

for f=1:size(fileList,1)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Data Preprocessing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [PATHSTR,NAME,~] = fileparts(fileList{f,1});
    %     fileList{f,1}
    
    series = NAME(1:strfind(NAME,'_')-1);
    season = str2num(NAME(strfind(NAME,'s')+1:strfind(NAME,'s')+2));
    episode = str2num(NAME(strfind(NAME,'e')+1:strfind(NAME,'e')+2));
    
    temp = strfind(NAME,'_');
    method = NAME(temp(end)+1:end);
    
    if strcmp (eval_type,'track_level')
        [data_track, track_labels, Tracks]= prep_tvseries_track(fileList{f,1}, path2InputFeats);
        raw_features = data_track;
        labels = track_labels;
        clear data_track  track_labels
        
    elseif strcmp(eval_type,'frame_level')
        [data_frame, ~, frame_labels, ~, Tracks]= prep_tvseries(fileList{f,1}, path2InputFeats);
        raw_features = data_frame;
        labels = frame_labels;
        clear data_frame frame_labels
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Label processing: removing false accepts and unknowns.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Remove unknowns and false accepts ids
    rm = [labels_to_remove max(labels)];
    
    [remove_idx] = remove_labels(labels,rm);
    labels(remove_idx) = [];
    raw_features(remove_idx,:) = [];
    
    % Number of clusters
    Num_clusters = size(unique(labels),1);
    
    for j=1:size(NC,2)
        %         nc
        nc = NC(1,j);
        
        features = raw_features;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Agglomerative Clustering
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [c] = clustering(features,Num_clusters,clustering_type);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  Accuracy only
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [ClustScores,bcube,~]=computescores(c,length(unique(labels)), labels);
        WCP = ClustScores.wcp;
        
        fprintf('MLP_Model:%s. Series: %s_s%02de%02d. Method: %s.  Removed_Labels: %s. WCP: %s, B3-P: %s, B3-R: %s, B3-F1: %s\n',...
            mlp_model_name, series, season, episode, method,  num2str(rm), num2str(ClustScores.wcp), num2str(bcube.Precision), num2str(bcube.Recall), num2str(bcube.Fscore));
    end
end

end