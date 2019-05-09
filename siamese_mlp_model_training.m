function siamese_mlp_model_training(layers,reqEpochs,series,path2InputFeats,path2SaveModels,path2SaveOutputFeats)

for k=1:size(layers,1)
    intermediateLayers = ([layers{k,1}, layers{k,2}]); intermediateLayers(intermediateLayers==0) = [];
    % Train the network
    opts.hiddenLayer = intermediateLayers;
    opts.reqEpochs = reqEpochs;
    opts.series = series;
    opts.path2InputFeats = path2InputFeats;
    opts.path2SaveModels = path2SaveModels ;
    opts.path2SaveOutputFeats =  path2SaveOutputFeats;
    
    [net, info, ~] = cnn_unsup_siamese('hiddenLayer',opts.hiddenLayer,'reqEpochs',opts.reqEpochs,'series',opts.series,...
        'path2InputFeats',opts.path2InputFeats,'path2SaveModels',opts.path2SaveModels,'path2SaveOutputFeats',opts.path2SaveOutputFeats);
    
end

end
