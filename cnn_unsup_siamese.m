function [net, info, imdb] = cnn_unsup_siamese(varargin)
% CNN_MNIST_SIAMESE  Demonstrated MatConNet on MNIST Siamese network

opts.batchNormalization = false ;
opts.hiddenLayer=[512];
opts.network = [] ;
opts.reqEpochs = 1000;
opts.series = 'SERIES';

opts.path2InputFeats = 'INPUT_PATH';
opts.path2SaveModels = 'MODEL_PATH' ;
opts.path2SaveOutputFeats =  'OUTPUT_PATH';

series = opts.series;
opts.train.continue = true ;
[opts, varargin] = vl_argparse(opts, varargin) ;


% BBT
if strcmp(opts.series,'bbt')
    series = 'bbt';  season = 1; episode = 1;
else
    display('Define variables for your datasets!');
end


sfx = 'siam-dagnn' ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir =   sprintf('%s/%s/%s_%d',opts.path2SaveModels,series,series,opts.hiddenLayer);
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = sprintf('%s/%s',opts.path2InputFeats,series);

opts.seed = 1;
opts.train = struct('numEpochs', opts.reqEpochs);
opts = vl_argparse(opts, varargin) ;

if ~isfield(opts.train, 'gpus')
    opts.train.gpus = [1];
end

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if isempty(opts.network)
    net = cnn_mlp_init('batchNormalization', opts.batchNormalization, ...
        'networkType', 'dagnn','hiddenLayer',opts.hiddenLayer) ;
    %   net.removeLayer({'layer7', 'layer8', 'top1err', 'top5err'});
    if size(opts.hiddenLayer,2) ==1
        net.removeLayer({'layer3', 'layer4', 'top1err', 'top5err'});
        net.addLayer('emb', ...
            dagnn.Conv('size', [1 1 opts.hiddenLayer(1,1) 2], 'stride', 1, 'pad', 0), ...
            'x2', 'emb', {'emb_f', 'emb_b'}) ;
    end
    net.initParams(); net.rebuild();
    net = vl_create_siamese(net, net, 'mergeParams', true);
    net.addLayer('loss', dagnn.ContrastiveLoss(), ...
        [net.getOutputs(), 'label'], {'objective'}, {});
else
    net = opts.network ;
    opts.network = [] ;
end


% loading data and setting imdb 
imdb.dataDir = opts.dataDir;
imdb.series = series;
imdb.episode = episode;
imdb.season = season;


imdb.vggfeaturespath  = [imdb.dataDir, '/', sprintf('%s_s%02de%02d_features_VGG2.mat',imdb.series,imdb.season,imdb.episode)];
[imdb.images.rawData.features_vgg, ~, ~, ~, ~]= prep_tvseries_no_norm(imdb.vggfeaturespath);
imdb.images.data_mean = mean(imdb.images.rawData.features_vgg)';
imdb.images.rawData.features_vgg = imdb.images.rawData.features_vgg';

imdb = cnn_metriclearning_pdist2_setup_data(opts,imdb);

q = RandStream('mt19937ar','Seed',opts.seed);


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, build_get_batch(opts,q), ...
    'expDir', opts.expDir,...
    net.meta.trainOpts, opts.train, ...
    'train', find(imdb.images.set==1)) ;



% --------------------------------------------------------------------
function fn = build_get_batch(opts,q)
    % --------------------------------------------------------------------
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) get_siamese_batch(bopts,x,y,q) ;
end

% --------------------------------------------------------------------
function inputs = get_siamese_batch(opts, imdb, batch,q)
    % --------------------------------------------------------------------


    max_number_of_samples = 3000; % increased max samples from 1000 to 3000
    index_to_max_number_of_samples =  randi(size(imdb.images.id,2),1,max_number_of_samples);

    temp= imdb.images.rawData.features_vgg(:,imdb.images.data(1,index_to_max_number_of_samples));


    temp = temp';
    orig_dist= fast_euc_dist(temp);

    [distance, initial_rank]=sort(orig_dist,2,'ascend');

    pos_dist = distance(:,2); [pos_val,pos_index]=sort(pos_dist,'descend');  %changed to ascend
    ir=initial_rank(:,2);
    pos_index1=ir(pos_index);

    batch_size=max(size(batch));
    neg_dist = distance(:,end); [neg_val,neg_index]=sort(neg_dist,'ascend');
    ir_=initial_rank(:,end);
    neg_index1=ir_(neg_index);


    temp = temp';
    if mod(batch_size,2)==0
        imagesA(1,1,1:2048,1:size(batch,2)) = [temp(:,pos_index(1:batch_size/2)), temp(:,neg_index(1:batch_size/2))];
        imagesB(1,1,1:2048,1:size(batch,2)) = [temp(:,pos_index1(1:batch_size/2)), temp(:,neg_index1(1:batch_size/2))];

        labels = zeros(1,batch_size);
        labels(1:batch_size/2) = 1;
        labels(batch_size/2+1:end) = 0;
    elseif mod(batch_size,2)==1
        batch_size = batch_size-1;
        imagesA(1,1,1:2048,1:size(batch,2)) = [temp(:,pos_index(1:batch_size/2)), temp(:,neg_index(1:batch_size/2+1))];
        imagesB(1,1,1:2048,1:size(batch,2)) = [temp(:,pos_index1(1:batch_size/2)), temp(:,neg_index1(1:batch_size/2+1))];

        labels = zeros(1,batch_size+1);
        labels(1:batch_size/2) = 1;
        labels(batch_size/2+1:end) = 0;

    end

    if opts.numGpus > 0
        imagesA = gpuArray(imagesA) ;
        imagesB = gpuArray(imagesB) ;
    end
    inputs = {'input_a', imagesA, 'input_b', imagesB, 'label', labels} ;
end
end
