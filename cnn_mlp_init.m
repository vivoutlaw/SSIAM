function net = cnn_mlp_init(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts.hiddenLayer = 9;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/100 ;
net.layers = {} ;
if size(opts.hiddenLayer,2) ==1
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{f*randn(1,1,2048,opts.hiddenLayer(1,1), 'single'),  zeros(1,opts.hiddenLayer(1,1),'single')}}, ...
        'stride', 1, ...
        'pad', 0) ;
    net.layers{end+1} = struct('type', 'relu') ;
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{f*randn(1,1,opts.hiddenLayer(1,1),10, 'single'), zeros(1,10,'single')}}, ...
        'stride', 1, ...
        'pad', 0) ;
    
elseif size(opts.hiddenLayer,2) ==2
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{f*randn(1,1,2048,opts.hiddenLayer(1,1), 'single'),  zeros(1,opts.hiddenLayer(1,1),'single')}}, ...
        'stride', 1, ...
        'pad', 0) ;
    net.layers{end+1} = struct('type', 'relu') ;
        net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{f*randn(1,1,opts.hiddenLayer(1,1),opts.hiddenLayer(1,2), 'single'),  zeros(1,opts.hiddenLayer(1,2),'single')}}, ...
        'stride', 1, ...
        'pad', 0) ;
    net.layers{end+1} = struct('type', 'relu') ;
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{f*randn(1,1,opts.hiddenLayer(1,2),10, 'single'), zeros(1,10,'single')}}, ...
        'stride', 1, ...
        'pad', 0) ;
    
end
net.layers{end+1} = struct('type', 'softmaxloss') ;

% optionally switch to batch normalization
if opts.batchNormalization
    net = insertBnorm(net, 1) ;
    net = insertBnorm(net, 4) ;
    net = insertBnorm(net, 7) ;
end

% Meta parameters
net.meta.inputSize = [1 2048 1] ;
net.meta.trainOpts.learningRate = 0.001; %logspace(-2, -4, 20) ; % 0.001
net.meta.trainOpts.numEpochs = 20 ; % 20
net.meta.trainOpts.batchSize = 128 ; % 128

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
            {'prediction', 'label'}, 'error') ;
        net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
            'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
    otherwise
        assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{l}.weights{2} = [] ;  % eliminate bias in previous conv layer
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
