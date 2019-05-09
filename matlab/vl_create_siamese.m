function net = vl_create_siamese(net_a, net_b, varargin)
%VL_CREATE_SIAMESE Create a siamese network
%  NET = VL_CREATE_SIAMESE(NET_A, NET_B) Creates a siamese network by
%  combining the NET_A and NET_B. Does not add a loss at top, which can be
%  easily done with:
%
%  NET.addLayer('loss', dagnn.PDist('aggregate', true), ...
%      NET.getOutputs(), {'objective'}, {});
%
%  By default renames all network parameters, variables and layers to
%  prevent any name clashes.
%
%  Additionally accepts the following arguments:
%
%  mergeParams :: [false]
%    If true, does not rename the parameters but merges the parameters
%    between two networks. Useful for weight sharing between two networks.
%
%  cutoffA :: ''
%    If specified, cutoff layer of NET_A. Will keep only layers of NET_A up
%    to NET_A.getLayerIndex(cutoffA)+cutOffAOffset.
%   
%  cutOffAOffset :: 0
%    If not 0, keep +cutOffAOffset layers in NET_A.
%
%  cutOffB, cutOffBOffset :: ;;, 0
%    Same as cutOffA and cutOffAOffset but for the second network.

% Copyright (C) 2017 Karel Lenc.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


opts.mergeParams = false;
opts.cutoffA = [];
opts.cutoffAOffset = 0;
opts.cutoffB = [];
opts.cutoffBOffset = 0;
opts.singleInput = false;
opts = vl_argparse(opts, varargin);

if iscell(net_a.layers)
  net_a = dagnn.DagNN.fromSimpleNN(net_a, 'CanonicalNames', true);
end
if iscell(net_b.layers)
  net_b = dagnn.DagNN.fromSimpleNN(net_b, 'CanonicalNames', true);
end

% Work on copies
net_a = copy(net_a);
net_b = copy(net_b);

  function net = cutoff(net, layer, offset)
    lidx = min(net.getLayerIndex(layer) + offset, numel(net.layers));
    eord = net.getLayerExecutionOrder();
    lidx_eord = find(eord == lidx);
    remove_layers = eord((lidx_eord+1):end);
    for rli = numel(remove_layers):-1:1
      % Params and variable are automatically removed
      net.removeLayer(net.layers(remove_layers(rli)).name);
    end
  end

if ~isempty(opts.cutoffA)
  cutoff(net_a, opts.cutoffA, opts.cutoffAOffset);
end
if ~isempty(opts.cutoffB)
  cutoff(net_b, opts.cutoffB, opts.cutoffBOffset);
end

  function rename(type, net, appendix)
    switch type
      case 'vars'
        names = {net.vars.name};
        if opts.singleInput, names(ismember('input', names)) = []; end
        rename_fun = @net.renameVar;
      case 'layers'
        names = {net.layers.name};
        rename_fun = @net.renameLayer;
      case 'params'
        names = {net.params.name};
        rename_fun = @net.renameParam;
    end
    for vi = 1:numel(names)
      rename_fun(names{vi}, [names{vi}, appendix]);
    end
  end

rename('vars', net_a, '_a');
rename('vars', net_b, '_b');
rename('layers', net_a, '_a');
rename('layers', net_b, '_b');

if opts.mergeParams
  all_params = unique([{net_a.params.name}, {net_b.params.name}]);
  [param_in_a, pia_idxs] = ismember(all_params, {net_a.params.name});
  [param_in_b, pib_idxs] = ismember(all_params, {net_b.params.name});
  param_in_b(param_in_a) = false;
  net_a_ = net_a.saveobj(); net_b_ = net_b.saveobj();
  params = [net_a_.params(pia_idxs(param_in_a)), ...
    net_b_.params(pib_idxs(param_in_b))];
else
  rename('params', net_a, '_a');
  rename('params', net_b, '_b');
  net_a_ = net_a.saveobj(); net_b_ = net_b.saveobj();
  params = [net_a_.params, net_b_.params];
end

net_ = struct('vars', [net_a_.vars, net_b_.vars], 'params', params, ...
  'layers', [net_a_.layers, net_b_.layers], 'meta', net_a.meta);

net = dagnn.DagNN.loadobj(net_);

end