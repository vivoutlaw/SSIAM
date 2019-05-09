classdef nncontrloss < nntest
  properties (TestParameter)
    margin = {1, 10, 100}
  end

  methods
    function [x1, x2, c, dzdy] = getx(test)
      numAttributes = 3 ;
      numImages = 6 ;
      w = 4 ;
      h = 5 ;
      c = single(sign(test.randn(1,1,1, numImages)) > 0) ;
      x1 = test.randn(h, w, numAttributes, numImages) / test.range ;
      x2 = test.randn(h, w, numAttributes, numImages) / test.range ;
      dzdy = test.randn(1,1) / test.range ;
    end
  end

  methods (Test)
    function constmargin(test, margin)
      [x1, x2, c, dzdy] = test.getx() ;
      test.dotest(x1, x2, c, dzdy, margin);
    end

    function varmargin(test, margin)
      [x1, x2, c, dzdy] = test.getx() ;
      margin = single(test.randn(1, 1, 1, numel(c))) * margin / test.range ;
      test.dotest(x1, x2, c, dzdy, margin);
    end
  end

  methods
    function dotest(test, x1, x2, c, dzdy, margin)
      y = vl_nncontrloss(x1, x2, c, 'margin', margin) ;
      [dzdx1, dzdx2] = vl_nncontrloss(x1, x2, c, dzdy, 'margin', margin) ;
      test.der(@(x1) vl_nncontrloss(x1, x2, c, 'margin', margin), x1, ...
        dzdy, dzdx1, 1e-4*test.range) ;
      test.der(@(x2) vl_nncontrloss(x1, x2, c, 'margin', margin), x2, ...
        dzdy, dzdx2, 1e-4*test.range) ;
    end
  end
end
