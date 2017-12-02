netfile = 'model.h5'; 
layers = importKerasLayers(netfile, 'ImportWeights', true); 
layers.Layers(3)=[];
layers.Layers(15,1)=[];