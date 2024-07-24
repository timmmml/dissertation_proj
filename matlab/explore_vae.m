numLatent = 32; 
imageSize = [64 64 3];

encoder = [
  imageInputLayer(imageSize, 'Name', 'input', 'Normalization', 'none') 
  convolution2dLayer(3, 32, 'Stride', 2, 'Padding', "same", 'Name', 'conv1')
  reluLayer('Name', 'relu1')
  convolution2dLayer(3, 64, 'Stride', 2, 'Padding', "same", 'Name', 'conv2')
  reluLayer('Name', 'relu2')
  fullyConnectedLayer(2 * numLatent, 'Name', 'fc_encoder')
  samplingLayer('Name', 'encoder_output', 'Z', numLatent)];
];

projectionSize = 
decoder = [
  featureInputLayer(numLatent, 'Name', 'decoder_input')
  transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', "same", 'Name', 'deconv1')
  reluLayer('Name', 'relu1')
  transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', "same", 'Name', 'deconv2')
  reluLayer
