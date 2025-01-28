% Define input image size for grayscale images
inputSize = [227 227 1]; 

% Load your initial imageDatastore
imdsTraining = imageDatastore('C:/Users/Admin/Downloads/Downloads/227x227','IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the dataset into training and validation
[imdsTraining, imdsValidation] = splitEachLabel(imdsTraining, 0.7, 'randomize');

% Define your image augmenter
augmenter = imageDataAugmenter(...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10]);

% Create augmented image datastore for training
augmentedImageDataTraining = augmentedImageDatastore(inputSize, imdsTraining, ...
    'DataAugmentation', augmenter);
augmentedImageDataValidation = augmentedImageDatastore(inputSize, imdsValidation, 'DataAugmentation', augmenter);

net = trainedNetwork_1;

% Modify the network for transfer learning
layersTransfer = net.Layers(1:end-3);

% Get true labels from the original image datastore
trueLabelsValidation = imdsValidation.Labels;

%numClasses = numel(categories(trueLabelsValidation));
numClasses = numel(unique(trueLabelsValidation));

% Ensure numClasses is positive
numClasses1 = max(numClasses, 1);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses1,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam',...
    'MiniBatchSize', 128,...
    'MaxEpochs', 2,...
    'InitialLearnRate', 1e-4,...
    'Shuffle', 'every-epoch',...
    'ValidationData', augmentedImageDataValidation,...
    'ValidationFrequency', 50,...
    'Verbose', false,...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(imdsTraining,layers,options);

% Perform network pruning manually
prunedLayers = layersTransfer;

for i = 1:numel(prunedLayers)
    if isa(prunedLayers(i), 'nnet.cnn.layer.FullyLayer') || ...
       isa(prunedLayers(i), 'nnet.cnn.layerolution2DLayer') || ...
       isa(prunedLayers(i), 'nnet.cnn.layerolution3DLayer')
        weights = prunedLayers(i).Weights;
        biases = pruned(i).Bias;
        
        % Define your pruning criterion
        threshold = 0.01; % Example threshold
        
        % Apply pruning criterion (set small weights to zero)
        prunedIndices = abs(weights) < threshold;
        weights(prunedIndices) = 0;
        
        % Apply the pruning criterion to biases array (ensure dimensions match)
        biasedIndices = any(prunedIndices, 1); % Identify columns with pruned weights
        biases(:, biasedIndices) = 0; % Set biases corresponding touned weights to 0
        
        % Set pruned weights and biases back to the layer
        prunedLayers(i).Weights = weights;
        pruned(i).Bias = biases;
    end
end

% Define the rest of the layers for the pruned network
layersPruned = [
    prunedLayers
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Train the pruned network
netPruned = trainNetwork(imdsTraining, layersPruned, options);

% Evaluate the pruned network
YPred = classify(netPruned, augmentedImageDataValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == trueLabelsValidation);

% Optionally fine-tune the pruned network
optionsFineTune = trainingOptions('adam', ...
    'MiniBatchSize', 128, ...
    'MaxEpochs', 2, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedImageDataValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

netPrunedFineTuned = trainNetwork(imdsTraining, layersPruned, optionsFineTune);

% Calculate the confusion matrix
confMatrix = confusionmat(YValidation, YPred);

% Display the confusion matrix
figure;
confusionchart(confMatrix, categories(imdsValidation.Labels));

save('prunednet.mat', 'netPrunedFineTuned');