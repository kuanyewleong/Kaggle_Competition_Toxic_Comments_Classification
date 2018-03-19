fprintf('Reading data...\n');

data = readtable("train_data_label1.xlsx",'TextType','string');

% convert label into categorical
data.Label1 = categorical(data.Label1);


% Partition the data into a training partition and a held-out test set. 
% Specify the holdout percentage to be 10%.
cvp = cvpartition(data.Label1,'Holdout',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

% Extract the text data and labels from the partitioned tables
fprintf('Extract the text data and labels from the partitioned tables...\n');
textDataTrain = dataTrain.comment_text;
textDataTest = dataTest.comment_text;
YTrain = dataTrain.Label1;
YTest = dataTest.Label1;

% Preprocess the text
textDataTrain = erasePunctuation(textDataTrain);
textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);

%Train a word embedding with dimension 100
fprintf('Start embedding training...\n');
embeddingDimension = 250;
embeddingEpochs = 1000;

emb = trainWordEmbedding(documentsTrain, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...    
    'Verbose',1)

% Convert document to sequences
documentLengths = doclength(documentsTrain);
sequenceLength = round(mean(documentLengths));

documentsTruncatedTrain = docfun(@(words) words(1:min(sequenceLength,end)),documentsTrain);

% Pad the documents with fewer tokens than the fixed length with zeros.
XTrain = doc2sequence(emb,documentsTruncatedTrain);
for i = 1:numel(XTrain)
    XTrain{i} = leftPad(XTrain{i},sequenceLength);
end

save embedding.mat;

%% Train LSTM
fprintf('Prepare to run LSTM training...\n');
inputSize = embeddingDimension;
outputSize = 512;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


maxEpochs = 2000;
miniBatchSize = 1000;
shuffle = 'every-epoch';

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle,...
    'Verbose',1);

net = trainNetwork(XTrain,YTrain,layers,options);

% Test the trained LSTM
textDataTest = erasePunctuation(textDataTest);
textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);

documentsTruncatedTest = docfun(@(words) words(1:min(sequenceLength,end)),documentsTest);
XTest = doc2sequence(emb,documentsTruncatedTest);
for i=1:numel(XTest)
    XTest{i} = leftPad(XTest{i},sequenceLength);
end


YPred = classify(net,XTest);
YProbability = predict(net,XTest);

accuracy = sum(YPred == YTest)/numel(YPred)

save trained_LSTM_256nodes_minibatch2000.mat;






