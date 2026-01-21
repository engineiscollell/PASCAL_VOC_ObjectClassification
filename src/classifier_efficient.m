function example_classifier_efficient
% Add the VOCcode directory to the MATLAB path
addpath([cd '/VOCcode']);
% Load the pre-trained EfficientNet-B0 network
net = efficientnetb0;

featureLayer = 'efficientnet-b0|model|head|dense|MatMul'; % extract features from the selected layer
% Initialize VOC options
VOCinit;
for i=1:VOCopts.nclasses
    cls = VOCopts.classes{i};
    classifier = train(VOCopts, cls, net, featureLayer); % train with SVM
    test(VOCopts, cls, classifier, net, featureLayer);   % test with SVM
    [fp, tp, auc] = VOCroc(VOCopts, 'comp1', cls, true); % ROC curve
    allFP{i} = fp;% store for collage
    allTP{i} = tp;
    allAUC(i) = auc;
    if i < VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
end


fprintf('Display ROC curve collage and line plot for easier data visualization.\n');


% ROC curve collage
figure;
for i = 1:VOCopts.nclasses
    subplot(2,5,i);
    plot(allFP{i}, allTP{i}, 'b-', 'LineWidth', 2);
    axis([0 1 0 1]);
    xlabel('FPR');
    ylabel('TPR');
    title(sprintf('%s (AUC=%.2f)', VOCopts.classes{i}, allAUC(i)));
    grid on;
end
sgtitle('ROC curves per class (EfficientNet-B0, dense layer, SVM, Euclidean distance)');

% Line plot
figure;
plot(allAUC, '-o');
xticks(1:VOCopts.nclasses);
xticklabels(VOCopts.classes);
xtickangle(45);
ylabel('AUC');
title('AUC per class (EfficientNet-B0, dense layer, SVM, Euclidean distance)');
grid on;

% Classifier training
function classifier = train(VOCopts,cls, net, featureLayer)
    [ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
    tic;
    % Pre-allocate feature matrix with the correct dimensions
    classifier.FD = [];
    for i=1:length(ids)
        if toc>1
            fprintf('%s: train: %d/%d\n',cls,i,length(ids));
            drawnow;
            tic;
        end
        try
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            I=imread(sprintf(VOCopts.imgpath,ids{i}));
            fd=extractfd(I, net, featureLayer);
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        
        % Make sure fd is a column vector
        fd = fd(:);
        
        % On first iteration, initialize FD matrix with correct dimensions
        if i == 1
            % Use single precision for feature data to match what kNN expects
            classifier.FD = zeros(length(fd), length(ids), 'single');
        end
        
        % Ensure consistent data type
        fd = single(fd);
        classifier.FD(:,i) = fd;
    end
    
% SVM Classifier training
    classifier.model = fitcsvm(classifier.FD', classifier.gt', ...
    'KernelFunction', 'linear', ...         %%% Modified by SVM
    'Standardize', true, ...                %%% Modified by SVM
    'ClassNames', [-1 1]);  

% Classifier test
function test(VOCopts,cls,classifier, net, featureLayer)
    [ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');
    fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');
    tic;
    for i=1:length(ids)
        if toc>1
            fprintf('%s: test: %d/%d\n',cls,i,length(ids));
            drawnow;
            tic;
        end
        try
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            I=imread(sprintf(VOCopts.imgpath,ids{i}));
            fd=extractfd(I, net, featureLayer);
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        c=classify(classifier,fd); % kNN classification
        fprintf(fid,'%s %f\n',ids{i},c);
    end
    fclose(fid);

% AlexNet feature extraction
function fd = extractfd(I, net, featureLayer)
    inputSize = net.Layers(1).InputSize;
    I = imresize(I, [inputSize(1), inputSize(2)]);
    % Convertir a RGB si és en escala de grisos
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    features = activations(net, I, featureLayer, 'OutputAs', 'columns');  %%% Modified by RESNET-101
    fd = single(features);  % ja és vector columna %%% Modified by RESNET-101
    
    % Ensure consistent feature dimensionality
    expectedLength = 1000;  % MobileNet-v2
    if numel(fd) ~= expectedLength
        warning('Warning in extractfd: feature vector has incorrect size %d', numel(fd), expectedLength);
        fd = zeros(expectedLength, 1, 'single');
    end

% kNN Classifier
function c = classify(classifier, fd)
    % Ensure fd is a row vector with the same number of columns as expected
    fd = fd(:)';  % Convert to row vector 
    
    % Check dimensions
    if size(fd, 2) ~= size(classifier.FD, 1)
        error('Error: Feature dimension mismatch. Expected %d features but got %d.', ...
            size(classifier.FD, 1), size(fd, 2));
    end
    
    % Explicitly convert to the same data type as training data to avoid warnings
    trainType = class(classifier.FD);
    fd = cast(fd, trainType);
    
    [~, score] = predict(classifier.model, fd); % predicció amb SVM
    c = score(2); % puntuació per la classe positiva
