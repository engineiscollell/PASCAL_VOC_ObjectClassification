function example_classifier_alex
% Add the VOCcode directory to the MATLAB path
addpath([cd '/VOCcode']);
% Load the pre-trained AlexNet network
net = alexnet;

featureLayer = 'fc7'; % Extract features from the fc7 layer
% Initialize VOC options
VOCinit;
for i=1:VOCopts.nclasses
    cls = VOCopts.classes{i};
    classifier = train(VOCopts, cls, net, featureLayer); % train with kNN
    test(VOCopts, cls, classifier, net, featureLayer);   % test with kNN
    [fp, tp, auc] = VOCroc(VOCopts, 'comp1', cls, true); % ROC curve
    allFP{i} = fp; % Modify for collage
    allTP{i} = tp;
    allAUC(i) = auc;
    if i < VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
end


fprintf('Display ROC curve collage and line plot for easier data visualization:.\n');


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
sgtitle('ROC curves per class (AlexNet, fc7, SVM, Euclidean distance)');

% Line plot
figure;
plot(allAUC, '-o');
xticks(1:VOCopts.nclasses);
xticklabels(VOCopts.classes);
xtickangle(45);
ylabel('AUC');
title('AUC per class (AlexNet, fc7, SVM, Euclidean distance)');
grid on;

% CLASSIFIER TRAINING
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
   % SVM classifier training
    classifier.model = fitcsvm(classifier.FD', classifier.gt', 'KernelFunction','linear', 'Standardize',true);


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
    % Convert to RGB if image is grayscale
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    features = activations(net, I, featureLayer);
    fd = single(features(:));  % Ensure single precision
    
    % Ensure consistent feature dimensionality
    expectedLength = 4096;
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
    
    [~, score] = predict(classifier.model, fd); % kNN prediction
    c = score(2); % positive class score
