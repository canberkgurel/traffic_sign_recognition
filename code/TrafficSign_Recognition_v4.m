%Canberk Suat Gurel, UID: 115595972

clc; close all;
%% Initiate the VL library
run('F:\vlfeat-0.9.21-bin\vlfeat-0.9.21\toolbox\vl_setup')

%% User Input

prompt = {'Is there a trained model named classifier in the workspace? (y for Yes, n for No) If "n" is selected, the classifier will be trained.', ...
    'Do you want to generate a confusion matrix? (y for Yes, n for No)',...
    'Do you want to display the frames while the code is running? (y for Yes, n for No)'};
GUItitle = 'User Input';
dims = [1 82];
definput = {'n','y','y'};
answer = inputdlg(prompt,GUItitle,dims,definput);

%% CLASSIFIER Training
if(answer{1,1} == 'n')
    clearvars -except answer
    training_folder = fullfile('Training-20180505T155054Z-001','Training');
    trainingSet = imageSet(training_folder,'recursive');
    
    trainingFeatures = [];
    trainingLabels   = [];
    
    for i = 1:numel(trainingSet)
        
        numImages = trainingSet(i).Count;
        hog = [];
        
        for j = 1:numImages
            img = read(trainingSet(i), j);
            
            %Resize Image to 64x64
            img = im2single(imresize(img,[64 64]));
            %Get HOG Features
            hog_cl = vl_hog(img, 4);
            [hog_1, hog_2] = size(hog_cl);
            dim = hog_1*hog_2;
            hog_cl_trans = permute(hog_cl, [2 1 3]);
            hog_cl=reshape(hog_cl_trans,[1 dim]);
            hog(j,:) = hog_cl;
        end
        labels = repmat(trainingSet(i).Description, numImages, 1);
        
        trainingFeatures = [trainingFeatures; hog];
        trainingLabels = [trainingLabels; labels];
    end
    
    %SVM
    classifier = fitcecoc(trainingFeatures, trainingLabels);
end
%% CLASSIFIER Testing
if(answer{2,1} == 'y')
    testing_folder = fullfile('Testing-20180505T161422Z-001','Testing');
    testingSet = imageSet(testing_folder,'recursive');
    
    testingFeatures = [];
    testingLabels   = [];
    
    for i = 1:numel(testingSet)
        
        numImages = testingSet(i).Count;
        hog = [];
        for j = 1:numImages
            img = read(testingSet(i), j);
            
            %Resize Image to 64x64
            img = im2single(imresize(img,[64 64]));
            %Get HOG Features
            hog_cl = vl_hog(img, 4);
            [hog_1, hog_2] = size(hog_cl);
            dim = hog_1*hog_2;
            hog_cl_trans = permute(hog_cl, [2 1 3]);
            hog_cl=reshape(hog_cl_trans,[1 dim]);
            hog(j,:) = hog_cl;
        end
        labels = repmat(testingSet(i).Description, numImages, 1);
        
        testingFeatures = [testingFeatures; hog];
        testingLabels = [testingLabels; labels];
    end
    predictedLabels = predict(classifier, testingFeatures);
    confMat = confusionmat(testingLabels, predictedLabels);
    plotConfMat(confMat);
end
%% MAIN ALGORITHM
imagepath = 'Signs';
fileS = dir(fullfile(imagepath, '*.png'));
predLab = [1,14,17,19,21,35,38,45];

files = ls('input-20180505T161721Z-001\input\*jpg');
names = files;
fileDirectory = dir(fullfile('input-20180505T161721Z-001\input', '*image*.jpg'));

v = VideoWriter('Output_8');
v.FrameRate = 30;
open(v);

%% Traffic Sign Detection

h = figure(3);
if(answer{3,1} == 'y')
    set(h, 'Visible', 'on');
elseif(answer{3,1} == 'n')
    set(h, 'Visible', 'off');
end

for fr = 75:1965 %size(names,1) returns the number of frames in the imput folder
    disp(fr)
    full_name= fullfile('input-20180505T161721Z-001\input', fileDirectory(fr).name);
    img = imread(full_name);
    
    %Gaussian Filter
    mu = 5;
    sigma = 2;
    index = -floor(mu/2) : floor(mu/2);
    [X,Y] = meshgrid(index, index);
    H = exp(-(X.^2 + Y.^2) / (2*sigma*sigma));
    H = H / sum(H(:));
    I = imfilter(img, H);
    
    if fr>=2020 && fr<=2034
        R = I(:,:,1);
        G = I(:,:,2);
        B = I(:,:,3);
        R=imadjust(I, stretchlim(I),[0.001 0.999], 0.1);
        I=cat(3,R,G,B);
    end
    
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    
    red = uint8(max(0, min(R-B, R-G)));
    blue = uint8(max(0, min(B-R, B-G)));
    
    bb = im2bw(blue,.15);
    br = im2bw(red,.15);
    
    rb = red + blue;
    
    %Crop the frame
    [yy,xx,~] = size(img);
    
    %Apply mask to the image
    x = [1 xx xx 1];
    y = [1 1 yy/2 yy/2];
    mask = poly2mask(x,y, yy, xx);
    crop = uint8(immultiply(rb,mask));
    
    % MSER
    [r,f] = vl_mser(crop,'MinDiversity',0.7,'MaxVariation',0.2,'Delta',8,'MaxArea',1,'MinArea',0.00043,'DarkOnBright',1);
    
    % Compute matrix M whose value is equal to the number of overlapping extremal regions
    M = zeros(size(crop));
    for x=r'
        s = vl_erfill(crop,x);
        M(s) = M(s) + 1;
    end
    
    thresh = graythresh(M);
    M = im2bw(M, thresh);
    se = strel('octagon',6);
    M = imdilate(M, se);
    
    % Area filter
    M = bwareafilt(M, [950 10000]);
    
    regions = regionprops( M, 'BoundingBox');
    
    %Get Bounding boxes for the blobs given by MSER    
    clf; imagesc(img); hold on ; axis equal off; colormap gray ;
    for k = 1 : length(regions)
        box = regions(k).BoundingBox;
        ratio = box(3)/box(4);
        if ratio < 1.1 && ratio > 0.6 %Aspect Ration of detections
            
            sign = imcrop(img, box);
            sign = im2single(imresize(sign,[64 64]));
            
            %% Traffic Sign Classification
            %Get HOG Features of detections
            hog = [];
            hog_det = vl_hog(sign, 4);
            [hog_1, hog_2] = size(hog_det);
            dim = hog_1 * hog_2;
            hog_det_trans = permute(hog_det, [2 1 3]);
            hog_det=reshape(hog_det_trans,[1 dim]);
            hog = hog_det;
            [predictedLabels, score] = predict(classifier, hog);
            label = str2num(predictedLabels);
            for j = 1:length(score)
                if (score(j) > -0.04)
                    
                    [~,col]=find(predLab== label);
                    if(isempty(col))
                        disp('Detected sign does not belong to the set of desired Signs')
                        continue
                    else
                        rectangle('Position', box,'EdgeColor','g','LineWidth',2 )
                    end
                    
                    full_name= fullfile(imagepath, strcat(predictedLabels,'.png'));
                    im = imread(full_name);
                    im = im2single(imresize(im,[box(4) box(3)]));
                    image([int64(box(1)-box(3)) int64(box(1)-box(3)) ],[int64(box(2)) int64(box(2))],im);
                end
            end
        end
    end
    
    frame = getframe(gca);
    writeVideo(v,frame);
    
end

close(v)
