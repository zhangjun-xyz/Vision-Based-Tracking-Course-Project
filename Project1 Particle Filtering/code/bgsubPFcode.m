% Particle filtering using backgroud subtraction as the detection methods
% zj

%load precomputed background image into bgimage
load soccerbgimage.mat
bgimage = double(bgimage);

%prepare sequence structure for genfilename.m
startframe = 1;
endframe = 2499;
prefix = 'Soccer/Frame';
postfix = '.jpg';
sequence = struct('prefix',prefix,'postfix',postfix,'digits',4,'startframe',startframe,'endframe',endframe);

%initialize by choosing a subsequence and one person to track
fstart = startframe; 
fend = endframe;
fend = fstart+100;  %I just want to run for a 100 frames for now, to demonstrate
fnum = fstart;

%get image frame and draw it
fname = genfilename(sequence,fnum);
imrgb = imread(fname);
figure(1); imagesc(imrgb);

%intialize prior by clicking mouse near center of person 
%you want to track
[x0,y0] = ginput(1);

%number of particles for particle filtering
nsamples = 100;
%prior distribution will be gaussian
priorsigmax = 10;
priorsigmay = 10;
priorsigmavx = 1;
priorsigmavy = 1;
%generate particles from prior distribution
sampx = x0 + priorsigmax*randn(1,nsamples);
sampy = y0 + priorsigmay*randn(1,nsamples);
sampvx = priorsigmavx*randn(1,nsamples);
sampvy = priorsigmavy*randn(1,nsamples);
weights = ones(1,nsamples)/nsamples;
%plot particles
figure(1); imagesc(imrgb); hold on
plot(sampx,sampy,'b.');
hold off; drawnow;

%now start tracking
deltaframe = 2;  %set to 1 for every frame
% just a guess of sample time
dt = 0.03;
boundingBoxW = 14;
boundingBoxH = 46;
imgW = size(imrgb, 1);
imgH = size(imrgb, 2);
for fnum = (fstart+deltaframe): deltaframe : fend
    %get image frame and draw it
    fname = genfilename(sequence,fnum);
    imrgb = imread(fname);
    figure(1); imagesc(imrgb);
    
    %do motion prediction step of Bayes filtering 
    %we will use a deterministic motion model plus
    %additive gaussian noise.
    %we are using simple constant position model 
    %as a simple demonstration; it would be better
    %to use constant velocity.
    motpredsigmax = 10;
    motpredsigmay = 10;
    predx = sampx + sampvx*dt + motpredsigmax*randn(1,nsamples);
    predy = sampy + sampvy*dt + motpredsigmay*randn(1,nsamples);
    
    %compute weights based on likelihood
    %recall weights should be oldweight * likelihood
    %but all old weights are equal, so new weight will
    %just be the likelihood.
    %For measuring likelihood, we are using a mixture
    %model (parzen estimate) based on the locations of
    %the ground truth bounding boxes  Note that this is
    %a semiparametric, multimodal distribtion
    obssigmax = 5;
    obssigmay = 5;
    
    %do background subtraction and thresholding
    bgthresh = 30;
    rgbabsdiff = abs(double(imrgb)-bgimage);
    maxdiff = max(rgbabsdiff,[],3);  %max diff in red green or blue
    bgmask = roicolor(maxdiff,bgthresh,Inf);

    %there surely must be a more efficient way to do the
    %following as a vectorized computation rather than
    %a loop, but I want to just get it right the first time
    weights = ones(1,nsamples);
    for i=1:nsamples
        prob = 0;
        x = predx(i); y=predy(i);
        x1 = int16(x-boundingBoxW/2); x1= min(max(1, x1), imgW);
        y1 = int16(y-boundingBoxH/2); y1= min(max(1, y1), imgH);
        imgCrop = imcrop(bgmask, [x1, y1, boundingBoxW, boundingBoxH]);
        prob = length(find(imgCrop))/boundingBoxW/boundingBoxH;
        weights(i) = prob;
    end
    
    %resample particles according to likelihood weights
    %the resulting samples will then have equal weight
    indices = resampindex(weights);
    sampx = predx(indices);
    sampy = predy(indices);
    %plot resampled particles
    %jitter with a little noise so multiple copies can be seen
    figure(1); imagesc(imrgb); hold on
    plot(sampx+1*randn(1,nsamples),sampy+1*randn(1,nsamples),'b.');
    drawnow
    
end