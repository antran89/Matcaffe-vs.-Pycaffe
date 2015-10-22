addpath('/home/tranlaman/BLVC-caffe/matlab')

model_dir = '/home/tranlaman/BLVC-caffe/models/bvlc_reference_caffenet/';
net_model = [model_dir 'deploy.prototxt'];
net_weights = [model_dir 'bvlc_reference_caffenet.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)

% load data
load('test.mat')

% Initialize a network
caffe.set_mode_gpu();
caffe.set_device(0);
tic
net = caffe.Net(net_model, net_weights, phase);
out = net.forward({input_data});
toc

out = out{1};
scores = mean(out, 2);  % take average scores over 10 crops


% print prediction of center cropping
[maxscore, maxlabel] = max(out(:, 9));
fprintf('Predicted class of center cropping is %f.\n', maxlabel)
fprintf('Confidence of prediction is %f.\n', maxscore)

% print final prediction
[maxscore, maxlabel] = max(scores);
fprintf('Final prediction of ten croppings is %f\n', maxlabel)
fprintf('Confidence of prediction is %f.\n', maxscore)

% call caffe.reset_all() to reset caffe
caffe.reset_all();