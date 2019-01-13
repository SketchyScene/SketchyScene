%% set the dataset type here, ['test', 'val']
dataset_type = 'test';
image_id = 1;
data_base_dir = '../../../data/';

%% 1
% Read the sample image in
img_path = [data_base_dir, dataset_type, '/DRAWING_GT/'];
img_path = strcat(img_path, sprintf( 'L0_sample%d.png', image_id));
im = imread(img_path);
im = im(:, :, 1);
edgeim = 1 - im;
figure(1), imshow(edgeim);

%% 2

% Link edge pixels together into lists of sequential edge points, one
% list for each edge contour. A contour/edgelist starts/stops at an 
% ending or a junction with another contour/edgelist.
% Here we discard contours less than 10 pixels long.
[edgelist, labelededgeim] = edgelink(edgeim, 10);
figure(4), imshow(labelededgeim);

% Display the edgelists with random colours for each distinct edge 
% in figure 2
drawedgelist(edgelist, size(im), 1, 'rand', 2); axis off   

%% 3

% Fit line segments to the edgelists
tol = 2; % Line segments are fitted with maximum deviation from
            % original edge of 2 pixels.
seglist = lineseg(edgelist, tol);

% Draw the fitted line segments stored in seglist in figure window 3 with
% a linewidth of 2 and random colours
drawedgelist(seglist, size(im), 2, 'rand', 3); axis off