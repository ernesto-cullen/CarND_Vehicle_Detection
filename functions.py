import glob
import cv2
import numpy as np
from numpy import random
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Read in cars and notcars filenames.
def getTrainingImageLists(smallset=True, maxImages=0):
    print("Reading image filenames")
    if smallset:
        dir_veh = r'D:\archi\ernesto\cursos\self-driving car\proj5\vehicles_smallset\**\*.jpeg'
        dir_notveh = r'D:\archi\ernesto\cursos\self-driving car\proj5\non-vehicles_smallset\**\*.jpeg'
    else:
        dir_veh = r'D:\archi\ernesto\cursos\self-driving car\proj5\vehicles\**\*.png'
        dir_notveh = r'D:\archi\ernesto\cursos\self-driving car\proj5\non-vehicles\**\*.png'

    images = glob.glob(dir_veh, recursive=True)
    cars = []
    i = 0
    for image in images:
        if maxImages > 0 and i >= maxImages:
            break
        cars.append(image)
        i += 1

    i = 0
    images = glob.glob(dir_notveh, recursive=True)
    notcars = []
    for image in images:
        if maxImages > 0 and i >= maxImages:
            break
        notcars.append(image)
        i += 1

    random.shuffle(cars)
    random.shuffle(notcars)
    return cars, notcars


# return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# compute binned color features
def bin_spatial(img, color_space='RGB', size=(64, 64)):
    image = convertColor(img, color_space)
    features = cv2.resize(image, size).ravel()
    return features


# compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


# extract features from a list of images given their filenames
# this function calls bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(64, 64),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, window_size=(64,64),
                     spatial_feat=True, hist_feat=True, hog_feat=True,
                     hog_colorspace='YCrCb'):
    features = []
    for file in imgs:
        # load file always in RGB
        feature_image = loadRGBImage(file)
        # and compute its feature vector
        file_features = img_features(img=feature_image, spatial_size=spatial_size, color_space=color_space,
                                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block, hog_channel=hog_channel, window_size=window_size,
                                     spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                     hog_colorspace=hog_colorspace)
        features.append(file_features)
    return features


def convertColor(image, color_space):
    if color_space == 'RGB':
        feature_image = image
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    return feature_image  # extract features from a single image


def img_features(img, spatial_size, color_space, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                 window_size, spatial_feat, hist_feat, hog_feat, hog_colorspace):
    # resize the image to the processing size. It must be the same as in training to get same feature vector length
    feature_image = cv2.resize(img, window_size)
    file_features = []
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, color_space=color_space, size=spatial_size)
        # print("spatial features: ",len(spatial_features))
        file_features.append(spatial_features)
    if hist_feat:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # print("hist features: ",len(hist_features))
        file_features.append(hist_features)
    if hog_feat:
        hog_image = convertColor(feature_image, hog_colorspace)
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(hog_image.shape[2]):
                hog_features.extend(get_hog_features(hog_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(hog_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # print("hog features: ",len(hog_features))
        file_features.append(hog_features)
    return np.concatenate((file_features))


# takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
# returns a list of windows
def slide_window(img_shape=(1280,768), x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img_shape[0]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img_shape[1]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)+1
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)+1
    # Initialize a list to append window positions to
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            # adjust last column to span all width
            if x_start_stop[1] - startx < xy_window[0]:
                endx = x_start_stop[1]
            else:
                endx = startx + xy_window[0]

            starty = ys * ny_pix_per_step + y_start_stop[0]
            # adjust last row to span all height
            if y_start_stop[1] - starty < xy_window[1]:
                endy = y_start_stop[1]
            else:
                endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
# the image is already in desired color space
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(64, 64), hist_bins=32,
                   orient=9, pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, window_size=(64,64), spatial_feat=True,
                   hist_feat=True, hog_feat=True,
                   hog_colorspace='YCrCb'):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    img = convertColor(img, color_space)
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], spatial_size)
        # 4) Extract features for that window using single_img_features()
        features = img_features(test_img, color_space=None,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, window_size=window_size,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                hog_colorspace=hog_colorspace)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(features.reshape(1, -1))
        # 6) Predict
        prediction = clf.predict(test_features)
        decision = clf.decision_function(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1 and decision > 0.4:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# loads an rgb image
def loadRGBImage(path):
    # print("Load RGB image")
    # img = mpimg.imread(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# display an image, opcionally with color map and title
def displayImage(img, cmap=None, title=""):
    if not cmap is None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.show()
