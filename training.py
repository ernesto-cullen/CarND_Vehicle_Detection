import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
from proj5.CarND_Vehicle_Detection.functions import *


cars, notcars = getTrainingImageLists(smallset=True, maxImages=4000)

print("cars: %d, not cars: %d" %(len(cars),len(notcars)))

# display a random car and not car images
# index = random.randint(0, len(cars))
# print(cars[index])
# image = cv2.cvtColor(cv2.imread(cars[index]), cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.show()

# index = random.randint(0, len(notcars))
# print(notcars[index])
# image = mpimg.imread(notcars[index])
# plt.imshow(image)
# plt.show()

#### HOG
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

#### spatial binning
spatial_size = (64, 64)  # Spatial binning dimensions
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

#### histograms
hist_bins = 16  # Number of histogram bins

image_size = (64, 64) #size of training images to match feature vectors
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, window_size=image_size,
                                spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, window_size=image_size,
                                   spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)
print('X_test shape: ',X_test.shape)
print('y_test shape: ',y_test.shape)

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
print('Training...')
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
print('Save trained model')
with open('svc.p', mode='wb') as f:
    pickle.dump({'svc': svc, 'scaler': X_scaler, 'orient': orient, 'pix_per_cell': pix_per_cell, 'hog_channel': hog_channel,
                 'cell_per_block': cell_per_block, 'spatial_size': spatial_size, 'hist_bins': hist_bins,
                 'color_space':color_space, 'training_size':image_size}, f)
