import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from proj5.CarND_Vehicle_Detection.functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque

windows = []

def process_image(image):
    global hm_history, windows
    imgx = image.shape[1]
    imgy = image.shape[0]
    if len(windows)==0:
        # search boxes on two sizes, smaller for cars far away (up in the image), bigger ones for nearer cars
        windows.extend(slide_window((imgx, imgy), [imgx//6, 5*imgx//6], [380, 500], xy_window=(64, 64), xy_overlap=(0.75,0.75)))
        if debug:
            window_img = draw_boxes(image, windows, color=(0, 255, 0), thick=4)
            displayImage(window_img)

        windows.extend(slide_window((imgx, imgy), [None, None], [380, 700], xy_window=(112, 100), xy_overlap=(0.75,0.75)))
        if debug:
            window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=4)
            displayImage(window_img)

    # classify content of search boxes using the trained linear svc
    hot_windows = search_windows(img=image, windows=windows, clf=svc, scaler=scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                 spatial_feat=True, hist_feat=True, hog_feat=True)
    # intermediate result: bounding boxes on detected cars
    if debug:
        window_img = draw_boxes(image, hot_windows, color=(255, 0, 0), thick=4)
        displayImage(window_img, title='hot windows')

    # construct heat map to join multiple detection windows
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    if debug:
        displayImage(heat, cmap='hot', title='heatmap')

    if single_image:
        heat = apply_threshold(heat, 3)
    else:
        hm_history.append(heat)
        if len(hm_history) == max_history:
            # Apply threshold to help remove false positives
            heat = apply_threshold(sum(hm_history), max_history+4)
        else:
            heat = apply_threshold(heat, 3)

    if debug:
        displayImage(heat, cmap='hot', title='hot with threshold')
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image, labels)
    if debug:
        displayImage(draw_img, title='Detected cars')
    return draw_img


def save_image(img):
    global i
    # imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print("saving image %d" % i)
    cv2.imwrite("output_images/image%d.jpg" % i, img)
    i += 1
    return img


print("Reading classifier from pickle file")
dist_pickle = pickle.load(open("svc.p", "rb"))
svc = dist_pickle["svc"]
scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
hog_channel = dist_pickle["hog_channel"]
color_space = dist_pickle["color_space"]
training_size = dist_pickle["training_size"]

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
max_history = 10
hm_history = deque(maxlen=max_history)

single_image = True
debug = False

if single_image:
    print("reading image")
    image = loadRGBImage(r'D:\archi\ernesto\cursos\self-driving car\proj5\CarND_Vehicle_Detection\test_images\test6.jpg')
    print("processing")
    final_img = process_image(image)
    plt.imshow(final_img)
    plt.show()
else:
    i = 0
    video = VideoFileClip(r"D:\archi\ernesto\cursos\self-driving car\proj5\CarND_Vehicle_Detection\project_video.mp4")
    video_clip = video.fl_image(process_image)
    video_clip.write_videofile(
        r"D:\archi\ernesto\cursos\self-driving car\proj5\CarND_Vehicle_Detection\output_images\project_video.mp4",
        audio=False)
