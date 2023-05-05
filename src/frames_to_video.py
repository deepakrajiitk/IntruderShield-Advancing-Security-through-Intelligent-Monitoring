import cv2
import os

# set the directory path where the images are stored
directory = "../dataset/test/ch24"

# set the output video file path
output_path = "test.mp4"

# get a list of all the image file names in the directory
image_files = sorted([os.path.join(directory, f)
                     for f in os.listdir(directory) if f.endswith(".jpg")])

# get the dimensions of the first image in the sequence
frame = cv2.imread(image_files[0])
height, width, channels = frame.shape

# create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # specify the codec
video_writer = cv2.VideoWriter(output_path, fourcc, 25, (width, height))

# loop through all the images and add them to the video writer
for image_file in image_files:
    # read the image and write it to the video writer
    frame = cv2.imread(image_file)
    video_writer.write(frame)

# release the video writer and close the window
video_writer.release()
cv2.destroyAllWindows()
