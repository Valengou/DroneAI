import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import ultralytics
import numpy as np

from pytube import YouTube
from ultralytics import YOLO

from IPython.display import display, Image
#import supervision as svp #SEE https://www.youtube.com/watch?v=Mi9iHFd0_Bo to work with supervision package

model = YOLO("models/yolov8concretecrack.pt")

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
  #Define COCO Labels
  if labels == []: # ME FALTA CAMBIAR LOS LABEL NAMES DE LAS DIFERENTES CRACKS
    label_names=model.names
    labels = {0: label_names[0], 1: label_names[1], 2: label_names[2],3: label_names[3]}
  #Define colors
  if colors == []:
    #colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
    colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]

  #plot each boxes
  for box in boxes:
    #add score in label if score=True
    if score :
      label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])]
    #filter every box under conf threshold if conf threshold setted
    if conf :
      if box[-2] > conf:
        color = colors[int(box[-1])]
        box_label(image, box, label, color)
    else:
      color = colors[int(box[-1])]
      box_label(image, box, label, color)

  #show image
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image
  #cv2.imshow('',image) #if used in Python




import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import tempfile


def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        #img_path = glob.glob('data/sample_images/*')
        #img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = Image.open('data/sample_images/1.jpg')

    else:
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file:
            img_file=Image.open(uploaded_file)

    return img_file

def video_input(data_src,confidence):
    f = None
    st.title("Video Frame Extraction")
    desired_fps=st.slider('FPS', min_value=1, max_value=500, value=10)
    if data_src == 'Sample data':
        f = "data/sample_videos/sample_video.mp4"
    elif data_src == 'Upload your own data':
        f = st.file_uploader("Upload a video")
    else:
        url = st.text_input("Enter the YouTube video URL")
        if url:
            video = YouTube(url)
            stream = video.streams.get_highest_resolution()
            f = stream.download(skip_existing=True)
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        if data_src == 'Sample data' or data_src == 'URL':
            vf = cv2.VideoCapture(f)
        else:
            tfile.write(f.read())
            vf = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        # Set the desired FPS here


        # Calculate the time to wait between frames
        frame_time = int(1000 / desired_fps)

        # Get the tick frequency
        freq = cv2.getTickFrequency()

        while vf.isOpened():
            # Get the current time in ticks
            start_time = cv2.getTickCount()

            ret, frame = vf.read()
            # Stop the loop if the video ends or there is an error
            if not ret:
                break

            # Pass the frame to your YOLO model here for inference
            results = model.predict(frame)
            #label=labels[int(results[0].boxes.cls)]
            frame =plot_bboxes(frame, results[0].boxes.boxes,  conf=confidence)

            # Display the frame
            stframe.image(frame)

            # Calculate the time it took to process the frame
            elapsed_time = (cv2.getTickCount() - start_time) / freq

            # Calculate the remaining time to wait
            remaining_time = max(0, frame_time - int(elapsed_time * 1000))

            # Wait for the remaining time
            cv2.waitKey(remaining_time)

        vf.release()



def main():
    # global variables
    global model, img, cfg_model_path
    st.title("Concrecte Crack Recognition Model")
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.25)
    data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data','URL'])
    input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

    if input_option== 'image':
        image = np.asarray(image_input(data_src))

        if image.any():
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Selected Image")
            with col2:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Define range of black color in grayscale
                lower_black = np.array([0])
                upper_black = np.array([150])
                mask = cv2.inRange(gray, lower_black, upper_black)
                edges = cv2.Canny(mask, 100, 200)
                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edges=cv2.imwrite("edges.jpg", edges_color)
                edges = Image.open('edges.jpg')

                results1 = model.predict(edges_color)

                results = model.predict(image)
                #st.text(results)
                #st.text(results[0].boxes.cls)
                result_size=results[0].boxes.boxes.size()[0]
                if result_size>0:
                    #label=labels[int(results[0].boxes.cls)]
                    img =plot_bboxes(image, results[0].boxes.boxes,  conf=confidence)
                    img1 =plot_bboxes(edges_color, results1[0].boxes.boxes,  conf=confidence)
                    st.image(img, caption="Model prediction")
                    edge_crack = st.radio("Crack edge: ", ['Hide', 'Show'])
                    if edge_crack== 'Show':
                        st.image(img1)
                    st.text('We have found a crack')
                else:
                    st.image(image, caption='No crack found')
                    #st.image(edges)
    else:
         video_input(data_src,confidence)
### FOOTER
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/valengou/" target="_blank">Valengou</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
