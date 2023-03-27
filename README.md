# Yolov8 Real-time Inference using Streamlit
A web interface for real-time yolo inference using streamlit. It supports CPU and GPU inference, supports both images and videos.


### [Live Demo](https://valengou-yolov8-app-p7kyf7.streamlit.app/)


## Features
- **Caches** the model for faster inference on both CPU and GPU.
- Supports uploading model files (<200MB) and downloading models from URL (any size)
- Supports both images and videos.
- Supports both CPU and GPU inference.
- Supports:
  - Custom Classes
  - Changing Confidence
  - Changing input/frame size for videos


## How to run
After cloning the repo:
1. Install requirements
   - `pip install -r requirements.txt`
2. Add sample images to `data/sample_images`
3. Add sample video to `data/sample_videos` and call it `sample.mp4` or change name in the code.
4. Add the model file to `models/` and change `cfg_model_path` to its path.
```bash
git clone https://github.com/valengou/yolov8
cd yolov8
streamlit run app.py
```

### To-do Next
- [] Allow model upload (file / url).
- [] resizing video frames for faster processing.
- [ ] batch processing, processes the whole video and then show the results.
