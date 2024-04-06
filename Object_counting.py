import streamlit as st
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import tempfile
import os
import base64

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Function to process video
def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define region points
    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

    # Init Object Counter
    model = YOLO("yolov8n.pt")
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False,
                      reg_pts=region_points,
                      classes_names=model.names,
                      draw_tracks=True)

    # Read and process video frames
    st.header("Processed Video")
    progress_bar = st.progress(0)
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        tracks = model.track(frame, persist=True, show=False)
        processed_frame = counter.start_counting(frame, tracks)

        # Display the processed frame
        video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

        # Update progress bar
        progress_bar.progress(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release video capture
    cap.release()

    # Generate download link for the processed video
    st.markdown(get_video_download_link(video_path), unsafe_allow_html=True)

# Main function
def main():
    st.title('Object Counting App')

    # Upload video file
    uploaded_file = st.file_uploader("Upload Video File (MP4):", type=["mp4"])
    if uploaded_file:
        # Temporarily save the uploaded file
        temp_file_path = save_uploaded_file(uploaded_file)

        # Process and display the video
        process_video(temp_file_path)

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file in the temporary directory
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_file_path

# Function to generate a download link for a file
def get_video_download_link(file_path):
    with open(file_path, "rb") as file:
        video_bytes = file.read()
    encoded_video = base64.b64encode(video_bytes).decode()
    href = f"<a href='data:video/mp4;base64,{encoded_video}' download='processed_video.mp4'>Download Processed Video</a>"
    return href

if __name__ == '__main__':
    main()
