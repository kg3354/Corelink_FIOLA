
import os
import cv2
import tifffile

def get_avi_files(directory):
    """Get a list of AVI files in the specified directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.avi')]

def extract_frames_from_avi(avi_file):
    """Extract frames from an AVI file."""
    cap = cv2.VideoCapture(avi_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale (if needed)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    return frames

def main(input_directory, output_tif_file):
    avi_files = get_avi_files(input_directory)
    
    with tifffile.TiffWriter(output_tif_file, bigtiff=True) as tif_writer:
        for avi_file in avi_files:
            frames = extract_frames_from_avi(avi_file)
            for frame in frames:
                tif_writer.write(frame, contiguous=True)

if __name__ == "__main__":
    input_directory = 'C:/Users/Research/Desktop/temp/GarrettBlair/PKCZ_imaging/test/2024_06_24/09_54_00/miniscope1'
    output_tif_file = "darkminiscope.tif"
    main(input_directory, output_tif_file)
