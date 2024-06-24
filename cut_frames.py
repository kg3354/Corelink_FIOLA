# # # # import caiman as cm

# # # for i in range(1, 5):
# # #     mov = cm.load('C:/Users/29712/fiola/CaImAn/example_movies/msCam_continuous.tif', subindices=range(i))
        
# # #     mov.save(f'C:/Users/29712/fiola/CaImAn/example_movies/frame_sample/msCam_continuous_frame_{i}.tif')
# # # import caiman as cm

# # # # Load the entire movie once
# # # mov = cm.load('C:/Users/29712/fiola/CaImAn/example_movies/msCam_continuous.tif')

# # # # Loop through each frame and save it separately
# # # for i in range(mov.shape[0]):
# # #     frame = mov[i]
# # #     frame.save(f'C:/Users/29712/fiola/CaImAn/example_movies/frame_sample/msCam_continuous_frame_{i}.tif')
# import os
# import caiman as cm

# # Define the directory for saving frames
# output_dir = 'C:/Users/29712/fiola/CaImAn/example_movies/frame_sample'

# # Create the directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Load the entire movie once
# mov = cm.load('C:/Users/29712/fiola/CaImAn/example_movies/msCam_continuous.tif')

# # Loop through each frame and save it separately
# # for i in range(mov.shape[0]):
# #     frame = mov[i]
# #     frame.save(f'{output_dir}/msCam_continuous_frame_{i}.tif')

# for i in range(1, 5):
#     mov = cm.load('C:/Users/29712/fiola/CaImAn/example_movies/msCam_continuous.tif', subindices=range(i))
        
#     mov.save(f'C:/Users/29712/fiola/CaImAn/example_movies/frame_sample/msCam_continuous_frame_{i}.tif')

import os
import cv2
import tifffile

# # Define the input and output paths
input_path = 'C:/Users/Research/desktop/fiola/CaImAn/example_movies/msCam_continuous.tif'
output_dir = 'C:/Users/Research/desktop/fiola/CaImAn/example_movies/frame_sample'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the entire movie using tifffile
with tifffile.TiffFile(input_path) as tif:
    frames = tif.asarray()

# Loop through each frame and save it as a separate TIFF file
for i, frame in enumerate(frames):
    
    # Check the number of channels in the frame
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Convert to grayscale if the frame has 3 channels (BGR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # If the frame is already grayscale or has an unexpected number of channels, use it as is
        gray_frame = frame

    output_path = f'{output_dir}/msCam_continuous_tf_{i}.tif'
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif_writer:
        tif_writer.write(gray_frame, contiguous=True)

print("Frames saved successfully.")
