# import os
# import tifffile

# def get_tiff_files(directory, limit=100):
#     """Get a list of the first 'limit' TIFF files in the specified directory."""
#     all_tiff_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
#     return all_tiff_files[:limit]

# def extract_frames_from_tiff(tiff_file):
#     """Extract frames from a TIFF file."""
#     with tifffile.TiffFile(tiff_file) as tif:
#         frames = [page.asarray() for page in tif.pages]
#     return frames

# def main(input_directory, output_tif_file):
#     tiff_files = get_tiff_files(input_directory)
    
#     with tifffile.TiffWriter(output_tif_file, bigtiff=True) as tif_writer:
#         for tiff_file in tiff_files:
#             frames = extract_frames_from_tiff(tiff_file)
#             for frame in frames:
#                 tif_writer.write(frame, contiguous=True)

# if __name__ == "__main__":
#     input_directory = 'C:/Users/29712/fiola/CaImAn/example_movies/frame_sample'
#     output_tif_file = "combined_output_100.tif"
#     main(input_directory, output_tif_file)
import os
import tifffile

def get_tiff_files(directory, limit=1000):
    """Get a list of the first 'limit' TIFF files in the specified directory."""
    all_tiff_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
    return all_tiff_files[:limit]

def extract_frames_from_tiff(tiff_file):
    """Extract frames from a TIFF file."""
    with tifffile.TiffFile(tiff_file) as tif:
        frames = [page.asarray() for page in tif.pages]
    return frames

def main(input_directory, output_tif_file):
    try:
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"The input directory '{input_directory}' does not exist.")
        
        tiff_files = get_tiff_files(input_directory)
        
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in the directory '{input_directory}'.")
        
        with tifffile.TiffWriter(output_tif_file, bigtiff=True) as tif_writer:
            for tiff_file in tiff_files:
                print(f"Processing file: {tiff_file}")
                frames = extract_frames_from_tiff(tiff_file)
                for frame in frames:
                    tif_writer.write(frame, contiguous=True)
        print(f"Output file saved as: {output_tif_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_directory = 'C:/Users/29712/fiola/CaImAn/example_movies/frame_sample'
    output_tif_file = "combined_output_1000.tif"
    main(input_directory, output_tif_file)
