
import sys
import imageio
import numpy as np
from io import BytesIO
import tifftools
import json
#import fiola_pipeline

def convert_avi_to_tiff(input_buffer, output_path):
    try:
        # Create a reader for the AVI buffer
        reader = imageio.get_reader(input_buffer, 'avi')
        
        frames = []
        for i, frame in enumerate(reader):
            try:
                # Convert each frame to a numpy array and append to the frames list
                frames.append(frame)
            except Exception as frame_error:
                print(f"Error processing frame {i}: {frame_error}", file=sys.stderr)
        
        # Write frames to a TIFF file
        if frames:
            print(f"Writing {len(frames)} frames to {output_path}")
            imageio.mimwrite(output_path, frames, format='TIFF') #try tiffile imwrite ... 
            print(f"TIFF image saved at {output_path}")
        else:
            print("No frames were successfully processed.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error processing AVI buffer: {e}", file=sys.stderr)
        sys.exit(1)

def combine_tiffs(input_paths, output_path):
    try:
        print(f"Received input paths for combining: {input_paths}")
        
        # Read the list of TIFF files from the input
        tiff_paths = json.loads(input_paths)
        print(f"Parsed TIFF paths: {tiff_paths}")
        
        # Combine TIFF files using tifftools
        tiff = tifftools.read_tiff(tiff_paths[0])
        print(f"Loaded first TIFF: {tiff_paths[0]}")
        
        for other in tiff_paths[1:]:
            print(f"Adding TIFF: {other}")
            othertiff = tifftools.read_tiff(other)
            tiff['ifds'].extend(othertiff['ifds'])
            print(f"Added TIFF: {other}")
        
        tifftools.write_tiff(tiff, output_path)
        print(f"Combined TIFF image saved at {output_path}")
        
        # fiola_pipeline.run_pipeline(output_path)
    except Exception as e:
        print(f"Error combining TIFF files: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_avi_to_tiff.py <output_path> <mode>", file=sys.stderr)
        sys.exit(1)
    
    output_path = sys.argv[1]
    mode = sys.argv[2]

    if mode == 'generate':
        buffer = BytesIO(sys.stdin.buffer.read())
        print(f"Generating TIFF from buffer, saving to {output_path}")
        convert_avi_to_tiff(buffer, output_path)
    elif mode == 'combine':
        input_paths = sys.stdin.read()
        print(f"Combining TIFFs from paths, saving to {output_path}")
        combine_tiffs(input_paths, output_path)
    else:
        print("Invalid mode. Use 'generate' or 'combine'.", file=sys.stderr)
        sys.exit(1)
