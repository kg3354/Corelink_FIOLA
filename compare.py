import tifffile as tiff 
import numpy as np 
import sys 
import os 

def compare_tiff_files(file1, file2, num_frames=20): 
    print('comparing')
    try: 
        tiff1 = tiff.TiffFile(file1) 
    except Exception as e: 
        print(f"Error opening file1 ({file1}): {e}") 
        return 
    try: 
        tiff2 = tiff.TiffFile(file2) 
    except Exception as e: 
        print(f"Error opening file2 ({file2}): {e}") 
        tiff1.close() 
        return 
    
    # Extract image data for the first num_frames
    image1 = [tiff1.pages[i].asarray() for i in range(min(num_frames, len(tiff1.pages)))]
    image2 = [tiff2.pages[i].asarray() for i in range(min(num_frames, len(tiff2.pages)))]

    # Compare metadata for the first page
    metadata1 = tiff1.pages[0].tags 
    metadata2 = tiff2.pages[0].tags 
    print("Metadata comparison:") 
    keys1 = set(metadata1.keys()) 
    keys2 = set(metadata2.keys()) 
    common_keys = keys1.intersection(keys2) 
    for key in common_keys: 
        value1 = metadata1[key].value 
        value2 = metadata2[key].value 
        if value1 != value2: 
            print(f"Difference in tag {key}:") 
            print(f"File 1: {value1}") 
            print(f"File 2: {value2}") 
    unique_keys1 = keys1 - keys2 
    unique_keys2 = keys2 - keys1 
    if unique_keys1: 
        print("Unique tags in file 1:") 
        for key in unique_keys1: 
            print(f"Tag {key}: {metadata1[key].value}") 
    if unique_keys2: 
        print("Unique tags in file 2:") 
        for key in unique_keys2: 
            print(f"Tag {key}: {metadata2[key].value}") 

    # Compare image data
    min_frames = min(len(image1), len(image2))
    for i in range(min_frames):
        if image1[i].shape != image2[i].shape: 
            print(f"Different image shapes in frame {i}:") 
            print(f"File 1: {image1[i].shape}") 
            print(f"File 2: {image2[i].shape}") 
        else: 
            difference = np.sum(np.abs(image1[i] - image2[i])) 
            if difference == 0: 
                print(f"The images in frame {i} are identical.") 
            else: 
                print(f"Total difference in pixel values for frame {i}: {difference}") 

    # Detailed page-by-page comparison for the first num_frames
    print("\nPage-by-page comparison for the first 20 frames:") 
    min_pages = min(len(tiff1.pages), len(tiff2.pages), num_frames) 
    for i in range(min_pages): 
        page1 = tiff1.pages[i] 
        page2 = tiff2.pages[i] 
        print(f"\nComparing page {i}:") 
        compare_page_metadata(page1, page2) 

    # Close the files 
    tiff1.close() 
    tiff2.close() 

def compare_page_metadata(page1, page2): 
    metadata1 = page1.tags 
    metadata2 = page2.tags 
    keys1 = set(metadata1.keys()) 
    keys2 = set(metadata2.keys()) 
    common_keys = keys1.intersection(keys2) 
    for key in common_keys: 
        value1 = metadata1[key].value 
        value2 = metadata2[key].value 
        if value1 != value2: 
            print(f"Difference in tag {key}:") 
            print(f"Page 1: {value1}") 
            print(f"Page 2: {value2}") 
    unique_keys1 = keys1 - keys2 
    unique_keys2 = keys2 - keys1 
    if unique_keys1: 
        print("Unique tags in page 1:") 
        for key in unique_keys1: 
            print(f"Tag {key}: {metadata1[key].value}") 
    if unique_keys2: 
        print("Unique tags in page 2:") 
        for key in unique_keys2: 
            print(f"Tag {key}: {metadata2[key].value}") 

if __name__ == "__main__": 
    file1 = 'C:/Users/Research/Desktop/fiola/1000miniscope.tif'
    #file2 = 'C:/Users/Research/Desktop/fiola/trash/combined_output_20.tif' 
    file2 = 'C:/Users/Research/Desktop/fiola/darkminiscope.tif'
    compare_tiff_files(file1, file2)
