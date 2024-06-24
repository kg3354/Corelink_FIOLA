import pickle
import numpy as np
import os
import time

def convert_pickle_to_npy(pickle_file, output_folder):
    with open(pickle_file, 'rb') as f:
        data = f.read()

    # Debug print to check data content and delimiters
    print(f"Data length: {len(data)}")
    print(f"Start delimiter position: {data.find(b'--PICKLE-START--')}")
    print(f"End delimiter position: {data.find(b'--PICKLE-END--')}")

    # Extract the pickle data between the delimiters
    start_delim = b'--PICKLE-START--'
    end_delim = b'--PICKLE-END--'
    start = data.find(start_delim) + len(start_delim)
    end = data.find(end_delim)
    
    if start == -1 or end == -1:
        raise ValueError("Delimiters not found in the file.")
    
    pickle_data = data[start:end].strip()

    # Debug print to inspect the extracted pickle data
    print("Extracted pickle data length:", len(pickle_data))

    # Deserialize the pickle data
    estimates = pickle.loads(pickle_data)

    # Save the estimates as an npy file
    output_file = os.path.join(output_folder, f"fiola_result_{int(time.time())}.npy")
    np.save(output_file, estimates)
    print(f"Saved estimates to {output_file}")
 
if __name__ == "__main__":
    # Example usage
    pickle_file = './results/fiola_result_1718992270399.pkl'  # Replace with your pickle file path
    output_folder = './results'  # Replace with your desired output folder
    convert_pickle_to_npy(pickle_file, output_folder)
