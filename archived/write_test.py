import os
import shutil
import time

def write_input_to_output(input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input directory {input_dir} does not exist.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    avi_files = [f for f in os.listdir(input_dir) if f.endswith('.avi')]
    if not avi_files:
        raise FileNotFoundError(f"No .avi files found in the directory {input_dir}.")

    interval = 1 / 30  # interval in seconds (approximately 30 times per second)

    try:
        count = 0
        start_time = time.time()
        
        for avi_file in avi_files:
            elapsed = time.time() - start_time
            target_count = int(elapsed * 30)
            while count <= target_count:
                output_path = os.path.join(output_dir, f"{count}_{avi_file}")
                shutil.copy(os.path.join(input_dir, avi_file), output_path)
                count += 1
                print(f"Written {output_path}")
                if count > target_count:
                    break
                
            # Sleep some time to avoid excessive CPU usage
            time_to_sleep = (count / 30) - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
                
            if count >= len(avi_files) * 30:
                print("All files have been copied.")
                return

    except KeyboardInterrupt:
        # Stop the loop if Ctrl+C is pressed
        print("Stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_dir = "/Users/guobuzai/Downloads/nyu corelink 3/mHPC24457/2024_07_02/17_16_48/HPC_miniscope1/"  # Replace with the actual directory containing .avi files
    output_dir = "../curr"
    write_input_to_output(input_dir, output_dir)
