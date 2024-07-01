import os
import shutil
import time

def write_input_to_output(input_path, output_dir):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_filename = os.path.basename(input_path)
    interval = 1 / 30  # interval in seconds (approximately 30 times per second)

    try:
        count = 0
        start_time = time.time()
        
        while True:
            # Calculate elapsed time to adjust sleeping time to maintain ~30 writes per second
            elapsed = time.time() - start_time
            target_count = int(elapsed * 30)
            while count <= target_count:
                output_path = os.path.join(output_dir, f"{count}_{input_filename}")
                shutil.copy(input_path, output_path)
                count += 1
                print(f"Written {output_path}")
            
            # Sleep some time to avoid excessive CPU usage
            time_to_sleep = (count / 30) - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    except KeyboardInterrupt:
        # Stop the loop if Ctrl+C is pressed
        print("Stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_dir = "../sample/msCam_continuous_tf_2.tif"
    output_dir = "../curr"
    write_input_to_output(input_dir, output_dir)
