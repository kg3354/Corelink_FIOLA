import psutil
 
import time
from datetime import datetime
 

def log_cpu_usage():
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(psutil.cpu_percent(interval=1))
            
 
if __name__ == "__main__":
    from threading import Thread

    cpu_thread = Thread(target=log_cpu_usage)
 

    cpu_thread.start()
 
    cpu_thread.join()
 
