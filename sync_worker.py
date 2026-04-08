import os
import time
import requests
import threading
from data_processor import get_config

def sync_job():
    while True:
        try:
            config = get_config()
            years = config.get("years", {})
            for year, info in years.items():
                link = info.get("link")
                if link:
                    base_dir = os.path.dirname(__file__)
                    data_dir = os.path.join(base_dir, "data_files")
                    os.makedirs(data_dir, exist_ok=True)
                    
                    filepath = os.path.join(data_dir, f"data_{year}.xlsx")
                    
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(link, headers=headers, stream=True, timeout=15)
                    
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    else:
                        print(f"[Sync Worker] Failed to fetch data for {year} from link. HTTP Status: {response.status_code}")
        except Exception as e:
            print(f"[Sync Worker] Exception during background sync: {e}")
            
        # Sleep for 60 seconds
        time.sleep(60)

def start_sync_scheduler():
    thread = threading.Thread(target=sync_job, daemon=True)
    thread.start()
    print("[Sync Worker] Background live-sync scheduler loop initialized.")
