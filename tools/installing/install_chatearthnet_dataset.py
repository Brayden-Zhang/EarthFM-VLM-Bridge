import os
import time
import requests
from tqdm import tqdm

def download_zenodo_record(record_id, download_dir="./chatearthnet_data"):
    # Zenodo API endpoint for the record
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")
    print(f"Fetching metadata for record {record_id}...")
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()

    files = data.get('files', [])
    # if not files:
    #     files = data.get('entries', {}).values()

    print(f"Found {len(files)} files. Starting download...")

    for file_info in files:
        # Newer Zenodo API uses 'links', older uses 'links' or 'download'
        file_url = file_info['links']['self']
        file_name = file_info['key'] if 'key' in file_info else file_info['filename']
        file_path = os.path.join(download_dir, file_name)
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"Skipping {file_name} (already exists).")
            continue

        print(f"Downloading {file_name}...")
        
        # Download with progress bar
        max_retries = 10
        base_delay = 60

        for attempt in range(max_retries):
            try:
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    
                    with open(file_path, 'wb') as f, tqdm(
                        desc=file_name,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            bar.update(size)
                break # Success
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Use Retry-After header if available, otherwise exponential backoff
                        wait_time = int(e.response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                        print(f"\nRate limited (429). Waiting {wait_time} seconds before retrying (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                raise e

    print("\nDownload complete!")

if __name__ == "__main__":
    RECORD_ID = "11003436"
    download_zenodo_record(RECORD_ID)