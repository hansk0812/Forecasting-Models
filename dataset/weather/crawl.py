import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import os

import shutil

def download_files(url_list):
    # Configure retry logic
    retry_strategy = Retry(
        total=5, # Total retries
        backoff_factor=1, # Exponential backoff: 1, 2, 4, 8, 16 seconds
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    with requests.Session() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        for url in url_list:
            try:
                # Stream the download
                response = session.get(url, stream=True, timeout=10)
                response.raise_for_status()

                filename = url.split('/')[-1]
                with open(f"{filename}", 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {filename}")
                time.sleep(0.5) # Gentle pause
                
                shutil.unpack_archive(filename, "./")
                os.remove(filename)
            
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url}: {e}")

if __name__ == "__main__":

    URL = "https://www.bgc-jena.mpg.de/wetter/weather_data.html"
    r = requests.get(URL)
    urls = [x.split("<a href=\"")[-1].split("\">")[0] \
                for x in r.text.split('\n') \
                    if "<a href=" in x and ".zip" in x]

    urls = ['/'.join(URL.split('/')[:-1]) + '/' + fname for fname in urls]
    
    download_files(urls)
