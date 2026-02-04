from PIL import Image
import requests
import time
import cv2

url1 = "https://ristmikud.tallinn.ee/last/cam112.jpg"
url2 = "https://ristmikud.tallinn.ee/last/cam103.jpg"
while True:
    try:
        response1 = requests.get(url1, timeout=5) 
        response2 = requests.get(url2, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        time.sleep(0.1)
        continue

    response1.raise_for_status()
    response2.raise_for_status()
    with open(f"/Users/hugo/desktop/traffic_images_112/cam112_{time.strftime('%Y-%m-%d %H:%M:%S')}.jpg", "wb") as f:
        f.write(response1.content)
    with open(f"/Users/hugo/desktop/traffic_images_103/cam103_{time.strftime('%Y-%m-%d %H:%M:%S')}.jpg", "wb") as f:
        f.write(response2.content)
    print(f"saved image at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if cv2.waitKey(1) == ord("q"):
        break
    time.sleep(60)
    