from re import I
import cv2
import requests
import csv
from PIL import Image
from io import BytesIO
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

MAX_DISTANCE = 50  # pixels - max distance to match same object
MAX_FRAMES_UNSEEN = 5 # delete object after not seen for that many frames
CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
DURATION_SECONDS = 30  # 2 minutes

filename = 'vehicle_counts_on_Liivalaia_street.csv'
# with open(filename, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['timestamp', 'car', 'motorcycle', 'bus', 'truck', 'total'])

# Class mapping for counting
model = YOLO('yolo11s.pt')
device = 'mps'

url_1 = 'https://ristmikud.tallinn.ee/last/cam112.jpg?'
url_2 = 'https://ristmikud.tallinn.ee/last/cam00.jpg?'
tripwire_1_start = (815,167)
tripwire_1_end = (726,212)
tripwire_2_start = (114,312)
tripwire_2_end = (201,361)

tracked_objects = {} # {object_id: {'center': (x, y), 'class': class_id, 'frames_since_seen': 0, 'side': None, 'counted': False}}
next_id = 0
counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

# 2-minute timer setup
start_time = time.time()


def which_side_of_tripwire(tripwire_start, tripwire_end, point):
    x, y = point
    x1, y1 = tripwire_start
    x2, y2 = tripwire_end

    cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    return 'before' if cross_product > 0 else 'after'


while(True):
    # Check if 2 minutes have passed
    elapsed = time.time() - start_time
    if elapsed >= DURATION_SECONDS:
        print(f"2 minutes elapsed. Total counts: {counts}")
        
        # Write to CSV with proper datetime
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            total_count = counts['car'] + counts['motorcycle'] + counts['bus'] + counts['truck']
            writer.writerow([current_time, counts['car'], counts['motorcycle'], counts['bus'], counts['truck'], total_count])
        break

    timestamp = int(time.time() * 1000)
    url_1 = f'{url_1}?={timestamp}'
    try: 
        response = requests.get(url_1, timeout=5) # get the image from the URL
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        results = model(img, save=True, show=False, classes=[0,1,2,3,5,7])
        annotated_frame = results[0].plot()
        
        # Draw tripwire line on the frame
        cv2.line(annotated_frame, tripwire_1_start, tripwire_1_end, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Street Cam", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            print()
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                total_count = counts['car'] + counts['motorcycle'] + counts['bus'] + counts['truck']
                writer.writerow([current_time, counts['car'], counts['motorcycle'], counts['bus'], counts['truck'], total_count])
                print(f"Wrote to file: /Users/hugo/AI projects/vehicle_counts_on_Liivalaia_street.csv, Total counts: {total_count}")
            break

        time.sleep(0.1)
    except requests.exceptions.RequestException as e:
        print(f'Error fetching image: {e}')
        time.sleep(0.1)

    currently_tracking = []
    for box in results[0].boxes: # get the boxes, center and class id from the results
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center = (center_x, center_y)
        class_id = int(box.cls[0])
        currently_tracking.append({'center': center, 'class': class_id})

    matched_tracks = set() # list of object ids that have been matched from previous frames
    for obj in currently_tracking:
        best_match_id = None
        best_dist = MAX_DISTANCE

        for id, obj_info, in tracked_objects.items(): # find the best match for object in the new frame
            if id in matched_tracks:
                continue
            dist = np.sqrt(
                (obj['center'][0] - obj_info['center'][0])**2 +
                (obj['center'][1] - obj_info['center'][1])**2
            )
            if dist < best_dist:
                best_dist = dist
                best_match_id = id

        if best_match_id is not None: # if a match is found, update the tracked object's center
            tracked_objects[best_match_id]['center'] = obj['center']
            tracked_objects[best_match_id]['frames_since_seen'] = 0
            matched_tracks.add(best_match_id)
            
            current_side = which_side_of_tripwire(
                tripwire_1_start, 
                tripwire_1_end, 
                obj['center']
            )
            
            # Get previous side (if exists)
            previous_side = tracked_objects[best_match_id].get('side')
            
            # Check if object crossed the line
            if previous_side is not None and previous_side != current_side:
                # Object crossed the tripwire!
                if not tracked_objects[best_match_id].get('counted', False):
                    # Map class_id to class name and increment count
                    class_id = tracked_objects[best_match_id]['class']
                    if class_id in CLASS_NAMES:
                        counts[CLASS_NAMES[class_id]] += 1
                        tracked_objects[best_match_id]['counted'] = True
            
            # Update the side for next frame
            tracked_objects[best_match_id]['side'] = current_side

        else:
            # Get initial side for new object
            initial_side = which_side_of_tripwire(
                tripwire_1_start, 
                tripwire_1_end, 
                obj['center']
            )
            
            tracked_objects[next_id] = { # otherwise create a new tracked object
                'center': obj['center'],
                'class': obj['class'],
                'frames_since_seen': 0,
                'side': initial_side,
                'counted': False
            }
            next_id += 1

    # Add missing frame to lost objects (moved outside the loop)
    for id in tracked_objects.keys():
        if id not in matched_tracks:
            tracked_objects[id]['frames_since_seen'] += 1

    # Remove objects that are unseen for too long
    tracked_objects = {
        id: data for id, data in tracked_objects.items()
            if data['frames_since_seen'] < MAX_FRAMES_UNSEEN
    }

cv2.destroyAllWindows()
