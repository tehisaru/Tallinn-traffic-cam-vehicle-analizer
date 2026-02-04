import os
import cv2
import csv
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime

# ---- Constants ----

MAX_DISTANCE = 70  # pixels - max distance to match same object
MAX_FRAMES_UNSEEN = 2  # delete object after not seen for that many frames
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DURATION_SECONDS = 120
CSV_FILENAME = "vehicle_counts_on_Liivalaia_street.csv"

# Load YOLO model once and reuse
model = YOLO("yolo11s.pt")
device = "mps"


# ---- Helpers / small utilities ----

def init_csv(filename: str) -> None:
    """Create CSV file with header if it does not exist yet."""
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "car", "motorcycle", "bus", "truck", "total"])


def which_side_of_tripwire(tripwire_start, tripwire_end, point):
    """Return 'before' or 'after' depending on which side of the tripwire the point is."""
    x, y = point
    x1, y1 = tripwire_start
    x2, y2 = tripwire_end

    cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    return "before" if cross_product > 0 else "after"


def get_camera_configs():

    return {
        "cam112": {
            "id": "cam112",
            "name": "Liivalaia tn (suund Pärnu mnt)- Juhkentali tn - Lembitu tn*",
            "base_url": "https://ristmikud.tallinn.ee/last/cam112.jpg",
            # Tripwire tuned for cam112
            "tripwire_start": (815, 167),
            "tripwire_end": (726, 212),
            "expected_movement": (-180, -80),
        },
        "cam103": {
            "id": "cam103",
            "name": "Viru väljak (suund Mere pst ja Narva mnt)",
            "base_url": "https://ristmikud.tallinn.ee/last/cam103.jpg",
            # Tripwire tuned for cam103
            "tripwire_start": (685, 205),
            "tripwire_end": (470, 300),
            "expected_movement": (45, 20),
        },
    }


def prompt_camera_choice(camera_configs):
    """
    Ask the user which camera to use.

    Shows human-readable names (from cam_array) instead of just numbers.
    Returns the chosen camera configuration dict.
    """
    # We only care about cam112 and cam103, keep deterministic order
    order = ["cam112", "cam103"]
    options = [camera_configs[key] for key in order]

    print("Choose camera:")
    for idx, cfg in enumerate(options, start=1):
        print(f"{idx}: {cfg['name']} ({cfg['id']})")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in {"1", "2"}:
            return options[int(choice) - 1]
        print("Please enter 1 or 2.")


def write_counts_to_csv(filename, counts):
    """Append current counts to CSV with timestamp."""
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_count = (
            counts["car"]
            + counts["motorcycle"]
            + counts["bus"]
            + counts["truck"]
        )
        writer.writerow(
            [
                current_time,
                counts["car"],
                counts["motorcycle"],
                counts["bus"],
                counts["truck"],
                total_count,
            ]
        )
    print(
        f"Wrote to file: /Users/hugo/AI projects/{CSV_FILENAME}, "
        f"Total counts: {total_count}"
    )


def run_counter(camera_cfg):
    """
    Main loop for a single camera.

    Uses:
    - camera_cfg['base_url']
    - camera_cfg['tripwire_start'] / ['tripwire_end']
    - camera_cfg['expected_movement']
    """
    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    tracked_objects = {}  # id -> info
    next_id = 0

    tripwire_start = camera_cfg["tripwire_start"]
    tripwire_end = camera_cfg["tripwire_end"]
    tripwire_expected_movement = camera_cfg["expected_movement"]
    base_url = camera_cfg["base_url"]

    start_time = time.time()

    while True:
        # Check if the desired duration has passed
        elapsed = time.time() - start_time
        if elapsed >= DURATION_SECONDS:
            print(
                f"{round(DURATION_SECONDS / 60)} minutes elapsed. "
                f"Total counts: {counts}"
            )
            write_counts_to_csv(CSV_FILENAME, counts)
            break

        # Fetch image
        timestamp = int(time.time() * 1000)
        url = f"{base_url}?{timestamp}"

        try:
            response = requests.get(url, timeout=5)  # get the image from the URL
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))


        except (requests.exceptions.RequestException, Image.UnidentifiedImageError) as e:
            print(f"Error fetching or processing image: {e}")
            time.sleep(0.1)
            continue

        # Run YOLO
        results = model(img, save=False, show=False, classes=[0, 1, 2, 3, 5, 7])
        annotated_frame = results[0].plot()

        # ---- Tracking logic ----
        currently_tracking = []
        for box in results[0].boxes:  # get the boxes, center and class id from results
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center = (center_x, center_y)
            class_id = int(box.cls[0])
            currently_tracking.append({"center": center, "class": class_id})

        matched_tracks = set()  # object ids that have been matched from previous frames
        for obj in currently_tracking:
            best_match_id = None
            best_dist = MAX_DISTANCE

            # find the best match for object in the new frame
            for prev_obj_id, prev_obj_info in tracked_objects.items():
                if prev_obj_id in matched_tracks:
                    continue

                # Distance: current detection vs where we expect this track to be this frame (prev + movement)
                pred_x = prev_obj_info["center"][0] + tripwire_expected_movement[0]
                pred_y = prev_obj_info["center"][1] + tripwire_expected_movement[1]
                dist = np.sqrt((obj["center"][0] - pred_x) ** 2+ (obj["center"][1] - pred_y) ** 2)

                if dist < best_dist:
                    best_dist = dist
                    best_match_id = prev_obj_id

            if best_match_id is not None:  # match: same car from previous frame
                # Save previous frame position before we overwrite it (for green line)
                prev_center = tracked_objects[best_match_id]["center"]
                tracked_objects[best_match_id]["center"] = obj["center"]
                tracked_objects[best_match_id]["frames_since_seen"] = 0
                matched_tracks.add(best_match_id)

                # Green line: previous frame center -> current frame center (same car)
                pt_prev = (int(prev_center[0]), int(prev_center[1]))
                pt_curr = (int(obj["center"][0]), int(obj["center"][1]))
                cv2.line(annotated_frame, pt_prev, pt_curr, (0, 255, 0), 2) 


                current_side = which_side_of_tripwire(
                    tripwire_start,
                    tripwire_end,
                    obj["center"],
                )

                # Get previous side (if exists)
                previous_side = tracked_objects[best_match_id].get("side")

                # Check if object crossed the line
                if previous_side is not None and previous_side != current_side:
                    # Object crossed the tripwire!
                    if not tracked_objects[best_match_id].get("counted", False):
                        # Map class_id to class name and increment count
                        class_id = tracked_objects[best_match_id]["class"]
                        if class_id in CLASS_NAMES:
                            counts[CLASS_NAMES[class_id]] += 1
                            tracked_objects[best_match_id]["counted"] = True

                # Update the side for next frame
                tracked_objects[best_match_id]["side"] = current_side

            else:
                # New object: get initial side
                initial_side = which_side_of_tripwire(
                    tripwire_start,
                    tripwire_end,
                    obj["center"],
                )

                tracked_objects[next_id] = {
                    "center": obj["center"],
                    "class": obj["class"],
                    "frames_since_seen": 0,
                    "side": initial_side,
                    "counted": False,
                }
                next_id += 1

        # Add "missing frame" to objects not seen in this frame 
        for prev_obj_id in list(tracked_objects.keys()):
            if prev_obj_id not in matched_tracks:
                tracked_objects[prev_obj_id]["frames_since_seen"] += 1

        # Remove objects that are unseen for too long
        tracked_objects = {
            obj_id: data
            for obj_id, data in tracked_objects.items()
            if data["frames_since_seen"] < MAX_FRAMES_UNSEEN
        }

        # Draw tripwire line on the frame
        cv2.line(annotated_frame, tripwire_start, tripwire_end, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Street Cam", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            write_counts_to_csv(CSV_FILENAME, counts)
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()


def main():
    # Entry point: ask user for camera, then start counting. 
    init_csv(CSV_FILENAME)

    camera_configs = get_camera_configs()
    chosen_camera = prompt_camera_choice(camera_configs)

    print(
        f"Using camera: {chosen_camera['name']} "
        f"({chosen_camera['id']})"
    )

    run_counter(chosen_camera)


if __name__ == "__main__":
    main()

