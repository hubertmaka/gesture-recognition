import cv2
import os
import logging
import pandas as pd

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
)

classes = {
    'Swiping Left': 0,
    'Swiping Right': 1,
    'Swiping Down': 2,
    'Swiping Up': 3,
    'Pushing Hand Away': 4,
    'Pulling Hand In': 5,
    'Sliding Two Fingers Left': 6,
    'Sliding Two Fingers Right': 7,
    'Sliding Two Fingers Down': 8,
    'Sliding Two Fingers Up': 9,
    'Pushing Two Fingers Away': 10,
    'Pulling Two Fingers In': 11,
    'Rolling Hand Forward': 12,
    'Rolling Hand Backward': 13,
    'Turning Hand Clockwise': 14,
    'Turning Hand Counterclockwise': 15,
    'Zooming In With Full Hand': 16,
    'Zooming Out With Full Hand': 17,
    'Zooming In With Two Fingers': 18,
    'Zooming Out With Two Fingers': 19,
    'Thumb Up': 20,
    'Thumb Down': 21,
    'Shaking Hand': 22,
    'Stop Sign': 23,
    'Drumming Fingers': 24,
    'No gesture': 25,
    'Doing other things': 26
}

num_samples_per_class = 10
frames_per_sample = 38

output_dir = "gesture_data"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "own_dataset.csv")
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    existing_paths = set(existing_df["path"])
    logging.info("EXISTING DATAFRAME LOADED.")
else:
    existing_df = pd.DataFrame(columns=["path", "label_id"])
    existing_paths = set()
    logging.info("NO EXISTING DATAFRAME FOUND. CREATING NEW ONE.")

new_data = []

cap = cv2.VideoCapture(0)

try:
    for class_name, class_label in classes.items():
        logging.info(f"COLLECTING DATA FOR CLASS: {class_name} ({class_label})")

        for sample_num in range(num_samples_per_class):
            sample_dir = os.path.join(output_dir, f"{class_name}_{sample_num}")
            sample_dir = sample_dir.replace(" ", "_")

            if os.path.exists(sample_dir):
                logging.info(f"Directory {sample_dir} already exists. Skipping this sample.")
                continue

            os.makedirs(sample_dir, exist_ok=True)
            logging.info(
                f"COLLECTING PROBES {sample_num + 1} z {num_samples_per_class} FOR CLASS {class_name}. PRESS 's' TO START")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("CANNOT READ FRAME FROM WEBCAM. TRY AGAIN.")
                    break

                cv2.imshow("Collecting frames", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    logging.info("Starting collecting frames...")
                    break
                elif key == ord('q'):
                    logging.info("Stopping collecting frames...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            for frame_num in range(1, frames_per_sample + 1):
                ret, frame = cap.read()
                if not ret:
                    logging.error("CANNOT READ FRAME FROM WEBCAM. TRY AGAIN.")
                    break

                resized_frame = cv2.resize(frame, (224, 224))

                frame_path = os.path.join(sample_dir, f"frame_{frame_num:03d}.jpg")

                cv2.imwrite(frame_path, resized_frame)
                new_data.append({"path": frame_path, "label_id": class_label})

                cv2.imshow("Collecting data", resized_frame)
                cv2.waitKey(50)

            logging.info(f"PROBE {sample_num + 1} FOR CLASS {class_name} SAVED.")

except KeyboardInterrupt:
    logging.info("COLLECTING FRAMES STOPPED")
finally:
    cap.release()
    cv2.destroyAllWindows()
