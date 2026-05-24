"""
1_data_preprocessing_and_stratification.py

Grabs the top 1500 highest-confidence images per emotion class from ExpW,
runs MTCNN to detect and crop faces, converts to grayscale, saves as 64x64.
Already ran this and pushed the output to Drive — you probably don't need to re-run.
"""

import os
import cv2
import pandas as pd
from mtcnn import MTCNN
from tqdm import tqdm
import tensorflow as tf
import gc
import glob


# ENVIRONMENT SETUP (For Google Colab Reference Only)
# If recreating this dataset from scratch on Google Colab, the following commands were used to set up the environment and extract the raw archives. Do not uncomment these lines if running locally.
# !pip install mtcnn tqdm
# from google.colab import drive
# drive.mount('/content/drive')
# !7z x "./data/origin.7z.001" -o"./expw_images/"


class ExpWPreprocessor:

    def __init__(self, raw_metadata_path, stratified_csv_path, input_dir_pattern, output_dir):
        self.raw_metadata_path = raw_metadata_path
        self.stratified_csv_path = stratified_csv_path
        self.input_dir_pattern = input_dir_pattern
        self.output_dir = output_dir
        self.detector = None  # lazy-load MTCNN, it's slow to init

    def build_balanced_metadata(self, samples_per_class=1500):
        """
        The raw ExpW labels are heavily imbalanced — some emotions like 'happy'
        have way more samples than others like 'disgust'. To stop the model from
        just learning to predict the majority class, we cap each emotion at 1500
        samples and pick the ones with the highest MTCNN detection confidence,
        since those tend to be cleaner, more frontal faces.
        """
        print("building metadata...")
        cols = ['image_name', 'face_id', 'top', 'left', 'right', 'bottom', 'confidence', 'label']
        df_raw = pd.read_csv(self.raw_metadata_path, sep=r'\s+', names=cols, engine='python')

        # sort descending by confidence before grouping so .head(1500) always
        # picks the best-detected faces, not just the first ones in the file
        df_stratified = (
            df_raw
            .sort_values(by='confidence', ascending=False)
            .groupby('label', group_keys=False)
            .apply(lambda x: x.head(samples_per_class))
        )

        os.makedirs(os.path.dirname(self.stratified_csv_path), exist_ok=True)
        df_stratified.to_csv(self.stratified_csv_path, index=False)

        print(df_stratified['label'].value_counts())
        return df_stratified['image_name'].tolist()

    def find_input_dir(self):
        """
        The 7z archive extracts into a nested folder structure we don't fully
        control, so instead of hardcoding a path we just glob for any .jpg and
        use its parent directory. Assumes all images are in the same folder.
        """
        found = glob.glob(self.input_dir_pattern, recursive=True)
        if not found:
            raise SystemExit("no .jpg files found — did the extraction finish?")
        return os.path.dirname(found[0]) + "/"

    def run(self, images, input_dir):
        """
        Main preprocessing loop. For each image we:
          1. run MTCNN to get all face bounding boxes
          2. keep only the highest-confidence detection (avoids background faces)
          3. crop to that box, convert to grayscale, resize to 64x64
          4. save to output_dir under the original filename

        We greyscale to keep the input space small and removes color
        as a confounding feature.

        The loop is resumable: anything already in output_dir gets skipped,
        so if it crashes halfway through you can just re-run it.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        already_done = set(os.listdir(self.output_dir))
        queue = [f for f in images if f not in already_done]
        print(f"already processed: {len(already_done)}, left: {len(queue)}")

        if not queue:
            return

        self.detector = MTCNN()

        for i, filename in enumerate(tqdm(queue)):
            try:
                img = cv2.imread(os.path.join(input_dir, filename))

                # images smaller than 20px are usually thumbnails or corrupt — skip them
                if img is None or img.shape[0] < 20 or img.shape[1] < 20:
                    continue

                # MTCNN expects RGB, OpenCV loads BGR by default
                results = self.detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if results:
                    x, y, w, h = max(results, key=lambda b: b['confidence'])['box']
                    x, y = max(0, x), max(0, y)  # clamp negatives, MTCNN can return them
                    face_crop = img[y:y+h, x:x+w]

                    if face_crop.size > 0:
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (64, 64))
                        cv2.imwrite(os.path.join(self.output_dir, filename), resized)

                # TF holds onto GPU/CPU memory between detections — clearing it
                # every 10 steps keeps things stable on long runs
                if i % 10 == 0:
                    tf.keras.backend.clear_session()
                    gc.collect()

            except Exception:
                continue  # silently skip corrupt or unreadable images

        print(f"done — output saved to: {self.output_dir}")


if __name__ == "__main__":
    preprocessor = ExpWPreprocessor(
        raw_metadata_path="./data/label/label.lst",
        stratified_csv_path="./data/Stratified_10k_Metadata.csv",
        input_dir_pattern="./expw_images/**/*.jpg",
        output_dir="./Stratified_10k_Cleaned_64x64/"
    )

    images = preprocessor.build_balanced_metadata()
    input_dir = preprocessor.find_input_dir()
    preprocessor.run(images, input_dir)