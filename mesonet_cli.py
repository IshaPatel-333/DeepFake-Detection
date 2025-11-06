import numpy as np
import argparse
from classifiers import Meso4, MesoInception4  # Import specifics
from pipeline import compute_accuracy  # For videos
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import os
import pandas as pd

def predict_on_images(classifier, img_dir, batch_size=32, use_ensemble=False):
    if not os.path.exists(img_dir):
        print(f"Error: Directory '{img_dir}' not found.")
        return None, None, None, None, None, None
    
    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
        img_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Ensures order matches filenames
    )
    
    if generator.samples == 0:
        print("No images found.")
        return None, None, None, None, None, None
    
    all_predictions = []
    all_labels = []
    print(f"Processing {generator.samples} images...")
    
    # Optional ensemble
    if use_ensemble:
        inc_classifier = MesoInception4()
        inc_classifier.load('weights/MesoInception4_DF.h5')  # Assumes weights exist
    
    for batch_X, batch_y in generator:
        if use_ensemble:
            batch_pred1 = classifier.predict(batch_X)
            batch_pred2 = inc_classifier.predict(batch_X)
            batch_pred = 0.5 * batch_pred1 + 0.5 * batch_pred2  # Simple average
        else:
            batch_pred = classifier.predict(batch_X)
        
        all_predictions.extend(batch_pred.flatten())
        all_labels.extend(batch_y.flatten())
        if len(all_predictions) >= generator.samples:
            break
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_filenames = generator.filenames  # Add this: relative paths to images
    
    mean_pred = np.mean(all_predictions)
    pred_classes = (all_predictions > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, pred_classes) * 100
    
    return all_predictions, all_labels, mean_pred, accuracy, generator.class_indices, all_filenames  # Return filenames too


def main():
    parser = argparse.ArgumentParser(description="MesoNet Deepfake Detector CLI")
    parser.add_argument('--img_dir', type=str, default='test_images', help="Path to image directory (default: test_images)")
    parser.add_argument('--video_dir', type=str, default=None, help="Path to video directory (e.g., test_videos)")
    parser.add_argument('--model', type=str, default='Meso4', choices=['Meso4', 'MesoInception4'], help="Model to use")
    parser.add_argument('--weights', type=str, default='weights/Meso4_DF.h5', help="Weights file path")
    parser.add_argument('--ensemble', action='store_true', help="Use ensemble (Meso4 + MesoInception4)")
    parser.add_argument('--subsample', type=int, default=30, help="Frame subsample for videos")
    parser.add_argument('--output', type=str, default='results.csv', help="Output CSV file")
    
    args = parser.parse_args()
    
    # Load model
    if args.model == 'Meso4':
        classifier = Meso4()
    else:
        classifier = MesoInception4()
    classifier.load(args.weights)
    
    results = []
    
    # Images
    if args.img_dir:
        print(f"\n--- Predicting on Images in '{args.img_dir}' ---")
        all_preds, all_labels, mean_pred, acc, class_map, all_filenames = predict_on_images(
            classifier, args.img_dir, use_ensemble=args.ensemble
        )
        if all_preds is not None:
            final_class = 'FAKE' if mean_pred > 0.5 else 'REAL'
            print(f"Mean Score: {mean_pred:.4f} | Class: {final_class} | Accuracy: {acc:.2f}%")
            results.append({
                'Type': 'Images',
                'Mean_Score': mean_pred,
                'Accuracy_%': acc,
                'Class_Map': class_map
            })
            # Per-image details
            df = pd.DataFrame({
                'Filename': all_filenames,
                'Pred_Prob': all_preds,
                'Pred_Class': (all_preds > 0.5).astype(int),
                'True_Class': all_labels
            })
            df.to_csv(args.output, index=False)  # Overwrite for images
            print(f"Detailed results saved to '{args.output}' (with filenames)")
    
    # Videos (if specified)
    if args.video_dir:
        print(f"\n--- Predicting on Videos in '{args.video_dir}' ---")
        classifier.load('weights/Meso4_F2F.h5')  # Switch for video-tuned if needed
        video_preds = compute_accuracy(classifier, args.video_dir, frame_subsample_count=args.subsample)
        vid_df = pd.DataFrame([
            {'Video': name, 'Mean_Score': score, 'Class': 'FAKE' if score > 0.5 else 'REAL'}
            for name, (score, _) in video_preds.items()
        ])
        vid_df.to_csv('video_results.csv', index=False)
        print("Video results saved to 'video_results.csv'")
        print(vid_df)
    
    if not results:
        print("No data processed. Check paths.")

if __name__ == "__main__":
    main()