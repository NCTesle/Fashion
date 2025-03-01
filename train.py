import torch
import os
from ultralytics import YOLO

# Ensure GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# Define dataset path and YAML file
DATASET_PATH = r"E:\College\miniproject\Fashion\Dataset\fash"
YAML_FILE = os.path.join(DATASET_PATH, "data.yaml")

if __name__ == '__main__':
    # Load Pretrained YOLOv8 Model
    model = YOLO("E:/College/miniproject/Fashion/Training/yolov8m_fashion4/weights/best.pt")

    # Training Parameters
    epochs = 50
    batch_size = 8
    lr = 0.0003
    weight_decay = 0.0005

    # Store Previous Validation Loss & mAP for Overfitting Detection
    best_val_loss = float('inf')
    prev_train_loss = float('inf')
    best_val_mAP = 0
    patience = 5  # Number of epochs to wait before stopping
    patience_counter = 0

    for epoch in range(epochs):
        results = model.train(
            data=YAML_FILE,
            epochs=1,  # Train one epoch at a time
            imgsz=640,
            batch=batch_size,
            device=device,
            project="Training",
            name="yolov8m_fashion_finetuning",
            save=True,
            amp=True,
            exist_ok=True,
            lr0=lr,
            weight_decay=weight_decay
        )

        # Extract Losses and Metrics Correctly
        metrics = results.results_dict  # Get dictionary of results

        train_loss = metrics.get('train/box_loss', 0) + metrics.get('train/cls_loss', 0) + metrics.get('train/dfl_loss', 0)
        val_loss = metrics.get('val/box_loss', 0) + metrics.get('val/cls_loss', 0) + metrics.get('val/dfl_loss', 0)
        val_mAP = metrics.get('metrics/mAP50', 0)

        print(f"📊 Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mAP50: {val_mAP:.4f}")

        # Check for Overfitting
        if val_loss > best_val_loss and train_loss < prev_train_loss:
            patience_counter += 1
            print(f"⚠️ Overfitting detected! Validation loss increased while training loss decreased ({patience_counter}/{patience})")

            # Stop training if overfitting continues for "patience" epochs
            if patience_counter >= patience:
                print("🛑 Stopping training due to overfitting!")
                break
        else:
            patience_counter = 0  # Reset counter if no overfitting

        # Update Best Values
        best_val_loss = min(best_val_loss, val_loss)
        prev_train_loss = train_loss
        best_val_mAP = max(best_val_mAP, val_mAP)

    print("✅ Fine-tuning complete! Model saved.")
