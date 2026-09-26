"""
metrics.py
Generates accuracy/loss graphs and evaluation metrics
(precision, recall, F1, confusion matrix) after training.
"""

import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for Flask
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    acc_path = os.path.join(out_dir, 'accuracy.png')
    plt.savefig(acc_path, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    loss_path = os.path.join(out_dir, 'loss.png')
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()

    return acc_path, loss_path


def evaluate_model(model, val_gen, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    val_gen.reset()

    y_true = val_gen.classes
    preds = model.predict(val_gen)
    y_pred = (preds > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    classes = list(val_gen.class_indices.keys())

    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                      ha='center', va='center',
                      color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_path': cm_path
    }
