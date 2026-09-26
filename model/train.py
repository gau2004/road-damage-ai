"""
train.py
Standalone command-line training script. Use this once to train
the very first model before starting the Flask app (though you can
also do the first training run from the web UI's Configure & Retrain
page once the app is running).

Example:
    cd model
    python train.py --dataset_dir ../dataset --epochs 15
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_builder import build_model
from utils.preprocess import get_generators
from utils.metrics import plot_history, evaluate_model


def train(args):
    train_gen, val_gen = get_generators(
        args.dataset_dir, args.image_size, args.batch_size, args.augmentation
    )

    model = build_model(
        args.image_size, args.activation, args.dropout,
        args.learning_rate, args.optimizer
    )

    print(f"Class indices: {train_gen.class_indices}")

    start = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)
    training_time = time.time() - start

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'road_model.keras')
    model.save(model_path)

    with open(os.path.join(args.output_dir, 'class_indices.json'), 'w') as f:
        json.dump(train_gen.class_indices, f)

    acc_path, loss_path = plot_history(history, args.output_dir)
    eval_metrics = evaluate_model(model, val_gen, args.output_dir)

    results = {
        'training_accuracy': float(history.history['accuracy'][-1]),
        'validation_accuracy': float(history.history['val_accuracy'][-1]),
        'training_loss': float(history.history['loss'][-1]),
        'validation_loss': float(history.history['val_loss'][-1]),
        'precision': eval_metrics['precision'],
        'recall': eval_metrics['recall'],
        'f1_score': eval_metrics['f1_score'],
        'training_time_sec': training_time,
        'model_path': model_path,
        'accuracy_graph': acc_path,
        'loss_graph': loss_path,
        'confusion_matrix': eval_metrics['confusion_matrix'],
        'class_indices': train_gen.class_indices
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nModel saved to: {model_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the road damage classifier')
    parser.add_argument('--dataset_dir', default='../dataset',
                         help='Path to dataset folder with "Broken Road" and "Not Broken Road" subfolders')
    parser.add_argument('--output_dir', default='saved_model')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--activation', default='relu', choices=['relu', 'tanh', 'sigmoid'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--no-augmentation', dest='augmentation', action='store_false')
    args = parser.parse_args()
    train(args)
