import torch
import numpy as np
import itertools
from tqdm import tqdm
import json
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import ImprovedEmotionTransformer, NilsHMeierCNN, VGGStyleCNN
import pytorch_helpers as pth  # Using your existing pytorch_helpers module
from data_preparation import INT_TO_LABEL


def load_models(device):
    """Load all three models with their weights"""
    print("Loading models...")

    # Load CNN model
    cnn = NilsHMeierCNN(feature='melspecs')
    cnn.load_state_dict(torch.load('mood_model.pth', map_location=device))
    cnn.eval().to(device)

    # Load Transformer model
    transformer = ImprovedEmotionTransformer()
    transformer.load_state_dict(torch.load(
        'balanced2.pth', map_location=device))
    transformer.eval().to(device)

    # Load VGG model
    vgg = VGGStyleCNN(feature='melspecs')
    vgg.load_state_dict(torch.load('best_jig.pt', map_location=device))
    vgg.eval().to(device)

    return cnn, transformer, vgg


def load_test_data(dataset_name='test'):
    """Load the test dataset using the correct function from pytorch_helpers"""
    print(f"Loading {dataset_name} dataset...")
    # Use the correct function from your pytorch_helpers module
    features, labels = pth.load_data(dataset_name)
    return features, labels


def get_model_predictions(model, features, device):
    """Get predictions from a single model"""
    model.eval()
    all_preds = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(features['melspecs']), batch_size):
            batch_features = {
                'melspecs': features['melspecs'][i:i+batch_size].to(device)
            }
            outputs = model(batch_features)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu())

    return torch.cat(all_preds, dim=0)


def ensemble_predict(predictions, weights):
    """Combine model predictions using the given weights"""
    cnn_preds, transformer_preds, vgg_preds = predictions
    weighted_sum = (
        weights[0] * cnn_preds +
        weights[1] * transformer_preds +
        weights[2] * vgg_preds
    )
    return weighted_sum


def evaluate_ensemble(predictions, weights, true_labels):
    """Evaluate ensemble performance with given weights"""
    ensemble_preds = ensemble_predict(predictions, weights)
    pred_labels = torch.argmax(ensemble_preds, dim=1).numpy()

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weights': weights,
        'predictions': pred_labels
    }


def grid_search(predictions, true_labels, step=0.1):
    """Perform grid search to find optimal weights"""
    best_result = {
        'accuracy': 0,
        'macro_f1': 0,
        'weights': (0, 0, 0)
    }

    results = []

    # Create weight combinations that sum to 1
    weight_options = np.arange(0, 1 + step, step)
    weight_combinations = []

    for w1, w2 in itertools.product(weight_options, weight_options):
        w3 = 1 - w1 - w2
        if 0 <= w3 <= 1:
            weight_combinations.append((w1, w2, w3))

    print(
        f"Searching through {len(weight_combinations)} weight combinations...")

    for weights in tqdm(weight_combinations):
        result = evaluate_ensemble(predictions, weights, true_labels)
        results.append(result)

        # Update best result based on macro F1 score
        if result['macro_f1'] > best_result['macro_f1']:
            best_result = result

    return best_result, results


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def get_individual_model_performance(predictions, true_labels):
    """Evaluate each model individually"""
    individual_results = {}
    model_names = ['CNN', 'Transformer', 'VGG']

    for i, preds in enumerate(predictions):
        pred_labels = torch.argmax(preds, dim=1).numpy()
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')

        individual_results[model_names[i]] = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1)
        }

    return individual_results


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    cnn, transformer, vgg = load_models(device)

    # Load test data - try loading validation data if available, otherwise try 'test'
    dataset_name = 'test'
    try:
        features, labels = load_test_data('validation')
        dataset_name = 'validation'
    except:
        try:
            features, labels = load_test_data('test')
        except:
            print("Could not find validation or test dataset. Trying to load all data...")
            try:
                features, labels = pth.load_all_data()
                print("Using all available data for evaluation")
            except Exception as e:
                print(f"Error loading data: {e}")
                return

    print(f"Successfully loaded {dataset_name} dataset")

    # Get predictions from each model
    print("Getting predictions from CNN...")
    cnn_preds = get_model_predictions(cnn, features, device)

    print("Getting predictions from Transformer...")
    transformer_preds = get_model_predictions(transformer, features, device)

    print("Getting predictions from VGG...")
    vgg_preds = get_model_predictions(vgg, features, device)

    predictions = (cnn_preds, transformer_preds, vgg_preds)

    # Convert labels tensor to numpy for evaluation
    true_labels = labels.numpy()

    # Get individual model performance
    individual_results = get_individual_model_performance(
        predictions, true_labels)
    print("\nIndividual Model Performance:")
    for model, metrics in individual_results.items():
        print(
            f"{model}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['macro_f1']:.4f}")

    # Find best weights
    print("\nFinding optimal weights...")
    best_result, all_results = grid_search(predictions, true_labels, step=0.1)

    # Fine-tune around best weights with smaller step
    print("\nFine-tuning weights...")
    best_w1, best_w2, best_w3 = best_result['weights']
    fine_tune_range = 0.05
    step = 0.01

    fine_tune_predictions = []
    for w1 in np.arange(max(0, best_w1 - fine_tune_range), min(1, best_w1 + fine_tune_range) + step, step):
        for w2 in np.arange(max(0, best_w2 - fine_tune_range), min(1, best_w2 + fine_tune_range) + step, step):
            w3 = 1 - w1 - w2
            if 0 <= w3 <= 1:
                fine_tune_predictions.append((w1, w2, w3))

    for weights in tqdm(fine_tune_predictions):
        result = evaluate_ensemble(predictions, weights, true_labels)
        all_results.append(result)

        if result['macro_f1'] > best_result['macro_f1']:
            best_result = result

    # Print best results
    print("\nBest Ensemble Weights:")
    print(f"CNN: {best_result['weights'][0]:.3f}")
    print(f"Transformer: {best_result['weights'][1]:.3f}")
    print(f"VGG: {best_result['weights'][2]:.3f}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Macro F1 Score: {best_result['macro_f1']:.4f}")

    # Plot confusion matrix for best ensemble
    plot_confusion_matrix(
        true_labels,
        best_result['predictions'],
        [INT_TO_LABEL[i] for i in range(len(INT_TO_LABEL))]
    )

    # Save results
    output = {
        'best_weights': {
            'cnn': float(best_result['weights'][0]),
            'transformer': float(best_result['weights'][1]),
            'vgg': float(best_result['weights'][2])
        },
        'best_performance': {
            'accuracy': float(best_result['accuracy']),
            'macro_f1': float(best_result['macro_f1'])
        },
        'individual_model_performance': individual_results
    }

    with open('ensemble_weights.json', 'w') as f:
        json.dump(output, f, indent=4)

    print("\nResults saved to ensemble_weights.json")


if __name__ == "__main__":
    main()
