import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
from RDDataset import RDDataset
import os
import numpy as np

from facial_image_classification.src.models.train_model import num_classes

# Paths
image_folder = "/home/hafsa/Documents/MSc_thesis_paper/facial_image_classification/data"
n_splits = 5  # Number of folds
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Transform
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

# Load data
images = os.listdir(image_folder)
labels = []
for file in images:
    if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png"):
        if file.__contains__("Normal") or file.__contains__("Neutral"):
            labels.append(0)
        else:
            labels.append(1)

X = np.array(images)
y = np.array(labels)

# K-Fold Cross-Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

model_dir = "/home/hafsa/Documents/MSc_thesis_paper/facial_image_classification/src/models"

# Evaluate each fold
for fold, (train_val_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"Evaluating Fold {fold + 1}/{n_splits}")

    # Split data
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Test dataset and loader
    test_dataset = RDDataset(list(zip(X_test, y_test)), image_folder, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Load the best model for this fold
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    model_path = os.path.join(model_dir, f"best_model_fold{fold + 1}.pkl")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Test the model
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []  # To store predicted probabilities for AUROC and AUPRC

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for the positive class
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Store predictions, probabilities, and targets for metrics calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calculate metrics
    accuracy = 100 * correct / total
    precision = precision_score(all_targets, all_predictions, average='binary')
    recall = recall_score(all_targets, all_predictions, average='binary')
    f1 = f1_score(all_targets, all_predictions, average='binary')

    # Compute AUROC
    fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
    roc_auc = auc(fpr, tpr)

    # Compute AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(all_targets, all_probabilities)
    pr_auc = auc(recall_vals, precision_vals)

    print(f"Fold {fold + 1} Metrics:")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUROC: {roc_auc:.4f}")
    print(f"AUPRC: {pr_auc:.4f}")

    # Optional: Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))

    # Plot AUROC
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUROC (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Fold {fold + 1})')
    plt.legend(loc="lower right")
    #plt.show()
    roc_curve_path = os.path.join('/home/hafsa/Documents/MSc_thesis_paper/facial_image_classification/src/models', f"roc_curve_fold{fold+1}.png")
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"AUROC plot saved to{roc_curve_path}")

    # Plot AUPRC
    plt.figure()
    plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'AUPRC (area = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Fold {fold + 1})')
    plt.legend(loc="lower left")
    pr_curve_path = os.path.join('/home/hafsa/Documents/MSc_thesis_paper/facial_image_classification/src/models',
                                  f"pr_curve_fold{fold + 1}.png")
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"AUPRC plot saved to{roc_curve_path}")

