import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os

# Custom Dataset
class RDDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# Function to calculate class distribution
def calculate_class_distribution(image_sampling_counter, train_dataset):
    """
    Calculate the class-wise count of images sampled during an epoch.
    """
    class_distribution = Counter()
    for image_name, count in image_sampling_counter.items():
        label = next(label for (name, label) in train_dataset.data if name == image_name)
        class_distribution[label] += count
    return class_distribution

# Function to compute class weights
def compute_class_weights(labels):
    total_elements = len(labels)
    class_counts = Counter(labels)
    weights = {cls: total_elements / count for cls, count in class_counts.items()}
    return weights

# Paths
image_folder = "/home/hafsa/Documents/MSc_thesis_paper/facial_image_classification/data"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load data
images = os.listdir(image_folder)
labels = [0 if "Normal" in img or "Neutral" in img else 1 for img in images]
X = np.array(images)
y = np.array(labels)

# Data transformations
transform = transforms.Compose([transforms.ToTensor()])

# Hyperparameters
num_epochs = 5
batch_size = 16
learning_rate = 0.001
num_classes = 2
n_splits = 5

# Stratified K-Fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = []

# Training Loop
if __name__ == "__main__":
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}")

        # Split data
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        # Further split train+val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
        )

        # Compute weights and sampler
        weights = compute_class_weights(y_train.tolist())
        sample_weights = torch.tensor([weights[label] for label in y_train.tolist()])
        sampler = WeightedRandomSampler(sample_weights, int(len(sample_weights)))

        # Prepare datasets and loaders
        train_dataset = RDDataset(list(zip(X_train, y_train)), image_folder, transform)
        val_dataset = RDDataset(list(zip(X_val, y_val)), image_folder, transform)
        test_dataset = RDDataset(list(zip(X_test, y_test)), image_folder, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


        # Training Loop
        best_val_loss = float('inf')
        best_model_path = f"best_model_fold{fold+1}.pkl"

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            image_sampling_counter = Counter()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                batch_image_names = [train_dataset.data[idx][0] for idx in list(train_loader.sampler)[
                    batch_size * batch_idx: batch_size * (batch_idx + 1)]]
                image_sampling_counter.update(batch_image_names)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            train_loss /= len(train_loader.dataset)

            # Calculate class distribution
            class_distribution = calculate_class_distribution(image_sampling_counter, train_dataset)
            print(f"Epoch {epoch + 1}, Fold {fold + 1} Class Distribution: {class_distribution}")

            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    val_loss += loss.item() * data.size(0)

            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch + 1}, Fold {fold + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Save the best model for the current fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model for fold{fold+1} at epoch{epoch+1}")

    fold_results.append(best_val_loss)

print(f"Cross validation results: {fold_results}")