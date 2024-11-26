import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
from sklearn.manifold import TSNE

class CIFAR100Trainer:
    def __init__(self, batch_size=2048, learning_rate=0.001, num_epochs=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Transformations
        self.transform = transforms.Compose([
        ])

        # Load datasets with long-tailed distribution

        self.trainloader, self.testloader, self.long_tailed_label_counts = self.load_long_tailed_data()
        # trainloader = class_distribution 
        self.model = self.build_model()

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # For tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # CIFAR-100 Class Names
        self.cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'y', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

    def load_long_tailed_data(self):
        """Create a long-tailed version of the CIFAR-100 dataset."""
        # Load the full CIFAR-100 dataset
        trainset = CIFAR100(root='./data', train=True, download=True, transform=self.transform)

        # Get the class indices from the dataset
        targets = np.array(trainset.targets)

        # Create a long-tailed distribution: randomly reduce samples in some classes
        long_tailed_label_counts = {}
        max_samples = 500  # Maximum number of samples for the most frequent class
        min_samples = 20    # Minimum number of samples for the least frequent class

        # Use a decay factor for reducing the number of samples per class
        decay_factor = 0.95
        for label in range(100):
            long_tailed_label_counts[label] = max(int(max_samples * (decay_factor ** label)), min_samples)

        # Create the new indices based on the long-tailed distribution
        indices = []
        for label in range(100):
            label_indices = np.where(targets == label)[0]
            np.random.shuffle(label_indices)
            selected_indices = label_indices[:long_tailed_label_counts[label]]  # Select only a portion of data
            indices.extend(selected_indices)

        # Create a subset of the dataset with the selected indices
        long_tailed_trainset = torch.utils.data.Subset(trainset, indices)

        # Create DataLoader with the new long-tailed trainset
        trainloader = torch.utils.data.DataLoader(long_tailed_trainset, batch_size=self.batch_size, shuffle=True)

        # Use the standard test set
        testset = CIFAR100(root='./data', train=False, download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return trainloader, testloader, long_tailed_label_counts

    def plot_top_bottom_kde_gradients(self, label_accuracies, epoch):
        gradients = self.model.fc.weight.grad.cpu().numpy()  # Get gradients of the fc layer
        gradients_per_class = {i: [] for i in range(100)}  # Store gradients for each label

        # Accumulate gradients per class by summing across classes
        for class_idx in range(100):
            gradients_per_class[class_idx].append(gradients[class_idx].flatten())

        # Flatten the gradient arrays for each label class
        for label in gradients_per_class:
            if gradients_per_class[label]:
                gradients_per_class[label] = np.concatenate(gradients_per_class[label])

        # Sort labels by accuracy
        sorted_labels = sorted(label_accuracies.items(), key=lambda x: x[1], reverse=True)
        top_5_labels = [x[0] for x in sorted_labels[:5]]  # Top 5 performing labels
        bottom_5_labels = [x[0] for x in sorted_labels[-5:]]  # Bottom 5 performing labels

        # Gather the gradients for the rest of the labels
        rest_labels_gradients = []
        for label, grads in gradients_per_class.items():
            if label not in top_5_labels and label not in bottom_5_labels and len(grads) > 0:
                rest_labels_gradients.append(grads)
        rest_labels_gradients = np.concatenate(rest_labels_gradients)

        # Create subplots
        plt.figure(figsize=(14, 10))
        
        # Top 5 Subplot
        plt.subplot(2, 1, 1)
        sns.set(style="whitegrid")
        
        for label in top_5_labels:
            if len(gradients_per_class[label]) > 0:
                class_name = self.cifar100_classes[label]
                sns.kdeplot(gradients_per_class[label], label=f'Top 5: {class_name}', fill=True, lw=2)

        if len(rest_labels_gradients) > 0:
            sns.kdeplot(rest_labels_gradients, label='Rest of the Labels', fill=True, lw=2, linestyle="--")

        plt.xlabel('Gradient Values', fontsize=14, fontweight='bold')
        plt.ylabel('Density', fontsize=14, fontweight='bold')
        plt.title(f'KDE Plot for Top 5 Performing Labels - Epoch {epoch}', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, which="both", ls="--", lw=0.5)

        # Bottom 5 Subplot
        plt.subplot(2, 1, 2)
        sns.set(style="whitegrid")
        
        for label in bottom_5_labels:
            if len(gradients_per_class[label]) > 0:
                class_name = self.cifar100_classes[label]
                sns.kdeplot(gradients_per_class[label], label=f'Worst 5: {class_name}', fill=True, lw=2)

        if len(rest_labels_gradients) > 0:
            sns.kdeplot(rest_labels_gradients, label='Rest of the Labels', fill=True, lw=2, linestyle="--")

        plt.xlabel('Gradient Values', fontsize=14, fontweight='bold')
        plt.ylabel('Density', fontsize=14, fontweight='bold')
        plt.title(f'KDE Plot for Worst 5 Performing Labels - Epoch {epoch}', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, which="both", ls="--", lw=0.5)

        plt.tight_layout()

        # Save the KDE plot as PNG with high resolution
        self.save_plot(f"kde_gradients_epoch_{epoch}.png")
        plt.show()

    def plot_confusion_matrix_and_correlation(self, true_labels, predicted_labels):
        """Plot confusion matrix and correlation matrix between true and predicted labels with label names."""
        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(100))

        # Normalize confusion matrix to get a correlation-like matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix as heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(conf_matrix_norm, annot=False, cmap="Blues", fmt='.2f', linewidths=0.5,
                    xticklabels=self.cifar100_classes, yticklabels=self.cifar100_classes)

        plt.title("Correlation Matrix between True Labels and Predicted Labels", fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=14, fontweight='bold')
        plt.ylabel('True Labels', fontsize=14, fontweight='bold')

        # Rotate labels for better visibility
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(False)

        # Save the correlation matrix plot as PNG with high resolution
        self.save_plot("label_label_correlation_matrix_with_names.png")
        plt.show()


    def build_model(self):
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust final layer for CIFAR-100
        return model.to(self.device)

    def extract_features(self, dataloader):
        """Extract features from the dataset using the trained model."""
        self.model.eval()  # Set model to evaluation mode
        features_list = []
        labels_list = []

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in tqdm(dataloader, desc="Extracting Features"):
                inputs = inputs.to(self.device)
                
                # Forward pass through the model up to the last fully connected layer (before softmax/logits)
                features = self.model.avgpool(self.model.layer4(self.model.layer3(self.model.layer2(self.model.layer1(self.model.conv1(inputs))))))
                features = torch.flatten(features, 1)  # Flatten the features
                features_list.append(features.cpu().numpy())  # Convert features to numpy array
                labels_list.append(labels.cpu().numpy())

        return np.concatenate(features_list), np.concatenate(labels_list)

    def validate_model_with_accuracy(self):
        """Validation method to calculate accuracy per label and return predictions/labels."""
        model = self.model.eval()
        running_val_loss = 0.0
        correct_per_class = {i: 0 for i in range(100)}
        total_per_class = {i: 0 for i in range(100)}
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.testloader, desc="Validating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()

                # Get predictions
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Accuracy per class
                for label, pred in zip(labels, preds):
                    label = label.item()
                    total_per_class[label] += 1
                    if label == pred.item():
                        correct_per_class[label] += 1

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate accuracy per label
        label_accuracies = {label: correct_per_class[label] / total_per_class[label] if total_per_class[label] > 0 else 0
                            for label in range(100)}

        val_loss = running_val_loss / len(self.testloader)
        return val_loss, label_accuracies, all_preds, all_labels

    def save_plot(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

    def plot_long_tailed_distribution(self):
        """Plot the long-tailed label distribution highlighting labels with >270 and <25 samples.
        Label names are printed for these thresholds, otherwise, the index is shown."""
        
        label_counts = pd.Series(self.long_tailed_label_counts).sort_values(ascending=True)

        plt.figure(figsize=(20, 10))
        sns.set(style="whitegrid")

        # Plot label distribution
        sns.lineplot(x=label_counts.index, y=label_counts.values, marker="o", color="royalblue", linewidth=3.5)

        # Highlight markers for labels with >270 samples (green) and <25 samples (red)
        high_samples = label_counts[label_counts > 270]
        low_samples = label_counts[label_counts < 25]
        sns.scatterplot(x=high_samples.index, y=high_samples.values, color='green', s=50, label='Labels > 270 samples', zorder=5)
        sns.scatterplot(x=low_samples.index, y=low_samples.values, color='red', s=50, label='Labels < 25 samples', zorder=5)

        # Customize x-axis labels: use class names for labels with samples >270 or <25, otherwise use index
        xtick_labels = []
        for i, count in label_counts.items():
            if count > 270 or count < 25:
                xtick_labels.append(self.cifar100_classes[i])
            else:
                xtick_labels.append(str(i))  # Show the index for others

        # Add class names to the x-axis
        plt.xticks(ticks=label_counts.index, labels=xtick_labels, rotation=90, fontsize=12)

        # Add labels and title
        plt.xlabel("Labels", fontsize=16, fontweight='bold', labelpad=15)
        plt.ylabel("Number of Samples", fontsize=16, fontweight='bold', labelpad=15)  
        plt.title("Long-Tailed Label Distribution with Highlighted Labels CIFAR-100", fontsize=18, fontweight='bold', pad=20)

        # Add a legend
        plt.legend(fontsize=12)

        # Refine grid lines
        plt.grid(False)
        # plt.grid(False, which="both", linestyle='--', linewidth=0.6)

        # Save the plot as a high-resolution image
        self.save_plot("long_tailed_label_distribution_beautified.png")
        plt.tight_layout()
        plt.show()

    def plot_tsne_features(self, train_features, train_labels, test_features, test_labels):
        """Generates and plots t-SNE projections for the train and test set features with legend outside."""
        tsne = TSNE(n_components=2, random_state=42)

        # Fit t-SNE on train and test features
        train_tsne = tsne.fit_transform(train_features)
        test_tsne = tsne.fit_transform(test_features)

        plt.figure(figsize=(14, 10))

        # Subplot for Train Set
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_labels, cmap='tab20', s=10, alpha=0.8)
        plt.title('t-SNE of Train Set Features', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        
        # Legend outside
        plt.legend(handles=scatter.legend_elements()[0], labels=self.cifar100_classes, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

        # Subplot for Test Set
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c=test_labels, cmap='tab20', s=10, alpha=0.8)
        plt.title('t-SNE of Test Set Features', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        
        # Legend outside
        plt.legend(handles=scatter.legend_elements()[0], labels=self.cifar100_classes, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

        plt.tight_layout()
        self.save_plot("tsne_train_test_features.png")
        plt.show()

    def plot_training_validation_curves(self):
        epochs = range(1, self.num_epochs + 1)

        plt.figure(figsize=(14, 10))

        # Subplot 1: Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', marker='o', color='blue', linewidth=2.5, markersize=7)
        plt.plot(epochs, self.val_losses, label='Validation Loss', marker='s', color='orange', linewidth=2.5, markersize=7)
        plt.fill_between(epochs, self.train_losses, self.val_losses, color='gray', alpha=0.2)  # Optional shading between curves
        plt.title('Training and Validation Loss', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Epochs', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, linestyle='--', linewidth=0.6)

        # Subplot 2: Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy', marker='o', color='green', linewidth=2.5, markersize=2)
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', marker='s', color='red', linewidth=2.5, markersize=2)
        plt.fill_between(epochs, self.train_accuracies, self.val_accuracies, color='gray', alpha=0.2)  # Optional shading between curves
        plt.title('Training and Validation Accuracy', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Epochs', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, linestyle='--', linewidth=0.6)

        # Apply tight layout and save the plot
        plt.tight_layout(pad=3)
        self.save_plot("training_validation_curves.png")
        plt.show()

    def train(self):
        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            pbar_train = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Training]", leave=False)

            for inputs, labels in pbar_train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Update optimizer
                self.optimizer.step()

                # Update running training loss
                running_train_loss += loss.item()

                # Calculate training accuracy
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            avg_train_loss = running_train_loss / len(self.trainloader)
            train_accuracy = correct_train / total_train
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            print(f'Epoch {epoch+1}/{self.num_epochs}, Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

            # Validation loop
            val_loss, label_accuracies, all_preds, all_labels = self.validate_model_with_accuracy()
            val_accuracy = np.mean([label_accuracies[label] for label in label_accuracies])
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

            # Plot KDE gradients every 5 epochs
            # if epoch == 0:
            #     self.plot_top_bottom_kde_gradients(label_accuracies, epoch+1)
            # elif (epoch + 1) % 1 == 0:
            #     self.plot_top_bottom_kde_gradients(label_accuracies, epoch + 1)

            if (epoch + 1) == self.num_epochs:
                # Plot confusion matrix for labels and predictions
                # self.plot_confusion_matrix_and_correlation(all_labels, all_preds)


                train_features, train_labels = self.extract_features(self.trainloader)
                test_features, test_labels = self.extract_features(self.testloader)
                self.plot_tsne_features(train_features, train_labels, test_features, test_labels)

        # Plot Training and Validation curves
        # self.plot_long_tailed_distribution()
        self.plot_training_validation_curves()