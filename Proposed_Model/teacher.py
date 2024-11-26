import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import Frame_model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from datetime import datetime

def create_results_dir(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, save_path):
    # Update rcParams for aesthetic settings
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.figsize': (14, 8),
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.5,
        'grid.linestyle': '--'
    })
    
    epochs = range(1, len(train_losses) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors and markers
    colors = ['navy', 'darkgreen', 'maroon']
    markers = ['o', 's', '^']
    
    # Plot Losses
    axs[0].plot(epochs, train_losses, color=colors[0], marker=markers[0], label='Train Loss', markevery=1)
    axs[0].plot(epochs, val_losses, color=colors[1], marker=markers[1], label='Val Loss', markevery=1)
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # Plot Accuracies
    axs[1].plot(epochs, train_accuracies, color=colors[0], marker=markers[0], label='Train Accuracy', markevery=1)
    axs[1].plot(epochs, val_accuracies, color=colors[1], marker=markers[1], label='Val Accuracy', markevery=1)
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    
    # Plot F1 Scores
    axs[2].plot(epochs, train_f1s, color=colors[0], marker=markers[0], label='Train F1 Score', markevery=1)
    axs[2].plot(epochs, val_f1s, color=colors[1], marker=markers[1], label='Val F1 Score', markevery=1)
    axs[2].set_title('F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('F1 Score')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_cifar10_lt(root, imb_factor=0.01, transform=None):
    dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    num_classes = 10
    img_num_per_cls = make_imbalance_data(num_classes, imb_factor)
    new_data = []
    new_targets = []
    targets_np = np.array(dataset.targets)
    classes = np.arange(num_classes)
    for the_class, img_num in zip(classes, img_num_per_cls):
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selected_idx = idx[:img_num]
        new_data.append(dataset.data[selected_idx])
        new_targets.extend([the_class] * img_num)
    dataset.data = np.vstack(new_data)
    
    # Change targets to class indices instead of one-hot encoding
    dataset.targets = new_targets
    return dataset, img_num_per_cls


def make_imbalance_data(num_classes, imb_factor):
    max_img = 500
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = max_img * (imb_factor ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def random_minute_mask(img, mask_ratio=0.1):
    """Randomly mask minute portions of the image."""
    masked_img = img.clone()
    c, h, w = img.size()
    num_pixels = int(h * w * mask_ratio)
    for _ in range(num_pixels):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        masked_img[:, x, y] = 0  # Set pixel to 0 to mask it
    return masked_img


class Noise2VoidLoss(nn.Module):
    """Noise2Void loss for reconstructing masked regions."""
    def __init__(self):
        super(Noise2VoidLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """Calculate MSE loss only on masked regions."""
        return self.mse_loss(predictions, targets)


class DecoupledLossWithMasking(nn.Module):
    def __init__(self, head_classes, tail_classes):
        super(DecoupledLossWithMasking, self).__init__()
        self.head_classes = head_classes
        self.tail_classes = tail_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Calculate decoupled loss for head and tail classes.

        Parameters:
            outputs (Tensor): Model predictions (logits) with shape (batch_size, num_classes).
            targets (Tensor): Ground truth class indices with shape (batch_size).

        Returns:
            Tensor: Combined loss.
        """
        head_mask = torch.zeros_like(targets, dtype=torch.bool)
        tail_mask = torch.zeros_like(targets, dtype=torch.bool)
        for cls in self.head_classes:
            head_mask |= (targets == cls)
        for cls in self.tail_classes:
            tail_mask |= (targets == cls)
        loss = 0.0
        if head_mask.any():
            head_loss = self.ce_loss(outputs[head_mask], targets[head_mask])
            loss += head_loss
        if tail_mask.any():
            tail_loss = self.ce_loss(outputs[tail_mask], targets[tail_mask])
            loss += tail_loss
        return loss


def train(model, device, train_loader, criterion_cls, criterion_mask, optimizer, scheduler, epoch, mask_ratio=0.1):
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1} [Train]", leave=False)

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Apply random masking
        masked_inputs = torch.stack([random_minute_mask(img, mask_ratio) for img in inputs])
        optimizer.zero_grad()
        
        recon_output, class_output = model(masked_inputs)

        # Calculate combined loss
        loss_cls = criterion_cls(class_output, targets)  # targets are class indices
        loss_mask = criterion_mask(recon_output, inputs)
        loss = loss_cls + loss_mask
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())
        all_preds.append(torch.argmax(class_output, dim=1))  # No need to apply softmax before argmax
        all_labels.append(targets)

        pbar.set_postfix(loss=loss.item())
        pbar.update(1)

    pbar.close()

    # Metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = np.mean(losses)
    accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
    f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')

    return avg_loss, accuracy, f1


def evaluate(results_dir, model, device, val_loader, criterion_cls, criterion_mask, num_samples_per_class=10, num_classes=10):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    class_samples = {i: [] for i in range(num_classes)}  # Dictionary to store samples for each class

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            recon_output, class_output = model(inputs)

            # Calculate combined loss
            loss_cls = criterion_cls(class_output, targets)
            loss_mask = criterion_mask(recon_output, inputs)
            loss = loss_cls + loss_mask

            losses.append(loss.item())
            preds = torch.argmax(class_output, dim=1)
            labels = targets  # targets are class indices

            all_preds.append(preds)
            all_labels.append(labels)

            # Store random samples for visualization
            for img, pred, label in zip(inputs.cpu(), preds.cpu(), labels.cpu()):
                if len(class_samples[label.item()]) < num_samples_per_class:
                    class_samples[label.item()].append((img, pred.item(), label.item()))

    # Metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = np.mean(losses)
    accuracy = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
    f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')

    # Plot samples
    plot_class_predictions(class_samples, num_classes, os.path.join(results_dir, 'teacher_plots.pdf'))

    return avg_loss, accuracy, f1


def plot_class_predictions(class_samples, num_classes, save_path):
    """
    Plot 1 random sample from each class with predicted and actual labels in a 2x5 subplot grid.
    
    Parameters:
        class_samples (dict): Dictionary of class samples, where each entry is a list of (image, pred, label).
        num_classes (int): Number of classes.
        save_path (str): Path to save the plot.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Create a 2x5 grid
    fig.suptitle("Predictions vs Actual Labels", fontsize=16)

    for class_idx in range(num_classes):
        row, col = divmod(class_idx, 5)  # Determine subplot position (row, column)
        ax = axes[row, col]

        # Check if there is a sample for this class
        if len(class_samples[class_idx]) > 0:
            img, pred, label = class_samples[class_idx][0]  # Pick the first sample
            img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for plotting
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for visualization

            ax.imshow(img)
            ax.set_title(f"Pred: {pred}\nActual: {label}", fontsize=10)
        else:
            ax.set_title(f"No Sample", fontsize=10)

        ax.axis('off')  # Turn off axis for a cleaner look

    # Remove unused subplots (if any)
    for i in range(num_classes, 10):  # CIFAR-10 has exactly 10 classes
        row, col = divmod(i, 5)
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()



def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    results_dir = create_results_dir('./results_teacher')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Prepare datasets
    dataset, img_num_per_cls = get_cifar10_lt(root='./data', imb_factor=0.01, transform=transform)

    # Define head and tail classes based on img_num_per_cls
    # For example, let's define classes with more samples as head and fewer as tail
    sorted_classes = sorted(range(len(img_num_per_cls)), key=lambda x: img_num_per_cls[x], reverse=True)
    split = len(sorted_classes) // 2  # Split classes into head and tail
    head_classes = sorted_classes[:split]
    tail_classes = sorted_classes[split:]

    print(f"Head classes: {head_classes}")
    print(f"Tail classes: {tail_classes}")

    # Stratified split
    indices = list(range(len(dataset)))
    targets = np.array(dataset.targets)  # Targets are class indices

    train_indices, temp_indices, train_targets, temp_targets = train_test_split(
        indices, targets, test_size=0.3, stratify=targets, random_state=42
    )
    val_indices, test_indices, val_targets, test_targets = train_test_split(
        temp_indices, temp_targets, test_size=0.3333, stratify=temp_targets, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = Frame_model(in_channels=3, out_channels=3, num_classes=10).to(device)

    # Loss and optimizer
    # Replace BCEWithLogitsLoss with DecoupledLossWithMasking
    criterion_cls = DecoupledLossWithMasking(head_classes=head_classes, tail_classes=tail_classes)
    criterion_mask = Noise2VoidLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0,
        betas=(0.9, 0.99)
    )
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = 0.0
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train(model, device, train_loader, criterion_cls, criterion_mask, optimizer, scheduler, epoch)
        val_loss, val_acc, val_f1 = evaluate(results_dir, model, device, val_loader, criterion_cls, criterion_mask)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1  # Epochs are 1-indexed for readability
            # Save the student model's state_dict
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_teacher_model.pth'))
            print(f"New best F1 score: {best_val_f1:.4f} at epoch {best_epoch}. Model saved.")

    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, os.path.join(results_dir, 'teacher_training_curves.pdf'))
    print(f"Training complete. Results saved in {results_dir}.")
    
    # Test Evaluation
    test_loss, test_acc, test_f1 = evaluate(results_dir, model, device, test_loader, criterion_cls, criterion_mask)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")

    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))
    print("Model saved as 'model.pth'")


if __name__ == '__main__':
    main()