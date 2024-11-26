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
from model import Frame_model, StudentFrameModel  # Ensure these models are correctly defined
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from datetime import datetime

# ----------------------------
# Utility Functions
# ----------------------------

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
    """
    Create a long-tailed CIFAR-10 dataset.
    
    Args:
        root (str): Root directory for dataset.
        imb_factor (float): Imbalance factor.
        transform (callable, optional): Transform to apply to the data.
    
    Returns:
        Tuple[Dataset, List[int]]: Imbalanced dataset and number of images per class.
    """
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
    
    # Convert targets to tensor
    dataset.targets = torch.tensor(new_targets)
    return dataset, img_num_per_cls

def make_imbalance_data(num_classes, imb_factor):
    """
    Generate the number of images per class for imbalance.
    
    Args:
        num_classes (int): Number of classes.
        imb_factor (float): Imbalance factor.
    
    Returns:
        List[int]: Number of images per class.
    """
    max_img = 500
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = max_img * (imb_factor ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls

def compute_class_weights(subset, num_classes=10):
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        subset (Subset): Subset of the dataset.
        num_classes (int, optional): Number of classes.
    
    Returns:
        Tensor: Class weights.
    """
    # Access the targets of the underlying dataset
    targets = np.array([subset.dataset.targets[i] for i in subset.indices])
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize
    return weights

def random_minute_mask(img, mask_ratio=0.1):
    """
    Randomly mask minute portions of the image.
    
    Args:
        img (Tensor): Image tensor of shape (C, H, W).
        mask_ratio (float, optional): Ratio of pixels to mask.
    
    Returns:
        Tensor: Masked image tensor.
    """
    masked_img = img.clone()
    c, h, w = img.size()
    num_pixels = int(h * w * mask_ratio)
    for _ in range(num_pixels):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        masked_img[:, x, y] = 0  # Set pixel to 0 to mask it
    return masked_img

# ----------------------------
# Loss Functions
# ----------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Args:
        alpha (Tensor, optional): A tensor of shape (num_classes,) assigning weight to each class.
        gamma (float, optional): Focusing parameter to reduce the loss contribution from easy examples.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss between `inputs` and `targets`.
        
        Args:
            inputs (Tensor): Predictions from the model (logits) of shape (batch_size, num_classes).
            targets (Tensor): Ground truth class indices of shape (batch_size).
        
        Returns:
            Tensor: Computed loss.
        """
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)  # Probability of the true class
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            logpt = logpt * at

        loss = -1 * ((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DistillationLoss(nn.Module):
    """
    Combines classification loss with distillation loss.
    
    Args:
        class_weights (Tensor): Weights for each class to handle class imbalance.
        alpha (float, optional): Weighting factor between classification and distillation loss.
        temperature (float, optional): Temperature scaling factor for distillation.
        gamma (float, optional): Focusing parameter for Focal Loss.
    """
    def __init__(self, class_weights, alpha=0.5, temperature=4.0, gamma=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.gamma = gamma
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=self.gamma, reduction='mean')
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Compute the combined loss.
        
        Args:
            student_outputs (Tensor): Student model predictions (logits) of shape (batch_size, num_classes).
            teacher_outputs (Tensor): Teacher model predictions (logits) of shape (batch_size, num_classes).
            targets (Tensor): Ground truth class indices of shape (batch_size).
        
        Returns:
            Tensor: Combined loss.
        """
        # Classification loss using Focal Loss
        loss_ce = self.focal_loss(student_outputs, targets)

        # Distillation loss using KL Divergence
        # Soft targets
        teacher_probs = F.log_softmax(teacher_outputs / self.temperature, dim=1)
        student_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        loss_kd = self.kd_loss(student_probs, teacher_probs) * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha * loss_ce + (1. - self.alpha) * loss_kd
        return loss

# ----------------------------
# Training and Evaluation Functions
# ----------------------------

def train_kd(student_model, teacher_model, device, train_loader, criterion, optimizer, scheduler, epoch, mask_ratio=0.1):
    """
    Train the student model with knowledge distillation.
    
    Args:
        student_model (nn.Module): The student model to train.
        teacher_model (nn.Module): The pre-trained teacher model.
        device (torch.device): Device to perform computations on.
        train_loader (DataLoader): DataLoader for training data.
        criterion (DistillationLoss): Combined loss function.
        optimizer (Optimizer): Optimizer for training.
        scheduler (LR_Scheduler): Learning rate scheduler.
        epoch (int): Current epoch number.
        mask_ratio (float, optional): Ratio for random masking.
    
    Returns:
        Tuple[float, float, float]: Average loss, accuracy, and F1 score.
    """
    student_model.train()
    teacher_model.eval()
    losses = []
    all_preds = []
    all_labels = []

    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1} [Train]", leave=False)

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply random masking to inputs for teacher model
        masked_inputs = torch.stack([random_minute_mask(img, mask_ratio) for img in inputs])

        optimizer.zero_grad()

        with torch.no_grad():
            _, teacher_outputs = teacher_model(masked_inputs)
            teacher_logits = teacher_outputs  # Assuming teacher_model returns logits directly

        student_outputs = student_model(inputs)

        loss = criterion(student_outputs, teacher_logits, targets)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())
        all_preds.append(torch.argmax(F.softmax(student_outputs, dim=1), dim=1))
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

    scheduler.step()

    return avg_loss, accuracy, f1

def evaluate_kd(results_dir, student_model, teacher_model, device, val_loader, criterion, num_samples_per_class=10, num_classes=10):
    """
    Evaluate the student model with knowledge distillation.
    
    Args:
        results_dir (str): Directory to save evaluation plots.
        student_model (nn.Module): The student model to evaluate.
        teacher_model (nn.Module): The pre-trained teacher model.
        device (torch.device): Device to perform computations on.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (DistillationLoss): Combined loss function.
        num_samples_per_class (int, optional): Number of samples per class for visualization.
        num_classes (int, optional): Number of classes.
    
    Returns:
        Tuple[float, float, float]: Average loss, accuracy, and F1 score.
    """
    student_model.eval()
    teacher_model.eval()
    losses = []
    all_preds = []
    all_labels = []
    class_samples = {i: [] for i in range(num_classes)}  # Dictionary to store samples for each class

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Apply random masking to inputs for teacher model (no masking during evaluation)
            masked_inputs = inputs.clone()  # No masking

            teacher_outputs = teacher_model(masked_inputs)
            _, teacher_logits = teacher_outputs  # Assuming teacher_model returns logits directly
            student_outputs = student_model(inputs)

            loss = criterion(student_outputs, teacher_logits, targets)

            losses.append(loss.item())
            preds = torch.argmax(F.softmax(student_outputs, dim=1), dim=1)
            labels = targets  # Already in class indices

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
    plot_class_predictions(class_samples, num_classes, os.path.join(results_dir, 'student_plots.pdf'))

    return avg_loss, accuracy, f1

def plot_class_predictions(class_samples, num_classes, save_path):
    """
    Plot 1 random sample from each class with predicted and actual labels in a 2x5 subplot grid.
    
    Args:
        class_samples (dict): Dictionary of class samples, where each entry is a list of (image, pred, label).
        num_classes (int): Number of classes.
        save_path (str): Path to save the plot.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Create a 2x5 grid
    fig.suptitle("Student Predictions vs Actual Labels", fontsize=16)

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

# ----------------------------
# Main Function
# ----------------------------

def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    results_dir = create_results_dir('./results_student')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Prepare datasets
    dataset, img_num_per_cls = get_cifar10_lt(root='./data', imb_factor=0.01, transform=transform_train)

    # Stratified split
    indices = list(range(len(dataset)))
    targets = np.array([t.item() for t in dataset.targets])  # Convert tensor targets to numpy

    train_indices, temp_indices, train_targets, temp_targets = train_test_split(
        indices, targets, test_size=0.3, stratify=targets, random_state=42
    )
    val_indices, test_indices, val_targets, test_targets = train_test_split(
        temp_indices, temp_targets, test_size=0.3333, stratify=temp_targets, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Compute class weights for balancing
    class_weights = compute_class_weights(train_dataset, num_classes=10).to(device)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize teacher model
    teacher_model = Frame_model(in_channels=3, out_channels=3, num_classes=10).to(device)
    # Load the pre-trained teacher model
    teacher_checkpoint_path = './results_teacher/20241125_165325/best_teacher_model.pth'  # Update this path
    if not os.path.exists(teacher_checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint_path}")
    teacher_model.load_state_dict(torch.load(teacher_checkpoint_path, map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize student model
    student_model = StudentFrameModel(in_channels=3, out_channels=10, num_classes=10).to(device)

    # Loss and optimizer
    criterion = DistillationLoss(class_weights=class_weights, alpha=0.5, temperature=4.0, gamma=2.0)
    optimizer = optim.Adam(
        student_model.parameters(),
        lr=0.001,
        weight_decay=0,
        betas=(0.9, 0.99)
    )
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    epoch_times = []

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = 0.0
    num_epochs = 100
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_f1 = train_kd(student_model, teacher_model, device, train_loader, criterion, optimizer, scheduler, epoch)
        val_loss, val_acc, val_f1 = evaluate_kd(results_dir, student_model, teacher_model, device, val_loader, criterion)
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")

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
            torch.save(student_model.state_dict(), os.path.join(results_dir, 'best_student_model.pth'))
            print(f"New best F1 score: {best_val_f1:.4f} at epoch {best_epoch}. Model saved.")

    # Plot training curves
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, os.path.join(results_dir, 'student_training_curves.pdf'))
    print(f"Training complete. Results saved in {results_dir}.")

    # Test Evaluation
    test_loss, test_acc, test_f1 = evaluate_kd(results_dir, student_model, teacher_model, device, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")

    # Save the student model checkpoint
    torch.save(student_model.state_dict(), os.path.join(results_dir, 'student_model.pth'))
    print("Student model saved as 'student_model.pth'")

if __name__ == '__main__':
    main()