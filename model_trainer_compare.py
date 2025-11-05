#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  28 19:06:09 2025

@author: otto
"""

# =============================("="*60===============================
# REUSABLE MODEL TRAINER FOR MLP, GCN, AND GAT
# ============================================================

import torch
#import torch.nn.functional as F
#from torch.nn import Linear
#from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
#import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
import pandas as pd


# ============================================================
# ENHANCED VISUALIZATION WITH ERROR ANALYSIS
# ============================================================


def visualize_predictions(model_output, true_labels, title="Model Predictions", 
                          save_path=None, show_legend=True):
    """
    Visualize model predictions with t-SNE, color-coding correct and incorrect predictions.
    
    Args:
        model_output: Model output logits/embeddings (numpy array) [N, num_classes]
        true_labels: Ground truth labels (numpy array) [N]
        title: Plot title
        save_path: Path to save figure (None = don't save)
        show_legend: Whether to show legend
    
    Color scheme:
        - True Negative (Licit correctly predicted): Light Blue (#52D1DC)
        - True Positive (Illicit correctly predicted): Dark Red (#8D0004)
        - False Positive (Licit wrongly predicted as Illicit): Orange (#FF7800)
        - False Negative (Illicit wrongly predicted as Licit): Purple (#845218)
    """
    # Apply t-SNE
    print(f"Applying t-SNE for visualization: {title}...")
    tsne = TSNE(n_components=2, init='pca', random_state=7, perplexity=30)
    tsne_res = tsne.fit_transform(model_output)
    
    # Get predictions
    pred_labels = np.argmax(model_output, axis=1)
    
    # Create classification categories
    # 0 = Licit, 1 = Illicit
    categories = []
    category_names = []
    
    for true, pred in zip(true_labels, pred_labels):
        if true == 0 and pred == 0:
            categories.append(0)  # True Negative (TN)
            category_names.append('TN: Licit → Licit')
        elif true == 1 and pred == 1:
            categories.append(1)  # True Positive (TP)
            category_names.append('TP: Illicit → Illicit')
        elif true == 0 and pred == 1:
            categories.append(2)  # False Positive (FP)
            category_names.append('FP: Licit → Illicit')
        elif true == 1 and pred == 0:
            categories.append(3)  # False Negative (FN)
            category_names.append('FN: Illicit → Licit')
    
    # Create DataFrame
    df = pd.DataFrame({
        'dim1': tsne_res[:, 0],
        'dim2': tsne_res[:, 1],
        'category': categories,
        'category_name': category_names,
        'true_label': true_labels,
        'pred_label': pred_labels
    })
    
    # Count each category
    category_counts = df['category'].value_counts().sort_index()
    tn_count = category_counts.get(0, 0)
    tp_count = category_counts.get(1, 0)
    fp_count = category_counts.get(2, 0)
    fn_count = category_counts.get(3, 0)
    
    # Calculate accuracy
    correct = tn_count + tp_count
    total = len(df)
    accuracy = correct / total if total > 0 else 0
    
    # Define colors
    colors = {
        0: '#52D1DC',  # TN - Light Blue
        1: '#8D0004',  # TP - Dark Red
        2: '#FF7800',  # FP - Orange
        3: '#845218'   # FN - Brown/Purple
    }
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Plot each category
    for cat in sorted(df['category'].unique()):
        subset = df[df['category'] == cat]
        plt.scatter(
            subset['dim1'], 
            subset['dim2'],
            c=colors[cat],
            label=f"{subset['category_name'].iloc[0]} ({len(subset)})",
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.title(f'{title}\nAccuracy: {accuracy:.2%} | Correct: {correct}/{total}',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    if show_legend:
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Prediction Statistics: {title}")
    print(f"{'='*60}")
    print(f"True Negatives (Licit → Licit):     {tn_count:5d} ({tn_count/total*100:5.2f}%)")
    print(f"True Positives (Illicit → Illicit):  {tp_count:5d} ({tp_count/total*100:5.2f}%)")
    print(f"False Positives (Licit → Illicit):   {fp_count:5d} ({fp_count/total*100:5.2f}%)")
    print(f"False Negatives (Illicit → Licit):   {fn_count:5d} ({fn_count/total*100:5.2f}%)")
    print(f"{'-'*60}")
    print(f"Total Correct: {correct:5d} ({accuracy*100:.2f}%)")
    print(f"Total Errors:  {fp_count + fn_count:5d} ({(fp_count + fn_count)/total*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return df


def visualize_comparison(untrained_output, trained_output, true_labels, 
                        model_name="Model", save_path=None):
    """
    Side-by-side comparison of untrained vs trained model predictions.
    
    Args:
        untrained_output: Untrained model predictions [N, num_classes]
        trained_output: Trained model predictions [N, num_classes]
        true_labels: Ground truth labels [N]
        model_name: Name of the model
        save_path: Path to save figure
    """
    # Apply t-SNE to both
    print(f"Creating comparison visualization for {model_name}...")
    
    tsne = TSNE(n_components=2, init='pca', random_state=7, perplexity=30)
    
    # Combine data for consistent t-SNE projection
    combined = np.vstack([untrained_output, trained_output])
    tsne_res = tsne.fit_transform(combined)
    
    # Split back
    n = len(untrained_output)
    untrained_tsne = tsne_res[:n]
    trained_tsne = tsne_res[n:]
    
    # Get predictions
    untrained_pred = np.argmax(untrained_output, axis=1)
    trained_pred = np.argmax(trained_output, axis=1)
    
    # Define colors
    colors = {
        0: '#52D1DC',  # TN - Light Blue
        1: '#8D0004',  # TP - Dark Red
        2: '#FF7800',  # FP - Orange
        3: '#845218'   # FN - Brown
    }
    
    def get_categories(pred, true):
        cats = []
        for p, t in zip(pred, true):
            if t == 0 and p == 0: cats.append(0)  # TN
            elif t == 1 and p == 1: cats.append(1)  # TP
            elif t == 0 and p == 1: cats.append(2)  # FP
            elif t == 1 and p == 0: cats.append(3)  # FN
        return np.array(cats)
    
    untrained_cats = get_categories(untrained_pred, true_labels)
    trained_cats = get_categories(trained_pred, true_labels)
    
    # Calculate accuracies
    untrained_acc = np.mean(untrained_pred == true_labels)
    trained_acc = np.mean(trained_pred == true_labels)
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Untrained model
    for cat in range(4):
        mask = untrained_cats == cat
        if mask.sum() > 0:
            axes[0].scatter(
                untrained_tsne[mask, 0],
                untrained_tsne[mask, 1],
                c=colors[cat],
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
    
    axes[0].set_title(f'{model_name} - Untrained\nAccuracy: {untrained_acc:.2%}',
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Trained model
    for cat in range(4):
        mask = trained_cats == cat
        if mask.sum() > 0:
            axes[1].scatter(
                trained_tsne[mask, 0],
                trained_tsne[mask, 1],
                c=colors[cat],
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
    
    axes[1].set_title(f'{model_name} - Trained\nAccuracy: {trained_acc:.2%}',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Shared legend
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='TN: Licit → Licit'),
        Patch(facecolor=colors[1], edgecolor='black', label='TP: Illicit → Illicit'),
        Patch(facecolor=colors[2], edgecolor='black', label='FP: Licit → Illicit (Error)'),
        Patch(facecolor=colors[3], edgecolor='black', label='FN: Illicit → Licit (Error)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {save_path}")
    
    plt.show()
    
    # Print improvement
    improvement = (trained_acc - untrained_acc) * 100
    print(f"\n{'='*60}")
    print(f"{model_name} - Training Improvement")
    print(f"{'='*60}")
    print(f"Untrained Accuracy: {untrained_acc:.2%}")
    print(f"Trained Accuracy:   {trained_acc:.2%}")
    print(f"Improvement:        {improvement:+.2f}%")
    print(f"{'='*60}\n")


def visualize_error_analysis(model_output, true_labels, data_x, 
                             feature_names=None, title="Error Analysis"):
    """
    Detailed error analysis showing which types of transactions are misclassified.
    
    Args:
        model_output: Model predictions [N, num_classes]
        true_labels: Ground truth [N]
        data_x: Original feature matrix [N, num_features]
        feature_names: List of feature names (optional)
        title: Plot title
    """
    pred_labels = np.argmax(model_output, axis=1)
    
    # Identify errors
    fp_mask = (true_labels == 0) & (pred_labels == 1)  # False Positives
    fn_mask = (true_labels == 1) & (pred_labels == 0)  # False Negatives
    
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    
    print(f"\n{'='*60}")
    print(f"Error Analysis: {title}")
    print(f"{'='*60}")
    print(f"False Positives (Licit predicted as Illicit): {fp_count}")
    print(f"False Negatives (Illicit predicted as Licit): {fn_count}")
    
    if fp_count == 0 and fn_count == 0:
        print("Perfect predictions! No errors to analyze.")
        return
    
    # Analyze feature differences for errors
    if data_x is not None:
        correct_licit = data_x[(true_labels == 0) & (pred_labels == 0)]
        correct_illicit = data_x[(true_labels == 1) & (pred_labels == 1)]
        fp_features = data_x[fp_mask]
        fn_features = data_x[fn_mask]
        
        print(f"\nFeature Statistics:")
        print(f"  Correct Licit:   Mean = {correct_licit.mean():.4f}, Std = {correct_licit.std():.4f}")
        print(f"  Correct Illicit: Mean = {correct_illicit.mean():.4f}, Std = {correct_illicit.std():.4f}")
        
        if fp_count > 0:
            print(f"  False Positives: Mean = {fp_features.mean():.4f}, Std = {fp_features.std():.4f}")
        if fn_count > 0:
            print(f"  False Negatives: Mean = {fn_features.mean():.4f}, Std = {fn_features.std():.4f}")
    
    print(f"{'='*60}\n")
    
    # Visualize confidence distribution for errors
    probs = np.exp(model_output) / np.exp(model_output).sum(axis=1, keepdims=True)
    pred_confidence = probs[np.arange(len(probs)), pred_labels]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # False Positives confidence
    if fp_count > 0:
        axes[0].hist(pred_confidence[fp_mask], bins=20, color='#FF7800', 
                    alpha=0.7, edgecolor='black')
        axes[0].set_title(f'False Positive Confidence Distribution\n(n={fp_count})',
                         fontweight='bold')
        axes[0].set_xlabel('Prediction Confidence')
        axes[0].set_ylabel('Count')
        axes[0].axvline(pred_confidence[fp_mask].mean(), color='red', 
                       linestyle='--', linewidth=2, label='Mean')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # False Negatives confidence
    if fn_count > 0:
        axes[1].hist(pred_confidence[fn_mask], bins=20, color='#845218', 
                    alpha=0.7, edgecolor='black')
        axes[1].set_title(f'False Negative Confidence Distribution\n(n={fn_count})',
                         fontweight='bold')
        axes[1].set_xlabel('Prediction Confidence')
        axes[1].set_ylabel('Count')
        axes[1].axvline(pred_confidence[fn_mask].mean(), color='red', 
                       linestyle='--', linewidth=2, label='Mean')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}_error_analysis.png', dpi=150)
    plt.show()



# ============================================================
# Universal Model Trainer
# ============================================================

class ModelTrainer:
    """
    Universal trainer for MLP, GCN, and GAT models.
    Handles training, evaluation, visualization, and comparison.
    """
    
    def __init__(self, model, data, model_name, lr=0.01, weight_decay=1e-4):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model (MLP, GCN, or GAT)
            data: PyG data object containing graph
            model_name: String name for the model (e.g., "MLP", "GCN", "GAT")
            lr: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.data = data
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model and data to device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0
        self.best_epoch = 0
        self.best_model_state = None
        
        print(f"\n{'='*60}")
        print(f"Initialized {model_name} Trainer")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Parameters: {self.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self):
        """Execute one training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass (works for both MLP and GCN/GAT)
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss on training nodes
        loss = self.criterion(
            out[self.data.train_mask], 
            self.data.y[self.data.train_mask]
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mask):
        """
        Evaluate model on given mask.
        
        Args:
            mask: Boolean tensor for node selection
            
        Returns:
            Dictionary with accuracy and F1 score
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
        
        pred = out.argmax(dim=1)
        
        # Calculate accuracy
        correct = pred[mask] == self.data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        
        # Calculate F1 score
        y_true = self.data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {'accuracy': acc, 'f1': f1}
    
    def train(self, num_epochs=1000, print_every=100, early_stopping_patience=None):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of training epochs
            print_every: Print progress every N epochs
            early_stopping_patience: Stop if no improvement for N epochs (None = disabled)
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            loss = self.train_epoch()
            
            # Evaluation
            train_metrics = self.evaluate(self.data.train_mask)
            val_metrics = self.evaluate(self.data.val_mask)
            
            # Track history
            self.history['train_loss'].append(loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.best_model_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:04d} | "
                      f"Loss: {loss:.4f} | "
                      f"Train Acc: {train_metrics['accuracy']:.4f} | "
                      f"Train F1: {train_metrics['f1']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"No improvement for {early_stopping_patience} epochs")
                break
        
        # Load best model
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print(f"{'='*60}")
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
    
    def test(self):
        """Evaluate on test set and return detailed metrics."""
        print(f"\n{'='*60}")
        print(f"{self.model_name} - Final Test Evaluation")
        print(f"{'='*60}")
        
        test_metrics = self.evaluate(self.data.test_mask)
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1']:.4f}")
        
        # Detailed classification report
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
        
        pred = out.argmax(dim=1)
        y_true = self.data.y[self.data.test_mask].cpu().numpy()
        y_pred = pred[self.data.test_mask].cpu().numpy()
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Licit', 'Illicit'],
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Licit  Illicit")
        print(f"Actual Licit     {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"       Illicit   {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        if cm.size == 4:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            print(f"\nAdditional Metrics:")
            print(f"  True Negatives:  {tn:5d}")
            print(f"  False Positives: {fp:5d}")
            print(f"  False Negatives: {fn:5d}")
            print(f"  True Positives:  {tp:5d}")
            if (fp + tn) > 0:
                print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
            if (fn + tp) > 0:
                print(f"  False Negative Rate: {fn/(fn+tp):.4f}")
        
        print(f"{'='*60}\n")
        
        return test_metrics
    
    def plot_learning_curves(self, save_path=None):
        """Plot comprehensive learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(len(self.history['train_loss']))
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title(f'{self.model_name} - Training Loss', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', 
                       label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', 
                       label='Validation', linewidth=2)
        axes[0, 1].set_title(f'{self.model_name} - Accuracy', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.history['train_f1'], 'b-', 
                       label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', 
                       label='Validation', linewidth=2)
        axes[1, 0].axhline(y=self.best_val_f1, color='g', 
                          linestyle='--', alpha=0.5, label='Best Val F1')
        axes[1, 0].set_title(f'{self.model_name} - F1 Score', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting Gap
        gap = np.array(self.history['train_f1']) - np.array(self.history['val_f1'])
        axes[1, 1].plot(epochs, gap, 'g-', linewidth=2)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title(f'{self.model_name} - Overfitting Gap', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train F1 - Val F1')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{self.model_name}_learning_curves.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning curves: {save_path}")
        plt.show()
    
    def get_predictions(self, mask):
        """Get predictions and probabilities for given mask."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
        
        pred = out.argmax(dim=1)
        probs = torch.softmax(out, dim=1)
        
        return {
            'predictions': pred[mask].cpu().numpy(),
            'probabilities': probs[mask].cpu().numpy(),
            'true_labels': self.data.y[mask].cpu().numpy()
        }
    
    def visualize_predictions_tsne(self, mask, mask_name="Test", save_path=None):
        """
        Visualize predictions on given mask using t-SNE with error highlighting.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
        
        # Get data for visualization
        output = out[mask].cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()
        
        if save_path is None:
            save_path = f'{self.model_name}_{mask_name}_predictions.png'
        
        title = f'{self.model_name} - {mask_name} Set Predictions'
        df = visualize_predictions(output, labels, title=title, save_path=save_path)
        
        return df

    def visualize_before_after(self, mask, untrained_model, mask_name="Test"):
        """
        Compare untrained vs trained model predictions.
        """
        untrained_model = untrained_model.to(self.device)
        # Untrained predictions
        untrained_model.eval()
        with torch.no_grad():
            untrained_out = untrained_model(self.data.x, self.data.edge_index)
        
        # Trained predictions
        self.model.eval()
        with torch.no_grad():
            trained_out = self.model(self.data.x, self.data.edge_index)
        
        # Get data
        untrained_output = untrained_out[mask].cpu().numpy()
        trained_output = trained_out[mask].cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()
        
        save_path = f'{self.model_name}_{mask_name}_comparison.png'
        visualize_comparison(untrained_output, trained_output, labels, 
                            model_name=self.model_name, save_path=save_path)

    def analyze_errors(self, mask, mask_name="Test"):
        """
        Detailed error analysis for the model.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
        
        output = out[mask].cpu().numpy()
        labels = self.data.y[mask].cpu().numpy()
        features = self.data.x[mask].cpu().numpy()
        
        visualize_error_analysis(output, labels, features, 
                                title=f'{self.model_name} {mask_name} Set')



# ============================================================
# Model Comparison Utility
# ============================================================

class ModelComparison:
    """Compare multiple trained models."""
    
    def __init__(self):
        self.results = {}
    
    def add_model(self, model_name, trainer):
        """Add a trained model's results."""
        test_metrics = trainer.evaluate(trainer.data.test_mask)
        self.results[model_name] = {
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'best_val_f1': trainer.best_val_f1,
            'best_epoch': trainer.best_epoch,
            'num_parameters': trainer.count_parameters(),
            'history': trainer.history
        }
    
    def print_comparison(self):
        """Print comparison table."""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"{'Model':<10} | {'Params':<10} | {'Test Acc':<10} | {'Test F1':<10} | {'Best Val F1':<12} | {'Best Epoch':<10}")
        print(f"{'-'*80}")
        
        for name, metrics in self.results.items():
            print(f"{name:<10} | "
                  f"{metrics['num_parameters']:<10,} | "
                  f"{metrics['test_accuracy']:<10.4f} | "
                  f"{metrics['test_f1']:<10.4f} | "
                  f"{metrics['best_val_f1']:<12.4f} | "
                  f"{metrics['best_epoch']:<10}")
        
        print(f"{'='*80}\n")
    
    def plot_comparison(self, save_path='model_comparison.png'):
        """Plot side-by-side comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = list(self.results.keys())
        
        # Test Accuracy
        test_accs = [self.results[m]['test_accuracy'] for m in models]
        axes[0].bar(models, test_accs, color=['blue', 'green', 'red'])
        axes[0].set_title('Test Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim([0.9, 1.0])
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(test_accs):
            axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Test F1
        test_f1s = [self.results[m]['test_f1'] for m in models]
        axes[1].bar(models, test_f1s, color=['blue', 'green', 'red'])
        axes[1].set_title('Test F1 Score', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_ylim([0.9, 1.0])
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(test_f1s):
            axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Parameters
        params = [self.results[m]['num_parameters'] for m in models]
        axes[2].bar(models, params, color=['blue', 'green', 'red'])
        axes[2].set_title('Model Parameters', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Number of Parameters')
        axes[2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(params):
            axes[2].text(i, v + max(params)*0.02, f'{v:,}', 
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")
        plt.show()
    
    def plot_f1_curves(self, save_path='f1_comparison.png'):
        """Plot F1 curves for all models."""
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (name, metrics) in enumerate(self.results.items()):
            epochs = range(len(metrics['history']['val_f1']))
            plt.plot(epochs, metrics['history']['val_f1'], 
                    label=f"{name} (Best: {metrics['best_val_f1']:.4f})",
                    linewidth=2, color=colors[i % len(colors)])
        
        plt.title('Validation F1 Score Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved F1 comparison: {save_path}")
        plt.show()

