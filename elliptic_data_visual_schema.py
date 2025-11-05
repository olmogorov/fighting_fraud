#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 18:18:54 2025

@author: otto
"""

# ============================================================
# ELLIPTIC BITCOIN DATASET VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt

#import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
#import numpy as np

def visualize_elliptic_dataset_overview(data, dataset, save_path='elliptic_dataset_overview.png'):
    """
    Create a comprehensive overview visualization of the Elliptic Bitcoin dataset.
    
    Args:
        data: PyG data object
        dataset: PyG dataset object
        save_path: Path to save the figure
    """
    # Calculate statistics
    total_nodes = data.x.shape[0]
    total_edges = data.edge_index.shape[1]
    num_features = data.x.shape[1]
    
    # Count labels
    y_values = data.y.numpy()
    licit_count = (y_values == 0).sum()
    illicit_count = (y_values == 1).sum()
    unlabeled_count = (y_values == 2).sum()
    
    # Train/Test split
    train_total = data.train_mask.sum().item()
    test_total = data.test_mask.sum().item()
    
    # Count train/test by class
    train_licit = ((data.y == 0) & data.train_mask).sum().item()
    train_illicit = ((data.y == 1) & data.train_mask).sum().item()
    test_licit = ((data.y == 0) & data.test_mask).sum().item()
    test_illicit = ((data.y == 1) & data.test_mask).sum().item()
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(18, 13))  # Increased height
    
    # ============================================================
    # 1. GRAPH VISUALIZATION (Top)
    # ============================================================
    
    ax_graph = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=1)
    ax_graph.set_xlim(-1, 11)
    ax_graph.set_ylim(-1, 6)
    ax_graph.axis('off')
    
    # Title
    ax_graph.text(0, 5.5, 'Elliptic Bitcoin Dataset', 
                  fontsize=20, fontweight='bold', va='top')
    
    # Node positions for graph visualization
    nodes = {
        # Unlabeled nodes (gray)
        'u1': (1, 2.5), 'u2': (2, 1.5), 'u3': (3, 0.5), 
        'u4': (4, 1), 'u5': (5, 2), 'u6': (7, 3.5), 'u7': (8, 2),
        
        # Training nodes (white with "Train" label)
        't1': (3.5, 4), 't2': (4.5, 2.5), 't3': (6, 3.5),
        
        # Test nodes (white with "Test" label)
        'test1': (7.5, 4), 'test2': (9, 3),
        
        # Special nodes (black - illicit)
        'illicit_train': (5, 4.5), 'illicit_test': (7, 2.5)
    }
    
    # Draw edges (dashed lines)
    edges = [
        ('illicit_train', 't1'), ('illicit_train', 't2'), 
        ('illicit_train', 't3'), ('t1', 'u1'), ('t1', 't2'),
        ('t2', 'u2'), ('t2', 'illicit_test'), ('t3', 'test1'),
        ('illicit_test', 'u4'), ('illicit_test', 'u5'),
        ('test1', 'u6'), ('test1', 'test2'), ('u6', 'u7'),
        ('u1', 'u2'), ('u2', 'u3'), ('u3', 'u4'), ('u4', 'u5'),
        ('u5', 't3'), ('u6', 'illicit_test')
    ]
    
    for start, end in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                               arrowstyle='->', mutation_scale=15,
                               linestyle='--', linewidth=1, 
                               color='gray', alpha=0.6, zorder=1)
        ax_graph.add_patch(arrow)
    
    # Draw nodes
    node_styles = {
        # Unlabeled (gray)
        'unlabeled': (['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7'], 
                      'lightgray', 'black', ''),
        # Train nodes (white)
        'train': (['t1', 't2', 't3'], 'white', 'black', 'Train'),
        # Test nodes (white)
        'test': (['test1', 'test2'], 'white', 'black', 'Test'),
        # Illicit nodes (black)
        'illicit_train': (['illicit_train'], 'black', 'white', 'Train'),
        'illicit_test': (['illicit_test'], 'black', 'white', 'Test')
    }
    
    for style_name, (node_list, facecolor, textcolor, label) in node_styles.items():
        for node in node_list:
            x, y = nodes[node]
            circle = Circle((x, y), 0.3, facecolor=facecolor, 
                          edgecolor='black', linewidth=2, zorder=2)
            ax_graph.add_patch(circle)
            if label:
                ax_graph.text(x, y, label, ha='center', va='center',
                            fontsize=8, fontweight='bold', color=textcolor, zorder=3)
    
    # ============================================================
    # 2. LEGEND (Bottom Left)
    # ============================================================
    
    ax_legend = plt.subplot2grid((4, 3), (1, 0), rowspan=3)
    ax_legend.set_xlim(0, 10)
    ax_legend.set_ylim(0, 15)
    ax_legend.axis('off')
    
    ax_legend.text(1, 14, 'Legend', fontsize=16, fontweight='bold')
    
    legend_items = [
        (12, 'Licit\ntransaction', 'white', 'black'),
        (10, 'Illicit\ntransaction', 'black', 'white'),
        (8, 'Unlabeled\ntransaction', 'lightgray', 'black'),
        (6, 'Payment\nflow', None, None),
        (4, 'Training set\ntransaction', 'white', 'black'),
        (2, 'Test set\ntransaction', 'white', 'black')
    ]
    
    for i, (y, label, facecolor, textcolor) in enumerate(legend_items):
        if facecolor:  # Circle items
            circle = Circle((1.5, y), 0.4, facecolor=facecolor, 
                          edgecolor='black', linewidth=2)
            ax_legend.add_patch(circle)
            
            # Add "Train" or "Test" label for last two items
            if i >= 4:
                ax_legend.text(1.5, y, 'Train' if i == 4 else 'Test',
                             ha='center', va='center', fontsize=7, 
                             fontweight='bold', color='black')
        else:  # Arrow item
            arrow = FancyArrowPatch((1, y), (2, y), arrowstyle='->',
                                  mutation_scale=15, linewidth=2, color='black')
            ax_legend.add_patch(arrow)
        
        ax_legend.text(3, y, label, fontsize=11, va='center')
    
    # ============================================================
    # 3. STATISTICS (Bottom Middle)
    # ============================================================
    
    ax_stats = plt.subplot2grid((4, 3), (1, 1), rowspan=3)
    ax_stats.set_xlim(0, 10)
    ax_stats.set_ylim(0, 15)
    ax_stats.axis('off')
    
    ax_stats.text(1, 14, 'Statistics', fontsize=16, fontweight='bold')
    
    # Overall statistics
    y_pos = 12
    stats_items = [
        (licit_count, 'white', 'black'),
        (illicit_count, 'black', 'white'),
        (unlabeled_count, 'lightgray', 'black')
    ]
    
    for count, facecolor, textcolor in stats_items:
        circle = Circle((1.5, y_pos), 0.4, facecolor=facecolor, 
                       edgecolor='black', linewidth=2)
        ax_stats.add_patch(circle)
        ax_stats.text(3.5, y_pos, f'{count:,}', fontsize=12, va='center')
        y_pos -= 1.8
    
    # Total nodes - moved higher with larger font
    ax_stats.text(2.5, y_pos - 0.2, f'Total  {total_nodes:,}', 
                  fontsize=12, va='center', ha='center', fontweight='bold')
    
    # Edges with better spacing
    y_pos -= 1.8
    arrow = FancyArrowPatch((1, y_pos), (2, y_pos), 
                           arrowstyle='->', mutation_scale=15, 
                           linewidth=2, color='black')
    ax_stats.add_patch(arrow)
    ax_stats.text(3.5, y_pos, f'{total_edges:,}', fontsize=12, va='center')
    
    # Train/Test split with more spacing
    y_pos = 4
    
    # Train section
    train_circle_licit = Circle((1.5, y_pos), 0.35, facecolor='white', 
                               edgecolor='black', linewidth=2)
    ax_stats.add_patch(train_circle_licit)
    ax_stats.text(1.5, y_pos, 'Train', ha='center', va='center', 
                  fontsize=7, fontweight='bold')
    ax_stats.text(3, y_pos, f'{train_licit:,}', fontsize=11, va='center')
    
    train_circle_illicit = Circle((1.5, y_pos - 1), 0.35, facecolor='black', 
                                 edgecolor='black', linewidth=2)
    ax_stats.add_patch(train_circle_illicit)
    ax_stats.text(1.5, y_pos - 1, 'Train', ha='center', va='center', 
                  fontsize=7, fontweight='bold', color='white')
    ax_stats.text(3, y_pos - 1, f'{train_illicit:,}', fontsize=11, va='center')
    
    # Arrow for train total
    arrow_train = FancyArrowPatch((1, y_pos - 1.8), (2, y_pos - 1.8),
                                 arrowstyle='->', mutation_scale=15,
                                 linewidth=2, color='black')
    ax_stats.add_patch(arrow_train)
    ax_stats.text(3, y_pos - 1.8, f'{total_edges:,}', 
                  fontsize=11, va='center')
    ax_stats.text(1.5, y_pos - 2.4, f'Total  {train_total:,}', 
                  fontsize=10, ha='center', fontweight='bold')
    
    # Test section
    test_circle_licit = Circle((6, y_pos), 0.35, facecolor='white', 
                              edgecolor='black', linewidth=2)
    ax_stats.add_patch(test_circle_licit)
    ax_stats.text(6, y_pos, 'Test', ha='center', va='center', 
                  fontsize=7, fontweight='bold')
    ax_stats.text(7.5, y_pos, f'{test_licit:,}', fontsize=11, va='center')
    
    test_circle_illicit = Circle((6, y_pos - 1), 0.35, facecolor='black', 
                                edgecolor='black', linewidth=2)
    ax_stats.add_patch(test_circle_illicit)
    ax_stats.text(6, y_pos - 1, 'Test', ha='center', va='center', 
                  fontsize=7, fontweight='bold', color='white')
    ax_stats.text(7.5, y_pos - 1, f'{test_illicit:,}', fontsize=11, va='center')
    
    # Arrow for test total
    arrow_test = FancyArrowPatch((5.5, y_pos - 1.8), (6.5, y_pos - 1.8),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color='black')
    ax_stats.add_patch(arrow_test)
    ax_stats.text(7.5, y_pos - 1.8, f'{test_illicit:,}', 
                  fontsize=11, va='center')
    ax_stats.text(6, y_pos - 2.4, f'Total  {test_total:,}', 
                  fontsize=10, ha='center', fontweight='bold')
    
    # ============================================================
    # 4. FEATURES (Bottom Right)
    # ============================================================
    
    ax_features = plt.subplot2grid((4, 3), (1, 2), rowspan=3)
    ax_features.set_xlim(0, 10)
    ax_features.set_ylim(0, 15)
    ax_features.axis('off')
    
    ax_features.text(1, 14, 'Features', fontsize=16, fontweight='bold')
    
    # Main circle
    circle = Circle((2, 11), 0.6, facecolor='white', edgecolor='black', linewidth=2)
    ax_features.add_patch(circle)
    
    ax_features.text(2, 11, '165', ha='center', va='center', 
                     fontsize=11, fontweight='bold')
    ax_features.text(2, 12.3, 'transaction', ha='center', va='center', fontsize=10)
    ax_features.text(2, 12.0, 'features', ha='center', va='center', fontsize=10)
    
    # 94 local features box (LARGER - fixed size)
    local_box = FancyBboxPatch((3.5, 9.0), 5.5, 3.8,  # Increased height to 3.2
                               boxstyle="round,pad=0.15", 
                               edgecolor='black', facecolor='white', linewidth=2)
    ax_features.add_patch(local_box)
    
    ax_features.text(6.25, 12.4, '94 local', fontsize=11, ha='center', fontweight='bold')
    ax_features.text(6.25, 12.0, 'features', fontsize=11, ha='center', fontweight='bold')
    
    # Local features list (removed one "..." to prevent overflow)
    local_features = [
        '- transaction_time',
        '- num_inputs',
        '- num_outputs',
        '- transaction_fee',
        '- avg_btc_per_input',
        '- avg_btc_per_output',
        '- avg_num_transactions_per_input'
    ]
    
    y_text = 11.5
    for feat in local_features:
        ax_features.text(3.8, y_text, feat, fontsize=8.5, va='top', family='monospace')
        y_text -= 0.30
    
    # Add "..." at the end inside the box
    ax_features.text(3.8, y_text, '...', fontsize=8.5, va='top', family='monospace')
    
    # Arrow from circle to local box
    arrow = FancyArrowPatch((2.6, 11.0), (3.4, 11.2), arrowstyle='->',
                           mutation_scale=20, linewidth=2, color='black')
    ax_features.add_patch(arrow)
    
    # 71 aggregate features box (LARGER)
    aggregate_box = FancyBboxPatch((3.5, 5.5), 5.5, 3.2,  # Increased height to 4.2
                                  boxstyle="round,pad=0.15", 
                                  edgecolor='black', facecolor='white', linewidth=2)
    ax_features.add_patch(aggregate_box)
    
    ax_features.text(6.25, 8.4, '71 aggregate neighbor', 
                     fontsize=11, ha='center', fontweight='bold')
    ax_features.text(6.25, 8.0, 'features', fontsize=11, ha='center', fontweight='bold')
    
    aggregate_features = [
        '- max_transaction_time_of_neighbor_',
        '  for_same_inputs_and_outputs',
        '- min_transaction_btc_of_neighbor_',
        '  for_same_transaction_fee',
        '- std_num_transactions_of_neighbors_',
        '  for_same_outputs'
    ]
    
    y_text = 7.5
    for feat in aggregate_features:
        ax_features.text(3.8, y_text, feat, fontsize=8.5, va='top', family='monospace')
        y_text -= 0.30
    
    # Add blank line and "..." inside the aggregate box
    ax_features.text(3.8, y_text - 0.2, '...', fontsize=8.5, va='top', family='monospace')
    
    # Arrow from circle to aggregate box (with curve)
    arrow2 = FancyArrowPatch((2.0, 10.4), (3.4, 8.5), arrowstyle='->',
                            mutation_scale=20, linewidth=2, color='black',
                            connectionstyle="arc3,rad=0.3")
    ax_features.add_patch(arrow2)
    
    # ============================================================
    # Caption
    # ============================================================
    
    caption = (
        "Demonstration of the Elliptic Bitcoin dataset with key statistics and feature descriptions. The\n"
        "dataset contains a total of 203769 nodes connected via 234355 edges. Each node is a transaction and is\n"
        "labeled licit, illicit, or neither. The dataset is divided into a training and a test set. Each node contains 165\n"
        "features representing different pieces of information associated with a Bitcoin transaction."
    )
    
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', 
             fontsize=11, wrap=True, weight='bold')
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ELLIPTIC BITCOIN DATASET SUMMARY")
    print("="*60)
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print('======================')
    print(f"Total Nodes:       {total_nodes:,}")
    print(f"Total Edges:       {total_edges:,}")
    print(f"Features per Node: {num_features}")
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print('======================')
    print(f"\nClass Distribution:")
    print(f"  Licit:      {licit_count:,} ({licit_count/total_nodes*100:.1f}%)")
    print(f"  Illicit:    {illicit_count:,} ({illicit_count/total_nodes*100:.1f}%)")
    print(f"  Unlabeled:  {unlabeled_count:,} ({unlabeled_count/total_nodes*100:.1f}%)")
    print(f"\nTrain/Test Split:")
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f"  Training:   {train_total:,} nodes")
    print(f"    - Licit:   {train_licit:,}")
    print(f"    - Illicit: {train_illicit:,}")
    print(f"  Testing:    {test_total:,} nodes")
    print(f"    - Licit:   {test_licit:,}")
    print(f"    - Illicit: {test_illicit:,}")
    print("="*60 + "\n")
    print(data, '\n')
    print("="*60)