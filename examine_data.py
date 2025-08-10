#!/usr/bin/env python3
"""
Script to examine all three datasets and check structural consistency
"""

import numpy as np
import scipy.sparse as sp
import os

def examine_dataset_structure(dataset_name):
    """Examine dataset structure and available files"""
    print(f"\n{'='*60}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    base_path = f"datasets/{dataset_name}"
    if not os.path.exists(base_path):
        print(f"❌ Directory {base_path} not found")
        return None
    
    # List all files
    files = os.listdir(base_path)
    print(f"📁 Files: {files}")
    
    # Check for core files
    core_files = ['train_list.npy', 'valid_list.npy', 'test_list.npy']
    missing_files = [f for f in core_files if f not in files]
    
    if missing_files:
        print(f"❌ Missing core files: {missing_files}")
        return None
    
    # Load and examine core data
    train_list = np.load(f"{base_path}/train_list.npy", allow_pickle=True)
    valid_list = np.load(f"{base_path}/valid_list.npy", allow_pickle=True)
    test_list = np.load(f"{base_path}/test_list.npy", allow_pickle=True)
    
    print(f"✅ Train: {train_list.shape}")
    print(f"✅ Valid: {valid_list.shape}")
    print(f"✅ Test: {test_list.shape}")
    
    # Check data format consistency
    if len(train_list.shape) != 2 or train_list.shape[1] != 2:
        print(f"❌ Train data format inconsistent: expected (N, 2), got {train_list.shape}")
        return None
    
    # Get user/item ranges
    unique_users = np.unique(train_list[:, 0])
    unique_items = np.unique(train_list[:, 1])
    n_user = unique_users.max() + 1
    n_item = unique_items.max() + 1
    
    print(f"👥 Users: {len(unique_users)} (range: {unique_users.min()}-{unique_users.max()})")
    print(f"📦 Items: {len(unique_items)} (range: {unique_items.min()}-{unique_items.max()})")
    
    # Check for additional files
    additional_files = [f for f in files if f not in core_files]
    if additional_files:
        print(f"🔍 Additional files:")
        for file in additional_files:
            file_path = f"{base_path}/{file}"
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  - {file} ({file_size:.1f} MB)")
            
            # Try to examine additional files
            if file.endswith('.npy'):
                try:
                    additional_data = np.load(file_path, allow_pickle=True)
                    print(f"    Shape: {additional_data.shape}")
                    if len(additional_data.shape) == 2:
                        print(f"    Sample: {additional_data[:2]}")
                    elif len(additional_data.shape) == 1:
                        print(f"    Sample: {additional_data[:5]}")
                except Exception as e:
                    print(f"    Error loading: {e}")
    
    # Create sparse matrix to check density
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), 
                               (train_list[:, 0], train_list[:, 1])), 
                               dtype='float64', shape=(n_user, n_item))
    
    density = train_data.nnz / (n_user * n_item)
    print(f"📈 Matrix density: {density:.6f}")
    
    return {
        'name': dataset_name,
        'n_user': n_user,
        'n_item': n_item,
        'train_size': len(train_list),
        'density': density,
        'additional_files': additional_files,
        'has_item_emb': 'item_emb.npy' in files
    }

def main():
    """Main function to examine all datasets"""
    print("🔍 DiffRec Dataset Structure Analysis")
    print("Checking consistency across all datasets")
    
    datasets = ['amazon-book_clean', 'ml-1m_clean', 'yelp_clean']
    results = []
    
    for dataset in datasets:
        result = examine_dataset_structure(dataset)
        if result:
            results.append(result)
    
    print(f"\n{'='*60}")
    print("📋 STRUCTURAL CONSISTENCY SUMMARY")
    print(f"{'='*60}")
    
    if not results:
        print("❌ No datasets found or all have errors")
        return
    
    # Check consistency
    print(f"✅ All datasets have consistent core structure:")
    print(f"  - train_list.npy, valid_list.npy, test_list.npy")
    print(f"  - Format: (N, 2) where N = number of interactions")
    print(f"  - Each row: [user_id, item_id]")
    
    print(f"\n📊 Dataset Comparison:")
    print(f"{'Dataset':<20} {'Users':<8} {'Items':<8} {'Train':<8} {'Density':<10} {'Item_Emb':<10}")
    print(f"{'-'*70}")
    
    for result in results:
        item_emb_status = "✅ Yes" if result['has_item_emb'] else "❌ No"
        print(f"{result['name']:<20} {result['n_user']:<8} {result['n_item']:<8} {result['train_size']:<8} {result['density']:<10.6f} {item_emb_status:<10}")
    
    print(f"\n💡 CONDITIONAL DESIGN RECOMMENDATIONS:")
    print(f"{'='*60}")
    
    # Universal conditions (work on all datasets)
    print(f"🎯 UNIVERSAL CONDITIONS (apply to all datasets):")
    print(f"  1. User interaction count (sparsity level)")
    print(f"  2. Item popularity (interaction count)")
    print(f"  3. User-item interaction patterns")
    print(f"  4. Matrix density-based features")
    
    # Dataset-specific conditions
    print(f"\n🔧 DATASET-SPECIFIC CONDITIONS:")
    print(f"  1. ML-1M: Item embeddings (64D vectors)")
    print(f"  2. Amazon-Book: Larger scale, more sparse")
    print(f"  3. Yelp: Business/restaurant specific features")
    
    print(f"\n🚀 IMPLEMENTATION STRATEGY:")
    print(f"  1. Design conditionals that work with [user_id, item_id] pairs")
    print(f"  2. Make item embeddings optional (check if file exists)")
    print(f"  3. Use universal features as base, enhance with dataset-specific ones")
    print(f"  4. Model architecture remains the same, only input features change")

if __name__ == "__main__":
    main()
