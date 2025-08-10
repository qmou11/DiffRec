# DiffRec Conditional Training Implementation

This document describes the implementation of conditional training functionality for the DiffRec (Diffusion-based Recommendation) model. The conditional approach allows the model to incorporate user group information to guide recommendation generation.

## 🎯 Overview

We implemented a conditional diffusion model that can incorporate user group information (active vs. inactive users) to guide the recommendation generation process. This enables personalized recommendations based on user behavior patterns.

## 🔧 Files Modified

### 1. `data_utils.py` - Data Preparation & Loading

**Changes Made:**
- **Added `create_user_groups()` function**: Creates binary user classification based on median interaction count
  - Group 0: Inactive users (≤42 interactions)
  - Group 1: Active users (>42 interactions)
- **Modified `data_load()` function**: Added `create_conditions` parameter to optionally generate user groups
- **Updated `DataDiffusion` class**: 
  - Added `user_groups` parameter to constructor
  - Modified `__getitem__()` to return both interaction data and one-hot encoded user group tensor

**Key Code Changes:**
```python
# New function for binary user grouping
def create_user_groups(train_data, n_groups=2):
    user_interactions = np.array(train_data.sum(axis=1)).flatten()
    median_interactions = np.median(user_interactions)
    user_groups = (user_interactions > median_interactions).astype(int)
    return user_groups, group_stats

# Modified DataDiffusion.__getitem__
def __getitem__(self, index):
    item = self.data[index]
    if self.user_groups is not None:
        user_group = self.user_groups[index]
        user_group_onehot = torch.zeros(2, dtype=torch.float32)
        user_group_onehot[user_group] = 1.0
        return item, user_group_onehot
    return item
```

### 2. `models/DNN.py` - Neural Network Architecture

**Changes Made:**
- **Added conditional parameters**: `use_conditionals` (boolean) and `conditional_dim` (integer, default 2)
- **Added conditional embedding layer**: `self.conditional_emb_layer = nn.Linear(2, 2)`
- **Modified input dimensions**: Adjusted first layer to accommodate conditional input
- **Updated `forward()` method**: Added `conditionals` parameter and conditional processing logic
- **Enhanced weight initialization**: Added initialization for conditional embedding layer

**Key Code Changes:**
```python
def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=True, 
             dropout=0.5, use_conditionals=False, conditional_dim=2):
    # ... existing code ...
    if self.use_conditionals:
        self.conditional_emb_layer = nn.Linear(self.conditional_dim, self.conditional_dim)
    
    # Adjust input layer dimensions for conditionals
    conditional_input_size = self.conditional_dim if self.use_conditionals else 0
    self.in_layers.append(nn.Linear(self.in_dims[0] + self.time_emb_dim + conditional_input_size, self.in_dims[1]))

def forward(self, x, timesteps, conditionals=None):
    # ... existing time embedding code ...
    
    # Add conditionals if provided and enabled
    if self.use_conditionals and conditionals is not None:
        conditional_emb = self.conditional_emb_layer(conditionals)
        h = torch.cat([h, conditional_emb], dim=-1)
    
    # ... rest of forward pass ...
```

### 3. `models/gaussian_diffusion.py` - Diffusion Process

**Changes Made:**
- **Modified `training_losses()`**: Added `conditionals` parameter and passed it to model calls
- **Modified `p_mean_variance()`**: Added `conditionals` parameter and passed it to model calls  
- **Modified `p_sample()`**: Added `conditionals` parameter and passed it through the sampling process
- **Fixed `UnboundLocalError`**: Added `loss = mse` in the else branch

**Key Code Changes:**
```python
def training_losses(self, model, x_start, reweight=False, conditionals=None):
    # ... existing code ...
    if reweight:
        # ... reweighted loss calculation ...
    else:
        loss = mse  # Fixed: ensure loss is always defined
    
    # Pass conditionals to model
    model_output = model(x_t, ts, conditionals)
    
def p_mean_variance(self, model, x, t, conditionals=None):
    # ... existing code ...
    # Pass conditionals to model
    model_output = model(x, t, conditionals)
    
def p_sample(self, model, x_start, steps, sampling_noise=False, conditionals=None):
    # ... existing code ...
    # Pass conditionals through the sampling process
    out = self.p_mean_variance(model, x_t, t, conditionals)
```

### 4. `main_conditional.py` - New Main Training Script

**New File Created:**
- **Added `--use_conditionals` argument**: Boolean flag to enable/disable conditional training
- **Modified data loading**: Calls `data_utils.data_load()` with `create_conditions=args.use_conditionals`
- **Updated dataset creation**: Passes `user_groups` to `DataDiffusion` when conditionals enabled
- **Modified model initialization**: Sets `use_conditionals=True` and `conditional_dim=2` for conditional DNN
- **Updated training loop**: Handles both standard and conditional data formats
- **Modified diffusion calls**: Passes `batch_conditionals` to diffusion methods when appropriate

**Key Code Changes:**
```python
# New argument
parser.add_argument('--use_conditionals', action='store_true', 
                   help='Enable conditional training with user groups')

# Conditional data loading
if args.use_conditionals:
    train_data, valid_y_data, test_y_data, n_user, n_item, user_groups, group_stats = \
        data_utils.data_load(train_path, valid_path, test_path, create_conditions=True)
else:
    train_data, valid_y_data, test_y_data, n_user, n_item, _, _ = \
        data_utils.data_load(train_path, valid_path, test_path, create_conditions=False)

# Conditional dataset creation
if args.use_conditionals:
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()), user_groups)
else:
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))

# Conditional model initialization
if args.use_conditionals:
    model = DNN(..., use_conditionals=True, conditional_dim=2)
else:
    model = DNN(..., use_conditionals=False)

# Conditional training loop
if args.use_conditionals:
    for batch_data, batch_conditionals in train_loader:
        # ... training with conditionals ...
        loss = diffusion.training_losses(model, batch_data, reweight=args.reweight, 
                                       conditionals=batch_conditionals)
else:
    for batch_data in train_loader:
        # ... standard training ...
        loss = diffusion.training_losses(model, batch_data, reweight=args.reweight)
```

### 5. `compare_conditionals.py` - New Comparison Script

**New File Created:**
- **Automated comparison**: Runs both standard and conditional training automatically
- **Performance tracking**: Captures training time, exit codes, and logs
- **Results analysis**: Provides comparison summary and saves detailed logs
- **Configurable parameters**: Allows easy modification of dataset, epochs, and batch size

**Key Features:**
```python
def run_training(use_conditionals, dataset, epochs, batch_size):
    # Build command with appropriate flags
    cmd = f"python3 main_conditional.py --dataset {dataset} --epochs {epochs} --batch_size {batch_size}"
    if use_conditionals:
        cmd += " --use_conditionals"
    
    # Execute and capture output
    # ... implementation ...

def main():
    # Run both training approaches
    standard_result = run_training(False, dataset, epochs, batch_size)
    conditional_result = run_training(True, dataset, epochs, batch_size)
    
    # Analyze and compare results
    analyze_results(standard_result, conditional_result)
```

## 🔄 Data Flow Changes

### Before (Standard):
```
User-Item Interactions → Sparse Matrix → DataDiffusion → DNN → Diffusion Process
```

### After (Conditional):
```
User-Item Interactions → Sparse Matrix → User Group Classification → DataDiffusion → DNN + Conditionals → Diffusion Process
```

## 📊 Technical Specifications

- **Conditional Dimension**: 2 (binary: active/inactive users)
- **Conditional Format**: One-hot encoded tensors `[1, 0]` or `[0, 1]`
- **Integration Method**: Concatenation with main features and time embeddings
- **Parameter Increase**: +2,006 parameters (+0.04% overhead)
- **Training Overhead**: +1.3 seconds (+2.8% time increase)

## 🚀 Usage Examples

### Standard Training:
```bash
python3 main_conditional.py --dataset ml-1m_clean --epochs 20 --batch_size 400
```

### Conditional Training:
```bash
python3 main_conditional.py --dataset ml-1m_clean --epochs 20 --batch_size 400 --use_conditionals
```

### Automated Comparison:
```bash
python3 compare_conditionals.py
```

## 📈 Performance Analysis

### Current Results (ml-1m_clean dataset, 20 epochs):

| Metric | Standard Model | Binary Conditional | Performance Change |
|--------|----------------|-------------------|-------------------|
| **Training Time** | 47.6s | 48.9s | **+1.3s (+2.8%)** |
| **Model Parameters** | 5,633,920 | 5,635,926 | **+2,006 (+0.04%)** |
| **Precision@10** | 0.0498 | 0.0098 | **-80.3%** ⬇️ |
| **Recall@10** | 0.0721 | 0.0234 | **-67.5%** ⬇️ |
| **NDCG@10** | 0.0710 | 0.0141 | **-80.1%** ⬇️ |
| **MRR** | 0.1518 | 0.0398 | **-73.8%** ⬇️ |

## 🔍 Key Findings

✅ **Technical Implementation**: Successfully implemented and working  
✅ **Training Execution**: Both models train without errors  
❌ **Performance Impact**: Conditional model shows degraded performance  
💡 **Next Steps**: Investigate alternative conditional integration strategies  

## 🛠️ Implementation Details

### User Group Classification
- **Method**: Binary classification based on median interaction count
- **Threshold**: 42 interactions (dataset-specific)
- **Distribution**: 
  - Inactive Users (Group 0): 2,985 users (≤42 interactions)
  - Active Users (Group 1): 2,964 users (>42 interactions)

### Conditional Integration
- **Architecture**: Concatenation with main features and time embeddings
- **Processing**: Linear embedding layer for conditional features
- **Gradient Flow**: Full backpropagation through conditional pathway

### Training Process
- **Data Loading**: Conditional data generated during preprocessing
- **Batch Processing**: Handles both standard and conditional data formats
- **Loss Computation**: Conditional information integrated into diffusion process

## 🔮 Future Improvements

### Alternative Conditional Approaches
- **Gated conditioning**: Use gates to control conditional influence
- **Attention mechanism**: Add attention between conditional and main features
- **Separate pathways**: Process conditionals through separate network branches

### Hyperparameter Optimization
- **Learning rate adjustment**: Conditional model may need slower learning
- **Regularization**: Add dropout or weight decay to prevent overfitting
- **Batch normalization**: Stabilize training with conditional inputs

### Data Strategy
- **Feature engineering**: Include more meaningful user/item features
- **Balanced sampling**: Ensure equal representation of both user groups
- **Progressive conditioning**: Gradually introduce conditionals during training

## 📚 Dependencies

- **PyTorch**: Neural network framework
- **NumPy**: Numerical computations
- **SciPy**: Sparse matrix operations
- **TorchUtils**: Dataset and DataLoader utilities

## 🐛 Known Issues

1. **Performance Degradation**: Conditional model shows worse performance than standard model
2. **Overfitting**: Model may be overfitting to conditional information
3. **Architecture Limitations**: Current concatenation approach may not be optimal

## 📝 Notes

- The conditional framework is fully functional and ready for experimentation
- Performance issues suggest the need for alternative integration strategies
- Binary user grouping provides a stable foundation for further development
- All changes maintain backward compatibility with existing code

---

**Last Updated**: 2025-08-10  
**Version**: 1.0  
**Status**: Functional but requires performance optimization
