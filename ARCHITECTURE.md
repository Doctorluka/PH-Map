# PH-Map Multi-Task Learning Architecture

## 完整架构示意图

本文档详细描述了 PH-Map 多任务学习框架的完整架构，包括数据流、网络结构、损失计算和优化过程。

```
═══════════════════════════════════════════════════════════════════════════════
                          PH-MAP MTL TRAINING ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA                                         │
│                                                                              │
│  AnnData Object                                                              │
│  ┌────────────────────────────────────────────────────────────┐            │
│  │  Gene Expression Matrix (n_cells × n_genes)                │            │
│  │  X: scipy.sparse.csr_matrix or np.ndarray                  │            │
│  └────────────────────────────────────────────────────────────┘            │
│                                                                              │
│  Labels (adata.obs)                                                          │
│  ┌────────────────────────────────────────────────────────────┐            │
│  │  anno_lv1: [6 classes]   → LabelEncoder → y[0]            │            │
│  │  anno_lv2: [19 classes]  → LabelEncoder → y[1]            │            │
│  │  anno_lv3: [30 classes]  → LabelEncoder → y[2]            │            │
│  │  anno_lv4: [55 classes]  → LabelEncoder → y[3]            │            │
│  └────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA PREPROCESSING                                    │
│                                                                              │
│  1. Binary Conversion: X_binary = (X > 0).astype(np.float32)               │
│     └─> Convert gene expression to binary (expressed = 1, not = 0)         │
│                                                                              │
│  2. Label Encoding:                                                          │
│     └─> LabelEncoder.fit_transform() for each task                         │
│                                                                              │
│  3. Train/Val Split (80/20, stratified by anno_lv1)                        │
│                                                                              │
│  4. DataLoader: Batch size = 128, shuffle = True                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARD PARAMETER SHARING LAYERS                             │
│                    (Shared Feature Extraction)                               │
│                                                                              │
│  Input:  X_binary [batch_size, n_genes]                                     │
│          ↓                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  Shared Layer 1: Linear(n_genes → 200)                       │          │
│  │  └─> ReLU Activation                                         │          │
│  │  └─> Dropout(p=0.4)                                          │          │
│  └──────────────────────────────────────────────────────────────┘          │
│          ↓                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  Shared Layer 2: Linear(200 → 100)                           │          │
│  │  └─> ReLU Activation                                         │          │
│  │  └─> Dropout(p=0.4)                                          │          │
│  └──────────────────────────────────────────────────────────────┘          │
│          ↓                                                                    │
│  Shared Features: [batch_size, 100]  ← ALL TASKS SHARE THESE PARAMETERS    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TASK-SPECIFIC OUTPUT LAYERS                               │
│                    (Task-Specific Classification Heads)                      │
│                                                                              │
│  Shared Features [batch_size, 100]                                          │
│          ↓                                                                    │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐            │
│  │ Task 1       │ Task 2       │ Task 3       │ Task 4       │            │
│  │ anno_lv1     │ anno_lv2     │ anno_lv3     │ anno_lv4     │            │
│  │              │              │              │              │            │
│  │ Linear       │ Linear       │ Linear       │ Linear       │            │
│  │ (100→6)      │ (100→19)     │ (100→30)     │ (100→55)     │            │
│  │              │              │              │              │            │
│  │ Output 1     │ Output 2     │ Output 3     │ Output 4     │            │
│  │ [B, 6]       │ [B, 19]      │ [B, 30]      │ [B, 55]      │            │
│  └──────────────┴──────────────┴──────────────┴──────────────┘            │
│                                                                              │
│  Outputs: [output_1, output_2, output_3, output_4]                         │
│           Each output: [batch_size, num_classes]                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOSS COMPUTATION                                     │
│                                                                              │
│  For each task:                                                              │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  Loss_i = CrossEntropyLoss(output_i, label_i)                │          │
│  │                                                               │          │
│  │  L1 = CrossEntropyLoss(output_1, y[0])  # anno_lv1          │          │
│  │  L2 = CrossEntropyLoss(output_2, y[1])  # anno_lv2          │          │
│  │  L3 = CrossEntropyLoss(output_3, y[2])  # anno_lv3          │          │
│  │  L4 = CrossEntropyLoss(output_4, y[3])  # anno_lv4          │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  Task Weights (default):                                                     │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  task_weights = [0.3, 0.8, 1.5, 2.0]                        │          │
│  │  └─> w1 = 0.3  (anno_lv1)                                    │          │
│  │  └─> w2 = 0.8  (anno_lv2)                                    │          │
│  │  └─> w3 = 1.5  (anno_lv3)                                    │          │
│  │  └─> w4 = 2.0  (anno_lv4)                                    │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  Weighted Loss:                                                              │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  L_total = w1*L1 + w2*L2 + w3*L3 + w4*L4                    │          │
│  │                                                               │          │
│  │  L_total = 0.3*L1 + 0.8*L2 + 1.5*L3 + 2.0*L4                │          │
│  └──────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION (ADAM OPTIMIZER)                           │
│                                                                              │
│  Optimizer: Adam                                                             │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  optimizer = optim.Adam(                                     │          │
│  │      model.parameters(),  # All parameters (shared + task-specific)│    │
│  │      lr = 0.001                                              │          │
│  │  )                                                           │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  Training Step (per batch):                                                  │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  1. optimizer.zero_grad()     # Clear gradients              │          │
│  │  2. loss.backward()           # Backpropagation              │          │
│  │     └─> Compute gradients for:                               │          │
│  │         • Shared layers (shared_fc)                          │          │
│  │         • Task-specific layers (output_layers)               │          │
│  │  3. optimizer.step()          # Update parameters            │          │
│  │     └─> Update all parameters using Adam algorithm           │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  Note: Gradients from all tasks flow back through shared layers,            │
│        enabling shared feature learning across tasks.                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                        │
│                                                                              │
│  For epoch in range(num_epochs=100):                                         │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  For batch in train_loader:                                  │          │
│  │    • Forward pass → outputs                                  │          │
│  │    • Compute weighted loss                                   │          │
│  │    • Backward pass → gradients                               │          │
│  │    • Optimizer step → update parameters                      │          │
│  │                                                               │          │
│  │  Validation:                                                  │          │
│  │    • Evaluate on validation set                              │          │
│  │    • Calculate accuracy for each task                        │          │
│  │                                                               │          │
│  │  Early Stopping:                                              │          │
│  │    • Monitor validation accuracy                             │          │
│  │    • Patience = 10 epochs                                    │          │
│  └──────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION OUTPUT                                    │
│                                                                              │
│  During Inference:                                                           │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  Input: Query AnnData (gene expression)                      │          │
│  │    ↓                                                          │          │
│  │  Binary conversion → Shared layers → Task-specific layers    │          │
│  │    ↓                                                          │          │
│  │  Outputs: [logits_1, logits_2, logits_3, logits_4]          │          │
│  │    ↓                                                          │          │
│  │  Softmax: [prob_1, prob_2, prob_3, prob_4]                  │          │
│  │    ↓                                                          │          │
│  │  Predictions:                                                 │          │
│  │    • predicted_anno_lv1 = argmax(prob_1)                     │          │
│  │    • predicted_anno_lv2 = argmax(prob_2)                     │          │
│  │    • predicted_anno_lv3 = argmax(prob_3)                     │          │
│  │    • predicted_anno_lv4 = argmax(prob_4)                     │          │
│  │                                                               │          │
│  │  Probabilities:                                               │          │
│  │    • predicted_anno_lv1_prob = max(prob_1)                   │          │
│  │    • predicted_anno_lv2_prob = max(prob_2)                   │          │
│  │    • predicted_anno_lv3_prob = max(prob_3)                   │          │
│  │    • predicted_anno_lv4_prob = max(prob_4)                   │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                              │
│  Final Output:                                                               │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │  DataFrame with columns:                                      │          │
│  │    • predicted_anno_lv1, predicted_anno_lv1_prob             │          │
│  │    • predicted_anno_lv2, predicted_anno_lv2_prob             │          │
│  │    • predicted_anno_lv3, predicted_anno_lv3_prob             │          │
│  │    • predicted_anno_lv4, predicted_anno_lv4_prob             │          │
│  └──────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              KEY FEATURES
═══════════════════════════════════════════════════════════════════════════════

1. HARD PARAMETER SHARING:
   • All tasks share the same feature extraction layers (shared_fc)
   • Task-specific layers are independent (output_layers)
   • Efficient parameter usage: shared parameters learned once, used by all tasks

2. MULTI-TASK LEARNING:
   • Simultaneous learning of hierarchical cell type labels
   • Shared features capture common patterns across all classification levels
   • Task-specific heads learn level-specific distinctions

3. WEIGHTED LOSS:
   • Different weights for different tasks (higher weights for finer-grained tasks)
   • Balances learning across hierarchical levels
   • Default: [0.3, 0.8, 1.5, 2.0] for [lv1, lv2, lv3, lv4]

4. ADAM OPTIMIZER:
   • Adaptive learning rate for each parameter
   • Efficient gradient-based optimization
   • Learning rate: 0.001 (default)

5. REGULARIZATION:
   • Dropout (p=0.4) in shared layers to prevent overfitting
   • Early stopping based on validation accuracy

═══════════════════════════════════════════════════════════════════════════════
                              PARAMETER COUNT
═══════════════════════════════════════════════════════════════════════════════

Example: n_genes = 2000, hidden_sizes = [200, 100]

Shared Layers:
  • Linear(2000 → 200): 2000*200 + 200 = 400,200 parameters
  • Linear(200 → 100):  200*100 + 100 = 20,100 parameters
  • Total Shared: 420,300 parameters

Task-Specific Layers:
  • Task 1 (6 classes):  100*6 + 6 = 606 parameters
  • Task 2 (19 classes): 100*19 + 19 = 1,919 parameters
  • Task 3 (30 classes): 100*30 + 30 = 3,030 parameters
  • Task 4 (55 classes): 100*55 + 55 = 5,555 parameters
  • Total Task-Specific: 11,110 parameters

Total Model Parameters: 431,410 parameters

═══════════════════════════════════════════════════════════════════════════════

