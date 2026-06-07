# 🍔 AI Food Fraud Detection
🚀 **Live Demo:** [Try it here](https://huggingface.co/spaces/Gargi002/ai-food-fraud-detection)
Detects AI-generated or manipulated food images used in fraudulent refund claims on food delivery platforms like Uber Eats and DoorDash.

## 🎯 What This Project Does
Refund fraud on food delivery platforms often involves users uploading edited or AI-generated food photos to claim missing/incorrect orders.
This system automatically analyzes an uploaded food image and classifies it as:
- ✅ AUTHENTIC — genuine food photo
- 🚨 AI-GENERATED / FRAUD — fake or manipulated image
- ⚠️ SUSPICIOUS — uncertain, requires manual review

### Example workflow
Customer uploads food image
↓
AI detection pipeline runs
↓
Fraud score + forensic report generated
↓
Approve / Flag / Manual Review

## 💼 Why It Matters
Food delivery platforms lose revenue from fraudulent refund claims.
This project helps by:
- reducing manual review workload
- detecting synthetic/manipulated food evidence
- lowering refund fraud losses
- generating explainable forensic reports for reviewers

## 📊 Results
| Metric | Score |
|--------|-------|
| Accuracy | 86% |
| ROC-AUC | 0.94 |
| F1-Score (macro) | 0.86 |
| Equal Error Rate | Low |

## 🧠 Detection Pipeline
Input Image
│
├── [Layer 1] Vision Transformer (ViT)
│     Fine-tuned google/vit-base-patch16-224-in21k
│     → Fraud probability
│
├── [Layer 2] Advanced Error Level Analysis
│     Multi-quality JPEG Ghost + Spatial Blob Detection
│     → Suspicion score + manipulation regions
│
└── [Layer 3] 8-Feature Forensic Analyzer
ELA · Noise · Texture-LBP · FFT
Illumination · Shadow · Bokeh · Color
→ Fused forensic score
Final Score = (ViT × 55%) + (ELA × 25%) + (Forensic × 20%)

## 🛠️ Tech Stack
| Technology | Purpose |
|-----------|---------|
| Python | Core language |
| PyTorch | Deep learning |
| HuggingFace Transformers | Vision Transformer |
| OpenCV | Image processing |
| scikit-image | Texture / blob analysis |
| NumPy / SciPy | Feature computation |
| Matplotlib / Seaborn | Visualization |

## 📁 Dataset
CIFAKE — Real vs AI-Generated Images
- Source: Kaggle (birdy654/cifake)
- 10,000 training images
- 2,000 test images
- Labels: REAL / FAKE

> **Note:** CIFAKE contains general real-vs-synthetic imagery. This project adapts those forensic detection techniques specifically for food image fraud detection. CIFAKE's forensic patterns (ELA anomalies, frequency artifacts) transfer directly to food image fraud detection.

## 🚀 Running the Project

### 1. Open notebook in Google Colab
Upload: `AI_Food_Fraud_FINAL_COMBINED.ipynb`

### 2. Enable GPU
Runtime → Change runtime type → T4 GPU

### 3. Run cells sequentially
| Cell | Purpose |
|------|---------|
| 1 | Install dependencies |
| 2 | Download dataset |
| 3 | Load ELA engine |
| 4 | Load forensic analyzer |
| 5–6 | Configure model + dataset |
| 7–8 | Train ViT |
| 9–10 | Evaluate + plot |
| 11 | Save model |
| 12 | Upload custom image |

### 4. Add Kaggle credentials
```python
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_api_key"
```

## 🖼️ Per-Image Output
Each uploaded image generates:
- Original image with highlighted suspicious region
- ELA map (quality 85)
- ELA map + JPEG Ghost detection
- Spatial blob heatmap
- 8-feature forensic score chart
- ViT fraud probability
- Console summary
FINAL VERDICT : 🚨 AI-GENERATED / FRAUD
RISK LEVEL    : 🔴 HIGH RISK
Final Score   : 78.3 / 100

## 📐 Evaluation Metrics
Uses ISO/IEC 30107-3 anti-spoofing metrics:
- APCER — fraud missed as real
- BPCER — real flagged as fraud
- ACER — average error rate
- EER — equal error rate

## ⚙️ Tuning Tips
| Goal | Setting |
|------|---------|
| Low RAM | max_train_samples = 5000 |
| Better accuracy | num_epochs = 10 |
| Catch more fraud | Lower threshold → 0.3 |
| Skip retraining | Load saved model + run Cell 12 |

## 🔮 Future Improvements
- Train on real-world food delivery refund images
- Add metadata / EXIF validation
- Deploy as API for refund review pipelines
- Dashboard for fraud analysts
- Real-time batch image screening
