ChestXray-Report-Generation:

-A multimodal deep learning model for chest X-rays that generates diagnostic reports, predicts disease labels, and learns image–report alignment. The architecture combines ResNet-50 as the image encoder and a BERT-based decoder for report generation, trained jointly with language modeling, image–report matching, and multi-label classification tasks.

📊 Dataset:
After cleaning, the dataset contains 59,030 chest X-ray images with paired radiology reports.
• Train: 47,222
• Validation: 5,912
• Test: 5,896

⚙️ Model Architecture:
- Encoder: ResNet-50 → extracts visual features
- Decoder: BERT → generates textual reports
Prediction heads:
• Masked Language Modeling (MLM)
• Image–Report Matching (IRM)
• Multi-label classification (14 disease labels)
• Loss function: Combined (MLM + IRM + BCE)

🏋️ Training:
- Optimizer: Adam
- Epochs: 20
- Batch size: 4
- Metrics: Training/Validation Loss, F1 Score
- Best Model: Saved at epoch 19 with F1 = 0.5502

📈 Results:
Training Metrics:
• Train Loss: 1.7191
• Validation Loss: 2.0652
• Best F1 Score: 0.5502

Classification Report (Test Set):
• Micro F1: 0.55
• Macro F1: 0.21
• Weighted F1: 0.44
• Test Loss: 2.0713

Report Generation (BERTScore with SciBERT):
• Precision: 0.6380
• Recall: 0.6238
• F1: 0.6292
Notable cases:

Pleural Effusion → FP = 368, TP = 556

Pneumothorax → FP = 329, TP = 554

Atelectasis → FP = 117, TP = 134
