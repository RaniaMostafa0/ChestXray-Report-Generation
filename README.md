ChestXray Report Generation:

- A multimodal deep learning model for chest X-rays that generates diagnostic reports, predicts disease labels, and learns image–report alignment. The architecture combines ResNet-50 as the image encoder and a BERT-based decoder for report generation, trained jointly with language modeling, image–report matching, and multi-label classification tasks.

📊 Dataset:
This project is based on the Curated CXR Report Generation Dataset (Kaggle) (https://www.kaggle.com/datasets/financekim/curated-cxr-report-generation-dataset), originally sourced from MIMIC-CXR and OpenI.

A processed version of the dataset was used, resulting in 59,030 chest X-ray images with paired radiology reports:

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

Multi-label Classification:

• Micro F1: 0.55

• Macro F1: 0.21

• Weighted F1: 0.44

• Test Loss: 2.07


Report Generation (BERTScore with SciBERT):

• Precision: 0.6380

• Recall: 0.6238

• F1: 0.6292
