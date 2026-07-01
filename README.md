# Chest-Xray-Report-Generation

Multimodal deep learning model that generates free-text diagnostic reports from chest X-rays, jointly trained to predict 10 disease/finding labels and learn image-report alignment. Combines a **ResNet-50** image encoder with a **BERT-based** decoder, trained with a language modeling, image-report matching, and multi-label classification loss.

## Overview

Report generation was framed as a single multi-task model rather than separate classification and generation pipelines, so the image encoder learns representations useful for both diagnosis and narrative description. The image is compressed to a single embedding token that the BERT decoder cross-attends to at every generation step. An auxiliary image-report matching (IRM) task — distinguishing a genuine image-report pair from a randomly mismatched one — was added to encourage the shared representation to stay grounded in the actual image content rather than defaulting to generic radiology phrasing.

## Architecture

- **Image encoder:** ResNet-50 (ImageNet-pretrained), final FC layer removed, 2048-dim features projected to a 768-dim embedding via a linear layer.
- **Decoder:** `BertModel` configured as a decoder (`is_decoder=True, add_cross_attention=True`), cross-attending to the single image embedding token. Tokenizer: `bert-base-uncased`.
- **Heads:** `lm_head` (vocab-size, next-token prediction), `irm_head` (1, binary match prediction), `label_head` (10, multi-label disease classification) — all sharing the same encoder/decoder backbone.
- **Loss:** `CrossEntropyLoss` (language modeling, pad-masked) + `BCEWithLogitsLoss` (IRM) + `BCEWithLogitsLoss` (multi-label classification), summed.

Transfer learning was used by initializing the ResNet-50 backbone with ImageNet-pretrained weights; the decoder was trained from scratch with a BERT-style architecture.

## Dataset

- **Source:** [MIMIC-CXR Dataset (Kaggle)](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset) — ~91,685 chest X-ray image–report pairs originally.
- **Cleaning impact:** ~35% of the original data was dropped — 31,978 rows had placeholder text (`___`) instead of a real report, and a further 677 rows became empty after boilerplate comparison phrases (e.g. *"Compared to prior..."*, *"No significant change as compared..."*) were stripped via regex. Final clean dataset: 59,030 pairs.
- **Preprocessing:** images resized to 512×512; reports tokenized with `bert-base-uncased`, max length 256.
- **Labels:** 10 disease/finding classes extracted via keyword/regex matching directly on report text (**not** clinician-annotated ground truth): `cardiomegaly, edema, consolidation, atelectasis, pleural_effusion, pneumonia, pneumothorax, lung_opacity, fracture, support_devices`.
- **Split:** 59,030 image-report pairs total — 47,222 train / 5,912 val / 5,896 test.

## Results

| Metric | Value |
|---|---|
| Best validation micro-F1 | 0.5502 (epoch 19) |
| Test micro-F1 | 0.55 |
| Test macro-F1 | 0.21 |
| Test loss (summed) | 2.0713 |
| Training epochs | 20 |
| Best checkpoint | Epoch 19 |
| Hardware | NVIDIA RTX 2080 |
| Optimizer | Adam, lr 1e-4, batch size 4 |

**BERTScore** (`allenai/scibert_scivocab_uncased`, test set, n=5,896)

| Metric | Value |
|---|---|
| Precision | 0.6380 |
| Recall | 0.6238 |
| F1 | 0.6292 |

### Per-Label Results (test set)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| pleural_effusion | 0.60 | 0.94 | 0.73 | 3488 |
| pneumothorax | 0.61 | 0.91 | 0.73 | 3506 |
| atelectasis | 0.55 | 0.37 | 0.45 | 2149 |
| edema | 0.65 | 0.10 | 0.17 | 1830 |
| cardiomegaly | 0.59 | 0.03 | 0.05 | 894 |
| consolidation | 0.00 | 0.00 | 0.00 | 1333 |
| pneumonia | 0.00 | 0.00 | 0.00 | 894 |
| fracture | 0.00 | 0.00 | 0.00 | 280 |
| support_devices | 0.00 | 0.00 | 0.00 | 199 |
| lung_opacity | 0.00 | 0.00 | 0.00 | 31 |

Large, high-prevalence classes (pleural effusion, pneumothorax) classify well; low-prevalence or visually subtle classes (fracture, support devices, lung opacity, pneumonia, consolidation) collapse to zero — driven by both class imbalance and weak (regex-derived) label quality rather than model capacity alone.

## Qualitative Testing

Generated reports were spot-checked against reference reports on the validation and test sets (see notebook cell outputs).

**Strengths:** fluent radiology-style prose; frequently correct on major findings (edema, effusion, atelectasis) and device/line positioning language.

**Limitations:**
- Hallucinates specific findings or devices (e.g. endotracheal tube position, PICC line) not present in the ground-truth report.
- Comparison-to-prior phrasing is inconsistent, plausibly a residual effect of the boilerplate-stripping step during cleaning.
- No radiologist review was performed — all metrics are automatic (F1, BERTScore) only.

## Project Structure
├── 01_Data_Cleaning.ipynb     # image/CSV consistency checks, placeholder-row removal, boilerplate stripping, label extraction
├── 02_Model_Training.ipynb    # dataset class, model definition, training loop, evaluation, BERTScore scoring
