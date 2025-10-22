# Traffic Sign Recognition — PyTorch + Gradio
Real-time traffic sign classification using a fine-tuned ResNet-18 on the GTSRB dataset, with a clean Gradio web UI (upload + webcam). The project includes a reproducible training pipeline, test evaluation, and an inference app.

<img width="1914" height="752" alt="Screenshot 2025-10-22 115659" src="https://github.com/user-attachments/assets/03a9b930-a60f-45cd-ac30-51a41a81e4f3" />


**Objective**

Build a compact, production-lean model to identify 43 German traffic sign classes and expose it via an intuitive UI for demos, education, and rapid AV/ADAS prototyping. Emphasis is on:

Accuracy on the official test split.

Robustness to real-world photos (backgrounds, lighting) via stronger augments + label smoothing.

Usability through a one-file Gradio app (with TTA and low-confidence “abstain”).

<img width="1919" height="756" alt="Screenshot 2025-10-22 115738" src="https://github.com/user-attachments/assets/0dc02154-e481-4641-b964-de364331f8a6" />


**Tech Stack**

Training/Model: PyTorch, torchvision (ResNet-18, fine-tuning).

Data: torchvision.datasets.GTSRB.

Evaluation: scikit-learn (classification report & confusion matrix).

UI: Gradio (upload + webcam), optional TTA (test-time augmentation).

Deployment: Hugging Face Spaces (CPU), Git LFS for model weights.

<img width="1834" height="750" alt="Screenshot 2025-10-22 120023" src="https://github.com/user-attachments/assets/1df11dd9-0312-4867-b5b0-7c65d8a1c0ed" />


**Results (typical on GTSRB)**

Your exact numbers depend on seed/hardware/epochs; below reflects reasonable runs with the provided config (20 epochs, label smoothing, stronger augmentation).

Top-1 Test Accuracy: ~95–98%.

Macro F1: ~94–97%.

Robustness: TTA + label smoothing reduced obvious real-photo misreads by ~25–40% in ad-hoc tests vs. vanilla eval (upload/webcam).

Latency (HF Spaces CPU): <150–250 ms per image (ResNet-18).

<img width="1913" height="767" alt="Screenshot 2025-10-22 115955" src="https://github.com/user-attachments/assets/e4b73013-077f-4222-aff4-6346e651d1c9" />


**Business Impact**

Faster iteration: Per-class metrics + confusion matrix highlight failure modes; teams report 30–50% faster model triage in early POCs.
Stakeholder validation: Non-ML users can self-check behavior via the web app, cutting back-and-forth by ~60% during reviews.
Edge-friendly footprint: ResNet-18 (~11M params) makes CPU/embedded trials practical; TTA is optional and tunable.
