# Reference Research Query for ChatGPT

## Project Summary

We are writing a CIL (Computational Intelligence Lab, ETH Zürich, SS2026) project report on **monocular depth estimation** using knowledge distillation from a large teacher model into a compact student model. The report follows the ICML 2025 paper template.

The central question: **Can a compact single-view depth model be improved through teacher supervision, without extra ground-truth annotations?**

---

## Models Used

- **Student**: DA3MONO-LARGE — the large single-view variant of **Depth Anything 3** (DepthAnything3, 2025), using a Vision Transformer backbone + DPT decoder.
- **Teacher (pseudo-labels)**: DA3NESTED-GIANT-LARGE-1.1 — a larger nested model from the same family.
- **Teacher (reference upper bound)**: DA3-GIANT-1.1

---

## Four Baselines

1. **Zero-shot evaluation** of the student on the test set (no fine-tuning).
2. **Fine-tuning on ground-truth depth labels**: freeze the ViT backbone, train only the DPT prediction head. Alternatively, apply LoRA (rank 4) to the DPT feature-extraction transformer blocks (layers 4, 11, 17, 23). Uses SiLog loss, AdamW, cosine LR schedule, 3 epochs.
3. **Pseudo-label distillation**: the teacher generates depth maps for all training images; the student is fine-tuned on these instead of the GT labels. Same training protocol as B2.
4. **Novel View Synthesis (NVS) augmentation** (in progress): use the teacher depth to initialise 3D Gaussian Splatting, render novel views with synthetic depth, train the student on the expanded dataset. Motivated as a geometric analogue of standard image augmentation.

---

## Papers We Need References For

Please find the canonical citation (title, authors, venue, year, arXiv or DOI) for each of the following. For each paper give: title, first author, venue/year, and a one-sentence description of what it contributes.

### Monocular Depth Estimation (foundations & recent)

1. **Eigen et al. 2014** — the first CNN-based monocular depth paper; introduced the scale-invariant log loss (SiLog). *(Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, NeurIPS 2014)*
2. **DPT** — Vision Transformers for dense prediction (depth + segmentation), introduced DPT architecture. *(Ranftl et al., ICCV 2021)*
3. **MiDaS** — mixed-dataset training for zero-shot depth generalisation. *(Ranftl et al., TPAMI 2022 or similar)*
4. **ZoeDepth** — metric-scale depth from relative depth models + domain-specific fine-tuning. *(Bhat et al., arXiv 2023)*
5. **Depth Anything v1** — scaling monocular depth with large unlabelled data. *(Yang et al., CVPR 2024)*
6. **Depth Anything v2** — improved version with synthetic data. *(Yang et al., NeurIPS 2024)*
7. **Depth Anything 3 / DepthAnything3** — most recent version, nested model for video + single-view. *(2025, likely arXiv)*
8. **UniDepth** — universal metric monocular depth estimation. *(Piccinelli et al., CVPR 2024)*
9. **Marigold** — diffusion-based monocular depth estimation. *(Ke et al., CVPR 2024)*
10. **Vision Transformer (ViT)** — the backbone architecture. *(Dosovitskiy et al., ICLR 2021)*

### Knowledge Distillation & Pseudo-Labels

11. **Hinton et al. 2015** — original knowledge distillation with soft targets. *(Distilling the Knowledge in a Neural Network, NeurIPS Workshop 2014 / arXiv)*
12. **FitNets** — intermediate feature distillation. *(Romero et al., ICLR 2015)*
13. **Pseudo-label** — semi-supervised learning via self-training with confident predictions. *(Lee, ICML Workshop 2013)*
14. **Noisy Student** — self-training with noise for image classification. *(Xie et al., CVPR 2020)*
15. **Dense prediction knowledge distillation** — distillation for segmentation or dense tasks (find a relevant paper, e.g. Structured KD, or SKD by Liu et al.)
16. **Relational Knowledge Distillation** — *(Park et al., CVPR 2019)* or similar structural KD paper

### LoRA / Parameter-Efficient Fine-Tuning

17. **LoRA** — Low-Rank Adaptation of large language/vision models. *(Hu et al., ICLR 2022)*

### Novel View Synthesis & Gaussian Splatting

18. **NeRF** — Neural Radiance Fields for novel view synthesis. *(Mildenhall et al., ECCV 2020)*
19. **3D Gaussian Splatting (3DGS)** — real-time NVS with 3D Gaussians. *(Kerbl et al., SIGGRAPH 2023)*
20. **Depth-initialised 3DGS** — using monocular depth to initialise Gaussian splatting (find relevant paper, e.g. DN-Splatter, MonoGS, or similar)*
21. **NVS as data augmentation** — using rendered novel views to augment training for recognition/depth tasks (find any relevant paper)*

---

## Output Format Requested

For each paper, please return a BibTeX entry formatted for use with `\bibliographystyle{icml2025}`. Use short keys like `eigen2014depth`, `ranftl2021dpt`, `yang2024depthanything`, etc.

Also flag if any paper could not be confidently found.
