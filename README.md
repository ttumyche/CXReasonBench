<h1 align="center">CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays</h1>

<p align="center">
  <a href="[https://arxiv.org/abs/2403.12345](https://arxiv.org/pdf/2505.18087)">
    <img src="https://img.shields.io/badge/arXiv-2505.18087-b31b1b.svg" alt="arXiv">
  </a>
   <a href="[https://huggingface.co/datasets/ttumyche/CheXStruct](https://huggingface.co/datasets/ttumyche/CheXStruct)">
    <img src="https://img.shields.io/badge/dataset-HuggingFace-yellow.svg" alt="dataset">
  </a>
</p>


## 🖼️ Overview

Recent progress in Large Vision-Language Models (LVLMs) has enabled promising applications in medical tasks, such as report generation and visual question answering. However, existing benchmarks mainly focus on the final diagnostic answer, providing limited insight into whether models engage in clinically meaningful reasoning.

To address this, we present **CheXStruct** and **CXReasonBench**:

- **CheXStruct**: A fully automated pipeline extracting structured clinical information directly from chest X-rays. It performs anatomical segmentation, derives anatomical landmarks and diagnostic measurements, computes diagnostic indices, and applies clinical thresholds based on expert guidelines (see Figure 1).

- **CXReasonBench**: A multi-path, multi-stage evaluation framework that assesses a model’s ability to perform structured diagnostic reasoning. The benchmark includes 18,988 QA pairs across 12 diagnostic tasks and 1,200 cases, each with up to 4 visual inputs, enabling detailed evaluation of reasoning steps including visual grounding and diagnostic measurements (see Figure 2).

Even the strongest LVLMs evaluated struggle with structured reasoning and generalization, showing the importance of this benchmark.

<p align="center">
  <img src="images/overview_chexstruct.png" alt="CheXStruct Pipeline" width="70%">
</p>
<p align="center"><em>Figure 1. CheXStruct: Automated pipeline for <br> extracting structured clinical information from chest X-rays.</em></p>

<p align="center">
  <img src="images/overview_cxreasonbench.png" alt="CXReasonBench Evaluation Pipeline" width="70%">
</p>
<p align="center"><em>Figure 2. CXReasonBench: Multi-path, multi-stage evaluation framework <br> for assessing structured diagnostic reasoning in LVLMs.</em></p>

