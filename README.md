## CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays (NeurIPS 2025 D&B Track - Spotlight)

<p align="center">
  <a href="https://arxiv.org/pdf/2505.18087" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/arXiv-2505.18087-b31b1b.svg" alt="arXiv">
  </a>
   <a href="https://physionet.org/content/chexstruct-cxreasonbench" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/dataset-Physionet-green.svg" alt="dataset">  
  </a>
   <a href="https://huggingface.co/datasets/ttumyche/CheXStruct" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/dataset-HuggingFace-yellow.svg" alt="dataset">
  </a>
</p>

## Overview

Recent progress in Large Vision-Language Models (LVLMs) has enabled promising applications in medical tasks, such as report generation and visual question answering. However, existing benchmarks mainly focus on the final diagnostic answer, providing limited insight into whether models engage in clinically meaningful reasoning.

To address this, we present **CheXStruct** and **CXReasonBench**:

- **CheXStruct**: A fully automated pipeline extracting structured clinical information directly from chest X-rays. It performs anatomical segmentation, derives anatomical landmarks and diagnostic measurements, computes diagnostic indices, and applies clinical thresholds based on expert guidelines.

- **CXReasonBench**: A multi-path, multi-stage evaluation framework that assesses a model’s ability to perform structured diagnostic reasoning. The benchmark includes 18,988 QA pairs across 12 diagnostic tasks and 1,200 cases, each with up to 4 visual inputs, enabling detailed evaluation of reasoning steps including visual grounding and diagnostic measurements.

Even the strongest LVLMs evaluated struggle with structured reasoning and generalization, showing the importance of this benchmark.

## Citation

If you use this code or dataset in your research, please cite our work:
```
@inproceedings{leecxreasonbench,
  title={CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays},
  author={Lee, Hyungyung and Choi, Geon and Lee, Jung-Oh and Yoon, Hangyul and Hong, Hyuk Gi and Choi, Edward},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}
```

## Contact

If you have any questions, feedback, or issues regarding this project, please reach out to us via email: **ttumyche@kaist.ac.kr**
