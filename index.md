# Inference Engineering Guide

Welcome to the **Inference Engineering Guide**. This repository contains technical documentation, hardware optimization guides, and LLM deployment blueprints.

## Technical Guides & Benchmarks

* 🚀 **[High-Throughput & Low-Latency LLM Agent Deployment: Tuning Qwen on vLLM & Blackwell](technical_docs/kv-caching/qwen-vllm-guide.md)**
  * An in-depth engineering guide to optimizing open-weight LLM inference for asynchronous agent workloads using vLLM on NVIDIA RTX 6000 Blackwell. Covers Qwen 3.6-35B MoE vs Qwen 3.8-27B Dense, FP8/FP4 quantization mechanics, PagedAttention, and Automatic Prefix Caching for tool calling.

---

## Architecture & Hardware Chapters

* **Chapter 0:** [Preface](chapter_0_preface/)
* **Chapter 1:** [Transformers Implementation](chapter_1_transformers/)
* **Chapter 2:** [Hardware Optimization](chapter_2_hardware_optimization/)
* **Chapter 3:** [Dense LLM Architecture](chapter_3_dense_llm_architecture/)
