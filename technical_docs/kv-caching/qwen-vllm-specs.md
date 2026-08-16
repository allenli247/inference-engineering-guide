Deployed on RTX 6000 Pro Blackwell with ~97GB VRAM with 1 replica
Model: Qwen3.6-35b-a3b-fp8
Inference Server: vLLM
max-model-len: 131072
gpu-memory-utilization: 0.60
max-num-seqs: 512
enable-prefix-caching: True
enable-auto-tool-choice: True
tool-call-parser: True

Workload Description: In our ETF Lab, we have several dozen hosts. Qwen monitors the health of each host, using telemetry data to take action if a host exhibits certain health behaviors such as low memory or over-utilized compute. A Qwen agent calls a series of defined tools to remediate the health of the host. As a side note, we also have MPS to run TEI on the same GPU, about 8GB VRAM

Problem Description: We're interested in deploying the latest Qwen release - Qwen 3.8-27B on our GPU. Right now though, it looks like we're severely underutilizing our GPU due to the async and infrequent nature of the LLM agent being called. Yet at the same time, we're looking for how we can optimize for what we currently have. Should we increase gpu-memory-utilization of Qwen and vLLM and increase workload on Qwen or should we decrease gpu-memory-utilization and make room for yet another model, say, a draft model? We need recommendations on inference tuning. Address the underutilization with brief recommendations but we're more interested in inference optimization irrespective of the workloads we're currently running

Article description
1. Give an overview of the paremters for Qwen3.6-35b with fp8 and Qwen3.8-27b (both fp8 and fp4) including the architecture. Explain MoE and quantization (going down to the depth explaining stuff like mantissas and exponents) 
2. Overview of the flow of inference during LLMs, both with a single request and batched requests including the role of the KV cache and prefix caching. Explain token IDs and how they reference the embedding table and what keys, values, and queries do and how they are created for each token. 
3. Explain the KV cache and its purpose, why KV can be cached and how KV caching works in the context of attention. Explain how vLLM implements pagedattention and what problem pagedattention solves. Explain how prefix caching works as well and the benefits it offers.
4. Give recommendations for tuning our current vLLM parameters, model deployment (Qwen3.6 vs 3.8) and improving our inference service overall. 