This repository is meant to guide those eager to learn about LLM inference and optimization. 
After getting my hands on a few incredibly insightful books on the topic, upon which most of the information here is based on, I was eager to learn how to implement the techniques mentioned in the books. 
In particular, 

1. AI Engineering - Chip Huyen Nguyen (2025)
2. AI Systems Performance Engineering: Optimizing Model Training and Inference Workloads with GPUs, CUDA, and PyTorch - Chris Fregley (2025)
3. Inference Engineering - Philip Kiely (2026)

These books omit in-depth implementation examples for a good reason - technology and frameworks surrounding LLM optimization and deployment change rapidly. Who knows? In 5 years, it is quite possible that the world has moved on from LLMs because the promised AGI (Artificial General Intelligence) has finally been brought into the world. 
The inspiration behind creating examples is the fact that I know almost nothing about this field but I became fascinated by the idea of making systems faster and cheaper.

For full transparency, as of this writing in May 2026, here is a list of things I know

1. I have a Masters in Data Science in UT Austin where I learned concepts of deep learning and training as well as transformers. A novel course called Advanced in Deep Learning was impactful and launched my interest in this field. This course covered GPU architecture, generative AI, quantization, LoRA fine-tuning, and I appreciated the professor's intent for the course content to come from recent papers in the field. 
2. I have basic Python scripting abilities and know the basics of Pytorch. 
3. I have vibe-coded a research agent using Gemini. This agent is capable of taking a user prompt in the form of a topic the user wants to learn more about (e.g. The Roman Empire) and make tool calls to the Wikipedia API to collect information and then write a brief research report.
4. I have a basic understanding of Docker 
5. I have a basic understanding of writing code in C++

And concerning this field, that is all I know. I am a beginner. If you find yourself at a similar position, then below are the concepts that we will learn together. 

1. Deploying on the cloud (I will use GCP)
2. Understanding CUDA
3. Inference frameworks as listed in the Table of Contents
4. Concepts of High performance computing

The fact that the points regarding what I don't know are so broad speaks to how much I don't know about these fields. Over time, we will explore how we can make this list more specific as we learn together. Note that there are many other related concepts that aren't covered in the Table of Contents (yet) such as agentic frameworks and orchestration, retrieval augmented generation, data engineering, MLOps, model training, etc. 

My goal isn't to cover every single topic but rather to provide a solid foundation for explaining how LLMs work and how we can optimize them for inference and deployment. Along the way, I hope to have fun and learn a lot. 

