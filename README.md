# 💡 LLM & Generative AI Study Plan (10 Hrs/Week - 6 Months)

Welcome! This repo is your roadmap to master Large Language Models (LLMs) and Generative AI. It’s built for developers (like MERN stack devs) who can dedicate ~10 hours/week.

---

## 📅 Study Structure
- **Duration**: 6 Months
- **Time Commitment**: 10 hours/week
- **Phases**:
  - 📘 Phase 1: Foundations of AI & Deep Learning (Month 1–2)
  - 🤖 Phase 2: LLMs, Prompting & Tools (Month 3–4)
  - 🚀 Phase 3: Fine-tuning, Deployment, Final Project (Month 5–6)

---

## ✅ Phase 1: Core AI & Deep Learning (Month 1–2)

### Week 1: Python for AI
- [ ] Learn Python, Numpy, Pandas: https://jovian.com/learn/data-analysis-with-python-zero-to-pandas
- [ ] Project: Data Analysis on a dataset
- 🗂 Starter: `phase-1-foundations/week-01-python/data_analysis_starter.ipynb`

### Week 2: Math Basics
- [ ] Linear Algebra: https://www.youtube.com/watch?v=fNk_zzaMoSs
- [ ] Project: Build matrix ops in NumPy
- 🗂 Starter: `phase-1-foundations/week-02-math/matrix_numpy_ops.py`

### Week 3–4: Neural Networks
- [ ] Deep Learning Specialization - Course 1: https://www.coursera.org/learn/neural-networks-deep-learning
- [ ] Project: MNIST Digit Classifier
- 🗂 Starter: `phase-1-foundations/week-03-neural-nets/mnist_classifier.py`

### Week 5: CNNs, RNNs, Intro to Transformers
- [ ] Continue Courses 2 & 4 from above
- [ ] Project: Image classifier or RNN text gen

### Week 6–8: Transformers Deep Dive
- [ ] Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- [ ] Karpathy’s Zero to Hero: https://www.youtube.com/playlist?list=PLpVm0N1j1PzCkzZkMfV8AZt0X_qzZ72sM
- [ ] Project: Build char-level GPT
- 🗂 Starter: `phase-1-foundations/week-06-transformers/char_gpt_minimal.py`

---

## ✅ Phase 2: LLMs, Hugging Face, LangChain (Month 3–4)

### Week 9: Tokenization & NLP Intro
- [ ] Hugging Face Course: Chapter 1–2: https://huggingface.co/learn/nlp-course
- 🗂 Starter: `phase-2-llm-tools/huggingface-tasks/tokenization_demo.ipynb`

### Week 10–11: Transformers in Practice
- [ ] Hugging Face Chapters 3–4
- [ ] Project: Text classification with DistilBERT
- 🗂 Starter: `phase-2-llm-tools/huggingface-tasks/distilbert_text_classifier.py`

### Week 12: Prompt Engineering
- [ ] DeepLearning.ai Prompt Course: https://www.deeplearning.ai/short-courses/prompt-engineering/
- [ ] Project: Prompt-based rewriting tool

### Week 13–14: RAG (Retrieval Augmented Generation)
- [ ] LangChain Docs: https://docs.langchain.com/docs/
- [ ] Project: Document Q&A Bot
- 🗂 Starter: `phase-2-llm-tools/langchain-projects/pdf_qa_bot.py`

### Week 15–16: Vector Stores & Embeddings
- [ ] Learn FAISS / Pinecone / Weaviate
- [ ] Project: Semantic Search App
- 🗂 Starter: `phase-2-llm-tools/langchain-projects/semantic_search_app.py`

---

## ✅ Phase 3: Fine-Tuning & Projects (Month 5–6)

### Week 17: Fine-Tuning Basics
- [ ] Hugging Face Fine-tuning Guide: https://huggingface.co/blog/fine-tune-transformers
- [ ] Project: Fine-tune BERT on sentiment
- 🗂 Starter: `phase-3-projects/fine-tuning/bert_finetune_sentiment.py`

### Week 18–19: LoRA, PEFT
- [ ] PEFT Repo: https://github.com/huggingface/peft
- [ ] Project: LoRA fine-tuning on a custom dataset
- 🗂 Starter: `phase-3-projects/fine-tuning/lora_finetune_custom.py`

### Week 20: Model Hosting
- [ ] Learn Gradio: https://gradio.app/
- [ ] Hugging Face Spaces: https://huggingface.co/spaces
- [ ] Project: Deploy a Text Generator
- 🗂 Starter: `phase-3-projects/deployment/gradio_textgen_app.py`

### Week 21: LangChain + OpenAI API
- [ ] LangChain Docs
- [ ] Project: AI Assistant or Chatbot
- 🗂 Starter: `phase-3-projects/langchain-assistant/openai_chatbot.py`

### Week 22–24: Final Project
- [ ] Build & polish your final app

---

## 💡 Final Project Ideas
- AI Resume/Cover Letter Writer
- PDF Chatbot (LangChain + OpenAI)
- Image Caption Generator
- AI Recipe Generator

---

## 🧰 Tools Used
- Python, PyTorch, Hugging Face, Gradio
- LangChain, FAISS/Pinecone/Weaviate
- OpenAI API / LLaMA / Mistral
- FastAPI for backend deployment

---

## 🗂 Suggested Repo Folder Structure
```
llm-genai-study-plan/
│
├── phase-1-foundations/
│   ├── week-01-python/
│   ├── week-02-math/
│   ├── week-03-neural-nets/
│   ├── week-06-transformers/
│   └── ...
│
├── phase-2-llm-tools/
│   ├── huggingface-tasks/
│   ├── langchain-projects/
│   └── ...
│
├── phase-3-projects/
│   ├── fine-tuning/
│   ├── deployment/
│   ├── langchain-assistant/
│   └── final-project/
│
├── notebooks/
│   ├── huggingface-tutorials.ipynb
│   ├── peft-finetuning.ipynb
│   └── ...
│
├── datasets/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💻 Starter Code Resources
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Hugging Face Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples)
- [LangChain Examples](https://github.com/langchain-ai/langchain)
- [Awesome GenAI GitHub List](https://github.com/freddyaboulton/awesome-genai)

---

## 🌱 Contributions
Feel free to fork and modify the plan, or share your progress in Issues or Discussions!

---

## ⭐️ License
MIT
