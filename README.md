# Personal Memory Chatbot (PMC)

### Project Overview
The Personal Memory Chatbot (PMC) is an AI-powered chatbot designed to engage in conversations while utilizing stored memories for context-aware responses. It retrieves relevant memories from past interactions and generates responses using a locally run large language model. The chatbot improves its performance over time by remembering user interactions.

PMC uses **Sentence Transformers** to generate embeddings for each conversation and **FAISS** for fast similarity-based retrieval. The conversational model is powered by **Mistral-7B-Instruct** (via llama.cpp), running natively on your machine with support for Apple Silicon via the Metal backend.

---

### Features

- **Memory System:** Stores past user interactions and retrieves relevant memories to inform responses.
- **Local LLM Inference:** Uses llama.cpp to run **Mistral-7B-Instruct** locally without needing cloud inference or internet connectivity.
- **Fast Memory Retrieval:** Uses **FAISS** for fast similarity search among stored embeddings.
- **Apple Silicon Optimized:** Leverages Metal backend for efficient model execution on macOS (M1/M2/M3).
- **Clean Response Extraction:** Post-processing of model output ensures concise and relevant replies.

---

### Installation

#### 1. Clone the Repository

git clone https://github.com/your-username/Personal-Memory-Chatbot.git
cd Personal-Memory-Chatbot


#### 2. Set Up Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


#### 3. Install Python Dependencies

pip install -r requirements.txt


#### 4. Build llama.cpp

Navigate to the llama.cpp directory and build the binary:

cd models/llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_METAL=on
make -j4


Ensure the llama-cli binary is available at:  
llama.cpp/build/bin/llama-cli

#### 5. Download the Mistral-7B Model (.gguf)

Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF

Download mistral-7b-instruct.Q4_K_M.gguf

Place it inside:
 
llama.cpp/models/mistral-7b.Q4_K_M.gguf

---

### Usage

#### 1. Run the Chatbot

python main.py


#### 2. Chat with the Bot

You can now have a contextual conversation with the bot. Just type your input and get a response based on relevant past interactions.

Type exit to stop the conversation.

#### 3. Memory Handling

Memories are automatically stored in data/memory_store.json and indexed in memory_index.faiss. 
This allows semantic retrieval of past chats to provide better context.

---

### Project Structure

```plaintext
personal-memory-chatbot/
│
├── main.py                # Entry point: runs the chatbot and handles memory and inference
│
├── memory/
│   ├── memory_store.py     # FAISS + JSON memory management
│   └── memory_index.faiss  # FAISS index file
│
├── models/
│   └── model_loader.py     # Loads llama.cpp-based LLM (Mistral) and handles subprocess inference
    ├── llama.cpp/             # Submodule for running Mistral-7B locally
        └── build/bin/llama-cli  # CLI binary for model inference
│
├── utils/
│   ├── prompts.py          # Prompt templates for the chatbot
│   └── logger.py           # Logging utilities
│
├── data/
│   └── memory_store.json   # Stores conversation memory in JSON format
│
└── requirements.txt        # Python dependencies
```

---

### Dependencies

This project requires the following Python libraries:

- torch – Required for Sentence Transformers
- faiss-cpu – Fast vector similarity search
- sentence-transformers – Embedding generation
- numpy – Array operations
- transformers – Optional, for embedding pipelines
- huggingface_hub – For model authentication (optional)

---

### Known Issues & Future Work

- **Multi-platform support:** Current build setup is optimised for macOS with Metal; Linux/Windows instructions pending.
- **Performance Optimisation:** Improvements for managing long memory histories and response latency.
- **Interactive Memory Viewer:** Planned UI to inspect, delete, or tag memories.
- **Customisation:** User-specific personality tuning, memory filtering, and long-term storage.

---

### License

This project is licensed under the [MIT License](./LICENSE).  
© 2025 Saurabh Kohad

Made with ❤️ by **Saurabh Kohad**

