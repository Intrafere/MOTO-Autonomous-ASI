# MOTO - Mathematics Variant V1
# Autonomous AI Deep Research Harness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 16+](https://img.shields.io/badge/node-16+-green.svg)](https://nodejs.org/)

**An autonomous AI research system that generates mathematical research papers through multi-agent aggregation and compilation. Useful for any discipline that has an interest in novel math: physicists, engineers, mathematicians, chemists, etc. This harness can easily be modified for other research topics such as general academic research, chat bots, niche research, or anything requiring useful creative output. MOTO's novel brainstorming and rejection/validation stage allows autonomous long term runtime without user intervention - if desired, research can be conducted for days or weeks without user input.**

**Created by [Intrafere LLC](https://intrafere.com)** | [Donate](https://intrafere.com/donate/) | [News & Updates](https://intrafere.com/moto-news/)

---

## Description of "MOTO - Math Variant"

MOTO (Multi-Output Token Orchestrator) is a high-risk high-reward (novelty seeking AI) mathematics researcher designed to run for days at a time after pressing start without user interaction. This program can support multiple simultaneous models working in parallel from either local host LM studio, OpenRouter API key, or both.

### Key Features

- 🤖 **Autonomous Topic Selection, Brainstorming, and Paper Generation**: AI chooses research avenues based on high-level goals and produces you a final answer with ZERO extra user input. Let MOTO run for days using the best models without touching it, or for a few hours using a faster draft model. How deep you research and how long it takes is left up to you, the user.
- **OpenRouter Integration**: Supports both local (LM Studio) and cloud (OpenRouter) models. Use your local LM studio models ran offline from your computer or add your OpenRouter API key to compete and team up 3rd party models from the largest closed source LLMs like ChatGPT, Claude, DeepSeek, Gemini and Perplexity

---

## 🚀 Quick Start

### Prerequisites

Before installation, you need:

1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
2. **Node.js 16+** - [Download here](https://nodejs.org/)
3. **LM Studio** (optional but HIGHLY recommended - otherwise your system will need to pay OpenRouter for RAG embedding calls, which is very slow compared to LM studio's local embeddings) - [Download here](https://lmstudio.ai/)
   - If using open router, then download and load at least one model (e.g., DeepSeek, Llama, Qwen - older models and some models below 12 billion parameters may struggle, however it is always worth a try!)
   - **Load the LM Studio RAG agent [optional but HIGHLY recommended for much faster outputs/answers]**: Load the embedding model `nomic-ai/nomic-embed-text-v1.5` in your LM studio "Developer" tab (server tab) (search for "nomic-ai/nomic-embed-text-v1.5" to download it in the LM studio downloads center). Please note: you may need to enable "Power User" or "Developer" to see this developer tab - this server will let you load the amount and capacity of simultaneous models that your PC will suport. In this develop tab is where you load both your nomic-ai embedding agent and any optional local hosted agents you want to use in the program (I.e. GPT OSS 20b, DeepSeek 32B, etc). **If you do not not download LM studio and enable the Nomic agent the system will run much slower and cost a slightly more due to having to use the paid service OpenRouter for RAG calls.**
   - Start the local server (port 1234)
4. **If using cloud AI - Get an OpenRouter API key**: Sign up at OpenRouter.ai and get a paid or free API key to use the most powerful cloud models available from your favorite providers. OpenRouter may also offer a certain amount of free API calls per day with your account key. When you download the MOTO deep research harness, you can see which models are free by checking the "show only free models" check box(es) in the MOTO app settings.

### Installation

#### Windows (One-Click Launcher)

1. Clone or download this repository
2. Start LM Studio and load your models and "nomic-embed-text-v1.5" agent **and/or** enter OpenRouter API key
3. **Double-click `launch.bat`**
4. The launcher will:
   - Check all prerequisites
   - Install Python and Node.js dependencies automatically
   - Create necessary directories
   - Start backend and frontend servers
   - Open the UI in your browser

**That's it!** The system will be running at `http://localhost:5173`

#### Manual Installation (All Platforms)

```bash
# Clone the repository
git clone https://github.com/Intrafere/MOTO
cd moto-math-variant

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..

# Create necessary directories
mkdir -p backend/data/user_uploads
mkdir -p backend/logs

# Start the backend (in one terminal)
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

# Start the frontend (in another terminal)
cd frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## 📖 Usage Guide

### Mode 1 - Autonomous Mode (multi-paper writer):

1. Go to **Autonomous Research** tab
2. Enter high-level research goal (e.g., "Solve the Langlands Bridge problem")
3. Configure model settings for all roles
4. Click **Start Autonomous Research**
5. System will:
   - Autonomously select research topics
   - Build brainstorm databases
   - Generate complete papers
   - Create final answer synthesis (after 5 papers)

### Mode 2 - Single Paper Writer (advanced/manual mode):

#### Mode 2, step 1: Aggregator (Knowledge Building)

1. Go to **Aggregator Interface** tab
2. Enter your research prompt (e.g., "Explore connections between modular forms and Galois representations")
3. Configure settings:
   - Select submitter and validator models
   - Set context window sizes (default: 131072 tokens)
   - Configure 1-10 submitters (default: 3)
4. Click **Start Aggregator**
5. Monitor progress in **Aggregator Logs** tab
6. View accepted submissions in **Live Results** tab

#### Mode 2, step 2: Compiler (Paper Generation)

1. Go to **Compiler Interface** tab
2. Enter compiler-directing prompt (e.g., "Build a paper titled 'Modular Forms in the Langlands Program'")
3. Configure settings:
   - Select validator, high-context, and high-parameter models
   - Set context windows and output token limits
4. Click **Start Compiler**
5. Watch real-time paper construction in **Live Paper** tab
6. Monitor metrics in **Compiler Logs** tab

---

## 🛠️ System Architecture

### Technology Stack

- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Frontend**: React, Vite, Tailwind CSS
- **AI**: LM Studio API, OpenRouter API
- **RAG**: ChromaDB, Nomic Embeddings, or OpenRouter embeddings fallback if LM studio is unavailable (not recommended - slower).
- **WebSocket**: Real-time updates

### Key Components

- **RAG System**: 4-stage retrieval (query rewriting, hybrid recall, reranking, packing)
- **Multi-Agent Coordinator**: Manages parallel submitters and sequential validation
- **Context Allocator**: Direct injection vs RAG routing based on token budgets
- **Workflow Predictor**: Predicts next 20 API calls for boost selection
- **Boost Manager**: Selective task acceleration with OpenRouter

---

## 📁 Project Structure

```
moto-math-variant/
├── backend/
│   ├── aggregator/          # Tier 1: Multi-agent knowledge aggregation
│   ├── compiler/            # Tier 2: Paper compilation and validation
│   ├── autonomous/          # Tier 3: Autonomous topic selection and synthesis
│   ├── api/                 # FastAPI routes and WebSocket
│   ├── shared/              # Shared utilities, models, API clients
│   └── data/                # Persistent storage (databases, papers, logs)
├── frontend/
│   └── src/
│       ├── components/      # React components for UI
│       └── services/        # API and WebSocket clients
├── .cursor/
│   └── rules/               # AI agent design specifications (full system documentation)
├── launch.bat               # One-click Windows launcher
├── launch.ps1               # PowerShell launcher
├── requirements.txt         # Python dependencies
└── package.json             # Node.js dependencies
```

---

## ⚙️ Configuration

### Model Selection

**Aggregator**:
- 1-10 submitters (configurable, default 3)
- Each submitter can use different models
- Single validator model (for coherent Markov chain)

**Compiler**:
- Validator model (coherence/rigor checking)
- High-context model (outline, construction, review)
- High-parameter model (rigor enhancement)

**Autonomous Research**:
- All aggregator and compiler roles configurable
- Separate models for topic selection, completion review, etc.

### OpenRouter Integration

Each role supports:
- **Provider**: LM Studio (local) or OpenRouter (cloud)
- **Model Selection**: Choose from available models
- **Host/Provider**: Select specific OpenRouter provider (e.g., Anthropic, Google)
- **Fallback**: Optional LM Studio fallback if OpenRouter fails

### Context and Output Settings

All configurable per role:
- **Context Window**: Default 131072 tokens (user-adjustable)
- **Max Output Tokens**: Default 25000 tokens (recommended for reasoning models)

---

## 🔧 Troubleshooting

### Installation Issues

**"Python not recognized"**
- Reinstall Python and check "Add Python to PATH"
- Verify: `python --version` in terminal

**"Node not recognized"**
- Install Node.js from nodejs.org
- Verify: `node --version` in terminal

**"pip install failed"**
- Check internet connection
- Try: `python -m pip install --upgrade pip`
- Run as administrator if permission errors

### Runtime Issues

**"Failed to connect to LM Studio"**
- Ensure LM Studio is running
- Start the local server in LM Studio (port 1234)
- Load at least one model
- Load embedding model: `nomic-ai/nomic-embed-text-v1.5`

**"Port already in use"**
- Close other apps using ports 8000 or 5173
- Restart computer if needed
- Use different ports in config

**High rejection rate**
- Check models are generating valid JSON
- Review validator reasoning in logs
- Ensure prompt is clear and specific
- Use larger models for better results
- View the learning curve analysis on the Intrafere.com website and ensure you are not just at a learning curve wall - 100's of rejections in a row before the first acceptance in the brainstorming session can be common.

**System running slow**
- Use faster/smaller models
- Reduce context window size
- Close resource-intensive apps
- Check RAG cache performance in logs

### Common Error Messages

**"ChromaDB corruption detected"**
- Delete `backend/data/chroma_db` folder
- Restart the system (launcher cleans ChromaDB automatically)

**"Context window exceeded"**
- Reduce context size in settings
- System will automatically offload to RAG
- Check logs for detailed token usage

**JSON and output errors**
- Monitor your model(s) output(s) occassionally to see if its stuck in output loops and is repeatedly utilizing its entire output token budget - this is a sign the model runtime instance from either LM Studio or OpenRouter has corrupted. If this is the case you will either need to Switch OpenRouter hosts for that model, switch models, or if using LM studio you must unload and reload the model. However if this happens once, it is likely to happen again so you should either try to switch hosts (if using OpenRouter), switch runtime engines (if using LM studio), or switch models entirely as some models may be more vulnerable to this than others. This issue does not appear to be related to MOTO harness and the MOTO developers have no control over this deterministic model-loop corruption state. The repetitive nature of the harness appear to stress certain engines, for example when using AMD comptable engines, ROCm *may* have more instabilities than Vulkan as of 1/11/2026. This is an odd bug and it is unclear if this is related to the 3rd party runtime engine's K/V caching mechanism or some other feature of the code. If you find any information on this bug please submit it to the GitHub.

**JSON truncation errors**
- Ensure you are not experiencing the output error looping mentioned above that some LLM runtime engines seem to experience, if your JSON truncation is not a result of looping then you should try increasing your models max output tokens. It is highly likely that your model was truncated because you did not set enough output tokens (20% or more of your token budget being allotted for token output is standard practice, longer thinking models like DeepSeek V3.2 Speciale may require much larger splits such as (164K total tokens, 64K reserved for output tokens), however most models function great closer to the 20% output budget mark.
---

## 📚 Documentation

- **QUICKSTART.md**: Step-by-step setup guide
- **START_HERE.txt**: Quick reference and prompting tips
- **.cursor/rules/**: Complete system design specifications
  - `part-1-aggregator-tool-design-specifications.mdc`
  - `part-2-compiler-tool-design-specification.mdc`
  - `part-3-autonomous-research-mode.mdc`
  - `rag-design-for-overall-program.mdc`
  - `program-directory-and-file-definitions.mdc`

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Development Setup

For development with AI assistance:
- Install [Cursor](https://cursor.com/) - AI-powered IDE
- The `.cursor/rules/` folder contains complete design specifications
- Cursor can help you understand and modify the system

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Intrafere LLC](https://intrafere.com)** - Creator and maintainer
- **LM Studio** for local model hosting
- **OpenRouter** for cloud model access
- **Nomic AI** for embedding models
- **ChromaDB** for vector storage
- **FastAPI** and **React** frameworks

---

## ⚠️ Disclaimer

Papers generated by this system have not been peer-reviewed and were autonomously generated without user oversight beyond the original prompts. Content may contain errors. Papers often contain ambitious content and/or extraordinary claims - all content should be viewed with extreme scrutiny. All users must follow terms of service, conditions, etc from all 3rd party applications.

---

## 🔗 Links

- **Website**: https://intrafere.com
- **Program Info**: https://intrafere.com/moto-autonomous-home-ai/
- **News & Updates**: https://intrafere.com/moto-news/
- **Donate**: https://intrafere.com/donate/
- **Documentation**: See `.cursor/rules/` folder
- **Issues**: https://github.com/Intrafere/MOTO/issues
- **LM Studio**: https://lmstudio.ai/
- **OpenRouter**: https://openrouter.ai/
- **Cursor IDE**: https://cursor.com/

---

## 📊 System Requirements

### Minimum

- **OS**: Windows 10+, macOS 10.15+, Linux
- **RAM**: 16GB (for running local models)
- **Storage**: 10GB free space
- **Internet**: Required for package installation and OpenRouter

### Recommended

- **OS**: Windows 11, macOS 12+, Linux
- **RAM**: 32GB+ (for larger models)
- **Storage**: 50GB+ (for multiple models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for faster inference)

---

**Built for autonomous mathematical research. Powered by multi-agent AI.**

