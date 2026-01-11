# MOTO - Math Variant V1 (Autonomous Solution Intelligence)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 16+](https://img.shields.io/badge/node-16+-green.svg)](https://nodejs.org/)

**An autonomous AI research system that generates mathematical research papers through multi-agent aggregation and compilation.**

**Created by [Intrafere LLC](https://intrafere.com)** | [Donate](https://intrafere.com/donate/) | [News & Updates](https://intrafere.com/moto-news/)

---

## 🎯 What This System Does

MOTO is a three-tier autonomous AI research system:

- **Tier 1 (Aggregator)**: 1-10 AI agents work in parallel to explore mathematical concepts and build validated knowledge databases
- **Tier 2 (Compiler)**: Sequential AI agents compile knowledge into coherent academic papers with rigorous validation
- **Tier 3 (Autonomous Research)**: System autonomously selects topics, generates papers, and synthesizes final answers

### Key Features

- 🤖 **Multi-Agent Architecture**: Configurable 1-10 parallel submitters + 1 validator
- 🧠 **Advanced RAG System**: 4-stage retrieval pipeline with semantic search and citation tracking
- 📄 **Automated Paper Generation**: Creates structured academic papers with mathematical rigor
- 🔄 **Autonomous Topic Selection**: AI chooses research avenues based on high-level goals
- 🔒 **OpenRouter Integration**: Supports both local (LM Studio) and cloud (OpenRouter) models
- ⚡ **API Boost Mode**: Selective task acceleration with usage tracking
- 📊 **Real-time Monitoring**: Live metrics, acceptance rates, and workflow visualization

---

## 🚀 Quick Start

### Prerequisites

Before installation, you need:

1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
2. **Node.js 16+** - [Download here](https://nodejs.org/)
3. **LM Studio** (optional but recommended) - [Download here](https://lmstudio.ai/)
   - Download and load at least one model (e.g., DeepSeek, Llama, Qwen)
   - **Optional But Recommended (faster outputs)**: Load the embedding model `nomic-ai/nomic-embed-text-v1.5`
   - Start the local server (port 1234)

### Installation

#### Windows (One-Click Launcher)

1. Clone or download this repository
2. Start LM Studio and load your models
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

### Part 1: Aggregator (Knowledge Building)

1. Go to **Aggregator Interface** tab
2. Enter your research prompt (e.g., "Explore connections between modular forms and Galois representations")
3. Configure settings:
   - Select submitter and validator models
   - Set context window sizes (default: 131072 tokens)
   - Configure 1-10 submitters (default: 3)
4. Click **Start Aggregator**
5. Monitor progress in **Aggregator Logs** tab
6. View accepted submissions in **Live Results** tab

### Part 2: Compiler (Paper Generation)

1. Go to **Compiler Interface** tab
2. Enter compiler-directing prompt (e.g., "Build a paper titled 'Modular Forms in the Langlands Program'")
3. Configure settings:
   - Select validator, high-context, and high-parameter models
   - Set context windows and output token limits
4. Click **Start Compiler**
5. Watch real-time paper construction in **Live Paper** tab
6. Monitor metrics in **Compiler Logs** tab

### Part 3: Autonomous Research

1. Go to **Autonomous Research** tab
2. Enter high-level research goal (e.g., "Solve the Langlands Bridge problem")
3. Configure model settings for all roles
4. Click **Start Autonomous Research**
5. System will:
   - Autonomously select research topics
   - Build brainstorm databases
   - Generate complete papers
   - Create final answer synthesis (after 5 papers)

---

## 🛠️ System Architecture

### Technology Stack

- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Frontend**: React, Vite, Tailwind CSS
- **AI**: LM Studio API, OpenRouter API
- **RAG**: ChromaDB, Nomic Embeddings
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

