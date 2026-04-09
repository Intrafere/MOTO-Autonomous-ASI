# MOTO Autonomous ASI - Novel S.T.E.M. Solution Variant for ASI Automated Theory Generation
# Autonomous Superintelligence Deep Research Harness
**Version: 1.0.5**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 16+](https://img.shields.io/badge/node-16+-green.svg)](https://nodejs.org/)

**A breakthrough in AI automated theorem generation. An autonomous AI/ASI research system that generates novel and publication-worthy research papers autonomously powered by Intrafere Research Group's new ASI discovery of [Top-P Exploration Through Structured Brainstorming & Validated Feedback](https://intrafere.com/structured-brainstorming-validated-feedback/). Top-P exploration changes how we look at AI weights, a specific combination of reiterative brainstorming, validation, feedback, and pruning allows for superintelligence exploration and data extraction from nearly any combination of AI models. This is useful for any discipline with an interest in creative and novel solution generation for mathematics: physicists, engineers, mathematicians, chemists, etc. This harness can also easily be modified for other research topics such as general academic research, chatbots, niche research, robotics, or anything requiring creative output and/or general autonomy. MOTO's novel brainstorming and rejection/validation stage allows autonomous long-term runtime without user intervention — if desired, research can be conducted for days or weeks without user input.**

### The Core Discovery: Top-P Exploration (Solution Basin Aggregation)

MOTO is built on a [key insight](https://intrafere.com/structured-brainstorming-validated-feedback/) about how generative transformers operate: **transformers predict what tokens come next, so providing them with their own prior ideas enables deeper probing of the solution space**. This is called **solution basin aggregation** — each brainstorming pass explores a richer, more informed landscape, and the cross-recombination of "mined" knowledge compounds to create new insights that do not exist from the model's training alone. Intrafere considers this the mechanism that produces [ASI-like results](https://intrafere.com/structured-brainstorming-validated-feedback/) in practice. MOTO essentially "mines" creativity from a transformer's knowledge set, and this compounding effect is what differentiates it from traditional single-pass AI.

The brainstorming phase runs **multiple submitters in parallel**, each independently exploring the solution space, funneled into a **single bottleneck validator** — a completely separate model instance whose only job is to decide whether each submission genuinely advances the knowledge base. This architectural separation between creative exploration and critical evaluation mitigates the hallucination loops and drift that plague single-model autonomous agents. Every rejection carries specific feedback that steers the next round of exploration, so failure is never wasted. Iterative pruning continuously removes entries that become redundant as stronger ideas emerge, producing an ever-denser, self-refining knowledge base. [View the learning curve data](https://intrafere.com/motos-brainstorming-potential-data/) for empirical evidence of this approach.

### How Research Compounds Across Tiers

Once a brainstorm is sufficiently explored, MOTO writes a research paper from it. This step then repeats — papers become a new "Tier 2" brainstorm. Previous papers are referenced in future brainstorms and future papers. This set of Tier 2 papers is another higher-order brainstorm-like database, which also self-prunes newly discovered incorrect or redundant papers just like the Tier 1 short-hand idea brainstorm does. A third tier generates the final answer, capable of producing book-length volumes.

MOTO may produce many brilliant papers as it runs; these intermediate papers are answers that rival traditional paid cloud deep research. As the user, observe MOTO as often or as little as you'd like — skip its autonomy and force it into final answer generation, or stop it early and select one of its highly creative pre-final answer papers. If the operator allows, let MOTO run for many hours and produce a final answer from its experimental mode. MOTO autonomously decides whether to output a short-form answer or collect existing papers into a long-form academic volume. With models over 131,000 token context limits, the harness easily produces final volumes exceeding 40,000 words autonomously. The built-in "critique" feature allows the user to direct-inject the full volume into nearly any AI model of their choice for evaluation. MOTO writes papers in reverse order — body first, conclusion second, introduction last — to avoid constraining the creative process with premature structural commitments. MOTO is an experimental system; the AI(s) are producing this content partially unguided and all papers should be judged with extreme scrutiny.

Give the program a try, MOTO is as cool as it sounds – there is a one-click installer. Use the two links below to download Python and Node.js, they should automatically install in seconds. Once those are downloaded, click the green “< > Code” drop-down menu on the top right of this GitHub page, download the zip file, extract it to your desktop then double-click "Press to Launch MOTO.bat". Put in your OpenRouter.AI API key (or optionally connect LM Studio for faster performance), select your agents in the settings profile – if desired and you are unsure you may use the preselected “fastest” profile.

***Now you are set up and every time you press launch your home lab is ready for your prompt!*** **Give MOTO the toughest question you can think of and press start to begin YOUR creations!**

**Created by [Intrafere™ LLC](https://intrafere.com)** | [News & Updates](https://intrafere.com/moto-news/)

---

## Outline of "MOTO - S.T.E.M. Mathematics Variant"

MOTO (Multi-Output Token Orchestrator) is a high-risk high-reward (novelty seeking AI) mathematics researcher designed to run for days at a time after pressing start without user interaction. This program can support multiple simultaneous models working in parallel from either local host LM Studio, OpenRouter API key, or both.

### Key Features

- 🤖 **Autonomous Topic Selection, Brainstorming, and Paper Generation**: AI chooses research avenues based on high-level goals and produces you a final answer with ZERO extra user input. Let MOTO run for days using the best models without touching it, or for a few hours using a faster draft model. How deep you research and how long it takes is left up to you, the user.
- **OpenRouter Integration**: Supports both local (LM Studio) and cloud (OpenRouter) models. Use your local LM Studio models run offline from your computer or add your OpenRouter API key to compete and team up 3rd party models from the largest closed source LLMs like ChatGPT, Claude, DeepSeek, Gemini and Perplexity

---

## 🚀 Quick Start

### Prerequisites

Before installation, you need:

1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
2. **Node.js 16+** - [Download here](https://nodejs.org/)
3. **LM Studio** (optional but HIGHLY recommended - otherwise your system will need to pay OpenRouter for RAG embedding calls, which is very slow compared to LM Studio's local embeddings) - [Download here](https://lmstudio.ai/)
   - If using OpenRouter, then download and load at least one model (e.g., DeepSeek, Llama, Qwen - older models and some models below 12 billion parameters may struggle, however it is always worth a try!)
   - **Load the LM Studio RAG agent [optional but HIGHLY recommended for much faster outputs/answers]**: Load the embedding model `nomic-ai/nomic-embed-text-v1.5` in your LM Studio "Developer" tab (server tab) (search for "nomic-ai/nomic-embed-text-v1.5" to download it in the LM Studio downloads center). Please note: you may need to enable "Power User" or "Developer" to see this developer tab - this server will let you load the amount and capacity of simultaneous models that your PC will support. In this developer tab is where you load both your nomic-ai embedding agent and any optional local hosted agents you want to use in the program (I.e. GPT OSS 20b, DeepSeek 32B, etc). **If you do not download LM Studio and enable the Nomic agent the system will run much slower and cost slightly more due to having to use the paid service OpenRouter for RAG calls.**
   - Start the local server (port 1234)
4. **If using cloud AI - Get an OpenRouter API key**: Sign up at OpenRouter.ai and get a paid or free API key to use the most powerful cloud models available from your favorite providers. OpenRouter may also offer a certain amount of free API calls per day with your account key. When you download the MOTO deep research harness, you can see which models are free by checking the "show only free models" check box(es) in the MOTO app settings.
5. **On first startup, pick your provider path**: After you acknowledge the disclaimer, MOTO will prompt you to either enter an OpenRouter key or confirm that LM Studio is running. If you save an OpenRouter key there, the recommended default autonomous profile is applied immediately so you can open Settings and see it already selected.

### Installation

#### Windows (One-Click Launcher)

1. Clone or download this repository
2. Start LM Studio and load your models and "nomic-embed-text-v1.5" agent **and/or** have your OpenRouter API key ready
3. **Double-click `Press to Launch MOTO.bat`**
4. After acknowledging the disclaimer, choose one of the startup setup paths:
   - Enter your OpenRouter API key
   - Confirm that LM Studio is already running with a loaded model
   - Then open Settings to keep the recommended profile or switch to your saved team profile / another default profile
5. The launcher will:
   - Check all prerequisites
   - Install Python and Node.js dependencies automatically
   - Create necessary directories
   - Start backend and frontend servers
   - Open the UI in your browser

**That's it!** The system will be running at `http://localhost:5173`

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
- **RAG**: ChromaDB, Nomic Embeddings, or OpenRouter embeddings fallback if LM Studio is unavailable (not recommended - slower).
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
├── Press to Launch MOTO.bat  # One-click Windows launcher
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
- View the learning curve analysis on the Intrafere.com website and ensure you are not just at a learning curve wall - 100s of rejections in a row before the first acceptance in the brainstorming session can be common.

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
- Monitor your model(s) output(s) occasionally to see if it's stuck in output loops and is repeatedly utilizing its entire output token budget - this is a sign the model runtime instance from either LM Studio or OpenRouter has corrupted. If this is the case you will either need to switch OpenRouter hosts for that model, switch models, or if using LM Studio you must unload and reload the model. However if this happens once, it is likely to happen again so you should either try to switch hosts (if using OpenRouter), switch runtime engines (if using LM Studio), or switch models entirely as some models may be more vulnerable to this than others. This issue does not appear to be related to the MOTO harness and the MOTO developers have no control over this deterministic model-loop corruption state. The repetitive nature of the harness appears to stress certain engines, for example when using AMD compatible engines, ROCm *may* have more instabilities than Vulkan as of 1/11/2026. This is an odd bug and it is unclear if this is related to the 3rd party runtime engine's K/V caching mechanism or some other feature of the code. If you find any information on this bug please submit it to GitHub.

**JSON truncation errors**
- Ensure you are not experiencing the output error looping mentioned above that some LLM runtime engines seem to experience. If your JSON truncation is not a result of looping then you should try increasing your model's max output tokens. It is highly likely that your model was truncated because you did not set enough output tokens (20% or more of your token budget being allotted for token output is standard practice, longer thinking models like DeepSeek V3.2 Speciale may require much larger splits such as (164K total tokens, 64K reserved for output tokens), however most models function great closer to the 20% output budget mark.
---

## 📚 Documentation

- **.cursor/rules/**: Complete system design specifications
  - `part-1-aggregator-tool-design-specifications.mdc`
  - `part-2-compiler-tool-design-specification.mdc`
  - `part-3-autonomous-research-mode.mdc`
  - `rag-design-for-overall-program.mdc`
  - `program-directory-and-file-definitions.mdc`

#### Manual Installation (All Platforms)

```bash
# Clone the repository
git clone https://github.com/Intrafere/MOTO-Autonomous-ASI
cd MOTO-Autonomous-ASI

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright Chromium browser (required for PDF generation - one-time ~150MB download)
python -m playwright install chromium

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

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

**Quick steps:**

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

### Security

Found a security vulnerability? Please review our [Security Policy](SECURITY.md) for responsible disclosure procedures.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Intrafere™ LLC](https://intrafere.com)** - Creator and maintainer
- **LM Studio** for local model hosting
- **OpenRouter** for cloud model access
- **Nomic AI** for embedding models
- **ChromaDB** for vector storage
- **FastAPI** and **React** frameworks

---

## ⚠️ Disclaimer

All content generated by this system is for informational purposes only. Papers are autonomously generated with the novelty-seeking MOTO harness without peer review or user oversight beyond the original prompt. AI-generated content may contain fabricated or unverified claims presented with high confidence - all content should be viewed with extreme scrutiny and independently verified before use. Users are responsible for how they use generated content. All users must follow terms of service, conditions, etc. from all 3rd party applications.

---

## 🔗 Links

- **Website**: https://intrafere.com
- **Top-P Exploration (ASI Discovery)**: https://intrafere.com/structured-brainstorming-validated-feedback/
- **Learning Curve Data**: https://intrafere.com/motos-brainstorming-potential-data/
- **Program Info**: https://intrafere.com/moto-autonomous-home-ai/
- **News & Updates**: https://intrafere.com/moto-news/
- **Donate**: https://intrafere.com/donate/
- **Agentic Programmers**: See `.cursor/rules/` folder and have your agent use the rules as scaffolding to edit the program. Don't forget to keep your rules updated as you go! We recommend you keep your rules on read-only access to prevent your agent from accidentally deviating from your programming plans.
- **Issues**: https://github.com/Intrafere/MOTO-Autonomous-ASI/issues
- **LM Studio**: https://lmstudio.ai/
- **OpenRouter**: https://openrouter.ai/
- **Cursor IDE**: https://cursor.com/

---

## 📊 System Requirements

### Option 1 - Local Large-Model / Large-MoE Setup

Best if you want to run local models in LM Studio, especially models above 20B parameters or larger MoE-style models.

- **OS**: Windows 10+, macOS 12+, Linux
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space for models and project data
- **GPU**: 16GB+ VRAM recommended for practical local inference on 20B+ class models
- **Internet**: Required for installation; optional afterward if staying local-only

### Option 2 - OpenRouter-Only Setup

Best if you want the lightest local hardware requirements and are comfortable running inference in the cloud through OpenRouter.

- **OS**: Windows, macOS, Linux, or Raspberry Pi OS
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 5GB+ free space
- **GPU**: Not required
- **Internet**: Required

Because the heavy model inference happens on OpenRouter, MOTO can run on very modest local hardware in this mode, including a Raspberry Pi, as long as it can run Python, Node.js, and maintain a stable internet connection.

---

**Built for autonomous mathematical research in STEM. Powered by multi-agent AI.**

---

Intrafere™ and Intrafere Research Group™ are trademarks of Intrafere LLC. All rights reserved.

