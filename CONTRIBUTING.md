# Contributing to MOTO Math Variant

Thank you for your interest in contributing to MOTO! This project is developed and maintained by [Intrafere LLC](https://intrafere.com).

This document provides guidelines for contributing to the project.

---

## 🎯 Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with detailed reproduction steps
- **Feature Requests**: Have an idea? Propose it in an issue first
- **Code Contributions**: Submit pull requests for bug fixes or features
- **Documentation**: Improve documentation, guides, or examples
- **Testing**: Test the system with different models and report results
- **Feedback**: Share your experience using the system

---

## 🚀 Getting Started

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/Intrafere/MOTO-Autonomous-ASI
   cd MOTO-Autonomous-ASI
   ```

3. **Install dependencies**:
   ```bash
   # Python dependencies
   pip install -r requirements.txt
   
   # Frontend dependencies
   cd frontend
   npm install
   cd ..
   ```

4. **Set up development environment**:
   - Install [Cursor IDE](https://cursor.com/) (recommended for AI-assisted development)
   - The `.cursor/rules/` folder contains complete design specifications
   - Cursor can help you understand and modify the codebase

5. **Create a branch for your work**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

---

## 📝 Development Guidelines

### Code Style

**Python**:
- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to classes and functions
- Keep functions focused and modular

**JavaScript/React**:
- Use functional components with hooks
- Follow JSX best practices
- Use meaningful variable names
- Add comments for complex logic

### Project Structure

Understand the three-tier architecture:
- **Tier 1 (Aggregator)**: `backend/aggregator/` - Multi-agent knowledge building
- **Tier 2 (Compiler)**: `backend/compiler/` - Paper compilation and validation
- **Tier 3 (Autonomous)**: `backend/autonomous/` - Autonomous research workflow

Key directories:
- `backend/shared/`: Shared utilities, API clients, models
- `backend/api/`: FastAPI routes and WebSocket
- `frontend/src/components/`: React UI components
- `.cursor/rules/`: Complete system design specifications

### Design Specifications

**CRITICAL**: Before making significant changes, read the relevant design docs in `.cursor/rules/`:

1. **part-1-aggregator-tool-design-specifications.mdc**: Multi-agent aggregation workflow
2. **part-2-compiler-tool-design-specification.mdc**: Paper compilation system
3. **part-3-autonomous-research-mode.mdc**: Autonomous topic selection and synthesis
4. **rag-design-for-overall-program.mdc**: RAG architecture and 4-stage pipeline
5. **program-directory-and-file-definitions.mdc**: File structure and purpose

These specifications are used by AI agents (like Cursor) to assist with development.

---

## 🐛 Reporting Bugs

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** in `.cursor/rules/` and README.md
3. **Test with latest version** from main branch

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.10.0]
- Node.js version: [e.g., 18.0.0]
- LM Studio version: [e.g., 0.2.9]
- Models used: [e.g., DeepSeek R1 70B, Llama 3.1 70B]

**Logs**
Paste relevant logs from:
- Backend terminal output
- Browser console (F12)
- `backend/logs/system.log`

**Additional context**
Any other relevant information.
```

---

## 💡 Proposing Features

### Before Proposing

1. **Check if already proposed** in issues
2. **Review design specs** to understand current architecture
3. **Consider system constraints** (multi-agent coordination, RAG pipeline, etc.)

### Feature Request Template

```markdown
**Is your feature related to a problem?**
Clear description of the problem.

**Describe the solution you'd like**
Clear description of desired functionality.

**Describe alternatives considered**
Alternative solutions or features you've considered.

**Architecture impact**
How would this affect:
- Aggregator (Tier 1)?
- Compiler (Tier 2)?
- Autonomous Research (Tier 3)?
- RAG system?
- UI/UX?

**Additional context**
Mockups, examples, or relevant information.
```

---

## 🔀 Pull Request Process

### 1. Make Your Changes

- Follow code style guidelines
- Add tests if applicable
- Update documentation if needed
- Test thoroughly with different models

### 2. Commit Guidelines

Use descriptive commit messages:

```bash
# Good
git commit -m "Fix: Prevent duplicate section headers in compiler validation"
git commit -m "Feature: Add batch validation for 2-3 submissions"
git commit -m "Docs: Update RAG pipeline documentation"

# Bad
git commit -m "fix bug"
git commit -m "changes"
git commit -m "update"
```

### 3. Push and Create PR

```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub with:

**Title**: Clear, concise description of changes

**Description**:
```markdown
## Changes
- List of changes made

## Related Issue
Fixes #123

## Testing
- [ ] Tested with LM Studio models
- [ ] Tested with OpenRouter models
- [ ] Tested aggregator workflow
- [ ] Tested compiler workflow
- [ ] Tested autonomous research
- [ ] UI changes work in Chrome/Firefox/Edge

## Screenshots (if UI changes)
[Add screenshots]

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings/errors
- [ ] Tested on Windows/Mac/Linux (if applicable)
```

### 4. Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

---

## 🧪 Testing

### Manual Testing

1. **Aggregator Testing**:
   - Start aggregator with test prompt
   - Monitor acceptance/rejection rates
   - Verify submissions appear in live results
   - Check RAG retrieval quality

2. **Compiler Testing**:
   - Start compiler with aggregator database
   - Verify outline creation
   - Check paper construction quality
   - Test review and rigor phases

3. **Autonomous Research Testing**:
   - Start autonomous mode with research goal
   - Verify topic selection works
   - Check brainstorm aggregation
   - Test paper compilation
   - Verify Tier 3 final answer generation

### Testing Different Models

Test with various model combinations:
- Small models (7B-13B)
- Medium models (30B-40B)
- Large models (70B+)
- OpenRouter models (GPT-4, Claude, etc.)

### Load Testing

- Test with long prompts
- Test with large file uploads
- Test with multiple concurrent operations
- Monitor memory usage and performance

---

## 📚 Documentation

### When to Update Docs

Update documentation when you:
- Add new features
- Change existing functionality
- Fix bugs that affect usage
- Modify API endpoints
- Change configuration options

### Documentation Locations

- **README.md**: Main project documentation
- **.cursor/rules/**: System design specifications (update if architecture changes)
- **Code comments**: Add inline documentation
- **Docstrings**: Document classes and functions

---

## 💬 Getting Help

- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check `.cursor/rules/` for detailed design specs
- **Cursor IDE**: Use Cursor with the rules folder for AI-assisted development

---

## 🏆 Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes
- Project acknowledgments

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MOTO! 🎉

