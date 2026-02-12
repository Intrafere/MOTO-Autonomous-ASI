# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in MOTO, please report it privately:

### 📧 Contact

Email security reports to: **[security@intrafere.com](mailto:security@intrafere.com)**

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Status Updates**: Every 7 days until resolved
- **Fix Release**: Depends on severity (critical: 7 days; high: 30 days; medium: 90 days)

---

## Security Best Practices for Users

### API Key Protection

**NEVER commit API keys to the repository:**
- OpenRouter API keys should be entered through the UI only
- Keys are stored in browser localStorage, not in code
- Use `.gitignore` to exclude sensitive data files
- Check `.gitignore` includes `backend/data/` subdirectories

### LM Studio Security

**Local model hosting:**
- LM Studio runs on localhost (127.0.0.1:1234)
- No external network access required for local models
- Models execute on your machine only
- No data leaves your system when using LM Studio exclusively

### OpenRouter Usage

**When using OpenRouter:**
- Your API key is sent only to OpenRouter API endpoints
- Research content may be sent to OpenRouter for model inference
- Review OpenRouter's privacy policy: https://openrouter.ai/privacy
- Free models may require data sharing consent (check privacy settings)
- Paid models typically have stricter privacy protections

### Generated Content

**AI-generated papers contain disclaimers:**
- Papers include "AUTONOMOUS AI SOLUTION" disclaimers
- Content has not been peer-reviewed
- May contain errors or unverified claims
- All content should be verified independently

---

## Known Security Considerations

### 1. XSS Prevention in LaTeX Rendering

**Component**: `frontend/src/components/LatexRenderer.jsx`

**Protection**: DOMPurify sanitization
- All LaTeX-rendered content is sanitized before display
- Prevents malicious script injection in generated papers
- Configuration blocks: `<script>`, `<iframe>`, `<form>`, event handlers
- See `.cursor/rules/latex-renderer-security.mdc` for details

**Status**: ✅ Fixed (DOMPurify v3.2.4+ includes CVE-2025-26791 fix)

### 2. PDF Generation Security

**Dependencies**:
- `html2pdf.js` v0.14.0+ (fixes GHSA-w8x4-x68c-m6fc XSS vulnerability)
- `jspdf` v4.0.0+ (fixes CVE-2025-68428 LFI/Path Traversal)

**Status**: ✅ Fixed (both vulnerabilities patched)

### 3. JSON Parsing

**Component**: `backend/shared/json_parser.py`

**Protection**:
- Sanitizes LLM outputs before parsing
- Removes reasoning tokens, markdown wrappers, control tokens
- Validates structure before execution
- Rejects truncated or malformed JSON

### 4. File Upload Handling

**Component**: `backend/api/routes/aggregator.py`

**Protection**:
- Files stored in isolated `backend/data/user_uploads/` directory
- No code execution on uploaded files
- Files processed as text only
- Maximum file size enforced by FastAPI

---

## Security Updates

### Recent Security Fixes

**2026-01-15**: html2pdf.js XSS vulnerability (GHSA-w8x4-x68c-m6fc)
- Updated html2pdf.js from v0.12.1 to v0.14.0
- Affects PDF download functionality in all components
- See COMMITS_PENDING.txt for details

**2025-12-20**: jspdf LFI/Path Traversal (CVE-2025-68428)
- Pinned jspdf to v4.0.0 via overrides
- Affects PDF generation in all download features
- Both direct dependency and npm overrides enforce v4.0.0

**2025-12-15**: DOMPurify mXSS vulnerability (CVE-2025-26791)
- Updated DOMPurify to v3.2.4
- Affects all LaTeX rendering components
- Prevents mutation XSS attacks

---

## Dependency Security

### Automated Scanning

We use:
- **npm audit** for frontend dependencies
- **pip-audit** for Python dependencies (recommended)
- **Dependabot** (GitHub) for automated vulnerability alerts

### Manual Reviews

Security-sensitive dependencies reviewed regularly:
- `dompurify` (HTML sanitization)
- `html2pdf.js` and `jspdf` (PDF generation)
- `fastapi` (API framework)
- `chromadb` (vector database)

### Updating Dependencies

```bash
# Check for vulnerabilities
npm audit                    # Frontend
pip-audit                    # Backend (requires: pip install pip-audit)

# Update dependencies
npm update                   # Frontend
pip install --upgrade -r requirements.txt  # Backend
```

---

## Secure Development Practices

### For Contributors

1. **Never hardcode secrets** - use environment variables or UI configuration
2. **Sanitize all user inputs** - especially in prompts and file uploads
3. **Validate LLM outputs** - use structured JSON schemas
4. **Use DOMPurify** for any HTML rendering of untrusted content
5. **Review `.gitignore`** - ensure sensitive files are excluded
6. **Test with malicious inputs** - verify sanitization works
7. **Update dependencies regularly** - check for security advisories

### Code Review Checklist

Before merging:
- [ ] No hardcoded API keys or secrets
- [ ] User inputs are sanitized
- [ ] LLM outputs are validated
- [ ] HTML content uses DOMPurify
- [ ] Dependencies are up to date
- [ ] No new security warnings from `npm audit`
- [ ] Sensitive data excluded by `.gitignore`

---

## Security Audit History

| Date | Component | Issue | Status |
|------|-----------|-------|--------|
| 2026-01-15 | html2pdf.js | XSS vulnerability (GHSA-w8x4-x68c-m6fc) | ✅ Fixed |
| 2025-12-20 | jspdf | LFI/Path Traversal (CVE-2025-68428) | ✅ Fixed |
| 2025-12-15 | DOMPurify | mXSS vulnerability (CVE-2025-26791) | ✅ Fixed |
| 2025-12-05 | LatexRenderer | Missing XSS sanitization | ✅ Fixed |

---

## Scope

### In Scope

- Security vulnerabilities in MOTO code
- Dependency vulnerabilities
- XSS, injection, or code execution issues
- Data leakage or privacy concerns
- Authentication/authorization issues (if applicable)

### Out of Scope

- Issues in third-party services (LM Studio, OpenRouter)
- Model-generated content quality
- Performance optimization
- Feature requests
- General support questions

---

## Security Resources

- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **GitHub Security Advisories**: https://github.com/advisories
- **npm Security Advisories**: https://www.npmjs.com/advisories
- **Python Security**: https://python.org/dev/security/

---

## Attribution

We credit security researchers who responsibly disclose vulnerabilities:
- Reports will be acknowledged in release notes (unless reporter prefers anonymity)
- Significant findings may be eligible for recognition on our website

---

**Thank you for helping keep MOTO secure!** 🔒

For non-security issues, please use GitHub Issues: https://github.com/Intrafere/MOTO-Autonomous-ASI/issues

