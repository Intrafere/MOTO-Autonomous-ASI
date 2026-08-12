# Security Policy

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
- Enter desktop provider credentials through MOTO's credential UI. Hosted operators should inject credentials through the platform's secret environment or authenticated API path.
- In desktop/default mode, OpenRouter and Wolfram API keys, OAuth tokens, and supported subscription-provider keys are persisted through the operating system's credential store. MOTO delegates protection to that OS keyring; it does not encrypt the credentials itself.
- In hosted/generic mode, provider keys are injected through environment variables or authenticated API requests and are kept in backend process memory rather than persisted to the desktop keyring.
- Credentials necessarily exist in backend process memory while they are in use. Do not put provider keys in prompts, uploaded files, source code, browser storage, or runtime settings.
- Use `.gitignore` to exclude sensitive data files
- Check `.gitignore` includes `backend/data/` subdirectories

### Privacy and External Data Flow

MOTO can operate close to offline during normal research, but this depends on the selected providers and configuration. The closest-to-offline setup uses LM Studio at its default loopback address for both model inference and embeddings, with OpenRouter, desktop cloud providers, Wolfram|Alpha, and other optional network integrations disabled. This is not an air-gap guarantee: installation, updates, package or model downloads, and optional Lean/Mathlib setup can still require internet access. Operators may also override the LM Studio address to a non-local server.

| Provider path | What may leave the machine | Credential handling | Processing and retention expectation |
|---|---|---|---|
| **LM Studio at the default loopback address** | Model prompts and embedding text stay between MOTO and the LM Studio server on `127.0.0.1`. | LM Studio does not require a MOTO API key. | Inference is local when the configured server is genuinely local. If LM Studio embeddings fail and OpenRouter fallback is available, text selected for embedding may be sent to OpenRouter. |
| **OpenRouter** | The complete request assembled for a role is sent to OpenRouter and may be forwarded to the selected upstream model provider. It can include the user prompt, system instructions, uploaded or retrieved text, accepted research, drafts, proof material, feedback, and tool results when included in that role's context. Embedding fallback can also send text selected for embedding. | Desktop keys use the OS keyring; hosted keys come from environment/authenticated API input and remain in process memory. The key is transmitted to OpenRouter for authentication. | OpenRouter and the upstream provider control processing and retention. OpenRouter offers privacy, data-collection, and Zero Data Retention controls, but compatibility varies by endpoint. Verify the selected endpoint and current account settings before use. |
| **Wolfram\|Alpha** | When the optional tool is enabled and invoked, MOTO sends the model-generated computational query and the Wolfram App ID to Wolfram. Wolfram's result is returned to the active writing model, so it is also sent to that model's cloud provider when the role is cloud-hosted. The full MOTO research prompt is not automatically sent to Wolfram. | Desktop App IDs use the OS keyring; hosted App IDs remain in process memory. | Wolfram's policies allow collection of query information and govern its retention and use. Do not assume that a query has zero retention. |
| **Desktop OAuth/subscription providers** | When selected for a role, MOTO sends that role's complete assembled model request directly to the provider, such as OpenAI Codex/ChatGPT, xAI Grok/SuperGrok, or Sakana Fugu. These are separate paths, not OpenRouter relays. | OAuth tokens and supported subscription API keys use the desktop OS keyring and also exist in process memory while active. These providers are unavailable in hosted/generic mode. | The selected provider's current privacy, training, and retention terms apply. MOTO cannot impose a provider-side retention guarantee. |
| **Hosted/generic MOTO** | Browser requests first enter the private hosted MOTO sandbox. Embeddings use in-process FastEmbed, but assembled LLM requests are sent from the sandbox to OpenRouter. | Provider keys are environment-injected or supplied through authenticated API routes and are held in sandbox process memory. | Hosted transport controls protect access to the sandbox but do not prevent intentional cloud-model processing. Hosted mode is not a near-offline deployment. |

Exact transmitted content depends on the workflow role, phase, enabled integrations, and context allocation. Some sources are injected in full, some contribute retrieved excerpts, and others may be excluded. Do not submit secrets, regulated data, or confidential source material to a cloud provider unless the provider's current terms and your organization's policies permit it.

Third-party policies can change. Review the current sources before handling sensitive workloads:
- [OpenRouter data collection and prompt logging](https://openrouter.ai/docs/guides/privacy/data-collection)
- [OpenRouter Zero Data Retention controls](https://openrouter.ai/docs/guides/features/zdr)
- [Wolfram Privacy Policy](https://www.wolfram.com/legal/privacy/wolfram/index.html)
- [Wolfram|Alpha API Terms of Use](https://products.wolframalpha.com/api/termsofuse)
- The privacy and data-use terms published by the selected upstream model or OAuth/subscription provider

### Local MOTO Logging

- MOTO's default API-call logs do not persist full prompt and response bodies, but they do persist bounded, credential-redacted content previews, payload sizes, and hashes under the active data root.
- Pattern-based credential redaction is not a general personal-data or confidential-information classifier. Ordinary sensitive research text may still appear in a bounded preview.
- Desktop operators can explicitly enable full-payload debug logging with `MOTO_API_LOG_STORE_FULL_PAYLOADS` / `API_LOG_STORE_FULL_PAYLOADS`; do not enable it for sensitive workloads. Generic/hosted mode keeps full-payload logging disabled.
- Wolfram activity logs record redacted metadata and lengths rather than raw query/result text.

### Generated Content

**AI-generated papers contain disclaimers:**
- All generated content is for informational purposes only
- Papers include "AUTONOMOUS AI SOLUTION" disclaimers
- Content has not been peer-reviewed
- May contain fabricated or unverified claims presented with high confidence
- All content should be independently verified before use

---

## Known Security Considerations

### 1. XSS Prevention in LaTeX Rendering

**Component**: `frontend/src/components/LatexRenderer.jsx`

**Protection**: DOMPurify sanitization
- All LaTeX-rendered content is sanitized before display
- Prevents malicious script injection in generated papers
- Configuration blocks: `<script>`, `<iframe>`, `<form>`, event handlers
- See `.cursor/rules/latex-renderer.mdc` for details

**Status**: ✅ Fixed (DOMPurify v3.2.4+ includes CVE-2025-26791 fix)

### 2. PDF Generation Security

**Component**: `backend/api/routes/download.py` + `frontend/src/utils/downloadHelpers.js`

**Approach**: Backend Playwright (headless Chromium) PDF rendering
- All content is DOMPurify-sanitized on the frontend **before** being sent to the backend
- Backend receives only sanitized HTML — no raw LLM output ever reaches the PDF renderer
- User-supplied metadata (title, outline) is HTML-escaped via `_escape_html()` before interpolation into the HTML template
- Playwright runs as an isolated subprocess — no impact on the FastAPI event loop
- `html2pdf.js` and `jspdf` (and their CVEs) have been removed entirely

**Status**: ✅ Secure (html2pdf.js and jspdf CVEs eliminated by removal)

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

**2026-03-20**: PDF generation migrated from html2pdf.js/jspdf to Playwright (headless Chromium)
- Removed `html2pdf.js` and `jspdf` and all associated CVEs from the dependency tree
- PDF generation now runs server-side via Playwright in a thread pool (non-blocking)
- DOMPurify sanitization still applied client-side before content is sent to the backend
- Eliminates GHSA-w8x4-x68c-m6fc (html2pdf.js XSS), CVE-2025-68428 and CVE-2026-24737 (jspdf)

**2026-01-15**: html2pdf.js XSS vulnerability (GHSA-w8x4-x68c-m6fc)
- Updated html2pdf.js from v0.12.1 to v0.14.0
- Affects PDF download functionality in all components
- See COMMITS_PENDING.txt for details

**2025-12-20**: jspdf LFI/Path Traversal (CVE-2025-68428)
- Pinned jspdf to v4.1.0 via overrides
- Affects PDF generation in all download features
- Both direct dependency and npm overrides enforce v4.1.0

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
- `playwright` (headless Chromium PDF generation)
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

## Scope for Reporting

### In Scope

- Security vulnerabilities in MOTO code
- Dependency vulnerabilities
- XSS, injection, or code execution issues
- Data leakage or privacy concerns
- Authentication/authorization issues (if applicable)

### Out of Scope

- Issues in third-party services (LM Studio, OpenRouter)
- Model-generated content quality (including incorrect LaTeX)
- Performance optimization
- Feature requests (use the discussion section of the GitHub)
- General support questions

---

## Security Resources

- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **GitHub Security Advisories**: https://github.com/advisories
- **npm Security Advisories**: https://www.npmjs.com/advisories
- **DOMPurify**: https://github.com/cure53/DOMPurify
- **Python Security**: https://python.org/dev/security/

---

## Attribution

We credit security researchers who responsibly disclose vulnerabilities:
- Reports will be acknowledged in release notes (unless reporter prefers anonymity)
- Significant findings may be eligible for recognition on our website

---

**Thank you for helping keep MOTO secure!** 🔒

For non-security issues, please use GitHub Issues: https://github.com/Intrafere/MOTO-Autonomous-ASI/issues

