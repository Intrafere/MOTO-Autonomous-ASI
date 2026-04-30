"""
PDF download route - uses Playwright (headless Chromium) for full-fidelity rendering.
Runs in a thread pool so the FastAPI event loop is never blocked.
"""
import asyncio
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/download", tags=["download"])

# KaTeX CSS — read from the installed npm package to avoid CDN dependency
# Path: frontend/node_modules/katex/dist/katex.min.css (installed via npm)
_KATEX_CSS_PATH = Path(__file__).parent.parent.parent.parent / "frontend" / "node_modules" / "katex" / "dist" / "katex.min.css"
_KATEX_CSS: str = ""

try:
    _KATEX_CSS = _KATEX_CSS_PATH.read_text(encoding="utf-8")
except Exception as e:
    logger.warning(f"Could not read katex.min.css from node_modules: {e}. Math rendering may be degraded.")

# Path to LatexRenderer.css — read once at module load so it's always available
_LATEX_RENDERER_CSS_PATH = Path(__file__).parent.parent.parent.parent / "frontend" / "src" / "components" / "LatexRenderer.css"
_LATEX_RENDERER_CSS: str = ""

try:
    _LATEX_RENDERER_CSS = _LATEX_RENDERER_CSS_PATH.read_text(encoding="utf-8")
except Exception as e:
    logger.warning(f"Could not read LatexRenderer.css for PDF generation: {e}")


class PDFRequest(BaseModel):
    html_body: str           # The rendered HTML content (output of renderLatexToHtml)
    title: str = "Document"
    word_count: Optional[int] = None
    date: Optional[str] = None
    models: Optional[str] = None
    outline: Optional[str] = None
    filename: str = "document"


def _build_html_document(req: PDFRequest) -> str:
    """
    Wrap the rendered HTML body in a complete standalone HTML document
    with all required CSS (KaTeX + LatexRenderer styles + print overrides).
    """
    # Metadata header block
    meta_parts = []
    if req.word_count:
        meta_parts.append(f"Word Count: {req.word_count:,}")
    if req.date:
        meta_parts.append(f"Generated: {req.date}")
    if req.models:
        meta_parts.append(f"AI Models: {req.models}")
    meta_line = " &nbsp;|&nbsp; ".join(meta_parts) if meta_parts else ""

    outline_section = ""
    if req.outline:
        outline_section = f"""
        <div class="outline-section">
            <h2 class="outline-heading">OUTLINE</h2>
            <pre class="outline-content">{_escape_html(req.outline)}</pre>
        </div>
        <hr class="section-divider"/>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{_escape_html(req.title)}</title>
<style>
/* ── KaTeX styles ── */
{_KATEX_CSS}

/* ── LatexRenderer component styles ── */
{_LATEX_RENDERER_CSS}

/* ── PDF-specific overrides ── */
*, *::before, *::after {{
    box-sizing: border-box;
}}

html, body {{
    margin: 0;
    padding: 0;
    background: #ffffff;
    color: #111111;
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.7;
}}

/* Override dark-theme colors for PDF printing */
.latex-rendered-content {{
    color: #111111 !important;
    background: transparent !important;
    font-size: 11pt;
    line-height: 1.75;
}}

.latex-rendered-content h1,
.latex-rendered-content h2,
.latex-rendered-content h3,
.latex-rendered-content h4,
.latex-rendered-content h5,
.latex-rendered-content h6 {{
    color: #111111 !important;
}}

.latex-chapter  {{ color: #111111 !important; border-bottom-color: #333 !important; }}
.latex-section  {{ border-bottom-color: #555 !important; }}
.latex-subsection {{ border-bottom-color: #888 !important; color: #222 !important; }}

.latex-display {{
    background: #f8f8f8 !important;
    border-left-color: #555 !important;
}}

.latex-theorem  {{ background: #fffef0 !important; border-color: #b8960a !important; }}
.latex-theorem strong {{ color: #7a6200 !important; }}
.latex-lemma    {{ background: #f0f7ff !important; border-color: #2c6fa8 !important; }}
.latex-lemma strong {{ color: #1a4f7a !important; }}
.latex-proposition {{ background: #f0fff4 !important; border-color: #2a7a3a !important; }}
.latex-proposition strong {{ color: #1a5a28 !important; }}
.latex-corollary {{ background: #fdf0ff !important; border-color: #7a3a9a !important; }}
.latex-corollary strong {{ color: #5a1a7a !important; }}
.latex-definition {{ background: #f0feff !important; border-color: #1a8a9a !important; }}
.latex-definition strong {{ color: #0a6a7a !important; }}
.latex-example {{ background: #fff8f0 !important; border-color: #a86020 !important; }}
.latex-example strong {{ color: #7a4010 !important; }}
.latex-remark, .latex-note {{ background: #f8f8f8 !important; border-color: #666 !important; }}
.latex-remark strong, .latex-note strong {{ color: #444 !important; }}
.latex-proof {{ background: #f5fff5 !important; border-color: #3a7a3a !important; }}
.latex-proof strong {{ color: #2a5a2a !important; }}
.latex-claim {{ background: #fff5f5 !important; border-color: #9a3a3a !important; }}
.latex-claim strong {{ color: #7a1a1a !important; }}
.latex-conjecture {{ background: #fdf5ff !important; border-color: #7a3a9a !important; }}
.latex-conjecture strong {{ color: #5a1a7a !important; }}
.latex-axiom, .latex-assumption {{ background: #fffff0 !important; border-color: #8a8a20 !important; }}
.latex-axiom strong, .latex-assumption strong {{ color: #5a5a00 !important; }}

.qed {{ color: #333 !important; }}

.latex-texttt, code {{
    background: #f0f0f0 !important;
    color: #1a5c1a !important;
}}

.latex-rendered-content pre {{
    background: #f5f5f5 !important;
    border-color: #ccc !important;
}}

.latex-table td, .latex-table th {{
    border-color: #bbb !important;
}}

.latex-table th {{
    background: #f0f0f0 !important;
    color: #111 !important;
}}

.latex-hrule {{
    border-top-color: #aaa !important;
}}

.latex-tikz-placeholder {{
    background: #eef3ff !important;
    border-color: #6495ed !important;
}}

.latex-tikz-placeholder .tikz-label {{
    color: #3060c0 !important;
}}

.latex-tikz-placeholder .tikz-code {{
    background: #e8eef8 !important;
    color: #222 !important;
}}

/* KaTeX: force black math for PDF */
.katex {{ color: #111 !important; }}
.katex-display {{ overflow-x: visible; }}
.katex .katex-mathml {{ display: none; }}
.katex .frac-line {{ border-bottom-color: #111 !important; }}

/* Metadata header */
.pdf-header {{
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #444;
}}

.pdf-header h1 {{
    margin: 0 0 0.5rem 0;
    font-size: 1.6rem;
    color: #111;
    line-height: 1.3;
}}

.pdf-header .pdf-meta {{
    font-size: 0.85rem;
    color: #555;
}}

/* Outline section */
.outline-section {{
    margin-bottom: 2rem;
    padding: 1rem 1.5rem;
    background: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 4px;
}}

.outline-heading {{
    margin: 0 0 1rem 0;
    font-size: 1.1rem;
    color: #333;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.5rem;
}}

.outline-content {{
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: Georgia, serif;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #222;
    margin: 0;
}}

.section-divider {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.5rem 0;
}}

/* Page wrapper */
.page-wrapper {{
    max-width: 170mm;
    margin: 0 auto;
    padding: 10mm 0;
}}

@media print {{
    .latex-toggle-bar, .latex-large-doc-banner {{
        display: none !important;
    }}
}}
</style>
</head>
<body>
<div class="page-wrapper">
    <div class="pdf-header">
        <h1>{_escape_html(req.title)}</h1>
        {f'<div class="pdf-meta">{meta_line}</div>' if meta_line else ""}
    </div>
    {outline_section}
    <div class="latex-rendered-content">
        {req.html_body}
    </div>
</div>
</body>
</html>"""


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _generate_pdf_sync(html: str) -> bytes:
    """
    Synchronous PDF generation via Playwright — runs in a thread pool executor.
    Returns raw PDF bytes.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load", timeout=60000)
            pdf_bytes = page.pdf(
                format="A4",
                margin={"top": "15mm", "right": "20mm", "bottom": "20mm", "left": "20mm"},
                print_background=True,
            )
            return pdf_bytes
        finally:
            browser.close()


@router.post("/pdf")
async def generate_pdf(req: PDFRequest):
    """
    Generate a PDF from rendered HTML content using Playwright (headless Chromium).

    Accepts the already-rendered HTML body (output of the frontend's renderLatexToHtml),
    wraps it in a complete HTML document with KaTeX CSS and LatexRenderer styles,
    then renders to PDF via Playwright running in a thread pool.

    The event loop is never blocked — PDF generation runs in a worker thread.
    """
    if system_config.generic_mode:
        raise HTTPException(
            status_code=501,
            detail="PDF generation unavailable in web mode. Use raw text download.",
        )

    if not req.html_body or not req.html_body.strip():
        raise HTTPException(status_code=400, detail="html_body is required and cannot be empty")

    try:
        html_document = _build_html_document(req)

        loop = asyncio.get_running_loop()
        pdf_bytes = await loop.run_in_executor(None, _generate_pdf_sync, html_document)

        safe_filename = (
            req.filename
            .replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
            [:200]
        ) or "document"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_filename}.pdf"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    except ImportError:
        logger.error("Playwright not installed. Run: python -m playwright install chromium")
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable. Run 'python -m playwright install chromium' and restart."
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
