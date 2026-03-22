/**
 * LatexRenderer - A robust component that renders text with LaTeX math expressions.
 * 
 * COMPREHENSIVE UPGRADE - Handles ANY model output including:
 * - Properly delimited math ($...$, $$...$$, \(...\), \[...\])
 * - UNWRAPPED LaTeX commands (auto-detected and wrapped)
 * - LaTeX environments (theorem, proof, definition, lemma, etc.)
 * - LaTeX sectioning commands (\section, \subsection, etc.)
 * - LaTeX text formatting (\textbf, \textit, \emph, etc.)
 * - Complex mathematical notation without delimiters
 * 
 * Uses KaTeX for fast client-side rendering with extensive error recovery.
 */
import React, { useState, useMemo, useCallback, useRef, useEffect, memo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import './LatexRenderer.css';
import DOMPurify from 'dompurify';

/**
 * Helper function to escape HTML special characters
 */
const escapeHtml = (text) => {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};

/**
 * DOMPurify configuration for XSS prevention.
 * Allows LaTeX/KaTeX/MathML rendering while blocking dangerous content.
 */
const DOMPURIFY_CONFIG = {
  ALLOWED_TAGS: [
    // Layout & structure
    'div', 'span', 'p', 'br', 'hr',
    // Text formatting
    'strong', 'b', 'em', 'i', 'u', 's', 'sub', 'sup', 'small',
    // Headings
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    // Lists
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    // Tables
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
    // KaTeX specific (MathML)
    'math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 
    'mfrac', 'mroot', 'msqrt', 'mtext', 'mspace', 'mtable', 'mtr', 'mtd',
    'annotation', 'annotation-xml',
    // SVG (for KaTeX rendering)
    'svg', 'path', 'line', 'rect', 'circle', 'g', 'use', 'defs', 'clippath',
  ],
  ALLOWED_ATTR: [
    // Standard attributes
    'class', 'id', 'title', 'style',
    // KaTeX/MathML attributes
    'mathvariant', 'encoding', 'xmlns', 'displaystyle', 'scriptlevel',
    'columnalign', 'rowalign', 'columnspacing', 'rowspacing', 'stretchy',
    'symmetric', 'fence', 'separator', 'lspace', 'rspace', 'accent',
    'accentunder', 'movablelimits', 'minsize', 'maxsize', 'width', 'height',
    // SVG attributes
    'd', 'viewBox', 'preserveAspectRatio', 'fill', 'stroke', 'stroke-width',
    'transform', 'x', 'y', 'dx', 'dy', 'x1', 'y1', 'x2', 'y2', 'r', 'cx', 'cy',
    'href', 'xlink:href', 'clip-path',
  ],
  ALLOW_DATA_ATTR: false,
  ALLOW_ARIA_ATTR: false,
  FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button',
                'textarea', 'select', 'option', 'link', 'style', 'base', 'meta'],
  FORBID_ATTR: ['onerror', 'onclick', 'onload', 'onmouseover', 'onfocus', 'onblur',
                'onchange', 'onsubmit', 'onkeydown', 'onkeyup', 'onmousedown', 'onmouseup'],
  SANITIZE_DOM: true,  // Prevent DOM clobbering attacks
};

/**
 * Extended macro definitions for common mathematical symbols and operators
 */
const extendedMacros = {
  // Number sets
  "\\R": "\\mathbb{R}",
  "\\N": "\\mathbb{N}",
  "\\Z": "\\mathbb{Z}",
  "\\Q": "\\mathbb{Q}",
  "\\C": "\\mathbb{C}",
  "\\F": "\\mathbb{F}",
  "\\P": "\\mathbb{P}",
  "\\A": "\\mathbb{A}",
  "\\H": "\\mathbb{H}",
  "\\K": "\\mathbb{K}",
  // Common groups and Lie theory
  "\\GL": "\\mathrm{GL}",
  "\\SL": "\\mathrm{SL}",
  "\\PSL": "\\mathrm{PSL}",
  "\\PGL": "\\mathrm{PGL}",
  "\\SO": "\\mathrm{SO}",
  "\\Sp": "\\mathrm{Sp}",
  "\\SU": "\\mathrm{SU}",
  "\\Op": "\\mathrm{O}",
  "\\Lie": "\\mathrm{Lie}",
  // Common operators and functions  
  "\\Aut": "\\mathrm{Aut}",
  "\\End": "\\mathrm{End}",
  "\\Hom": "\\mathrm{Hom}",
  "\\Gal": "\\mathrm{Gal}",
  "\\Spec": "\\mathrm{Spec}",
  "\\Proj": "\\mathrm{Proj}",
  "\\Ad": "\\mathrm{Ad}",
  "\\ad": "\\mathrm{ad}",
  "\\Val": "\\mathrm{Val}",
  "\\Ind": "\\mathrm{Ind}",
  "\\Res": "\\mathrm{Res}",
  "\\Tr": "\\mathrm{Tr}",
  "\\Vol": "\\mathrm{Vol}",
  "\\rank": "\\mathrm{rank}",
  "\\Out": "\\mathrm{Out}",
  "\\Inn": "\\mathrm{Inn}",
  "\\Cent": "\\mathrm{Cent}",
  "\\Stab": "\\mathrm{Stab}",
  "\\Orb": "\\mathrm{Orb}",
  "\\supp": "\\mathrm{supp}",
  "\\sgn": "\\mathrm{sgn}",
  "\\id": "\\mathrm{id}",
  "\\Id": "\\mathrm{Id}",
  "\\diag": "\\mathrm{diag}",
  "\\disc": "\\mathrm{disc}",
  // Langlands/automorphic forms specific
  "\\STr": "\\mathrm{STr}",
  "\\Irr": "\\mathrm{Irr}",
  "\\Rep": "\\mathrm{Rep}",
  "\\cusp": "\\mathrm{cusp}",
  "\\temp": "\\mathrm{temp}",
  "\\tw": "\\mathrm{tw}",
  // Common symbols
  "\\im": "\\mathrm{im}",
  "\\re": "\\mathrm{re}",
  "\\ker": "\\mathrm{ker}",
  "\\coker": "\\mathrm{coker}",
  "\\deg": "\\mathrm{deg}",
  "\\codim": "\\mathrm{codim}",
  "\\dim": "\\mathrm{dim}",
  "\\ord": "\\mathrm{ord}",
  "\\char": "\\mathrm{char}",
  "\\Char": "\\mathrm{Char}",
  // Algebraic geometry
  "\\Pic": "\\mathrm{Pic}",
  "\\Div": "\\mathrm{Div}",
  "\\Cl": "\\mathrm{Cl}",
  "\\NS": "\\mathrm{NS}",
  "\\Br": "\\mathrm{Br}",
  // Homological algebra
  "\\Ext": "\\mathrm{Ext}",
  "\\Tor": "\\mathrm{Tor}",
  "\\colim": "\\mathrm{colim}",
  "\\holim": "\\mathrm{holim}",
  "\\hocolim": "\\mathrm{hocolim}",
};

/**
 * List of LaTeX commands that indicate mathematical content
 * Used for auto-detection of unwrapped LaTeX
 */
const MATH_COMMAND_PATTERNS = [
  // Greek letters (lowercase)
  '\\\\alpha', '\\\\beta', '\\\\gamma', '\\\\delta', '\\\\epsilon', '\\\\varepsilon',
  '\\\\zeta', '\\\\eta', '\\\\theta', '\\\\vartheta', '\\\\iota', '\\\\kappa',
  '\\\\lambda', '\\\\mu', '\\\\nu', '\\\\xi', '\\\\omicron', '\\\\pi', '\\\\varpi',
  '\\\\rho', '\\\\varrho', '\\\\sigma', '\\\\varsigma', '\\\\tau', '\\\\upsilon',
  '\\\\phi', '\\\\varphi', '\\\\chi', '\\\\psi', '\\\\omega',
  // Greek letters (uppercase)
  '\\\\Gamma', '\\\\Delta', '\\\\Theta', '\\\\Lambda', '\\\\Xi', '\\\\Pi',
  '\\\\Sigma', '\\\\Upsilon', '\\\\Phi', '\\\\Psi', '\\\\Omega',
  // Math fonts
  '\\\\mathcal', '\\\\mathbb', '\\\\mathfrak', '\\\\mathscr', '\\\\mathbf',
  '\\\\mathrm', '\\\\mathsf', '\\\\mathtt', '\\\\mathit', '\\\\boldsymbol',
  // Text in math mode
  '\\\\text', '\\\\textrm', '\\\\textit', '\\\\textbf', '\\\\textsf', '\\\\texttt',
  // Operators
  '\\\\frac', '\\\\tfrac', '\\\\dfrac', '\\\\sqrt', '\\\\root',
  '\\\\sum', '\\\\prod', '\\\\coprod', '\\\\int', '\\\\oint', '\\\\iint', '\\\\iiint',
  '\\\\bigcup', '\\\\bigcap', '\\\\bigsqcup', '\\\\bigvee', '\\\\bigwedge', '\\\\bigoplus', '\\\\bigotimes',
  '\\\\lim', '\\\\limsup', '\\\\liminf', '\\\\sup', '\\\\inf', '\\\\max', '\\\\min',
  '\\\\log', '\\\\ln', '\\\\exp', '\\\\sin', '\\\\cos', '\\\\tan', '\\\\cot',
  '\\\\sec', '\\\\csc', '\\\\arcsin', '\\\\arccos', '\\\\arctan',
  '\\\\sinh', '\\\\cosh', '\\\\tanh', '\\\\coth',
  '\\\\det', '\\\\gcd', '\\\\lcm', '\\\\Pr', '\\\\hom', '\\\\arg',
  // Relations
  '\\\\leq', '\\\\geq', '\\\\neq', '\\\\approx', '\\\\equiv', '\\\\cong', '\\\\sim', '\\\\simeq',
  '\\\\subset', '\\\\supset', '\\\\subseteq', '\\\\supseteq', '\\\\subsetneq', '\\\\supsetneq',
  '\\\\in', '\\\\notin', '\\\\ni', '\\\\owns',
  '\\\\prec', '\\\\succ', '\\\\preceq', '\\\\succeq',
  '\\\\ll', '\\\\gg', '\\\\lll', '\\\\ggg',
  '\\\\perp', '\\\\parallel', '\\\\mid', '\\\\nmid',
  '\\\\vdash', '\\\\dashv', '\\\\models', '\\\\vDash', '\\\\Vdash',
  // Arrows
  '\\\\to', '\\\\rightarrow', '\\\\leftarrow', '\\\\leftrightarrow',
  '\\\\Rightarrow', '\\\\Leftarrow', '\\\\Leftrightarrow',
  '\\\\mapsto', '\\\\longmapsto', '\\\\hookrightarrow', '\\\\hookleftarrow',
  '\\\\twoheadrightarrow', '\\\\rightarrowtail',
  '\\\\uparrow', '\\\\downarrow', '\\\\updownarrow',
  '\\\\nearrow', '\\\\searrow', '\\\\nwarrow', '\\\\swarrow',
  '\\\\leadsto', '\\\\rightsquigarrow',
  // Brackets/delimiters
  '\\\\left', '\\\\right', '\\\\big', '\\\\Big', '\\\\bigg', '\\\\Bigg',
  '\\\\langle', '\\\\rangle', '\\\\lfloor', '\\\\rfloor', '\\\\lceil', '\\\\rceil',
  '\\\\lvert', '\\\\rvert', '\\\\lVert', '\\\\rVert',
  '\\\\{', '\\\\}', '\\\\|',
  // Accents and decorations
  '\\\\hat', '\\\\widehat', '\\\\tilde', '\\\\widetilde', '\\\\bar', '\\\\overline',
  '\\\\vec', '\\\\overrightarrow', '\\\\overleftarrow',
  '\\\\dot', '\\\\ddot', '\\\\dddot', '\\\\acute', '\\\\grave', '\\\\breve', '\\\\check',
  '\\\\underline', '\\\\overbrace', '\\\\underbrace',
  // Spacing
  '\\\\quad', '\\\\qquad', '\\\\,', '\\\\;', '\\\\!',
  // Misc symbols
  '\\\\infty', '\\\\partial', '\\\\nabla', '\\\\forall', '\\\\exists', '\\\\nexists',
  '\\\\emptyset', '\\\\varnothing', '\\\\setminus', '\\\\times', '\\\\otimes', '\\\\oplus',
  '\\\\circ', '\\\\bullet', '\\\\cdot', '\\\\cdots', '\\\\ldots', '\\\\vdots', '\\\\ddots',
  '\\\\star', '\\\\ast', '\\\\dagger', '\\\\ddagger',
  '\\\\pm', '\\\\mp', '\\\\div', '\\\\wedge', '\\\\vee', '\\\\cap', '\\\\cup',
  '\\\\neg', '\\\\lnot', '\\\\land', '\\\\lor',
  '\\\\prime', '\\\\backprime',
  '\\\\aleph', '\\\\beth', '\\\\gimel',
  '\\\\ell', '\\\\wp', '\\\\Re', '\\\\Im', '\\\\hbar', '\\\\hslash',
  '\\\\triangle', '\\\\square', '\\\\Diamond', '\\\\Box',
  '\\\\clubsuit', '\\\\diamondsuit', '\\\\heartsuit', '\\\\spadesuit',
  // Stacking/matrices
  '\\\\stackrel', '\\\\overset', '\\\\underset', '\\\\xrightarrow', '\\\\xleftarrow',
  '\\\\binom', '\\\\choose', '\\\\atop',
  '\\\\matrix', '\\\\pmatrix', '\\\\bmatrix', '\\\\vmatrix', '\\\\Vmatrix',
  '\\\\array', '\\\\cases',
  // Custom operators from our macros
  '\\\\GL', '\\\\SL', '\\\\PSL', '\\\\PGL', '\\\\SO', '\\\\Sp', '\\\\SU',
  '\\\\Aut', '\\\\End', '\\\\Hom', '\\\\Gal', '\\\\Spec', '\\\\Proj',
  '\\\\Tr', '\\\\Ad', '\\\\ad', '\\\\Val', '\\\\Ind', '\\\\Res',
  '\\\\Cent', '\\\\Stab', '\\\\Orb', '\\\\STr', '\\\\Irr', '\\\\Rep',
  '\\\\Ext', '\\\\Tor', '\\\\Pic', '\\\\Div', '\\\\Cl', '\\\\Br',
];

/**
 * Find the matching closing brace for a LaTeX command
 */
const findMatchingBrace = (text, startPos) => {
  let braceCount = 1;
  let i = startPos;
  while (i < text.length && braceCount > 0) {
    if (text[i] === '\\' && i + 1 < text.length) {
      i += 2; // Skip escaped char
      continue;
    }
    if (text[i] === '{') braceCount++;
    if (text[i] === '}') braceCount--;
    i++;
  }
  return braceCount === 0 ? i - 1 : -1;
};

/**
 * Find extent of math expression starting at position
 * Returns the end position of the math expression
 */
const findMathExtent = (text, startPos) => {
  let i = startPos;
  let braceDepth = 0;
  let parenDepth = 0;
  
  while (i < text.length) {
    const char = text[i];
    const prevChar = i > 0 ? text[i - 1] : '';
    
    // Skip escaped characters
    if (prevChar === '\\') {
      i++;
      continue;
    }
    
    // Track braces and parentheses
    if (char === '{') braceDepth++;
    if (char === '}') {
      braceDepth--;
      if (braceDepth < 0) break; // Unmatched brace
    }
    if (char === '(') parenDepth++;
    if (char === ')') {
      parenDepth--;
      if (parenDepth < 0 && braceDepth === 0) {
        // Include the closing paren if it's part of function notation
        i++;
        break;
      }
    }
    
    // Check for natural breaks (end of math expression)
    if (braceDepth === 0 && parenDepth === 0) {
      // Break on sentence-ending punctuation followed by space
      if ((char === '.' || char === ',' || char === ';' || char === ':') && 
          (i + 1 >= text.length || /\s/.test(text[i + 1]))) {
        break;
      }
      // Break on double newline
      if (char === '\n' && i + 1 < text.length && text[i + 1] === '\n') {
        break;
      }
      // Break if we hit a word character after whitespace (new sentence)
      if (/\s/.test(char) && i + 1 < text.length && /[A-Z]/.test(text[i + 1])) {
        // Check if it's actually a new sentence vs math continuation
        const ahead = text.substring(i + 1, Math.min(i + 20, text.length));
        if (!/^[A-Z]\s*[=<>]/.test(ahead) && !/^[A-Z]_/.test(ahead)) {
          break;
        }
      }
    }
    
    i++;
  }
  
  return i;
};

/**
 * Auto-detect and wrap unwrapped LaTeX math expressions
 * 
 * IMPORTANT: Uses robust placeholder markers that survive DOM operations.
 * Previous implementation used \x00 null characters which get stripped by
 * DOMPurify and browser DOM operations, causing cascading render failures.
 */
const autoWrapMath = (text) => {
  if (!text) return text;
  
  // Skip expensive regex processing if content already has LaTeX math delimiters
  // This prevents catastrophic backtracking on properly-formatted LLM output
  if (text.includes('$') || text.includes('\\[') || text.includes('\\(')) {
    return text;
  }
  
  let result = text;
  
  // Build a comprehensive regex to find unwrapped LaTeX commands
  // This matches LaTeX commands that appear outside of existing delimiters
  
  // First, mark existing delimited regions to avoid double-wrapping
  // CRITICAL: Use markers that survive DOM operations (no null chars!)
  // Format: ⟦MATH⟧index⟦/MATH⟧ - uses Unicode brackets unlikely in math content
  const PLACEHOLDER_START = '⟦MATHRGN⟧';
  const PLACEHOLDER_END = '⟦/MATHRGN⟧';
  const delimitedRegions = [];
  let regionIndex = 0;
  
  // Protect existing delimited math
  const delimiterPatterns = [
    /\$\$[\s\S]*?\$\$/g,           // $$...$$
    /\\\[[\s\S]*?\\\]/g,           // \[...\]
    /(?<!\$)\$(?!\$)(?:[^$\\]|\\.)+?\$(?!\$)/g,  // $...$
    /\\\([\s\S]*?\\\)/g,           // \(...\)
  ];
  
  delimiterPatterns.forEach(pattern => {
    result = result.replace(pattern, (match) => {
      const placeholder = `${PLACEHOLDER_START}${regionIndex++}${PLACEHOLDER_END}`;
      delimitedRegions.push(match);
      return placeholder;
    });
  });
  
  // Now process unwrapped LaTeX commands
  // Pattern matches: \command followed by optional arguments/subscripts/superscripts
  const unwrappedMathPattern = new RegExp(
    // Match a backslash followed by a command name
    '(' + MATH_COMMAND_PATTERNS.join('|') + ')' +
    // Followed by optional arguments, subscripts, superscripts, etc.
    '(?:[_{^](?:\\{[^}]*\\}|[^\\s{},;.!?)]))*' +
    // And potentially more content in braces
    '(?:\\{[^}]*\\})*' +
    // And trailing subscripts/superscripts
    '(?:[_{^](?:\\{[^}]*\\}|[^\\s{},;.!?]))*',
    'g'
  );
  
  // Find and wrap unwrapped math expressions
  result = result.replace(unwrappedMathPattern, (match, cmd) => {
    // Skip if it's a single backslash command that doesn't make sense alone
    if (match.trim().length < 3) return match;
    return `$${match}$`;
  });
  
  // NOTE: We intentionally do NOT auto-wrap single-letter subscript patterns like "G_x"
  // because they often appear in regular prose (e.g., "Let G_x be a set")
  // The model should use proper delimiters. Only wrap when there's clear LaTeX context.
  
  // Only wrap subscript/superscript patterns when they're adjacent to other LaTeX commands
  // This is more conservative - only wrap when we see clear math context nearby
  result = result.replace(
    /(\$[^$]+\$)\s*([A-Za-z])([_^])(\{[^}]+\}|[A-Za-z0-9])/g,
    (match, mathBefore, letter, op, sub) => {
      return `${mathBefore} $${letter}${op}${sub}$`;
    }
  );
  
  // Clean up consecutive $ signs from over-wrapping
  result = result.replace(/\$\s*\$/g, '');
  result = result.replace(/\$\$([^$]+)\$\$/g, (match, inner) => {
    // If this was supposed to be inline, keep as inline
    if (!inner.includes('\n') && inner.length < 100) {
      return `$${inner}$`;
    }
    return match;
  });
  
  // Restore the protected regions
  // CRITICAL: Restore in reverse order to handle any nested placeholders correctly
  for (let i = delimitedRegions.length - 1; i >= 0; i--) {
    const placeholder = `${PLACEHOLDER_START}${i}${PLACEHOLDER_END}`;
    result = result.replace(placeholder, delimitedRegions[i]);
  }
  
  // Safety check: warn if any placeholders remain (indicates a bug)
  if (result.includes(PLACEHOLDER_START)) {
    console.error('LaTeX placeholder restoration failed - some placeholders remain:', 
      result.match(/⟦MATHRGN⟧\d+⟦\/MATHRGN⟧/g));
  }
  
  return result;
};

/**
 * Replace LaTeX sectioning commands while respecting nested braces
 */
const replaceSectionCommand = (text, command, tag, endTag) => {
  const regex = new RegExp(`\\\\${command}\\s*\\{`, 'g');
  let result = '';
  let lastIndex = 0;
  let match;
  
  while ((match = regex.exec(text)) !== null) {
    const openBracePos = match.index + match[0].length - 1;
    const closeBracePos = findMatchingBrace(text, openBracePos + 1);
    
    if (closeBracePos !== -1) {
      const content = text.substring(openBracePos + 1, closeBracePos);
      result += text.substring(lastIndex, match.index);
      result += `${tag}${content}${endTag}`;
      lastIndex = closeBracePos + 1;
    }
  }
  result += text.substring(lastIndex);
  return result;
};

/**
 * Decode HTML entities that may have been encoded upstream.
 * This fixes issues where content arrives with &#x27; instead of ' etc.
 */
const decodeHtmlEntities = (text) => {
  if (!text) return text;
  
  // Use a textarea to decode HTML entities properly
  const textarea = document.createElement('textarea');
  textarea.innerHTML = text;
  let decoded = textarea.textContent;
  
  // Also handle common named entities that might not be decoded
  decoded = decoded
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'")
    .replace(/&apos;/g, "'");
  
  return decoded;
};

/**
 * Clean up TikZ diagram content for display as code block
 */
const cleanTikzContent = (content) => {
  return content.trim()
    .replace(/&lt;br\/&gt;/g, '\n')   // Fix HTML-encoded line breaks
    .replace(/&amp;amp;/g, '&')       // Fix double-encoded ampersands
    .replace(/&amp;/g, '&')           // Fix encoded ampersands
    .replace(/<br\s*\/?>/g, '\n');    // Fix actual HTML line breaks
};

/**
 * Safely render LaTeX with KaTeX, handling errors gracefully
 * @param {string} latex - The LaTeX content to render
 * @param {boolean} displayMode - Whether to render in display mode
 * @param {string} originalMatch - The original matched string (for error display)
 */
const renderKatexSafely = (latex, displayMode, originalMatch) => {
  const trimmedLatex = latex.trim();
  
  // Skip empty content
  if (!trimmedLatex) {
    return displayMode ? '' : '';
  }
  
  try {
    const html = katex.renderToString(trimmedLatex, {
      displayMode: displayMode,
      throwOnError: false,
      strict: false,
      trust: true,
      macros: extendedMacros,
      maxExpand: 5000,   // Balanced: high enough for complex expressions, safe from stack overflow
      maxSize: 500
    });
    
    if (displayMode) {
      return `<div class="latex-display">${html}</div>`;
    }
    return `<span class="latex-inline">${html}</span>`;
  } catch (e) {
    // Log error for debugging
    const errorType = displayMode ? 'display' : 'inline';
    console.warn(`KaTeX ${errorType} error:`, e.message, 'Content:', trimmedLatex.substring(0, 100));
    
    // Create error display
    const errorClass = displayMode ? 'latex-error latex-display-error' : 'latex-error';
    const tag = displayMode ? 'div' : 'span';
    return `<${tag} class="${errorClass}" title="${escapeHtml(e.message)}">${escapeHtml(originalMatch)}</${tag}>`;
  }
};

/**
 * Process LaTeX theorem-like environments
 */
const processTheoremEnvironments = (text) => {
  let result = text;
  
  // FIRST: Handle tikzcd and other TikZ environments that KaTeX doesn't support
  // These are displayed as styled code blocks since client-side TikZ rendering isn't possible
  // CRITICAL: Must also handle cases where tikz is wrapped in math delimiters like \[...\] or $$...$$
  const unsupportedTikzEnvs = ['tikzcd', 'tikzpicture', 'pgfpicture'];
  
  unsupportedTikzEnvs.forEach(envName => {
    // Pattern 1: tikz wrapped in \[...\] display math delimiters
    const displayBracketPattern = new RegExp(
      `\\\\\\[\\s*\\\\begin\\{${envName}\\}([\\s\\S]*?)\\\\end\\{${envName}\\}\\s*\\\\\\]`,
      'gi'
    );
    result = result.replace(displayBracketPattern, (match, content) => {
      const cleanContent = cleanTikzContent(content);
      return `<div class="latex-tikz-placeholder"><div class="tikz-label">[Commutative Diagram - ${envName}]</div><pre class="tikz-code">${escapeHtml(cleanContent)}</pre></div>`;
    });
    
    // Pattern 2: tikz wrapped in $$...$$ display math delimiters
    const displayDollarPattern = new RegExp(
      `\\$\\$\\s*\\\\begin\\{${envName}\\}([\\s\\S]*?)\\\\end\\{${envName}\\}\\s*\\$\\$`,
      'gi'
    );
    result = result.replace(displayDollarPattern, (match, content) => {
      const cleanContent = cleanTikzContent(content);
      return `<div class="latex-tikz-placeholder"><div class="tikz-label">[Commutative Diagram - ${envName}]</div><pre class="tikz-code">${escapeHtml(cleanContent)}</pre></div>`;
    });
    
    // Pattern 3: Standalone tikz environment (not wrapped in math delimiters)
    const standalonePattern = new RegExp(
      `\\\\begin\\{${envName}\\}([\\s\\S]*?)\\\\end\\{${envName}\\}`,
      'gi'
    );
    result = result.replace(standalonePattern, (match, content) => {
      const cleanContent = cleanTikzContent(content);
      return `<div class="latex-tikz-placeholder"><div class="tikz-label">[Commutative Diagram - ${envName}]</div><pre class="tikz-code">${escapeHtml(cleanContent)}</pre></div>`;
    });
  });
  
  // Define theorem-like environments and their styling
  const environments = [
    { name: 'theorem', label: 'Theorem', class: 'latex-theorem' },
    { name: 'lemma', label: 'Lemma', class: 'latex-lemma' },
    { name: 'proposition', label: 'Proposition', class: 'latex-proposition' },
    { name: 'corollary', label: 'Corollary', class: 'latex-corollary' },
    { name: 'definition', label: 'Definition', class: 'latex-definition' },
    { name: 'example', label: 'Example', class: 'latex-example' },
    { name: 'remark', label: 'Remark', class: 'latex-remark' },
    { name: 'note', label: 'Note', class: 'latex-note' },
    { name: 'proof', label: 'Proof', class: 'latex-proof' },
    { name: 'claim', label: 'Claim', class: 'latex-claim' },
    { name: 'conjecture', label: 'Conjecture', class: 'latex-conjecture' },
    { name: 'axiom', label: 'Axiom', class: 'latex-axiom' },
    { name: 'assumption', label: 'Assumption', class: 'latex-assumption' },
  ];
  
  environments.forEach(({ name, label, class: className }) => {
    // Match both \begin{env}...\end{env} and standalone markers
    const envPattern = new RegExp(
      `\\\\begin\\{${name}\\}(?:\\[([^\\]]+)\\])?([\\s\\S]*?)\\\\end\\{${name}\\}`,
      'gi'
    );
    
    result = result.replace(envPattern, (match, optionalTitle, content) => {
      const title = optionalTitle ? ` (${optionalTitle})` : '';
      const qedSymbol = name === 'proof' ? '<span class="qed">∎</span>' : '';
      return `<div class="${className}"><strong>${label}${title}.</strong> ${content.trim()}${qedSymbol}</div>`;
    });
  });
  
  // Handle equation environment
  result = result.replace(
    /\\begin\{equation\}([\s\S]*?)\\end\{equation\}/gi,
    (match, content) => `$$${content.trim()}$$`
  );
  
  // Handle equation* environment (no numbering)
  result = result.replace(
    /\\begin\{equation\*\}([\s\S]*?)\\end\{equation\*\}/gi,
    (match, content) => `$$${content.trim()}$$`
  );
  
  // Handle align environment
  result = result.replace(
    /\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}/gi,
    (match, content) => {
      // Convert align to gathered for KaTeX compatibility
      const processed = content.replace(/&/g, '').replace(/\\\\/g, '\\\\');
      return `$$\\begin{gathered}${processed}\\end{gathered}$$`;
    }
  );
  
  // Handle gather environment
  result = result.replace(
    /\\begin\{gather\*?\}([\s\S]*?)\\end\{gather\*?\}/gi,
    (match, content) => `$$\\begin{gathered}${content.trim()}\\end{gathered}$$`
  );
  
  // Handle split environment
  result = result.replace(
    /\\begin\{split\}([\s\S]*?)\\end\{split\}/gi,
    (match, content) => `$$\\begin{aligned}${content.trim()}\\end{aligned}$$`
  );
  
  // Handle multline environment
  result = result.replace(
    /\\begin\{multline\*?\}([\s\S]*?)\\end\{multline\*?\}/gi,
    (match, content) => `$$\\begin{gathered}${content.replace(/\\\\/g, '\\\\')}\\end{gathered}$$`
  );
  
  return result;
};

/**
 * Main function to parse and render LaTeX to HTML
 */
const renderLatexToHtml = (text) => {
  if (!text) return '';
  
  // CRITICAL FIX: Decode HTML entities FIRST before any LaTeX processing
  // Content may arrive with &#x27; instead of ', &amp; instead of &, etc.
  // which corrupts the LaTeX and causes parsing errors
  let result = decodeHtmlEntities(text);
  
  // FIX: LLM output sometimes contains literal \n characters inside math environments
  // These are not valid LaTeX and cause rendering errors. Strip them.
  // Pattern: \[\n...\] or $$\n...$$ - remove the literal \n at start/end of math blocks
  result = result.replace(/\\\[\s*\\n/g, '\\[');  // \[\n → \[
  result = result.replace(/\\n\s*\\\]/g, '\\]');  // \n\] → \]
  result = result.replace(/\$\$\s*\\n/g, '$$');   // $$\n → $$
  result = result.replace(/\\n\s*\$\$/g, '$$');   // \n$$ → $$
  // Also strip \n that appears mid-equation (literal backslash-n, not actual newline)
  result = result.replace(/\\n(?=[^a-zA-Z])/g, ' '); // \n followed by non-letter → space
  
  // FIX: LLM sometimes outputs \igl( and \igr) which are NOT valid LaTeX
  // These should be \bigl( and \bigr) respectively
  result = result.replace(/\\igl\(/g, '\\bigl(');
  result = result.replace(/\\igr\)/g, '\\bigr)');
  result = result.replace(/\\igl\[/g, '\\bigl[');
  result = result.replace(/\\igr\]/g, '\\bigr]');
  result = result.replace(/\\igl\{/g, '\\bigl\\{');
  result = result.replace(/\\igr\}/g, '\\bigr\\}');
  // Also handle generic \igl and \igr without immediate delimiter
  result = result.replace(/\\igl(?![a-zA-Z])/g, '\\bigl');
  result = result.replace(/\\igr(?![a-zA-Z])/g, '\\bigr');
  // FIX: \ig| should be \big| (missing 'b')
  result = result.replace(/\\ig\|/g, '\\big|');
  
  // FIX: LLM sometimes outputs double-backslash for LaTeX commands (\\underline instead of \underline)
  // This happens due to JSON escaping issues. Fix common math commands.
  result = result.replace(/\\\\underline\{/g, '\\underline{');
  result = result.replace(/\\\\overline\{/g, '\\overline{');
  result = result.replace(/\\\\widehat\{/g, '\\widehat{');
  result = result.replace(/\\\\widetilde\{/g, '\\widetilde{');
  result = result.replace(/\\\\mathcal\{/g, '\\mathcal{');
  result = result.replace(/\\\\mathbb\{/g, '\\mathbb{');
  result = result.replace(/\\\\mathrm\{/g, '\\mathrm{');
  result = result.replace(/\\\\mathbf\{/g, '\\mathbf{');
  result = result.replace(/\\\\mathit\{/g, '\\mathit{');
  result = result.replace(/\\\\mathsf\{/g, '\\mathsf{');
  result = result.replace(/\\\\operatorname\{/g, '\\operatorname{');
  
  // FIX: \rho sometimes gets corrupted to \ho (the \r is consumed as carriage return)
  // This pattern appears as backslash + newline/space + "ho_" or backslash + newline/space + "ho{"
  // Also handle cases where it appears mid-text as "\ho_" which should be "\rho_"
  result = result.replace(/\\\s*\nho_/g, '\\rho_');
  result = result.replace(/\\\s*\nho\{/g, '\\rho{');
  result = result.replace(/\\\s*\nho\(/g, '\\rho(');
  result = result.replace(/\\\s*\nho\|/g, '\\rho|');
  // Also handle the case where \ho appears directly (no split)
  result = result.replace(/\\ho_/g, '\\rho_');
  result = result.replace(/\\ho\{/g, '\\rho{');
  result = result.replace(/\\ho\(/g, '\\rho(');
  result = result.replace(/\\ho\|/g, '\\rho|');
  
  // FIX: \to sometimes gets corrupted because \t is interpreted as tab character
  // The source has backslash + tab + "o" which should be \to
  // In regex, \t matches the actual tab character (0x09)
  result = result.replace(/\\\to/g, '\\to');
  
  // Step 1: Auto-wrap unwrapped LaTeX commands
  result = autoWrapMath(result);
  
  // Step 2: Process theorem-like environments BEFORE math rendering
  result = processTheoremEnvironments(result);
  
  // Step 3: Handle LaTeX sectioning commands
  result = replaceSectionCommand(result, 'chapter', '<h1 class="latex-chapter">', '</h1>');
  result = replaceSectionCommand(result, 'subsubsection', '<h4 class="latex-subsubsection">', '</h4>');
  result = replaceSectionCommand(result, 'subsection', '<h3 class="latex-subsection">', '</h3>');
  result = replaceSectionCommand(result, 'section', '<h2 class="latex-section">', '</h2>');
  result = replaceSectionCommand(result, 'paragraph', '<h5 class="latex-paragraph">', '</h5>');
  
  // Step 4: Handle \cite commands (with nested brace support)
  {
    const citeRegex = /\\cite\s*\{/g;
    let citeResult = '';
    let citeLastIndex = 0;
    let citeMatch;
    
    while ((citeMatch = citeRegex.exec(result)) !== null) {
      const openBracePos = citeMatch.index + citeMatch[0].length - 1;
      const closeBracePos = findMatchingBrace(result, openBracePos + 1);
      
      if (closeBracePos !== -1) {
        const content = result.substring(openBracePos + 1, closeBracePos);
        citeResult += result.substring(citeLastIndex, citeMatch.index);
        citeResult += `[${content}]`;
        citeLastIndex = closeBracePos + 1;
      }
    }
    citeResult += result.substring(citeLastIndex);
    result = citeResult || result;
  }
  
  // Step 5: Handle text formatting commands (outside math mode)
  // Uses findMatchingBrace to properly handle nested braces like \textbf{hello \textit{world}}
  const textCommands = [
    { cmd: 'textbf', tag: 'strong', endTag: '</strong>' },
    { cmd: 'textit', tag: 'em', endTag: '</em>' },
    { cmd: 'texttt', tag: 'code class="latex-texttt"', endTag: '</code>' },
    { cmd: 'emph', tag: 'em', endTag: '</em>' },
    { cmd: 'underline', tag: 'u', endTag: '</u>' },
    { cmd: 'textsc', tag: 'span class="latex-smallcaps"', endTag: '</span>' },
    { cmd: 'textsf', tag: 'span class="latex-sans"', endTag: '</span>' },
  ];
  
  // Process text commands with proper nested brace handling
  textCommands.forEach(({ cmd, tag, endTag }) => {
    const regex = new RegExp(`\\\\${cmd}\\s*\\{`, 'g');
    let newResult = '';
    let lastIndex = 0;
    let match;
    
    while ((match = regex.exec(result)) !== null) {
      const openBracePos = match.index + match[0].length - 1;
      const closeBracePos = findMatchingBrace(result, openBracePos + 1);
      
      if (closeBracePos !== -1) {
        const content = result.substring(openBracePos + 1, closeBracePos);
        newResult += result.substring(lastIndex, match.index);
        newResult += `<${tag}>${content}${endTag}`;
        lastIndex = closeBracePos + 1;
      }
    }
    newResult += result.substring(lastIndex);
    result = newResult || result;
  });
  
  // Step 6: Handle footnotes (with nested brace support)
  {
    const footnoteRegex = /\\footnote\s*\{/g;
    let footnoteResult = '';
    let footnoteLastIndex = 0;
    let footnoteMatch;
    
    while ((footnoteMatch = footnoteRegex.exec(result)) !== null) {
      const openBracePos = footnoteMatch.index + footnoteMatch[0].length - 1;
      const closeBracePos = findMatchingBrace(result, openBracePos + 1);
      
      if (closeBracePos !== -1) {
        const content = result.substring(openBracePos + 1, closeBracePos);
        footnoteResult += result.substring(footnoteLastIndex, footnoteMatch.index);
        footnoteResult += `<sup class="latex-footnote">[${content}]</sup>`;
        footnoteLastIndex = closeBracePos + 1;
      }
    }
    footnoteResult += result.substring(footnoteLastIndex);
    result = footnoteResult || result;
  }
  
  // Step 7: Handle LaTeX list environments
  result = result.replace(/\\begin\{enumerate\}([\s\S]*?)\\end\{enumerate\}/g, (match, content) => {
    let listContent = content.replace(/\\item\s*/g, '</li><li>').trim();
    if (listContent.startsWith('</li>')) listContent = listContent.substring(5);
    if (!listContent.endsWith('</li>')) listContent += '</li>';
    return `<ol class="latex-enumerate">${listContent}</ol>`;
  });
  
  result = result.replace(/\\begin\{itemize\}([\s\S]*?)\\end\{itemize\}/g, (match, content) => {
    let listContent = content.replace(/\\item\s*/g, '</li><li>').trim();
    if (listContent.startsWith('</li>')) listContent = listContent.substring(5);
    if (!listContent.endsWith('</li>')) listContent += '</li>';
    return `<ul class="latex-itemize">${listContent}</ul>`;
  });
  
  result = result.replace(/\\begin\{description\}([\s\S]*?)\\end\{description\}/g, (match, content) => {
    let listContent = content.replace(/\\item\s*\[([^\]]+)\]\s*/g, '</dd><dt>$1</dt><dd>').trim();
    if (listContent.startsWith('</dd>')) listContent = listContent.substring(5);
    if (!listContent.endsWith('</dd>')) listContent += '</dd>';
    return `<dl class="latex-description">${listContent}</dl>`;
  });
  
  // Step 8: Handle tables
  result = result.replace(
    /\\begin\{(?:table|tabular)\}(?:\[[^\]]*\])?(?:\{[^}]*\})?([\s\S]*?)\\end\{(?:table|tabular)\}/gi,
    (match, content) => {
      // Simplified table handling - convert to basic HTML table
      const rows = content.split('\\\\').filter(r => r.trim());
      const htmlRows = rows.map(row => {
        const cells = row.split('&').map(cell => 
          `<td>${cell.replace(/\\hline/g, '').trim()}</td>`
        ).join('');
        return `<tr>${cells}</tr>`;
      }).join('');
      return `<table class="latex-table">${htmlRows}</table>`;
    }
  );
  
  // Step 9: Standalone \item commands
  result = result.replace(/\\item\s+/g, '• ');
  
  // Step 10: Handle QED symbol
  result = result.replace(/\\qed/g, '<span class="qed">∎</span>');
  result = result.replace(/\\blacksquare/g, '<span class="qed">■</span>');
  result = result.replace(/\\square/g, '<span class="qed">□</span>');
  
  // Step 11: Now render math with KaTeX
  // CRITICAL: Must happen BEFORE line break conversion because \\ is valid LaTeX
  // inside math mode (e.g., in aligned, matrix environments)
  // Display math patterns: $$...$$ or \[...\]
  const displayMathPatterns = [
    /\$\$([\s\S]*?)\$\$/g,
    /\\\[([\s\S]*?)\\\]/g
  ];
  
  displayMathPatterns.forEach(pattern => {
    result = result.replace(pattern, (match, latex) => {
      // Skip if the content looks like it contains HTML (placeholder that wasn't removed)
      if (latex.includes('<div') || latex.includes('class=')) {
        return match; // Return unchanged - let it display as-is
      }
      return renderKatexSafely(latex, true, match);
    });
  });
  
  // Inline math patterns: $...$ or \(...\)
  const inlineMathPatterns = [
    /(?<!\$)\$(?!\$)((?:[^$\\]|\\.|\\)+?)\$(?!\$)/g,
    /\\\(([\s\S]*?)\\\)/g
  ];
  
  inlineMathPatterns.forEach(pattern => {
    result = result.replace(pattern, (match, latex) => {
      // Skip if the content looks like it contains HTML
      if (latex.includes('<div') || latex.includes('class=')) {
        return match;
      }
      return renderKatexSafely(latex, false, match);
    });
  });
  
  // Step 12: Handle line breaks (AFTER KaTeX - \\ is valid inside math mode)
  // Now that math has been rendered, remaining \\ are text line breaks
  // But we must NOT touch backslashes inside HTML attributes (SVG paths use them)
  result = result.replace(/\\\\(?![^<]*>)/g, '<br/>');
  result = result.replace(/\\newline(?![^<]*>)/g, '<br/>');
  result = result.replace(/\\linebreak(?![^<]*>)/g, '<br/>');
  
  // Step 13: Handle horizontal rules
  result = result.replace(/\\hrule/g, '<hr class="latex-hrule"/>');
  result = result.replace(/\\rule\{[^}]*\}\{[^}]*\}/g, '<hr class="latex-hrule"/>');
  
  // Step 14: Convert remaining newlines to <br>
  // CRITICAL: Do NOT replace newlines inside HTML tags - this corrupts KaTeX SVG paths!
  // KaTeX generates SVG elements with newlines in path data like <path d="M1 2\n3 4"/>
  // Replacing \n with <br/> inside these paths breaks SVG rendering completely.
  // 
  // Safe approach: Use a function to only replace newlines in TEXT CONTENT, not inside tags.
  // This iterates through the string and only replaces \n when we're outside any HTML tag.
  {
    const parts = [];
    let inTag = false;
    let segStart = 0;
    for (let i = 0; i < result.length; i++) {
      const char = result[i];
      if (char === '<') {
        inTag = true;
      } else if (char === '>') {
        inTag = false;
      } else if (char === '\n' && !inTag) {
        if (i > segStart) parts.push(result.slice(segStart, i));
        parts.push('<br/>');
        segStart = i + 1;
      }
    }
    if (segStart < result.length) parts.push(result.slice(segStart));
    result = parts.join('');
  }
  
  // Step 15: Clean up multiple consecutive <br> tags
  result = result.replace(/(<br\s*\/?>\s*){3,}/g, '<br/><br/>');
  
  return result;
};

/**
 * Fast string hash for stable React keys. Not cryptographic — just needs to
 * produce different values for different chunk contents so React reconciles correctly.
 */
const simpleHash = (str) => {
  let h = 0;
  const len = Math.min(str.length, 128);
  for (let i = 0; i < len; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0;
  }
  return h.toString(36);
};

/**
 * Split content into chunks at section boundaries for progressive rendering.
 * Splits at double-newlines, section headers (Roman numerals, "Abstract", etc.),
 * and hard-coded markers. Ensures math environments are not split mid-expression.
 */
const CHUNK_TARGET_SIZE = 3000;

const splitIntoChunks = (text) => {
  if (!text || text.length <= CHUNK_TARGET_SIZE) return [text];

  const sectionPattern = /\n(?=(?:#{1,6}\s|[IVXLCDM]+\.\s|Abstract|Introduction|Conclusion|(?:\\section|\\subsection|\\chapter)\s*\{|\[HARD CODED))/gi;

  const chunks = [];
  let lastIdx = 0;

  let match;
  while ((match = sectionPattern.exec(text)) !== null) {
    if (match.index - lastIdx >= CHUNK_TARGET_SIZE * 0.3) {
      chunks.push(text.slice(lastIdx, match.index));
      lastIdx = match.index + 1;
    }
  }
  if (lastIdx < text.length) {
    chunks.push(text.slice(lastIdx));
  }

  const result = [];
  for (const chunk of chunks) {
    if (chunk.length <= CHUNK_TARGET_SIZE * 2) {
      result.push(chunk);
      continue;
    }
    let remaining = chunk;
    while (remaining.length > CHUNK_TARGET_SIZE * 2) {
      let splitAt = -1;
      const searchEnd = Math.min(remaining.length, CHUNK_TARGET_SIZE * 1.5);
      for (let i = CHUNK_TARGET_SIZE * 0.5; i < searchEnd; i++) {
        if (remaining[i] === '\n' && i + 1 < remaining.length && remaining[i + 1] === '\n') {
          splitAt = i;
          break;
        }
      }
      if (splitAt === -1) splitAt = Math.round(CHUNK_TARGET_SIZE);
      result.push(remaining.slice(0, splitAt));
      remaining = remaining.slice(splitAt);
    }
    if (remaining.length > 0) result.push(remaining);
  }

  return result.length > 0 ? result : [text];
};

/**
 * A single rendered chunk with IntersectionObserver-based lazy rendering.
 * Only runs renderLatexToHtml when the chunk scrolls into or near the viewport.
 */
const RenderedChunk = memo(({ text, index }) => {
  const containerRef = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { rootMargin: '600px 0px' }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const renderedHtml = useMemo(() => {
    if (!isVisible || !text) return null;
    const rawHtml = renderLatexToHtml(text);
    return DOMPurify.sanitize(rawHtml, DOMPURIFY_CONFIG);
  }, [text, isVisible]);

  if (!isVisible) {
    const estimatedHeight = Math.max(40, Math.round(text.length * 0.15));
    return (
      <div
        ref={containerRef}
        className="latex-chunk latex-chunk-placeholder"
        style={{ minHeight: `${estimatedHeight}px` }}
        data-chunk={index}
      />
    );
  }

  return (
    <div
      ref={containerRef}
      className="latex-chunk"
      data-chunk={index}
      dangerouslySetInnerHTML={{ __html: renderedHtml }}
    />
  );
});

const LARGE_DOC_THRESHOLD = 50000;

/**
 * LatexRenderer Component — chunked, virtualized rendering for large documents.
 */
const LatexRenderer = ({ 
  content, 
  className = '', 
  defaultRaw = false,
  showToggle = true,
  showLatex
}) => {
  const isLargeDoc = content && content.length > LARGE_DOC_THRESHOLD;
  const [internalViewMode, setInternalViewMode] = useState(
    defaultRaw ? 'raw' : 'rendered'
  );
  const [largeDocWarningDismissed, setLargeDocWarningDismissed] = useState(false);

  // Auto-switch to raw when content grows past threshold (for live/growing documents).
  // Only fires if the user has not explicitly opted into rendered mode.
  useEffect(() => {
    if (isLargeDoc && !largeDocWarningDismissed) {
      setInternalViewMode('raw');
    }
  }, [isLargeDoc, largeDocWarningDismissed]);

  const debouncedContent = useDebouncedValue(content, 1500);

  const viewMode = showLatex !== undefined 
    ? (showLatex ? 'rendered' : 'raw')
    : internalViewMode;

  const renderContent = viewMode === 'rendered' ? debouncedContent : content;

  const chunks = useMemo(() => {
    if (viewMode === 'raw' || !renderContent) return [];
    return splitIntoChunks(renderContent);
  }, [renderContent, viewMode]);

  const renderedHtmlSmall = useMemo(() => {
    if (viewMode === 'raw' || !renderContent) return null;
    if (chunks.length > 1) return null;
    const rawHtml = renderLatexToHtml(renderContent);
    return DOMPurify.sanitize(rawHtml, DOMPURIFY_CONFIG);
  }, [renderContent, viewMode, chunks.length]);

  const hasLatex = useMemo(() => {
    if (!content) return false;
    const sample = content.length > 2000 ? content.slice(0, 2000) : content;
    return /\$[\s\S]*?\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\\[a-zA-Z]+[\s{]/.test(sample);
  }, [content]);

  const wordCount = useMemo(() => {
    if (!content) return 0;
    // Estimate: ~5 chars per word on average (avoids allocating a 250k-element array)
    return Math.round(content.length / 5);
  }, [content]);

  if (!content) {
    return <div className={`latex-renderer ${className}`}>No content</div>;
  }

  const handleSwitchToRendered = () => {
    setLargeDocWarningDismissed(true);
    setInternalViewMode('rendered');
  };

  return (
    <div className={`latex-renderer ${className}`}>
      {showToggle && showLatex === undefined && (
        <div className="latex-toggle-bar">
          <div className="latex-toggle-buttons">
            <button
              className={`latex-toggle-btn ${viewMode === 'rendered' ? 'active' : ''}`}
              onClick={() => {
                if (isLargeDoc && !largeDocWarningDismissed) {
                  handleSwitchToRendered();
                } else {
                  setInternalViewMode('rendered');
                }
              }}
              title="Show rendered LaTeX"
            >
              📐 Rendered {isLargeDoc ? '' : '(Experimental)'}
            </button>
            <button
              className={`latex-toggle-btn ${viewMode === 'raw' ? 'active' : ''}`}
              onClick={() => setInternalViewMode('raw')}
              title="Show raw text"
            >
              Raw Text
            </button>
          </div>
          {viewMode === 'rendered' && chunks.length > 1 && (
            <span className="latex-indicator latex-chunked-indicator">
              {chunks.length} sections
            </span>
          )}
          {hasLatex && viewMode === 'rendered' && (
            <span className="latex-indicator">
              ✓ LaTeX rendered
            </span>
          )}
          {!hasLatex && viewMode === 'rendered' && (
            <span className="latex-indicator no-latex">
              No LaTeX detected
            </span>
          )}
        </div>
      )}
      
      {isLargeDoc && viewMode === 'raw' && !largeDocWarningDismissed && showLatex === undefined && (
        <div className="latex-large-doc-banner">
          <span>Large document ({wordCount.toLocaleString()} words). Rendered view uses progressive loading for performance.</span>
          <button onClick={handleSwitchToRendered} className="latex-large-doc-btn">
            Switch to Rendered View
          </button>
        </div>
      )}

      <div className="latex-content-container">
        {viewMode === 'raw' ? (
          <pre className="latex-raw-content">{content}</pre>
        ) : chunks.length <= 1 ? (
          <div 
            className="latex-rendered-content"
            dangerouslySetInnerHTML={{ __html: renderedHtmlSmall }}
          />
        ) : (
          <div className="latex-rendered-content">
            {chunks.map((chunk, i) => (
              <RenderedChunk key={`${i}-${simpleHash(chunk)}`} text={chunk} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Custom hook: debounce a value so rendered-mode processing doesn't fire on every rapid update.
 */
function useDebouncedValue(value, delayMs) {
  const [debounced, setDebounced] = useState(value);
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setDebounced(value), delayMs);
    return () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); };
  }, [value, delayMs]);

  return debounced;
}

export { renderLatexToHtml, DOMPURIFY_CONFIG };
export default LatexRenderer;
