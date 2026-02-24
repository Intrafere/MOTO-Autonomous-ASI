"""
JSON parsing utilities with sanitization for LLM responses.

Handles common LLM output quirks:
- Reasoning tokens (<think>...</think>)
- Markdown code blocks
- Control tokens
- LaTeX escape sequences
"""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def sanitize_json_response(raw_content: str) -> str:
    """
    Sanitize JSON response to handle LaTeX expressions and invalid escape sequences.
    
    Models often generate LaTeX math notation (\\(, \\), \\[, \\], \\tau, etc.) which
    creates invalid JSON escape sequences. Additionally, some models prepend control
    tokens (e.g., <|channel|>, <|constrain|>, <|message|>) or reasoning tokens
    (e.g., <think>...</think>) before JSON output. Some models also wrap JSON in
    markdown code blocks (```json\n...\n```).
    
    This function preprocesses the content to handle all these issues.
    
    Strategy:
    1. Strip markdown code blocks (```json ... ```)
    2. Strip reasoning tokens (<think>...</think>)
    3. Strip control tokens and any prefix before first { or [
    4. Detect and REJECT truncated JSON (raise ValueError if incomplete)
    5. Pre-escape dangerous LaTeX commands (\\to, \\text, \\tau, etc.)
    6. Handle control characters in strings only
    
    IMPORTANT: This function NO LONGER repairs truncated JSON. If the response
    is incomplete (unclosed braces/brackets/strings or trailing newline spam),
    it raises ValueError. Agents handle this by retrying with feedback asking
    for more concise output.
    
    Args:
        raw_content: Raw string from LLM
        
    Returns:
        Sanitized JSON string ready for parsing
    """
    if not raw_content or not raw_content.strip():
        return raw_content
    
    content = raw_content.strip()
    
    # STEP 1: Strip reasoning/thinking tokens (<think>...</think>)
    # Used by DeepSeek R1 and similar reasoning models
    # Do this FIRST since thinking tokens often appear before markdown blocks
    # Use case-insensitive matching and multiple passes to ensure complete removal
    think_pattern = r'<think>.*?</think>'
    original_len = len(content)
    
    # Multiple passes to handle nested or malformed thinking blocks
    max_iterations = 3
    for _ in range(max_iterations):
        if re.search(think_pattern, content, re.DOTALL | re.IGNORECASE):
            content = re.sub(think_pattern, '', content, flags=re.DOTALL | re.IGNORECASE).strip()
        else:
            break
    
    if len(content) < original_len:
        logger.debug(f"Stripped <think>...</think> reasoning tokens ({original_len} -> {len(content)} chars)")
        logger.debug(f"Content after think removal (first 300 chars): {repr(content[:300])}")
    
    # Extra safety: Remove any remaining thinking-related tags
    content = re.sub(r'</think\s*>', '', content, flags=re.IGNORECASE).strip()
    content = re.sub(r'<think\s*>', '', content, flags=re.IGNORECASE).strip()
    
    # STEP 2: Strip markdown code blocks (```json\n...\n``` or ```\n...\n```)
    # Pattern: ```json (or just ```) followed by content followed by ```
    if content.startswith('```'):
        # Find the closing ```
        lines = content.split('\n')
        if len(lines) > 2:
            # First line is ```json or ```
            # Find the line with closing ```
            closing_index = -1
            for i in range(1, len(lines)):  # Start from line 1 (after opening ```)
                if lines[i].strip() == '```':
                    closing_index = i
                    break
            
            if closing_index > 0:
                # Extract content between the code block markers
                # Skip first line (```json or ```)
                # Extract up to (but not including) closing ```
                # Then append any remaining lines after closing ```
                start_line = 1
                end_line = closing_index
                content_lines = lines[start_line:end_line]
                
                # If there's content after the closing ```, append it
                if closing_index + 1 < len(lines):
                    # Join remaining lines (skip empty lines immediately after ```)
                    remaining = '\n'.join(lines[closing_index + 1:]).strip()
                    if remaining:
                        # Add newlines to separate from extracted content
                        content_lines.append('')
                        content_lines.append('')
                        content_lines.append(remaining)
                
                content = '\n'.join(content_lines).strip()
                logger.debug(f"Stripped markdown code block wrapper")
    
    # STEP 3: Strip control tokens that some models emit
    # Common patterns: <|channel|>...<|message|>, <|constrain|>JSON, etc.
    # Some models emit these BEFORE the JSON, some WITHIN the content
    # Strategy: Remove ALL control token patterns using regex
    
    # Pattern for control tokens: <|word|> or <|word|>word (with optional trailing word)
    control_token_pattern = r'<\|[a-zA-Z_]+\|>(?:[a-zA-Z_]+\s*)?'
    
    if re.search(control_token_pattern, content):
        original_content = content
        content = re.sub(control_token_pattern, '', content).strip()
        logger.debug(
            f"Stripped control tokens: "
            f"'{original_content[:150]}...' -> '{content[:150]}...'"
        )
    
    # Additional cleanup: Remove any remaining angle bracket artifacts
    # that might be partial control tokens
    if '<|' in content:
        # Remove any remaining <|...> patterns
        content = re.sub(r'<\|[^>]*\|>', '', content).strip()
        logger.debug("Removed remaining control token artifacts")
    
    # STEP 4: Extract only the first complete JSON object if multiple exist
    # Some models (especially reasoning models) may output multiple JSON objects
    # We only want the first valid one
    # Find JSON start position
    json_start = -1
    for i, char in enumerate(content):
        if char in ['{', '[']:
            json_start = i
            break
    
    # If no JSON start found, raise explicit error
    if json_start < 0:
        logger.warning(f"No JSON start character found in content (length={len(content)})")
        logger.warning(f"Content preview: {repr(content[:200])}...")
        
        # NEW: Don't continue - this is pure reasoning text with no JSON
        # Raise explicit error for retry mechanism
        raise ValueError(
            f"No JSON found in response - only conversational reasoning text "
            f"({len(content)} chars). Model likely hit max_tokens before writing JSON. "
            f"Content starts with: {repr(content[:200])}"
        )
    else:
        # Strip everything before the JSON start (handles reasoning models that output
        # plain text reasoning without <think> tags before the JSON)
        if json_start > 0:
            stripped_prefix = content[:json_start]
            content = content[json_start:]
            json_start = 0  # Reset to 0 since we stripped the prefix
            logger.debug(f"Stripped {len(stripped_prefix)} chars of non-JSON prefix")
            logger.debug(f"Stripped prefix preview: {repr(stripped_prefix[:200])}...")
    
    if json_start >= 0:
        try:
            # Try to find where the first JSON object ends
            # Use a simple brace counter for objects or bracket counter for arrays
            start_char = content[json_start] if json_start < len(content) else ''
            if start_char == '{':
                # Track opening/closing braces
                brace_count = 0
                in_string = False
                escape_next = False
                end_pos = -1
                
                for i in range(json_start, len(content)):
                    char = content[i]
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not in_string:
                        in_string = True
                    elif char == '"' and in_string:
                        in_string = False
                    elif char == '{' and not in_string:
                        brace_count += 1
                    elif char == '}' and not in_string:
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                # If we found the end of the first JSON object, extract only that
                if end_pos > 0 and end_pos < len(content):
                    original_length = len(content)
                    content = content[:end_pos]
                    logger.debug(f"Extracted first JSON object (truncated from {original_length} to {end_pos} chars)")
                elif end_pos == -1 and brace_count > 0:
                    # JSON object was never closed - model hit max_tokens
                    last_complete = content.rfind('",')
                    last_complete_context = ""
                    if last_complete > json_start:
                        last_complete_context = f"Last completed property at char {last_complete}: ...{repr(content[max(0,last_complete-50):last_complete+50])}..."
                    
                    logger.error(
                        f"JSON TRUNCATION DETECTED: Model hit max_tokens during generation. "
                        f"Response has {brace_count} unclosed braces, in_string={in_string}. "
                        f"Response length: {len(content)} chars. {last_complete_context}"
                    )
                    raise ValueError(
                        f"JSON response truncated at max_tokens: {brace_count} unclosed braces, "
                        f"in_string={in_string}, response length {len(content)} chars. "
                        f"Model needs to generate more concise output that fits within token limits. "
                        f"{last_complete_context}"
                    )
            
            elif start_char == '[':
                # Track opening/closing brackets
                bracket_count = 0
                in_string = False
                escape_next = False
                end_pos = -1
                
                for i in range(json_start, len(content)):
                    char = content[i]
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not in_string:
                        in_string = True
                    elif char == '"' and in_string:
                        in_string = False
                    elif char == '[' and not in_string:
                        bracket_count += 1
                    elif char == ']' and not in_string:
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i + 1
                            break
                
                # If we found the end of the first JSON array, extract only that
                if end_pos > 0 and end_pos < len(content):
                    original_length = len(content)
                    content = content[:end_pos]
                    logger.debug(f"Extracted first JSON array (truncated from {original_length} to {end_pos} chars)")
                elif end_pos == -1 and bracket_count > 0:
                    # JSON array was never closed - model hit max_tokens
                    logger.error(
                        f"JSON TRUNCATION DETECTED: Model hit max_tokens during generation. "
                        f"Response has {bracket_count} unclosed brackets, in_string={in_string}. "
                        f"Response length: {len(content)} chars."
                    )
                    raise ValueError(
                        f"JSON response truncated at max_tokens: {bracket_count} unclosed brackets, "
                        f"in_string={in_string}, response length {len(content)} chars. "
                        f"Model needs to generate more concise output that fits within token limits."
                    )
        
        except Exception as e:
            # If extraction fails, continue with full content
            logger.debug(f"Failed to extract first JSON object: {e}")
    
    # Safety check: ensure content is not empty after preprocessing
    if not content or not content.strip():
        logger.error(f"Sanitization resulted in empty content! Original length: {len(raw_content)}")
        logger.error(f"Original content preview: {raw_content[:500]}...")
        # Return original content and let the caller handle the error
        return raw_content.strip()
    
    # STEP 4.7: Fix common invalid \uXXXX patterns from LaTeX
    # Pattern: \u{widehat}, \u{infty}, etc. → these are NOT valid \uXXXX Unicode escapes
    # Replace \u{ with \\u{ to escape the backslash
    
    # Fix \u{word} patterns (invalid Unicode escapes from LaTeX)
    invalid_unicode_pattern = r'\\u\{([a-zA-Z]+)\}'
    if re.search(invalid_unicode_pattern, content):
        content = re.sub(invalid_unicode_pattern, r'\\\\u{\1}', content)
        logger.debug("Fixed invalid \\u{...} LaTeX patterns")
    
    # Fix other common invalid \uXXXX where XXXX is not exactly 4 hex digits
    # Valid: \u03B1, \u0041
    # Invalid: \uinfty, \uphi, \uwidehat
    invalid_u_escape = r'\\u([^0-9a-fA-F{])'  # \u followed by non-hex (except opening brace)
    if re.search(invalid_u_escape, content):
        content = re.sub(invalid_u_escape, r'\\\\u\1', content)
        logger.debug("Fixed invalid \\u escape sequences")
    
    # STEP 4.8: Detect truncation patterns and REJECT (no repair)
    # Two patterns indicate truncation:
    # 1. Trailing newline spam (\n\n\n... or \\n\\n\\n...)
    # 2. Incomplete JSON structure (unclosed braces/brackets/strings)
    
    # Check for trailing spam patterns
    trailing_newline_match = re.search(r'((?:\\n){10,})$', content)
    if not trailing_newline_match:
        trailing_newline_match = re.search(r'(\n{10,})$', content)
    
    if trailing_newline_match:
        matched_text = trailing_newline_match.group(1)
        newline_count = len(matched_text) // 2 if matched_text.startswith('\\n') else len(matched_text)
        logger.error(
            f"JSON TRUNCATION DETECTED: Trailing newline spam pattern detected. "
            f"{newline_count} consecutive newlines at end of response (length: {len(content)} chars). "
            f"This indicates model hit max_tokens and filled remaining space with newlines."
        )
        raise ValueError(
            f"JSON response truncated at max_tokens: detected {newline_count} consecutive newlines "
            f"at end of {len(content)} char response. Model needs to generate more concise output."
        )
    
    # Check for incomplete JSON structure (backup check for other truncation patterns)
    # Track braces/brackets/strings to detect incomplete structure
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False
    
    for char in content:
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            elif char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1
    
    # If JSON structure is incomplete, raise error
    if open_braces != 0 or open_brackets != 0 or in_string:
        logger.error(
            f"JSON TRUNCATION DETECTED: Incomplete JSON structure. "
            f"open_braces={open_braces}, open_brackets={open_brackets}, in_string={in_string}. "
            f"Response length: {len(content)} chars. Model hit max_tokens during generation."
        )
        raise ValueError(
            f"JSON response truncated at max_tokens: incomplete structure with "
            f"{open_braces} unclosed braces, {open_brackets} unclosed brackets, "
            f"in_string={in_string}. Response length: {len(content)} chars. "
            f"Model needs to generate more concise output that fits within token limits."
        )
    
    # STEP 5a: Pre-escape DANGEROUS LaTeX commands that start with valid JSON escape characters
    # This MUST happen BEFORE any json.loads() attempt, including the fast path!
    # 
    # These commands cause the parser to incorrectly interpret them as valid JSON escapes:
    #   \beta (looks like \b backspace + "eta")
    #   \frac (looks like \f form-feed + "rac") 
    #   \nu (looks like \n newline + "u")
    #   \tau (looks like \t tab + "au")
    #   \to (looks like \t tab + "o")  <-- CRITICAL: Very common in math!
    #   \text (looks like \t tab + "ext")  <-- CRITICAL: Very common in math!
    #   \rightarrow (looks like \r carriage-return + "ightarrow")
    # 
    # IMPORTANT: We only pre-escape these specific dangerous patterns, not all LaTeX.
    # The character-by-character parser will handle other LaTeX like \pi, \phi, etc.
    
    # List of dangerous LaTeX commands that start with valid JSON escape characters
    # Format: (pattern to match, safe replacement)
    # NOTE: Order matters for overlapping patterns - longer patterns first!
    # 
    # CRITICAL: Each pattern uses (?<!\\) negative lookbehind to ONLY match
    # unescaped backslashes. Without this, when a model properly escapes its JSON
    # (outputting \\begin), the regex matches \begin WITHIN \\begin and creates
    # \\\begin, which json.loads() then interprets as \ + backspace + egin.
    # The lookbehind ensures \\begin (already escaped) is left untouched.
    dangerous_latex_commands = [
        # Commands starting with \b (backspace) - longer patterns first
        (r'(?<!\\)\\boldsymbol', r'\\\\boldsymbol'),
        (r'(?<!\\)\\bigotimes', r'\\\\bigotimes'),
        (r'(?<!\\)\\bigoplus', r'\\\\bigoplus'),
        (r'(?<!\\)\\bigcap', r'\\\\bigcap'),
        (r'(?<!\\)\\bigcup', r'\\\\bigcup'),
        (r'(?<!\\)\\binom', r'\\\\binom'),
        (r'(?<!\\)\\boxed', r'\\\\boxed'),
        (r'(?<!\\)\\begin', r'\\\\begin'),
        (r'(?<!\\)\\beta', r'\\\\beta'),
        (r'(?<!\\)\\bar', r'\\\\bar'),
        (r'(?<!\\)\\big', r'\\\\big'),
        # Commands starting with \f (form-feed)
        (r'(?<!\\)\\forall', r'\\\\forall'),
        (r'(?<!\\)\\frac', r'\\\\frac'),
        # Commands starting with \n (newline) - longer patterns first
        (r'(?<!\\)\\nabla', r'\\\\nabla'),
        (r'(?<!\\)\\newline', r'\\\\newline'),
        (r'(?<!\\)\\notin', r'\\\\notin'),
        (r'(?<!\\)\\neq', r'\\\\neq'),
        (r'(?<!\\)\\neg', r'\\\\neg'),
        (r'(?<!\\)\\not', r'\\\\not'),
        (r'(?<!\\)\\nu', r'\\\\nu'),
        # Commands starting with \t (tab) - longer patterns first
        # CRITICAL: These are extremely common in mathematical LaTeX!
        (r'(?<!\\)\\textbf', r'\\\\textbf'),
        (r'(?<!\\)\\textit', r'\\\\textit'),
        (r'(?<!\\)\\textrm', r'\\\\textrm'),
        (r'(?<!\\)\\textsc', r'\\\\textsc'),
        (r'(?<!\\)\\textsf', r'\\\\textsf'),
        (r'(?<!\\)\\texttt', r'\\\\texttt'),
        (r'(?<!\\)\\triangle', r'\\\\triangle'),
        (r'(?<!\\)\\times', r'\\\\times'),
        (r'(?<!\\)\\tilde', r'\\\\tilde'),
        (r'(?<!\\)\\theta', r'\\\\theta'),
        (r'(?<!\\)\\text', r'\\\\text'),
        (r'(?<!\\)\\top', r'\\\\top'),
        (r'(?<!\\)\\tau', r'\\\\tau'),
        (r'(?<!\\)\\to', r'\\\\to'),  # CRITICAL: Very common arrow command!
        # Commands starting with \r (carriage-return) - longer patterns first
        (r'(?<!\\)\\rightarrow', r'\\\\rightarrow'),
        (r'(?<!\\)\\Rightarrow', r'\\\\Rightarrow'),
        (r'(?<!\\)\\right', r'\\\\right'),
        (r'(?<!\\)\\rho', r'\\\\rho'),
        (r'(?<!\\)\\real', r'\\\\real'),
        (r'(?<!\\)\\ref', r'\\\\ref'),
        # Commands starting with \u (unicode escape prefix)
        # Note: \uXXXX is handled separately, but these are LaTeX commands
        (r'(?<!\\)\\upsilon', r'\\\\upsilon'),
        (r'(?<!\\)\\underset', r'\\\\underset'),
        (r'(?<!\\)\\underline', r'\\\\underline'),
        (r'(?<!\\)\\uparrow', r'\\\\uparrow'),
    ]
    
    sanitized = content
    
    # Apply pre-escaping for dangerous LaTeX commands
    # We do this BEFORE any json.loads() attempt to prevent misinterpretation
    pre_escaped = sanitized
    pre_escape_applied = False
    for pattern, replacement in dangerous_latex_commands:
        if re.search(pattern, pre_escaped):
            pre_escaped = re.sub(pattern, replacement, pre_escaped)
            pre_escape_applied = True
    
    if pre_escape_applied:
        logger.debug("Pre-escaped dangerous LaTeX commands that start with JSON escape chars")
    
    sanitized = pre_escaped
    
    # Fast path: try parsing as-is first (AFTER pre-escaping dangerous LaTeX)
    try:
        json.loads(sanitized)
        return sanitized  # Already valid
    except json.JSONDecodeError:
        pass  # Need further sanitization
    
    # STEP 5b: Robust escape handling for remaining LaTeX expressions
    # Parse character-by-character to properly handle all remaining escape sequences
    # (e.g., \pi, \phi, \epsilon, \alpha, \gamma, \delta, etc.)
    
    def robust_escape_latex(text):
        """
        Parse JSON string and escape invalid backslash sequences.
        Handles complex nested LaTeX like \\phi_{\\\\\\pi_v} correctly.
        
        Strategy:
        - Track if we're inside a JSON string value (between quotes)
        - Inside strings: Replace \X where X is not a valid JSON escape with \\X
        - Outside strings: Preserve structural JSON (braces, brackets, commas)
        - Valid JSON escapes: ", \, /, b, f, n, r, t, uXXXX
        
        NOTE: This function expects dangerous LaTeX commands (\beta, \frac, \nu, \tau, \rightarrow)
        to have been pre-escaped before calling. It handles other LaTeX safely.
        """
        result = []
        i = 0
        in_string = False
        
        while i < len(text):
            char = text[i]
            
            # Toggle string state on unescaped quotes
            if char == '"':
                # Check if this quote is escaped by counting preceding backslashes
                num_backslashes = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    num_backslashes += 1
                    j -= 1
                
                # Quote is unescaped if preceded by even number of backslashes (including 0)
                if num_backslashes % 2 == 0:
                    in_string = not in_string
                
                result.append(char)
                i += 1
                continue
            
            # Handle backslashes inside strings
            if in_string and char == '\\':
                # Look at next character to determine if this is a valid escape
                if i + 1 < len(text):
                    next_char = text[i + 1]
                    
                    # Check if this is already an escaped backslash (\\)
                    if next_char == '\\':
                        # This is \\, which is a valid escaped backslash - pass through both
                        result.append(char)
                        result.append(next_char)
                        i += 2
                        continue
                    
                    # Valid JSON escape sequences  
                    if next_char in '"\/':
                        # \", \/, - valid escapes, keep as-is
                        result.append(char)
                        i += 1
                        continue
                    
                    # For b, f, n, r, t - these COULD be valid JSON escapes,
                    # but the dangerous LaTeX commands should have been pre-escaped.
                    # If we see \n, \t, etc. at this point, they're likely actual JSON escapes.
                    if next_char in 'bfnrt':
                        # Trust that pre-escaping handled dangerous LaTeX
                        # This is likely a real JSON escape
                        result.append(char)
                        i += 1
                        continue
                    
                    # Check for valid \uXXXX (4 hex digits)
                    if next_char == 'u':
                        if i + 5 < len(text):
                            hex_chars = text[i+2:i+6]
                            try:
                                int(hex_chars, 16)  # Valid hex
                                result.append(char)  # Keep original backslash
                                i += 1
                                continue
                            except ValueError:
                                # Not valid \uXXXX - escape it
                                result.append('\\\\')
                                i += 1
                                continue
                        else:
                            # Not enough chars for \uXXXX - escape it
                            result.append('\\\\')
                            i += 1
                            continue
                    
                    # Invalid escape - need to escape the backslash
                    # This handles remaining LaTeX like \pi, \phi, \epsilon, etc.
                    result.append('\\\\')
                    i += 1
                    continue
                else:
                    # Backslash at end of string - escape it
                    result.append('\\\\')
                    i += 1
                    continue
            
            # Outside strings or non-backslash characters - keep as-is
            result.append(char)
            i += 1
        
        return ''.join(result)
    
    sanitized = robust_escape_latex(sanitized)
    
    # STEP 6: Escape raw control characters (ASCII 0x00-0x1F) ONLY INSIDE STRING VALUES
    # These cause "Invalid control character" errors in json.loads()
    # 
    # CRITICAL: We must ONLY escape control chars inside JSON strings, NOT structural whitespace!
    # Structural whitespace (newlines/tabs between tokens) is valid JSON and must be preserved.
    # 
    # Strategy: Parse character by character, track whether we're inside a string value,
    # and only escape control chars when inside strings.
    try:
        json.loads(sanitized)
        return sanitized  # Already valid after LaTeX fix
    except json.JSONDecodeError as e:
        if "control character" in str(e).lower():
            def escape_control_chars_in_strings(s):
                """Escape control characters ONLY inside JSON string values, preserving structural whitespace."""
                result = []
                in_string = False
                escape_next = False
                
                for char in s:
                    code = ord(char)
                    
                    if escape_next:
                        # Previous char was backslash, this char is escaped
                        result.append(char)
                        escape_next = False
                        continue
                    
                    if char == '\\' and in_string:
                        # Backslash inside string - next char is escaped
                        result.append(char)
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        # Quote toggles string state
                        in_string = not in_string
                        result.append(char)
                        continue
                    
                    # Control characters: 0x00-0x1F (excluding common whitespace outside strings)
                    if code < 0x20:
                        if in_string:
                            # Inside string - escape the control character
                            if code == 0x09:    # Tab -> \t
                                result.append('\\t')
                            elif code == 0x0A:  # Newline -> \n
                                result.append('\\n')
                            elif code == 0x0D:  # Carriage return -> \r
                                result.append('\\r')
                            else:
                                # Other control chars -> \uXXXX
                                result.append(f'\\u{code:04x}')
                        else:
                            # Outside string - preserve structural whitespace
                            # Only preserve newline, tab, carriage return as structural whitespace
                            if code in (0x09, 0x0A, 0x0D):
                                result.append(char)
                            else:
                                # Other control chars outside strings - escape them
                                result.append(f'\\u{code:04x}')
                    else:
                        result.append(char)
                
                return ''.join(result)
            
            sanitized = escape_control_chars_in_strings(sanitized)
            logger.debug("Escaped raw control characters in JSON string values (preserved structural whitespace)")
    
    return sanitized


def parse_json(response_content: str) -> dict:
    """
    Parse JSON response from LLM with sanitization.
    
    Handles common LLM output quirks like reasoning tokens, markdown blocks,
    control tokens, and LaTeX escape sequences.
    
    Args:
        response_content: Raw response string to parse
        
    Returns:
        Parsed JSON dict
        
    Raises:
        json.JSONDecodeError: If parsing fails after sanitization
        ValueError: If response is empty or too short
    """
    # Handle dict responses that are already parsed
    if isinstance(response_content, dict):
        logger.debug("parse_json: Received already-parsed dict, returning as-is")
        return response_content
    
    # Handle non-string responses
    if not isinstance(response_content, str):
        logger.error(f"parse_json: Received non-string response of type {type(response_content)}")
        raise ValueError(f"Expected string or dict, got {type(response_content)}")
    
    # Check for empty or whitespace-only response
    if not response_content or not response_content.strip():
        logger.error("parse_json: Received empty or whitespace-only response")
        raise ValueError("Empty or whitespace-only response")
    
    # Check for anomalously short response
    if len(response_content.strip()) < 10:
        logger.error(f"parse_json: Response too short ({len(response_content)} chars)")
        logger.error(f"Short response content: {repr(response_content)}")
        raise ValueError(f"Response too short ({len(response_content)} chars)")
    
    # Sanitize and parse
    try:
        sanitized_content = sanitize_json_response(response_content)
        result = json.loads(sanitized_content)
    except ValueError as e:
        # Handle explicit "No JSON found" errors from sanitization
        if "No JSON found" in str(e):
            logger.error("🚨 MODEL OUTPUT CONTAINS NO JSON STRUCTURE")
            logger.error(f"This indicates the model spent all tokens on reasoning text")
            logger.error(f"Consider: shorter prompts, or explicit 'JSON ONLY' instruction")
        logger.error(f"parse_json: {e}")
        raise
    except json.JSONDecodeError as e:
        # Enhanced error logging for JSON decode failures
        error_msg = str(e)
        
        # Detect common truncation patterns
        is_likely_truncated = False
        truncation_hints = []
        
        if "unterminated string" in error_msg.lower():
            is_likely_truncated = True
            truncation_hints.append("unterminated string (string never closed)")
        
        if "expecting" in error_msg.lower() and "eof" in error_msg.lower():
            is_likely_truncated = True
            truncation_hints.append("unexpected end of data")
        
        # Check for unclosed braces/brackets at end
        stripped = sanitized_content.rstrip()
        if stripped and stripped[-1] not in '}]':
            is_likely_truncated = True
            truncation_hints.append(f"JSON doesn't end with }} or ] (ends with: {repr(stripped[-20:])})")
        
        # Count unclosed braces/brackets (rough check)
        open_braces = sanitized_content.count('{') - sanitized_content.count('}')
        open_brackets = sanitized_content.count('[') - sanitized_content.count(']')
        if open_braces > 0 or open_brackets > 0:
            is_likely_truncated = True
            if open_braces > 0:
                truncation_hints.append(f"{open_braces} unclosed braces")
            if open_brackets > 0:
                truncation_hints.append(f"{open_brackets} unclosed brackets")
        
        # Log the error with enhanced diagnostics
        logger.error(f"parse_json: JSON decode failed - {e}")
        
        if is_likely_truncated:
            logger.error(f"🚨 LIKELY TRUNCATED LLM OUTPUT: {', '.join(truncation_hints)}")
            logger.error("This usually means the LLM hit max_tokens limit before completing the JSON response")
        
        logger.error(f"Original response length: {len(response_content)} chars")
        logger.error(f"Original response (first 500 chars): {repr(response_content[:500])}")
        logger.error(f"Original response (last 200 chars): {repr(response_content[-200:])}")
        logger.error(f"Sanitized content length: {len(sanitized_content)} chars")
        logger.error(f"Sanitized content (first 500 chars): {repr(sanitized_content[:500])}")
        logger.error(f"Sanitized content (last 200 chars): {repr(sanitized_content[-200:])}")
        logger.error(f"Error position: line {e.lineno}, column {e.colno}, char {e.pos}")
        if e.pos is not None and e.pos < len(sanitized_content):
            # Show context around error position
            start = max(0, e.pos - 50)
            end = min(len(sanitized_content), e.pos + 50)
            logger.error(f"Error context: ...{repr(sanitized_content[start:end])}...")
        raise
    except Exception as e:
        # Catch any other parsing errors
        logger.error(f"parse_json: Unexpected error during parsing - {type(e).__name__}: {e}")
        logger.error(f"Response content: {repr(response_content[:1000])}")
        raise
    
    # Handle array responses - extract first element
    if isinstance(result, list):
        if len(result) > 0:
            logger.warning("LLM returned array instead of object - using first element")
            result = result[0]
        else:
            logger.error("parse_json: LLM returned empty array")
            raise ValueError("LLM returned empty array")
    
    return result
