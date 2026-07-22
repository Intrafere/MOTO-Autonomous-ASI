r"""
JSON validator - validates LLM JSON responses against schemas.
Implements multi-strategy JSON repair to handle LaTeX, malformed unicode, and other LLM errors.

CHANGELOG:
- Replaced unsafe \x00 placeholder markers with safe <<<TOKEN>>> system
- Added multi-strategy repair pipeline: unicode fix → LaTeX escape → placeholder processing → aggressive mode
- Handles LaTeX notation: \(, \), \[, \], \mathcal{}, \Delta, etc.
- Handles malformed unicode: \u20101 (5 digits) → \u2010 + literal "1"
- Handles incomplete unicode: \u12 → \\u12 (escaped)
- Never crashes during preprocessing (safe exception handling)
- Logs repair strategy used for observability
"""
import json
import re
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class JSONValidator:
    r"""
    Validates JSON responses from LLM with robust multi-strategy repair.
    
    Repair Pipeline:
    1. Strategy A: Fix malformed unicode escapes
    2. Strategy B: Escape LaTeX notation to valid JSON
    3. Strategy C: Process with safe placeholder tokens
    4. Strategy E: Aggressive mode (strip all invalid escapes)
    
    All strategies preserve valid JSON escape sequences: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    """
    
    # Safe placeholder tokens (no problematic escape sequences)
    PLACEHOLDERS = {
        'DBLSLASH': '\\\\',
        'QUOTE': '\\"',
        'SLASH': '\\/',
        'NEWLINE': '\\n',
        'RETURN': '\\r',
        'TAB': '\\t',
        'BACKSPACE': '\\b',
        'FORMFEED': '\\f',
        'LPAREN': '\\\\(',
        'RPAREN': '\\\\)',
        'LBRACK': '\\\\[',
        'RBRACK': '\\\\]',
        'LBRACE': '\\\\{',
        'RBRACE': '\\\\}',
    }
    
    def _fix_malformed_unicode(self, text: str) -> Tuple[str, bool]:
        r"""
        Fix malformed unicode escape sequences.
        
        Strategy A: Handle 5+ hex digits and incomplete escapes.
        - \u20101 → \u2010 + literal "1"
        - \u12 → \\u12 (escaped as literal)
        
        Returns:
            (fixed_text, was_modified)
        """
        modified = False
        
        # Fix 5+ hex digit unicode (take first 4, literalize rest)
        def fix_long_unicode(match):
            nonlocal modified
            modified = True
            hex_digits = match.group(1)
            valid_part = hex_digits[:4]
            extra_part = hex_digits[4:]
            return f'\\u{valid_part}{extra_part}'
        
        text = re.sub(r'\\u([0-9a-fA-F]{5,})', fix_long_unicode, text)
        
        # Fix incomplete unicode (1-3 hex digits) - escape the backslash
        def fix_short_unicode(match):
            nonlocal modified
            modified = True
            hex_digits = match.group(1)
            return f'\\\\u{hex_digits}'
        
        text = re.sub(r'\\u([0-9a-fA-F]{1,3})(?![0-9a-fA-F])', fix_short_unicode, text)
        
        return text, modified
    
    def _escape_latex_notation(self, text: str) -> Tuple[str, bool]:
        r"""
        Escape LaTeX math notation to valid JSON.
        
        Strategy B: Convert LaTeX commands to double-escaped form.
        - \( → \\(
        - \mathcal{ → \\mathcal{
        - \Delta → \\Delta
        
        Returns:
            (fixed_text, was_modified)
        """
        modified = False
        
        # Escape LaTeX delimiters
        latex_delimiters = ['\\(', '\\)', '\\[', '\\]', '\\{', '\\}']
        for delim in latex_delimiters:
            if delim in text:
                # Only escape if not already escaped
                text = text.replace(delim, delim[0] + delim)
                modified = True
        
        # Escape LaTeX commands: \word or \word{
        # But preserve valid JSON unicode escapes (\uXXXX)
        def escape_latex_command(match):
            nonlocal modified
            full_match = match.group(0)
            command = match.group(1)
            
            # Skip if it's a unicode escape
            if command.startswith('u') and len(command) == 5:
                return full_match
            
            # Skip if already escaped
            if match.start() > 0 and text[match.start() - 1] == '\\':
                return full_match
            
            modified = True
            return '\\' + full_match
        
        text = re.sub(r'\\([a-zA-Z]+\{?)', escape_latex_command, text)
        
        return text, modified
    
    def _process_with_safe_placeholders(self, json_str: str) -> Tuple[str, bool]:
        """
        Process JSON strings using safe placeholder tokens.
        
        Strategy C: Replace valid escapes with safe placeholders, escape remaining backslashes,
        then restore. Uses <<<TOKEN>>> format to avoid any escape sequence issues.
        
        Returns:
            (fixed_json, was_modified)
        """
        modified = False
        
        def fix_string_escapes(match):
            nonlocal modified
            content = match.group(1)
            original_content = content
            
            # Apply safe placeholders for valid JSON escapes
            temp = content
            temp = temp.replace('\\\\', '<<<DBLSLASH>>>')
            temp = temp.replace('\\"', '<<<QUOTE>>>')
            temp = temp.replace('\\/', '<<<SLASH>>>')
            temp = temp.replace('\\n', '<<<NEWLINE>>>')
            temp = temp.replace('\\r', '<<<RETURN>>>')
            temp = temp.replace('\\t', '<<<TAB>>>')
            temp = temp.replace('\\b', '<<<BACKSPACE>>>')
            temp = temp.replace('\\f', '<<<FORMFEED>>>')
            
            # Preserve unicode escapes \uXXXX (valid 4 hex digits only)
            temp = re.sub(r'\\u([0-9a-fA-F]{4})', r'<<<UNICODE\1>>>', temp)
            
            # Preserve LaTeX notation (already double-escaped)
            temp = temp.replace('\\\\(', '<<<LPAREN>>>')
            temp = temp.replace('\\\\)', '<<<RPAREN>>>')
            temp = temp.replace('\\\\[', '<<<LBRACK>>>')
            temp = temp.replace('\\\\]', '<<<RBRACK>>>')
            temp = temp.replace('\\\\{', '<<<LBRACE>>>')
            temp = temp.replace('\\\\}', '<<<RBRACE>>>')
            
            # Now escape any remaining backslashes (these are the problematic ones)
            if '\\' in temp:
                temp = temp.replace('\\', '\\\\')
                modified = True
            
            # Restore valid escapes from placeholders
            for key, value in self.PLACEHOLDERS.items():
                temp = temp.replace(f'<<<{key}>>>', value)
            
            # Restore unicode escapes
            temp = re.sub(r'<<<UNICODE([0-9a-fA-F]{4})>>>', r'\\u\1', temp)
            
            if temp != original_content:
                modified = True
            
            return f'"{temp}"'
        
        try:
            # Match JSON string values with improved pattern
            # This handles escaped quotes within strings properly
            result = re.sub(r'"((?:[^"\\]|\\.)*)?"', fix_string_escapes, json_str)
            return result, modified
        except Exception as e:
            logger.warning(f"Safe placeholder processing failed: {e}")
            return json_str, False
    
    def _repair_json_aggressive(self, json_str: str) -> str:
        """
        Aggressive JSON repair as last resort.
        
        Strategy E: Strip all problematic backslashes except valid JSON escapes.
        This may lose some information but produces valid JSON.
        
        Returns:
            Aggressively repaired JSON string
        """
        def aggressive_fix(match):
            content = match.group(1)
            
            # Keep only valid JSON escape sequences
            # Remove all backslashes, then re-add only valid ones
            result = []
            i = 0
            while i < len(content):
                if content[i] == '\\' and i + 1 < len(content):
                    next_char = content[i + 1]
                    # Valid JSON escapes: " \ / b f n r t u
                    if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']:
                        result.append('\\')
                        result.append(next_char)
                        i += 2
                    elif next_char == 'u' and i + 5 < len(content):
                        # Check if it's a valid \uXXXX
                        hex_part = content[i+2:i+6]
                        if len(hex_part) == 4 and all(c in '0123456789abcdefABCDEF' for c in hex_part):
                            result.append('\\u')
                            result.append(hex_part)
                            i += 6
                        else:
                            # Invalid unicode escape - skip the backslash
                            i += 1
                    else:
                        # Invalid escape - skip the backslash
                        i += 1
                else:
                    result.append(content[i])
                    i += 1
            
            return f'"{"".join(result)}"'
        
        try:
            return re.sub(r'"((?:[^"\\]|\\.)*)"', aggressive_fix, json_str)
        except Exception as e:
            logger.error(f"Aggressive repair failed: {e}")
            return json_str
    
    def _repair_json_string(self, json_str: str) -> Tuple[str, bool, str]:
        """
        Multi-strategy JSON repair pipeline.
        
        Returns:
            (repaired_string, was_modified, strategy_used)
        """
        try:
            # Try strict parsing first
            json.loads(json_str)
            return json_str, False, "none"
        except json.JSONDecodeError:
            pass
        
        # Strategy A: Fix malformed unicode
        result, mod_a = self._fix_malformed_unicode(json_str)
        
        # Strategy B: Escape LaTeX commands
        result, mod_b = self._escape_latex_notation(result)
        
        # Strategy C: Process strings with safe placeholders
        result, mod_c = self._process_with_safe_placeholders(result)
        
        # Validate repair succeeded
        try:
            json.loads(result)
            strategy = []
            if mod_a:
                strategy.append("unicode")
            if mod_b:
                strategy.append("latex")
            if mod_c:
                strategy.append("placeholders")
            strategy_name = "+".join(strategy) if strategy else "minimal"
            logger.info(f"JSON repair succeeded using strategy: {strategy_name}")
            return result, True, strategy_name
        except json.JSONDecodeError:
            pass
        
        # Strategy E: Aggressive mode
        logger.warning("Standard repair strategies failed, attempting aggressive mode")
        result_aggressive = self._repair_json_aggressive(json_str)
        try:
            json.loads(result_aggressive)
            logger.info("JSON repair succeeded using aggressive strategy")
            return result_aggressive, True, "aggressive"
        except json.JSONDecodeError:
            # No repair worked
            logger.error("All JSON repair strategies failed")
            return json_str, False, "failed"
    
    def extract_and_validate_json(
        self,
        llm_output: str,
        expected_schema: Dict[str, type]
    ) -> Tuple[bool, Optional[Dict], str]:
        """
        Extract and validate JSON from LLM output with multi-strategy repair.
        
        Args:
            llm_output: Raw output from LLM
            expected_schema: Dict mapping field names to expected types
        
        Returns:
            (valid, parsed_json, error_message)
        """
        # Extract JSON from markdown or raw output
        try:
            # Try to extract JSON from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return False, None, "No JSON found in LLM output"
        except re.error as e:
            return False, None, f"Invalid escape sequences in LLM output prevented JSON extraction: {str(e)}"
        
        # Multi-strategy JSON repair
        parsed = None
        
        try:
            # Attempt repair (includes strict parsing as first step)
            repaired_json, was_repaired, repair_strategy = self._repair_json_string(json_str)
            
            if was_repaired:
                logger.info(f"JSON repair applied using strategy: {repair_strategy}")
            
            # Parse repaired JSON
            try:
                parsed = json.loads(repaired_json)
            except json.JSONDecodeError as e:
                # Log detailed error for debugging
                logger.error(f"JSON parsing failed after repair. First 500 chars: {json_str[:500]}")
                logger.error(f"Repair strategy used: {repair_strategy}")
                error_msg = (
                    f"Invalid JSON at line {e.lineno} col {e.colno} (char {e.pos}): {e.msg}. "
                    f"Repair strategy '{repair_strategy}' was attempted but failed."
                )
                return False, None, error_msg
                
        except Exception as e:
            # Catch any unexpected errors during repair
            logger.error(f"JSON repair failed with unexpected error: {str(e)}", exc_info=True)
            logger.error(f"First 500 chars of extracted JSON: {json_str[:500]}")
            return False, None, f"JSON repair error: {str(e)}"
        
        # Validate schema
        for field, expected_type in expected_schema.items():
            if field not in parsed:
                return False, None, f"Missing required field: {field}"
            
            if not isinstance(parsed[field], expected_type):
                return False, None, (
                    f"Field '{field}' has wrong type. "
                    f"Expected {expected_type.__name__}, got {type(parsed[field]).__name__}"
                )
        
        return True, parsed, f"JSON validation passed (repair: {repair_strategy})"
    
    def validate_submission_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate submitter JSON output."""
        schema = {
            "submission": str,
            "reasoning": str
        }
        return self.extract_and_validate_json(llm_output, schema)
    
    def validate_validation_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate validator JSON output."""
        schema = {
            "decision": str,
            "reasoning": str,
            "summary": str
        }
        valid, parsed, error = self.extract_and_validate_json(llm_output, schema)
        
        if valid:
            # Additional validation for decision field
            if parsed["decision"] not in ["accept", "reject"]:
                return False, None, f"Invalid decision value: {parsed['decision']}. Must be 'accept' or 'reject'"
        
        return valid, parsed, error
    
    # Compiler-specific validation methods
    
    def validate_construction_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Validate construction mode JSON output.
        
        Supports both legacy and phase-based construction formats:
        - Legacy: needs_construction, content, placement, reasoning
        - Phase-based: needs_construction, content, placement, section_complete, reasoning
        """
        # Minimal required schema (section_complete is optional for backward compatibility)
        minimal_schema = {
            "needs_construction": bool,
            "reasoning": str
        }
        
        valid, parsed, error = self.extract_and_validate_json(llm_output, minimal_schema)
        
        if not valid:
            return valid, parsed, error
        
        # Ensure string fields are strings (may be empty if needs_construction=False)
        if "content" in parsed and not isinstance(parsed["content"], str):
            return False, None, f"Field 'content' must be string, got {type(parsed['content']).__name__}"
        
        if "old_string" in parsed and not isinstance(parsed["old_string"], str):
            return False, None, f"Field 'old_string' must be string, got {type(parsed['old_string']).__name__}"
        
        if "new_string" in parsed and not isinstance(parsed["new_string"], str):
            return False, None, f"Field 'new_string' must be string, got {type(parsed['new_string']).__name__}"
        
        # Validate operation if present
        if "operation" in parsed:
            valid_ops = ["replace", "insert_after", "delete", "full_content"]
            if parsed["operation"] not in valid_ops:
                return False, None, f"Field 'operation' must be one of {valid_ops}, got '{parsed['operation']}'"
        
        # Validate section_complete if present (optional field for phase-based construction)
        if "section_complete" in parsed and not isinstance(parsed["section_complete"], bool):
            return False, None, f"Field 'section_complete' must be boolean, got {type(parsed['section_complete']).__name__}"
        
        # Set defaults for optional fields
        if "content" not in parsed:
            parsed["content"] = ""
        if "operation" not in parsed:
            parsed["operation"] = "replace"
        if "old_string" not in parsed:
            parsed["old_string"] = ""
        if "new_string" not in parsed:
            parsed["new_string"] = ""
        if "section_complete" not in parsed:
            parsed["section_complete"] = False
        
        return True, parsed, error
    
    def validate_outline_create_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate outline creation JSON output."""
        schema = {
            "content": str,
            "reasoning": str
        }
        return self.extract_and_validate_json(llm_output, schema)
    
    def validate_outline_update_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate outline update JSON output."""
        schema = {
            "needs_update": bool,
            "content": str,
            "placement_context": str,
            "reasoning": str
        }
        return self.extract_and_validate_json(llm_output, schema)
    
    def validate_review_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate review mode JSON output."""
        schema = {
            "needs_edit": bool,
            "edit_type": str,
            "content": str,
            "placement_context": str,
            "reasoning": str
        }
        valid, parsed, error = self.extract_and_validate_json(llm_output, schema)
        
        if valid:
            # Additional validation for edit_type field
            if parsed["edit_type"] not in ["replace", "delete", "none"]:
                return False, None, f"Invalid edit_type value: {parsed['edit_type']}. Must be 'replace', 'delete', or 'none'"
        
        return valid, parsed, error
    
    def validate_rigor_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate rigor enhancement JSON output."""
        schema = {
            "needs_enhancement": bool,
            "content": str,
            "placement_context": str,
            "reasoning": str
        }
        return self.extract_and_validate_json(llm_output, schema)
    
    def validate_compiler_validator_json(self, llm_output: str) -> Tuple[bool, Optional[Dict], str]:
        """Validate compiler validator JSON output."""
        schema = {
            "decision": str,
            "reasoning": str,
            "coherence_check": bool,
            "rigor_check": bool,
            "placement_check": bool,
        }
        valid, parsed, error = self.extract_and_validate_json(llm_output, schema)
        
        if valid:
            # Additional validation for decision field
            if parsed["decision"] not in ["accept", "reject"]:
                return False, None, f"Invalid decision value: {parsed['decision']}. Must be 'accept' or 'reject'"
            if not parsed["reasoning"].strip():
                return False, None, "reasoning must be a non-empty string"
        
        return valid, parsed, error


# Global JSON validator instance
json_validator = JSONValidator()

