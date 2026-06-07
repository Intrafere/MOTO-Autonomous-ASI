import unittest

from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import extract_message_text, extract_response_text


def _response(message):
    return {"choices": [{"message": message}]}


class ResponseExtractionTests(unittest.TestCase):
    def test_prefers_content_over_reasoning_fallback(self):
        text = extract_response_text(
            _response({
                "content": '{"submission": "visible"}',
                "reasoning": '{"submission": "fallback"}',
            })
        )

        self.assertEqual(text, '{"submission": "visible"}')

    def test_uses_reasoning_when_provider_puts_only_answer_there(self):
        text = extract_response_text(
            _response({
                "content": "",
                "reasoning": '{"submission": "works", "reasoning": "schema field"}',
            })
        )

        parsed = parse_json(text)
        self.assertEqual(parsed["submission"], "works")
        self.assertEqual(parsed["reasoning"], "schema field")

    def test_can_disable_fallback_fields(self):
        text = extract_response_text(
            _response({
                "content": "",
                "reasoning": '{"submission": "hidden"}',
            }),
            allow_fallback_fields=False,
        )

        self.assertEqual(text, "")

    def test_supports_thinking_field_compatibility(self):
        text = extract_response_text(
            _response({
                "content": "",
                "thinking": '{"decision": "accept"}',
            })
        )

        self.assertEqual(parse_json(text)["decision"], "accept")

    def test_extracts_text_from_content_parts(self):
        text = extract_message_text({
            "content": [
                {"type": "text", "text": "part one"},
                {"type": "text", "text": " part two"},
            ]
        })

        self.assertEqual(text, "part one part two")


if __name__ == "__main__":
    unittest.main()
