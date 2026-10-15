import unittest

from backend.shared.json_parser import parse_json
from backend.shared.response_extraction import (
    drop_unfinished_thinking,
    extract_message_text,
    extract_response_text,
)


def _response(message, finish_reason=None):
    choice = {"message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {"choices": [choice]}


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

    def test_uses_reasoning_fallback_when_reply_finished_normally(self):
        text = extract_response_text(
            _response(
                {"content": "", "reasoning": '{"submission": "works"}'},
                finish_reason="stop",
            )
        )

        self.assertEqual(parse_json(text)["submission"], "works")

    def test_cut_off_reply_never_uses_unfinished_thinking_as_answer(self):
        for finish_reason in ("length", "max_tokens", "incomplete", "LENGTH"):
            for field in ("reasoning", "reasoning_content", "thinking"):
                with self.subTest(finish_reason=finish_reason, field=field):
                    text = extract_response_text(
                        _response(
                            {"content": "", field: "Let me think about the lemma first..."},
                            finish_reason=finish_reason,
                        )
                    )

                    self.assertEqual(text, "")

    def test_cut_off_reply_keeps_visible_answer_text(self):
        text = extract_response_text(
            _response(
                {"content": '{"submission": "visible"}', "reasoning": "private"},
                finish_reason="length",
            )
        )

        self.assertEqual(text, '{"submission": "visible"}')

    def test_drop_unfinished_thinking_strips_cut_off_empty_answers(self):
        response = _response(
            {
                "role": "assistant",
                "content": "",
                "reasoning": "unfinished thought",
                "reasoning_content": "unfinished thought",
                "thinking": "unfinished thought",
            },
            finish_reason="length",
        )

        drop_unfinished_thinking(response)

        message = response["choices"][0]["message"]
        self.assertEqual(message, {"role": "assistant", "content": ""})
        self.assertEqual(extract_message_text(message), "")

    def test_drop_unfinished_thinking_leaves_other_replies_alone(self):
        finished = _response({"content": "", "reasoning": '{"ok": true}'}, finish_reason="stop")
        answered = _response({"content": "answer", "reasoning": "thought"}, finish_reason="length")

        drop_unfinished_thinking(finished)
        drop_unfinished_thinking(answered)
        drop_unfinished_thinking(None)
        drop_unfinished_thinking({"choices": "not-a-list"})

        self.assertEqual(finished["choices"][0]["message"]["reasoning"], '{"ok": true}')
        self.assertEqual(answered["choices"][0]["message"]["reasoning"], "thought")

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

    def test_parses_lm_studio_legacy_channel_json_after_private_thought(self):
        text = (
            '<|channel>thought\n'
            'The model considered proof_044 and proof_011 first, with many '
            'private notes containing non-json braces like {not valid json}.\n'
            '<channel|>{"selected_search_ids":["proof_044","proof_011"],'
            '"reasoning":"Relevant modular certificate proofs."}'
        )

        parsed = parse_json(text)

        self.assertEqual(parsed["selected_search_ids"], ["proof_044", "proof_011"])
        self.assertEqual(parsed["reasoning"], "Relevant modular certificate proofs.")


if __name__ == "__main__":
    unittest.main()
