from unittest import IsolatedAsyncioTestCase, mock

from backend.shared.free_model_manager import FreeModelManager, supports_text_chat_model
from backend.shared.openrouter_client import OpenRouterClient


def _free_model(model_id, *, architecture=None, context_length=1000):
    return {
        "id": model_id,
        "context_length": context_length,
        "pricing": {
            "prompt": "0",
            "completion": "0",
        },
        "architecture": architecture or {},
    }


def test_free_model_rotation_caches_only_text_chat_models():
    manager = FreeModelManager()
    manager.update_cached_models(
        [
            _free_model(
                "google/lyria-clip-preview:free",
                architecture={
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                },
                context_length=999999,
            ),
            _free_model(
                "image-only/free:free",
                architecture={
                    "input_modalities": ["image"],
                    "output_modalities": ["text"],
                },
                context_length=500000,
            ),
            _free_model(
                "openrouter/text-model:free",
                architecture={
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                context_length=1000,
            ),
            _free_model(
                "missing-modality/free:free",
                architecture={},
                context_length=750000,
            ),
        ]
    )

    assert (
        manager.get_alternative_free_model("exhausted/model:free")
        == "openrouter/text-model:free"
    )


def test_text_chat_filter_requires_modality_metadata():
    assert supports_text_chat_model(
        _free_model("chat/free:free", architecture={"modality": "text->text"})
    )
    assert not supports_text_chat_model(_free_model("missing-modality/free:free"))


def test_free_model_rotation_understands_compact_modality_string():
    manager = FreeModelManager()
    manager.update_cached_models(
        [
            _free_model(
                "image-generator/free:free",
                architecture={"modality": "text->image"},
                context_length=999999,
            ),
            _free_model(
                "chat/free:free",
                architecture={"modality": "text->text"},
                context_length=1000,
            ),
        ]
    )

    assert manager.get_alternative_free_model("exhausted/model:free") == "chat/free:free"


def test_failed_rotated_model_is_skipped_next_time():
    manager = FreeModelManager()
    manager.update_cached_models(
        [
            _free_model(
                "bad/free:free",
                architecture={"modality": "text->text"},
                context_length=2000,
            ),
            _free_model(
                "good/free:free",
                architecture={"modality": "text->text"},
                context_length=1000,
            ),
        ]
    )

    assert manager.get_alternative_free_model("original/free:free") == "bad/free:free"
    manager.mark_model_failed("bad/free:free")
    assert manager.get_alternative_free_model("original/free:free") == "good/free:free"


class OpenRouterFreeModelListTests(IsolatedAsyncioTestCase):
    async def test_free_only_list_returns_only_text_chat_models(self):
        client = OpenRouterClient("test-key")
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [
                _free_model(
                    "google/lyria-clip-preview:free",
                    architecture={"modality": "text->image"},
                ),
                _free_model(
                    "speech/free:free",
                    architecture={
                        "input_modalities": ["audio"],
                        "output_modalities": ["text"],
                    },
                ),
                _free_model(
                    "chat/free:free",
                    architecture={"modality": "text->text"},
                ),
                _free_model("missing-modality/free:free"),
                {
                    "id": "paid/chat",
                    "pricing": {"prompt": "0.1", "completion": "0.1"},
                    "architecture": {"modality": "text->text"},
                },
            ]
        }

        with mock.patch.object(client.client, "get", mock.AsyncMock(return_value=response)):
            models = await client.list_models(free_only=True, raise_on_error=True)

        await client.close()
        assert [model["id"] for model in models] == ["chat/free:free"]
