import ssl
import sys
import types
from unittest import TestCase, main, mock
from urllib.request import Request

import launcher_https


class LauncherHttpsTests(TestCase):
    def test_windows_context_uses_native_truststore(self) -> None:
        context = mock.Mock(verify_mode=ssl.CERT_REQUIRED, check_hostname=True)
        truststore_module = types.SimpleNamespace(
            SSLContext=mock.Mock(return_value=context)
        )
        with mock.patch.object(launcher_https.sys, "platform", "win32"):
            with mock.patch.dict(launcher_https.os.environ, {}, clear=True):
                with mock.patch.dict(sys.modules, {"truststore": truststore_module}):
                    with mock.patch.object(
                        launcher_https.ssl,
                        "create_default_context",
                    ) as default_context:
                        self.assertIs(
                            launcher_https.create_launcher_ssl_context(),
                            context,
                        )

        truststore_module.SSLContext.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
        default_context.assert_not_called()

    def test_windows_explicit_ca_config_keeps_stdlib_semantics(self) -> None:
        context = mock.Mock(verify_mode=ssl.CERT_REQUIRED, check_hostname=True)
        with mock.patch.object(launcher_https.sys, "platform", "win32"):
            with mock.patch.dict(
                launcher_https.os.environ,
                {"SSL_CERT_FILE": r"C:\certs\operator.pem"},
                clear=True,
            ):
                with mock.patch.object(
                    launcher_https.ssl,
                    "create_default_context",
                    return_value=context,
                ) as default_context:
                    self.assertIs(launcher_https.create_launcher_ssl_context(), context)

        default_context.assert_called_once_with()

    def test_windows_missing_truststore_keeps_verified_stdlib_fallback(self) -> None:
        context = mock.Mock(verify_mode=ssl.CERT_REQUIRED, check_hostname=True)
        real_import = __import__

        def import_without_truststore(name, *args, **kwargs):
            if name == "truststore":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(launcher_https.sys, "platform", "win32"):
            with mock.patch.dict(launcher_https.os.environ, {}, clear=True):
                with mock.patch("builtins.__import__", side_effect=import_without_truststore):
                    with mock.patch.object(
                        launcher_https.ssl,
                        "create_default_context",
                        return_value=context,
                    ):
                        self.assertIs(
                            launcher_https.create_launcher_ssl_context(),
                            context,
                        )

    def test_non_windows_context_keeps_default_verified_context(self) -> None:
        context = mock.Mock(verify_mode=ssl.CERT_REQUIRED, check_hostname=True)
        with mock.patch.object(launcher_https.sys, "platform", "linux"):
            with mock.patch.object(launcher_https.ssl, "create_default_context", return_value=context):
                self.assertIs(launcher_https.create_launcher_ssl_context(), context)


    def test_verified_urlopen_passes_verified_context(self) -> None:
        context = mock.Mock(verify_mode=ssl.CERT_REQUIRED, check_hostname=True)
        response = mock.Mock()
        request = Request("https://api.github.com/repos/example/project")
        with mock.patch.object(launcher_https, "create_launcher_ssl_context", return_value=context):
            with mock.patch.object(launcher_https, "urlopen", return_value=response) as open_url:
                self.assertIs(
                    launcher_https.verified_urlopen(request, timeout=30),
                    response,
                )

        open_url.assert_called_once_with(request, timeout=30, context=context)

    def test_verified_urlopen_rejects_http(self) -> None:
        with self.assertRaisesRegex(ValueError, "only accepts HTTPS"):
            launcher_https.verified_urlopen("http://example.com")

    def test_verified_urlopen_fails_closed_for_unverified_context(self) -> None:
        context = mock.Mock(verify_mode=ssl.CERT_NONE, check_hostname=False)
        with mock.patch.object(launcher_https, "create_launcher_ssl_context", return_value=context):
            with mock.patch.object(launcher_https, "urlopen") as open_url:
                with self.assertRaisesRegex(RuntimeError, "verification is not enabled"):
                    launcher_https.verified_urlopen("https://example.com")

        open_url.assert_not_called()


if __name__ == "__main__":
    main()
