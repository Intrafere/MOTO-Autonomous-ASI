"""Desktop subscription providers (OAuth logins and subscription API keys).

``metadata`` is the dependency-free list of every cloud provider and is safe to
import from models, route identity, and error helpers. API-key subscription
clients live in ``registry`` and are imported only by routing and REST code.
"""
