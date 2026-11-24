# ADR-005: Local REST/WebSocket API Security Architecture

## Status
Accepted

## Context
When QERLIO operates in developer or desktop server mode, it provides an HTTP/REST endpoint (compatible with OpenAI-style endpoints). Exposing an unauthenticated local server on a developer workstation allows malicious websites running in a local browser to issue cross-origin requests (CSRF / DNS rebinding) to execute prompts or read private model data.

## Decision
1. **Loopback Binding by Default:** The local server binds strictly to `127.0.0.1` (IPv4 loopback) or `::1` (IPv6 loopback). Binding to `0.0.0.0` is prohibited unless an explicit `--allow-remote-network` flag is supplied by the user.
2. **Ephemeral Bearer Authentication:** Every server launch generates a high-entropy random token saved in a protected local configuration file unless an explicit token is passed. All HTTP endpoints require `Authorization: Bearer <token>`.
3. **CORS Restrictions:** Default CORS configuration forbids arbitrary web origins (`Access-Control-Allow-Origin: *` is rejected). An explicit origin allowlist is enforced.
4. **Host Header Validation:** The HTTP server validates the `Host` header to mitigate DNS rebinding attacks.

## Consequences
- **Positive:** Protects the user from browser-based drive-by attacks and unauthorized local network access.
- **Negative:** Local developer scripts must read and pass the local authentication token.

## Implementation Status (v0.3.1)
`server/local_api/server.py` defaults `start_server(host="127.0.0.1")` and supports optional bearer authentication via `QERLIO_API_KEY`. Automatic ephemeral token generation on startup and strict CORS origin allowlisting remain target hardening items not yet enforced by default.
