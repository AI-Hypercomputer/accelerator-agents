---
title: 'Module: easydel/workers/esurge/auth/zmq_client.py'
type: catalog
provenance: extracted
module: easydel/workers/esurge/auth/zmq_client.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.esurge.auth.zmq_client`/
symbols:
  AuthWorkerClient._deserialize_metadata: AuthWorkerClient#_deserialize_metadata().
  AuthWorkerClient._request: AuthWorkerClient#_request().
  AuthWorkerClient.generate_api_key: AuthWorkerClient#generate_api_key().
  AuthWorkerClient.list_keys: AuthWorkerClient#list_keys().
  AuthWorkerClient.update_key: AuthWorkerClient#update_key().
  AuthWorkerClient._socket: AuthWorkerClient#_socket.
  AuthWorkerClient.validate_key: AuthWorkerClient#validate_key().
  AuthWorkerClient.authorize_request: AuthWorkerClient#authorize_request().
  AuthWorkerClient.get_key_by_id: AuthWorkerClient#get_key_by_id().
  AuthWorkerClient.rotate_key: AuthWorkerClient#rotate_key().
  AuthWorkerClient: AuthWorkerClient#
  AuthWorkerClient.shutdown: AuthWorkerClient#shutdown().
  AuthWorkerClient.close: AuthWorkerClient#close().
  AuthWorkerClient.record_usage: AuthWorkerClient#record_usage().
  AuthWorkerClient.revoke_key: AuthWorkerClient#revoke_key().
  AuthWorkerClient.suspend_key: AuthWorkerClient#suspend_key().
  AuthWorkerClient.reactivate_key: AuthWorkerClient#reactivate_key().
  AuthWorkerClient.delete_key: AuthWorkerClient#delete_key().
  AuthWorkerClient.get_audit_logs: AuthWorkerClient#get_audit_logs().
  AuthWorkerClient.get_statistics: AuthWorkerClient#get_statistics().
  PermissionDenied: PermissionDenied#
  RateLimitExceeded: RateLimitExceeded#
  QuotaExceeded: QuotaExceeded#
  AuthWorkerClient._context: AuthWorkerClient#_context.
  AuthWorkerClient._lock: AuthWorkerClient#_lock.
  AuthWorkerClient.__init__: AuthWorkerClient#__init__().
  AuthWorkerClient._endpoint: AuthWorkerClient#_endpoint.
  AuthWorkerClient.enabled: AuthWorkerClient#enabled().
---
# Module: [`easydel/workers/esurge/auth/zmq_client.py`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py)

## Classes
### `AuthWorkerClient`
- def: [`easydel/workers/esurge/auth/zmq_client.py:45`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L45)
- doc: Client for communicating with auth worker process via ZMQ.
- signature: `class AuthWorkerClient:`
- members:
  - `_deserialize_metadata(self, data: dict[str, tp.Any])` — [`L407`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L407) — Deserialize metadata dict to ApiKeyMetadata object.
  - `_request(self, payload: dict)` — [`L64`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L64) — Send a request to the worker and return the response.
  - `authorize_request(self, raw_key: str | None, ip_address: str | None = None, endpoint: str | None = None, model: str | None = None, requested_tokens: int = 0)` — [`L153`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L153) — Authorize a request and perform all security checks.
  - `close(self)` — [`L398`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L398) — Close the ZeroMQ socket.
  - `delete_key(self, key_id: str, deleted_by: str | None = None)` — [`L251`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L251) — Permanently delete an API key.
  - `enabled(self)` — [`L403`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L403) — Check if auth worker is enabled (always True for worker client).
  - `generate_api_key(self, name: str, role: ApiKeyRole = ApiKeyRole.USER, description: str | None = None, created_by: str | None = None, expires_in_days: int | None = None, rate_limits: RateLimitConfig | None = None, quota: QuotaConfig | None = None, permissions: ApiKeyPermissions | None = None, tags: list[str] | None = None, metadata: dict[str, tp.Any] | None = None)` — [`L93`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L93) — Generate a new API key.
  - `get_audit_logs(self, limit: int = 100, key_id: str | None = None, action: str | None = None)` — [`L361`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L361) — Get audit log entries.
  - `get_key_by_id(self, key_id: str)` — [`L264`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L264) — Get key metadata by key ID.
  - `get_statistics(self)` — [`L380`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L380) — Get overall statistics about API keys and usage.
  - `list_keys(self, role: ApiKeyRole | None = None, status: ApiKeyStatus | None = None, tags: list[str] | None = None)` — [`L277`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L277) — List API keys with optional filtering.
  - `reactivate_key(self, key_id: str, reactivated_by: str | None = None)` — [`L238`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L238) — Reactivate a suspended API key.
  - `record_usage(self, raw_key: str | None, prompt_tokens: int, completion_tokens: int)` — [`L190`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L190) — Record token usage for a key.
  - `revoke_key(self, key_id: str, revoked_by: str | None = None)` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L212) — Revoke an API key.
  - `rotate_key(self, key_id: str, rotated_by: str | None = None)` — [`L346`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L346) — Rotate an API key.
  - `shutdown(self)` — [`L389`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L389) — Shutdown the auth worker and close the connection.
  - `suspend_key(self, key_id: str, suspended_by: str | None = None)` — [`L225`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L225) — Suspend an API key.
  - `update_key(self, key_id: str, name: str | None = None, description: str | None = None, role: ApiKeyRole | None = None, expires_in_days: int | None = None, rate_limits: RateLimitConfig | None = None, quota: QuotaConfig | None = None, permissions: ApiKeyPermissions | None = None, tags: list[str] | None = None, metadata: dict[str, tp.Any] | None = None, updated_by: str | None = None)` — [`L296`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L296) — Update API key configuration.
  - `validate_key(self, raw_key: str | None)` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L140) — Validate a raw API key and return its metadata.
- protocol/private: `__init__`[`L55`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L55), `_context`[`L58`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L58), `_endpoint`[`L62`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L62), `_lock`[`L61`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L61), `_socket`[`L59`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L59)
- uses (calls/refs, reference-scoped): [`ApiKeyMetadata`](auth_models.md#ApiKeyMetadata), [`ApiKeyRole`](auth_models.md#ApiKeyRole), [`ApiKeyStatus`](auth_models.md#ApiKeyStatus), [`permissions`](auth_models.md#ApiKeyMetadata.permissions), [`ApiKeyPermissions`](auth_models.md#ApiKeyPermissions), [`QuotaConfig`](auth_models.md#QuotaConfig), [`RateLimitConfig`](auth_models.md#RateLimitConfig), [`status`](auth_models.md#ApiKeyMetadata.status), [`rate_limits`](auth_models.md#ApiKeyMetadata.rate_limits), [`role`](auth_models.md#ApiKeyMetadata.role), [`quota`](auth_models.md#ApiKeyMetadata.quota), [`key_id`](auth_models.md#ApiKeyMetadata.key_id), [`name`](auth_models.md#ApiKeyMetadata.name), [`expires_at`](auth_models.md#ApiKeyMetadata.expires_at), [`total_completion_tokens`](auth_models.md#ApiKeyMetadata.total_completion_tokens), [`total_prompt_tokens`](auth_models.md#ApiKeyMetadata.total_prompt_tokens), [`monthly_requests`](auth_models.md#ApiKeyMetadata.monthly_requests), [`monthly_tokens`](auth_models.md#ApiKeyMetadata.monthly_tokens), [`tags`](auth_models.md#ApiKeyMetadata.tags), [`total_requests`](auth_models.md#ApiKeyMetadata.total_requests), [`USER`](auth_models.md#ApiKeyRole.USER), [`description`](auth_models.md#ApiKeyMetadata.description), [`hashed_key`](auth_models.md#ApiKeyMetadata.hashed_key), [`key_prefix`](auth_models.md#ApiKeyMetadata.key_prefix), [`metadata`](auth_models.md#ApiKeyMetadata.metadata), [`created_by`](auth_models.md#ApiKeyMetadata.created_by), [`last_reset_month`](auth_models.md#ApiKeyMetadata.last_reset_month), [`last_rotated_at`](auth_models.md#ApiKeyMetadata.last_rotated_at), [`last_used_at`](auth_models.md#ApiKeyMetadata.last_used_at), [`created_at`](auth_models.md#ApiKeyMetadata.created_at), [`PermissionDenied`](zmq_client.md#PermissionDenied), [`QuotaExceeded`](zmq_client.md#QuotaExceeded), [`RateLimitExceeded`](zmq_client.md#RateLimitExceeded)
- used by: [`start`](worker_manager.md#AuthWorkerManager.start), [`shutdown`](worker_manager.md#AuthWorkerManager.shutdown), [`_auth_client`](worker_manager.md#AuthWorkerManager._auth_client), [`auth_client`](worker_manager.md#AuthWorkerManager.auth_client)

### `PermissionDenied`  ·  implements/extends Exception
- def: [`easydel/workers/esurge/auth/zmq_client.py:27`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L27)
- doc: Raised when permission check fails.
- signature: `class PermissionDenied(Exception):`
- used by: [`_request`](zmq_client.md#AuthWorkerClient._request)

### `QuotaExceeded`  ·  implements/extends Exception
- def: [`easydel/workers/esurge/auth/zmq_client.py:39`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L39)
- doc: Raised when quota limit is exceeded.
- signature: `class QuotaExceeded(Exception):`
- used by: [`_request`](zmq_client.md#AuthWorkerClient._request)

### `RateLimitExceeded`  ·  implements/extends Exception
- def: [`easydel/workers/esurge/auth/zmq_client.py:33`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/zmq_client.py#L33)
- doc: Raised when rate limit is exceeded.
- signature: `class RateLimitExceeded(Exception):`
- used by: [`_request`](zmq_client.md#AuthWorkerClient._request)

