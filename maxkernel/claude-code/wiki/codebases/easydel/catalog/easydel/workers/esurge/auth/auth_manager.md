---
title: 'Module: easydel/workers/esurge/auth/auth_manager.py'
type: catalog
provenance: extracted
module: easydel/workers/esurge/auth/auth_manager.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.esurge.auth.auth_manager`/
symbols:
  EnhancedApiKeyManager.create_api_key: EnhancedApiKeyManager#create_api_key().
  EnhancedApiKeyManager.update_key: EnhancedApiKeyManager#update_key().
  EnhancedApiKeyManager.rotate_key: EnhancedApiKeyManager#rotate_key().
  EnhancedApiKeyManager._log_audit: EnhancedApiKeyManager#_log_audit().
  EnhancedApiKeyManager.authorize_request: EnhancedApiKeyManager#authorize_request().
  EnhancedApiKeyManager.get_statistics: EnhancedApiKeyManager#get_statistics().
  EnhancedApiKeyManager._check_rate_limits: EnhancedApiKeyManager#_check_rate_limits().
  EnhancedApiKeyManager._check_quotas: EnhancedApiKeyManager#_check_quotas().
  EnhancedApiKeyManager.reactivate_key: EnhancedApiKeyManager#reactivate_key().
  EnhancedApiKeyManager._keys: EnhancedApiKeyManager#_keys.
  EnhancedApiKeyManager._load_from_storage: EnhancedApiKeyManager#_load_from_storage().
  EnhancedApiKeyManager.revoke_key: EnhancedApiKeyManager#revoke_key().
  EnhancedApiKeyManager.suspend_key: EnhancedApiKeyManager#suspend_key().
  EnhancedApiKeyManager.delete_key: EnhancedApiKeyManager#delete_key().
  EnhancedApiKeyManager.storage: EnhancedApiKeyManager#storage.
  EnhancedApiKeyManager._save_to_storage: EnhancedApiKeyManager#_save_to_storage().
  EnhancedApiKeyManager.record_usage: EnhancedApiKeyManager#record_usage().
  EnhancedApiKeyManager._record_token_rate_limit: EnhancedApiKeyManager#_record_token_rate_limit().
  EnhancedApiKeyManager.list_keys: EnhancedApiKeyManager#list_keys().
  EnhancedApiKeyManager._create_initial_admin_key: EnhancedApiKeyManager#_create_initial_admin_key().
  EnhancedApiKeyManager.generate_api_key: EnhancedApiKeyManager#generate_api_key().
  logger: logger.
  EnhancedApiKeyManager._mark_dirty_and_save: EnhancedApiKeyManager#_mark_dirty_and_save().
  EnhancedApiKeyManager.validate_key: EnhancedApiKeyManager#validate_key().
  RateLimitExceeded: RateLimitExceeded#
  EnhancedApiKeyManager._check_ip_permissions: EnhancedApiKeyManager#_check_ip_permissions().
  EnhancedApiKeyManager._key_id_to_hash: EnhancedApiKeyManager#_key_id_to_hash.
  EnhancedApiKeyManager._lock: EnhancedApiKeyManager#_lock.
  EnhancedApiKeyManager.get_audit_logs: EnhancedApiKeyManager#get_audit_logs().
  EnhancedApiKeyManager._auto_save_if_needed: EnhancedApiKeyManager#_auto_save_if_needed().
  EnhancedApiKeyManager.get_key_by_id: EnhancedApiKeyManager#get_key_by_id().
  QuotaExceeded: QuotaExceeded#
  PermissionDenied: PermissionDenied#
  EnhancedApiKeyManager._clean_window: EnhancedApiKeyManager#_clean_window().
  EnhancedApiKeyManager._audit_log: EnhancedApiKeyManager#_audit_log.
  EnhancedApiKeyManager._check_endpoint_permissions: EnhancedApiKeyManager#_check_endpoint_permissions().
  EnhancedApiKeyManager._check_model_permissions: EnhancedApiKeyManager#_check_model_permissions().
  EnhancedApiKeyManager._token_usage_windows: EnhancedApiKeyManager#_token_usage_windows.
  EnhancedApiKeyManager._raw_to_hash_cache: EnhancedApiKeyManager#_raw_to_hash_cache.
  EnhancedApiKeyManager.enabled: EnhancedApiKeyManager#enabled().
  EnhancedApiKeyManager._rate_limit_windows: EnhancedApiKeyManager#_rate_limit_windows.
  EnhancedApiKeyManager: EnhancedApiKeyManager#
  EnhancedApiKeyManager._hash_key: EnhancedApiKeyManager#_hash_key().
  EnhancedApiKeyManager.enable_audit_logging: EnhancedApiKeyManager#enable_audit_logging.
  EnhancedApiKeyManager.require_api_key: EnhancedApiKeyManager#require_api_key.
  EnhancedApiKeyManager.max_audit_entries: EnhancedApiKeyManager#max_audit_entries.
  EnhancedApiKeyManager.__init__: EnhancedApiKeyManager#__init__().
  EnhancedApiKeyManager.enable_persistence: EnhancedApiKeyManager#enable_persistence.
---
# Module: [`easydel/workers/esurge/auth/auth_manager.py`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py)

## Classes
### `EnhancedApiKeyManager`
- def: [`easydel/workers/esurge/auth/auth_manager.py:61`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L61)
- doc: Production-grade API key manager with security, RBAC, rate limiting, and audit logging.
- signature: `class EnhancedApiKeyManager:`
- members:
  - `__init__(self, require_api_key: bool = False, admin_key: str | None = None, enable_audit_logging: bool = True, max_audit_entries: int = 10000, storage_dir: str | Path | None = None, enable_persistence: bool = True, auto_save: bool = True, save_interval: float = 60)` — [`L75`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L75) — Initialize the enhanced API key manager.
  - `_auto_save_if_needed(self)` — [`L192`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L192) — Trigger auto-save if conditions are met.
  - `_check_endpoint_permissions(self, metadata: ApiKeyMetadata, endpoint: str | None)` — [`L532`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L532) — Check if the endpoint is allowed.
  - `_check_ip_permissions(self, metadata: ApiKeyMetadata, ip_address: str | None)` — [`L514`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L514) — Check if the IP address is allowed.
  - `_check_model_permissions(self, metadata: ApiKeyMetadata, model: str | None)` — [`L543`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L543) — Check if the model is allowed.
  - `_check_quotas(self, metadata: ApiKeyMetadata, requested_tokens: int)` — [`L611`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L611) — Check if quotas are exceeded.
  - `_check_rate_limits(self, metadata: ApiKeyMetadata, requested_tokens: int = 0)` — [`L554`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L554) — Check if rate limits are exceeded.
  - `_clean_window(self, window: deque, current_time: float, window_size: int)` — [`L643`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L643) — Remove expired entries from a time window.
  - `_create_initial_admin_key(self, key: str)` — [`L203`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L203) — Create initial admin key during initialization.
  - `_hash_key(self, key: str)` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L137) — Hash an API key using SHA-256.
  - `_load_from_storage(self)` — [`L148`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L148) — Load auth data from persistent storage.
  - `_log_audit(self, action: str, key_id: str | None = None, actor: str | None = None, ip_address: str | None = None, details: dict[str, tp.Any] | None = None, success: bool = True)` — [`L224`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L224) — Log an audit entry.
  - `_mark_dirty_and_save(self)` — [`L197`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L197) — Mark storage as dirty and trigger auto-save if needed.
  - `_record_token_rate_limit(self, metadata: ApiKeyMetadata, tokens: int)` — [`L687`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L687) — Record token usage for rate limiting.
  - `_save_to_storage(self)` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L171) — Save auth data to persistent storage.
  - `authorize_request(self, raw_key: str | None, ip_address: str | None = None, endpoint: str | None = None, model: str | None = None, requested_tokens: int = 0)` — [`L419`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L419) — Authorize a request and perform all security checks.
  - `create_api_key(self, raw_key: str, name: str, role: ApiKeyRole = ApiKeyRole.USER, description: str | None = None, created_by: str | None = None, expires_in_days: int | None = None, rate_limits: RateLimitConfig | None = None, quota: QuotaConfig | None = None, permissions: ApiKeyPermissions | None = None, tags: list[str] | None = None, metadata: dict[str, tp.Any] | None = None)` — [`L302`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L302) — Create an API key with a user-provided raw key.
  - `delete_key(self, key_id: str, deleted_by: str | None = None)` — [`L807`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L807) — Permanently delete an API key.
  - `enabled(self)` — [`L1077`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L1077) — Check if API key management is enabled.
  - `generate_api_key(self, name: str, role: ApiKeyRole = ApiKeyRole.USER, description: str | None = None, created_by: str | None = None, expires_in_days: int | None = None, rate_limits: RateLimitConfig | None = None, quota: QuotaConfig | None = None, permissions: ApiKeyPermissions | None = None, tags: list[str] | None = None, metadata: dict[str, tp.Any] | None = None)` — [`L256`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L256) — Generate a new random API key with configuration.
  - `get_audit_logs(self, limit: int = 100, key_id: str | None = None, action: str | None = None)` — [`L1019`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L1019) — Get audit log entries.
  - `get_key_by_id(self, key_id: str)` — [`L853`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L853) — Get key metadata by key ID.
  - `get_statistics(self)` — [`L1044`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L1044) — Get overall statistics about API keys and usage.
  - `list_keys(self, role: ApiKeyRole | None = None, status: ApiKeyStatus | None = None, tags: list[str] | None = None)` — [`L867`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L867) — List API keys with optional filtering.
  - `reactivate_key(self, key_id: str, reactivated_by: str | None = None)` — [`L774`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L774) — Reactivate a suspended API key.
  - `record_usage(self, raw_key: str | None, prompt_tokens: int, completion_tokens: int)` — [`L654`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L654) — Record token usage for a key.
  - `revoke_key(self, key_id: str, revoked_by: str | None = None)` — [`L708`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L708) — Revoke an API key.
  - `rotate_key(self, key_id: str, rotated_by: str | None = None)` — [`L973`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L973) — Rotate an API key (generate new key, preserve metadata).
  - `suspend_key(self, key_id: str, suspended_by: str | None = None)` — [`L741`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L741) — Suspend an API key (can be reactivated later).
  - `update_key(self, key_id: str, name: str | None = None, description: str | None = None, role: ApiKeyRole | None = None, expires_in_days: int | None = None, rate_limits: RateLimitConfig | None = None, quota: QuotaConfig | None = None, permissions: ApiKeyPermissions | None = None, tags: list[str] | None = None, metadata: dict[str, tp.Any] | None = None, updated_by: str | None = None)` — [`L895`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L895) — Update API key configuration.
  - `validate_key(self, raw_key: str | None)` — [`L390`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L390) — Validate a raw API key and return its metadata.
  - `enable_audit_logging` — [`L99`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L99)
  - `enable_persistence` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L101)
  - `max_audit_entries` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L100)
  - `require_api_key` — [`L98`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L98)
  - `storage` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L124)
- protocol/private: `_audit_log`[`L121`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L121), `_key_id_to_hash`[`L106`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L106), `_keys`[`L104`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L104), `_lock`[`L113`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L113), `_rate_limit_windows`[`L116`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L116), `_raw_to_hash_cache`[`L108`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L108), `_token_usage_windows`[`L118`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L118)
- uses (calls/refs, reference-scoped): [`ApiKeyMetadata`](auth_models.md#ApiKeyMetadata), [`ApiKeyRole`](auth_models.md#ApiKeyRole), [`ApiKeyStatus`](auth_models.md#ApiKeyStatus), [`save_keys`](auth_storage.md#AuthStorage.save_keys), [`permissions`](auth_models.md#ApiKeyMetadata.permissions), [`ApiKeyPermissions`](auth_models.md#ApiKeyPermissions), [`QuotaConfig`](auth_models.md#QuotaConfig), [`RateLimitConfig`](auth_models.md#RateLimitConfig), [`status`](auth_models.md#ApiKeyMetadata.status), [`logger`](auth_manager.md#logger), [`save_audit_logs`](auth_storage.md#AuthStorage.save_audit_logs), [`rate_limits`](auth_models.md#ApiKeyMetadata.rate_limits), [`load_audit_logs`](auth_storage.md#AuthStorage.load_audit_logs), [`load_keys`](auth_storage.md#AuthStorage.load_keys), [`role`](auth_models.md#ApiKeyMetadata.role), [`quota`](auth_models.md#ApiKeyMetadata.quota), [`key_id`](auth_models.md#ApiKeyMetadata.key_id), [`AuditLogEntry`](auth_models.md#AuditLogEntry), [`RateLimitExceeded`](auth_manager.md#RateLimitExceeded), [`is_active`](auth_models.md#ApiKeyMetadata.is_active), [`name`](auth_models.md#ApiKeyMetadata.name), [`save_usage_stats`](auth_storage.md#AuthStorage.save_usage_stats), [`should_auto_save`](auth_storage.md#AuthStorage.should_auto_save), [`PermissionDenied`](auth_manager.md#PermissionDenied), [`QuotaExceeded`](auth_manager.md#QuotaExceeded), [`expires_at`](auth_models.md#ApiKeyMetadata.expires_at), [`reset_monthly_counters_if_needed`](auth_models.md#ApiKeyMetadata.reset_monthly_counters_if_needed), [`tokens_per_day`](auth_models.md#RateLimitConfig.tokens_per_day), [`tokens_per_hour`](auth_models.md#RateLimitConfig.tokens_per_hour), [`tokens_per_minute`](auth_models.md#RateLimitConfig.tokens_per_minute), [`total_completion_tokens`](auth_models.md#ApiKeyMetadata.total_completion_tokens), [`total_prompt_tokens`](auth_models.md#ApiKeyMetadata.total_prompt_tokens), [`max_tokens_per_request`](auth_models.md#ApiKeyPermissions.max_tokens_per_request), [`max_total_requests`](auth_models.md#QuotaConfig.max_total_requests), [`max_total_tokens`](auth_models.md#QuotaConfig.max_total_tokens), [`monthly_request_limit`](auth_models.md#QuotaConfig.monthly_request_limit), [`monthly_requests`](auth_models.md#ApiKeyMetadata.monthly_requests), [`monthly_token_limit`](auth_models.md#QuotaConfig.monthly_token_limit), [`monthly_tokens`](auth_models.md#ApiKeyMetadata.monthly_tokens), [`requests_per_day`](auth_models.md#RateLimitConfig.requests_per_day)  (+29 more)
- used by: [`get_metrics`](../../../inference/esurge/server/api_server.md#eSurgeApiServer.get_metrics), [`_authorize_request`](../../../inference/esurge/server/api_server.md#eSurgeApiServer._authorize_request), [`auth_manager`](../../../inference/esurge/server/api_server.md#eSurgeApiServer.auth_manager), [`_auth_system_enabled`](../../../inference/esurge/server/api_server.md#eSurgeApiServer._auth_system_enabled), [`_record_api_key_usage`](../../../inference/esurge/server/api_server.md#eSurgeApiServer._record_api_key_usage), [`generate_api_key`](../../../inference/esurge/server/api_server.md#eSurgeApiServer.generate_api_key)

### `PermissionDenied`  ·  implements/extends Exception
- def: [`easydel/workers/esurge/auth/auth_manager.py:55`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L55)
- doc: Raised when permission check fails.
- signature: `class PermissionDenied(Exception):`
- used by: [`authorize_request`](auth_manager.md#EnhancedApiKeyManager.authorize_request), [`_authorize_request`](../../../inference/esurge/server/api_server.md#eSurgeApiServer._authorize_request)

### `QuotaExceeded`  ·  implements/extends Exception
- def: [`easydel/workers/esurge/auth/auth_manager.py:49`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L49)
- doc: Raised when quota limit is exceeded.
- signature: `class QuotaExceeded(Exception):`
- used by: [`_authorize_request`](../../../inference/esurge/server/api_server.md#eSurgeApiServer._authorize_request), [`_check_quotas`](auth_manager.md#EnhancedApiKeyManager._check_quotas)

### `RateLimitExceeded`  ·  implements/extends Exception
- def: [`easydel/workers/esurge/auth/auth_manager.py:43`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L43)
- doc: Raised when rate limit is exceeded.
- signature: `class RateLimitExceeded(Exception):`
- used by: [`_authorize_request`](../../../inference/esurge/server/api_server.md#eSurgeApiServer._authorize_request), [`_check_rate_limits`](auth_manager.md#EnhancedApiKeyManager._check_rate_limits)

## Module values
- `logger` — [`L40`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_manager.py#L40)

