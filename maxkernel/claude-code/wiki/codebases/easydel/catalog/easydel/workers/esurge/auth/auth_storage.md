---
title: 'Module: easydel/workers/esurge/auth/auth_storage.py'
type: catalog
provenance: extracted
module: easydel/workers/esurge/auth/auth_storage.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.esurge.auth.auth_storage`/
symbols:
  AuthStorage._serialize_key_metadata: AuthStorage#_serialize_key_metadata().
  AuthStorage._deserialize_key_metadata: AuthStorage#_deserialize_key_metadata().
  logger: logger.
  AuthStorage.save_keys: AuthStorage#save_keys().
  AuthStorage._serialize_audit_log: AuthStorage#_serialize_audit_log().
  AuthStorage._deserialize_audit_log: AuthStorage#_deserialize_audit_log().
  AuthStorage.save_audit_logs: AuthStorage#save_audit_logs().
  AuthStorage.load_keys: AuthStorage#load_keys().
  AuthStorage.load_audit_logs: AuthStorage#load_audit_logs().
  AuthStorage.clear_all: AuthStorage#clear_all().
  AuthStorage.save_usage_stats: AuthStorage#save_usage_stats().
  AuthStorage.should_auto_save: AuthStorage#should_auto_save().
  AuthStorage._last_save: AuthStorage#_last_save.
  AuthStorage.keys_file: AuthStorage#keys_file.
  AuthStorage.load_usage_stats: AuthStorage#load_usage_stats().
  AuthStorage._lock: AuthStorage#_lock.
  AuthStorage.audit_logs_file: AuthStorage#audit_logs_file.
  AuthStorage.stats_file: AuthStorage#stats_file.
  AuthStorage.storage_dir: AuthStorage#storage_dir.
  AuthStorage._ensure_storage_dir: AuthStorage#_ensure_storage_dir().
  AuthStorage._atomic_write: AuthStorage#_atomic_write().
  AuthStorage: AuthStorage#
  AuthStorage.mark_dirty: AuthStorage#mark_dirty().
  AuthStorage._dirty: AuthStorage#_dirty.
  AuthStorage.auto_save: AuthStorage#auto_save.
  AuthStorage.save_interval: AuthStorage#save_interval.
  AuthStorage.__init__: AuthStorage#__init__().
---
# Module: [`easydel/workers/esurge/auth/auth_storage.py`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py)

## Classes
### `AuthStorage`
- def: [`easydel/workers/esurge/auth/auth_storage.py:40`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L40)
- doc: Persistent storage manager for authentication data.
- signature: `class AuthStorage:`
- members:
  - `__init__(self, storage_dir: str | Path | None = None, auto_save: bool = True, save_interval: float = 60)` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L60) — Initialize the auth storage manager.
  - `_atomic_write(self, file_path: Path, data: str)` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L106) — Write data to file atomically (write to temp, then rename).
  - `_deserialize_audit_log(self, data: dict[str, tp.Any])` — [`L381`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L381) — Deserialize JSON dict to AuditLogEntry.
  - `_deserialize_key_metadata(self, data: dict[str, tp.Any])` — [`L341`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L341) — Deserialize JSON dict to ApiKeyMetadata.
  - `_ensure_storage_dir(self)` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L97) — Create storage directory if it doesn't exist.
  - `_serialize_audit_log(self, log: AuditLogEntry)` — [`L369`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L369) — Serialize AuditLogEntry to JSON-compatible dict.
  - `_serialize_key_metadata(self, metadata: ApiKeyMetadata)` — [`L293`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L293) — Serialize ApiKeyMetadata to JSON-compatible dict.
  - `clear_all(self)` — [`L407`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L407) — Clear all stored data (for testing/reset).
  - `load_audit_logs(self)` — [`L232`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L232) — Load audit logs from disk.
  - `load_keys(self)` — [`L168`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L168) — Load API key metadata from disk.
  - `load_usage_stats(self)` — [`L276`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L276) — Load aggregated usage statistics from disk.
  - `mark_dirty(self)` — [`L393`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L393) — Mark data as needing save.
  - `save_audit_logs(self, logs: list[AuditLogEntry])` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L208) — Save audit logs to disk.
  - `save_keys(self, keys: dict[str, ApiKeyMetadata])` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L141) — Save API key metadata to disk.
  - `save_usage_stats(self, stats: dict[str, tp.Any])` — [`L255`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L255) — Save aggregated usage statistics to disk.
  - `should_auto_save(self)` — [`L397`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L397) — Check if auto-save should be triggered.
  - `audit_logs_file` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L84)
  - `auto_save` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L79)
  - `keys_file` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L83)
  - `save_interval` — [`L80`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L80)
  - `stats_file` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L85)
  - `storage_dir` — [`L78`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L78)
- protocol/private: `_dirty`[`L89`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L89), `_last_save`[`L90`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L90), `_lock`[`L88`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L88)
- uses (calls/refs, reference-scoped): [`ApiKeyMetadata`](auth_models.md#ApiKeyMetadata), [`ApiKeyRole`](auth_models.md#ApiKeyRole), [`logger`](auth_storage.md#logger), [`ApiKeyStatus`](auth_models.md#ApiKeyStatus), [`permissions`](auth_models.md#ApiKeyMetadata.permissions), [`ApiKeyPermissions`](auth_models.md#ApiKeyPermissions), [`QuotaConfig`](auth_models.md#QuotaConfig), [`RateLimitConfig`](auth_models.md#RateLimitConfig), [`status`](auth_models.md#ApiKeyMetadata.status), [`rate_limits`](auth_models.md#ApiKeyMetadata.rate_limits), [`role`](auth_models.md#ApiKeyMetadata.role), [`quota`](auth_models.md#ApiKeyMetadata.quota), [`key_id`](auth_models.md#ApiKeyMetadata.key_id), [`AuditLogEntry`](auth_models.md#AuditLogEntry), [`name`](auth_models.md#ApiKeyMetadata.name), [`expires_at`](auth_models.md#ApiKeyMetadata.expires_at), [`tokens_per_day`](auth_models.md#RateLimitConfig.tokens_per_day), [`tokens_per_hour`](auth_models.md#RateLimitConfig.tokens_per_hour), [`tokens_per_minute`](auth_models.md#RateLimitConfig.tokens_per_minute), [`total_completion_tokens`](auth_models.md#ApiKeyMetadata.total_completion_tokens), [`total_prompt_tokens`](auth_models.md#ApiKeyMetadata.total_prompt_tokens), [`max_tokens_per_request`](auth_models.md#ApiKeyPermissions.max_tokens_per_request), [`max_total_requests`](auth_models.md#QuotaConfig.max_total_requests), [`max_total_tokens`](auth_models.md#QuotaConfig.max_total_tokens), [`monthly_request_limit`](auth_models.md#QuotaConfig.monthly_request_limit), [`monthly_requests`](auth_models.md#ApiKeyMetadata.monthly_requests), [`monthly_token_limit`](auth_models.md#QuotaConfig.monthly_token_limit), [`monthly_tokens`](auth_models.md#ApiKeyMetadata.monthly_tokens), [`requests_per_day`](auth_models.md#RateLimitConfig.requests_per_day), [`requests_per_hour`](auth_models.md#RateLimitConfig.requests_per_hour), [`requests_per_minute`](auth_models.md#RateLimitConfig.requests_per_minute), [`tags`](auth_models.md#ApiKeyMetadata.tags), [`total_requests`](auth_models.md#ApiKeyMetadata.total_requests), [`allowed_endpoints`](auth_models.md#ApiKeyPermissions.allowed_endpoints), [`allowed_ip_addresses`](auth_models.md#ApiKeyPermissions.allowed_ip_addresses), [`allowed_models`](auth_models.md#ApiKeyPermissions.allowed_models), [`blocked_ip_addresses`](auth_models.md#ApiKeyPermissions.blocked_ip_addresses), [`description`](auth_models.md#ApiKeyMetadata.description), [`hashed_key`](auth_models.md#ApiKeyMetadata.hashed_key), [`key_prefix`](auth_models.md#ApiKeyMetadata.key_prefix)  (+15 more)
- used by: [`create_api_key`](auth_manager.md#EnhancedApiKeyManager.create_api_key), [`_load_from_storage`](auth_manager.md#EnhancedApiKeyManager._load_from_storage), [`storage`](auth_manager.md#EnhancedApiKeyManager.storage), [`_save_to_storage`](auth_manager.md#EnhancedApiKeyManager._save_to_storage), [`_mark_dirty_and_save`](auth_manager.md#EnhancedApiKeyManager._mark_dirty_and_save), [`_auto_save_if_needed`](auth_manager.md#EnhancedApiKeyManager._auto_save_if_needed)

## Module values
- `logger` — [`L37`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_storage.py#L37)

