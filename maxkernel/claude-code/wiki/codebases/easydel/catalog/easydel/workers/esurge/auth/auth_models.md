---
title: 'Module: easydel/workers/esurge/auth/auth_models.py'
type: catalog
provenance: extracted
module: easydel/workers/esurge/auth/auth_models.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.workers.esurge.auth.auth_models`/
symbols:
  ApiKeyMetadata.as_dict: ApiKeyMetadata#as_dict().
  ApiKeyMetadata: ApiKeyMetadata#
  ApiKeyRole: ApiKeyRole#
  ApiKeyStatus: ApiKeyStatus#
  ApiKeyMetadata.permissions: ApiKeyMetadata#permissions.
  ApiKeyPermissions.as_dict: ApiKeyPermissions#as_dict().
  RateLimitConfig: RateLimitConfig#
  QuotaConfig: QuotaConfig#
  ApiKeyPermissions: ApiKeyPermissions#
  ApiKeyMetadata.status: ApiKeyMetadata#status.
  AuditLogEntry.as_dict: AuditLogEntry#as_dict().
  RateLimitConfig.as_dict: RateLimitConfig#as_dict().
  ApiKeyMetadata.rate_limits: ApiKeyMetadata#rate_limits.
  ApiKeyMetadata.role: ApiKeyMetadata#role.
  ApiKeyMetadata.quota: ApiKeyMetadata#quota.
  ApiKeyMetadata.key_id: ApiKeyMetadata#key_id.
  QuotaConfig.as_dict: QuotaConfig#as_dict().
  ApiKeyMetadata.name: ApiKeyMetadata#name.
  ApiKeyMetadata.is_active: ApiKeyMetadata#is_active().
  AuditLogEntry: AuditLogEntry#
  ApiKeyMetadata.expires_at: ApiKeyMetadata#expires_at.
  ApiKeyMetadata.reset_monthly_counters_if_needed: ApiKeyMetadata#reset_monthly_counters_if_needed().
  RateLimitConfig.tokens_per_minute: RateLimitConfig#tokens_per_minute.
  RateLimitConfig.tokens_per_hour: RateLimitConfig#tokens_per_hour.
  RateLimitConfig.tokens_per_day: RateLimitConfig#tokens_per_day.
  ApiKeyMetadata.total_prompt_tokens: ApiKeyMetadata#total_prompt_tokens.
  ApiKeyMetadata.total_completion_tokens: ApiKeyMetadata#total_completion_tokens.
  RateLimitConfig.requests_per_minute: RateLimitConfig#requests_per_minute.
  RateLimitConfig.requests_per_hour: RateLimitConfig#requests_per_hour.
  RateLimitConfig.requests_per_day: RateLimitConfig#requests_per_day.
  QuotaConfig.max_total_tokens: QuotaConfig#max_total_tokens.
  QuotaConfig.max_total_requests: QuotaConfig#max_total_requests.
  QuotaConfig.monthly_token_limit: QuotaConfig#monthly_token_limit.
  QuotaConfig.monthly_request_limit: QuotaConfig#monthly_request_limit.
  ApiKeyPermissions.max_tokens_per_request: ApiKeyPermissions#max_tokens_per_request.
  ApiKeyMetadata.total_requests: ApiKeyMetadata#total_requests.
  ApiKeyMetadata.monthly_requests: ApiKeyMetadata#monthly_requests.
  ApiKeyMetadata.monthly_tokens: ApiKeyMetadata#monthly_tokens.
  ApiKeyMetadata.tags: ApiKeyMetadata#tags.
  ApiKeyRole.USER: ApiKeyRole#USER.
  ApiKeyPermissions.allowed_models: ApiKeyPermissions#allowed_models.
  ApiKeyPermissions.allowed_endpoints: ApiKeyPermissions#allowed_endpoints.
  ApiKeyPermissions.allowed_ip_addresses: ApiKeyPermissions#allowed_ip_addresses.
  ApiKeyPermissions.blocked_ip_addresses: ApiKeyPermissions#blocked_ip_addresses.
  ApiKeyMetadata.key_prefix: ApiKeyMetadata#key_prefix.
  ApiKeyMetadata.hashed_key: ApiKeyMetadata#hashed_key.
  ApiKeyMetadata.description: ApiKeyMetadata#description.
  ApiKeyMetadata.metadata: ApiKeyMetadata#metadata.
  ApiKeyMetadata.is_expired: ApiKeyMetadata#is_expired().
  ApiKeyMetadata.created_by: ApiKeyMetadata#created_by.
  ApiKeyMetadata.last_used_at: ApiKeyMetadata#last_used_at.
  ApiKeyMetadata.last_rotated_at: ApiKeyMetadata#last_rotated_at.
  ApiKeyMetadata.last_reset_month: ApiKeyMetadata#last_reset_month.
  AuditLogEntry.key_id: AuditLogEntry#key_id.
  AuditLogEntry.action: AuditLogEntry#action.
  ApiKeyStatus.ACTIVE: ApiKeyStatus#ACTIVE.
  ApiKeyPermissions.enable_streaming: ApiKeyPermissions#enable_streaming.
  ApiKeyPermissions.enable_function_calling: ApiKeyPermissions#enable_function_calling.
  ApiKeyMetadata.created_at: ApiKeyMetadata#created_at.
  AuditLogEntry.actor: AuditLogEntry#actor.
  AuditLogEntry.ip_address: AuditLogEntry#ip_address.
  AuditLogEntry.details: AuditLogEntry#details.
  AuditLogEntry.success: AuditLogEntry#success.
  ApiKeyMetadata.update_last_used: ApiKeyMetadata#update_last_used().
  ApiKeyStatus.REVOKED: ApiKeyStatus#REVOKED.
  AuditLogEntry.timestamp: AuditLogEntry#timestamp.
  ApiKeyRole.ADMIN: ApiKeyRole#ADMIN.
  ApiKeyStatus.SUSPENDED: ApiKeyStatus#SUSPENDED.
  ApiKeyRole.READONLY: ApiKeyRole#READONLY.
  ApiKeyRole.SERVICE: ApiKeyRole#SERVICE.
  ApiKeyStatus.EXPIRED: ApiKeyStatus#EXPIRED.
---
# Module: [`easydel/workers/esurge/auth/auth_models.py`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py)

## Classes
### `ApiKeyMetadata`
- def: [`easydel/workers/esurge/auth/auth_models.py:159`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L159) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
- doc: Complete metadata record for a managed API key.
- signature: `class ApiKeyMetadata:`
- members:
  - `as_dict(self, include_sensitive: bool = False)` — [`L245`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L245) — Serialize key metadata to dictionary. — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `is_active(self)` — [`L229`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L229) — Check if the key is active and usable. — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `is_expired(self)` — [`L223`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L223) — Check if the key has expired.
  - `reset_monthly_counters_if_needed(self)` — [`L237`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L237) — Reset monthly counters if we're in a new month.
  - `update_last_used(self)` — [`L233`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L233) — Update the last used timestamp.
  - `created_at` — [`L200`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L200)
  - `created_by` — [`L201`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L201)
  - `description` — [`L197`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L197)
  - `expires_at` — [`L202`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L202) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `hashed_key` — [`L195`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L195)
  - `key_id` — [`L193`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L193) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `key_prefix` — [`L194`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L194)
  - `last_reset_month` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L212)
  - `last_rotated_at` — [`L204`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L204)
  - `last_used_at` — [`L203`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L203)
  - `metadata` — [`L221`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L221)
  - `monthly_requests` — [`L210`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L210) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `monthly_tokens` — [`L211`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L211) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `name` — [`L196`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L196) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `permissions` — [`L217`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L217) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `quota` — [`L216`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L216) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `rate_limits` — [`L215`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L215) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `role` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L198) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `status` — [`L199`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L199) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `tags` — [`L220`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L220) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `total_completion_tokens` — [`L209`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L209) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `total_prompt_tokens` — [`L208`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L208) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `total_requests` — [`L207`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L207) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
- uses (calls/refs, reference-scoped): [`ApiKeyRole`](auth_models.md#ApiKeyRole), [`ApiKeyStatus`](auth_models.md#ApiKeyStatus), [`as_dict`](auth_models.md#ApiKeyPermissions.as_dict), [`ApiKeyPermissions`](auth_models.md#ApiKeyPermissions), [`QuotaConfig`](auth_models.md#QuotaConfig), [`RateLimitConfig`](auth_models.md#RateLimitConfig), [`as_dict`](auth_models.md#RateLimitConfig.as_dict), [`as_dict`](auth_models.md#QuotaConfig.as_dict), [`USER`](auth_models.md#ApiKeyRole.USER), [`ACTIVE`](auth_models.md#ApiKeyStatus.ACTIVE)
- used by: [`_serialize_key_metadata`](auth_storage.md#AuthStorage._serialize_key_metadata), [`_deserialize_metadata`](zmq_client.md#AuthWorkerClient._deserialize_metadata), [`_deserialize_key_metadata`](auth_storage.md#AuthStorage._deserialize_key_metadata), [`create_api_key`](auth_manager.md#EnhancedApiKeyManager.create_api_key), [`update_key`](auth_manager.md#EnhancedApiKeyManager.update_key), [`rotate_key`](auth_manager.md#EnhancedApiKeyManager.rotate_key), [`authorize_request`](auth_manager.md#EnhancedApiKeyManager.authorize_request), [`get_statistics`](auth_manager.md#EnhancedApiKeyManager.get_statistics), [`_check_quotas`](auth_manager.md#EnhancedApiKeyManager._check_quotas), [`_check_rate_limits`](auth_manager.md#EnhancedApiKeyManager._check_rate_limits), [`reactivate_key`](auth_manager.md#EnhancedApiKeyManager.reactivate_key), [`_keys`](auth_manager.md#EnhancedApiKeyManager._keys), [`_load_from_storage`](auth_manager.md#EnhancedApiKeyManager._load_from_storage), [`delete_key`](auth_manager.md#EnhancedApiKeyManager.delete_key), [`revoke_key`](auth_manager.md#EnhancedApiKeyManager.revoke_key), [`suspend_key`](auth_manager.md#EnhancedApiKeyManager.suspend_key), [`record_usage`](auth_manager.md#EnhancedApiKeyManager.record_usage), [`_record_token_rate_limit`](auth_manager.md#EnhancedApiKeyManager._record_token_rate_limit), [`save_keys`](auth_storage.md#AuthStorage.save_keys), [`generate_api_key`](zmq_client.md#AuthWorkerClient.generate_api_key), [`list_keys`](auth_manager.md#EnhancedApiKeyManager.list_keys), [`generate_api_key`](auth_manager.md#EnhancedApiKeyManager.generate_api_key), [`validate_key`](auth_manager.md#EnhancedApiKeyManager.validate_key), [`load_keys`](auth_storage.md#AuthStorage.load_keys), [`list_keys`](zmq_client.md#AuthWorkerClient.list_keys), [`_check_ip_permissions`](auth_manager.md#EnhancedApiKeyManager._check_ip_permissions), [`get_key_by_id`](auth_manager.md#EnhancedApiKeyManager.get_key_by_id), [`_check_endpoint_permissions`](auth_manager.md#EnhancedApiKeyManager._check_endpoint_permissions), [`_check_model_permissions`](auth_manager.md#EnhancedApiKeyManager._check_model_permissions), [`authorize_request`](zmq_client.md#AuthWorkerClient.authorize_request), [`get_key_by_id`](zmq_client.md#AuthWorkerClient.get_key_by_id), [`rotate_key`](zmq_client.md#AuthWorkerClient.rotate_key), [`validate_key`](zmq_client.md#AuthWorkerClient.validate_key)

### `ApiKeyPermissions`
- def: [`easydel/workers/esurge/auth/auth_models.py:122`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L122) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
- doc: Granular permissions for an API key.
- signature: `class ApiKeyPermissions:`
- members:
  - `as_dict(self)` — [`L145`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L145) — Serialize the permissions to a plain dictionary. — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `allowed_endpoints` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L138)
  - `allowed_ip_addresses` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L139)
  - `allowed_models` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L137)
  - `blocked_ip_addresses` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L140)
  - `enable_function_calling` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L142)
  - `enable_streaming` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L141)
  - `max_tokens_per_request` — [`L143`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L143)
- used by: [`create_api_key_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.create_api_key_endpoint), [`update_api_key_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.update_api_key_endpoint), [`_serialize_key_metadata`](auth_storage.md#AuthStorage._serialize_key_metadata), [`_deserialize_metadata`](zmq_client.md#AuthWorkerClient._deserialize_metadata), [`_deserialize_key_metadata`](auth_storage.md#AuthStorage._deserialize_key_metadata), [`create_api_key`](auth_manager.md#EnhancedApiKeyManager.create_api_key), [`as_dict`](auth_models.md#ApiKeyMetadata.as_dict), [`update_key`](auth_manager.md#EnhancedApiKeyManager.update_key), [`authorize_request`](auth_manager.md#EnhancedApiKeyManager.authorize_request), [`permissions`](auth_models.md#ApiKeyMetadata.permissions), [`generate_api_key`](zmq_client.md#AuthWorkerClient.generate_api_key), [`generate_api_key`](auth_manager.md#EnhancedApiKeyManager.generate_api_key), [`update_key`](zmq_client.md#AuthWorkerClient.update_key), [`_check_ip_permissions`](auth_manager.md#EnhancedApiKeyManager._check_ip_permissions), [`_check_endpoint_permissions`](auth_manager.md#EnhancedApiKeyManager._check_endpoint_permissions), [`_check_model_permissions`](auth_manager.md#EnhancedApiKeyManager._check_model_permissions)

### `ApiKeyRole`  ·  implements/extends StrEnum
- def: [`easydel/workers/esurge/auth/auth_models.py:25`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L25) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
- doc: Role-based access control levels for API keys.
- signature: `class ApiKeyRole(StrEnum):`
- members:
  - `ADMIN` — [`L35`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L35)
  - `READONLY` — [`L37`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L37)
  - `SERVICE` — [`L38`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L38)
  - `USER` — [`L36`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L36)
- used by: [`_deserialize_metadata`](zmq_client.md#AuthWorkerClient._deserialize_metadata), [`_deserialize_key_metadata`](auth_storage.md#AuthStorage._deserialize_key_metadata), [`create_api_key`](auth_manager.md#EnhancedApiKeyManager.create_api_key), [`update_key`](auth_manager.md#EnhancedApiKeyManager.update_key), [`generate_api_key`](zmq_client.md#AuthWorkerClient.generate_api_key), [`list_keys`](auth_manager.md#EnhancedApiKeyManager.list_keys), [`_create_initial_admin_key`](auth_manager.md#EnhancedApiKeyManager._create_initial_admin_key), [`generate_api_key`](auth_manager.md#EnhancedApiKeyManager.generate_api_key), [`_require_admin_role`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin._require_admin_role), [`role`](auth_models.md#ApiKeyMetadata.role), [`list_keys`](zmq_client.md#AuthWorkerClient.list_keys), [`update_key`](zmq_client.md#AuthWorkerClient.update_key), [`generate_api_key`](../../../inference/esurge/server/api_server.md#eSurgeApiServer.generate_api_key), [`list_api_keys_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.list_api_keys_endpoint), [`role`](../../../inference/esurge/server/auth_endpoints.md#CreateApiKeyRequest.role), [`role`](../../../inference/esurge/server/auth_endpoints.md#UpdateApiKeyRequest.role)

### `ApiKeyStatus`  ·  implements/extends StrEnum
- def: [`easydel/workers/esurge/auth/auth_models.py:41`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L41) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
- doc: API key lifecycle status.
- signature: `class ApiKeyStatus(StrEnum):`
- members:
  - `ACTIVE` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L51)
  - `EXPIRED` — [`L53`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L53)
  - `REVOKED` — [`L54`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L54)
  - `SUSPENDED` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L52)
- used by: [`_deserialize_metadata`](zmq_client.md#AuthWorkerClient._deserialize_metadata), [`_deserialize_key_metadata`](auth_storage.md#AuthStorage._deserialize_key_metadata), [`get_statistics`](auth_manager.md#EnhancedApiKeyManager.get_statistics), [`reactivate_key`](auth_manager.md#EnhancedApiKeyManager.reactivate_key), [`revoke_key`](auth_manager.md#EnhancedApiKeyManager.revoke_key), [`suspend_key`](auth_manager.md#EnhancedApiKeyManager.suspend_key), [`list_keys`](auth_manager.md#EnhancedApiKeyManager.list_keys), [`status`](auth_models.md#ApiKeyMetadata.status), [`list_keys`](zmq_client.md#AuthWorkerClient.list_keys), [`is_active`](auth_models.md#ApiKeyMetadata.is_active), [`list_api_keys_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.list_api_keys_endpoint)

### `AuditLogEntry`
- def: [`easydel/workers/esurge/auth/auth_models.py:284`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L284)
- doc: Audit log entry for tracking API key operations.
- signature: `class AuditLogEntry:`
- members:
  - `as_dict(self)` — [`L305`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L305) — Serialize the audit log entry to a plain dictionary.
  - `action` — [`L299`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L299)
  - `actor` — [`L300`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L300)
  - `details` — [`L302`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L302)
  - `ip_address` — [`L301`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L301)
  - `key_id` — [`L298`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L298)
  - `success` — [`L303`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L303)
  - `timestamp` — [`L297`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L297)
- used by: [`_log_audit`](auth_manager.md#EnhancedApiKeyManager._log_audit), [`_deserialize_audit_log`](auth_storage.md#AuthStorage._deserialize_audit_log), [`_serialize_audit_log`](auth_storage.md#AuthStorage._serialize_audit_log), [`save_audit_logs`](auth_storage.md#AuthStorage.save_audit_logs), [`load_audit_logs`](auth_storage.md#AuthStorage.load_audit_logs), [`get_audit_logs`](auth_manager.md#EnhancedApiKeyManager.get_audit_logs), [`_audit_log`](auth_manager.md#EnhancedApiKeyManager._audit_log)

### `QuotaConfig`
- def: [`easydel/workers/esurge/auth/auth_models.py:93`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L93) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
- doc: Usage quota limits for an API key.
- signature: `class QuotaConfig:`
- members:
  - `as_dict(self)` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L111) — Serialize the quota config to a plain dictionary. — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `max_total_requests` — [`L107`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L107)
  - `max_total_tokens` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L106)
  - `monthly_request_limit` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L109)
  - `monthly_token_limit` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L108)
- used by: [`create_api_key_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.create_api_key_endpoint), [`update_api_key_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.update_api_key_endpoint), [`_serialize_key_metadata`](auth_storage.md#AuthStorage._serialize_key_metadata), [`_deserialize_metadata`](zmq_client.md#AuthWorkerClient._deserialize_metadata), [`_deserialize_key_metadata`](auth_storage.md#AuthStorage._deserialize_key_metadata), [`create_api_key`](auth_manager.md#EnhancedApiKeyManager.create_api_key), [`as_dict`](auth_models.md#ApiKeyMetadata.as_dict), [`update_key`](auth_manager.md#EnhancedApiKeyManager.update_key), [`_check_quotas`](auth_manager.md#EnhancedApiKeyManager._check_quotas), [`generate_api_key`](zmq_client.md#AuthWorkerClient.generate_api_key), [`generate_api_key`](auth_manager.md#EnhancedApiKeyManager.generate_api_key), [`quota`](auth_models.md#ApiKeyMetadata.quota), [`update_key`](zmq_client.md#AuthWorkerClient.update_key)

### `RateLimitConfig`
- def: [`easydel/workers/esurge/auth/auth_models.py:58`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L58) — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
- doc: Rate limiting configuration for an API key.
- signature: `class RateLimitConfig:`
- members:
  - `as_dict(self)` — [`L80`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L80) — Serialize the rate limit config to a plain dictionary. — documented in [easydel-workers-esurge-auth-auth_models](../../../../../concepts/easydel-workers-esurge-auth-auth_models.md)
  - `requests_per_day` — [`L75`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L75)
  - `requests_per_hour` — [`L74`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L74)
  - `requests_per_minute` — [`L73`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L73)
  - `tokens_per_day` — [`L78`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L78)
  - `tokens_per_hour` — [`L77`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L77)
  - `tokens_per_minute` — [`L76`](../../../../../../../../raw/code/EasyDeL/easydel/workers/esurge/auth/auth_models.py#L76)
- used by: [`create_api_key_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.create_api_key_endpoint), [`update_api_key_endpoint`](../../../inference/esurge/server/auth_endpoints.md#AuthEndpointsMixin.update_api_key_endpoint), [`_serialize_key_metadata`](auth_storage.md#AuthStorage._serialize_key_metadata), [`_deserialize_metadata`](zmq_client.md#AuthWorkerClient._deserialize_metadata), [`_deserialize_key_metadata`](auth_storage.md#AuthStorage._deserialize_key_metadata), [`create_api_key`](auth_manager.md#EnhancedApiKeyManager.create_api_key), [`as_dict`](auth_models.md#ApiKeyMetadata.as_dict), [`update_key`](auth_manager.md#EnhancedApiKeyManager.update_key), [`_check_rate_limits`](auth_manager.md#EnhancedApiKeyManager._check_rate_limits), [`_record_token_rate_limit`](auth_manager.md#EnhancedApiKeyManager._record_token_rate_limit), [`generate_api_key`](zmq_client.md#AuthWorkerClient.generate_api_key), [`generate_api_key`](auth_manager.md#EnhancedApiKeyManager.generate_api_key), [`rate_limits`](auth_models.md#ApiKeyMetadata.rate_limits), [`update_key`](zmq_client.md#AuthWorkerClient.update_key)

