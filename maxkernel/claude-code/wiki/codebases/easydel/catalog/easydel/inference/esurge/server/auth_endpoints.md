---
title: 'Module: easydel/inference/esurge/server/auth_endpoints.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/server/auth_endpoints.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.server.auth_endpoints`/
symbols:
  AuthEndpointsMixin.create_api_key_endpoint: AuthEndpointsMixin#create_api_key_endpoint().
  AuthEndpointsMixin.update_api_key_endpoint: AuthEndpointsMixin#update_api_key_endpoint().
  AuthEndpointsMixin._require_admin_role: AuthEndpointsMixin#_require_admin_role().
  AuthEndpointsMixin.list_api_keys_endpoint: AuthEndpointsMixin#list_api_keys_endpoint().
  CreateApiKeyRequest.role: CreateApiKeyRequest#role.
  AuthEndpointsMixin: AuthEndpointsMixin#
  UpdateApiKeyRequest.role: UpdateApiKeyRequest#role.
  AuthEndpointsMixin.get_api_key_endpoint: AuthEndpointsMixin#get_api_key_endpoint().
  AuthEndpointsMixin.revoke_api_key_endpoint: AuthEndpointsMixin#revoke_api_key_endpoint().
  AuthEndpointsMixin.suspend_api_key_endpoint: AuthEndpointsMixin#suspend_api_key_endpoint().
  AuthEndpointsMixin.reactivate_api_key_endpoint: AuthEndpointsMixin#reactivate_api_key_endpoint().
  AuthEndpointsMixin.delete_api_key_endpoint: AuthEndpointsMixin#delete_api_key_endpoint().
  AuthEndpointsMixin.rotate_api_key_endpoint: AuthEndpointsMixin#rotate_api_key_endpoint().
  AuthEndpointsMixin.get_api_key_stats_endpoint: AuthEndpointsMixin#get_api_key_stats_endpoint().
  AuthEndpointsMixin.get_audit_logs_endpoint: AuthEndpointsMixin#get_audit_logs_endpoint().
  CreateApiKeyRequest: CreateApiKeyRequest#
  UpdateApiKeyRequest: UpdateApiKeyRequest#
  CreateApiKeyRequest.name: CreateApiKeyRequest#name.
  CreateApiKeyRequest.description: CreateApiKeyRequest#description.
  CreateApiKeyRequest.expires_in_days: CreateApiKeyRequest#expires_in_days.
  CreateApiKeyRequest.requests_per_minute: CreateApiKeyRequest#requests_per_minute.
  CreateApiKeyRequest.requests_per_hour: CreateApiKeyRequest#requests_per_hour.
  CreateApiKeyRequest.requests_per_day: CreateApiKeyRequest#requests_per_day.
  CreateApiKeyRequest.tokens_per_minute: CreateApiKeyRequest#tokens_per_minute.
  CreateApiKeyRequest.tokens_per_hour: CreateApiKeyRequest#tokens_per_hour.
  CreateApiKeyRequest.tokens_per_day: CreateApiKeyRequest#tokens_per_day.
  CreateApiKeyRequest.max_total_tokens: CreateApiKeyRequest#max_total_tokens.
  CreateApiKeyRequest.max_total_requests: CreateApiKeyRequest#max_total_requests.
  CreateApiKeyRequest.monthly_token_limit: CreateApiKeyRequest#monthly_token_limit.
  CreateApiKeyRequest.monthly_request_limit: CreateApiKeyRequest#monthly_request_limit.
  CreateApiKeyRequest.allowed_models: CreateApiKeyRequest#allowed_models.
  CreateApiKeyRequest.allowed_endpoints: CreateApiKeyRequest#allowed_endpoints.
  CreateApiKeyRequest.allowed_ip_addresses: CreateApiKeyRequest#allowed_ip_addresses.
  CreateApiKeyRequest.blocked_ip_addresses: CreateApiKeyRequest#blocked_ip_addresses.
  CreateApiKeyRequest.enable_streaming: CreateApiKeyRequest#enable_streaming.
  CreateApiKeyRequest.enable_function_calling: CreateApiKeyRequest#enable_function_calling.
  CreateApiKeyRequest.max_tokens_per_request: CreateApiKeyRequest#max_tokens_per_request.
  CreateApiKeyRequest.tags: CreateApiKeyRequest#tags.
  CreateApiKeyRequest.metadata: CreateApiKeyRequest#metadata.
  UpdateApiKeyRequest.name: UpdateApiKeyRequest#name.
  UpdateApiKeyRequest.description: UpdateApiKeyRequest#description.
  UpdateApiKeyRequest.expires_in_days: UpdateApiKeyRequest#expires_in_days.
  UpdateApiKeyRequest.requests_per_minute: UpdateApiKeyRequest#requests_per_minute.
  UpdateApiKeyRequest.requests_per_hour: UpdateApiKeyRequest#requests_per_hour.
  UpdateApiKeyRequest.requests_per_day: UpdateApiKeyRequest#requests_per_day.
  UpdateApiKeyRequest.tokens_per_minute: UpdateApiKeyRequest#tokens_per_minute.
  UpdateApiKeyRequest.tokens_per_hour: UpdateApiKeyRequest#tokens_per_hour.
  UpdateApiKeyRequest.tokens_per_day: UpdateApiKeyRequest#tokens_per_day.
  UpdateApiKeyRequest.max_total_tokens: UpdateApiKeyRequest#max_total_tokens.
  UpdateApiKeyRequest.max_total_requests: UpdateApiKeyRequest#max_total_requests.
  UpdateApiKeyRequest.monthly_token_limit: UpdateApiKeyRequest#monthly_token_limit.
  UpdateApiKeyRequest.monthly_request_limit: UpdateApiKeyRequest#monthly_request_limit.
  UpdateApiKeyRequest.allowed_models: UpdateApiKeyRequest#allowed_models.
  UpdateApiKeyRequest.allowed_endpoints: UpdateApiKeyRequest#allowed_endpoints.
  UpdateApiKeyRequest.allowed_ip_addresses: UpdateApiKeyRequest#allowed_ip_addresses.
  UpdateApiKeyRequest.blocked_ip_addresses: UpdateApiKeyRequest#blocked_ip_addresses.
  UpdateApiKeyRequest.enable_streaming: UpdateApiKeyRequest#enable_streaming.
  UpdateApiKeyRequest.enable_function_calling: UpdateApiKeyRequest#enable_function_calling.
  UpdateApiKeyRequest.max_tokens_per_request: UpdateApiKeyRequest#max_tokens_per_request.
  UpdateApiKeyRequest.tags: UpdateApiKeyRequest#tags.
  UpdateApiKeyRequest.metadata: UpdateApiKeyRequest#metadata.
  ApiKeyResponse: ApiKeyResponse#
  ApiKeyResponse.key: ApiKeyResponse#key.
  ApiKeyResponse.key_id: ApiKeyResponse#key_id.
  ApiKeyResponse.key_prefix: ApiKeyResponse#key_prefix.
  ApiKeyResponse.name: ApiKeyResponse#name.
  ApiKeyResponse.description: ApiKeyResponse#description.
  ApiKeyResponse.role: ApiKeyResponse#role.
  ApiKeyResponse.status: ApiKeyResponse#status.
  ApiKeyResponse.created_at: ApiKeyResponse#created_at.
  ApiKeyResponse.expires_at: ApiKeyResponse#expires_at.
  ApiKeyResponse.last_used_at: ApiKeyResponse#last_used_at.
  ApiKeyResponse.total_requests: ApiKeyResponse#total_requests.
  ApiKeyResponse.total_tokens: ApiKeyResponse#total_tokens.
  ApiKeyResponse.message: ApiKeyResponse#message.
---
# Module: [`easydel/inference/esurge/server/auth_endpoints.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py)

## Classes
### `ApiKeyResponse`  ·  implements/extends BaseModel
- def: [`easydel/inference/esurge/server/auth_endpoints.py:143`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L143)
- doc: Response model for API key operations.
- signature: `class ApiKeyResponse(BaseModel):`
- members:
  - `created_at` — [`L177`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L177)
  - `description` — [`L174`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L174)
  - `expires_at` — [`L178`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L178)
  - `key` — [`L170`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L170)
  - `key_id` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L171)
  - `key_prefix` — [`L172`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L172)
  - `last_used_at` — [`L179`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L179)
  - `message` — [`L182`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L182)
  - `name` — [`L173`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L173)
  - `role` — [`L175`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L175)
  - `status` — [`L176`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L176)
  - `total_requests` — [`L180`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L180)
  - `total_tokens` — [`L181`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L181)

### `AuthEndpointsMixin`
- def: [`easydel/inference/esurge/server/auth_endpoints.py:185`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L185)
- doc: Mixin providing admin endpoints for API key management.
- signature: `class AuthEndpointsMixin:`
- members:
  - `_require_admin_role(self, raw_request: Request)` — [`L192`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L192) — Verify that the request is from an admin API key.
  - `create_api_key_endpoint(self, request: CreateApiKeyRequest, raw_request: Request)` — [`L237`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L237) — Admin endpoint to create a new API key.
  - `delete_api_key_endpoint(self, key_id: str, raw_request: Request)` — [`L642`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L642) — Admin endpoint to permanently delete an API key.
  - `get_api_key_endpoint(self, key_id: str, raw_request: Request)` — [`L366`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L366) — Admin endpoint to get details of a specific API key.
  - `get_api_key_stats_endpoint(self, raw_request: Request)` — [`L702`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L702) — Admin endpoint to get overall API key statistics.
  - `get_audit_logs_endpoint(self, raw_request: Request, limit: int = 100, key_id: str | None = None, action: str | None = None)` — [`L719`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L719) — Admin endpoint to get audit logs.
  - `list_api_keys_endpoint(self, raw_request: Request, role: str | None = None, status: str | None = None)` — [`L314`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L314) — Admin endpoint to list API keys.
  - `reactivate_api_key_endpoint(self, key_id: str, raw_request: Request)` — [`L617`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L617) — Admin endpoint to reactivate a suspended API key.
  - `revoke_api_key_endpoint(self, key_id: str, raw_request: Request)` — [`L567`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L567) — Admin endpoint to revoke an API key.
  - `rotate_api_key_endpoint(self, key_id: str, raw_request: Request)` — [`L667`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L667) — Admin endpoint to rotate an API key.
  - `suspend_api_key_endpoint(self, key_id: str, raw_request: Request)` — [`L592`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L592) — Admin endpoint to suspend an API key.
  - `update_api_key_endpoint(self, key_id: str, request: UpdateApiKeyRequest, raw_request: Request)` — [`L387`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L387) — Admin endpoint to update an API key.
- uses (calls/refs, reference-scoped): [`ApiKeyRole`](../../../workers/esurge/auth/auth_models.md#ApiKeyRole), [`ApiKeyStatus`](../../../workers/esurge/auth/auth_models.md#ApiKeyStatus), [`ApiKeyPermissions`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions), [`QuotaConfig`](../../../workers/esurge/auth/auth_models.md#QuotaConfig), [`RateLimitConfig`](../../../workers/esurge/auth/auth_models.md#RateLimitConfig), [`eSurgeApiServer`](api_server.md#eSurgeApiServer), [`tokens_per_day`](../../../workers/esurge/auth/auth_models.md#RateLimitConfig.tokens_per_day), [`tokens_per_hour`](../../../workers/esurge/auth/auth_models.md#RateLimitConfig.tokens_per_hour), [`tokens_per_minute`](../../../workers/esurge/auth/auth_models.md#RateLimitConfig.tokens_per_minute), [`max_tokens_per_request`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions.max_tokens_per_request), [`max_total_requests`](../../../workers/esurge/auth/auth_models.md#QuotaConfig.max_total_requests), [`max_total_tokens`](../../../workers/esurge/auth/auth_models.md#QuotaConfig.max_total_tokens), [`monthly_request_limit`](../../../workers/esurge/auth/auth_models.md#QuotaConfig.monthly_request_limit), [`monthly_token_limit`](../../../workers/esurge/auth/auth_models.md#QuotaConfig.monthly_token_limit), [`requests_per_day`](../../../workers/esurge/auth/auth_models.md#RateLimitConfig.requests_per_day), [`requests_per_hour`](../../../workers/esurge/auth/auth_models.md#RateLimitConfig.requests_per_hour), [`requests_per_minute`](../../../workers/esurge/auth/auth_models.md#RateLimitConfig.requests_per_minute), [`allowed_endpoints`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions.allowed_endpoints), [`allowed_ip_addresses`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions.allowed_ip_addresses), [`allowed_models`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions.allowed_models), [`blocked_ip_addresses`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions.blocked_ip_addresses), [`role`](auth_endpoints.md#CreateApiKeyRequest.role), [`enable_function_calling`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions.enable_function_calling), [`enable_streaming`](../../../workers/esurge/auth/auth_models.md#ApiKeyPermissions.enable_streaming), [`role`](auth_endpoints.md#UpdateApiKeyRequest.role), [`CreateApiKeyRequest`](auth_endpoints.md#CreateApiKeyRequest), [`UpdateApiKeyRequest`](auth_endpoints.md#UpdateApiKeyRequest), [`ADMIN`](../../../workers/esurge/auth/auth_models.md#ApiKeyRole.ADMIN), [`allowed_endpoints`](auth_endpoints.md#CreateApiKeyRequest.allowed_endpoints), [`allowed_endpoints`](auth_endpoints.md#UpdateApiKeyRequest.allowed_endpoints), [`allowed_ip_addresses`](auth_endpoints.md#CreateApiKeyRequest.allowed_ip_addresses), [`allowed_ip_addresses`](auth_endpoints.md#UpdateApiKeyRequest.allowed_ip_addresses), [`allowed_models`](auth_endpoints.md#CreateApiKeyRequest.allowed_models), [`allowed_models`](auth_endpoints.md#UpdateApiKeyRequest.allowed_models), [`blocked_ip_addresses`](auth_endpoints.md#CreateApiKeyRequest.blocked_ip_addresses), [`blocked_ip_addresses`](auth_endpoints.md#UpdateApiKeyRequest.blocked_ip_addresses), [`description`](auth_endpoints.md#CreateApiKeyRequest.description), [`description`](auth_endpoints.md#UpdateApiKeyRequest.description), [`enable_function_calling`](auth_endpoints.md#CreateApiKeyRequest.enable_function_calling), [`enable_function_calling`](auth_endpoints.md#UpdateApiKeyRequest.enable_function_calling)  (+32 more)
- used by: [`_endpoints`](api_server.md#eSurgeApiServer._endpoints), [`eSurgeApiServer`](api_server.md#eSurgeApiServer)

### `CreateApiKeyRequest`  ·  implements/extends BaseModel
- def: [`easydel/inference/esurge/server/auth_endpoints.py:49`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L49)
- doc: Request model for creating a new API key.
- signature: `class CreateApiKeyRequest(BaseModel):`
- members:
  - `allowed_endpoints` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L83)
  - `allowed_ip_addresses` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L84)
  - `allowed_models` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L82)
  - `blocked_ip_addresses` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L87)
  - `description` — [`L64`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L64)
  - `enable_function_calling` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L89)
  - `enable_streaming` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L88)
  - `expires_in_days` — [`L65`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L65)
  - `max_tokens_per_request` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L90)
  - `max_total_requests` — [`L77`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L77)
  - `max_total_tokens` — [`L76`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L76)
  - `metadata` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L94)
  - `monthly_request_limit` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L79)
  - `monthly_token_limit` — [`L78`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L78)
  - `name` — [`L62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L62)
  - `requests_per_day` — [`L70`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L70)
  - `requests_per_hour` — [`L69`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L69)
  - `requests_per_minute` — [`L68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L68)
  - `role` — [`L63`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L63)
  - `tags` — [`L93`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L93)
  - `tokens_per_day` — [`L73`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L73)
  - `tokens_per_hour` — [`L72`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L72)
  - `tokens_per_minute` — [`L71`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L71)
- uses (calls/refs, reference-scoped): [`ApiKeyRole`](../../../workers/esurge/auth/auth_models.md#ApiKeyRole), [`USER`](../../../workers/esurge/auth/auth_models.md#ApiKeyRole.USER)
- used by: [`create_api_key_endpoint`](auth_endpoints.md#AuthEndpointsMixin.create_api_key_endpoint)

### `UpdateApiKeyRequest`  ·  implements/extends BaseModel
- def: [`easydel/inference/esurge/server/auth_endpoints.py:97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L97)
- doc: Request model for updating an API key.
- signature: `class UpdateApiKeyRequest(BaseModel):`
- members:
  - `allowed_endpoints` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L131)
  - `allowed_ip_addresses` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L132)
  - `allowed_models` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L130)
  - `blocked_ip_addresses` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L133)
  - `description` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L111)
  - `enable_function_calling` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L135)
  - `enable_streaming` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L134)
  - `expires_in_days` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L113)
  - `max_tokens_per_request` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L136)
  - `max_total_requests` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L125)
  - `max_total_tokens` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L124)
  - `metadata` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L140)
  - `monthly_request_limit` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L127)
  - `monthly_token_limit` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L126)
  - `name` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L110)
  - `requests_per_day` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L118)
  - `requests_per_hour` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L117)
  - `requests_per_minute` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L116)
  - `role` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L112)
  - `tags` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L139)
  - `tokens_per_day` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L121)
  - `tokens_per_hour` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L120)
  - `tokens_per_minute` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/server/auth_endpoints.py#L119)
- uses (calls/refs, reference-scoped): [`ApiKeyRole`](../../../workers/esurge/auth/auth_models.md#ApiKeyRole)
- used by: [`update_api_key_endpoint`](auth_endpoints.md#AuthEndpointsMixin.update_api_key_endpoint)

