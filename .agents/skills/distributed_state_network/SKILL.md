# Skill: distributed_state_network — Network Protocol

## Overview

The Distributed State Network (DSN) is a peer-to-peer state replication framework that runs over HTTP. Each node owns its own key-value state and automatically broadcasts updates to all peers. The protocol uses AES-128-CBC encryption for transport confidentiality and ECDSA (secp256k1) signatures for authentication. It is based on gossip-protocol research (RFC 677, Demers et al. epidemic algorithms) and is designed for local-network operation within Language Pipes.

All source code lives under `src/language_pipes/distributed_state_network/`. The public API is exported from `__init__.py`: `DSNodeServer`, `DSNode`, `DSNodeConfig`, `Endpoint`.

---

## Architecture Summary

```
distributed_state_network/
├── __init__.py                 # Public exports: DSNodeServer, DSNode, DSNodeConfig, Endpoint
├── handler.py                  # DSNodeServer: Flask HTTP server, routes, encrypt/decrypt dispatch
├── dsnode.py                   # DSNode: core protocol logic, state management, gossip, bootstrap
├── objects/
│   ├── config.py               # DSNodeConfig frozen dataclass
│   ├── endpoint.py             # Endpoint(address, port) frozen dataclass
│   ├── msg_types.py            # Message type constants (1–4; 5 defined in handler/dsnode)
│   ├── signed_packet.py        # Base class: ECDSA sign/verify via to_bytes(include_signature=False)
│   ├── hello_packet.py         # HELLO: version, node_id, connection endpoint, ECDSA pubkey, detected_address
│   ├── peers_packet.py         # PEERS: node_id, dict of node_id→Endpoint connections
│   ├── state_packet.py         # UPDATE: node_id, last_update timestamp, JSON state_data dict
│   └── data_packet.py          # DATA: node_id, arbitrary bytes payload
└── util/
    ├── __init__.py             # int_to_bytes, bytes_to_int, float_to_bytes, get_dict_hash, stop_thread
    ├── byte_helper.py          # ByteHelper: length-prefixed serialization (4-byte LE int lengths)
    ├── aes.py                  # AES-128-CBC encrypt/decrypt (PKCS7 padding, random IV per message)
    ├── ecdsa.py                # ECDSA secp256k1 key generation, SHA-256 message signing/verification
    └── key_manager.py          # CredentialManager: filesystem-backed ECDSA key storage
```

---

## Transport Layer

- **Protocol**: HTTP/1.1 over TCP
- **Server**: Flask, threaded mode, bound to `0.0.0.0:<port>`
- **Request method**: POST with `Content-Type: application/octet-stream`
- **Timeout**: 2 seconds per request (`HTTP_TIMEOUT` in `dsnode.py`)
- **Retry policy**: Up to 3 attempts total (initial + 2 retries), 0.5 s delay between retries

### HTTP Endpoints

Each message type maps to a dedicated Flask route:

| Path | Message Type | Constant |
|------|-------------|----------|
| `/hello` | HELLO | `MSG_HELLO = 1` |
| `/peers` | PEERS | `MSG_PEERS = 2` |
| `/update` | UPDATE | `MSG_UPDATE = 3` |
| `/ping` | PING | `MSG_PING = 4` |
| `/data` | DATA | `MSG_DATA = 5` |

The mapping is defined in `dsnode.py` as `MSG_TYPE_TO_PATH`.

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success with response body |
| 204 | Success, no response body (e.g., HELLO when no data to return) |
| 400 | Malformed request or message type mismatch |
| 401 | ECDSA signature verification failed or unknown sender |
| 406 | Invalid data, stale update, duplicate packet, or self-addressed update |
| 500 | Unexpected server error or server not running |
| 505 | Protocol version mismatch |

---

## Wire Format

### Encryption Envelope

When `aes_key` is configured (non-`None`), every HTTP request and response body is encrypted:

```
[IV: 16 bytes random] [AES-128-CBC ciphertext with PKCS7 padding]
```

- **Key**: Shared AES-128 key (16 bytes / 32 hex characters), identical across all nodes
- **IV**: Fresh `os.urandom(16)` per message, prepended to ciphertext
- **Padding**: PKCS7 block padding (128-bit block size)
- Implemented in `util/aes.py`: `aes_encrypt(key, data) -> bytes`, `aes_decrypt(key, ciphertext) -> bytes`

When `aes_key` is `None`, bodies are sent in plaintext.

### Message Framing

After decryption (or if unencrypted), the body is:

```
[msg_type: 1 byte] [payload: variable length]
```

- The **sender** prepends the message type byte before encrypting
- The **receiver** decrypts, reads the first byte, and verifies it matches the expected type for the route
- The **response** follows the same format: `[msg_type: 1 byte] [response_payload]`, then encrypted

### Binary Serialization (ByteHelper)

All packet payloads use `ByteHelper` (`util/byte_helper.py`) for length-prefixed TLV-style serialization:

| Operation | Wire Format |
|-----------|-------------|
| `write_int(i)` | 4 bytes, little-endian unsigned |
| `write_float(f)` | 8 bytes, big-endian double (`struct.pack(">d", f)`) |
| `write_string(s)` | `write_bytes(s.encode('utf-8'))` |
| `write_bytes(b)` | `[4-byte LE length] [raw bytes]` |

All fields are length-prefixed so the reader can deserialize without knowing the schema ahead of time.

---

## ECDSA Authentication

- **Curve**: secp256k1 (via the `ecdsa` Python library)
- **Signing**: `SHA-256(message_bytes)` → sign the hash with the private key
- **Verification**: Reconstruct `SHA-256(message_bytes)` → verify with stored public key
- **Key storage**: `CredentialManager` in `util/key_manager.py`, stores keys at `<credential_dir>/<node_id>/<target_node_id>.pub` and `<credential_dir>/<node_id>/<node_id>.key`

### SignedPacket Base Class

All packet types (`HelloPacket`, `PeersPacket`, `StatePacket`, `DataPacket`) extend `SignedPacket`:

```python
class SignedPacket:
    def sign(self, private_key: bytes):
        self.ecdsa_signature = sign_message(private_key, self.to_bytes(False))

    def verify_signature(self, public_key: bytes):
        return verify_signature(public_key, self.to_bytes(False), self.ecdsa_signature)
```

The signature covers `to_bytes(include_signature=False)` — i.e., all fields **except** the signature itself. The signature bytes are included in `to_bytes(include_signature=True)` for transmission.

---

## Message Types in Detail

### 1. HELLO (Type 1) — `/hello`

**Purpose**: Node introduction, version check, and ECDSA public key exchange.

**Request payload** (`HelloPacket.to_bytes()`):

| Field | Encoding | Description |
|-------|----------|-------------|
| `version` | `write_string` | Protocol version (e.g., `"0.7.0"`) |
| `node_id` | `write_string` | Sender's unique node ID |
| `connection.address` | `write_bytes` | Sender's address (or empty bytes if `None`) |
| `connection.port` | `write_int` | Sender's listening port |
| `ecdsa_public_key` | `write_bytes` | Sender's ECDSA secp256k1 public key |
| `ecdsa_signature` | `write_bytes` | Signature over all preceding fields |
| `detected_address` | `write_string` | Empty string in requests; filled by responder |

**Response payload** (`HelloPacket.to_bytes()`): Same structure as request, but from the receiving node's perspective. The `detected_address` field in the response contains the IP address the server detected for the requester (from `request.remote_addr`).

**Processing logic** (in `DSNode.handle_hello()`):
1. Deserialize `HelloPacket.from_bytes(data)`
2. Verify `version` matches — if mismatch, raise Exception(505)
3. Store/verify the sender's ECDSA public key via `cred_manager.ensure_public()`
4. Write the sender's endpoint to the address book using the detected IP and the sender's port
5. Initialize empty state for the sender if not already tracked
6. Return own `HelloPacket` with the sender's detected IP in `detected_address`

**Client-side** (`DSNode.send_hello()`):
1. Send own `HelloPacket` to the target endpoint
2. Parse the response `HelloPacket`
3. Verify version compatibility
4. Store the peer's public key
5. If the response includes a `detected_address`, update own address book entry with that IP
6. Write the peer's endpoint to the address book

---

### 2. PEERS (Type 2) — `/peers`

**Purpose**: Request/share the peer list (address book) for network discovery.

**Request payload** (`PeersPacket.to_bytes()`):

| Field | Encoding | Description |
|-------|----------|-------------|
| `node_id` | `write_string` | Requester's node ID |
| `ecdsa_signature` | `write_bytes` | Signature over `node_id` |
| `num_connections` | `write_int` | Number of connections (0 in request) |

**Response payload** (`PeersPacket.to_bytes()`):

| Field | Encoding | Description |
|-------|----------|-------------|
| `node_id` | `write_string` | Responder's node ID |
| `ecdsa_signature` | `write_bytes` | Signature over all fields |
| `num_connections` | `write_int` | Number of peer entries |
| Per entry: `key` | `write_string` | Peer's node ID |
| Per entry: `endpoint_bytes` | `write_bytes` | Serialized `Endpoint` (address string + port int) |

**Processing logic** (in `DSNode.handle_peers()`):
1. Deserialize and verify the requester is in the address book (401 if not)
2. Verify ECDSA signature (406 if invalid)
3. Return all known peers (excluding self) as a signed `PeersPacket`

**Client-side** (`DSNode.request_peers()`):
1. Send a signed `PeersPacket` with empty connections
2. Parse and verify the response signature
3. For each new peer: write to address book, send HELLO, exchange state updates

---

### 3. UPDATE (Type 3) — `/update`

**Purpose**: Exchange state data between nodes (gossip protocol).

**Request payload** (`StatePacket.to_bytes()`):

| Field | Encoding | Description |
|-------|----------|-------------|
| `node_id` | `write_string` | Origin node ID (state owner) |
| `last_update` | `write_float` | Unix timestamp of last state modification |
| `ecdsa_signature` | `write_bytes` | Signature over node_id + timestamp + state_data |
| `state_data` | `write_string` | JSON-encoded `Dict[str, str]` of the node's state |

**Response payload**: The receiver's own `StatePacket` (same format), enabling bidirectional state sync in one round-trip.

**Processing logic** (in `DSNode.handle_update()` → `DSNode.update_state()`):
1. Deserialize `StatePacket.from_bytes(data)`
2. Reject if `node_id` matches self (406 — self-addressed)
3. Verify ECDSA signature against stored public key (401 if invalid)
4. **Staleness check**: If the stored `last_update` timestamp is newer than the incoming one, discard (return `False`)
5. **Duplicate check**: If the SHA-256 hash of the incoming `state_data` dict matches the stored hash, discard (return `False`)
6. If accepted, replace the stored state and trigger `update_cb()` if set
7. Return own state as response

---

### 4. PING (Type 4) — `/ping`

**Purpose**: Connection health check.

**Request payload**: Single space byte (`b' '`)

**Response payload**: Empty bytes (`b''`)

**Processing logic**: The handler simply returns an empty response if the server is running. Used by the network tick to detect dead peers.

---

### 5. DATA (Type 5) — `/data`

**Purpose**: Arbitrary data transfer between nodes (application-level messaging).

**Request payload** (`DataPacket.to_bytes()`):

| Field | Encoding | Description |
|-------|----------|-------------|
| `node_id` | `write_string` | Sender's node ID |
| `ecdsa_signature` | `write_bytes` | Signature over node_id + data |
| `data` | `write_bytes` | Arbitrary bytes payload |

**Response payload**: `b'OK'` on success.

**Processing logic** (in `DSNode.receive_data()`):
1. Deserialize `DataPacket.from_bytes(data)`
2. Verify sender is in address book (401 if not)
3. Verify ECDSA signature (401 if invalid)
4. Invoke `receive_cb(pkt.data)` if callback is set
5. Return `b'OK'`

---

## Network Lifecycle

### Bootstrap Sequence

When a node starts with `bootstrap_nodes` configured, it calls `DSNode.bootstrap(con)` for the first reachable bootstrap endpoint:

```
1. HELLO  →  Bootstrap node       (exchange versions, pubkeys, detect IP)
2. UPDATE →  Bootstrap node       (exchange state; get bootstrap's state back)
3. PEERS  →  Bootstrap node       (get full peer list)
   └─ For each new peer:
      4a. HELLO  → New peer       (introduce self)
      4b. UPDATE → New peer       (exchange state)
```

After bootstrap, the node is fully connected and will participate in gossip.

### Network Tick (Gossip Loop)

Every **3 seconds** (`TICK_INTERVAL`), `DSNode.network_tick()` runs:

1. **`test_connections()`**: PING every known peer. If PING fails (after retries), remove the peer from `node_states` and `address_book`, then invoke `disconnect_cb()`.
2. **`gossip()`**: Randomly select one peer and send an UPDATE with own state. This implements the epidemic/gossip protocol — random pairwise exchanges ensure eventual consistency.

The tick runs in a daemon thread that re-spawns itself via `threading.Thread(target=self.network_tick, daemon=True).start()`.

### State Ownership Model

- Each node **owns** exactly one state: `node_states[self.config.node_id]`
- Only the owning node can modify its state (via `update_data(key, val)`)
- When state is modified, the `StatePacket` is re-signed and broadcast to all peers
- Remote states are read-only replicas stored in `node_states[remote_node_id]`
- Timestamps prevent stale overwrites; SHA-256 hash comparison prevents duplicate processing

### Update Broadcast

When `DSNode.update_data(key, val)` is called:
1. The local `StatePacket` is updated: new value set, `last_update` set to `time.time()`, packet re-signed
2. An UPDATE is sent to **every** known peer (not just a random one)

This is distinct from gossip, which sends only the local state to one random peer per tick.

---

## Credential Management

ECDSA keys are managed by `CredentialManager` (subclass of `KeyManager`):

- **Storage layout**: `<credential_dir>/<own_node_id>/<target_node_id>.pub` for public keys, `<own_node_id>.key` for private key
- **Key generation**: Automatic on first start via `generate_keys()` — skipped if key file already exists
- **Trust-on-first-use**: `ensure_public(node_id, pubkey)` stores the key on first contact; on subsequent contacts, it verifies the key matches the stored one (raises 401 if mismatch)
- **Default credential dir**: `"credentials"` (relative to working directory)

---

## Configuration Reference

`DSNodeConfig` is a frozen dataclass:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `node_id` | `str` | Yes | — | Unique identifier for this node |
| `port` | `int` | Yes | — | HTTP listening port |
| `credential_dir` | `str` | No | `"credentials"` | Directory for ECDSA key storage |
| `network_ip` | `Optional[str]` | No | `None` | Advertised IP; auto-detected via bootstrap if `None` |
| `aes_key` | `Optional[str]` | No | `None` | Hex-encoded AES-128 key (32 hex chars); `None` disables encryption |
| `bootstrap_nodes` | `List[Endpoint]` | Yes | — | Bootstrap endpoints; empty list for the first node |

Create from dict via `DSNodeConfig.from_dict(data)`.

---

## Key Constants

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `TICK_INTERVAL` | `3` (seconds) | `dsnode.py` | Gossip/health-check interval |
| `HTTP_TIMEOUT` | `2` (seconds) | `dsnode.py` | Per-request timeout |
| `AES_KEY_LENGTH` | `16` (bytes) | `util/aes.py` | AES-128 key size |
| `AES_BLOCK_SIZE` | `16` (bytes) | `util/aes.py` | AES block/IV size |
| `VERSION` | `"0.7.0"` | `handler.py` | Protocol version string |
| Max retries | `3` (total attempts) | `dsnode.py` | `send_http_request` retry logic |
| Retry delay | `0.5` (seconds) | `dsnode.py` | Between retry attempts |

---

## Implementation Notes / Gotchas (learned during whitelist feature work)

1. **Whitelist enforcement should happen in both directions**
   - **Inbound**: enforce in `DSNodeServer._handle_request(...)` using `remote_addr` *before* handler-specific parsing.
   - **Outbound**: enforce in `DSNode.send_http_request(...)` using `endpoint.address`.
   - Enforcing only in HELLO/update paths is not enough; all message types should be covered.

2. **`DSNodeConfig` constructor order matters in tests**
   - Dataclass positional order is:
     `node_id, credential_dir, port, network_ip, aes_key, whitelist_ips, bootstrap_nodes`.
   - Existing tests that instantiate `DSNodeConfig(...)` directly must be updated when new fields are added.

3. **`from_dict` should normalize optional list fields**
   - For `whitelist_ips`, use `[]` for missing or `None` values to avoid runtime `None` checks.

4. **Language Pipes wiring points for DSN config**
   - DSN config is created in multiple places, not just one:
     - `src/language_pipes/cli.py` (`serve` flow)
     - `src/language_pipes/commands/start.py` (wizard start flow)
   - Keep these in sync when adding DSN config fields.

5. **Config/env/CLI integration details**
   - Add CLI flag in `cli.py` (e.g. `--whitelist-ips`).
   - Add env support in `config.py` (e.g. `LP_WHITELIST_IPS`), and parse comma-separated values into a list.
   - Reflect new config in interactive UX (`initialize.py`, `edit.py`, `view.py`) and docs.

6. **Good test targets for network access controls**
   - Config parsing/default test (`whitelist_ips` present + missing default).
   - Inbound rejection test (non-whitelisted request to `/ping` returns 401).
   - End-to-end allow test (both nodes whitelist each other and connect).
   - Outbound restriction behavior (node with non-matching whitelist cannot establish peer connectivity).

---

## Useful Discovery Commands

```bash
# Find all message type references
grep -rn "MSG_" src/language_pipes/distributed_state_network/

# Find all HTTP route definitions
grep -rn "@self.app.route" src/language_pipes/distributed_state_network/handler.py

# Find all packet classes
grep -rn "class.*Packet" src/language_pipes/distributed_state_network/objects/

# Find signature verification points
grep -rn "verify_signature" src/language_pipes/distributed_state_network/

# Find state update logic
grep -rn "update_state\|handle_update" src/language_pipes/distributed_state_network/dsnode.py

# Run DSN tests
python -m pytest tests/distributed_state_network/ -v

# Find all error status codes raised in protocol handling
grep -rn "raise Exception" src/language_pipes/distributed_state_network/dsnode.py
```

---

## Test Suite

Tests live in `tests/distributed_state_network/` and cover:

| Test File | What It Tests |
|-----------|--------------|
| `test_connectivity.py` | Bootstrap, peer discovery, multi-node joining |
| `test_state_data.py` | State updates, reads, broadcast, staleness rejection |
| `test_callbacks.py` | Disconnect, update, and receive callback invocation |
| `test_disconnect.py` | Peer removal on PING failure |
| `test_security.py` | ECDSA signature verification, key trust-on-first-use |
| `test_aes.py` | AES encryption/decryption, key validation |
| `test_configuration.py` | DSNodeConfig creation, from_dict, validation |
| `test_error_handling.py` | Malformed packets, unknown senders, version mismatch |
| `test_http_endpoints.py` | Flask route handling, status codes |
| `test_wrapper_methods.py` | DSNodeServer convenience method pass-through |
| `base.py` | Shared test utilities and base class |
