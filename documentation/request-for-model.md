---
title: Request For Model Protocol
description: The peer-to-peer protocol for transferring installed model files between nodes, so a node can install a model from the network instead of downloading it from HuggingFace.
---

The Request For Model (RFM) protocol lets a node install a model by copying it from a peer that already has it, instead of downloading it from HuggingFace. This is useful when several machines on a LAN need the same model: only one node pays the internet download, and the rest pull the weights over the local network.

The protocol is implemented by `RequestForModelHandler` (`src/language_pipes/request_for_model/rfm.py`). Every node runs one handler instance, which acts as both a **requester** (asking for models) and a **provider** (serving models it has installed).


## How It Is Used

### From the TUI

Navigate to **Models → Installed** and start a new model install. If the node is connected to a network, the download page offers two methods:

- **Download from Huggingface** — fetches the model directly from the HuggingFace Hub.
- **Request model locally** — runs the RFM protocol against connected peers.

The page polls `RequestForModelHandler.download_status()` and renders progress as `Downloading: N of M files completed (P%)`, followed by `Computing metadata...` and finally a `SUCCESSFULLY Downloaded model` or `ERROR: ...` line.

### From code

`ContentProvider.request_model(model_id, token)` resets any previous request state and calls `RequestForModelHandler.request_model()`. The handler is wired to the router when the network starts (`ContentProvider.set_router`), using the router's `peers()` and `send_to_node()` functions.


## Transport

RFM packets travel over the [Distributed State Network](./distributed-state-network/protocol.md) data channel (`MSG_DATA`), so they get the same encryption, signing, and peer authentication as all other node traffic. The first integer of every data payload is a protocol number that `ContentProvider._receive_data` uses to dispatch:

| Protocol | Consumer |
|----------|----------|
| `0` | `JobReceiver` (inference jobs) |
| `1` | `RequestForModelHandler` |


## Protocol Flow

```
Requester (A)                                Provider (B)
     │                                            │
     │  1. fetch manifest from HuggingFace API    │
     │     (file names, sizes, sha256 hashes)     │
     │                                            │
     ├── WHO_HAS_MODEL ──────────────────────────►│  has model installed?
     │                                            │
     │◄────────────────────────── I_HAVE_MODEL ───┤  first responder wins
     │                                            │
     ├── READY_TO_RECEIVE ───────────────────────►│  starts send thread
     │                                            │
     │◄──────────────── SENDING_DATA (idx 0) ─────┤  32MB chunks per file
     │◄──────────────── SENDING_DATA (idx 1) ─────┤
     │◄──────────────── SENDING_DATA (EOF) ───────┤  zero-length terminator
     │      (hash + write file to disk)           │  ...next file...
     │                                            │
     │◄──────────────────────── DONE_SENDING ─────┤
     │                                            │
     │  2. verify all manifest files received     │
     │  3. compute model metadata                 │
```

1. **Manifest fetch.** The requester calls the HuggingFace API (`HfApi.model_info` with `files_metadata=True`). Small non-LFS files (configs, tokenizer files) are downloaded directly from HuggingFace on the spot. LFS files (the weights) become the **expected manifest**: a map of file name → size and SHA-256 hash. Only manifest files will be accepted from the peer.
2. **`WHO_HAS_MODEL`** is broadcast to every peer (skipping the node itself — the manifest fetch already created a partial model directory locally, so the node would otherwise answer its own request).
3. **`I_HAVE_MODEL`** is sent back by any provider that has the model in its installed list. The requester locks onto the first responder and ignores later ones.
4. **`READY_TO_RECEIVE`** tells the chosen provider to start. The provider streams on a background thread so the handshake packet itself gets an immediate response.
5. **`SENDING_DATA`** packets carry each file in 32MB chunks with increasing packet indices, followed by a zero-length packet with the `file_done` flag set as an end-of-file terminator. The provider iterates every regular file in the model's `data` directory (subdirectories such as HuggingFace's `.cache` are skipped); the requester silently rejects any file not in its manifest.
6. **`DONE_SENDING`** signals the end of the transfer. Once all in-flight file writes finish, the requester verifies that every manifest file was received (a partial transfer produces an `ERROR` status, never a false success), computes the model metadata, and reports success.


## Packet Formats

All packets are serialized with `ByteHelper` (`rfm_packets.py`) and start with the same header:

| Field | Type | Value |
|-------|------|-------|
| protocol | int | `1` |
| packet type | int | `RFMPacketType` value |

| Packet | Type | Body fields |
|--------|------|-------------|
| `WHO_HAS_MODEL` | 0 | `model_id: str` |
| `I_HAVE_MODEL` | 1 | `model_id: str` |
| `READY_TO_RECEIVE` | 2 | `model_id: str` |
| `SENDING_DATA` | 3 | `model_id: str`, `file_name: str`, `packet_idx: int`, `file_done: int (0/1)`, `packet_data: bytes` |
| `DONE_SENDING` | 4 | `model_id: str` |


## Receiving and Writing Files

The requester buffers incoming chunks in memory (`RFMRequestState.file_data`, keyed by file name and packet index) until a file's terminator arrives and all indices are contiguous. It then hashes and writes the file on a **background write thread** — never on the packet-handler thread. This matters because the sender's HTTP timeout for the tiny EOF packet is only a couple of seconds; blocking its response on a multi-gigabyte write would time the request out and derail the rest of the transfer.

Writing is atomic: content is streamed to a `.tmp` file, the SHA-256 digest is checked against the manifest, and only then is the file renamed into place. A failed write deletes the temp file and aborts the download with an `ERROR` status.

Completion is coordinated by two pieces of state (`RFMRequestState`, guarded by its lock):

- `pending_writes` — count of files fully received but still being written.
- `done_received` — whether `DONE_SENDING` has arrived.

Whichever event happens last (the DONE packet, or the final write finishing) triggers finalization. `DONE_SENDING` itself is handled on a background thread too, so its HTTP response returns before metadata computation begins.

### Validation on every `SENDING_DATA` packet

- The sender must be the locked-in provider node for the active request.
- The file name must appear in the expected manifest, and the resolved path must stay inside the model directory (path-traversal guard).
- Duplicate packet indices are rejected.
- Data packets may not overflow the file's manifest size.

Rejected packets are logged and dropped; they never abort the transfer.


## Timeouts and Failure Handling

- **Inactivity watchdog.** A monitor thread checks every 10 seconds; if no packet activity has been seen for 30 seconds (`INACTIVITY_TIMEOUT_SECONDS`) the download is reset with `ERROR: Download timed out due to inactivity`. The watchdog stands down while a file write is in flight, since a long hash-and-write is legitimate quiet time.
- **Sender aborts.** If a packet send fails mid-transfer (e.g. the requester disconnected), the provider's send thread logs the error and resets its send state so it can serve the next request.
- **Partial transfers.** If `DONE_SENDING` arrives but manifest files are missing, the requester reports `ERROR: Transfer ended with N expected file(s) missing` instead of success.
- **One at a time.** A node handles one outgoing request and one incoming send at a time; a `WHO_HAS_MODEL` received mid-send is ignored.


## Implementation Layout

| File | Contents |
|------|----------|
| `src/language_pipes/request_for_model/rfm.py` | `RequestForModelHandler` — all packet handlers, download/send threads, watchdog |
| `src/language_pipes/request_for_model/rfm_packets.py` | Packet classes and wire serialization |
| `src/language_pipes/request_for_model/state.py` | `RFMRequestState` (requester side) and `RFMSendState` (provider side) |
| `src/language_pipes/request_for_model/util.py` | `read_packet()` dispatch and `assert_fn` |
| `tests/language_pipes/unit/test_rfm.py` | Packet round-trips, handler unit tests, and a full two-handler transfer integration test |


## Current Limitations

- The requester buffers the entire model in RAM during the transfer; chunks are only flushed to disk when a whole file has arrived. Very large models on small machines can exhaust memory.
- The first peer to answer `WHO_HAS_MODEL` is used for the whole transfer; there is no multi-source download or mid-transfer failover.
- A HuggingFace manifest is still required to validate the transfer, so the requester needs internet access (and a token for gated models) even when the weights come from a peer.
