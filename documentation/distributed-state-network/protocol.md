---
title: DSN Network Protocol
description: The wire protocol of the Distributed State Network — the transport, the packet formats, the encryption, and the signatures.
---

## Network protocol

### Transport layer

The network uses HTTP for all communication. Each node runs a threaded HTTP server. The server is based on `BaseHTTPRequestHandler`.

### Packet structure

A node sends an HTTP POST request to the path of the message type. The body of the request has this structure:

- A random IV of 16 bytes. The node makes a new IV for each message.
- The message type in 1 byte. The receiver uses this byte for verification.
- The payload of the message. The length of the payload is variable.

The body of the response has the same structure. The message type in the response is the same as the message type in the request.

**NOTE:** The node encrypts the body of the request and the body of the response only if the configuration has an AES key.

### Message types

| Type | Name | Path | Function |
|---|---|---|---|
| 1 | HELLO | `/hello` | The node sends its information and its credentials. |
| 2 | PEERS | `/peers` | The node asks for the list of peers, or sends the list. |
| 3 | UPDATE | `/update` | The node sends a change of its state. |
| 4 | PING | `/ping` | The node does a check of the connection. |
| 5 | DATA | `/data` | The node sends data to a different node. |

### Security

- The nodes encrypt all communication with AES-CBC and a shared AES-128 key.
- The node makes a new random IV for each message. The node puts the IV before the ciphertext.
- The node signs each packet with ECDSA. The receiver uses the signature for authentication.
- The encryption applies to the body of the request and to the body of the response.

**CAUTION:** Give the same AES-128 key to all the nodes in the network. A node that has a different key cannot communicate with the network.

### State synchronization

- Each node keeps a copy of the states of all its peers.
- A node sends each update to all the connected peers.
- Each update has a timestamp. A node ignores an update that is older than the data of the node.

### Configuration of the HTTP server

| Item | Value |
|---|---|
| Port | The value of the `port` parameter in the configuration |
| Threads | The server starts one thread for each request |
| Thread of the server | One daemon thread |
| Timeout of a request | 2 seconds minimum |
| Attempts | 3 maximum, with a delay of 0.5 seconds between the attempts |
| Network tick | 3 seconds |

For a large payload, the timeout increases. The rate of the increase is 1 second for each 1 MB.

### HTTP status codes

| Code | Name | Meaning |
|---|---|---|
| 200 | OK | The server accepted the request and sends response data. |
| 204 | No Content | The server accepted the request, but there is no response data. |
| 400 | Bad Request | The request is malformed, or the message type does not agree with the path. |
| 401 | Unauthorized | The key is incorrect, the node is unknown, or the node is not in the whitelist. |
| 404 | Not Found | The path is not a path of the protocol. |
| 406 | Not Acceptable | The signature is not correct, the update is stale, or the data is invalid. |
| 500 | Internal Server Error | An unexpected error occurred in the server. |
| 505 | HTTP Version Not Supported | The version of the node is different from the version of the peer. |

## Requirements

1. **Shared AES key**: All the nodes in the network must use the same AES-128 key. The key has 16 bytes, or 32 hexadecimal characters.
2. **Unique node IDs**: The `node_id` of each node must be different from the `node_id` of all the other nodes.
3. **Bootstrap nodes**: A node that joins an existing network must have a minimum of one bootstrap node.
4. **Network tick**: The network does maintenance checks every 3 seconds.
5. **Credentials**: The node makes the ECDSA keys automatically. The node stores the keys in the `credentials/` directory.
6. **Reliability**: A node makes a maximum of 3 attempts for each request that fails.
