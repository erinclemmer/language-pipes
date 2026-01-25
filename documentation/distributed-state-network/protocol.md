## Network Protocol

### Transport Layer
The network uses **HTTP** with a Flask server for all communication.

### Packet Structure
Each HTTP request follows this structure:
1. **HTTP POST request** to the appropriate endpoint
2. **Request Body**: AES Encrypted Payload containing:
   - Message type (1 byte) - for verification
   - Message payload (variable length)
3. **Response Body**: AES Encrypted Payload containing:
   - Message type (1 byte) - matches request type
   - Response payload (variable length)

### Message Types
Internal message type constants (used for verification):
- **Type 1 (HELLO)**: Exchange node information and credentials
- **Type 2 (PEERS)**: Request/share peer list
- **Type 3 (UPDATE)**: Send/receive state updates
- **Type 4 (PING)**: Connectivity check
- **Type 5 (DATA)**: Send data between nodes

### Security
- All communication is encrypted using AES with a shared key
- Messages are signed using ECDSA for authentication
- HTTP request/response bodies are encrypted end-to-end

### State Synchronization
- Nodes maintain a copy of all peers' states
- Updates are broadcast to all connected peers
- Timestamps prevent older updates from overwriting newer ones

### HTTP Server Configuration
- Flask server runs on specified port with threading enabled
- Default timeout is 5 seconds for requests
- Maximum retry attempts: 3 with 0.5 second delay between retries
- Server runs in a separate daemon thread

### HTTP Status Codes
- **200 OK**: Successful request with response data
- **204 No Content**: Successful request with no response data (e.g., some HELLO responses)
- **400 Bad Request**: Malformed request or message type mismatch
- **401 Unauthorized**: Signature verification failed
- **406 Not Acceptable**: Invalid data, stale update, or other validation error
- **500 Internal Server Error**: Unexpected server error
- **505 HTTP Version Not Supported**: Used for version mismatch errors

## Important Notes

1. **Shared AES Key**: All nodes in the network must use the same AES key file
2. **Unique Node IDs**: Each node must have a unique node_id
3. **Bootstrap Nodes**: At least one bootstrap node is required to join an existing network
4. **Network Tick**: The network performs maintenance checks every 3 seconds
5. **Credential Management**: ECDSA keys are automatically generated and stored in `credentials/` directory
6. **HTTP Reliability**: The protocol implements retry logic (up to 3 attempts) for failed requests

