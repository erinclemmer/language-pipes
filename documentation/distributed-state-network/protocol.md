## Network Protocol

### Transport Layer
The network now uses **HTTP (HyperText Transfer Protocol)** with a Flask server for all communication. This provides:
- Standard HTTP semantics for request/response patterns
- Better compatibility with firewalls and proxies
- Built-in error handling with HTTP status codes
- Easy integration with monitoring and debugging tools
- More flexible for future enhancements

### Message Types and Endpoints
All HTTP requests use specific endpoints to identify the packet type:

- **POST /hello**: Exchange node information and credentials
- **POST /peers**: Request/share peer list
- **POST /update**: Send/receive state updates
- **POST /ping**: Connectivity check

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

### Security
- All communication is encrypted using AES with a shared key
- Messages are signed using ECDSA for authentication
- HTTP request/response bodies are encrypted end-to-end
- HTTPS/TLS can be added by placing a reverse proxy (nginx, apache) in front of the Flask server

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
3. **Port Availability**: Ensure the specified HTTP port is available before starting
4. **Bootstrap Nodes**: At least one bootstrap node is required to join an existing network
5. **Network Tick**: The network performs maintenance checks every 3 seconds
6. **Credential Management**: ECDSA keys are automatically generated and stored in `credentials/` directory
7. **HTTP Reliability**: The protocol implements retry logic (up to 3 attempts) for failed requests
8. **HTTPS Support**: While the base implementation uses HTTP, you can add HTTPS by:
   - Using a reverse proxy (nginx, apache) with SSL certificates
   - Configuring Flask with SSL context (requires certificate files)

## Error Handling

Common error codes:
- **401**: Not Authorized (signature verification failed)
- **406**: Not Acceptable (invalid data, stale update, or version mismatch)
- **505**: Version not supported

## Migration from UDP

If you're migrating from the UDP-based version:
1. Install Flask: The dependency is now included in pyproject.toml
2. Update firewall rules to allow HTTP traffic on your ports instead of UDP
3. The API remains largely the same, but transport is now HTTP
4. HTTP endpoints replace UDP message types for routing
5. Consider adding HTTPS via reverse proxy for production deployments

## Example URLs

For a node running on `192.168.1.100:8080`:
- HELLO endpoint: `http://192.168.1.100:8080/hello`
- PEERS endpoint: `http://192.168.1.100:8080/peers`
- UPDATE endpoint: `http://192.168.1.100:8080/update`
- PING endpoint: `http://192.168.1.100:8080/ping`

## Request Flow

1. **Client prepares message**: Message type byte + payload
2. **Client encrypts**: AES encryption applied to entire message
3. **Client sends**: HTTP POST to appropriate endpoint
4. **Server receives**: Flask routes to appropriate handler
5. **Server decrypts**: AES decryption applied
6. **Server validates**: Message type verification, signature checks
7. **Server processes**: Business logic executed
8. **Server prepares response**: Message type byte + response payload
9. **Server encrypts**: AES encryption applied
10. **Server sends**: HTTP response with encrypted body
11. **Client receives**: Response status and body
12. **Client decrypts**: AES decryption applied
13. **Client validates**: Message type verification
14. **Client processes**: Response data used
