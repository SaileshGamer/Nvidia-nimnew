from flask import Flask, request, jsonify, Response
import requests
import os
import json
import time

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Add retry logic
def make_nvidia_request(payload, max_retries=3):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                stream=payload.get("stream", False),
                timeout=120  # Increased timeout
            )
            
            # Log the response for debugging
            print(f"NVIDIA API Status Code: {response.status_code}")
            
            if response.status_code == 429:
                # Rate limited
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(retry_after)
                    continue
                return None, {
                    "error": {
                        "message": "Rate limit exceeded. Please try again later.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }, 429
            
            if response.status_code == 401:
                return None, {
                    "error": {
                        "message": "Invalid NVIDIA API key",
                        "type": "authentication_error"
                    }
                }, 401
            
            if response.status_code != 200:
                error_text = response.text
                print(f"NVIDIA API Error: {error_text}")
                return None, {
                    "error": {
                        "message": f"NVIDIA API error: {error_text}",
                        "type": "api_error",
                        "status_code": response.status_code
                    }
                }, response.status_code
            
            return response, None, 200
            
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, {
                "error": {
                    "message": "Request timeout. NVIDIA API took too long to respond.",
                    "type": "timeout_error"
                }
            }, 504
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, {
                "error": {
                    "message": "Connection error. Could not reach NVIDIA API.",
                    "type": "connection_error"
                }
            }, 503
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None, {
                "error": {
                    "message": f"Unexpected error: {str(e)}",
                    "type": "server_error"
                }
            }, 500
    
    return None, {
        "error": {
            "message": "Max retries exceeded",
            "type": "server_error"
        }
    }, 500

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "NVIDIA NIM Proxy API",
        "api_key_configured": bool(NVIDIA_API_KEY)
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    if not NVIDIA_API_KEY:
        return jsonify({
            "error": {
                "message": "NVIDIA_API_KEY not configured on server",
                "type": "configuration_error"
            }
        }), 500
    
    try:
        data = request.json
        
        # Validate request
        if not data or 'messages' not in data:
            return jsonify({
                "error": {
                    "message": "Missing 'messages' in request",
                    "type": "invalid_request_error"
                }
            }), 400
        
        # Transform OpenAI format to NVIDIA NIM format
        nvidia_payload = {
            "model": data.get("model", "meta/llama-3.1-70b-instruct"),  # Using 70B for better availability
            "messages": data.get("messages", []),
            "temperature": min(data.get("temperature", 0.7), 1.0),
            "top_p": data.get("top_p", 1),
            "max_tokens": min(data.get("max_tokens", 512), 2048),  # Reduced default
            "stream": data.get("stream", False)
        }
        
        print(f"Request to NVIDIA: {nvidia_payload['model']}")
        
        response, error, status_code = make_nvidia_request(nvidia_payload)
        
        if error:
            return jsonify(error), status_code
        
        if nvidia_payload["stream"]:
            def generate():
                try:
                    for chunk in response.iter_lines():
                        if chunk:
                            yield chunk + b'\n'
                except Exception as e:
                    print(f"Streaming error: {str(e)}")
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'.encode()
            
            return Response(generate(), content_type='text/event-stream')
        else:
            return jsonify(response.json()), 200
            
    except Exception as e:
        print(f"Error in chat_completions: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "meta/llama-3.1-70b-instruct",
                "object": "model",
                "owned_by": "meta"
            },
            {
                "id": "meta/llama-3.1-405b-instruct",
                "object": "model",
                "owned_by": "meta"
            },
            {
                "id": "mistralai/mixtral-8x7b-instruct-v0.1",
                "object": "model",
                "owned_by": "mistralai"
            },
            {
                "id": "meta/llama-3.1-8b-instruct",
                "object": "model",
                "owned_by": "meta"
            }
        ]
    })

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)    try:
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nvidia_payload,
            stream=nvidia_payload["stream"],
            timeout=60
        )
        
        if nvidia_payload["stream"]:
            def generate():
                for chunk in response.iter_lines():
                    if chunk:
                        yield chunk + b'\n'
            
            return Response(generate(), content_type='text/event-stream')
        else:
            return jsonify(response.json()), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "meta/llama-3.1-405b-instruct",
                "object": "model",
                "owned_by": "meta"
            },
            {
                "id": "meta/llama-3.1-70b-instruct",
                "object": "model",
                "owned_by": "meta"
            },
            {
                "id": "mistralai/mixtral-8x7b-instruct-v0.1",
                "object": "model",
                "owned_by": "mistralai"
            }
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
