from flask import Flask, request, jsonify, Response
import requests
import os
import json

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/')
def home():
    return jsonify({"status": "running", "message": "NVIDIA NIM Proxy API"})

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    if not NVIDIA_API_KEY:
        return jsonify({"error": "NVIDIA_API_KEY not configured"}), 500
    
    data = request.json
    
    # Transform OpenAI format to NVIDIA NIM format
    nvidia_payload = {
        "model": data.get("model", "meta/llama-3.1-405b-instruct"),
        "messages": data.get("messages", []),
        "temperature": data.get("temperature", 0.7),
        "top_p": data.get("top_p", 1),
        "max_tokens": data.get("max_tokens", 1024),
        "stream": data.get("stream", False)
    }
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
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
