#!/usr/bin/env python3
"""Test chatting with the created Seed Agent"""

import json
import requests
import sys

BASE_URL = "http://localhost:8000/api/v1"

# Use the created holon
HOLON_ID = "holon_1d30e07827b4"

def chat_with_holon():
    """Send a test message to the holon"""
    print("=== Testing Chat with Seed Agent ===\n")

    # Test project request
    project_request = {
        "message": """Create a simple Todo List application specification.

Requirements:
1. Users can add, edit, delete tasks
2. Tasks have title, description, due date, priority
3. Filter tasks by status (pending, in-progress, done)
4. Simple UI (CLI or web-based)

Please provide:
1. Feature specification
2. Data model design
3. Implementation approach
4. Technology stack recommendation""",
        "conversation_id": "test-project-001",
        "context": {
            "project_name": "Simple Todo App",
            "project_type": "web_application",
            "complexity": "low"
        }
    }

    print(f"Sending request to Holon {HOLON_ID}...")
    print(f"Message: {project_request['message'][:100]}...")
    print()

    resp = requests.post(f"{BASE_URL}/chat/{HOLON_ID}", json=project_request)

    if resp.status_code == 200:
        data = resp.json()
        print(f"✓ Response received!")
        print(f"  - Holon ID: {data.get('holon_id')}")
        print(f"  - Route Decision: {data.get('route_decision')}")
        print(f"  - Latency: {data.get('latency_ms')}ms")
        print()
        print("=== Response Content ===")
        print(data.get('content', 'No content'))
        return True
    else:
        print(f"✗ Chat failed: {resp.status_code}")
        print(f"Response: {resp.text}")
        return False

if __name__ == "__main__":
    success = chat_with_holon()
    sys.exit(0 if success else 1)
