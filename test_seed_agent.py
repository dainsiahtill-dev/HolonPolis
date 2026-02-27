#!/usr/bin/env python3
"""
HolonPolis Seed Agent 测试脚本
使用 llm_config.json 中的配置来：
1. 配置 Providers (Kimi, MiniMax, Ollama)
2. 创建 Seed Agent (PM + Director + QA 组合)
3. 运行一个测试项目
"""

import os
import sys

# 强制设置环境变量
os.environ["HOLONPOLIS_LLM_DEFAULT_PROVIDER"] = "minimax"
os.environ["MINIMAX_API_KEY"] = "sk-cp-HOoOELqEewj2Q0tEv1MitCjHiPiExAPZEUjopxuHxjiA5dgxt8ONNgGFxKIT0-AyM0_no1cdYeOpJuXgGJYdL0a7EcS4CHf-esZ__zm9T93CkSU_WsxMR0Y"
os.environ["MINIMAX_BASE_URL"] = "https://api.minimaxi.com/v1"
os.environ["MINIMAX_MODEL"] = "MiniMax-M2.5"

import asyncio
import json
import requests
from typing import Dict, Any, Optional

BASE_URL = "http://localhost:8000/api/v1"


def create_provider(provider_data: Dict[str, Any]) -> bool:
    """创建或更新 Provider"""
    provider_id = provider_data["provider_id"]

    # 检查是否已存在
    resp = requests.get(f"{BASE_URL}/providers/{provider_id}")
    if resp.status_code == 200:
        # 更新现有 provider
        resp = requests.patch(f"{BASE_URL}/providers/{provider_id}", json=provider_data)
        if resp.status_code == 200:
            print(f"  ✓ Updated provider: {provider_id}")
            return True
        else:
            print(f"  ✗ Failed to update {provider_id}: {resp.text}")
            return False
    else:
        # 创建新 provider
        resp = requests.post(f"{BASE_URL}/providers", json=provider_data)
        if resp.status_code == 200:
            print(f"  ✓ Created provider: {provider_id}")
            return True
        else:
            print(f"  ✗ Failed to create {provider_id}: {resp.text}")
            return False


def setup_providers():
    """根据 llm_config.json 配置 Providers"""
    print("\n=== Step 1: Configuring LLM Providers ===\n")

    providers = [
        # Kimi Coding (Anthropic Compatible)
        {
            "provider_id": "kimi-coding",
            "provider_type": "anthropic_compat",
            "name": "Kimi Coding",
            "base_url": "https://api.kimi.com/coding",
            "api_key": "sk-kimi-EmPrvGQU7aZW1FDx9WdhjccnhzcRLAobsQkBqpN7negLgVfrr7HEu0Eb4TebKGM6",
            "api_path": "/v1/chat/completions",
            "models_path": "/v1/models",
            "timeout": 360,
            "retries": 2,
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        # MiniMax M2.5
        {
            "provider_id": "minimax",
            "provider_type": "openai_compat",
            "name": "MiniMax-M2.5",
            "base_url": "https://api.minimaxi.com/v1",
            "api_key": "sk-cp-HOoOELqEewj2Q0tEv1MitCjHiPiExAPZEUjopxuHxjiA5dgxt8ONNgGFxKIT0-AyM0_no1cdYeOpJuXgGJYdL0a7EcS4CHf-esZ__zm9T93CkSU_WsxMR0Y",
            "api_path": "/v1/chat/completions",
            "models_path": "/v1/models",
            "timeout": 360,
            "retries": 2,
            "temperature": 0.1,
            "max_tokens": 16384,
        },
        # Ollama (Local)
        {
            "provider_id": "ollama-local",
            "provider_type": "ollama",
            "name": "Ollama-Qwen3-Coder",
            "base_url": "http://127.0.0.1:11434",
            "api_key": "",
            "api_path": "/api/chat",
            "models_path": "/v1/models",
            "timeout": 600,
            "retries": 1,
            "temperature": 0.1,
            "max_tokens": 16384,
        },
    ]

    success_count = 0
    for provider in providers:
        if create_provider(provider):
            success_count += 1

    print(f"\n  Configured {success_count}/{len(providers)} providers")
    return success_count == len(providers)


def create_seed_agent() -> Optional[str]:
    """创建一个 Seed Agent (PM role)"""
    print("\n=== Step 2: Creating Seed Agent ===\n")

    # 发送聊天请求来触发 Genesis 创建 Holon
    test_request = {
        "message": """I need you to act as a Product Manager (PM) Seed Agent.

Your role:
1. Analyze user requirements and break them down into actionable tasks
2. Create product specifications and roadmaps
3. Coordinate with other agents (Director, Architect, QA) to execute projects
4. Track project progress and ensure quality delivery

For this test, please:
1. Create a simple task management system blueprint
2. Define the core features and architecture
3. Estimate the effort required

Please respond with:
- Your understanding of the role
- A brief project plan for the task management system
- What other agents you would need to collaborate with""",
        "context": {
            "role": "pm",
            "provider": "minimax",
            "model": "MiniMax-M2.5",
            "project_type": "seed_agent_test"
        }
    }

    print("  Sending request to Genesis...")
    resp = requests.post(f"{BASE_URL}/chat", json=test_request)

    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✓ Seed Agent created!")
        print(f"    - Holon ID: {data.get('holon_id')}")
        print(f"    - Holon Name: {data.get('holon_name')}")
        print(f"    - Route Decision: {data.get('route_decision')}")
        print(f"    - Latency: {data.get('latency_ms')}ms")
        print(f"\n  --- Response Preview ---")
        content = data.get('content', '')
        print(content[:500] + "..." if len(content) > 500 else content)
        return data.get('holon_id')
    else:
        print(f"  ✗ Failed to create Seed Agent: {resp.status_code}")
        print(f"    Response: {resp.text}")
        return None


def run_project_test(holon_id: str):
    """运行一个测试项目"""
    print("\n=== Step 3: Running Test Project ===\n")

    # 测试项目：创建一个简单的待办事项应用
    project_request = {
        "message": """Let's create a simple Todo List application.

Requirements:
1. Users can add, edit, delete tasks
2. Tasks have title, description, due date, priority
3. Filter tasks by status (pending, in-progress, done)
4. Simple UI (can be CLI or web-based)

Please:
1. Create a detailed specification document
2. Design the data model
3. Outline the implementation steps
4. Suggest which technology stack to use
5. Estimate the development effort""",
        "conversation_id": "test-project-001",
        "context": {
            "project_name": "Simple Todo App",
            "project_type": "web_application",
            "complexity": "low"
        }
    }

    print(f"  Sending project request to Holon {holon_id}...")
    resp = requests.post(f"{BASE_URL}/chat/{holon_id}", json=project_request)

    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✓ Project response received!")
        print(f"    - Latency: {data.get('latency_ms')}ms")
        print(f"\n  --- Project Plan ---")
        content = data.get('content', '')
        print(content[:1000] + "..." if len(content) > 1000 else content)
        return True
    else:
        print(f"  ✗ Project failed: {resp.status_code}")
        print(f"    Response: {resp.text}")
        return False


def list_holons():
    """列出所有 Holons"""
    print("\n=== Current Holons ===\n")
    resp = requests.get(f"{BASE_URL}/holons")
    if resp.status_code == 200:
        holons = resp.json()
        if holons:
            for h in holons:
                print(f"  - {h['holon_id']}: {h['name']} ({h['species_id']})")
                print(f"    Purpose: {h['purpose'][:80]}...")
        else:
            print("  No holons found.")
    else:
        print(f"  Error: {resp.status_code}")


def health_check_providers():
    """检查 Providers 健康状态"""
    print("\n=== Provider Health Checks ===\n")

    providers = ["kimi-coding", "minimax", "ollama-local"]
    for provider_id in providers:
        resp = requests.post(f"{BASE_URL}/providers/{provider_id}/health")
        if resp.status_code == 200:
            data = resp.json()
            status = "✓ Healthy" if data.get('healthy') else "✗ Unhealthy"
            print(f"  {provider_id}: {status}")
            if not data.get('healthy'):
                print(f"    Error: {data.get('error', 'Unknown')}")
        else:
            print(f"  {provider_id}: ✗ Check failed ({resp.status_code})")


def main():
    """主测试流程"""
    print("=" * 60)
    print("HolonPolis Seed Agent Test")
    print("=" * 60)

    # 检查服务是否运行
    try:
        resp = requests.get("http://localhost:8000/health")
        if resp.status_code != 200:
            print("\n✗ HolonPolis server is not running!")
            print("  Please run: python run.py")
            return
        print("\n✓ HolonPolis server is running")
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to HolonPolis server!")
        print("  Please run: python run.py")
        return

    # 1. 配置 Providers
    if not setup_providers():
        print("\n⚠ Some providers failed to configure, continuing anyway...")

    # 2. 健康检查
    health_check_providers()

    # 3. 列出当前 Holons
    list_holons()

    # 4. 创建 Seed Agent
    holon_id = create_seed_agent()
    if not holon_id:
        print("\n✗ Failed to create Seed Agent, aborting.")
        return

    # 5. 运行测试项目
    run_project_test(holon_id)

    # 6. 再次列出 Holons
    list_holons()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
