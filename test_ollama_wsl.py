#!/usr/bin/env python3
"""Quick test: can WSL reach Ollama via localhost?"""
import httpx
import asyncio

async def test():
    async with httpx.AsyncClient(timeout=30.0) as c:
        # Non-streaming test
        r = await c.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral:7b",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "stream": False,
            },
        )
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:300]}")

asyncio.run(test())
