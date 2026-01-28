"""
Test A2A Endpoint - Strict Mode Testing

This script demonstrates that:
1. /a2a REQUIRES @agent-id (no silent fallbacks)
2. /query is for direct questions
3. All A2A messages are logged
"""

import requests
import json

# Change this to your deployed URL
BASE_URL = "http://localhost:8000"

def print_response(response):
    """Pretty print response"""
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    print("-" * 80)

print("=" * 80)
print("Testing A2A Strict Mode")
print("=" * 80)

# Test 1: Direct query to /query endpoint (CORRECT)
print("\n✅ Test 1: Direct query to /query endpoint")
print("This should work - it's the correct endpoint for direct questions")
response = requests.post(
    f"{BASE_URL}/query",
    json={"question": "What is 2+2?"}
)
print_response(response)

# Test 2: Message without @agent-id to /a2a (SHOULD ERROR)
print("\n❌ Test 2: Message to /a2a without @agent-id")
print("This should FAIL - /a2a requires @agent-id")
response = requests.post(
    f"{BASE_URL}/a2a",
    json={
        "content": {
            "text": "What is the capital of France?",
            "type": "text"
        },
        "role": "user",
        "conversation_id": "test-no-target"
    }
)
print_response(response)

# Test 3: Register a dummy agent
print("\n✅ Test 3: Register a test agent")
DUMMY_AGENT_ID = "test-agent"
DUMMY_AGENT_URL = "https://httpbin.org/post"  # Echo service for testing

response = requests.post(
    f"{BASE_URL}/agents/register",
    params={
        "agent_id": DUMMY_AGENT_ID,
        "agent_url": DUMMY_AGENT_URL
    }
)
print_response(response)

# Test 4: Message with @agent-id to /a2a (SHOULD WORK)
print("\n✅ Test 4: Message to /a2a with @agent-id")
print(f"This should work - routing to @{DUMMY_AGENT_ID}")
response = requests.post(
    f"{BASE_URL}/a2a",
    json={
        "content": {
            "text": f"@{DUMMY_AGENT_ID} Can you help with this task?",
            "type": "text"
        },
        "role": "user",
        "conversation_id": "test-with-target"
    }
)
print_response(response)

# Test 5: Check logs
print("\n📝 Test 5: Check A2A logs")
print("After running these tests, check logs/a2a_messages.log to see:")
print("- INCOMING messages")
print("- ROUTING attempts")
print("- SUCCESS/ERROR logs")
print("- NO_TARGET errors for Test 2")
print("\nExample log entries:")
print("  INFO | INCOMING | conversation_id=test-with-target | message=@test-agent Can you help?")
print("  ERROR | NO_TARGET | conversation_id=test-no-target | message=What is the capital of France?")

print("\n" + "=" * 80)
print("Summary:")
print("  ✅ Use POST /query for direct questions to YOUR agent")
print("  ✅ Use POST /a2a ONLY for routing to other agents (requires @agent-id)")
print("  ❌ /a2a without @agent-id will ERROR - no silent fallbacks!")
print("  📝 All A2A activity is logged to logs/a2a_messages.log")
print("=" * 80)





