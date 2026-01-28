# A2A Strict Mode - No Fallbacks, Real Errors Only

## 🎯 Philosophy

The `/a2a` endpoint is **ONLY** for Agent-to-Agent routing. No silent fallbacks, no lies, real errors only.

## ✅ The Right Way

### Direct Query to Your Agent
**Use:** `POST /query`

```bash
curl -X POST https://your-agent.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 2+2?"
  }'
```

**What happens:**
- ✅ Your agent processes the question
- ✅ Returns an answer
- ✅ Uses memory and tools
- ✅ Logged to standard logs

### Route Message to Another Agent
**Use:** `POST /a2a` with `@agent-id`

```bash
curl -X POST https://your-agent.railway.app/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "text": "@john-agent Can you help with this?",
      "type": "text"
    },
    "role": "user",
    "conversation_id": "collab-1"
  }'
```

**What happens:**
1. ✅ Your agent extracts `@john-agent`
2. ✅ Looks up john-agent's URL from registry
3. ✅ Forwards message to john-agent's `/a2a` endpoint
4. ✅ Returns john-agent's response
5. ✅ Logged to `logs/a2a_messages.log`

## ❌ The Wrong Way

### Using /a2a Without @agent-id

```bash
# ❌ THIS WILL ERROR
curl -X POST https://your-agent.railway.app/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "text": "What is 2+2?",
      "type": "text"
    },
    "role": "user",
    "conversation_id": "test"
  }'
```

**What happens:**
- ❌ Returns 400 Bad Request
- ❌ Error message explains the problem
- ❌ Logged as `NO_TARGET` error
- ❌ Does NOT fallback to local processing

**Response:**
```json
{
  "detail": "❌ ERROR: /a2a endpoint requires @agent-id for routing.\n\nYour message: 'What is 2+2?'\n\nThis endpoint is ONLY for agent-to-agent communication.\nYou must include @agent-id to route to another agent.\n\nFor direct queries to THIS agent, use POST /query instead."
}
```

## 📝 Logging

All A2A activity is logged to `logs/a2a_messages.log`:

### Successful Routing
```
2025-01-24 10:30:15 | INFO | INCOMING | conversation_id=test123 | message=@john-agent Can you help?
2025-01-24 10:30:15 | INFO | ROUTING | conversation_id=test123 | target=john-agent | message=Can you help?
2025-01-24 10:30:16 | INFO | SUCCESS | conversation_id=test123 | target=john-agent | response_length=245
```

### Failed Routing (No @agent-id)
```
2025-01-24 10:31:00 | INFO | INCOMING | conversation_id=test456 | message=What is 2+2?
2025-01-24 10:31:00 | ERROR | NO_TARGET | conversation_id=test456 | message=What is 2+2?
```

### Failed Routing (Agent Not Found)
```
2025-01-24 10:32:00 | INFO | INCOMING | conversation_id=test789 | message=@unknown-agent Help!
2025-01-24 10:32:00 | INFO | ROUTING | conversation_id=test789 | target=unknown-agent | message=Help!
2025-01-24 10:32:00 | ERROR | ERROR | conversation_id=test789 | error=Agent 'unknown-agent' not found
```

## 🧪 Testing

Use `test_a2a_strict.py` to see this behavior in action:

```bash
# Start your agent
uvicorn main:app --reload

# In another terminal
python test_a2a_strict.py
```

**Tests:**
1. ✅ `/query` with direct question (works)
2. ❌ `/a2a` without @agent-id (errors - expected!)
3. ✅ Register test agent (works)
4. ✅ `/a2a` with @agent-id (works)
5. 📝 Check logs for all activity

## 🤔 Why This Design?

### 1. Explicit is Better Than Implicit
- No guessing about what endpoint does
- Clear separation of concerns
- Easier to debug

### 2. Real Errors Help You Learn
- If you use wrong endpoint, you know immediately
- Error message tells you exactly what to do
- No silent failures

### 3. Clean Logs
- A2A routing logged separately from direct queries
- Easy to trace agent-to-agent communication
- Debug network of agents

### 4. Scales Better
- As you build more agents, clear boundaries matter
- No confusion about message flow
- Easier to build multi-agent systems

## 📊 Quick Reference

| What You Want | Endpoint | Requires @agent-id? | Logged To |
|---------------|----------|---------------------|-----------|
| Query your agent | `POST /query` | No | Standard logs |
| Route to another agent | `POST /a2a` | **Yes** | `logs/a2a_messages.log` |
| Get agent info | `GET /agentfacts` | No | Standard logs |
| Register agent | `POST /agents/register` | No | Standard logs |
| List agents | `GET /agents` | No | Standard logs |

## 🎓 Learning Outcomes

By using strict mode, you'll understand:
- ✅ Different types of agent communication
- ✅ When to use direct vs routed messages
- ✅ How to debug multi-agent systems
- ✅ The importance of clear APIs
- ✅ How to read logs effectively

**No fallbacks = Better understanding = Better agents!** 🚀





