# Day 3: Deploy Your Agent to Railway

Goal: Deploy your Day 2 agent (with memory and tools) to the cloud using Railway, making it accessible via REST API from anywhere.

## Quick Start

```bash
# 1. Login to Railway
railway login

# 2. Link to your project
cd day-3
railway link

# 3. Deploy (will fail - that's expected!)
railway up

# 4. Add your OpenAI API key
# Go to railway.app → Your Service → Variables → Add OPENAI_API_KEY

# 5. Deploy again (now it will work!)
railway up

# 6. Get your URL
railway domain

# 7. Test it
curl https://your-url.up.railway.app/health
```

## What You'll Learn

- **What is FastAPI?** How to turn Python code into a web service
- **What is REST API?** How to make your agent accessible via HTTP
- **Cloud Deployment:** How to deploy to Railway using CLI
- **Environment Variables:** How to configure secrets in production

## Understanding FastAPI

### What is FastAPI?

**FastAPI** is a Python web framework that creates REST APIs. Think of it as a translator:

```
Before (Day 1-2):
You → Terminal → Python script → Agent → Response
(Only works on your computer)

After (Day 3):
Anyone → Internet → HTTP Request → FastAPI → Agent → Response
(Works from anywhere in the world!)
```

### Why Do We Need It?

Without FastAPI, your agent only runs locally. With FastAPI:
- Anyone can use your agent via HTTP requests
- Works from terminal, browser, or other apps
- Can be integrated into websites or mobile apps
- Industry-standard way to expose AI services

## Deployment Steps

### Prerequisites

1. **Railway Account:** Sign up at [railway.app](https://railway.app)
2. **Railway CLI:** Install Railway CLI
   ```bash
   # macOS/Linux
   curl -fsSL https://railway.app/install.sh | sh
   
   # Or via npm
   npm install -g @railway/cli
   ```
3. **OpenAI API Key:** From Day 1-2

### Step 1: Deploy Your Agent

```bash
# Make sure you're in the day-3 directory
cd day-3

# Login to Railway (first time only)
railway login

# Create/link project
railway link

# First deployment (creates the service)
railway up
```

This will fail because OPENAI_API_KEY is missing - that's expected! This step creates the service.

### Step 2: Add Environment Variables

Now add your OpenAI API key so the agent can work.

**Option A: Via Railway Dashboard**
1. Go to [railway.app](https://railway.app)
2. Click your project → Click your service
3. Go to **Variables** tab
4. Click **+ New Variable**
5. Add `OPENAI_API_KEY` with your key value
6. (Optional) Add `SERPER_API_KEY` for web search tool
7. Railway automatically redeploys

**Option B: Via CLI**
```bash
# Set environment variables AFTER first deploy
railway variables --set "OPENAI_API_KEY=your-key-here"

# Optional: Add SERPER_API_KEY for web search
railway variables --set "SERPER_API_KEY=your-serper-key"

# Deploy again with variables
railway up
```

**What Railway does:**
1. Reads `railway.json` for config
2. Installs packages from `requirements.txt`
3. Starts your FastAPI server
4. Gives you a public URL

### Step 3: Get Your Public URL

After deployment completes (takes 2-3 minutes):

**Option A: CLI**
```bash
railway domain
```

**Option B: Dashboard**
- Go to railway.app → Your service → "Settings" tab
- Look for "Public URL" or click "Generate Domain"

**Example output:**
```
https://day-3-agent-production.up.railway.app
```

Save this URL - you'll use it to interact with your agent.

## Testing Your Deployed Agent

### Test 1: Health Check

```bash
curl https://your-app.up.railway.app/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "memory_enabled": true,
  "tools_count": 5
}
```

### Test 2: Query Your Agent

```bash
curl -X POST https://your-app.up.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 123 * 456?"}'
```

**Expected response:**
```json
{
  "answer": "The result of 123 * 456 is 56,088.",
  "timestamp": "2026-01-24T10:30:00Z",
  "processing_time": 2.5
}
```

### Test 3: Test Memory

```bash
# First request - tell the agent something
curl -X POST https://your-app.up.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "My name is Alex and I love Python"}'

# Second request - see if it remembers
curl -X POST https://your-app.up.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is my name?"}'
```

**Expected:** Agent should remember your name from the first request.
