"""
Day 4: Agent-to-Agent Communication (A2A)
==========================================

This extends your Day 3 agent with A2A capabilities, allowing your agent
to communicate with other agents!

What's A2A?
- Agent-to-Agent communication protocol
- Agents can message each other using @agent-id syntax
- Simple, direct communication between agents
- Based on NEST/NANDA approach

Architecture:
- Service 1: Your agent (from Day 3)
- Service 2: Other agents (anywhere on the internet)
- They communicate via HTTP using the /a2a endpoint

⚠️ IMPORTANT: Two Endpoints for Different Purposes
=================================================

1. POST /query - Direct queries to YOUR agent
   Example: {"question": "What is 2+2?"}
   Use this when YOU want to ask YOUR agent something

2. POST /a2a - Agent-to-agent routing ONLY
   Example: {"content": {"text": "@other-agent Can you help?", "type": "text"}, ...}
   MUST include @agent-id to route to another agent
   Will ERROR if no @agent-id is provided (no silent fallbacks!)

Logging:
- All A2A messages logged to logs/a2a_messages.log
- Includes: INCOMING, ROUTING, SUCCESS, ERROR, NO_TARGET
- Check logs to debug A2A routing issues
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import os
import re
import httpx
import logging
import json
from typing import Optional, Dict, Any

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import FileReadTool, SerperDevTool, WebsiteSearchTool, YoutubeVideoSearchTool
from pydantic import Field
from typing import Type

# Load environment variables
load_dotenv()

# ==============================================================================
# Logging Setup
# ==============================================================================

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure A2A logging to file
a2a_logger = logging.getLogger("a2a")
a2a_logger.setLevel(logging.INFO)

# File handler for A2A messages
a2a_file_handler = logging.FileHandler("logs/a2a_messages.log")
a2a_file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
a2a_file_handler.setFormatter(formatter)

# Add handler to logger
a2a_logger.addHandler(a2a_file_handler)

# ==============================================================================
# FastAPI Application Setup
# ==============================================================================

app = FastAPI(
    title="Personal Agent Twin API with A2A",
    description="Your agent with memory, tools, AND agent-to-agent communication!",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Request/Response Models
# ==============================================================================

class QueryRequest(BaseModel):
    """Standard query request"""
    question: str
    user_id: str = "anonymous"

class QueryResponse(BaseModel):
    """Standard query response"""
    answer: str
    timestamp: str
    processing_time: float

class A2AMessage(BaseModel):
    """A2A message format (NEST-style)"""
    content: Dict[str, Any]  # {"text": "message", "type": "text"}
    role: str = "user"
    conversation_id: str

class A2AResponse(BaseModel):
    """A2A response format"""
    content: Dict[str, Any]
    role: str = "assistant"
    conversation_id: str
    timestamp: str
    agent_id: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    memory_enabled: bool
    tools_count: int
    a2a_enabled: bool

class SearchRequest(BaseModel):
    """Search request - finds and routes to suitable agent"""
    query: str
    conversation_id: str = "search-conv"
    user_id: str = "anonymous"

class SearchResponse(BaseModel):
    """Search response"""
    selected_agent: Dict[str, Any]
    agent_response: str
    timestamp: str
    processing_time: float

# ==============================================================================
# Agent Registry
# ==============================================================================

# Central registry URL - set this to your deployed registry
REGISTRY_URL = os.getenv("REGISTRY_URL", "https://nest.projectnanda.org/api/agents")

# Store known agents - fetched from central registry
KNOWN_AGENTS: Dict[str, str] = {
    # Format: "username": "http://agent-url/a2a"
    # Auto-populated from registry on startup
}

# ==============================================================================
# Agent Identity Configuration
# ==============================================================================
# 👇 EDIT THESE VALUES - This is your agent's public information

MY_AGENT_USERNAME = "personal-agent-twin"  # 👈 CHANGE THIS: Your unique username
MY_AGENT_NAME = "Personal Agent Twin"      # 👈 CHANGE THIS: Human-readable name
MY_AGENT_DESCRIPTION = "AI agent with memory and tools for research and assistance"  # 👈 CHANGE THIS
MY_AGENT_PROVIDER = "NANDA Student"        # 👈 CHANGE THIS: Your name
MY_AGENT_PROVIDER_URL = "https://nanda.mit.edu"  # 👈 CHANGE THIS: Your website

# Optional - usually don't need to change these
MY_AGENT_ID = MY_AGENT_USERNAME  # Uses username as ID
MY_AGENT_VERSION = "1.0.0"
MY_AGENT_JURISDICTION = "USA"

# Get public URL from environment (Railway sets this automatically)
PUBLIC_URL = os.getenv("PUBLIC_URL") or os.getenv("RAILWAY_PUBLIC_DOMAIN")
if PUBLIC_URL and not PUBLIC_URL.startswith("http"):
    PUBLIC_URL = f"https://{PUBLIC_URL}"

# ==============================================================================
# Tools Setup (from Day 3)
# ==============================================================================

# Tool 1: Calculator
class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Performs mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

calculator_tool = CalculatorTool()

# Tool 2: File Reading
file_tool = FileReadTool()

# Tool 3: Website Search (RAG)
web_rag_tool = WebsiteSearchTool()

# Tool 4: YouTube Search (RAG)
youtube_tool = YoutubeVideoSearchTool()

# Tool 5: Web Search (optional)
search_tool = None
if os.getenv('SERPER_API_KEY'):
    search_tool = SerperDevTool()

# Collect all tools
available_tools = [
    calculator_tool,
    file_tool,
    web_rag_tool,
    youtube_tool
]

if search_tool:
    available_tools.append(search_tool)

# ==============================================================================
# Agent Setup (from Day 3)
# ==============================================================================

# Initialize LLM
llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0.7,
)

# Create agent with memory and tools
my_agent_twin = Agent(
    role="Personal Digital Twin with Memory, Tools, and A2A Communication",
    
    goal="Answer questions, remember conversations, use tools, and communicate with other agents",
    
    backstory=f"""
    You are the digital twin of a student learning AI and CrewAI.
    Your agent ID is: {MY_AGENT_ID}
    
    Here's what you know about me:
    - I'm a student in the MIT IAP NANDA course
    - I'm learning about AI agents, memory systems, and deployment
    - I love experimenting with new AI technologies
    - My favorite programming language is Python
    - I'm building this as part of a 5-day intensive course
    
    MEMORY CAPABILITIES:
    You have four types of memory:
    1. Short-Term Memory (RAG): Recent conversation context
    2. Long-Term Memory: Important facts across sessions
    3. Entity Memory (RAG): People, places, concepts
    4. Contextual Memory: Combines all memory types
    
    TOOL CAPABILITIES:
    - FileReadTool: Read files
    - WebsiteSearchTool: Search websites (RAG)
    - YoutubeVideoSearchTool: Search video transcripts (RAG)
    - SerperDevTool: Web search (if API key configured)
    - Calculator: Math operations
    
    A2A COMMUNICATION:
    You can communicate with other agents! When you see a message mentioning
    another agent with @agent-id syntax, that means you should route the message
    to that agent. You'll receive responses from other agents and can continue
    the conversation.
    
    Use tools when you need external information. Use memory to provide
    personalized, context-aware responses. Use A2A to collaborate with other agents!
    """,
    
    tools=available_tools,
    llm=llm,
    verbose=False,
)

# ==============================================================================
# Registry Helper Functions
# ==============================================================================

async def fetch_agents_from_registry():
    """
    Fetch all registered agents from the central registry
    Updates the KNOWN_AGENTS dictionary with username -> A2A endpoint mappings
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(REGISTRY_URL)
            response.raise_for_status()
            data = response.json()
            
            # Handle both old and new API formats
            agents = data.get("agents", [])
            if not agents and isinstance(data, list):
                # New API might return list directly
                agents = data
            
            print(f"📥 Fetched {len(agents)} agents from registry")
            
            # Update KNOWN_AGENTS with username -> A2A endpoint mapping
            for agent in agents:
                # Support both old (username/url) and new (agent_id/endpoint) formats
                username = agent.get("agent_id") or agent.get("username")
                url = agent.get("endpoint") or agent.get("url", "")
                
                # Skip if no username or if it's this agent
                if not username or username == MY_AGENT_USERNAME:
                    continue
                
                # Ensure URL ends with /a2a
                if not url.endswith("/a2a"):
                    url = url.rstrip("/") + "/a2a"
                
                KNOWN_AGENTS[username] = url
                print(f"   ✅ Registered: @{username} -> {url}")
            
            return True
    except Exception as e:
        print(f"⚠️ Failed to fetch agents from registry: {str(e)}")
        return False

# ==============================================================================
# A2A Helper Functions
# ==============================================================================

async def send_message_to_agent(agent_id: str, message: str, conversation_id: str) -> str:
    """
    Send a message to another agent via A2A protocol
    
    Args:
        agent_id: The target agent's ID (e.g., "furniture-expert")
        message: The message to send
        conversation_id: Conversation tracking ID
    
    Returns:
        Response from the target agent
    """
    if agent_id not in KNOWN_AGENTS:
        return f"❌ Agent '{agent_id}' not found. Known agents: {list(KNOWN_AGENTS.keys())}"
    
    agent_url = KNOWN_AGENTS[agent_id]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                agent_url,
                json={
                    "content": {
                        "text": message,
                        "type": "text"
                    },
                    "role": "user",
                    "conversation_id": conversation_id
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", {}).get("text", str(data))
    
    except httpx.TimeoutException:
        return f"❌ Timeout connecting to agent '{agent_id}'"
    except httpx.HTTPError as e:
        return f"❌ Error communicating with agent '{agent_id}': {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def extract_agent_mentions(text: str) -> list[str]:
    """
    Extract @agent-id mentions from text
    
    Args:
        text: Input text
    
    Returns:
        List of mentioned agent IDs
    """
    # Match @agent-id pattern (alphanumeric, hyphens, underscores)
    pattern = r'@([\w-]+)'
    mentions = re.findall(pattern, text)
    return mentions

def parse_a2a_request(message: str) -> tuple[Optional[str], str]:
    """
    Parse A2A message to extract target agent and actual message
    
    Args:
        message: Input message (e.g., "@furniture-expert What sofa should I buy?")
    
    Returns:
        Tuple of (agent_id, message_without_mention)
    """
    mentions = extract_agent_mentions(message)
    
    if not mentions:
        return None, message
    
    # Take the first mention as the target agent
    target_agent = mentions[0]
    
    # Remove the @agent-id from the message
    clean_message = re.sub(r'@' + target_agent + r'\s*', '', message, count=1)
    
    return target_agent, clean_message

# ==============================================================================
# Search Helper Functions
# ==============================================================================

AGENTFACTS_DB_URL = "https://v0-agent-facts-database.vercel.app/api/agentfacts"

async def fetch_agentfacts_from_db() -> list[Dict[str, Any]]:
    """
    Fetch all agentfacts from the central database
    
    Returns:
        List of agentfacts dictionaries
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(AGENTFACTS_DB_URL)
            response.raise_for_status()
            agents = response.json()
            
            if isinstance(agents, list):
                return agents
            elif isinstance(agents, dict) and "agents" in agents:
                return agents["agents"]
            else:
                return []
    except Exception as e:
        print(f"⚠️ Failed to fetch agentfacts from database: {str(e)}")
        return []

async def select_best_agent(query: str, agentfacts: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Use LLM to select the best agent for a given query
    
    Args:
        query: User's query (e.g., "send an email")
        agentfacts: List of agentfacts from the database
    
    Returns:
        Selected agentfacts dictionary, or None if no suitable agent found
    """
    if not agentfacts:
        return None
    
    # Create a summary of available agents for the LLM
    agents_summary = []
    for agent in agentfacts:
        agent_summary = {
            "id": agent.get("id", ""),
            "label": agent.get("label", ""),
            "description": agent.get("description", ""),
            "skills": [skill.get("id", "") for skill in agent.get("skills", [])],
            "endpoints": agent.get("endpoints", {})
        }
        agents_summary.append(agent_summary)
    
    # Use LLM to select the best agent
    prompt = f"""You are an agent router. Given a user query and a list of available agents, select the single best agent to handle the query.

User Query: "{query}"

Available Agents:
{json.dumps(agents_summary, indent=2)}

Analyze the query and select the ONE agent that best matches the user's intent. Consider:
- The agent's description and label
- The agent's skills
- How well the agent's capabilities match the query

Respond with ONLY a JSON object in this exact format:
{{
    "selected_agent_id": "the id of the selected agent",
    "reasoning": "brief explanation of why this agent was selected"
}}

If no agent is suitable, respond with:
{{
    "selected_agent_id": null,
    "reasoning": "explanation of why no agent matches"
}}
"""
    
    try:
        # Use the existing LLM to get the selection
        selection_llm = LLM(model="openai/gpt-4o-mini", temperature=0.3)
        response = selection_llm.call(prompt)
        
        # Parse the response
        # Try to extract JSON from the response
        response_text = str(response).strip()
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        selection = json.loads(response_text)
        selected_id = selection.get("selected_agent_id")
        
        if not selected_id:
            return None
        
        # Find the full agentfacts for the selected agent
        for agent in agentfacts:
            if agent.get("id") == selected_id:
                return agent
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error selecting agent with LLM: {str(e)}")
        # Fallback: simple keyword matching
        query_lower = query.lower()
        for agent in agentfacts:
            description = agent.get("description", "").lower()
            label = agent.get("label", "").lower()
            if query_lower in description or query_lower in label:
                return agent
        return None

async def send_query_to_url(agent_url: str, question: str, user_id: str = "anonymous") -> str:
    """
    Send a direct query to an agent URL
    
    Args:
        agent_url: Full URL to the agent's query endpoint
        question: Question to ask
        user_id: User identifier
    
    Returns:
        Response from the agent
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                agent_url,
                json={
                    "question": question,
                    "user_id": user_id
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("answer", str(data))
    
    except httpx.TimeoutException:
        return f"Timeout connecting to agent at {agent_url}"
    except httpx.HTTPError as e:
        return f"Error communicating with agent: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

async def send_a2a_to_url(agent_url: str, message: str, conversation_id: str) -> str:
    """
    Send an A2A message directly to an agent URL
    
    Args:
        agent_url: Full URL to the agent's A2A endpoint
        message: Message to send
        conversation_id: Conversation tracking ID
    
    Returns:
        Response from the agent
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                agent_url,
                json={
                    "content": {
                        "text": message,
                        "type": "text"
                    },
                    "role": "user",
                    "conversation_id": conversation_id
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", {}).get("text", str(data))
    
    except httpx.TimeoutException:
        return f"Timeout connecting to agent at {agent_url}"
    except httpx.HTTPError as e:
        return f"Error communicating with agent: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def generate_agent_facts() -> Dict[str, Any]:
    """
    Generate AgentFacts JSON (NANDA schema)
    
    AgentFacts is an enhanced version of Google's AgentCard with additional
    fields for trust, verification, and operational metrics.
    
    Returns:
        Dict containing the AgentFacts schema
    """
    import uuid
    from datetime import datetime, timedelta
    
    # Generate unique ID if not set
    agent_uuid = os.getenv("AGENT_UUID", str(uuid.uuid4()))
    
    # Determine endpoint URL
    base_url = PUBLIC_URL or "http://localhost:8000"
    
    agent_facts = {
        # ========== IDENTITY & BASIC INFORMATION ==========
        # 🔵 Unique machine-readable identifier
        "id": f"nanda:{agent_uuid}",
        
        # 🔵 Agent URN identifier (can be DID or traditional)
        "agent_name": f"urn:agent:nanda:{MY_AGENT_USERNAME}",
        
        # 🟢 Human readable name (maps to AgentCard.name)
        "label": MY_AGENT_NAME,
        
        # 🟢 Agent description (maps to AgentCard.description)
        "description": MY_AGENT_DESCRIPTION,
        
        # 🟢 Version information (maps to AgentCard.version)
        "version": MY_AGENT_VERSION,
        
        # 🟢 Documentation URL
        "documentationUrl": f"{base_url}/docs",
        
        # 🔵 Jurisdiction: which country/entity covers the compliance for this agent
        "jurisdiction": MY_AGENT_JURISDICTION,
        
        # ========== PROVIDER INFORMATION ==========
        # 🟢 Provider details (maps to AgentCard.provider)
        "provider": {
            "name": MY_AGENT_PROVIDER,
            "url": MY_AGENT_PROVIDER_URL,
            # 🔵 Optional Decentralized identifier for provider verification
            "did": f"did:web:{MY_AGENT_PROVIDER_URL.replace('https://', '').replace('http://', '')}"
        },
        
        # ========== NETWORK ENDPOINTS ==========
        "endpoints": {
            # 🟢 Static endpoints (maps to AgentCard.url)
            "static": [
                f"{base_url}/a2a"
            ],
            # 🔵 Dynamic routing capabilities (optional - for advanced deployments)
            "adaptive_resolver": {
                "url": f"{base_url}/a2a",
                "policies": [
                    "load"  # Load balancing (if deployed across multiple instances)
                ]
            }
        },
        
        # ========== TECHNICAL CAPABILITIES ==========
        "capabilities": {
            # 🟢 Input/output modalities (maps to AgentCard.defaultInputModes/defaultOutputModes)
            "modalities": [
                "text"
            ],
            # 🔵 Real-time streaming support
            "streaming": False,
            # 🔵 Batch processing support
            "batch": False,
            # 🟢 Authentication methods (maps to AgentCard.securitySchemes & security)
            "authentication": {
                "methods": [
                    "none"  # No authentication required (add "bearer" or "oauth2" for production)
                ],
                "requiredScopes": []
            }
        },
        
        # ========== FUNCTIONAL SKILLS ==========
        # 🟢 Skills array (maps to AgentCard.skills)
        "skills": [
            {
                # 🟢 Skill identifier
                "id": "question_answering",
                # 🟢 Skill description
                "description": "Answer questions using memory and context",
                # 🟢 Input modes for this skill
                "inputModes": ["text"],
                # 🟢 Output modes for this skill
                "outputModes": ["text"],
                # 🔵 Language support specification
                "supportedLanguages": ["en"],
                # 🔵 Performance constraint
                "latencyBudgetMs": 5000
            },
            {
                "id": "calculation",
                "description": "Perform mathematical calculations",
                "inputModes": ["text"],
                "outputModes": ["text"],
                "supportedLanguages": ["en"],
                "latencyBudgetMs": 1000
            },
            {
                "id": "web_search",
                "description": "Search the web and websites for information",
                "inputModes": ["text"],
                "outputModes": ["text"],
                "supportedLanguages": ["en"],
                "latencyBudgetMs": 10000
            },
            {
                "id": "file_reading",
                "description": "Read and analyze file contents",
                "inputModes": ["text"],
                "outputModes": ["text"],
                "supportedLanguages": ["en"],
                "latencyBudgetMs": 3000
            }
        ],
        
        # ========== QUALITY METRICS ==========
        # 🔵 Certified performance and reliability metrics
        "evaluations": {
            "performanceScore": 4.5,
            "availability90d": "99.0%",  # Estimated for student project
            "lastAudited": datetime.now().isoformat(),
            "auditTrail": None,  # Optional: Add IPFS hash for immutable audit
            "auditorID": "Self-Reported v1.0"
        },
        
        # ========== OBSERVABILITY & MONITORING ==========
        # 🔵 Telemetry and monitoring configuration
        "telemetry": {
            "enabled": True,
            "retention": "7d",
            "sampling": 1.0,  # 100% sampling for development
            "metrics": {
                "latency_p95_ms": 2000,
                "throughput_rps": 10,
                "error_rate": 0.01,
                "availability": "99.0%"
            }
        },
        
        # ========== TRUST & VERIFICATION ==========
        # 🔵 Certification and trust framework
        "certification": {
            "level": "development",  # Options: "development", "verified", "certified"
            "issuer": MY_AGENT_PROVIDER,
            "issuanceDate": datetime.now().isoformat(),
            "expirationDate": (datetime.now() + timedelta(days=365)).isoformat()
        }
    }
    
    return agent_facts

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    """Root endpoint - shows API information"""
    return {
        "message": "🤖 Personal Agent Twin API with A2A - Day 4",
        "version": "2.0.0",
        "agent_id": MY_AGENT_ID,
        "agent_name": MY_AGENT_NAME,
        "agent_username": MY_AGENT_USERNAME,
        "memory_enabled": True,
        "tools_enabled": len(available_tools),
        "a2a_enabled": True,
        "known_agents": list(KNOWN_AGENTS.keys()),
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "a2a": "POST /a2a",
            "search": "POST /search (Auto-find and route to suitable agent)",
            "agentfacts": "GET /agentfacts",
            "agents": "GET /agents",
            "docs": "GET /docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        memory_enabled=True,
        tools_count=len(available_tools),
        a2a_enabled=True
    )

@app.get("/agents")
async def list_agents():
    """List known agents for A2A communication"""
    return {
        "my_agent_id": MY_AGENT_ID,
        "my_agent_name": MY_AGENT_NAME,
        "my_agent_username": MY_AGENT_USERNAME,
        "known_agents": KNOWN_AGENTS,
        "usage": "Send messages using @agent-id syntax in the /a2a endpoint"
    }

@app.get("/agentfacts")
async def get_agent_facts():
    """
    Get AgentFacts (NANDA Schema)
    
    AgentFacts is an enhanced version of Google's AgentCard with additional
    fields for trust, verification, and operational metrics.
    
    This endpoint allows other agents to discover this agent's:
    - Identity and capabilities
    - Skills and modalities
    - Performance metrics
    - Trust certifications
    - Network endpoints
    
    Example usage:
        curl https://your-agent.railway.app/agentfacts
    
    Note: In NANDA, this is similar to /.well-known/agent-card.json
    but with extended metadata for agent mesh networks.
    """
    return generate_agent_facts()

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the agent (original endpoint from Day 3)
    
    This is the standard query endpoint - no A2A routing.
    For A2A communication, use the /a2a endpoint instead.
    """
    start_time = datetime.now()
    
    try:
        # Create task for this query
        task = Task(
            description=f"""
            Answer the following question: {request.question}
            
            Use your memory to recall relevant context.
            Use your tools when you need external information or calculations.
            Provide accurate, helpful responses.
            """,
            expected_output="A clear, context-aware answer using memory and tools as needed",
            agent=my_agent_twin,
        )
        
        # Create crew with memory enabled
        crew = Crew(
            agents=[my_agent_twin],
            tasks=[task],
            memory=True,
            verbose=False,
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return QueryResponse(
            answer=str(result.raw),
            timestamp=end_time.isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/a2a", response_model=A2AResponse)
async def a2a_endpoint(message: A2AMessage):
    """
    A2A (Agent-to-Agent) Communication Endpoint
    
    ⚠️ IMPORTANT: This endpoint is ONLY for agent-to-agent routing!
    
    You MUST include @agent-id in your message to route to another agent.
    
    For direct queries to this agent, use POST /query instead.
    
    Usage:
        Send a message with @agent-id to route to another agent
    
    Example:
        {"content": {"text": "@furniture-expert What sofa should I buy?", "type": "text"}, "role": "user", "conversation_id": "123"}
    
    This will:
        1. Extract target agent: "furniture-expert"
        2. Look up agent URL from registry
        3. Forward message to that agent
        4. Return their response
    """
    
    try:
        text_content = message.content.get("text", "")
        conversation_id = message.conversation_id
        
        # Log incoming A2A message
        a2a_logger.info(f"INCOMING | conversation_id={conversation_id} | message={text_content}")
        
        # Check if this message is routing to another agent
        target_agent, clean_message = parse_a2a_request(text_content)
        
        if not target_agent:
            # NO @agent-id found - this is an ERROR!
            error_msg = (
                "❌ ERROR: /a2a endpoint requires @agent-id for routing.\n\n"
                f"Your message: '{text_content}'\n\n"
                "This endpoint is ONLY for agent-to-agent communication.\n"
                "You must include @agent-id to route to another agent.\n\n"
                "Examples:\n"
                "  - '@john-agent Can you help with this?'\n"
                "  - '@research-bot What's the latest on AI?'\n\n"
                "For direct queries to THIS agent, use POST /query instead."
            )
            
            a2a_logger.error(f"NO_TARGET | conversation_id={conversation_id} | message={text_content}")
            
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        # Route to target agent
        print(f"🔀 Routing message to agent: {target_agent}")
        a2a_logger.info(f"ROUTING | conversation_id={conversation_id} | target={target_agent} | message={clean_message}")
        
        agent_response = await send_message_to_agent(target_agent, clean_message, conversation_id)
        
        response_text = f"[Forwarded to @{target_agent}]\n\n{agent_response}"
        
        # Log successful routing
        a2a_logger.info(f"SUCCESS | conversation_id={conversation_id} | target={target_agent} | response_length={len(agent_response)}")
        
        end_time = datetime.now()
        
        return A2AResponse(
            content={
                "text": response_text,
                "type": "text"
            },
            role="assistant",
            conversation_id=conversation_id,
            timestamp=end_time.isoformat(),
            agent_id=MY_AGENT_ID
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like our 400 error above)
        raise
    except Exception as e:
        a2a_logger.error(f"ERROR | conversation_id={message.conversation_id} | error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing A2A message: {str(e)}"
        )

@app.post("/agents/register")
async def register_agent(agent_id: str, agent_url: str):
    """
    Register another agent for A2A communication
    
    This allows you to dynamically add agents to your known agents list.
    
    Args:
        agent_id: The agent's unique ID
        agent_url: The agent's A2A endpoint URL (e.g., http://agent.com/a2a)
    """
    KNOWN_AGENTS[agent_id] = agent_url
    return {
        "message": f"✅ Agent '{agent_id}' registered successfully",
        "agent_id": agent_id,
        "agent_url": agent_url,
        "total_known_agents": len(KNOWN_AGENTS)
    }

@app.post("/search", response_model=SearchResponse)
async def search_and_route(request: SearchRequest):
    """
    Search endpoint - automatically finds and routes to suitable agent
    
    This endpoint:
    1. Fetches all available agents from the agentfacts database
    2. Uses an LLM to select the best agent for the query
    3. Sends an A2A message to the selected agent
    4. Returns the agent's response
    
    Example:
        {"query": "send an email", "conversation_id": "conv-123"}
    
    The LLM will analyze the query and select the most suitable agent
    from the database, then route the message to that agent.
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Fetch all agentfacts from database
        print(f"🔍 Fetching agentfacts from database...")
        agentfacts = await fetch_agentfacts_from_db()
        
        if not agentfacts:
            raise HTTPException(
                status_code=503,
                detail="No agents available in the database"
            )
        
        print(f"📥 Found {len(agentfacts)} agents in database")
        
        # Step 2: Use LLM to select the best agent
        print(f"🤖 Selecting best agent for query: '{request.query}'")
        selected_agent = await select_best_agent(request.query, agentfacts)
        
        if not selected_agent:
            raise HTTPException(
                status_code=404,
                detail="No suitable agent found for this query"
            )
        
        print(f"✅ Selected agent: {selected_agent.get('label', 'Unknown')}")
        
        # Step 3: Extract the agent's endpoint URL
        endpoints = selected_agent.get("endpoints", {})
        agent_url = None
        
        # Try static endpoints first
        static_endpoints = endpoints.get("static", [])
        if static_endpoints:
            agent_url = static_endpoints[0]
        # Try adaptive resolver
        elif "adaptive_resolver" in endpoints:
            agent_url = endpoints["adaptive_resolver"].get("url")
        
        if not agent_url:
            raise HTTPException(
                status_code=500,
                detail=f"Selected agent '{selected_agent.get('label')}' has no valid endpoint"
            )

        # Ensure URL points to /query endpoint
        if agent_url.endswith("/a2a"):
            agent_url = agent_url.replace("/a2a", "/query")
        elif not agent_url.endswith("/query"):
            agent_url = agent_url.rstrip("/") + "/query"
        
        print(f"🔀 Routing to: {agent_url}")
        
        # Step 4: Send Query message to the selected agent
        # We use the direct query endpoint because we want to ask the agent a question,
        # not route a message through it to someone else.
        agent_response = await send_query_to_url(agent_url, request.query, request.user_id)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return SearchResponse(
            selected_agent={
                "id": selected_agent.get("id"),
                "label": selected_agent.get("label"),
                "description": selected_agent.get("description"),
                "endpoint": agent_url
            },
            agent_response=agent_response,
            timestamp=end_time.isoformat(),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search: {str(e)}"
        )

# ==============================================================================
# Startup Event
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Run when the API starts"""
    print("\n" + "="*70)
    print("🚀 Personal Agent Twin API with A2A Starting...")
    print("="*70)
    print(f"\n✅ Agent ID: {MY_AGENT_ID}")
    print(f"✅ Agent Name: {MY_AGENT_NAME}")
    print(f"✅ Agent Username: {MY_AGENT_USERNAME}")
    print(f"✅ Model: {llm.model}")
    print("✅ Memory: Enabled (4 types)")
    print(f"✅ Tools: {len(available_tools)} tools loaded")
    print("✅ A2A: Enabled (NANDA-style)")
    
    # Fetch agents from central registry
    print(f"\n🔍 Fetching agents from registry: {REGISTRY_URL}")
    await fetch_agents_from_registry()
    print(f"✅ Known Agents: {len(KNOWN_AGENTS)}")
    
    print("\n📚 Documentation: http://localhost:8000/docs")
    print("🤖 A2A Endpoint: http://localhost:8000/a2a")
    print("📋 AgentFacts: http://localhost:8000/agentfacts")
    if PUBLIC_URL:
        print(f"🌐 Public URL: {PUBLIC_URL}")
    print("="*70 + "\n")

# ==============================================================================
# Run Instructions
# ==============================================================================
"""
LOCAL TESTING:
    uvicorn main:app --reload
    
    Then test:
    # Standard query
    curl -X POST http://localhost:8000/query \\
      -H "Content-Type: application/json" \\
      -d '{"question": "What is 50 * 50?"}'
    
    # A2A message (local)
    curl -X POST http://localhost:8000/a2a \\
      -H "Content-Type: application/json" \\
      -d '{"content":{"text":"Hello! What can you help me with?","type":"text"},"role":"user","conversation_id":"test123"}'
    
    # A2A message (route to another agent)
    curl -X POST http://localhost:8000/a2a \\
      -H "Content-Type: application/json" \\
      -d '{"content":{"text":"@furniture-expert What sofa should I buy?","type":"text"},"role":"user","conversation_id":"test123"}'
    
    # Register another agent
    curl -X POST "http://localhost:8000/agents/register?agent_id=test-agent&agent_url=http://example.com/a2a"

RAILWAY DEPLOYMENT:
    Railway automatically detects and runs this with:
    uvicorn main:app --host 0.0.0.0 --port $PORT
    
    Set environment variables:
    - OPENAI_API_KEY (required)
    - AGENT_ID (optional, default: "personal-agent-twin")
    - AGENT_NAME (optional, default: "Personal Agent Twin")
    - SERPER_API_KEY (optional, for web search)
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

