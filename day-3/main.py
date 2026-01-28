"""
Day 3: Deploy Your Agent with Memory to Railway
================================================

This wraps your Day 2 agent (with memory and tools) in a FastAPI REST API
so anyone can interact with it via HTTP from anywhere in the world!

What's FastAPI?
- A Python web framework that creates REST APIs
- Turns your local Python code into a web service
- Allows HTTP requests (like from curl, browsers, or other apps)

Architecture:
- Service 1: ChromaDB (persistent memory storage)
- Service 2: This FastAPI app (your agent)
- They communicate via Railway's private network
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import os

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool, WebsiteSearchTool, YoutubeVideoSearchTool
from pydantic import Field
from typing import Type

# Load environment variables
load_dotenv()

# ==============================================================================
# FastAPI Application Setup
# ==============================================================================

app = FastAPI(
    title="Personal Agent Twin API",
    description="Your Day 2 agent with memory and tools, now accessible via REST API!",
    version="1.0.0"
)

# Enable CORS (allows browser requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Request/Response Models (API Input/Output)
# ==============================================================================

class QueryRequest(BaseModel):
    """What the API expects when you send a question"""
    question: str
    user_id: str = "anonymous"

class QueryResponse(BaseModel):
    """What the API returns after processing"""
    answer: str
    timestamp: str
    processing_time: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    memory_enabled: bool
    tools_count: int

# ==============================================================================
# Tools Setup (from Day 2)
# ==============================================================================

# Tool 1: Calculator (custom tool from Day 2)
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

# Tool 5: Web Search (optional - requires SERPER_API_KEY)
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
# Agent Setup (from Day 2, with memory!)
# ==============================================================================

# Initialize LLM
llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0.7,
)

# Create agent with memory and tools
my_agent_twin = Agent(
    role="Personal Digital Twin with Memory and Tools",
    
    goal="Answer questions about me, remember conversations, and use tools when needed",
    
    backstory="""
    You are the digital twin of a student learning AI and CrewAI.
    
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
    
    Use tools when you need external information. Use memory to provide
    personalized, context-aware responses.
    """,
    
    tools=available_tools,
    llm=llm,
    verbose=False,  # Set to True for debugging
)

# ==============================================================================
# Crew Setup (Create once, reuse for all requests)
# ==============================================================================

# Create a generic task that will be reused
answer_task = Task(
    description="Answer the user's question: {question}. Use memory to recall context and tools when needed.",
    expected_output="A clear, context-aware answer using memory and tools as needed",
    agent=my_agent_twin,
)

# Create crew with memory enabled - this persists across requests!
my_crew = Crew(
    agents=[my_agent_twin],
    tasks=[answer_task],
    memory=True,  # This enables all 4 memory types!
    verbose=False,
)

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    """Root endpoint - shows API information"""
    return {
        "message": "🤖 Personal Agent Twin API - Day 3",
        "version": "1.0.0",
        "memory_enabled": True,
        "tools_enabled": len(available_tools),
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "docs": "GET /docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns the current status of the API and agent.
    """
    return HealthResponse(
        status="healthy",
        memory_enabled=True,
        tools_count=len(available_tools)
    )

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the agent with memory and tools
    
    This is the main endpoint! Send a question and get an answer.
    The agent will:
    - Remember previous conversations (if memory is working)
    - Use tools when needed (calculator, web search, etc.)
    - Provide personalized responses
    
    Example:
        curl -X POST https://your-app.up.railway.app/query \\
          -H "Content-Type: application/json" \\
          -d '{"question": "What is 123 * 456?"}'
    """
    start_time = datetime.now()
    
    try:
        # Execute with the persistent crew (reuses memory across requests!)
        # Pass the question directly - the agent will handle it
        result = my_crew.kickoff(inputs={
            "question": request.question,
            "description": f"Answer the following question: {request.question}. Use your memory to recall relevant context and your tools when needed."
        })
        
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

# ==============================================================================
# Startup Event
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Run when the API starts"""
    print("\n" + "="*70)
    print("🚀 Personal Agent Twin API Starting...")
    print("="*70)
    print(f"\n✅ Model: {llm.model}")
    print(f"✅ Memory: Enabled (4 types)")
    print(f"✅ Tools: {len(available_tools)} tools loaded")
    print("✅ Agent: Initialized")
    print("\n📚 Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")

# ==============================================================================
# Run Instructions
# ==============================================================================
"""
LOCAL TESTING:
    uvicorn main:app --reload
    
    Then test:
    curl -X POST http://localhost:8000/query \\
      -H "Content-Type: application/json" \\
      -d '{"question": "What is 50 * 50?"}'

RAILWAY DEPLOYMENT:
    Railway automatically detects and runs this with:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
