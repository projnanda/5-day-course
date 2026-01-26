# MIT IAP NANDA - AI Agent Development Course

**5-Day Intensive Course on Building and Deploying AI Agents**

Welcome to the MIT IAP NANDA course! Over 5 days, you'll progress from building a simple AI agent to deploying a sophisticated agent capable of competing in an agent battle.

## Course Structure

### [Day 1: Agent Loop + AI Twin v0](./day-1/)
**What You'll Build:** A simple agent loop (max 5 turns) and your first AI twin using CrewAI (no tools/memory yet)

**Key Concepts:**
- Understanding agent loops
- CrewAI fundamentals
- Agent, Task, and Crew concepts
- GitHub repository setup
- Creating AgentFacts for your agent

**Prerequisites:** Python 3.10+, OpenAI API key, GitHub account

---

### [Day 2: Memory + MCP Tools](./day-2/)
**What You'll Build:** Enhance your agent with memory capabilities and MCP (Model Context Protocol) tools

**Key Concepts:**
- Short-term and long-term memory
- MCP tool integration
- Using external APIs (e.g., Spotify, web search)
- Tool selection and usage patterns

**Prerequisites:** API keys for chosen MCP server (e.g., Spotify, Weather, etc.)

---

### [Day 3: Deploy on Railway + REST API](./day-3/)
**What You'll Build:** Deploy your agent on Railway and expose it via REST API

**Key Concepts:**
- FastAPI integration
- Cloud deployment on Railway
- REST API design for agents
- Testing in NANDA testbed
- Environment variable management in production

**Prerequisites:** Railway account

**Bonus: [Chat Frontend](./frontend/)** - A simple Next.js chat interface where students can share and chat with each other's agents!

---

### [Day 4: A2A Communication Protocol](./day-4/)
**What You'll Build:** Agent-to-Agent (A2A) communication system

**Key Concepts:**
- Agent-to-agent (A2A) communication protocol
- Message routing and agent discovery
- AgentFacts for capability sharing
- Cross-agent collaboration
- Central registry integration

**Prerequisites:** Working agents (local or deployed)

---

### [Day 5: Coordination Protocol + Agent Battle](./day-5/)
**What You'll Build:** Advanced coordination and compete in the final agent battle

**Key Concepts:**
- Agent coordination protocols
- Multi-agent task decomposition
- Agent optimization strategies
- Using advanced tools (web search, stock data, specialized APIs)
- Memory optimization
- Response speed vs. accuracy tradeoffs

**The Challenge:** Build the most capable agent possible! Agents will be tested and evaluated using the Agent Smart Score system that measures:
- Accuracy and correctness
- Response speed and efficiency
- Reasoning quality
- Robustness across diverse topics
- Collaboration capabilities

**Prerequisites:** Deployed endpoint + consistent input/output format

---

## Getting Started

1. **Clone this repository:**
   ```bash
   git clone https://github.com/projnanda/5-day-course.git
   cd 5-day-course
   ```

2. **Start with Day 1:**
   ```bash
   cd day-1
   ```
   Follow the README in each day's folder!

3. **Progress through each day** at your own pace, or follow along with the course schedule.

4. **Optional: Use the Full Chat System** (after Day 3):
   
   See interface folder for complete instructions!
   
   **Quick version:**
   ```bash
   cd interface
   npm install && npm run dev
   ```
   
   Now students can share and chat with each other's agents!

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Model Context Protocol Servers](https://github.com/modelcontextprotocol/servers)
- [Railway Deployment Guide](https://docs.railway.app/)

## Course Goals

By the end of this course, you will:
- Understand AI agent architectures
- Build agents with memory and tools
- Deploy production-ready agent APIs
- Implement multi-agent coordination and communication protocols
- Compete in the Agent Battle with Agent Smart Score evaluation
- Have a portfolio-worthy project on GitHub

## Tips for Success

1. **Start early each day** - Building agents takes time!
2. **Experiment freely** - Try different approaches
3. **Ask questions** - Your instructors and classmates are here to help
4. **Test thoroughly** - Agents can behave unexpectedly
5. **Have fun** - AI agents are exciting technology!

## License

MIT License - Feel free to use this for learning!

---

**Built for MIT IAP 2026**

*Powered by [CrewAI](https://crewai.com) and [NANDA](https://nanda.ai)*
