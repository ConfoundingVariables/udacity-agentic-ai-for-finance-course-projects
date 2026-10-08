# Agentic EDGAR-RAG with Multi-Tool Coordination - Starter Code

## 🎯 Project Overview

In this project, you'll build a sophisticated **6-tool agentic system** that intelligently coordinates between document analysis, database queries, and real-time market data to provide comprehensive financial insights.

## 📊 Grading Rubric

Please review the **[GRADING_RUBRIC.md](GRADING_RUBRIC.md)** file for detailed evaluation criteria and requirements. The rubric covers:
- Document Tools Implementation
- Function Tools Implementation  
- Agent Coordinator Implementation
- System Integration & Testing
- Code Quality & Documentation

Understanding the rubric before starting will help you meet all project requirements and excel in your submission.

## 🏗️ What You'll Build

**Final System Architecture:**
- **3 Document Tools**: Individual SEC 10-K filing analysis for Apple, Google, and Tesla
- **3 Function Tools**: Database SQL queries, real-time market data, and PII protection
- **Smart Routing**: LLM-based intelligent tool selection with automatic coordination
- **Multi-Source Synthesis**: Combining information from multiple data sources

## 📋 Getting Started

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Initialize Database

```bash
# Run the provided database setup script
python build_database.py
```

This creates a SQLite database with sample customer portfolio data that you'll integrate with your agent.

### 3. Run the End-to-End Tests

```bash
# Live checks: 10-K routing, PII masking, market data, multi-tool routing
python -m pytest tests/test_e2e.py
```

**Important**: In Vocareum, LlamaIndex needs the explicit `api_base` parameter. Each helper module's `_configure_settings()` passes `OPENAI_API_BASE` to both `OpenAI` and `OpenAIEmbedding`.

### 4. Project Structure

```
starter_code/
├── README.md                          # This file
├── GRADING_RUBRIC.md                  # 📊 Detailed evaluation criteria
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── build_database.py                  # Database setup (provided)
├── financial_agent_walkthrough.ipynb  # 🚧 Complete testing walkthrough
├── helper_modules/                    # 🚧 Script directory - YOUR CODE HERE
│   ├── document_tools.py              # Document processing scripts
│   ├── function_tools.py              # SQL, market data, PII scripts
│   └── agent_coordinator.py           # Multi-tool coordination scripts
├── tests/
│   └── test_e2e.py                    # Live end-to-end checks
├── notebooks/                         # 🚧 Testing and examples
│   └── test_agent.ipynb               # Individual component testing
└── data/
    ├── financial.db                   # SQLite database (created by build_database.py)
    └── 10k_documents/                 # SEC filing PDFs (provided)
        ├── AAPL_10K_2024.pdf
        ├── GOOGL_10K_2024.pdf
        └── TSLA_10K_2024.pdf
```

## 🛠️ Implementation Guide

The three core scripts in your **script directory** (`helper_modules/`) form the foundation of your agentic system. Each script contains strategic `# YOUR CODE HERE` placeholders where you'll implement key functionality.

### Step 1: Document Tools (helper_modules/document_tools.py)
**Your Task**: Complete the DocumentToolsManager class for individual RAG systems

**Implementation Areas**:
- Complete `_configure_settings()`: Set up LlamaIndex configurations for optimal performance
- Complete `build_document_tools()`: Create individual QueryEngineTools for each company's 10-K filing

**Key Learning Goals**:
- PDF document processing with LlamaIndex
- Vector embedding and indexing strategies  
- Query engine creation for document retrieval
- Tool naming and metadata configuration

**What's Provided**:
- Class structure and method signatures
- Document loading utilities and chunking guidance
- Index creation templates with helpful comments
- Specific hints about Settings configuration

**Test Your Work**: Run the document tools test to validate your implementation

### Step 2: Function Tools (helper_modules/function_tools.py)  
**Your Task**: Complete the FunctionToolsManager class for database and market data functionality

**Implementation Areas**:
- Complete `create_function_tools()`: Build three core tools:
  - Database query tool: Natural language to SQL converter
  - Market search tool: Real-time market data fetcher
  - PII protection tool: Privacy protection system

**Key Learning Goals**:
- Natural language to SQL conversion using LLM
- Real-time API integration (Yahoo Finance)
- Privacy protection and data masking techniques
- FunctionTool creation and parameter validation

**What's Provided**:
- Database schema reference and example queries
- API integration templates with error handling patterns
- PII detection patterns and masking examples (abc@gmail.com → ***@gmail.com)
- Function structure with TODO markers for implementation

**Test Your Work**: Run the function tools test to validate each tool independently

### Step 3: Agent Coordination (helper_modules/agent_coordinator.py)
**Your Task**: Complete the AgentCoordinator class for intelligent multi-tool orchestration

**Implementation Areas**:
- Complete `_configure_settings()`: Set up LLM and embedding models with Vocareum compatibility
- Complete `_create_tools()`: Initialize document and function tools using helper modules
- Complete `_route_query()`: Implement intelligent LLM-based tool routing
- Complete `query()`: Main query processing with multi-tool coordination and result synthesis

**Key Learning Goals**:
- LLM-based intelligent tool routing and selection
- Multi-tool coordination and result synthesis
- Automatic PII detection and protection workflow
- Response formatting from multiple data sources

**What's Provided**:
- Complete class structure with helper methods
- Tool management and status tracking
- Error handling framework
- PII field detection patterns

**Test Your Work**: Run the agent coordinator test to validate routing and synthesis

### Step 4: Validation & Testing (financial_agent_walkthrough.ipynb)
**Your Task**: Once you complete the script directory, run this comprehensive walkthrough

**Testing Scenarios**:
- Individual tool functionality verification
- Multi-tool coordination examples
- PII protection validation  
- Complex financial analysis workflows

**What It Provides**:
- Complete testing framework for your implementations
- Example queries demonstrating system capabilities
- Performance validation and debugging guidance

## 📚 Learning Checkpoints

**Checkpoint 1**: Script directory foundation (helper_modules/)
- Complete `document_tools.py`: Each document tool can answer company-specific questions
- Complete `function_tools.py`: SQL tool generates valid queries, market data tool fetches prices, PII tool masks information  
- Complete `agent_coordinator.py`: Router correctly selects tools and synthesizes multi-source results

**Checkpoint 2**: Complete integration and testing
- Run the walkthrough notebook to validate your script implementations
- Verify tool coordination works across multiple data sources
- Confirm PII protection operates correctly during queries
- Demonstrate sophisticated multi-tool query capabilities

## 🎓 Success Criteria

By completion, your agent should handle queries like:
- "What are Apple's main business risks according to their 10-K filing?"
- "Show me customers who own Tesla stock and current TSLA price"
- "Compare my Google customers' holdings with GOOGL's revenue segments"
- "Analyze Tesla: customer holdings, stock price, and supply chain risks"

## 📖 Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)

## 🚀 Ready to Build?

Start with the **script directory** (`helper_modules/`) and work through each component systematically:

1. **Begin with `document_tools.py`** - Foundation RAG systems for document analysis
2. **Continue to `function_tools.py`** - Database and market data integration  
3. **Complete `agent_coordinator.py`** - Multi-tool orchestration and routing
4. **Test with `financial_agent_walkthrough.ipynb`** - Comprehensive validation of your script implementations

The provided database, documents, and test frameworks give you a solid foundation to build upon!

**Remember**: Focus on understanding how each script component contributes to multi-tool coordination and agentic AI principles. The `# YOUR CODE HERE` sections guide you through implementing sophisticated AI agent patterns.
