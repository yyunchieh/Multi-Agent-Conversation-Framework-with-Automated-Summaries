# Multi-Agent Conversation System on Education Topic

A sophisticated LangGraph application where two AI agents discuss a topic being assigned, with a third agent providing periodic summaries to enhance conversation structure and coherence.

## Features

- **Three-Agent Architecture**: Two conversational agents (GPT-4o and Perplexity Sonar) plus a summarizer agent(GPT-4o-mini)
- **Periodic Summaries**: Automatic summarization every N turns (default: 4) to maintain conversation focus
- **Configurable Parameters**: Adjust number of turns and summary interval
- **Rich Output**: Markdown files with formatted conversations and comprehensive summaries
- **Educational Focus**: Optimized for discussing educational inequalities and AI applications
- **Deployment-Ready**: Built with LangGraph for production use

## Architecture

```
┌─────────────┐
│   Agent 1   │  (GPT-4o)
│  (GPT-4o)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     Every N turns    ┌──────────────┐
│   Agent 2   │ ───────────────────► │  Summarizer  │
│ (Perplexity)│                      │ (GPT-4o-mini)│
└──────┬──────┘                      └──────┬───────┘
       │                                    │
       │◄───────────────────────────────────┘
                Resume Conversation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys in a `.env` file:
```bash
OPENAI_API_KEY=your_openai_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
```

## Usage

### Quick Start

Run the conversation system with default settings:

```bash
python run_agent.py
```

This will:
- Start a conversation about educational inequalities
- Run for 8 turns
- Generate a summary every 4 turns
- Save the conversation to `conversation_results/` as markdown

## Configuration

### Key Parameters

- **`topic`**: The discussion topic for the agents
- **`max_turns`**: Total number of conversation turns (excluding summaries)
- **`summary_interval`**: How often to generate summaries (e.g., 4 = every 4 turns)

### Customizing Agents

Edit `agent.py` to:
- Change model selections 
- Adjust system prompts for different personalities
- Modify summarizer behavior

## Project Structure

```
langgraph_interaction_agent/
├── agent.py              # Main LangGraph application with 3-agent system
├── summarizer.py         # Standalone summarization module
├── run_agent.py          # Test script with markdown export
├── langgraph.json        # LangGraph configuration
├── requirements.txt      # Python dependencies
├── conversation_results/ # Output directory for markdown files
└── README.md            
```

## How It Works

1. **Agent 1 (GPT-4o)** starts the conversation
2. **Agent 2 (Perplexity)** responds
3. After every N turns, the **Summarizer (GPT-4o-mini)** provides an objective summary
4. Conversation resumes with Agent 1
5. Process repeats until `max_turns` is reached
6. Final summary generated at the end

### Summary Features

Each periodic summary includes:
- Main points discussed in recent turns
- Areas of agreement/disagreement
- New insights or perspectives
- Objective third-person perspective

Final summary includes:
- Executive summary
- Topics discussed
- Key points 
- Main arguments from each agent
- Conclusions and takeaways

## Output Format

Generated markdown files include:

### Metadata Section
- Topic, date, turn count
- Agent models used

### Summary Section 
- Executive summary
- Topics discussed
- Key points
- Main arguments
- Conclusions

### Conversation Section
- Turn-by-turn dialogue
- Periodic summaries clearly marked

## Example Output

```markdown
### Turn 1: Agent 1 (GPT-4o)
AI can personalize learning experiences...

### Turn 2: Agent 2 (Perplexity (Llama 3.1 Sonar))
Additionally, AI can overcome accessibility challenges...

### Periodic Summary by Summarizer (GPT-4o-mini)
> The conversation discusses how AI can address educational
> inequalities through personalization and accessibility...
```

## Research Applications

This multi-agent framework offers a reproducible testbed for studying collective intelligence and argumentative dynamics in LLMs. With two heterogeneous agents engaging in structured deliberation and a third agent synthesizing the output, the system supports research in:

1. LLM comparative analysis — understanding reasoning divergences and convergence patterns

2. Human–AI policy research — generating well-reasoned multi-perspective outputs for domains such as education

3. Computational social science — analyzing how machine-generated debates resemble human deliberative processes