"""
Two-agent conversation system using LangGraph.
Two agents discuss a user-assigned topic back and forth.
"""

import re
import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
perplexity_key = os.getenv("PERPLEXITY_API_KEY")

# Define state for the conversation
class ConversationState(TypedDict):
    """State for tracking the two-agent conversation."""
    messages: Annotated[list, add_messages]
    topic: str
    current_speaker: Literal["agent_1", "agent_2", "summarizer"]
    turn_count: int
    max_turns: int
    summary_interval: int  # How often to summarize (every N turns)


# Initialize models - Agent 1 uses GPT-4o, Agent 2 uses Perplexity
model_agent1 = ChatOpenAI(
    model="gpt-4o",
    api_key=openai_key
)


# Perplexity uses OpenAI-compatible API
model_agent2 = ChatOpenAI(
    model="sonar",  # Perplexity's default model
    api_key=perplexity_key,
    base_url="https://api.perplexity.ai"
)

# Summarizer agent - uses GPT-4o 
model_summarizer = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_key,
    temperature=0.3  
)

def clean_response_content(response) -> str:
    """Extract and clean content from LLM response."""
    # Extract content based on type
    if isinstance(response.content, str):
        content = response.content
    elif isinstance(response.content, list):
        content = "".join(
            block.text if hasattr(block, 'text')
            else block.get('text', str(block)) if isinstance(block, dict)
            else str(block)
            for block in response.content
        )
    else:
        content = str(response.content)

    # Remove agent labels and clean whitespace
    content = re.sub(r'^\[Agent \d\]:\s*', '', content)
    content = re.sub(r'\n\[Agent \d\]:\s*.*$', '', content, flags=re.DOTALL)
    return content.strip()


def agent_1_node(state: ConversationState) -> dict:
    """Agent 1's turn to speak."""
    # Get turn count with default value
    turn_count = state.get('turn_count', 0)
    max_turns = state.get('max_turns', 8)

    system_prompt = f"""You are Agent 1 in a discussion about: {state['topic']}

You are having a thoughtful conversation with Agent 2. Your task is to provide ONE response as Agent 1.

Important guidelines:
- Only speak as Agent 1, never as Agent 2
- Do NOT include labels like "[Agent 1]:" or "[Agent 2]:" in your response
- Provide only YOUR response, not the entire conversation
- Share your perspective, ask questions, and engage meaningfully
- Keep your response concise (2-4 sentences)
- MAXIMUM 100 WORDS - keep your response brief and focused

This is turn {turn_count} of {max_turns}."""

    # Prepare messages
    messages = [SystemMessage(content=system_prompt)]

    if state["messages"]:
        messages.extend(state["messages"])
    else:
        messages.append(HumanMessage(content=f"Begin the discussion about: {state['topic']}"))

    # Get response and clean
    response = model_agent1.invoke(messages)
    content = clean_response_content(response)

    if not content:
        content = "I'd like to hear your thoughts on this topic."

    return {
        "messages": [AIMessage(content=f"[Agent 1]: {content}")],
        "current_speaker": "agent_2",
        "turn_count": turn_count + 1
    }


def agent_2_node(state: ConversationState) -> dict:
    """Agent 2's turn to speak."""
    # Get turn count with default value
    turn_count = state.get('turn_count', 0)
    max_turns = state.get('max_turns', 8)

    system_prompt = f"""You are Agent 2 in a discussion about: {state['topic']}

You are having a thoughtful conversation with Agent 1. Your task is to provide ONE response as Agent 2.

Important guidelines:
- Only speak as Agent 2, never as Agent 1
- Do NOT include labels like "[Agent 1]:" or "[Agent 2]:" in your response
- Provide only YOUR response, not the entire conversation
- Respond to Agent 1's points and offer your own insights
- Keep your response concise (2-4 sentences)
- MAXIMUM 100 WORDS - keep your response brief and focused

This is turn {turn_count} of {max_turns}."""

    # Prepare messages - swap roles so Agent 1's messages appear as user input
    messages = [SystemMessage(content=system_prompt)]

    for msg in state["messages"]:
        cleaned_content = re.sub(r'^\[Agent \d\]:\s*', '', msg.content).strip()

        # Determine which agent sent this message by checking the label
        if "[Agent 1]" in msg.content:
            # Agent 1's messages become user input for Agent 2
            messages.append(HumanMessage(content=cleaned_content))
        elif "[Agent 2]" in msg.content:
            # Agent 2's previous messages
            messages.append(AIMessage(content=cleaned_content))

    # Get response and clean
    response = model_agent2.invoke(messages)
    content = clean_response_content(response)

    if not content:
        content = "That's an interesting point. Let me share my perspective on this."

    return {
        "messages": [AIMessage(content=f"[Agent 2]: {content}")],
        "current_speaker": "agent_1",
        "turn_count": turn_count + 1
    }


def summarizer_node(state: ConversationState) -> dict:
    """Summarizer agent that provides periodic summaries of the conversation."""
    # Get recent messages (last summary_interval turns)
    interval = state.get("summary_interval", 4)
    recent_messages = state["messages"][-interval:] if len(state["messages"]) >= interval else state["messages"]

    # Build conversation context
    conversation_text = ""
    for msg in recent_messages:
        conversation_text += msg.content + "\n\n"

    system_prompt = f"""You are a neutral summarizer reviewing a discussion about: {state['topic']}

Your role is to provide a brief, objective summary of the last few turns of conversation.

Guidelines:
- Summarize the main points discussed in the recent exchange
- Highlight any agreements, disagreements, or new insights
- Keep it concise (3-4 sentences maximum)
- Be objective and balanced
- Use third person perspective"""

    user_prompt = f"""Summarize the following conversation segment:

{conversation_text}

Provide a brief summary of what has been discussed."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Get response
    response = model_summarizer.invoke(messages)
    content = clean_response_content(response)

    if not content:
        content = "Summary: The agents have been discussing various aspects of the topic."

    return {
        "messages": [AIMessage(content=f"[Summarizer]: {content}")],
        "current_speaker": "agent_1",  # Resume with agent_1 after summary
        "turn_count": state.get("turn_count", 0)  
    }


def should_continue(state: ConversationState) -> Literal["continue", "end"]:
    """Determine if the conversation should continue."""
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 8)
    return "end" if turn_count >= max_turns else "continue"


def route_after_agent(state: ConversationState) -> Literal["summarize", "continue", "end"]:
    """Route to summarizer every N turns, or continue/end conversation."""
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 8)

    # Check if we've reached max turns
    if turn_count >= max_turns:
        return "end"

    # Check if it's time to summarize
    interval = state.get("summary_interval", 4)
    if turn_count > 0 and turn_count % interval == 0:
        return "summarize"

    return "continue"


# Build the graph
graph_builder = StateGraph(ConversationState)

# Add nodes
graph_builder.add_node("agent_1", agent_1_node)
graph_builder.add_node("agent_2", agent_2_node)
graph_builder.add_node("summarizer", summarizer_node)

# Set entry point
graph_builder.add_edge(START, "agent_1")

# Add conditional edges with summarizer routing
graph_builder.add_conditional_edges(
    "agent_1",
    route_after_agent,
    {
        "summarize": "summarizer",
        "continue": "agent_2",
        "end": END
    }
)

graph_builder.add_conditional_edges(
    "agent_2",
    route_after_agent,
    {
        "summarize": "summarizer",
        "continue": "agent_1",
        "end": END
    }
)

# After summarizer, continue with agent_1
graph_builder.add_conditional_edges(
    "summarizer",
    should_continue,
    {"continue": "agent_1", "end": END}
)

# Compile and export
graph = graph_builder.compile()
app = graph
