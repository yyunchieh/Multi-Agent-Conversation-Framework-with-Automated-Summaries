"""
Conversation summarizer module for generating insights from agent conversations.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

class ConversationSummarizer:
    """Generates summaries and insights from agent conversations."""

    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3  
        )

    def generate_summary(self, messages: list, topic: str) -> dict:
       
        # Extract conversation text
        conversation_text = self._format_conversation(messages)

        # Generate different summary components
        summary = {
            "executive_summary": self._generate_executive_summary(conversation_text, topic),
            "key_points": self._extract_key_points(conversation_text, topic),
            "main_arguments": self._extract_arguments(conversation_text, topic),
            "conclusions": self._generate_conclusions(conversation_text, topic),
            "topics_discussed": self._extract_topics(conversation_text, topic)
        }

        return summary

    def _format_conversation(self, messages: list) -> str:
        """Format messages into readable conversation text."""
        conversation = []
        for i, msg in enumerate(messages, 1):
            content = msg.content
            # Remove agent labels for cleaner processing
            if "[Agent 1]:" in content:
                content = content.replace("[Agent 1]:", "Agent 1:").strip()
            elif "[Agent 2]:" in content:
                content = content.replace("[Agent 2]:", "Agent 2:").strip()
            conversation.append(f"Turn {i}: {content}")

        return "\n\n".join(conversation)

    def _generate_executive_summary(self, conversation: str, topic: str) -> str:
        system_prompt = """You are an expert at summarizing academic discussions.
Generate a concise executive summary (2-3 sentences) that captures the essence of the conversation."""

        user_prompt = f"""Topic: {topic}

Conversation:
{conversation}

Provide a 2-3 sentence executive summary that captures the main focus and outcome of this discussion."""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response.content.strip()

    def _extract_key_points(self, conversation: str, topic: str) -> list:
        system_prompt = """You are an expert at extracting key insights from discussions.
Identify the most important points made during the conversation."""

        user_prompt = f"""Topic: {topic}

Conversation:
{conversation}

Extract 5-7 key points from this conversation. Format as a bulleted list.
Focus on important insights, solutions, or perspectives discussed."""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Parse response into list
        points = response.content.strip().split('\n')
        # Clean up bullet points
        points = [p.strip('- •*').strip() for p in points if p.strip()]
        return points

    def _extract_arguments(self, conversation: str, topic: str) -> dict:
        """Extract main arguments from both agents."""
        system_prompt = """You are an expert at analyzing arguments in discussions.
Identify the main arguments or perspectives from each agent."""

        user_prompt = f"""Topic: {topic}

Conversation:
{conversation}

Identify the main arguments or perspectives from Agent 1 and Agent 2.
Format as:
Agent 1: [main argument/perspective]
Agent 2: [main argument/perspective]"""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Parse response
        content = response.content.strip()
        lines = content.split('\n')

        arguments = {
            "agent_1": "",
            "agent_2": ""
        }

        for line in lines:
            if line.startswith("Agent 1"):
                arguments["agent_1"] = line.replace("Agent 1:", "").strip()
            elif line.startswith("Agent 2"):
                arguments["agent_2"] = line.replace("Agent 2:", "").strip()

        return arguments

    def _generate_conclusions(self, conversation: str, topic: str) -> str:
        system_prompt = """You are an expert at synthesizing insights from discussions.
Generate conclusions that capture what was learned or agreed upon."""

        user_prompt = f"""Topic: {topic}

Conversation:
{conversation}

Generate a conclusion paragraph (3-4 sentences) that synthesizes the main insights,
areas of agreement, and any actionable takeaways from this discussion."""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response.content.strip()

    def _extract_topics(self, conversation: str, topic: str) -> list:
        system_prompt = """You are an expert at identifying themes in discussions.
Extract the specific sub-topics or themes that were discussed."""

        user_prompt = f"""Topic: {topic}

Conversation:
{conversation}

List 4-6 specific sub-topics or themes that were discussed in this conversation.
Format as a simple list, one per line."""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Parse response into list
        topics = response.content.strip().split('\n')
        topics = [t.strip('- •*0123456789.').strip() for t in topics if t.strip()]
        return topics

    def format_summary_as_markdown(self, summary: dict) -> str:
        md = "## Summary\n\n"

        md += "### Executive Summary\n\n"
        md += f"{summary['executive_summary']}\n\n"

        md += "### Topics Discussed\n\n"
        for topic in summary['topics_discussed']:
            md += f"- {topic}\n"
        md += "\n"

        md += "### Key Points\n\n"
        for i, point in enumerate(summary['key_points'], 1):
            md += f"{i}. {point}\n"
        md += "\n"

        md += "### Main Arguments\n\n"
        md += f"**Agent 1 (GPT-4o)**: {summary['main_arguments']['agent_1']}\n\n"
        md += f"**Agent 2 (Perplexity (Llama 3.1 Sonar))**: {summary['main_arguments']['agent_2']}\n\n"

        md += "### Conclusions\n\n"
        md += f"{summary['conclusions']}\n\n"

        return md
