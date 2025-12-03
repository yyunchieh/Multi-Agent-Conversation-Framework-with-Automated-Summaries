"""
Test script to run a conversation between two agents.
"""

import os
from datetime import datetime
from agent import app
from summarizer import ConversationSummarizer

def save_conversation_as_markdown(result, topic, max_turns, include_summary=True):
    # Create conversation_results directory
    output_dir = "conversation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/conversation_{timestamp}.md"

    # Build markdown content
    markdown_content = f"""# AI Agent Conversation

## Metadata
- **Topic**: {topic}
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Turns**: {result['turn_count']}
- **Max Turns**: {max_turns}
- **Agent 1**: GPT-4o
- **Agent 2**: Perplexity (Llama 3.1 Sonar)

---

"""

    # Generate and add summary if requested
    if include_summary:
        print("\nGenerating conversation summary...")
        summarizer = ConversationSummarizer()
        summary = summarizer.generate_summary(result["messages"], topic)
        summary_md = summarizer.format_summary_as_markdown(summary)
        markdown_content += summary_md
        markdown_content += "---\n\n"

    markdown_content += "## Full Conversation\n\n"

    # Add each message to the markdown
    turn_number = 0
    for message in result["messages"]:
        # Extract agent number from content
        if "[Agent 1]" in message.content:
            turn_number += 1
            agent = "Agent 1 (GPT-4o)"
            content = message.content.replace("[Agent 1]:", "").strip()
            markdown_content += f"### Turn {turn_number}: {agent}\n\n"
            markdown_content += f"{content}\n\n"
            markdown_content += "---\n\n"
        elif "[Agent 2]" in message.content:
            turn_number += 1
            agent = "Agent 2 (Perplexity)"
            content = message.content.replace("[Agent 2]:", "").strip()
            markdown_content += f"### Turn {turn_number}: {agent}\n\n"
            markdown_content += f"{content}\n\n"
            markdown_content += "---\n\n"
        elif "[Summarizer]" in message.content:
            agent = "Summarizer (GPT-4o)"
            content = message.content.replace("[Summarizer]:", "").strip()
            markdown_content += f"### Periodic Summary by {agent}\n\n"
            markdown_content += f"> {content}\n\n"
            markdown_content += "---\n\n"
        else:
            agent = "Unknown"
            content = message.content
            markdown_content += f"### {agent}\n\n"
            markdown_content += f"{content}\n\n"
            markdown_content += "---\n\n"

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return filename

def main():
    print("=" * 70)
    print("TWO-AGENT CONVERSATION SYSTEM TEST\nAgent 1: GPT-4o\nAgent 2: Perplexity (Llama 3.1 Sonar)")
    print("=" * 70)
    print()

    # Set up the conversation
    topic = "How AI can help address educational inequalities?"
    max_turns = 8
    summary_interval = 4  

    print(f"Topic: {topic}")
    print(f"Max turns: {max_turns}")
    print(f"Summary interval: Every {summary_interval} turns")
    print()
    print("-" * 70)
    print()

    # Run the conversation
    result = app.invoke({
        "messages": [],
        "topic": topic,
        "current_speaker": "agent_1",
        "turn_count": 0,
        "max_turns": max_turns,
        "summary_interval": summary_interval
    })

    # Print the conversation
    for message in result["messages"]:
        print(message.content)
        print()
        print("-" * 70)
        print()

    print(f"Conversation completed after {result['turn_count']} turns.")

    # Save to markdown
    filename = save_conversation_as_markdown(result, topic, max_turns)
    print(f"\nConversation saved to: {filename}")
    print()

if __name__ == "__main__":
    main()
