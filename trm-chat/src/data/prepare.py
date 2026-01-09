"""Data preparation scripts for TRM-Chat training.

This module handles:
1. Downloading public chat datasets (OpenAssistant, Dolly)
2. Generating tool-calling training data from Jarvis tool schemas
3. Creating profile-specific training samples
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


def download_public_datasets(output_dir: str, max_samples: int = 50000) -> List[Dict]:
    """Download and process public chat datasets.

    Args:
        output_dir: Directory to save processed data
        max_samples: Maximum total samples to collect

    Returns:
        List of processed samples
    """
    from datasets import load_dataset

    samples = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # OpenAssistant dataset
    print("Loading OpenAssistant dataset...")
    try:
        oasst = load_dataset("OpenAssistant/oasst1", split="train")

        # Process into conversation format
        for item in tqdm(oasst, desc="Processing OpenAssistant"):
            if len(samples) >= max_samples:
                break

            # OASST has tree structure, we'll extract simple Q&A pairs
            if item.get("role") == "prompter" and item.get("parent_id") is None:
                sample = {
                    "messages": [
                        {"role": "user", "content": item["text"]}
                    ],
                    "profile": "life",  # Default profile
                    "has_tool_call": False
                }
                samples.append(sample)

    except Exception as e:
        print(f"Warning: Could not load OpenAssistant: {e}")

    # Dolly dataset
    print("Loading Dolly dataset...")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

        for item in tqdm(dolly, desc="Processing Dolly"):
            if len(samples) >= max_samples:
                break

            messages = []

            # Add context as system if present
            if item.get("context"):
                messages.append({
                    "role": "system",
                    "content": f"Context: {item['context']}"
                })

            messages.append({"role": "user", "content": item["instruction"]})
            messages.append({"role": "assistant", "content": item["response"]})

            # Assign profile based on category
            category = item.get("category", "")
            if category in ["coding", "information_extraction"]:
                profile = "developer"
            elif category in ["brainstorming", "creative_writing"]:
                profile = "business"
            else:
                profile = "life"

            sample = {
                "messages": messages,
                "profile": profile,
                "has_tool_call": False
            }
            samples.append(sample)

    except Exception as e:
        print(f"Warning: Could not load Dolly: {e}")

    # Save combined dataset
    output_file = output_path / "public_chat.jsonl"
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} samples to {output_file}")
    return samples


def generate_tool_calling_data(
    output_dir: str,
    jarvis_tools_path: Optional[str] = None,
    samples_per_tool: int = 200
) -> List[Dict]:
    """Generate tool calling training data from Jarvis tool schemas.

    Args:
        output_dir: Directory to save data
        jarvis_tools_path: Path to Jarvis tools registry
        samples_per_tool: Samples to generate per tool

    Returns:
        List of tool calling samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define Jarvis tools and example usages
    # These are based on tools from src/jarvis/tools/registry.py
    tools = {
        "webSearch": {
            "description": "Search the web for information",
            "parameters": {"search_query": "string"},
            "examples": [
                ("What's the weather in London?", {"search_query": "weather London today"}),
                ("Find restaurants near me", {"search_query": "best restaurants nearby"}),
                ("Latest news about AI", {"search_query": "artificial intelligence news 2024"}),
                ("How tall is Mount Everest?", {"search_query": "Mount Everest height"}),
                ("Best coffee shops downtown", {"search_query": "top rated coffee shops downtown"}),
            ]
        },
        "screenshot": {
            "description": "Take a screenshot of the current screen",
            "parameters": {},
            "examples": [
                ("Take a screenshot", {}),
                ("Capture my screen", {}),
                ("Show me what's on my display", {}),
            ]
        },
        "fetchWebPage": {
            "description": "Fetch and read content from a web page",
            "parameters": {"url": "string"},
            "examples": [
                ("Read this article: https://example.com/article", {"url": "https://example.com/article"}),
                ("What does this page say: https://docs.python.org", {"url": "https://docs.python.org"}),
            ]
        },
        "recallConversation": {
            "description": "Search conversation history",
            "parameters": {"keywords": "list[string]", "from_time": "string", "to_time": "string"},
            "examples": [
                ("What did we talk about yesterday?", {"keywords": [], "from_time": "yesterday", "to_time": "today"}),
                ("Remind me about the meeting", {"keywords": ["meeting"], "from_time": "", "to_time": ""}),
            ]
        },
        "logMeal": {
            "description": "Log a meal for nutrition tracking",
            "parameters": {"meal_name": "string", "calories": "int", "meal_type": "string"},
            "examples": [
                ("I just had a salad for lunch, about 350 calories", {"meal_name": "salad", "calories": 350, "meal_type": "lunch"}),
                ("Log my breakfast: oatmeal with berries", {"meal_name": "oatmeal with berries", "calories": 300, "meal_type": "breakfast"}),
            ]
        }
    }

    samples = []

    # Generate variations for each tool
    for tool_name, tool_info in tools.items():
        print(f"Generating samples for {tool_name}...")

        for query, args in tool_info["examples"]:
            # Create sample with tool call
            sample = {
                "messages": [
                    {"role": "system", "content": "You are Jarvis, a helpful voice assistant. Use tools when needed."},
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "name": tool_name,
                            "arguments": args
                        }]
                    },
                    {"role": "tool", "content": f"[{tool_name} result: success]"},
                    {"role": "assistant", "content": _generate_tool_response(tool_name, query, args)}
                ],
                "profile": _infer_profile_from_tool(tool_name),
                "has_tool_call": True
            }
            samples.append(sample)

            # Generate variations
            for _ in range(samples_per_tool // len(tool_info["examples"])):
                varied_query = _vary_query(query)
                varied_sample = sample.copy()
                varied_sample["messages"] = [
                    {"role": "system", "content": "You are Jarvis, a helpful voice assistant. Use tools when needed."},
                    {"role": "user", "content": varied_query},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "name": tool_name,
                            "arguments": args
                        }]
                    },
                    {"role": "tool", "content": f"[{tool_name} result: success]"},
                    {"role": "assistant", "content": _generate_tool_response(tool_name, varied_query, args)}
                ]
                samples.append(varied_sample)

    # Shuffle and save
    random.shuffle(samples)
    output_file = output_path / "tool_calling.jsonl"
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} tool calling samples to {output_file}")
    return samples


def _vary_query(query: str) -> str:
    """Create variations of a query for data augmentation."""
    prefixes = [
        "Can you ", "Please ", "Hey, ", "I need you to ",
        "Would you ", "Could you ", "Help me ", ""
    ]
    suffixes = [
        "", " please", " for me", " quickly", "?"
    ]

    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)

    # Simple variation
    base = query.rstrip("?").rstrip(" please").rstrip(" for me")
    return f"{prefix}{base.lower()}{suffix}"


def _generate_tool_response(tool_name: str, query: str, args: Dict) -> str:
    """Generate a natural response after tool execution."""
    responses = {
        "webSearch": [
            "Based on what I found, ",
            "According to my search, ",
            "I found that ",
            "Here's what I discovered: "
        ],
        "screenshot": [
            "I've captured your screen.",
            "Screenshot taken.",
            "Here's what's on your screen."
        ],
        "fetchWebPage": [
            "The page contains information about ",
            "From that page, I can see ",
            "Here's a summary: "
        ],
        "recallConversation": [
            "From our previous conversations, ",
            "I remember we discussed ",
            "Looking back at our chat history, "
        ],
        "logMeal": [
            "Got it! I've logged your meal.",
            "Meal recorded successfully.",
            "I've added that to your nutrition log."
        ]
    }

    prefix = random.choice(responses.get(tool_name, ["Done. "]))
    return f"{prefix}[Response based on {tool_name} results]"


def _infer_profile_from_tool(tool_name: str) -> str:
    """Infer appropriate profile from tool type."""
    profile_map = {
        "webSearch": "life",
        "screenshot": "developer",
        "fetchWebPage": "developer",
        "recallConversation": "life",
        "logMeal": "life"
    }
    return profile_map.get(tool_name, "life")


def generate_profile_data(
    output_dir: str,
    samples_per_profile: int = 5000
) -> List[Dict]:
    """Generate profile-specific training data.

    Args:
        output_dir: Directory to save data
        samples_per_profile: Samples to generate per profile

    Returns:
        List of profile-specific samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = []

    # Developer profile examples
    developer_queries = [
        "How do I fix this Python error?",
        "What's the best way to implement a REST API?",
        "Can you explain recursion?",
        "Debug this code for me",
        "How do I use Docker containers?",
        "What design pattern should I use here?",
        "Explain the difference between threads and processes",
        "How do I optimize this database query?",
    ]

    # Business profile examples
    business_queries = [
        "Help me draft an email to the client",
        "What should we consider for the product roadmap?",
        "Analyze these sales figures",
        "How can we improve customer retention?",
        "Prepare talking points for the meeting",
        "What are the key metrics we should track?",
        "Help me prioritize these features",
        "Draft a project proposal",
    ]

    # Life profile examples
    life_queries = [
        "What's a good recipe for dinner tonight?",
        "Remind me about my schedule",
        "Help me plan my weekend",
        "What exercises can I do at home?",
        "Recommend a good book to read",
        "How can I improve my sleep?",
        "What's the weather like?",
        "Set a timer for 10 minutes",
    ]

    profile_data = [
        ("developer", developer_queries),
        ("business", business_queries),
        ("life", life_queries)
    ]

    for profile, queries in profile_data:
        print(f"Generating {profile} profile samples...")

        for _ in range(samples_per_profile // len(queries)):
            for query in queries:
                sample = {
                    "messages": [
                        {"role": "system", "content": f"You are Jarvis in {profile} mode. Focus on {profile}-related assistance."},
                        {"role": "user", "content": _vary_query(query)},
                        {"role": "assistant", "content": f"[{profile.capitalize()} response to: {query}]"}
                    ],
                    "profile": profile,
                    "has_tool_call": False
                }
                samples.append(sample)

    random.shuffle(samples)
    output_file = output_path / "profile_specific.jsonl"
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} profile samples to {output_file}")
    return samples


def prepare_training_data(
    output_dir: str = "./data",
    include_public: bool = True,
    include_tools: bool = True,
    include_profiles: bool = True,
    max_public_samples: int = 50000,
    samples_per_tool: int = 200,
    samples_per_profile: int = 5000
) -> str:
    """Prepare all training data and combine into a single file.

    Args:
        output_dir: Output directory for data files
        include_public: Include public chat datasets
        include_tools: Include tool calling data
        include_profiles: Include profile-specific data
        max_public_samples: Max samples from public datasets
        samples_per_tool: Samples per tool for tool calling
        samples_per_profile: Samples per profile

    Returns:
        Path to combined training file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_samples = []

    if include_public:
        samples = download_public_datasets(output_dir, max_public_samples)
        all_samples.extend(samples)

    if include_tools:
        samples = generate_tool_calling_data(output_dir, samples_per_tool=samples_per_tool)
        all_samples.extend(samples)

    if include_profiles:
        samples = generate_profile_data(output_dir, samples_per_profile=samples_per_profile)
        all_samples.extend(samples)

    # Shuffle all data
    random.shuffle(all_samples)

    # Split into train/val
    split_idx = int(len(all_samples) * 0.95)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Save combined files
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"

    with open(train_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nData preparation complete!")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Train file: {train_file}")
    print(f"  Val file: {val_file}")

    return str(train_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare TRM-Chat training data")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument("--no-public", action="store_true", help="Skip public datasets")
    parser.add_argument("--no-tools", action="store_true", help="Skip tool calling data")
    parser.add_argument("--no-profiles", action="store_true", help="Skip profile data")
    parser.add_argument("--max-public", type=int, default=50000, help="Max public samples")

    args = parser.parse_args()

    prepare_training_data(
        output_dir=args.output_dir,
        include_public=not args.no_public,
        include_tools=not args.no_tools,
        include_profiles=not args.no_profiles,
        max_public_samples=args.max_public
    )
