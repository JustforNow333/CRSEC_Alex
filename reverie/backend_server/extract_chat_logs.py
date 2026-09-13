"""
Extract non-null chat logs from simulation movement files into a text file.

Usage:
  python extract_chat_logs.py [sim_code]

If sim_code is omitted, reads from temp_storage/curr_sim_code.json.
Output: environment/frontend_server/storage/<sim_code>/conversations_extracted.txt
"""

import json
import os
import sys

# Paths relative to this script (reverie/backend_server)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from utils import fs_storage
except ImportError:
    fs_storage = os.path.join(SCRIPT_DIR, "../../environment/frontend_server/storage")


def get_sim_code():
    """Get sim_code from argument or from temp_storage/curr_sim_code.json."""
    if len(sys.argv) >= 2:
        return sys.argv[1].strip()
    temp_path = os.path.join(SCRIPT_DIR, "../../environment/frontend_server/temp_storage/curr_sim_code.json")
    if os.path.isfile(temp_path):
        with open(temp_path, encoding="utf-8") as f:
            return json.load(f).get("sim_code", "")
    return ""


def extract_chats_from_movement_file(filepath):
    """
    Read a single movement JSON; return list of (curr_time, chat) with chat = list of [speaker, utterance].
    Dedupes by participant pair (same conversation appears for both personas).
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    curr_time = meta.get("curr_time", "")
    persona = data.get("persona", data)
    if not isinstance(persona, dict):
        return []
    seen_keys = set()
    out = []
    for name, pdata in persona.items():
        chat = pdata.get("chat") if isinstance(pdata, dict) else None
        if not chat or len(chat) < 2:
            continue
        # Dedupe: same conversation for both participants
        a, b = chat[0][0], chat[1][0]
        key = tuple(sorted([a, b]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append((curr_time, chat))
    return out


def format_conversation(step, curr_time, chat):
    """Format one conversation for the txt file."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Step {step}  |  {curr_time}")
    a, b = chat[0][0], chat[1][0]
    lines.append(f"CONVERSATION: {a} & {b}")
    lines.append("=" * 60)
    for speaker, utterance in chat:
        lines.append(f"{speaker}: {utterance}")
    lines.append("")
    return "\n".join(lines)


def run_extract(sim_code=None):
    if not sim_code:
        sim_code = get_sim_code()
    if not sim_code:
        print("Usage: python extract_chat_logs.py <sim_code>")
        print("Or set curr_sim_code.json in environment/frontend_server/temp_storage/")
        sys.exit(1)

    movement_dir = os.path.join(fs_storage, sim_code, "movement")
    if not os.path.isdir(movement_dir):
        print(f"Movement directory not found: {movement_dir}")
        sys.exit(1)

    # Collect (step, curr_time, chat) for all steps, deduped per step
    step_conversations = []
    step_files = sorted(
        [f for f in os.listdir(movement_dir) if f.endswith(".json")],
        key=lambda f: int(f.replace(".json", "")) if f.replace(".json", "").isdigit() else -1
    )
    for filename in step_files:
        step = filename.replace(".json", "")
        if not step.isdigit():
            continue
        step = int(step)
        filepath = os.path.join(movement_dir, filename)
        for curr_time, chat in extract_chats_from_movement_file(filepath):
            step_conversations.append((step, curr_time, chat))

    out_path = os.path.join(fs_storage, sim_code, "conversations_extracted.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("EXTRACTED CONVERSATIONS FROM SIMULATION MOVEMENT FILES\n")
        f.write("=" * 60 + "\n")
        f.write(f"Simulation: {sim_code}\n")
        f.write(f"Total conversations extracted: {len(step_conversations)}\n")
        f.write("=" * 60 + "\n")
        for step, curr_time, chat in step_conversations:
            f.write(format_conversation(step, curr_time, chat))

    print(f"Wrote {len(step_conversations)} conversations to {out_path}")
    return out_path


if __name__ == "__main__":
    run_extract()
