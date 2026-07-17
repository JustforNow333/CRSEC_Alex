import numpy as np
import pandas as pd
import time
import os
import json
import argparse
import re
import shutil
from glob import glob


# =========================
# Part 1: Randomized setup
# =========================
AGENTS = [
    "Mary Smith",
    "Bob Johnson",
    "Robert Brown",
    "David Evans",
    "Linda Taylor",
    "Jennifer Moore",
    "Sam Moore",
    "Sarah Taylor",
    "Tom Brown",
    "Andrew Williams",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def classify(conviction, trust):
    if conviction < 0.0 and trust < 0.0:
        return "Apathetic Cynic"
    elif conviction < 0.0 and trust >= 0.0:
        return "Passive Conformist"
    elif conviction >= 0.0 and trust < 0.0:
        return "Grassroots Activist / Dissenter"
    else:
        return "Institutional Optimist / Policy Ally"


def assign_profiles():
    conviction = np.random.uniform(-1, 1, len(AGENTS))
    trust = np.random.uniform(-1, 1, len(AGENTS))
    archetypes = [classify(c, t) for c, t in zip(conviction, trust)]
    df = pd.DataFrame(
        {
            "Agent": AGENTS,
            "Conviction": conviction.round(2),
            "Trust": trust.round(2),
            "Archetype": archetypes,
        }
    )
    return df


def run_randomization(seed=None, minority_size=2, max_draws=10000):
    """
    Draw agent profiles until exactly `minority_size` Grassroots Activists are
    selected and Jennifer Moore is not among them.

    Parameters
    ----------
    seed : int or None
        RNG seed. If None, uses int(time.time()) (original behaviour).
    minority_size : int
        Target number of Grassroots Activist / Dissenter agents.
    max_draws : int
        Maximum rejection-sampling iterations before raising ValueError.

    Returns
    -------
    (seed_used, df)
    """
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    print("Seed used for this run:", seed)

    for _ in range(max_draws):
        df = assign_profiles()
        activists = df[df["Archetype"] == "Grassroots Activist / Dissenter"]
        # Constraint: exactly minority_size activists, Jennifer Moore cannot be one.
        if (len(activists) == minority_size and
                "Jennifer Moore" not in activists["Agent"].values):
            break
    else:
        raise ValueError(
            f"Could not find a sample with exactly {minority_size} Grassroots Activists "
            f"(excluding Jennifer Moore) after {max_draws} draws. "
            f"Try a different minority_size or increase max_draws."
        )

    print("\nFinal assignment:")
    print(df.to_string(index=False))
    print("\nGrassroots activists selected:")
    print(activists[["Agent", "Conviction", "Trust"]].to_string(index=False))

    output_path = os.path.join(BASE_DIR, "bio-initialization.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Seed used for this run: {seed}\n\n")
        f.write("Final assignment:\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    return seed, df


def _extract_json_object(text: str):
    """Best-effort extraction of first JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _build_prompt(base_scratch, info_text, stance_row):
    return f"""
Edit those biographies based on the language of the papers I provided and the the information summarized. Because the factors in my experiment are the level of conviction about climate change and the trust in government, make sure to explicitly mention the conviction and trust levels for each agent in the "learned" category. Change only "innate" and "learned" categories, and the "currently" category to make them aligned with our  climate change opinion experiment, but do not change the format, only the content. Here is a table indicating the agents' stance on the scale from -1 to 1. Make sure the language reflects the agent's numerical value of the attitude, but do not include the value in the updated biography.

Stance for this agent:
- Agent: {stance_row["Agent"]}
- Conviction: {stance_row["Conviction"]}
- Trust: {stance_row["Trust"]}
- Archetype: {stance_row["Archetype"]}

Reference information and papers summary:
{info_text}

Current biography fields:
- innate: {base_scratch.get("innate", "")}
- learned: {base_scratch.get("learned", "")}
- currently: {base_scratch.get("currently", "")}

Return JSON only with exactly these keys:
{{
  "innate": "<new innate text>",
  "learned": "<new learned text>",
  "currently": "<new currently text>"
}}
"""


def _call_model(prompt_text, model_name):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment.")
    try:
        # Newer SDK style (openai>=1.x)
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You edit persona biography fields and return strict JSON only.",
                },
                {"role": "user", "content": prompt_text},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except ImportError:
        # Older SDK style (openai<1.x)
        try:
            import openai  # type: ignore

            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You edit persona biography fields and return strict JSON only.",
                    },
                    {"role": "user", "content": prompt_text},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise ImportError(
                "OpenAI SDK is missing/incompatible. Install with: pip install -U openai"
            ) from exc


def update_biographies(df, information_path, base_storage_dir, model_name):
    with open(information_path, "r", encoding="utf-8") as f:
        info_text = f.read().strip()

    scratch_paths = sorted(
        glob(os.path.join(base_storage_dir, "personas", "*", "bootstrap_memory", "scratch.json"))
    )
    if not scratch_paths:
        raise FileNotFoundError(f"No scratch.json found under: {base_storage_dir}")

    stance_map = {row["Agent"]: row for _, row in df.iterrows()}
    grassroots_set = set(
        df[df["Archetype"] == "Grassroots Activist / Dissenter"]["Agent"].tolist()
    )
    updated = 0
    skipped = []
    updated_bios = []

    for scratch_path in scratch_paths:
        with open(scratch_path, "r", encoding="utf-8") as f:
            scratch = json.load(f)

        name = scratch.get("name", "").strip()
        if name not in stance_map:
            skipped.append(name or scratch_path)
            continue

        prompt = _build_prompt(scratch, info_text, stance_map[name])
        raw = _call_model(prompt, model_name=model_name)
        payload = _extract_json_object(raw)
        if not payload:
            raise ValueError(f"Model returned non-JSON output for {name}: {raw[:300]}")

        for key in ("innate", "learned", "currently"):
            if key not in payload:
                raise ValueError(f"Missing '{key}' in model output for {name}.")
            scratch[key] = str(payload[key]).strip()

        # Final rule: identity by archetype
        # - Grassroots Activist / Dissenter -> entrepreneur
        # - everyone else -> citizen
        scratch["identity"] = "entrepreneur" if name in grassroots_set else "citizen"

        updated_bios.append(
            {
                "name": name,
                "innate": scratch["innate"],
                "learned": scratch["learned"],
                "currently": scratch["currently"],
                "identity": scratch["identity"],
            }
        )

        with open(scratch_path, "w", encoding="utf-8") as f:
            json.dump(scratch, f, ensure_ascii=False, indent=2)
            f.write("\n")
        updated += 1
        print(f"Updated: {name}")

    print(f"\nBiography update complete. Updated {updated} personas.")
    if skipped:
        print(f"Skipped (no stance row): {', '.join(skipped)}")

    # Append rewritten biographies to bio-initialization.txt for run traceability.
    bio_log_path = os.path.join(BASE_DIR, "bio-initialization.txt")
    with open(bio_log_path, "a", encoding="utf-8") as f:
        f.write("\n\nUpdated biographies:\n")
        f.write("=" * 80 + "\n")
        for row in sorted(updated_bios, key=lambda x: x["name"]):
            f.write(f"Agent: {row['name']}\n")
            f.write(f"identity: {row['identity']}\n")
            f.write(f"innate: {row['innate']}\n")
            f.write(f"learned: {row['learned']}\n")
            f.write(f"currently: {row['currently']}\n")
            f.write("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize random ideology stances and rewrite base biographies."
    )
    parser.add_argument(
        "--information-file",
        default=os.path.join(BASE_DIR, "information.txt"),
        help="Path to information.txt with paper links and summary guidance.",
    )
    parser.add_argument(
        "--base-storage-dir",
        default=os.path.join(
            BASE_DIR, "..", "environment", "frontend_server", "storage", "base_the_ville_n10"
        ),
        help="Path to base simulation storage directory to update scratch.json files.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model name for biography rewriting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible assignments. Defaults to int(time.time()).",
    )
    parser.add_argument(
        "--minority-size",
        type=int,
        default=2,
        help="Number of Grassroots Activist / Dissenter agents to select (default: 2).",
    )
    parser.add_argument(
        "--out-base-name",
        default=None,
        help=(
            "If set, copy --base-storage-dir to a sibling directory with this name "
            "and run update_biographies on the copy, leaving the original untouched."
        ),
    )
    args = parser.parse_args()

    _, df = run_randomization(seed=args.seed, minority_size=args.minority_size)

    base_storage_dir = os.path.abspath(args.base_storage_dir)

    if args.out_base_name:
        parent = os.path.dirname(base_storage_dir)
        dest = os.path.join(parent, args.out_base_name)
        shutil.copytree(base_storage_dir, dest)
        target_dir = dest
    else:
        target_dir = base_storage_dir

    update_biographies(
        df=df,
        information_path=args.information_file,
        base_storage_dir=target_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
