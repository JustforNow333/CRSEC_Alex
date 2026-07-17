import os
from pathlib import Path

# Load .env from project root (two levels up from backend_server)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

# Set OPENAI_API_KEY in your environment (or a local .env file — do not commit .env).
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
# Put your name
key_owner = "<Astghik>"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose 
debug = True