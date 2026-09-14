import os

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

# --- local open-weight backend (inert unless CRSEC_LLM_BACKEND=local) --------
try:
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_llm"))
    import local_backend
    local_backend.activate()
except Exception as _e:
    print("local_backend not activated:", _e)
