#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOS_DIR="$SCRIPT_DIR/repos"

mkdir -p "$REPOS_DIR"

repos=(
  "https://github.com/warpem/warp.git"
  "https://github.com/warpem/warpylib.git"
  "https://github.com/warpem/torch-projectors.git"
  "https://github.com/warpem/warpem.github.io.git"
  "https://github.com/3dem/relion.git"
)

for repo_url in "${repos[@]}"; do
  repo_name=$(basename "$repo_url" .git)
  target="$REPOS_DIR/$repo_name"
  if [ -d "$target" ]; then
    echo "Skipping $repo_name (already cloned)"
  else
    echo "Cloning $repo_name..."
    git clone "$repo_url" "$target"
  fi
done

# Initialize empty state file if it doesn't exist
STATE_FILE="$SCRIPT_DIR/state.json"
if [ ! -f "$STATE_FILE" ]; then
  echo '{"processed_message_ids":[]}' > "$STATE_FILE"
  echo "Created state.json"
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^warpembot "; then
  echo "Creating conda environment..."
  conda create -n warpembot python=3.11 -y
fi

echo "Installing Python dependencies..."
conda run -n warpembot pip install -r "$SCRIPT_DIR/requirements.txt"

# Create threads directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/threads"
touch "$SCRIPT_DIR/threads/.gitkeep"

echo "Setup complete."
