# W&B login.

# This does not need an internet connection.
# OPTION 1: Set WANDB_API_KEY in the environment.
if [ -n "${WANDB_API_KEY}" ]; then
  echo "[TEMPLATE INFO] Logging in to W&B."
  wandb login "${WANDB_API_KEY}"
fi
# OPTION 2: Set WANDB_API_KEY_FILE_AT in the environment which points to a file containing the key.
if [ -n "${WANDB_API_KEY_FILE_AT}" ]; then
  echo "[TEMPLATE INFO] Logging in to W&B."
  wandb login "$(cat "${WANDB_API_KEY_FILE_AT}")"
fi

# Hugging Face login.
if [ -n "${HF_TOKEN_AT}" ]; then
  echo "[TEMPLATE INFO] Logging in to Hugging Face."
  huggingface-cli login --token "$(cat "${HF_TOKEN_AT}")"
fi

if [ -n "${HUGGINGFACE_KEY_FILE_AT}" ]; then
  echo "[TEMPLATE INFO] Logging in to HuggingFace."
  echo "$(cat "${HUGGINGFACE_KEY_FILE_AT}")"
  huggingface-cli login --token "$(cat "${HUGGINGFACE_KEY_FILE_AT}")"
fi
