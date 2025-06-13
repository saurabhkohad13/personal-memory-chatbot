import time
from memory.memory_store import (
    load_memory_from_file,
    build_faiss_index,
    store_memory,
    retrieve_relevant_memories
)
from models.model_loader import load_model
from utils.prompts import get_prompt
from utils.logger import setup_logger

# -------------------------
# Initialization
# -------------------------

# Set up the logger for debugging and monitoring
logger = setup_logger()

# Load existing memory from disk (if available)
load_memory_from_file()

# Load the local LLM model (Mistral via llama.cpp)
model = load_model()

# Build FAISS index from stored memory embeddings
index = build_faiss_index()
if index is None:
    print("[ERROR] FAISS index was not created properly. Check build_faiss_index() function.")
else:
    print("[INFO] FAISS index built successfully.")

# -------------------------
# Helper Functions
# -------------------------

def extract_bot_reply(output):
    """
    Extracts the bot's actual reply from the raw model output.
    Handles fallback conditions for hallucination or empty replies.
    """
    output = output.strip()

    # Remove any trailing end marker
    if "[end of text]" in output:
        output = output.split("[end of text]")[0].strip()

    # Check if the output is meaningful
    if not output or output.lower().startswith("you are a helpful assistant"):
        return "Sorry, I couldn't find your question in the output."

    return output

def generate_response(query):
    """
    Handles the full flow of:
    - Retrieving relevant past memories
    - Building prompt
    - Calling the LLM
    - Extracting the response
    - Storing the interaction in memory
    """
    # Retrieve top-k similar memories for the current query
    relevant_memories = retrieve_relevant_memories(query)
    context = "\n".join(relevant_memories)

    # Format the full prompt using memory context
    prompt = get_prompt(context, query)
    logger.debug(f"[DEBUG] Generated Prompt:\n{prompt}")

    # Call the LLM with the constructed prompt
    raw_output = model(prompt)
    logger.debug(f"[DEBUG] Raw Model Output:\n{raw_output}")

    # Extract only the meaningful bot reply
    bot_reply = extract_bot_reply(raw_output)

    # Store valid conversation in memory
    if bot_reply and "Sorry" not in bot_reply:
        store_memory(f"User: {query}\nBot: {bot_reply}")
        logger.info("[Memory] Saved 1 memory.")
    else:
        print("[INFO] Bot response was empty or invalid. Skipping memory storage.")

    return bot_reply or "Sorry, I didn't quite catch that."

# -------------------------
# Main Chat Loop
# -------------------------

def chat():
    """
    Starts an interactive chatbot loop in the terminal.
    """
    print("[Chat] Welcome to your personal memory assistant!")
    print("[Chat] Type 'exit' to stop chatting.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("[Chat] Goodbye! Memory saved. See you soon.")
            break

        bot_response = generate_response(user_input)
        print(f"Bot: {bot_response}")

# Entry point
if __name__ == "__main__":
    chat()