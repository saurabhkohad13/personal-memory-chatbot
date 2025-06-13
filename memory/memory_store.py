import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Embedding dimension for MiniLM model
DIMENSION = 384

# Global memory store (dictionary: key -> memory entry)
stored_memories = {}

# Global FAISS index for similarity search
index = None

# Load a lightweight sentence embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def load_memory_from_file(memory_file='data/memory_store.json'):
    """
    Loads stored conversation memories from a JSON file.
    If the file doesn't exist, it initializes an empty memory store.
    """
    global stored_memories
    try:
        with open(memory_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                stored_memories = data
                print(f"[Memory] Loaded {len(stored_memories)} memories.")
            else:
                print("[Memory] Warning: Memory file is not a dictionary. Resetting memory.")
                stored_memories = {}
    except FileNotFoundError:
        print("[Memory] No existing memory file found, starting fresh.")


def save_memory_to_file(memory_file='data/memory_store.json'):
    """
    Saves the current in-memory conversation store to a JSON file.
    """
    try:
        with open(memory_file, 'w') as f:
            json.dump(stored_memories, f, indent=2)
        print(f"[Memory] Saved {len(stored_memories)} memories.")
    except Exception as e:
        print(f"[Memory] Error saving memories: {e}")


def store_memory(memory_text):
    """
    Stores a single user-bot interaction and adds its embedding to the FAISS index.
    Only the user part of the message is used to generate the embedding.
    """
    global stored_memories, index

    # Use the first line (user message) for embedding generation
    user_line = memory_text.split("\n")[0].strip()
    embedding = embedder.encode(user_line)

    # Create a new unique memory key
    key = f"memory_{len(stored_memories) + 1}"
    stored_memories[key] = {
        "text": memory_text,
        "embedding": embedding.tolist()  # Save embedding in JSON-compatible format
    }

    # Add new embedding to the FAISS index
    vector = np.array([embedding], dtype="float32")
    index.add(vector)

    # Persist memory to disk
    save_memory_to_file()


def build_faiss_index():
    """
    Builds the FAISS index from the stored memories' embeddings.
    This should be called once during initialization.
    """
    global index
    index = faiss.IndexFlatL2(DIMENSION)

    if stored_memories:
        embeddings = np.array(
            [memory["embedding"] for memory in stored_memories.values()],
            dtype=np.float32
        )
        index.add(embeddings)

    return index


def retrieve_relevant_memories(query, top_k=5, similarity_threshold=0.4):
    """
    Retrieves the top-k most relevant memories for a given user query,
    based on cosine similarity approximation (L2 to cosine mapping).

    Args:
        query (str): User input
        top_k (int): Number of top memories to retrieve
        similarity_threshold (float): Minimum similarity to consider a match

    Returns:
        List of relevant memory text strings
    """
    global index

    if index is None or not stored_memories:
        return []

    # Generate query embedding
    query_embedding = embedder.encode(query)
    query_vector = np.expand_dims(query_embedding, axis=0).astype("float32")

    # Perform FAISS similarity search
    distances, indices = index.search(query_vector, top_k)

    memory_keys = list(stored_memories.keys())
    results = []

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        distance = distances[0][i]
        similarity = 1 / (1 + distance)  # Convert L2 to pseudo-similarity

        if similarity >= similarity_threshold:
            results.append(stored_memories[memory_keys[idx]]["text"])

    return results