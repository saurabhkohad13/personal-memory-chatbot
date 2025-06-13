import os
import subprocess

# Define the path to the llama.cpp binary and the GGUF model file
LLAMA_CPP_BIN = os.path.join(os.path.dirname(__file__), "llama.cpp", "build", "bin", "llama-cli")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "llama.cpp", "models", "mistral-7b.Q4_K_M.gguf")

def load_model():
    """
    Returns a callable function to run llama.cpp with the given prompt.

    This setup is optimized for Apple Silicon (M1/M2/M3) using Metal backend.
    It runs the Mistral 7B model locally and returns the response.
    """

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Make sure the .gguf file exists.")
    
    # Check if the llama-cli binary is built
    if not os.path.exists(LLAMA_CPP_BIN):
        raise FileNotFoundError(f"llama-cli binary not found at {LLAMA_CPP_BIN}.")

    def run_llama_cpp(prompt):
        """
        Runs the model using subprocess and returns the processed response.
        """
        # Construct the command to run llama.cpp with the specified settings
        command = [
            LLAMA_CPP_BIN,
            "--model", MODEL_PATH,
            "--prompt", prompt,
            "--n-predict", "256",        # Number of tokens to predict
            "--threads", "4",            # Number of CPU threads to use
            "--top-p", "0.9",            # Top-p sampling
            "--temp", "0.5",             # Temperature for randomness
            "--repeat-penalty", "1.2"    # Penalize repetition in output
        ]

        # Display the prompt being sent to the model
        print(f"[INFO] Running llama.cpp with prompt: \n{prompt}")

        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)

        # Handle any errors during execution
        if result.returncode != 0:
            print("[ERROR] llama.cpp failed:", result.stderr)
            return "Sorry, something went wrong."

        raw_output = result.stdout
        print("[DEBUG] Raw Model Output:\n", raw_output)

        # Try to extract only the actual bot response from the output
        if "Bot:" in raw_output:
            response = raw_output.split("Bot:", 1)[-1]
            for stop_token in ["User:", "Bot:"]:
                if stop_token in response:
                    response = response.split(stop_token)[0]
            return response.strip()
        else:
            return raw_output.strip()

    print("[INFO] llama.cpp model runner ready.")
    return run_llama_cpp