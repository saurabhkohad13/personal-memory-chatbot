def get_prompt(context, query):
    """
    Constructs the prompt for the language model based on retrieved memory and user query.

    Parameters:
    - context (str): The memory context retrieved from previous conversations.
    - query (str): The current question or message from the user.

    Returns:
    - str: A formatted prompt to be sent to the LLM for generating a response.
    """
    return f"""
                You are a helpful assistant. Use only the following past memory to answer the user. 
                If nothing is relevant, say so. Do not invent answers.

                Past Memory:
                {context if context else "No relevant memory found."}

                User: {query}
                Bot:
            """
