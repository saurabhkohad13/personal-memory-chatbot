import logging

def setup_logger():
    """
    Configures and returns a logger for debugging and information output.

    - Sets the logging level to DEBUG to capture all levels of log messages.
    - Outputs logs to the console with a timestamp, severity level, and message.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all logs from DEBUG and above
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format: timestamp - level - message
        handlers=[logging.StreamHandler()]  # Print logs to the console
    )
    logger = logging.getLogger()
    return logger