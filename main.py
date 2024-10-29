# main.py

from src.core.config import ConfigManager

def main():
    # Load configuration
    config_manager = ConfigManager()
    
    # Validate configuration
    if not config_manager.validate_config():
        print("Invalid configuration. Exiting.")
        return
    
    # Setup logging
    config_manager.setup_logging()

    # Placeholder for running the strategy
    run_strategy(config_manager)

def run_strategy(config_manager):
    # Implement strategy logic here
    # Example:
    print("Running Omnia Strategy")
    print("Configuration loaded:")
    print(config_manager.config)

if __name__ == "__main__":
    main()