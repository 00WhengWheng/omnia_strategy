import yaml
from pathlib import Path
from typing import Dict, Any
import logging

class ConfigManager:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
        self.setup_logging()

    def load_config(self) -> None:
        # Load the configuration file
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        # Get a value from the configuration
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
                
        return value

    def setup_logging(self) -> None:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/omnia.log'),
                logging.StreamHandler()
            ]
        )

    def validate_config(self) -> bool:
        # Validate the configuration
        required_keys = [
            'core.name',
            'core.version',
            'weights.macro',
            'weights.algorithmic',
            'risk'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logging.error(f"Missing required config key: {key}")
                return False
                
        return True

    def update_config(self, updates: Dict[str, Any]) -> None:
        # Update the configuration
        def update_recursive(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_recursive(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        self.config = update_recursive(self.config, updates)
        self._save_config()

    def _save_config(self) -> None:
        # Save the configuration
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            logging.info(f"Configuration saved successfully to {self.config_path}")
        except Exception as e:
            logging.error(f"Error saving config to {self.config_path}: {str(e)}")
            raise