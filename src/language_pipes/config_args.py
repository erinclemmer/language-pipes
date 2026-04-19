import toml
from typing import Dict, List, Optional

class ConfigurationArgs:
    config_file: Optional[str]
    auto_start: bool
    set_overrides: Dict[str, str]
    layer_models: List[str]
    end_models: List[str]

    def __init__(self, args):
        self.config_file = getattr(args, "config", None)
        self.auto_start = getattr(args, "start", False)
        self.set_overrides = self.parse_set_overrides(getattr(args, "set_overrides", []))
        self.layer_models = self.parse_layer_models(getattr(args, "layer_models", []))
        self.end_models = getattr(args, "end_models", [])

    def parse_set_overrides(self, values: List[str]) -> Dict[str, str]:
        overrides = {}
        for item in values:
            key, value = item.split("=", 1)
            try:
                overrides[key] = toml.loads(f"x = {value}")["x"]
            except toml.TomlDecodeError:
                overrides[key] = value
        return overrides

    def parse_layer_models(self, layer_models: List[str]):
        result = []
        for m in layer_models:
            model_config = { }
            for pair in m.split(","):
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid format '{pair}' in '{m}'. "
                        "Expected key=value pairs (e.g., id=Qwen/Qwen3-1.7B,device=cpu,memory=4)"
                    )
                key, value = pair.split("=", 1)
                model_config[key.strip()] = value.strip()
            
            required_keys = {"id", "device", "memory"}
            missing = required_keys - set(model_config.keys())
            if missing:
                raise ValueError(f"Missing required keys {missing} in '{m}'")
            
            result.append({
                "id": model_config["id"],
                "device": model_config["device"],
                "memory": float(model_config["memory"])
            })
        
        return result
