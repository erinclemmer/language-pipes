from typing import Optional

class ConfigurationArgs:
    config_file: Optional[str]
    auto_start: bool

    def __init__(self, args):
        self.config_file = getattr(args, "config", None)
        self.auto_start = getattr(args, "start", False)
