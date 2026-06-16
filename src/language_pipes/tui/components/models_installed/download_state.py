from typing import Dict, List, Optional

from ansinout import PressedKey

from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


class DownloadPageState(PageState):
    METHODS: List[str] = ["Download from Huggingface", "Request model locally"]

    def __init__(self):
        super().__init__('download')
        self.new_model_id = ""
        self.downloading = False
        self.download_status: Optional[str] = None
        self.choosing_method = False
        self.method_idx = 0

    def on_change(self, args: Dict):
        if "token" in args:
            self._start_download(args["token"])
            return
        if "fresh" in args:
            self.new_model_id = ""
            self.downloading = False
            self.download_status = None
            self.choosing_method = False
            self.method_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if self.choosing_method:
            self._on_method_key(key)
            return
        if key == PressedKey.Alpha:
            self._on_char(ch)
        if key == PressedKey.Backspace:
            self._on_backspace()
        if key == PressedKey.Enter:
            self._on_enter()
        if key == PressedKey.Escape:
            self._on_escape()

    def _on_method_key(self, key: PressedKey):
        if key == PressedKey.ArrowUp:
            self.method_idx = (self.method_idx - 1) % len(self.METHODS)
        if key == PressedKey.ArrowDown:
            self.method_idx = (self.method_idx + 1) % len(self.METHODS)
        if key == PressedKey.Enter:
            self._on_method_select()
        if key == PressedKey.Escape:
            self.choosing_method = False

    def _on_method_select(self):
        self.choosing_method = False
        if self.method_idx == 0:
            self._download_from_huggingface()
        else:
            self._request_locally()

    def _download_from_huggingface(self):
        if self.new_model_id in self.provider.model_provider.get_installed_models():
            def on_apply():
                self.provider.model_provider.delete_installed_model(self.new_model_id)
                self._request_token()

            self.confirm.open(
                f"{self.new_model_id} already exists\ndelete it and download again?",
                on_apply=on_apply,
                on_discard=lambda: None
            )
        else:
            self.confirm.open(
                f"Download {self.new_model_id}",
                on_apply=self._request_token,
                on_discard=lambda: None,
            )

    def _request_locally(self):
        # TODO: not yet hooked up to a backend
        pass

    def _on_char(self, ch: str):
        if not self.downloading:
            self.new_model_id += ch

    def _on_backspace(self):
        if not self.downloading:
            self.new_model_id = self.new_model_id[:-1]

    def _on_enter(self):
        if self.downloading:
            if self._download_done():
                self.downloading = False
                if  self.download_status is not None and "SUCCESS" in self.download_status:
                    self.change_state('list', {})
            else:
                self.confirm.open(
                    "Stop Current Download?",
                    on_apply=self._stop_download,
                    on_discard=lambda: None,
                )
        else:
            if not self._can_download():
                return

            self.choosing_method = True
            self.method_idx = 0

    def _download_done(self):
        return not self.downloading or self.download_status is not None and ("SUCCESS" in self.download_status or "ERROR" in self.download_status)

    def _on_escape(self):
        if not self._download_done():
            self.confirm.open(
                "Stop Current Download?",
                on_apply=self._stop_download,
                on_discard=lambda: None,
            )
        else:
            self.change_state('list', {})

    def _stop_download(self):
        self.provider.model_provider.stop_model_download()
        self.downloading = False

    def _request_token(self):
        cfg_token = ModelProvider.get_hf_config_token()
        if cfg_token is not None:
            self.confirm.open(
                "Use saved token?",
                on_apply=lambda: self._start_download(cfg_token),
                on_discard=lambda: None,
            )
            return

        def enter_key():
            self.change_state('api_key', {})

        self.confirm.open(
            "Enter Huggingface API Key?\nAPI Keys allow you to download gated models and better rate limits.",
            on_apply=enter_key,
            on_discard=lambda: self._start_download(None),
        )

    def _start_download(self, token: Optional[str] = None):
        self.provider.model_provider.start_download(self.new_model_id, token)
        self.downloading = True

    def _can_download(self) -> bool:
        if self.new_model_id == "":
            return False
        try:
            self.new_model_id.index("/")
        except ValueError:
            return False
        return True

    def get_view(self) -> List[str]:
        model_id = self.new_model_id[-40:] if len(self.new_model_id) > 40 else self.new_model_id
        lines = ["Install New Model:", "", "Type the huggingface model ID from huggingface.co", ""]

        cfg_key = ModelProvider.get_hf_config_token()
        if cfg_key is not None:
            lines.extend(["Global API Key loaded", ""])

        if self.choosing_method:
            lines.extend([f"How would you like to get {model_id}?", ""])
            for i, method in enumerate(self.METHODS):
                lines.append(make_selectable_text(method, self.method_idx == i))
                lines.append("")
        elif self.downloading:
            lines.extend([f"Downloading {model_id}...", ""])
            self.download_status = self.provider.model_provider.check_download_progress()
            if self.download_status is not None:
                lines.append(self.download_status)
                if "SUCCESS" in self.download_status or "ERROR" in self.download_status:
                    lines.extend(["", "Press Enter to continue..."])
        else:
            lines.extend([f"Model ID: {model_id}|", ""])
            if self.new_model_id == "":
                lines.append("!Warning: Model ID empty")
            else:
                try:
                    self.new_model_id.index("/")
                except ValueError:
                    lines.append("!Warning: model name must be in [organization]/[model] format")

            if cfg_key is None:
                lines.extend(["", "TIP: Create an access token at https://huggingface.co/settings/tokens"])

        return lines

    def get_footer(self) -> str:
        if self.choosing_method:
            return make_footer_text(["Arrow U/D: Move", "Enter: Select", "Esc: Back"])
        return make_footer_text(["Type: Model ID", "Enter: Download", "Esc: Back"])
