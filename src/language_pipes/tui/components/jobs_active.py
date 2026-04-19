from typing import Callable, List

from language_pipes.content_loader import ContentLoader
from language_pipes.content_provider.job_provider import MetaJob
from language_pipes.content_provider.provider_calls import ProviderCall
from language_pipes.tui.util.kb_utils import PressedKey


class JobsActive:
    provider: ContentProvider
    exit_page: Callable
    is_focused: Callable[[], bool]

    def __init__(self, provider: ContentProvider, exit_page: Callable, is_focused: Callable):
        self.provider = provider
        self.exit_page = exit_page
        self.is_focused = is_focused

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()
    
    def get_view(self) -> List[str]:
        lines = ["Active Jobs:", ""]

        jobs: List[MetaJob] = self.provider.call_provider(ProviderCall.get_active_jobs)
        for job in jobs:
            lines.extend([
                f"Model ID:      {job.model_id}",
                f"Origin Node:   {job.origin_node_id}", 
                f"Job ID:        {job.job_id[:8]}",
                f"Pipe ID:       {job.pipe_id[:8]}",
                f"Last active:   {job.last_update:.0f} seconds ago",
                f"Current Token: {job.current_token}"
                "", ""
            ])
        
        if len(jobs) == 0:
            lines.extend(["No Active Jobs..."])

        return lines

    def get_footer(self) -> str:
        return ""