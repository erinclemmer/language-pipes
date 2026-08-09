from typing import Callable, List

from ansinout import PressedKey
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.util.text import make_footer_text, make_window_text

class JobsActive:
    provider: ContentProvider
    exit_page: Callable

    selected_job_idx: int
    num_jobs: int

    def __init__(self, provider: ContentProvider, exit_page: Callable):
        self.provider = provider
        self.exit_page = exit_page
        self.selected_job_idx = 0
        self.num_jobs = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.exit_page()

    def on_next(self):
        if self.num_jobs == 0:
            return
        self.selected_job_idx = (self.selected_job_idx + 1) % self.num_jobs

    def on_prev(self):
        if self.num_jobs == 0:
            return
        self.selected_job_idx = (self.selected_job_idx - 1) % self.num_jobs
    
    def get_view(self) -> List[str]:
        lines = ["Active Jobs:", ""]

        jobs = self.provider.job_provider.get_active_jobs()
        self.num_jobs = len(jobs)
        entries = []
        for job in jobs:
            entry = [
                f"Model ID:      {job.model_id}",
                f"Origin Node:   {job.origin_node_id}", 
                f"Job ID:        {job.job_id[:8]}",
                f"Pipe ID:       {job.pipe_id[:8]}",
                f"Last active:   {job.last_update:.0f} seconds ago",
                f"Decode Token:  {job.current_token}" if not job.progress.prefilling else f"Prefill Token: {job.progress.prefill_tokens} of {job.progress.prompt_tokens}"
            ]

            prefill_speed = job.timing_stats.prefill_times.get_tokens_per_second()
            if prefill_speed > 0:
                entry.extend(["", f"Prefill speed: {prefill_speed:.2f} Tok/s", ""])

            decode_speed = job.timing_stats.output_times.get_tokens_per_second()
            if not job.progress.prefilling and decode_speed > 0:
                entry.extend([
                    "Decoding:",
                    f"Embed time: {job.timing_stats.output_times.get_avg_embed_time():.2f} ms",
                    f"Per layer time: {job.timing_stats.output_times.get_avg_layer_time():.2f} ms",
                    f"Decode speed: {decode_speed:.2f} Tok/s"
                ])

            entries.append(entry)

        lines.extend(make_window_text(entries, self.selected_job_idx, 20))
        
        if self.num_jobs == 0:
            lines.extend(["No Active Jobs..."])

        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrow U/D: Scroll Jobs"])