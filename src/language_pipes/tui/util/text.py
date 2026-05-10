from typing import List


def make_footer_text(items: List[str]) -> str:
    total_len = 0
    for item in items:
        total_len += len(item)
    spacing = int((76 - total_len) / (len(items) - 1))
    
    footer = ""
    for i, item in enumerate(items):
        footer += item
        if i < len(items) - 1:
            footer += (" " * spacing)
    return footer

def make_selectable_text(text: str, selected: bool) -> str:
    l_cursor = "|>" if selected else "  "
    r_cursor = "<|" if selected else "  "
    return f"{l_cursor} {text} {r_cursor}"