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

def make_window_text(entries: List[List[str]], selected_idx: int, max_height: int) -> List[str]:
    total = sum(len(e) for e in entries)
    if total <= max_height:
        out: List[str] = []
        for e in entries:
            out.extend(e)
        return out

    # Reserve a line for the truncation indicator on each clipped side.
    budget = max_height
    start = selected_idx
    end = selected_idx
    used = len(entries[selected_idx])

    def above_indicator() -> int:
        return 1 if start > 0 else 0

    def below_indicator() -> int:
        return 1 if end < len(entries) - 1 else 0

    progress = True
    while progress:
        progress = False
        if end + 1 < len(entries):
            cost = len(entries[end + 1])
            reserved = above_indicator() + 1  # new below indicator stays 1 if not last
            if end + 1 == len(entries) - 1:
                reserved = above_indicator()
            if used + cost + reserved <= budget:
                end += 1
                used += cost
                progress = True
        if start > 0:
            cost = len(entries[start - 1])
            reserved = below_indicator() + 1
            if start - 1 == 0:
                reserved = below_indicator()
            if used + cost + reserved <= budget:
                start -= 1
                used += cost
                progress = True

    out: List[str] = []
    if start > 0:
        out.append("           ^")
    else:
        out.append("")
    for i in range(start, end + 1):
        out.extend(entries[i])
    if end < len(entries) - 1:
        out.append("           V")

    if len(out) < max_height:
        out.append("")

    return out