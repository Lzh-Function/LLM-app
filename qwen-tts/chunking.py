"""Bounded sentence chunks that preserve the complete manuscript."""

import re


def chunks(text: str, limit=160):
    """Keep punctuation and every character, bounded even without sentence stops."""
    pieces = re.findall(r"[^。！？!?\n]+[。！？!?\n]*|[。！？!?\n]+", text)
    result = []
    pending = ""
    for piece in pieces:
        while len(piece) > limit:
            if pending:
                result.append(pending)
                pending = ""
            end = piece.rfind(" ", limit // 2, limit + 1)
            end = end + 1 if end >= 0 else limit
            result.append(piece[:end])
            piece = piece[end:]
        if pending and len(pending) + len(piece) > limit:
            result.append(pending)
            pending = ""
        pending += piece
    if pending:
        result.append(pending)
    return result
