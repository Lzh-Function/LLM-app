"""Bound upstream sentence groups so punctuation-free text stays GPU-sized."""
MAX_CHARS = 160


def bounded_chunks(chunks: list[str]) -> list[str]:
    result = []
    for text in chunks:
        while len(text) > MAX_CHARS:
            # Prefer a word boundary for Latin text, otherwise retain every character.
            end = text.rfind(" ", MAX_CHARS // 2, MAX_CHARS + 1)
            end = end + 1 if end != -1 else MAX_CHARS
            result.append(text[:end])
            text = text[end:]
        if text:
            result.append(text)
    return result
