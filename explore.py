import io
import zstandard

FILENAME = "lichess_db_standard_rated_2017-04.pgn.zst"

with open(FILENAME, "rb") as f:
    reader = zstandard.ZstdDecompressor().stream_reader(f)
    text = io.TextIOWrapper(reader, encoding="utf-8")

    for i, line in enumerate(text):
        print(line, end="")
        if i >= 49:
            break