import io
import zstandard
import chess.pgn

FILENAME = "lichess_db_standard_rated_2017-04.pgn.zst"
total_game = 0
bullet_games = 0
bullet_game_eval = 0
any_game_eval = 0
with open(FILENAME, "rb") as f:
    reader = zstandard.ZstdDecompressor().stream_reader(f)
    text = io.TextIOWrapper(reader, encoding="utf-8")
    while True:
        game = chess.pgn.read_game(text)
        if game is None:
            break
        total_game += 1
        if total_game == 20000:
            break
        time_control = game.headers["TimeControl"]
        if time_control == "60+0":
            bullet_games += 1
            for node in game.mainline():
                if node.eval() is not None:
                    bullet_game_eval += 1
                    break
        for node in game.mainline():
            if node.eval() is not None:
                any_game_eval += 1
                break
print("Total Games: ", total_game)
print("Total Bullet Games:", bullet_games)
print("Total Bullet Games with Eval:", bullet_game_eval)
print("Any game with Eval:",any_game_eval)