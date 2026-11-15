import io
import zstandard
import chess.pgn
import random
random.seed(42)
def material(board,color):
    knight_count = len(board.pieces(chess.KNIGHT,color))
    bishop_count = len(board.pieces(chess.BISHOP,color))
    rook_count = len(board.pieces(chess.ROOK,color))
    queen_count = len(board.pieces(chess.QUEEN,color))
    total = (knight_count * 3 + bishop_count * 3 + rook_count * 5 + queen_count * 9)
    return total
def iter_game_texts(text):
    buf = []
    for line in text:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = []
        buf.append(line)
    if buf:
        yield "".join(buf)
FILENAME = "lichess_db_standard_rated_2017-04.pgn.zst"
import csv

FIELDS = ["game_id", "white_elo", "black_elo", "result", "move_number",
          "white_to_move", "white_clock", "black_clock", "eval_cp", "mate",
          "material_white", "material_black", "phase"]

out = open("C:/Users/Duelz/positions_full.csv", "w", newline="")
writer = csv.DictWriter(out, fieldnames=FIELDS)
writer.writeheader()
games_done = 0
with open(FILENAME, "rb") as f:
    reader = zstandard.ZstdDecompressor().stream_reader(f)
    text = io.TextIOWrapper(reader, encoding="utf-8")
    for game_text in iter_game_texts(text):
        if '[TimeControl "60+0"]' not in game_text:
            continue
        if "%eval" not in game_text:
            continue
        game = chess.pgn.read_game(io.StringIO(game_text))


        white_clock = 60.0
        black_clock = 60.0
        middlegame_positions = []
        endgame_positions = []
        board = game.board()
        for node in game.mainline():
            if node.ply() % 2 == 1:
                white_clock = node.clock()
            else: 
                black_clock = node.clock()
            move_number = (node.ply() +1) //2 
            board.push(node.move)
            material_white = material(board, chess.WHITE)
            material_black = material(board, chess.BLACK)
            if move_number <= 10:
                phase = "opening"
            elif material_white <= 10 and material_black <= 10:
                phase = "endgame"
            else:
                phase = "middlegame"
            if node.eval() is None:
                eval_cp = None
                mate = None
            else:
                score = node.eval().white()
                if score.is_mate():
                    mate = score.mate()
                    eval_cp = None
                else:
                    eval_cp = score.score()
                    mate = None
            if phase == "opening" or (eval_cp is None and mate is None):
                continue
            position = {
                "game_id": game.headers["Site"],
                "white_elo": int(game.headers["WhiteElo"]),
                "black_elo": int(game.headers["BlackElo"]),
                "result": game.headers["Result"],
                "move_number": move_number,
                "white_to_move": board.turn,
                "white_clock": white_clock,
                "black_clock": black_clock,
                "eval_cp": eval_cp,
                "mate": mate,
                "material_white": material_white,
                "material_black": material_black,
                "phase": phase,
            }
            if phase == "middlegame":
                middlegame_positions.append(position)
            else:
                endgame_positions.append(position)
        if middlegame_positions:
            writer.writerow(random.choice(middlegame_positions))
        if endgame_positions:
            writer.writerow(random.choice(endgame_positions))
        games_done += 1
        if games_done % 1000 == 0:                  
            print(games_done)
        
out.close()
        