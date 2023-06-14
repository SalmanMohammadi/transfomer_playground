from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Union
import torch
from torch import Tensor
import pgn

BOARD_CHARS = list("abcdefgh")


def move_to_int(move: str):
    """
    Converts a string-formatted move to integer format e.g. A5 to 4

    Args:
        move: string representation of the move, where
            the first character is A-H, the second is 1-8
    """
    return BOARD_CHARS.index(move[0].lower()) * 8 + (int(move[1]) - 1)


def load_othello_game(game_path: str) -> Tuple[List[int], List[int]]:
    """
    Parses a list of moves and corresponding game result scores given a path to a `.pgm` format.

    Args:
        game_path: path to a `.pgm` game(s) of Othello.
    Returns:
        A tuple of board_moves, result where:
            board_moves is a list of integer encoded moves on the board, 0 being A1, and
            63 being H8, with black always playing first in Othello.

            result is a list with two integers describing the score at the end of the game,
            i.e. [16, 48] indicates that black scored 16, and white scored 48.
    """
    board_moves, results = [], []
    with open(game_path, "r") as f:
        games = pgn.loads(f.read())
        for game in games:
            valid_moves = filter(lambda x: not x[0].isdigit(), game.moves)
            valid_moves = filter(lambda x: x[0].lower() in BOARD_CHARS, valid_moves)
            int_moves = list(map(move_to_int, valid_moves))
            if len(int_moves) == 0:
                continue
            board_moves.append(int_moves)
            results.append(game.result.split("-"))
    return board_moves, results


def get_othello_championship_games(data_root: Path):
    game_paths = list(map(str, data_root.glob("*.pgn")))
    game_sequences, game_results = [], []
    for game_path in game_paths:
        game_sequence, game_result = load_othello_game(game_path)
        game_sequences.extend(game_sequence)
        game_results.extend(game_result)
    return game_sequences, game_results


class OthelloDataset(Dataset):
    def __init__(self, data_root: Union[str, Path], seq_len: int):
        data_root = Path(data_root)
        center_tiles = [27, 28, 35, 36]
        vocabulary = list(range(64))
        [vocabulary.remove(x) for x in center_tiles]
        self.vocabulary = sorted(vocabulary + [-100])
        self.vocabulary_size = len(self.vocabulary)
        self.seq_len = seq_len
        self.block_size = seq_len - 1
        self.board_to_vocabulary = {word: i for i, word in enumerate(self.vocabulary)}
        self.vocabulary_to_board = {i: word for i, word in enumerate(self.vocabulary)}
        self.game_sequences, self.game_results = get_othello_championship_games(data_root)

    def __len__(self) -> int:
        return len(self.game_sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        block = self.game_sequences[idx]
        if len(block) != self.seq_len:
            block += [
                -100,
            ] * (self.seq_len - len(block))
        block = [self.board_to_vocabulary[s] for s in block]
        return torch.tensor(block[:-1], dtype=torch.long), torch.tensor(block[1:], dtype=torch.long)
