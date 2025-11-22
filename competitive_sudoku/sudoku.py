#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
import io
import random
from typing import List, Tuple, Union, Any, Optional, Iterator

# A square consists of a row and column index. Both are zero-based.
Square = Tuple[int, int]


class SudokuSettings(object):
    print_ascii_states: bool = False  # Print game states in ascii format


class Move(object):
    """A Move is a tuple (square, value) that represents the action board.put(square, value) for a given
    sudoku configuration board."""

    def __init__(self, square: Square, value: int):
        """
        Constructs a move.
        @param square: A square with coordinates in the range [0, ..., N)
        @param value: A value in the range [1, ..., N]
        """
        self.square = square
        self.value = value

    def __str__(self):
        row, col = self.square
        return f'({row},{col}) -> {self.value}'

    def __eq__(self, other):
        return (self.square, self.value) == (other.square, other.value)


class TabooMove(Move):
    """A TabooMove is a Move that was flagged as illegal by the sudoku oracle. In other words, the execution of such a
    move would cause the sudoku to become unsolvable.
    """

    """
    Constructs a taboo move.
    @param square: A square with coordinates in the range [0, ..., N)
    @param value: A value in the range [1, ..., N]
    """
    def __init__(self, square: Square, value: int):
        super().__init__(square, value)


class SudokuBoard(object):
    """
    A simple board class for Sudoku. It supports arbitrary rectangular regions.
    """

    empty = 0  # Empty squares contain the value SudokuBoard.empty

    def __init__(self, m: int = 3, n: int = 3):
        """
        Constructs an empty Sudoku with regions of size m x n.
        @param m: The number of rows in a region.
        @param n: The number of columns in a region.
        """
        N = m * n
        self.m = m
        self.n = n
        self.N = N     # N = m * n, numbers are in the range [1, ..., N]
        self.squares = [SudokuBoard.empty] * (N * N)  # The N*N squares of the board

    def square2index(self, square: Square) -> int:
        """
        Converts row/column coordinates to the corresponding index in the board array.
        @param square: A square with coordinates in the range [0, ..., N)
        @return: The corresponding index k in the board array
        """
        i, j = square
        N = self.N
        return N * i + j

    def index2square(self, k: int) -> Square:
        """
        Converts an index in the board array to the corresponding row/column coordinates.
        @param k: A value in the range [0, ..., N * N)
        @return: The corresponding row/column coordinates
        """
        N = self.N
        i = k // N
        j = k % N
        return i, j

    def put(self, square: Square, value: int) -> None:
        """
        Puts a value on a square.
        @param square: A square with coordinates in the range [0, ..., N)
        @param value: A value in the range [1, ..., N]
        """
        k = self.square2index(square)
        self.squares[k] = value

    def get(self, square: Square) -> int:
        """
        Gets the value of the given square.
        @param square: A square with coordinates in the range [0, ..., N)
        @return: The value of the square.
        """
        k = self.square2index(square)
        return self.squares[k]

    def region_width(self):
        """
        Gets the number of columns in a region.
        @return: The number of columns in a region.
        """
        return self.n

    def region_height(self):
        """
        Gets the number of rows in a region.
        @return: The number of rows in a region.
        """
        return self.m

    def board_width(self):
        """
        Gets the number of columns of the board.
        @return: The number of columns of the board.
        """
        return self.N

    def board_height(self):
        """
        Gets the number of rows of the board.
        @return: The number of rows of the board.
        """
        return self.N

    def __str__(self) -> str:
        """
        Prints the board in a simple textual format. The first line contains the values m and n. Then the contents of
        the rows are printed as space separated lists, where a dot '.' is used to represent an empty square.
        @return: The generated string.
        """
        return print_sudoku_board(self)


# written by Gennaro Gala
def pretty_print_sudoku_board(board: SudokuBoard, gamestate = None) -> str:
    import io

    m = board.m
    n = board.n
    N = board.N
    out = io.StringIO()

    def print_square(square: Square):
        value = board.get(square)
        s = ' -' if value == 0 else f'{value:2}'
        
        if gamestate == None:
            return s + ' '
        if square in gamestate.occupied_squares1:
            return s + '+'
        elif square in gamestate.occupied_squares2:
            return s + '-'
        else:
            return s + ' '

    for i in range(N):

        # open the grid
        if i == 0:
            out.write('  ')
            for j in range(N):
                out.write(f'    {j}  ')
            out.write('\n')
            for j in range(N):
                if j % n != 0:
                    out.write('╤══════')
                elif j != 0:
                    out.write('╦══════')
                else:
                    out.write('   ╔══════')
            out.write('╗\n')

        # separate regions horizontally
        if i % m == 0 and i != 0:
            for j in range(N):
                if j % n != 0:
                    out.write('╪══════')
                elif j != 0:
                    out.write('╬══════')
                else:
                    out.write('   ╠══════')
            out.write('║\n')

        # plot values
        out.write(f'{i:2} ')
        for j in range(N):
            square = i, j
            symbol = print_square(square)
            if j % n != 0:
                out.write(f'│ {symbol}  ')
            else:
                out.write(f'║ {symbol}  ')
            if len(symbol) < 2:
                out.write(' ')
        out.write('║\n')

        # close the grid
        if i == N - 1:
            for j in range(N):
                if j % n != 0:
                    out.write('╧══════')
                elif j != 0:
                    out.write('╩══════')
                else:
                    out.write('   ╚══════')
            out.write('╝\n')

    return out.getvalue()


def print_sudoku_board(board: SudokuBoard) -> str:
    """
    Prints the board in a simple textual format. The first line contains the values m and n. Then the contents of
    the rows are printed as space separated lists, where a dot '.' is used to represent an empty square.
    @return: The generated string.
    """
    m = board.m
    n = board.n
    N = board.N
    out = io.StringIO()

    def print_square(square: Square):
        value = board.get(square)
        s = '   .' if value == 0 else f'{value:>4}'
        out.write(s)

    out.write(f'{m} {n}\n')
    for i in range(N):
        for j in range(N):
            square = i, j
            print_square(square)
        out.write('\n')
    return out.getvalue()


def parse_sudoku_board(text: str) -> SudokuBoard:
    """
    Loads a sudoku board from a string, in the same format as used by the SudokuBoard.__str__ function.
    @param text: A string representation of a sudoku board.
    @return: The generated Sudoku board.
    """
    words = text.split()
    if len(words) < 2:
        raise RuntimeError('The string does not contain a sudoku board')
    m = int(words[0])
    n = int(words[1])
    N = m * n
    if len(words) != N*N + 2:
        raise RuntimeError('The number of squares in the sudoku is incorrect.')
    result = SudokuBoard(m, n)
    N = result.N
    for k in range(N * N):
        s = words[k + 2]
        if s != '.':
            value = int(s)
            result.squares[k] = value
    return result


class GameState(object):
    def __init__(self,
                 initial_board: SudokuBoard = None,
                 board: SudokuBoard = None,
                 taboo_moves: List[TabooMove] = None,
                 moves: List[Union[Move, TabooMove]] = None,
                 scores: List[int] = None,
                 current_player: int = 1,
                 allowed_squares1: Optional[List[Square]] = None,
                 allowed_squares2: Optional[List[Square]] = None,
                 occupied_squares1: Optional[List[Square]] = None,
                 occupied_squares2: Optional[List[Square]] = None,
                ):
        """
        @param initial_board: A sudoku board. It contains the start position of a game.
        @param board: A sudoku board. It contains the current position of a game.
        @param taboo_moves: A list of taboo moves. Moves in this list cannot be played.
        @param moves: The history of a sudoku game, starting in initial_board. The history includes taboo moves.
        @param scores: The cumulative rewards of the first and the second player.
        @param current_player: The current player (1 or 2).
        @param allowed_squares1: The squares where player1 is always allowed to play (None if all squares are allowed).
        @param allowed_squares2: The squares where player2 is always allowed to play (None if all squares are allowed).
        @param occupied_squares1: The squares occupied by player1.
        @param occupied_squares2: The squares occupied by player2.
        """
        if taboo_moves is None:
            taboo_moves = []
        if moves is None:
            moves = []
        if scores is None:
            scores = [0, 0]
        if initial_board is None and board is None:
            initial_board = SudokuBoard(2, 2)
            board = SudokuBoard(2, 2)
        elif board is None:
            board = copy.deepcopy(initial_board)
            for move in moves:
                board.put(move.square, move.value)
        elif initial_board is None:
            initial_board = copy.deepcopy(board)
            for move in moves:
                initial_board.put(move.square, SudokuBoard.empty)
        self.initial_board = initial_board
        self.board = board
        self.taboo_moves = taboo_moves
        self.moves = moves
        self.scores = scores
        self.current_player = current_player
        self.allowed_squares1 = allowed_squares1
        self.allowed_squares2 = allowed_squares2
        self.occupied_squares1 = occupied_squares1
        self.occupied_squares2 = occupied_squares2

    def is_classic_game(self):
        """
        Returns True if the game is classic, i.e. all squares are allowed.
        """
        return self.allowed_squares1 is None and self.allowed_squares2 is None

    def occupied_squares(self):
        """
        Returns the occupied squares of the current player.
        """
        return self.occupied_squares1 if self.current_player == 1 else self.occupied_squares2

    def player_squares(self) -> Optional[List[Square]]:
        """
        Returns the squares where the current player can play, or None if all squares are allowed.
        """
        allowed_squares = self.allowed_squares1 if self.current_player == 1 else self.allowed_squares2
        occupied_squares = self.occupied_squares1 if self.current_player == 1 else self.occupied_squares2
        N = self.board.N

        if allowed_squares is None:
            return None

        def is_empty(square: Square) -> bool:
            return self.board.get(square) == SudokuBoard.empty

        def neighbors(square: Square) -> Iterator[Square]:
            row, col = square
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < N and 0 <= c < N:
                        yield r, c

        # add the empty allowed squares
        result = [s for s in allowed_squares if is_empty(s)]

        # add the empty neighbors to result
        for s1 in occupied_squares:
            for s2 in neighbors(s1):
                if is_empty(s2):
                    result.append(s2)

        # remove duplicates
        return sorted(list(set(result)))

    def __str__(self):
        return print_game_state(self)


def parse_properties(text: str) -> dict[str, str]:
    """
    Parses a string containing key-value pairs.
    Lines should have the format `key = value`. A value can be a multiline string. In that
    case subsequent lines should start with a space. Lines starting with '#' are ignored.
    @param text: A string.
    @return: A dictionary of key-value pairs.
    """
    result = {}
    key = None
    value = []

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("#") or not line.strip():
            continue
        elif line.startswith(" "):
            value.append(line.lstrip())
        else:
            if key:
                result[key] = "\n".join(value).strip()
            words = line.split('=', 1)
            if len(words) not in [1, 2]:
                raise ValueError(f"Unexpected line '{line}'")
            key = words[0].strip()
            value = [words[1].strip()] if len(words) > 1 else []

    if key:
        result[key] = "\n".join(value).strip()

    return result


def print_game_state(game_state: GameState) -> str:
    """
    Saves a game state as a string containing key-value pairs.
    @param game_state: A game state.
    """
    out = io.StringIO()

    is_classic_game = game_state.is_classic_game()

    board = game_state.board
    m = board.m
    n = board.n
    N = board.N

    def print_square(square: Square):
        value = board.get(square)
        if is_classic_game:
            s = '   .' if value == 0 else f'{value:>4}'
        else:
            occupied = '+' if square in game_state.occupied_squares1 else '-'
            s = '     .' if value == 0 else f' {value:>4}{occupied}'
        out.write(s)

    out.write(f'rows = {m}\n')
    out.write(f'columns = {n}\n')
    out.write(f'board =\n')
    for i in range(N):
        for j in range(N):
            square = i, j
            print_square(square)
        out.write('\n')
    taboo_moves = [f'{move}' for move in game_state.taboo_moves]
    out.write(f'taboo-moves = [{", ".join(taboo_moves)}]\n')
    moves = [f'{move}' for move in game_state.moves]
    out.write(f'moves = [{", ".join(moves)}]\n')
    out.write(f'scores = {game_state.scores}\n')
    out.write(f'current-player = {game_state.current_player}\n')
    if not game_state.is_classic_game():
        allowed_squares1 = [f'({square[0]},{square[1]})' for square in game_state.allowed_squares1]
        out.write(f'allowed-squares1 = {", ".join(allowed_squares1)}\n')
        allowed_squares2 = [f'({square[0]},{square[1]})' for square in game_state.allowed_squares2]
        out.write(f'allowed-squares2 = {", ".join(allowed_squares2)}\n')
        occupied_squares1 = [f'({square[0]},{square[1]})' for square in game_state.occupied_squares1]
        out.write(f'occupied-squares1 = {", ".join(occupied_squares1)}\n')
        occupied_squares2 = [f'({square[0]},{square[1]})' for square in game_state.occupied_squares2]
        out.write(f'occupied-squares2 = {", ".join(occupied_squares2)}\n')
    return out.getvalue()


def pretty_print_game_state(game_state: GameState) -> str:
    out = io.StringIO()
    out.write(pretty_print_sudoku_board(game_state.board, game_state))
    out.write(f'Score: {game_state.scores[0]} - {game_state.scores[1]}\n')
    out.write(f'Current player: player{game_state.current_player}\n')
    if not game_state.is_classic_game():
        out.write(f'Player1 allowed squares: {"None (all squares are allowed)" if game_state.allowed_squares1 is None else game_state.allowed_squares1}\n')
        out.write(f'Player2 allowed squares: {"None (all squares are allowed)" if game_state.allowed_squares2 is None else game_state.allowed_squares2}\n')
        out.write(f'Player1 occupied squares: {list(sorted(game_state.occupied_squares1))}\n')
        out.write(f'Player2 occupied squares: {list(sorted(game_state.occupied_squares2))}\n')
    return out.getvalue()


def generate_random_tuples(N):
    """
    Generates 2N random and distinct tuples of (i, j) where 0 <= i, j < N.

    Args:
        N: A positive integer.

    Returns:
        A list of 2N distinct tuples of (i, j) where 0 <= i, j < N.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer")

    unique_tuples = set()

    # Fill the set with random tuples until we have 2N elements
    while len(unique_tuples) < 2 * N:
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        unique_tuples.add((i, j))

    # Convert the set of tuples to a list
    return list(unique_tuples)


def allowed_squares(board: SudokuBoard, playmode: str) -> Tuple[List[Square], List[Square]]:
    """
    Generates allowed squares for player1 and player2.
    @param board: A SudokuBoard object.
    @param playmode: The playing playmode (classic, rows, random)
    """
    N = board.N
    if playmode == 'classic':
        return [], []
    elif playmode == 'rows':
        return [(0, j) for j in range(N)], [(N-1, j) for j in range(N)]
    elif playmode == 'border':
        top = [(0, j) for j in range(N)]
        bottom = [(N-1, j) for j in range(N)]
        right = [(i, 0) for i in range(1, N-1)]
        left = [(i, N-1) for i in range(1, N-1)]
        border = top + bottom + right + left
        return border, border
    elif playmode == 'random':
        squares = generate_random_tuples(N)
        return squares[:N], squares[N:]


def parse_game_state(text: str, playmode: str) -> GameState:
    """
    Loads a game state from a string containing key-value pairs.
    @param text: A string representation of a game state.
    """
    properties = parse_properties(text)
    is_classic_game = playmode == 'classic'

    def remove_special_characters(text):
        for char in '[](),->':
            text = text.replace(char, ' ')
        return text

    def parse_board(key: str, m: int, n: int) -> Tuple[Optional[SudokuBoard], Optional[List[Square]], Optional[List[Square]]]:
        text = properties.get(key)
        if text is None:
            return None, None, None
        if is_classic_game:
            return parse_sudoku_board(f'{m} {n}\n{text}'), None, None
        occupied_squares1 = []
        occupied_squares2 = []
        N = m * n
        words = text.strip().split()
        if len(words) != N * N:
            raise ValueError('The number of squares in the sudoku board is incorrect.')
        board = SudokuBoard(m, n)

        for k, word in enumerate(words):
            if word != '.':
                value, occupied = word[:-1], word[-1]
                value = int(value)
                board.squares[k] = value
                if occupied == '+':
                    occupied_squares1.append(board.index2square(k))
                else:
                    occupied_squares2.append(board.index2square(k))

        return board, occupied_squares1, occupied_squares2

    def parse_moves(key: str, move_class) -> Union[List[Move], List[TabooMove]]:
        text = properties.get(key)
        if text is None:
            return []
        result = []
        items = remove_special_characters(text).strip().split()
        items = [int(item) for item in items]
        assert len(items) % 3 == 0, "The number of elements in the a move list must be divisible by 3."
        for index in range(0, len(items), 3):
            i, j, value = items[index], items[index + 1], items[index + 2]
            result.append(move_class((i, j), value))
        return result

    def parse_scores(key: str) -> Optional[List[int]]:
        text = properties.get(key)
        if text is None:
            return None
        items = remove_special_characters(text).strip().split()
        items = [int(item) for item in items]
        assert len(items) == 2, "The number of elements in the scores list must be 2."
        return items

    def parse_squares(key: str) -> Optional[List[Square]]:
        text = properties.get(key)
        if text is None:
            return None
        result = []
        items = remove_special_characters(text).strip().split()
        assert len(items) % 2 == 0, "The number of elements in the a square list must be divisible by 2."
        items = [int(item) for item in items]
        for index in range(0, len(items), 2):
            i, j = items[index], items[index + 1]
            result.append((i, j))
        return result

    m = int(properties['rows'])
    n = int(properties['columns'])

    moves = parse_moves('moves', Move)
    taboo_moves = parse_moves('taboo-moves', TabooMove)
    scores = parse_scores('scores')
    current_player = int(properties.get('current-player', '1'))

    if is_classic_game:
        initial_board = None
        board = parse_sudoku_board(f'{m} {n}\n' + properties['board'])
        occupied_squares1 = None
        occupied_squares2 = None
        allowed_squares1 = None
        allowed_squares2 = None
    else:
        initial_board, _, _ = parse_board('initial-board', m, n)
        board, occupied_squares1, occupied_squares2 = parse_board('board', m, n)
        allowed_squares1 = parse_squares('allowed-squares1')
        allowed_squares2 = parse_squares('allowed-squares2')
        if allowed_squares1 is None or allowed_squares2 is None:
            allowed_squares1, allowed_squares2 = allowed_squares(board, playmode)

    return GameState(initial_board=initial_board,
                     board=board,
                     taboo_moves=taboo_moves,
                     moves=moves,
                     scores=scores,
                     current_player=current_player,
                     allowed_squares1=allowed_squares1,
                     allowed_squares2=allowed_squares2,
                     occupied_squares1=occupied_squares1,
                     occupied_squares2=occupied_squares2
                    )
