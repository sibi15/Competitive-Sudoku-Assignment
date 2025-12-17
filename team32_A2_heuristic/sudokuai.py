#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI with HEURISTIC constraint propagation and forced move detection.
    Uses alpha-beta pruning but prioritizes:
    1. Forced moves (naked singles)
    2. Moves that complete regions
    3. Moves that don't close opponent's options
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes best move using constraint propagation to find forced moves first,
        then searches with alpha-beta pruning.
        """
        N = game_state.board.N
        
        # Try to find forced moves (naked singles)
        forced_moves = self.find_forced_moves(game_state)
        
        if forced_moves:
            # Play forced move immediately - guaranteed to be correct
            best_move = forced_moves[0]
            self.propose_move(best_move)
            return
        
        # If no forced moves, fall back to intelligent search
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return
        
        # Order moves strategically
        legal_moves = self.order_moves(game_state, legal_moves)
        self.propose_move(legal_moves[0])
        
        # Iterative deepening with alpha-beta
        for depth in range(1, 100):
            try:
                move, score = self.alpha_beta_search(game_state, depth, legal_moves)
                if move:
                    self.propose_move(move)
            except:
                break

    def find_forced_moves(self, game_state: GameState):
        """
        Find "forced moves" - cells in our territory where only one value is legal.
        These moves are guaranteed to be part of the final solution.
        """
        N = game_state.board.N
        forced_moves = []
        allowed_squares = game_state.player_squares()
        
        for i in range(N):
            for j in range(N):
                square = (i, j)
                
                # Only check empty cells in our territory
                if game_state.board.get(square) != SudokuBoard.empty:
                    continue
                if square not in allowed_squares:
                    continue
                
                # Find valid values for this cell
                valid_values = []
                for value in range(1, N + 1):
                    if TabooMove(square, value) not in game_state.taboo_moves:
                        if self.is_valid_move(game_state.board, i, j, value):
                            valid_values.append(value)
                
                # Naked single: only one valid value
                if len(valid_values) == 1:
                    forced_moves.append(Move(square, valid_values[0]))
        
        return forced_moves

    def find_hidden_singles(self, game_state: GameState):
        """
        Find "hidden singles" - values that can only go in one cell within a region.
        More expensive computation, only used when we have time.
        """
        board = game_state.board
        N = board.N
        n = board.n
        m = board.m
        allowed_squares = game_state.player_squares()
        hidden_singles = []
        
        # Check for values that fit only one position per row
        for i in range(N):
            for value in range(1, N + 1):
                # Check if value already exists in row
                if any(board.get((i, j)) == value for j in range(N)):
                    continue
                
                possible_positions = []
                for j in range(N):
                    square = (i, j)
                    if (board.get(square) == SudokuBoard.empty and 
                        square in allowed_squares and
                        TabooMove(square, value) not in game_state.taboo_moves and
                        self.is_valid_move(board, i, j, value)):
                        possible_positions.append(square)
                
                # Hidden single: value fits only one position
                if len(possible_positions) == 1:
                    hidden_singles.append(Move(possible_positions[0], value))
        
        return hidden_singles

    def order_moves(self, game_state: GameState, moves):
        """
        Order moves to prioritize:
        1. Moves that complete regions (immediate points)
        2. Moves that create forced moves for us
        3. Moves in central positions
        """
        def move_priority(move):
            score = 0
            
            # Highest priority: regions we can complete
            regions = self.count_completed_regions(game_state.board, move)
            score += regions * 1000
            
            # Medium priority: moves that limit opponent
            board_copy = self.copy_board(game_state.board)
            board_copy.put(move.square, move.value)
            opponent_mobility = self.estimate_opponent_moves(board_copy, game_state)
            score -= opponent_mobility * 10  # Penalize if opponent gets many moves
            
            # Low priority: central positions
            N = game_state.board.N
            row, col = move.square
            center_distance = abs(row - N//2) + abs(col - N//2)
            score -= center_distance * 0.1
            
            return -score  # Negative for sorting
        
        return sorted(moves, key=move_priority)

    def get_legal_moves(self, game_state: GameState):
        """Generate all legal moves."""
        N = game_state.board.N
        legal_moves = []
        allowed_squares = game_state.player_squares()
        
        for i in range(N):
            for j in range(N):
                square = (i, j)
                if game_state.board.get(square) == SudokuBoard.empty and square in allowed_squares:
                    for value in range(1, N + 1):
                        if not TabooMove(square, value) in game_state.taboo_moves:
                            if self.is_valid_move(game_state.board, i, j, value):
                                legal_moves.append(Move(square, value))
        
        return legal_moves

    def is_valid_move(self, board: SudokuBoard, row, col, value):
        """Check sudoku constraint C0."""
        N = board.N
        n = board.n
        m = board.m
        
        for j in range(N):
            if board.get((row, j)) == value:
                return False
        
        for i in range(N):
            if board.get((i, col)) == value:
                return False
        
        block_row = (row // m) * m
        block_col = (col // n) * n
        
        for i in range(block_row, block_row + m):
            for j in range(block_col, block_col + n):
                if board.get((i, j)) == value:
                    return False
        
        return True

    def copy_board(self, board: SudokuBoard):
        """Create a copy of the board."""
        new_board = SudokuBoard(board.m, board.n)
        N = board.N
        for i in range(N):
            for j in range(N):
                value = board.get((i, j))
                if value != SudokuBoard.empty:
                    new_board.put((i, j), value)
        return new_board

    def estimate_opponent_moves(self, board: SudokuBoard, game_state: GameState):
        """Rough estimate of how many moves opponent can make."""
        N = board.N
        count = 0
        opponent_squares = set()
        
        # Get opponent's allowed squares (complement of our squares)
        all_squares = set((i, j) for i in range(N) for j in range(N))
        our_squares = game_state.player_squares()
        if our_squares:
            opponent_squares = all_squares - our_squares
        
        for i in range(N):
            for j in range(N):
                if board.get((i, j)) == SudokuBoard.empty and (i, j) in opponent_squares:
                    count += 1
        
        return count

    def alpha_beta_search(self, game_state: GameState, max_depth, legal_moves=None):
        """Perform alpha-beta search."""
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        if legal_moves is None:
            legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return None, 0
        
        for move in legal_moves:
            new_state = self.apply_move(game_state, move, maximizing=True)
            score = self.alpha_beta(new_state, max_depth - 1, alpha, beta, False)
            
            if score > alpha:
                alpha = score
                best_move = move
        
        return best_move, alpha

    def alpha_beta(self, game_state: GameState, depth, alpha, beta, maximizing_player):
        """Alpha-beta pruning with heuristic move ordering."""
        if depth == 0:
            return self.evaluate(game_state)
        
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return self.evaluate(game_state)
        
        legal_moves = self.order_moves(game_state, legal_moves)
        
        if maximizing_player:
            value = float('-inf')
            for move in legal_moves:
                new_state = self.apply_move(game_state, move, maximizing=True)
                value = max(value, self.alpha_beta(new_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                new_state = self.apply_move(game_state, move, maximizing=False)
                value = min(value, self.alpha_beta(new_state, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def apply_move(self, game_state: GameState, move: Move, maximizing: bool):
        """Apply a move to create new game state."""
        new_board = self.copy_board(game_state.board)
        new_board.put(move.square, move.value)
        new_scores = self.calculate_scores(game_state, move, maximizing)
        
        return GameState(
            new_board,
            game_state.taboo_moves[:],
            new_scores
        )

    def calculate_scores(self, game_state: GameState, move: Move, is_current_player: bool):
        """Calculate scores after a move."""
        current_scores = [game_state.scores[0], game_state.scores[1]]
        regions_completed = self.count_completed_regions(game_state.board, move)
        points_map = {0: 0, 1: 1, 2: 3, 3: 7}
        points = points_map.get(regions_completed, 0)
        
        if is_current_player:
            current_scores[0] += points
        else:
            current_scores[1] += points
        
        return current_scores

    def count_completed_regions(self, board: SudokuBoard, move: Move):
        """Count regions completed by this move."""
        N = board.N
        n = board.n
        m = board.m
        row, col = move.square
        completed = 0
        
        # Check row
        row_complete = True
        for j in range(N):
            if (row, j) == move.square:
                continue
            if board.get((row, j)) == SudokuBoard.empty:
                row_complete = False
                break
        if row_complete:
            completed += 1
        
        # Check column
        col_complete = True
        for i in range(N):
            if (i, col) == move.square:
                continue
            if board.get((i, col)) == SudokuBoard.empty:
                col_complete = False
                break
        if col_complete:
            completed += 1
        
        # Check block
        block_row = (row // m) * m
        block_col = (col // n) * n
        block_complete = True
        
        for i in range(block_row, block_row + m):
            for j in range(block_col, block_col + n):
                if (i, j) == move.square:
                    continue
                if board.get((i, j)) == SudokuBoard.empty:
                    block_complete = False
                    break
            if not block_complete:
                break
        
        if block_complete:
            completed += 1
        
        return completed

    def evaluate(self, game_state: GameState):
        """Enhanced evaluation function."""
        score_diff = game_state.scores[0] - game_state.scores[1]
        our_moves = len(self.get_legal_moves(game_state))
        almost_complete_bonus = self.count_almost_complete_regions(game_state)
        potential_points = self.count_potential_points(game_state)
        
        evaluation = (
            score_diff * 10.0 +
            almost_complete_bonus * 2.0 +
            potential_points * 3.0 +
            our_moves * 0.1
        )
        
        return evaluation

    def count_almost_complete_regions(self, game_state: GameState):
        """Count regions with only 1 empty cell."""
        board = game_state.board
        N = board.N
        n = board.n
        m = board.m
        count = 0
        
        for i in range(N):
            empty_count = sum(1 for j in range(N) if board.get((i, j)) == SudokuBoard.empty)
            if empty_count == 1:
                count += 1
        
        for j in range(N):
            empty_count = sum(1 for i in range(N) if board.get((i, j)) == SudokuBoard.empty)
            if empty_count == 1:
                count += 1
        
        for block_i in range(n):
            for block_j in range(m):
                block_row = block_i * m
                block_col = block_j * n
                empty_count = 0
                for i in range(block_row, block_row + m):
                    for j in range(block_col, block_col + n):
                        if board.get((i, j)) == SudokuBoard.empty:
                            empty_count += 1
                if empty_count == 1:
                    count += 1
        
        return count

    def count_potential_points(self, game_state: GameState):
        """Count potential points from available moves."""
        legal_moves = self.get_legal_moves(game_state)
        total_potential = 0
        
        for move in legal_moves:
            regions = self.count_completed_regions(game_state.board, move)
            if regions > 0:
                points_map = {1: 1, 2: 3, 3: 7}
                total_potential += points_map.get(regions, 0)
        
        return total_potential
