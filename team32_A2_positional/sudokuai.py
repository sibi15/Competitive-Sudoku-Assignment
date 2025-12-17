#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI with GAME ABSTRACTION LEVELS.
    Uses different strategies for early, mid, and late game phases:
    - Early game (0-30% filled): Focus on territorial control
    - Mid game (30-70% filled): Balance territory and scoring
    - Late game (70%+ filled): Maximize immediate points
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute best move based on game phase.
        """
        # Determine game phase
        phase = self.get_game_phase(game_state)
        
        # Get legal moves
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return
        
        # Order moves according to game phase
        if phase == "early":
            legal_moves = self.order_moves_early_game(game_state, legal_moves)
        elif phase == "mid":
            legal_moves = self.order_moves_mid_game(game_state, legal_moves)
        else:  # late
            legal_moves = self.order_moves_late_game(game_state, legal_moves)
        
        # Propose initial move
        self.propose_move(legal_moves[0])
        
        # Iterative deepening with phase-aware search
        depth_limit = 2 if phase == "early" else (3 if phase == "mid" else 4)
        
        for depth in range(1, depth_limit + 100):
            try:
                move, score = self.alpha_beta_search(game_state, depth, legal_moves, phase)
                if move:
                    self.propose_move(move)
            except:
                break

    def get_game_phase(self, game_state: GameState):
        """
        Determine game phase based on board fill percentage.
        """
        N = game_state.board.N
        total_cells = N * N
        filled_cells = sum(1 for i in range(N) for j in range(N) 
                          if game_state.board.get((i, j)) != SudokuBoard.empty)
        fill_percentage = filled_cells / total_cells
        
        if fill_percentage < 0.3:
            return "early"
        elif fill_percentage < 0.7:
            return "mid"
        else:
            return "late"

    def order_moves_early_game(self, game_state: GameState, moves):
        """
        Early game strategy: Claim territory and control key positions.
        Prioritize moves that block opponent's options and control regions.
        """
        def move_priority(move):
            score = 0
            
            # Claim boundary cells (between our and opponent territory)
            board_copy = self.copy_board(game_state.board)
            board_copy.put(move.square, move.value)
            
            # Count how many empty cells around this move
            empty_neighbors = self.count_empty_neighbors(board_copy, move.square)
            score += empty_neighbors * 10  # Prefer central positions
            
            # Slight preference for region completion, but not primary
            regions = self.count_completed_regions(game_state.board, move)
            score += regions * 5
            
            # Prefer boundary positions
            boundary_bonus = self.boundary_score(game_state.board, move.square)
            score += boundary_bonus * 8
            
            return -score
        
        return sorted(moves, key=move_priority)

    def order_moves_mid_game(self, game_state: GameState, moves):
        """
        Mid game strategy: Balance territorial expansion with scoring opportunities.
        """
        def move_priority(move):
            score = 0
            
            # Main factor: Complete regions
            regions = self.count_completed_regions(game_state.board, move)
            score += regions * 100
            
            # Secondary: Control moves
            empty_neighbors = self.count_empty_neighbors(self.copy_board(game_state.board), move.square)
            score += empty_neighbors * 5
            
            # Position bonus
            N = game_state.board.N
            row, col = move.square
            center_distance = abs(row - N//2) + abs(col - N//2)
            score -= center_distance * 0.5
            
            return -score
        
        return sorted(moves, key=move_priority)

    def order_moves_late_game(self, game_state: GameState, moves):
        """
        Late game strategy: Maximize immediate points.
        Prioritize moves that complete regions.
        """
        def move_priority(move):
            score = 0
            
            # Dominant factor: Score as many points as possible
            regions = self.count_completed_regions(game_state.board, move)
            if regions == 0:
                score = 0
            else:
                points_map = {1: 1, 2: 3, 3: 7}
                score = points_map.get(regions, 0) * 1000
            
            # Tiebreaker: prefer moves in regions we're close to completing
            almost_complete = self.is_almost_complete_region(game_state.board, move.square)
            score += almost_complete * 10
            
            return -score
        
        return sorted(moves, key=move_priority)

    def boundary_score(self, board: SudokuBoard, square):
        """
        Score a position based on how much it influences boundary between territories.
        Higher score = better boundary position.
        """
        N = board.N
        row, col = square
        boundary_value = 0
        
        # Check neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = row + di, col + dj
            if 0 <= ni < N and 0 <= nj < N:
                if board.get((ni, nj)) == SudokuBoard.empty:
                    boundary_value += 1
        
        return boundary_value

    def count_empty_neighbors(self, board: SudokuBoard, square):
        """Count empty cells adjacent to this square."""
        N = board.N
        row, col = square
        count = 0
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = row + di, col + dj
                if 0 <= ni < N and 0 <= nj < N:
                    if board.get((ni, nj)) == SudokuBoard.empty:
                        count += 1
        
        return count

    def is_almost_complete_region(self, board: SudokuBoard, square):
        """Check if this square is in a region that's almost complete."""
        N = board.N
        n = board.n
        m = board.m
        row, col = square
        
        # Check row
        empty_in_row = sum(1 for j in range(N) if board.get((row, j)) == SudokuBoard.empty)
        if empty_in_row <= 2:
            return 1
        
        # Check column
        empty_in_col = sum(1 for i in range(N) if board.get((i, col)) == SudokuBoard.empty)
        if empty_in_col <= 2:
            return 1
        
        # Check block
        block_row = (row // m) * m
        block_col = (col // n) * n
        empty_in_block = sum(1 for i in range(block_row, block_row + m)
                            for j in range(block_col, block_col + n)
                            if board.get((i, j)) == SudokuBoard.empty)
        if empty_in_block <= 2:
            return 1
        
        return 0

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
                        if TabooMove(square, value) not in game_state.taboo_moves:
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

    def alpha_beta_search(self, game_state: GameState, max_depth, legal_moves=None, phase="mid"):
        """Perform phase-aware alpha-beta search."""
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        if legal_moves is None:
            legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return None, 0
        
        for move in legal_moves:
            new_state = self.apply_move(game_state, move, maximizing=True)
            score = self.alpha_beta(new_state, max_depth - 1, alpha, beta, False, phase)
            
            if score > alpha:
                alpha = score
                best_move = move
        
        return best_move, alpha

    def alpha_beta(self, game_state: GameState, depth, alpha, beta, maximizing_player, phase="mid"):
        """Alpha-beta pruning with phase awareness."""
        if depth == 0:
            return self.evaluate(game_state, phase)
        
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return self.evaluate(game_state, phase)
        
        # Order based on phase
        if phase == "early":
            legal_moves = self.order_moves_early_game(game_state, legal_moves)
        elif phase == "mid":
            legal_moves = self.order_moves_mid_game(game_state, legal_moves)
        else:
            legal_moves = self.order_moves_late_game(game_state, legal_moves)
        
        if maximizing_player:
            value = float('-inf')
            for move in legal_moves:
                new_state = self.apply_move(game_state, move, maximizing=True)
                value = max(value, self.alpha_beta(new_state, depth - 1, alpha, beta, False, phase))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                new_state = self.apply_move(game_state, move, maximizing=False)
                value = min(value, self.alpha_beta(new_state, depth - 1, alpha, beta, True, phase))
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

    def evaluate(self, game_state: GameState, phase="mid"):
        """Phase-aware evaluation function."""
        score_diff = game_state.scores[0] - game_state.scores[1]
        our_moves = len(self.get_legal_moves(game_state))
        almost_complete_bonus = self.count_almost_complete_regions(game_state)
        potential_points = self.count_potential_points(game_state)
        
        if phase == "early":
            # Early game: prioritize position and mobility
            evaluation = (
                score_diff * 5.0 +
                our_moves * 2.0 +
                almost_complete_bonus * 1.0 +
                potential_points * 1.0
            )
        elif phase == "late":
            # Late game: prioritize points
            evaluation = (
                score_diff * 20.0 +
                potential_points * 5.0 +
                almost_complete_bonus * 2.0 +
                our_moves * 0.1
            )
        else:  # mid
            # Mid game: balance
            evaluation = (
                score_diff * 10.0 +
                almost_complete_bonus * 2.0 +
                potential_points * 3.0 +
                our_moves * 0.5
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
