#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI HYBRID: Combines A1 strengths with smart constraint propagation,
    using save/load to persist learned move patterns across turns.
    """

    def __init__(self):
        super().__init__()
        self.move_history = {}  # Track successful moves across turns

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Hybrid strategy: 
        1. Load history of good moves from previous turns
        2. Try forced moves first
        3. Use A1-style alpha-beta with enhanced move ordering
        """
        # Load learned patterns from previous turns
        self.move_history = self.load() or {}
        
        N = game_state.board.N
        
        # Step 1: Try forced moves (naked singles)
        forced_moves = self.find_forced_moves(game_state)
        if forced_moves:
            best_move = forced_moves[0]
            self.propose_move(best_move)
            return
        
        # Step 2: Get legal moves
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return
        
        # Step 3: Enhanced move ordering (A1 + history)
        legal_moves = self.order_moves_hybrid(game_state, legal_moves)
        
        # Propose initial move
        best_move = legal_moves[0]
        self.propose_move(best_move)
        
        # Step 4: Iterative deepening alpha-beta
        for depth in range(1, 100):
            try:
                move, score = self.alpha_beta_search(game_state, depth, legal_moves)
                
                if move:
                    best_move = move
                    self.propose_move(best_move)
                    # Update history with good moves
                    self.update_move_history(move, game_state, score)
            except:
                break
        
        # Save learned patterns for next turn
        self.save(self.move_history)

    def find_forced_moves(self, game_state: GameState):
        """Find naked singles (forced moves)."""
        N = game_state.board.N
        forced_moves = []
        allowed_squares = game_state.player_squares()
        
        for i in range(N):
            for j in range(N):
                square = (i, j)
                
                if game_state.board.get(square) != SudokuBoard.empty:
                    continue
                if square not in allowed_squares:
                    continue
                
                valid_values = []
                for value in range(1, N + 1):
                    if TabooMove(square, value) not in game_state.taboo_moves:
                        if self.is_valid_move(game_state.board, i, j, value):
                            valid_values.append(value)
                
                if len(valid_values) == 1:
                    forced_moves.append(Move(square, valid_values[0]))
        
        return forced_moves

    def update_move_history(self, move: Move, game_state: GameState, score: float):
        """Record successful moves in history for future learning."""
        key = (move.square, move.value)
        if key not in self.move_history:
            self.move_history[key] = {"wins": 0, "score_sum": 0.0}
        
        self.move_history[key]["wins"] += 1
        self.move_history[key]["score_sum"] += score

    def order_moves_hybrid(self, game_state: GameState, moves):
        """
        Smart move ordering combining:
        - A1 region completion heuristic
        - Learned patterns from move history
        - Positional strategy (central bias)
        """
        def move_priority(move):
            score = 0
            
            # Factor 1: Historical success (if we've seen this move before)
            key = (move.square, move.value)
            if key in self.move_history:
                hist = self.move_history[key]
                avg_score = hist["score_sum"] / hist["wins"]
                score += avg_score * 5
            
            # Factor 2: Region completion (A1 style)
            regions = self.count_completed_regions(game_state.board, move)
            score += regions * 100
            
            # Factor 3: Creates forced moves for us
            board_copy = self.copy_board(game_state.board)
            board_copy.put(move.square, move.value)
            temp_state = GameState(board_copy, game_state.taboo_moves[:], game_state.scores[:])
            future_forced = len(self.find_forced_moves(temp_state))
            score += future_forced * 50
            
            # Factor 4: Position (prefer central)
            N = game_state.board.N
            row, col = move.square
            center_distance = abs(row - N//2) + abs(col - N//2)
            score -= center_distance * 0.5
            
            return -score
        
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
        """Alpha-beta pruning."""
        if depth == 0:
            return self.evaluate(game_state)
        
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return self.evaluate(game_state)
        
        legal_moves = self.order_moves_hybrid(game_state, legal_moves)
        
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
        """Evaluation function."""
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
