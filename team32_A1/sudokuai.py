#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration using
    alpha-beta pruning with minimax search and improved evaluation.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes the best move using iterative deepening with alpha-beta pruning.
        This is an anytime algorithm that proposes increasingly better moves.
        """
        N = game_state.board.N
        
        # Generate all legal moves
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            # No legal moves available
            return
        
        # Sort moves by potential (moves that complete regions first)
        legal_moves = self.order_moves(game_state, legal_moves)
        
        # Propose the best looking move immediately
        best_move = legal_moves[0]
        self.propose_move(best_move)
        
        # Iterative deepening: search with increasing depth
        for depth in range(1, 100):
            try:
                # Search for best move at current depth
                move, score = self.alpha_beta_search(game_state, depth, legal_moves)
                
                if move:
                    best_move = move
                    self.propose_move(best_move)
                    
            except Exception as e:
                # If we run out of time or encounter an error, we still have a valid move
                break
    
    def order_moves(self, game_state: GameState, moves):
        """
        Order moves to examine promising moves first (improves pruning).
        Priority: moves that complete regions > moves in strategic positions.
        """
        def move_priority(move):
            # Count regions this move would complete
            regions = self.count_completed_regions(game_state.board, move)
            
            # Prioritize moves that score points
            if regions > 0:
                return -regions  # Negative because we sort ascending
            
            # Secondary priority: central positions
            N = game_state.board.N
            row, col = move.square
            center_distance = abs(row - N//2) + abs(col - N//2)
            
            return center_distance
        
        return sorted(moves, key=move_priority)
    
    def get_legal_moves(self, game_state: GameState):
        """
        Generate all legal moves for the current player.
        """
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
        """
        Check if placing a value at (row, col) respects the sudoku constraint C0.
        """
        N = board.N
        n = board.n
        m = board.m
        
        # Check row
        for j in range(N):
            if board.get((row, j)) == value:
                return False
        
        # Check column
        for i in range(N):
            if board.get((i, col)) == value:
                return False
        
        # Check block
        block_row = (row // m) * m
        block_col = (col // n) * n
        
        for i in range(block_row, block_row + m):
            for j in range(block_col, block_col + n):
                if board.get((i, j)) == value:
                    return False
        
        return True
    
    def alpha_beta_search(self, game_state: GameState, max_depth, legal_moves=None):
        """
        Perform alpha-beta pruning search to find the best move.
        """
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
        """
        Alpha-beta pruning algorithm with improved evaluation.
        """
        if depth == 0:
            return self.evaluate(game_state)
        
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return self.evaluate(game_state)
        
        # Order moves for better pruning
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
        """
        Apply a move to create a new game state.
        """
        new_board = SudokuBoard(game_state.board.m, game_state.board.n)
        N = game_state.board.N
        
        for i in range(N):
            for j in range(N):
                square = (i, j)
                value = game_state.board.get(square)
                if value != SudokuBoard.empty:
                    new_board.put(square, value)
        
        new_board.put(move.square, move.value)
        new_scores = self.calculate_scores(game_state, move, maximizing)
        
        new_state = GameState(
            new_board,
            game_state.taboo_moves[:],
            new_scores
        )
        
        return new_state
    
    def calculate_scores(self, game_state: GameState, move: Move, is_current_player: bool):
        """
        Calculate new scores after a move.
        """
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
        """
        Count how many regions (row, column, block) are completed by this move.
        """
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
        """
        IMPROVED evaluation function that considers multiple strategic factors.
        """
        # Base score: actual point difference
        score_diff = game_state.scores[0] - game_state.scores[1]
        
        # Factor 1: Mobility (number of legal moves available)
        our_moves = len(self.get_legal_moves(game_state))
        
        # Factor 2: Almost-complete regions (regions with only 1 empty cell)
        almost_complete_bonus = self.count_almost_complete_regions(game_state)
        
        # Factor 3: Potential immediate points
        potential_points = self.count_potential_points(game_state)
        
        # Combine factors with weights
        evaluation = (
            score_diff * 10.0 +              # Actual score is most important
            almost_complete_bonus * 2.0 +     # Value almost-complete regions
            potential_points * 3.0 +          # Value potential to score
            our_moves * 0.1                   # Small bonus for mobility
        )
        
        return evaluation
    
    def count_almost_complete_regions(self, game_state: GameState):
        """
        Count regions (rows, columns, blocks) that have only 1 empty cell.
        These are valuable because they're about to score points.
        """
        board = game_state.board
        N = board.N
        n = board.n
        m = board.m
        count = 0
        
        # Check rows
        for i in range(N):
            empty_count = sum(1 for j in range(N) if board.get((i, j)) == SudokuBoard.empty)
            if empty_count == 1:
                count += 1
        
        # Check columns
        for j in range(N):
            empty_count = sum(1 for i in range(N) if board.get((i, j)) == SudokuBoard.empty)
            if empty_count == 1:
                count += 1
        
        # Check blocks
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
        """
        Count how many points we could score with our available moves.
        """
        legal_moves = self.get_legal_moves(game_state)
        total_potential = 0
        
        for move in legal_moves:
            regions = self.count_completed_regions(game_state.board, move)
            if regions > 0:
                points_map = {1: 1, 2: 3, 3: 7}
                total_potential += points_map.get(regions, 0)
        
        return total_potential

