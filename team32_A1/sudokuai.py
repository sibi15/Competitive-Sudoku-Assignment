#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration using
    alpha-beta pruning with minimax search.
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
        
        # Propose a random initial move immediately to ensure we have something
        best_move = random.choice(legal_moves)
        self.propose_move(best_move)
        
        # Iterative deepening: search with increasing depth
        for depth in range(1, 100):  # Arbitrary max depth
            try:
                # Search for best move at current depth
                move, score = self.alpha_beta_search(game_state, depth)
                
                if move:
                    best_move = move
                    self.propose_move(best_move)
                    
            except Exception as e:
                # If we run out of time or encounter an error, we still have a valid move
                break
    
    def get_legal_moves(self, game_state: GameState):
        """
        Generate all legal moves for the current player.
        A move is legal if:
        1. The cell is empty
        2. The cell is allowed (in starting region or adjacent to occupied cell)
        3. The value respects sudoku constraint C0
        4. The move is not taboo
        """
        N = game_state.board.N
        legal_moves = []
        
        allowed_squares = game_state.player_squares()
        
        for i in range(N):
            for j in range(N):
                square = (i, j)
                # Check if cell is empty and allowed
                if game_state.board.get(square) == SudokuBoard.empty and square in allowed_squares:
                    # Try each possible value
                    for value in range(1, N + 1):
                        # Check if move is not taboo and respects C0
                        if not TabooMove(square, value) in game_state.taboo_moves:
                            if self.is_valid_move(game_state.board, i, j, value):
                                legal_moves.append(Move(square, value))
        
        return legal_moves
    
    def is_valid_move(self, board: SudokuBoard, row, col, value):
        """
        Check if placing a value at (row, col) respects the sudoku constraint C0.
        Returns True if the value doesn't conflict with existing values in the
        same row, column, or block.
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
    
    def alpha_beta_search(self, game_state: GameState, max_depth):
        """
        Perform alpha-beta pruning search to find the best move.
        Returns (best_move, best_score).
        """
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return None, 0
        
        for move in legal_moves:
            # Make the move
            new_state = self.apply_move(game_state, move, maximizing=True)
            
            # Evaluate the move
            score = self.alpha_beta(new_state, max_depth - 1, alpha, beta, False)
            
            # Update best move
            if score > alpha:
                alpha = score
                best_move = move
        
        return best_move, alpha
    
    def alpha_beta(self, game_state: GameState, depth, alpha, beta, maximizing_player):
        """
        Alpha-beta pruning algorithm.
        
        Args:
            game_state: Current game state
            depth: Remaining search depth
            alpha: Best value for maximizing player
            beta: Best value for minimizing player
            maximizing_player: True if current player is maximizing
        
        Returns:
            The evaluation score for this state
        """
        # Terminal condition: depth reached or game over
        if depth == 0:
            return self.evaluate(game_state)
        
        legal_moves = self.get_legal_moves(game_state)
        
        # If no legal moves, evaluate current state
        if not legal_moves:
            return self.evaluate(game_state)
        
        if maximizing_player:
            value = float('-inf')
            for move in legal_moves:
                new_state = self.apply_move(game_state, move, maximizing=True)
                value = max(value, self.alpha_beta(new_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # Beta cutoff
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                new_state = self.apply_move(game_state, move, maximizing=False)
                value = min(value, self.alpha_beta(new_state, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cutoff
            return value
    
    def apply_move(self, game_state: GameState, move: Move, maximizing: bool):
        """
        Apply a move to create a new game state.
        Note: This is a simplified simulation and doesn't call the oracle.
        """
        # Create a copy of the board
        new_board = SudokuBoard(game_state.board.m, game_state.board.n)
        N = game_state.board.N
        
        # Copy all values
        for i in range(N):
            for j in range(N):
                square = (i, j)
                value = game_state.board.get(square)
                if value != SudokuBoard.empty:
                    new_board.put(square, value)
        
        # Apply the move
        new_board.put(move.square, move.value)
        
        # Calculate new scores
        new_scores = self.calculate_scores(game_state, move, maximizing)
        
        # Create new game state (simplified - doesn't update all fields perfectly)
        new_state = GameState(
            new_board,
            game_state.taboo_moves[:],  # Copy taboo moves list
            new_scores
        )
        
        return new_state
    
    def calculate_scores(self, game_state: GameState, move: Move, is_current_player: bool):
        """
        Calculate new scores after a move.
        Awards points based on number of regions completed.
        """
        # Make a copy of the scores
        current_scores = [game_state.scores[0], game_state.scores[1]]
        
        # Count completed regions
        regions_completed = self.count_completed_regions(game_state.board, move)
        
        # Award points based on regions completed
        points_map = {0: 0, 1: 1, 2: 3, 3: 7}
        points = points_map.get(regions_completed, 0)
        
        # Add points to appropriate player
        if is_current_player:
            current_scores[0] += points
        else:
            current_scores[1] += points
        
        return current_scores
    
    def count_completed_regions(self, board: SudokuBoard, move: Move):
        """
        Count how many regions (row, column, block) are completed by this move.
        A region is completed if it has no empty cells after the move.
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
        Evaluation function that assigns a numerical score to a game state.
        Positive scores favor the current player (maximizing player).
        
        This uses score difference as the primary metric.
        """
        # Simple evaluation: difference in scores
        score_diff = game_state.scores[0] - game_state.scores[1]
        
        return score_diff


