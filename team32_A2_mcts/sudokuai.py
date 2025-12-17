#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import math
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, game_state: GameState, move: Move = None, parent=None):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score_sum = 0.0
        self.untried_moves = None
    
    def add_child(self, move: Move, game_state: GameState):
        """Add a child node."""
        child = MCTSNode(game_state, move, self)
        self.children.append(child)
        return child
    
    def update(self, score: float):
        """Update node with score from simulation."""
        self.visits += 1
        self.score_sum += score
    
    def ucb1_value(self, c=1.41):
        """Calculate UCB1 value for selection."""
        if self.visits == 0:
            return float('inf')
        return (self.score_sum / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def best_child(self):
        """Select child with highest UCB1 value."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1_value())


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI using Monte Carlo Tree Search (MCTS).
    Uses UCB1 for selection, random simulation for expansion, and score-based back-propagation.
    """

    def __init__(self):
        super().__init__()
        self.mcts_iterations = 1000
        self.simulation_depth = 5

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute best move using MCTS.
        """
        # Start with a quick move proposal
        legal_moves = self.get_legal_moves(game_state)
        
        if not legal_moves:
            return
        
        # Propose a quick move initially
        best_move = legal_moves[0]
        self.propose_move(best_move)
        
        # Run MCTS for several iterations, proposing best move each time
        root = MCTSNode(game_state)
        
        for iteration in range(self.mcts_iterations):
            try:
                # Selection and Expansion
                node = self.select_node(root)
                
                # Simulation
                score = self.simulate(node.game_state)
                
                # Back-propagation
                self.backpropagate(node, score)
                
                # Propose the best move found so far
                best_child = root.best_child()
                if best_child:
                    best_move = best_child.move
                    self.propose_move(best_move)
            except:
                # If we run out of time, the last proposed move is used
                break

    def select_node(self, node: MCTSNode):
        """
        Selection and Expansion phase:
        - Navigate down tree using UCB1
        - Expand with a new random child when reaching a leaf
        """
        while True:
            if node.untried_moves is None:
                # First visit to this node - generate its moves
                node.untried_moves = self.get_legal_moves(node.game_state)
            
            if node.untried_moves:
                # Expansion: add a new child
                move = node.untried_moves.pop(0)
                new_state = self.apply_move(node.game_state, move, maximizing=True)
                return node.add_child(move, new_state)
            else:
                # All children explored, use UCB1 to select
                if not node.children:
                    return node
                node = node.best_child()

    def simulate(self, game_state: GameState):
        """
        Simulation phase: play random moves from current state.
        Return score difference (our score - opponent score).
        """
        current_state = game_state
        is_our_turn = False  # We already made a move to reach this state
        
        for _ in range(self.simulation_depth):
            legal_moves = self.get_legal_moves(current_state)
            
            if not legal_moves:
                break
            
            # Random move selection
            move = random.choice(legal_moves)
            current_state = self.apply_move(current_state, move, maximizing=not is_our_turn)
            is_our_turn = not is_our_turn
        
        # Score difference favors us
        score_diff = current_state.scores[0] - current_state.scores[1]
        return score_diff

    def backpropagate(self, node: MCTSNode, score: float):
        """
        Back-propagation phase: update all parent nodes with the score.
        """
        while node is not None:
            node.update(score)
            node = node.parent

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
