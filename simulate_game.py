#!/usr/bin/env python3

#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import argparse
import copy
import importlib
import multiprocessing
import platform
import re
import time
import os
from pathlib import Path
from typing import Optional, Tuple

from competitive_sudoku.execute import solve_sudoku
from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove, parse_game_state, \
    SudokuSettings, print_game_state, pretty_print_game_state, allowed_squares
from competitive_sudoku.sudokuai import SudokuAI

SUDOKU_SOLVER = 'bin\\solve_sudoku.exe' if platform.system() == 'Windows' else 'bin/solve_sudoku'
#SUDOKU_SOLVER = 'bin\\Windows\\solve_sudoku.exe' if platform.system() == 'Windows' else 'bin/solve_sudoku'

GameResult = Tuple[float, float]


def check_oracle() -> None:
    board_text = '''2 2
       1   2   3   4
       3   4   .   2
       2   1   .   3
       .   .   .   1
    '''
    output = solve_sudoku(SUDOKU_SOLVER, board_text)
    result = 'has a solution' in output
    if result:
        print('The sudoku_solve program works.')
    else:
        print('The sudoku_solve program gives unexpected results.')
        print(output)


def warmup_players(player1: SudokuAI, player2: SudokuAI, calculation_time: float = 2.0) -> None:
    initial_board = SudokuBoard(3, 3)
    game_state = GameState(initial_board, copy.deepcopy(initial_board))
    move_number = 0
    number_of_moves = 2

    with multiprocessing.Manager() as manager:
        # use a lock to protect assignments to best_move
        lock = multiprocessing.Lock()
        player1.lock = lock
        player2.lock = lock

        # use shared variables to store the best move
        player1.best_move = manager.list([0, 0, 0])
        player2.best_move = manager.list([0, 0, 0])

        while move_number < number_of_moves:
            player, player_number = (player1, 1) if len(game_state.moves) % 2 == 0 else (player2, 2)
            player.best_move[0] = 0
            player.best_move[1] = 0
            player.best_move[2] = 0
            try:
                process = multiprocessing.Process(target=player.compute_best_move, args=(game_state,))
                process.start()
                time.sleep(calculation_time)
                lock.acquire()
                process.terminate()
                lock.release()
            except Exception as err:
                print(f'Error: an exception occurred during warm-up:\n{err}')
            move_number = move_number + 1


def simulate_game(game_state: GameState,
                  player1: SudokuAI,
                  player2: SudokuAI,
                  calculation_time: float = 0.5,
                  verbose=True,
                  warmup=False,
                  playmode='rows'
                  ) -> GameResult:
    """
    Simulates a game between two instances of SudokuAI.
    @param game_state: A game state.
    @param player1: The AI of the first player.
    @param player2: The AI of the second player.
    @param calculation_time: The amount of time in seconds for computing the best move.
    @param verbose: Print the positions and the moves.
    @param warmup: Let the engines play a move before the start of the game.
    @return The result of the game.
    """

    def log(text):
        if verbose:
            print(text)

    move_number = 0
    number_of_moves = game_state.board.squares.count(SudokuBoard.empty)

    log('Initial state')
    if SudokuSettings.print_ascii_states:
        log(print_game_state(game_state))
    else:
        log(pretty_print_game_state(game_state))

    if warmup:
        print('-- started warm-up --')
        warmup_players(player1, player2, 2.0)
        print('-- finished warm-up --')

    with multiprocessing.Manager() as manager:
        # use a lock to protect assignments to best_move
        lock = multiprocessing.Lock()
        player1.lock = lock
        player2.lock = lock

        # use shared variables to store the best move
        player1.best_move = manager.list([0, 0, 0])
        player2.best_move = manager.list([0, 0, 0])

        # used to detect if the game has ended
        finished_players = set()

        while move_number < number_of_moves and len(finished_players) < 2:
            player_number = game_state.current_player
            player = player1 if player_number == 1 else player2
            log(f'-----------------------------\nCalculate a move for player {player_number}')
            player_squares = None if playmode == 'classic' else game_state.player_squares()
            if player_squares == []:
                log(f'Player {player_number} cannot move')
                finished_players.add(player_number)
                game_state.current_player = 3 - game_state.current_player
                continue
            else:
                player.best_move[0] = 0
                player.best_move[1] = 0
                player.best_move[2] = 0
                try:
                    process = multiprocessing.Process(target=player.compute_best_move, args=(game_state,))
                    process.start()
                    time.sleep(calculation_time)
                    lock.acquire()
                    process.terminate()
                    lock.release()
                except Exception as err:
                    log(f'Error: an exception occurred:\n{err}')
                i, j, value = player.best_move
                square = (i, j)
                best_move = Move(square, value)
                log(f'Best move: {best_move}')
                player_score = 0
                if (i, j, value) != (0, 0, 0):
                    if TabooMove(square, value) in game_state.taboo_moves:
                        print(f'Error: {best_move} is a taboo move. Player {3-player_number} wins the game.')
                        return (0, 1) if player_number == 1 else (1, 0)
                    board_text = str(game_state.board)
                    options = f'--move "{game_state.board.square2index(square)} {value}"'
                    if player_squares is not None:
                       allowed = ' '.join(str(game_state.board.square2index(square)) for square in player_squares)
                       options += f' --allowed="{allowed}"'
                    output = solve_sudoku(SUDOKU_SOLVER, board_text, options)
                    if 'Invalid move' in output:
                        print(f'Error: {best_move} is not a valid move. Player {3-player_number} wins the game.')
                        return (0, 1) if player_number == 1 else (1, 0)
                    if 'Illegal move' in output:
                        print(f'Error: {best_move} is not a legal move. Player {3-player_number} wins the game.')
                        return (0, 1) if player_number == 1 else (1, 0)
                    if 'has no solution' in output:
                        log(f'The sudoku has no solution after the move {best_move}.')
                        player_score = 0
                        game_state.moves.append(TabooMove(square, value))
                        game_state.taboo_moves.append(TabooMove(square, value))
                    if 'The score is' in output:
                        match = re.search(r'The score is ([-\d]+)', output)
                        if match:
                            player_score = int(match.group(1))
                            game_state.board.put(square, value)
                            game_state.moves.append(best_move)
                            if playmode != 'classic':
                                game_state.occupied_squares().append(square)
                            move_number = move_number + 1
                        else:
                            raise RuntimeError(f'Unexpected output of sudoku solver: "{output}".')
                else:
                    print(f'No move was supplied. Player {3-player_number} wins the game.')
                    return (0, 1) if player_number == 1 else (1, 0)
            game_state.scores[player_number-1] = game_state.scores[player_number-1] + player_score
            game_state.current_player = 3 - game_state.current_player
            log(f'Reward: {player_score}')
            if SudokuSettings.print_ascii_states:
                log(print_game_state(game_state))
            else:
                log(pretty_print_game_state(game_state))
        if not verbose:
            print(f'Score: {game_state.scores[0]} - {game_state.scores[1]}')
        if game_state.scores[0] > game_state.scores[1]:
            print('Player 1 wins the game.')
            return 1, 0
        elif game_state.scores[0] == game_state.scores[1]:
            print('The game ends in a draw.')
            return 0.5, 0.5
        elif game_state.scores[0] < game_state.scores[1]:
            print('Player 2 wins the game.')
            return 0, 1


def play_game(board_file: Optional[str], name1: str, name2: str, calculation_time: float, verbose=True, warmup=False, playmode='rows') -> GameResult:
    """
    Simulates a game between two instances of SudokuAI.
    @param board_file: A text file containing a game state.
    @param name1: The module name of the first player.
    @param name2: The module name of the second player.
    @param calculation_time: The amount of time in seconds for computing the best move.
    @param verbose: Print the positions and the moves.
    @param warmup: Let the engines play a move before the start of the game.
    @param playmode: The playing mode (classic, rows, random)
    """
    
    if board_file:
        text = Path(board_file).read_text()
        game_state = parse_game_state(text, playmode)
    elif playmode == 'classic':
        game_state = GameState()
    else:
        initial_board = SudokuBoard(2, 2)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode)
        game_state = GameState(allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])

    module1 = importlib.import_module(name1 + '.sudokuai')
    module2 = importlib.import_module(name2 + '.sudokuai')
    player1 = module1.SudokuAI()
    player2 = module2.SudokuAI()
    player1.player_number = 1
    player2.player_number = 2
    if name1 in ('random_player', 'greedy_player', 'random_save_player'):
        player1.solve_sudoku_path = SUDOKU_SOLVER
    if name2 in ('random_player', 'greedy_player', 'random_save_player'):
        player2.solve_sudoku_path = SUDOKU_SOLVER

    # clean up pickle files
    if os.path.isfile(os.path.join(os.getcwd(), '-1.pkl')):
        os.remove(os.path.join(os.getcwd(), '-1.pkl'))
    if os.path.isfile(os.path.join(os.getcwd(), '1.pkl')):
        os.remove(os.path.join(os.getcwd(), '1.pkl'))
    if os.path.isfile(os.path.join(os.getcwd(), '2.pkl')):
        os.remove(os.path.join(os.getcwd(), '2.pkl'))

    return simulate_game(game_state, player1, player2, calculation_time=calculation_time, verbose=verbose, warmup=warmup, playmode=playmode)


def main():
    cmdline_parser = argparse.ArgumentParser(description='Script for simulating a competitive sudoku game.')
    cmdline_parser.add_argument('--first', help="the module name of the first player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--second', help="the module name of the second player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--time', help="the time (in seconds) for computing a move (default: 0.5)", type=float, default=0.5)
    cmdline_parser.add_argument('--check', help="check if the solve_sudoku program works", action='store_true')
    cmdline_parser.add_argument('--board', metavar='FILE', type=str, help='a text file containing a game state')
    cmdline_parser.add_argument('--quiet', help='print minimal output', action='store_true')
    cmdline_parser.add_argument('--warm-up', help='let the engines play a move before the start of the game', action='store_true')
    cmdline_parser.add_argument('--playmode', type=str, choices=['classic', 'rows', 'border', 'random'], default='rows', help='Choose the playing mode (classic, rows, border, random). Defaults to rows.')
    cmdline_parser.add_argument('--ascii', help=argparse.SUPPRESS, action='store_true')
    args = cmdline_parser.parse_args()

    SudokuSettings.print_ascii_states = args.ascii

    if args.check:
        check_oracle()
    else:
        play_game(args.board, args.first, args.second, args.time, verbose = not args.quiet, warmup=args.warm_up, playmode=args.playmode)


if __name__ == '__main__':
    main()
