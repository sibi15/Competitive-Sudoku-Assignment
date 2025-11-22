#!/usr/bin/env python3

#  (C) Copyright Wieger Wesselink 2023. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import argparse
import multiprocessing
from pathlib import Path
from simulate_game import play_game


# Prints 1 instead of 1.0
def print_score(x: float) -> str:
    return '0' if x == 0 else str(x).rstrip('0').rstrip('.')


# Play a match between player and opponent.
def play_match(player: str, opponent: str, count: int, board_file: str, calculation_time: float, verbose=False, warmup=False) -> None:
    player_score = 0.0
    opponent_score = 0.0
    result_lines = []

    for i in range(1, count+1):
        print(f'Playing game {i}')
        player_starts = i % 2 == 1
        first = player if player_starts else opponent
        second = opponent if player_starts else player
        result = play_game(board_file, first, second, calculation_time, verbose, warmup and i == 1)

        result_line = f'{first} - {second} {print_score(result[0])}-{print_score(result[1])}\n'
        result_lines.append(result_line)
        print(result_line)

        if player_starts:
            player_score += result[0]
            opponent_score += result[1]
        else:
            player_score += result[1]
            opponent_score += result[0]

    result_line = f'Match result: {player} - {opponent} {print_score(player_score)}-{print_score(opponent_score)}'
    result_lines.append(result_line)
    print(result_line)

    output_file = f'{player}-{opponent}-board={Path(board_file).stem}-time={calculation_time}-match-result.txt'
    Path(output_file).write_text('\n'.join(result_lines))


def main():
    multiprocessing.set_start_method('fork')
    cmdline_parser = argparse.ArgumentParser(description='Play a match between two sudoku players.')
    cmdline_parser.add_argument('first', help="The module name of the first player's SudokuAI class (default: random_player)", default='random_player', nargs='?')
    cmdline_parser.add_argument('second', help="The module name of the second player's SudokuAI class (default: random_player)", default='random_player', nargs='?')
    cmdline_parser.add_argument('--count', type=int, default=6, help='The number of games (default: 6)')
    cmdline_parser.add_argument('--board', type=str, default='boards/empty-2x2.txt', help='The text file containing the start position (default: boards/empty-2x2.txt)')
    cmdline_parser.add_argument('--time', type=float, default=3.0, help="The time (in seconds) for computing a move (default: 3.0)")
    cmdline_parser.add_argument('--verbose', help="Give verbose output", action="store_true")
    cmdline_parser.add_argument('--warm-up', help='Let the engines play a move before the start of the game', action='store_true')
    args = cmdline_parser.parse_args()

    play_match(args.first, args.second, args.count, args.board, args.time, args.verbose, args.warm_up)


if __name__ == '__main__':
    main()
