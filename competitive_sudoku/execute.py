#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import os
from pathlib import Path
import subprocess
import tempfile


def execute_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )
        output = result.stdout or result.stderr
    except Exception as e:
        output = str(e)
    return output.strip()


def solve_sudoku(solve_sudoku_path: str, board_text: str, options: str='') -> str:
    """
    Execute the solve_sudoku program.
    @param solve_sudoku_path: The location of the solve_sudoku executable.
    @param board_text: A string representation of a sudoku board.
    @param options: Additional command line options.
    @return: The output of solve_sudoku.
    """
    if not os.path.exists(solve_sudoku_path):
        raise RuntimeError(f'No oracle found at location "{solve_sudoku_path}"')
    filename = tempfile.NamedTemporaryFile(prefix='solve_sudoku_').name
    Path(filename).write_text(board_text)
    command = f'{solve_sudoku_path} {filename} {options}'
    return execute_command(command)
