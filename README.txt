Competitive sudoku
==================

This archive contains template code for a competitive sudoku assignment.
The goal of the assignment is to build an AI for a competitive sudoku player.

Contents
--------

- The script 'simulate_game.py' is used for running a competitive sudoku game.
- The script 'play_match.py' is used for running a match between two players.
- The folder 'bin' contains a sudoku solver that is used by simulate_game.py.
- The folder 'boards' contains files with starting positions for a game.
- The folder 'competitive_sudoku' is a python module with basic functionality
  needed for running a sudoku game.
- The folders 'greedy_player', 'naive_player', 'random_player' and
  'random_save_player' are four python modules with predefined sudoku AI's.
  All four of them play random moves.

  *) The naive_player does not check for duplicate entries in a region.
  *) The random_player checks for duplicate entries in a region.
  *) The greedy_player checks for duplicate entries in a region, and it
     does a 1 ply deep search to maximize the reward of a move.
  *) The random_save_player is a duplicate of random_player but using the save
     functionalities as defined in the SudokuAI base class

  Note that 'greedy_player', 'random_player' and 'random_save_player' make use of a sudoku
  solver,   which is expected to be found in the folder `bin/solve_sudoku`. In the assignments,
  it is not allowed to use this solver.

  On Windows and macOS, the first time a Python process is started may take more than
  a second. Due to this a time-out may occur on the first move if the calculation time
  is smaller than this. A command line parameter `--warm-up` has been added to deal with
  this problem.

Requirements
------------
Python 3.10 or higher is required to run the code. No additional python packages
need to be installed.

Running simulate_game.py and play_game.py
-----------------------------------------
Some examples of running the script are:

  simulate_game.py -h (print usage information)

  simulate_game.py --check
  (check if the solver works;
   it should give output "The sudoku_solve program works.")

  simulate_game.py
  (this will play a game between two random players on a board with 2x2 regions)

  simulate_game.py --first=random_player --second=greedy_player --board=boards/empty-3x3.txt --time=1.0
  (play a game between the random and the greedy player,
   starting on an empty board with 3x3 regions, and with 1 second per move)

  play_match.py --first=random_player --second=greedy_player --board=boards/empty-3x3.txt --time=1.0 --count=5
  (play a match of 5 games between the random and the greedy player)

File format
-----------
The file format for sudoku boards is as follows. A board with regions of size
m x n (with m the number of rows and n the number of columns) is stored as the fields 'rows' and 'columns' containing the numbers m and n, followed by a field 'board' containing the (m * n) * m * n values of the squares of the
board. Empty squares are printed as a dot '.'. When a square is non-empty, a '+' or '-' is appended to the value to indicate whether it is occupied by the first or second player, respectively.

Optionally, this may be followed by a field 'moves' containing the move history of the game represented in this file; a field 'taboo-moves' specifying moves that were already declared taboo, and a field 'scores' containing the scores of the two players. 
Note: No input check is performed to see whether the specified board is the result of a legal game, so take some care when constructing boards like these yourself.

Below an example is given of a board with 2x3 regions; note that this could never be the result of a legal game history, since allowed squares are not respected.

rows = 2
columns = 2
board =
   .   1+   .   2-
   2-   .   3-   .
   4-   2-   .   .
   1+   .   2+   4+
moves = [(0,1) -> 1, (2,0) -> 4, (3,2) -> 2, (0,3) -> 2, (3,3) -> 1, (1,0) -> 2, (3,0) -> 3, (1,2) -> 3, (3,0) -> 1, (3,3) -> 4, (2,1) -> 2]
taboo-moves = [(3,3) -> 1, (3,0) -> 3]
scores = [0, 0]


Assignment code organization and constraints
--------------------------------------------
Every team is assigned a number and every assignment has a code. Let's use '42'
and 'A1' as examples; replace them as appropriate. Then the module name should
be 'team42_A1', which is used as the folder name. The recommended way of doing
this is to copy the folder 'naive_player' to the folder 'team42_A1'.

The team should create their own AI in this module. This is done replacing the
'compute_best_move' method of the 'SudokuAI' class in the 'sudokuai.py' file
with your own implementation. This may involve adding other methods to the
class, further functions to the module (in the 'sudokuai.py' file or further
files).

Some constraints:
- The code must be written in python.
- The code must be single threaded.
- All code must be located inside the module folder of the team.
- If data files are used (for example data of a network) they must also be
  located inside the module folder of the team. See for example this page for
  an explanation on how to do that:

    https://dev.to/bowmanjd/easily-load-non-python-data-files-from-a-python-package-2e8g

- A requirement of a submission is that it should be possible to run it using
  simulate_game.py, without any modifications to this script, or to the code in
  the 'competitive_sudoku' folder. Test this!
- Transferring knowledge across moves is only possible through the save and load
  utility provided through the base class. This allows you to save any variable
  into a pickle file (.pkl) and load it back into the next move.
  Note that loading large amounts of data is costly.

Using python modules
--------------------
If a command prompt is opened in the root folder of the archive, then the
'simulate_game.py' script should work out of the box. For other usages, it may
be needed to add this root folder to the module search path, e.g. by adding this
folder to the PYTHONPATH environment variable. See

  https://docs.python.org/3/tutorial/modules.html

for an explanation about modules.
