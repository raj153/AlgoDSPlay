using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    348. Design Tic-Tac-Toe
    https://leetcode.com/problems/design-tic-tac-toe/description/

    */
    public class TicTacToe
    {
        /*
        Approach 1: Optimized Brute Force(OBF)

        Time Complexity: O(n), as for every move we are iterating over n cells 4 times to check for each of the column, row, diagonal row, and anti-diagonal. 
                        This gives us time complexity of O(4⋅n) which is equivalent to O(n).
        Space Complexity: O(n^2), as we are using 2-dimensional array board of size n * n.

        */
        public class TicTacToeOBF
        {
            private int[][] board;
            private int size;

            public TicTacToeOBF(int size)
            {
                board = new int[size][];
                for (int i = 0; i < size; i++)
                {
                    board[i] = new int[size];
                }
                this.size = size;
            }

            public int Move(int row, int col, int player)
            {
                board[row][col] = player;
                // check if the player wins
                if ((CheckRow(row, player)) ||
                    (CheckColumn(col, player)) ||
                    (row == col && CheckDiagonal(player)) ||
                    (col == size - row - 1 && CheckAntiDiagonal(player)))
                {
                    return player;
                }
                // No one wins
                return 0;
            }

            private bool CheckDiagonal(int player)
            {
                for (int row = 0; row < size; row++)
                {
                    if (board[row][row] != player)
                    {
                        return false;
                    }
                }
                return true;
            }

            private bool CheckAntiDiagonal(int player)
            {
                for (int row = 0; row < size; row++)
                {
                    if (board[row][size - row - 1] != player)
                    {
                        return false;
                    }
                }
                return true;
            }

            private bool CheckColumn(int col, int player)
            {
                for (int row = 0; row < size; row++)
                {
                    if (board[row][col] != player)
                    {
                        return false;
                    }
                }
                return true;
            }

            private bool CheckRow(int row, int player)
            {
                for (int col = 0; col < size; col++)
                {
                    if (board[row][col] != player)
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        /*
        Approach 2: Optimised Approach

        Let, n be the length of string s.
        Time Complexity: O(1) because for every move, we mark a particular row, column, diagonal, and anti-diagonal in constant time.

        Space Complexity: O(n) because we use arrays rows and cols of size n. The variables diagonal and antiDiagonal use constant extra space.


        */
        public class TicTacToeOptimal
        {
            private int[] rowCounts;
            private int[] columnCounts;
            private int diagonalCount;
            private int antiDiagonalCount;

            public TicTacToeOptimal(int boardSize)
            {
                rowCounts = new int[boardSize];
                columnCounts = new int[boardSize];
            }

            public int Move(int row, int column, int player)
            {
                int currentPlayer = (player == 1) ? 1 : -1;
                // update currentPlayer in rows and cols arrays
                rowCounts[row] += currentPlayer;
                columnCounts[column] += currentPlayer;
                // update diagonal
                if (row == column)
                {
                    diagonalCount += currentPlayer;
                }
                // update anti diagonal
                if (column == (columnCounts.Length - row - 1))
                {
                    antiDiagonalCount += currentPlayer;
                }
                int boardSize = rowCounts.Length;
                // check if the current player wins
                if (Math.Abs(rowCounts[row]) == boardSize ||
                    Math.Abs(columnCounts[column]) == boardSize ||
                    Math.Abs(diagonalCount) == boardSize ||
                    Math.Abs(antiDiagonalCount) == boardSize)
                {
                    return player;
                }
                // No one wins
                return 0;
            }
        }

    }
    /**
 * Your TicTacToe object will be instantiated and called as such:
 * TicTacToe obj = new TicTacToe(n);
 * int param_1 = obj.Move(row,col,player);
 */
}