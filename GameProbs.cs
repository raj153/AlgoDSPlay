using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace AlgoDSPlay
{
    public class GameProbs
    {
        //https://www.algoexpert.io/questions/reveal-minesweeper
        public static string[][] RevealMinesweeper(string[][] board, int row, int col)
        {
            //O(w * h) time | O(w * h) space - where w is the width of the board, and h is the height of the board
            if (board[row][col] == MINE)
            {
                board[row][col] = CLOSE;
                return board;
            }
            List<CellLocation> neighbors = GetNeighbors(board, row, col);

            int adjacentMinesCount = 0;
            foreach (var neighbor in neighbors)
            {
                if (board[neighbor.Row][neighbor.Col].Equals(MINE))
                    adjacentMinesCount++;
            }

            if (adjacentMinesCount > 0)
            {
                board[row][col] = adjacentMinesCount.ToString();
            }
            else
            {
                board[row][col] = "0";
                foreach (var neighbor in neighbors)
                {
                    if (board[neighbor.Row][neighbor.Col].Equals(NOMINE))
                        RevealMinesweeper(board, neighbor.Row, neighbor.Col);
                }

            }
            return board;
        }

        private static List<CellLocation> GetNeighbors(string[][] board, int row, int col)
        {
            int[,] directions = new int[8, 2]{
                {0,1},{0,-1}, //Same row - left and right
                {1,-1},{1,0},{1,1}, //next row - bottom-left, down and right and bottom-right diagonal
                {-1,-1},{-1,0},{-1,1} //previous row - top-left, up and top-right diagonal
            };

            List<CellLocation> neighbors = new List<CellLocation>();
            for (int i = 0; i < directions.GetLength(0); i++)
            {
                int newRow = row + directions[i, 0];
                int newCol = col + directions[i, 1];

                if (0 <= newRow && newRow < board.Length && 0 <= newCol && newCol < board[0].Length)
                    neighbors.Add(new CellLocation(newRow, newCol));

            }
            return neighbors;

        }

        public static string MINE = "M";
        public static string CLOSE = "X";
        public static string NOMINE = "H";
        public class CellLocation
        {
            public int Row;
            public int Col;

            public CellLocation(int row, int col)
            {
                this.Row = row;
                this.Col = col;
            }
        }
        //https://www.algoexpert.io/questions/blackjack-probability
        // O(t - s) time | O(t - s) space - where t is the target, and s is the
        // starting hand
        public double BlackjackProbability(int target, int startingHand)
        {
            Dictionary<int, double> memo = new Dictionary<int, double>();
            return Math.Round(
                     calculateProbability(target, startingHand, memo) * 1000f
                   ) /
                   1000f;
        }

        private double calculateProbability(
          int target, int currentHand, Dictionary<int, double> memo
        )
        {
            if (memo.ContainsKey(currentHand))
            {
                return memo[currentHand];
            }
            if (currentHand > target)
            {
                return 1;
            }
            if (currentHand + 4 >= target)
            {
                return 0;
            }

            double totalProbability = 0;
            for (int drawnCard = 1; drawnCard <= 10; drawnCard++)
            {
                totalProbability +=
                  .1 * calculateProbability(target, currentHand + drawnCard, memo);
            }

            memo[currentHand] = totalProbability;
            return totalProbability;
        }
        //https://www.algoexpert.io/questions/best-digits
        // O(n) time | O(n) space - where n is the length of the input string
        public string BestDigits(string number, int numDigits)
        {
            Stack<char> stack = new Stack<char>();

            for (int idx = 0; idx < number.Length; idx++)
            {
                char character = number[idx];
                while (numDigits > 0 && stack.Count > 0 && character > stack.Peek())
                {
                    numDigits--;
                    stack.Pop();
                }
                stack.Push(character);
            }

            while (numDigits > 0)
            {
                numDigits--;
                stack.Pop();
            }

            // build final string from stack
            StringBuilder bestDigitString = new StringBuilder();
            while (stack.Count > 0)
            {
                bestDigitString.Append(stack.Pop());
            }

            var charArray = bestDigitString.ToString().ToCharArray();
            Array.Reverse(charArray);
            return new string(charArray);
        }

        //https://www.algoexpert.io/questions/knight-connection
        // O(n * m) time | O(n * m) space - where n is horizontal distance between
        // the knights and m is the vertical distance between the knights
        public int KnightConnection(int[] knightA, int[] knightB)
        {
            int[,] possibleMoves = new int[8, 2] {
      { -2, 1 },
      { -1, 2 },
      { 1, 2 },
      { 2, 1 },
      { 2, -1 },
      { 1, -2 },
      { -1, -2 },
      { -2, -1 }
    };

            Queue<List<int>> queue = new Queue<List<int>>();
            queue.Enqueue(new List<int> { knightA[0], knightA[1], 0 });
            HashSet<string> visited = new HashSet<string>();
            visited.Add(knightA.ToString());

            while (queue.Count > 0)
            {
                List<int> currentPosition = queue.Dequeue();

                if (currentPosition[0] == knightB[0] && currentPosition[1] == knightB[1])
                {
                    return (int)Math.Ceiling((double)currentPosition[2] / 2);
                }

                for (var i = 0; i < possibleMoves.GetLength(0); i++)
                {
                    List<int> position = new List<int>();
                    position.Add(currentPosition[0] + possibleMoves[i, 0]);
                    position.Add(currentPosition[1] + possibleMoves[i, 1]);
                    string positionString =
                      String.Join(", ", position.ConvertAll<string>(x => x.ToString()));

                    if (!visited.Contains(positionString))
                    {
                        position.Add(currentPosition[2] + 1);
                        queue.Enqueue(position);
                        visited.Add(positionString);
                    }
                }
            }
            return -1;
        }
        //https://www.algoexpert.io/questions/solve-sudoku
        // O(1) time | O(1) space - assuming a 9x9 input board
        public List<List<int>> SolveSudoku(List<List<int>> board)
        {
            solvePartialSudoku(0, 0, board);
            return board;
        }

        public bool solvePartialSudoku(int row, int col, List<List<int>> board)
        {
            int currentRow = row;
            int currentCol = col;

            if (currentCol == board[currentRow].Count)
            {
                currentRow += 1;
                currentCol = 0;
                if (currentRow == board.Count)
                {
                    return true;  // board is completed
                }
            }

            if (board[currentRow][currentCol] == 0)
            {
                return tryDigitsAtPosition(currentRow, currentCol, board);
            }

            return solvePartialSudoku(currentRow, currentCol + 1, board);
        }
        public bool tryDigitsAtPosition(int row, int col, List<List<int>> board)
        {
            for (int digit = 1; digit < 10; digit++)
            {
                if (isValidAtPosition(digit, row, col, board))
                {
                    board[row][col] = digit;
                    if (solvePartialSudoku(row, col + 1, board))
                    {
                        return true;
                    }
                }
            }

            board[row][col] = 0;
            return false;
        }

        public bool isValidAtPosition(
          int value, int row, int col, List<List<int>> board
        )
        {
            bool rowIsValid = !board[row].Contains(value);
            bool columnIsValid = true;

            for (int r = 0; r < board.Count; r++)
            {
                if (board[r][col] == value) columnIsValid = false;
            }

            if (!rowIsValid || !columnIsValid)
            {
                return false;
            }

            // Check subgrid constraints
            int subgridRowStart = (row / 3) * 3;
            int subgridColStart = (col / 3) * 3;

            for (int rowIdx = 0; rowIdx < 3; rowIdx++)
            {
                for (int colIdx = 0; colIdx < 3; colIdx++)
                {
                    int rowToCheck = subgridRowStart + rowIdx;
                    int colToCheck = subgridColStart + colIdx;
                    int existingValue = board[rowToCheck][colToCheck];

                    if (existingValue == value)
                    {
                        return false;
                    }
                }
            }

            return true;
        }
        /*
        286. Walls and Gates
        https://leetcode.com/problems/walls-and-gates/description/      

        */
        public void WallsAndGates(int[][] rooms)
        {
            //1. Brute Force [Time Limit Exceeded]
            /*
            Time complexity : O(m^2*n^2)   For each point in the m×n size grid, the gate could be at most m×n steps away.
            Space complexity : O(mn). The space complexity depends on the queue's size. Since we won't insert points that have been visited before into the queue, we insert at most m×n points into the queue.
            */
            WallsAndGatesNaive(rooms);

            //2. Breadth-first Search
            /*
            •	Time complexity : O(mn).
            •	Space complexity : O(mn).The space complexity depends on the queue's size. We insert at most m×n points into the queue.
            */
            WallsAndGatesBFSOptimal(rooms);
        }

        private static readonly int EMPTY = int.MaxValue;
        private static readonly int GATE = 0;
        private static readonly int WALL = -1;
        private static readonly List<int[]> DIRECTIONS = new List<int[]>{
        new int[] { 1,  0},
        new int[] {-1,  0},
        new int[] { 0,  1},
        new int[] { 0, -1}
        };
        private void WallsAndGatesBFSOptimal(int[][] rooms)
        {
            int m = rooms.Length;
            if (m == 0) return;
            int n = rooms[0].Length;
            Queue<int[]> q = new Queue<int[]>();
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    if (rooms[row][col] == GATE)
                    {
                        q.Enqueue(new int[] { row, col });
                    }
                }
            }
            while (q.Count >0)
            {
                int[] point = q.Dequeue();
                int row = point[0];
                int col = point[1];
                foreach (int[] direction in DIRECTIONS)
                {
                    int r = row + direction[0];
                    int c = col + direction[1];
                    if (r < 0 || c < 0 || r >= m || c >= n || rooms[r][c] != EMPTY)
                    {
                        continue;
                    }
                    rooms[r][c] = rooms[row][col] + 1;
                    q.Enqueue(new int[] { r, c });
                }
            }

        }

        private void WallsAndGatesNaive(int[][] rooms)
        {
            if (rooms.Length == 0) return;
            for (int row = 0; row < rooms.Length; row++)
            {
                for (int col = 0; col < rooms[0].Length; col++)
                {
                    if (rooms[row][col] == EMPTY)
                    {
                        rooms[row][col] = DistanceToNearestGate(rooms, row, col);
                    }
                }
            }

        }

        private int DistanceToNearestGate(int[][] rooms, int startRow, int startCol)
        {
            int m = rooms.Length;
            int n = rooms[0].Length;
            int[][] distance = new int[m][];
            Queue<int[]> q = new Queue<int[]>();
            q.Enqueue(new int[] { startRow, startCol });
            while (q.Count > 0)
            {
                int[] point = q.Dequeue();
                int row = point[0];
                int col = point[1];
                foreach (int[] direction in DIRECTIONS)
                {
                    int r = row + direction[0];
                    int c = col + direction[1];
                    if (r < 0 || c < 0 || r >= m || c >= n || rooms[r][c] == WALL
                            || distance[r][c] != 0)
                    {
                        continue;
                    }
                    distance[r][c] = distance[row][col] + 1;
                    if (rooms[r][c] == GATE)
                    {
                        return distance[r][c];
                    }
                    q.Enqueue(new int[] { r, c });
                }
            }
            return int.MaxValue;

        }
    }
}