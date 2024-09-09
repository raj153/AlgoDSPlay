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
        /*
        2664. The Knight’s Tour
https://leetcode.com/problems/the-knights-tour/
https://algo.monster/liteproblems/2664

        */
        class TourOfKnightSol
        {
            private int[][] chessboard; // The chessboard representation
            private int numberOfRows;     // Number of rows in the chessboard
            private int numberOfColumns;   // Number of columns in the chessboard
            private bool isSolutionFound; // Flag to indicate if a solution is found

            // Method to generate the tour of a knight on a chessboard
            public int[][] Backtrack(int rows, int cols, int startRow, int startCol)
            {
                this.numberOfRows = rows;
                this.numberOfColumns = cols;
                this.chessboard = new int[rows][];
                for (int i = 0; i < rows; i++)
                {
                    chessboard[i] = new int[cols];
                }
                this.isSolutionFound = false;

                // Initialize all cells as unvisited by setting them to -1
                for (int rowIndex = 0; rowIndex < chessboard.Length; rowIndex++)
                {
                    Array.Fill(chessboard[rowIndex], -1);
                }

                // Start tour at the given starting position by setting it to 0
                chessboard[startRow][startCol] = 0;

                // Use Depth-First Search to explore all possible moves
                Dfs(startRow, startCol);
                return chessboard; // Return the completed tour grid
            }

            // Helper method for DFS traversal from a given cell (i, j)
            private void Dfs(int currentRow, int currentCol)
            {
                // Check if we've visited all cells, meaning a full tour is complete
                if (chessboard[currentRow][currentCol] == numberOfRows * numberOfColumns - 1)
                {
                    isSolutionFound = true;
                    return; // Found a solution, so backtrack
                }

                // Array of possible moves a knight can make (8 possible moves)
                int[] moveX = { -2, -1, 1, 2, 2, 1, -1, -2 };
                int[] moveY = { 1, 2, 2, 1, -1, -2, -2, -1 };

                // Explore all possible moves
                for (int moveIndex = 0; moveIndex < 8; ++moveIndex)
                {
                    int nextRow = currentRow + moveX[moveIndex];
                    int nextCol = currentCol + moveY[moveIndex];

                    // Check if the move is within bounds and the cell is not yet visited
                    if (IsValidMove(nextRow, nextCol))
                    {
                        chessboard[nextRow][nextCol] = chessboard[currentRow][currentCol] + 1; // Mark the cell with the move number
                        Dfs(nextRow, nextCol); // Continue dfs from the new cell

                        // If a solution is found, no need to explore further; start backtracking
                        if (isSolutionFound)
                        {
                            return;
                        }

                        // Backtrack: Unmark the cell as part of the path as it leads to no solution
                        chessboard[nextRow][nextCol] = -1;
                    }
                }
            }

            // Helper method to check if a move is valid and legal on the chessboard
            private bool IsValidMove(int x, int y)
            {
                return x >= 0 && x < numberOfRows && y >= 0 && y < numberOfColumns && chessboard[x][y] == -1;
            }
        }
        //https://www.algoexpert.io/questions/solve-sudoku
        // O(1) time | O(1) space - assuming a 9x9 input board
        public List<List<int>> SolveSudoku(List<List<int>> board)
        {
            SolvePartialSudoku(0, 0, board);
            return board;
        }

        public bool SolvePartialSudoku(int row, int col, List<List<int>> board)
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

            return SolvePartialSudoku(currentRow, currentCol + 1, board);
        }
        public bool tryDigitsAtPosition(int row, int col, List<List<int>> board)
        {
            for (int digit = 1; digit < 10; digit++)
            {
                if (isValidAtPosition(digit, row, col, board))
                {
                    board[row][col] = digit;
                    if (SolvePartialSudoku(row, col + 1, board))
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
            while (q.Count > 0)
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
        /*
        36. Valid Sudoku
        https://leetcode.com/problems/valid-sudoku/description/

        */
        public bool IsValidSudoku(char[][] board)
        {
            /*
  Approach 1: Hash Set  (HS)       
Complexity Analysis
Let N be the board length, which is 9 in this question. Note that since the value of N is fixed, the time and space complexity of this algorithm can be interpreted as O(1). However, to better compare each of the presented approaches, we will treat N as an arbitrary value in the complexity analysis below.
•	Time complexity: O(N^2) because we need to traverse every position in the board, and each of the four check steps is an O(1) operation.
•	Space complexity: O(N^2) because in the worst-case scenario, if the board is full, we need a hash set each with size N to store all seen numbers for each of the N rows, N columns, and N boxes, respectively.

            */
            bool isValidSudoku = IsValidSudokuHS(board);
            /*
  Approach 2: Array of Fixed Length  (AFL)        
  Complexity Analysis
Let N be the board length, which is 9 in this question. Note that since the value of N is fixed, the time and space complexity of this algorithm can be interpreted as O(1). However, to better compare each of the presented approaches, we will treat N as an arbitrary value in the complexity analysis below.
•	Time complexity: O(N^2) because we need to traverse every position in the board, and each of the four check steps is an O(1) operation.
•	Space complexity: O(N^2) because we need to create 3N arrays each with size N to store all previously seen numbers for all rows, columns, and boxes.
          
            */
            isValidSudoku = IsValidSudokuAFL(board);
            /*
   Approach 3: Bitmasking (BM)        
   Complexity Analysis
Let N be the board length, which is 9 in this question. Note that since the value of N is fixed, the time and space complexity of this algorithm can be interpreted as O(1). However, to better compare each of the presented approaches, we will treat N as an arbitrary value in the complexity analysis below.
•	Time complexity: O(N^2) because we need to traverse every position in the board, and each of the four check steps is an O(1) operation.
•	Space complexity: O(N) because in the worst-case scenario, if the board is full, we need 3N binary numbers to store all seen numbers in all rows, columns, and boxes. Using a binary number to record the occurrence of numbers is probably the most space-efficient method.
         
            */
            isValidSudoku = IsValidSudokuBM(board);

            return isValidSudoku;

        }
        public bool IsValidSudokuHS(char[][] board)
        {
            int N = 9;
            // Use hash set to record the status
            HashSet<char>[] rows = new HashSet<char>[N];
            HashSet<char>[] cols = new HashSet<char>[N];
            HashSet<char>[] boxes = new HashSet<char>[N];
            for (int r = 0; r < N; r++)
            {
                rows[r] = new HashSet<char>();
                cols[r] = new HashSet<char>();
                boxes[r] = new HashSet<char>();
            }

            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    char val = board[r][c];
                    // Check if the position is filled with number
                    if (val == '.')
                    {
                        continue;
                    }

                    // Check the row
                    if (rows[r].Contains(val))
                    {
                        return false;
                    }

                    rows[r].Add(val);
                    // Check the column
                    if (cols[c].Contains(val))
                    {
                        return false;
                    }

                    cols[c].Add(val);
                    // Check the box
                    int idx = (r / 3) * 3 + c / 3;
                    if (boxes[idx].Contains(val))
                    {
                        return false;
                    }

                    boxes[idx].Add(val);
                }
            }

            return true;
        }

        public bool IsValidSudokuAFL(char[][] board)
        {
            int N = 9;
            // Use an array to record the status
            int[][] rows = new int[N][];
            int[][] cols = new int[N][];
            int[][] boxes = new int[N][];
            for (int i = 0; i < N; i++)
            {
                rows[i] = new int[N];
                cols[i] = new int[N];
                boxes[i] = new int[N];
            }

            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    // Check if the position is filled with number
                    if (board[r][c] == '.')
                    {
                        continue;
                    }

                    int pos = board[r][c] - '1';
                    // Check the row
                    if (rows[r][pos] == 1)
                    {
                        return false;
                    }

                    rows[r][pos] = 1;
                    // Check the column
                    if (cols[c][pos] == 1)
                    {
                        return false;
                    }

                    cols[c][pos] = 1;
                    // Check the box
                    int idx = (r / 3) * 3 + c / 3;
                    if (boxes[idx][pos] == 1)
                    {
                        return false;
                    }

                    boxes[idx][pos] = 1;
                }
            }

            return true;
        }

        public bool IsValidSudokuBM(char[][] board)
        {
            int N = 9;
            // Use a binary number to record previous occurrence
            int[] rows = new int[N];
            int[] cols = new int[N];
            int[] boxes = new int[N];
            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    // Check if the position is filled with number
                    if (board[r][c] == '.')
                    {
                        continue;
                    }

                    int val = board[r][c] - '0';
                    int pos = 1 << (val - 1);
                    // Check the row
                    if ((rows[r] & pos) > 0)
                    {
                        return false;
                    }

                    rows[r] |= pos;
                    // Check the column
                    if ((cols[c] & pos) > 0)
                    {
                        return false;
                    }

                    cols[c] |= pos;
                    // Check the box
                    int idx = (r / 3) * 3 + c / 3;
                    if ((boxes[idx] & pos) > 0)
                    {
                        return false;
                    }

                    boxes[idx] |= pos;
                }
            }

            return true;
        }

        /*
        37. Sudoku Solver		
        https://leetcode.com/problems/sudoku-solver/description/

        Approach 0: Brute Force

        Approach 1: Backtracking
        Complexity Analysis
        •	Time complexity is constant here since the board size is fixed and there is no N-parameter to measure. Though let's discuss the number of operations needed : (9!)^9. Let's consider one row, i.e. not more than 9 cells to fill. There are not more than 9 possibilities for the first number to put, not more than 9×8 for the second one, not more than 9×8×7 for the third one, etc. In total that results in not more than 9! possibilities for just one row, which means no more than (9!)9 operations in total.
        Let's compare:
        o	981=196627050475552913618075908526912116283103450944214766927315415537966391196809
        for the brute force,
        o	and (9!)^9=109110688415571316480344899355894085582848000000000
        for the standard backtracking, i.e. the number of operations is reduced in 1027 times!
        •	Space complexity: the board size is fixed, and the space is used to store board, rows, columns, and box structures, each containing 81 elements.

        */
        public class SudokuSolution
        {
            // box size
            int n;

            // row size
            int N;
            int[][] rows;
            int[][] columns;
            int[][] boxes;
            char[][] board;
            bool sudokuSolved = false;

            public SudokuSolution()
            {
                n = 3;
                N = n * n;
                rows = new int[N][];
                columns = new int[N][];
                boxes = new int[N][];
                for (int k = 0; k < N; k++)
                {
                    rows[k] = new int[N + 1];
                    columns[k] = new int[N + 1];
                    boxes[k] = new int[N + 1];
                }
            }

            public bool CouldPlace(int d, int row, int col)
            {
                int idx = (row / n) * n + col / n;
                return rows[row][d] + columns[col][d] + boxes[idx][d] == 0;
            }

            public void PlaceNumber(int d, int row, int col)
            {
                int idx = (row / n) * n + col / n;
                rows[row][d]++;
                columns[col][d]++;
                boxes[idx][d]++;
                board[row][col] = (char)(d + '0');
            }

            public void RemoveNumber(int d, int row, int col)
            {
                int idx = (row / n) * n + col / n;
                rows[row][d]--;
                columns[col][d]--;
                boxes[idx][d]--;
                board[row][col] = '.';
            }

            public void PlaceNextNumbers(int row, int col)
            {
                if ((col == N - 1) && (row == N - 1))
                {
                    sudokuSolved = true;
                }
                else
                {
                    if (col == N - 1)
                        Backtrack(row + 1, 0);
                    else
                        Backtrack(row, col + 1);
                }
            }

            public void Backtrack(int row, int col)
            {
                if (board[row][col] == '.')
                {
                    for (int d = 1; d < 10; d++)
                    {
                        if (CouldPlace(d, row, col))
                        {
                            PlaceNumber(d, row, col);
                            PlaceNextNumbers(row, col);
                            if (!sudokuSolved)
                                RemoveNumber(d, row, col);
                        }
                    }
                }
                else
                    PlaceNextNumbers(row, col);
            }

            public void SolveSudoku(char[][] board)
            {
                this.board = board;
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        char num = board[i][j];
                        if (num != '.')
                        {
                            int d = (int)char.GetNumericValue(num);
                            PlaceNumber(d, i, j);
                        }
                    }
                }

                Backtrack(0, 0);
            }
        }

        /*
        174. Dungeon Game
        https://leetcode.com/problems/dungeon-game/description/

        */
        public int CalculateMinimumHP(int[][] dungeon)
        {
            /*
Approach 1: Dynamic Programming
Complexity

Time Complexity: O(M⋅N) where M⋅N is the size of the dungeon. We iterate through the entire dungeon once and only once.

Space Complexity: O(M⋅N) where M⋅N is the size of the dungeon. In the algorithm, we keep a dp matrix that is of the same size as the dungeon.
            
            */
            int minHealth = CalculateMinimumHPDP(dungeon);

            /*
Approach 2: Dynamic Programming with Circular Queue (DPCQ)
Complexity

Time Complexity: O(M⋅N) where M⋅N is the size of the dungeon. We iterate through the entire dungeon once and only once.

Space Complexity: O(N) where N is the number of columns in the dungeon
            
            */
            minHealth = CalculateMinimumHPDPCQ(dungeon);

            return minHealth;
        }

        const int inf = int.MaxValue;
        int[,] dp;
        int rows, cols;

        public int GetMinHealth(int currCell, int nextRow, int nextCol)
        {
            if (nextRow >= this.rows || nextCol >= this.cols)
                return inf;
            int nextCell = this.dp[nextRow, nextCol];
            // hero needs at least 1 point to survive
            return Math.Max(1, nextCell - currCell);
        }

        public int CalculateMinimumHPDP(int[][] dungeon)
        {
            this.rows = dungeon.Length;
            this.cols = dungeon[0].Length;
            this.dp = new int[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    dp[i, j] = inf;
                }
            }

            int currCell, rightHealth, downHealth, nextHealth, minHealth;
            for (int row = this.rows - 1; row >= 0; --row)
            {
                for (int col = this.cols - 1; col >= 0; --col)
                {
                    currCell = dungeon[row][col];

                    rightHealth = GetMinHealth(currCell, row, col + 1);
                    downHealth = GetMinHealth(currCell, row + 1, col);
                    nextHealth = Math.Min(rightHealth, downHealth);

                    if (nextHealth != inf)
                    {
                        minHealth = nextHealth;
                    }
                    else
                    {
                        minHealth = currCell >= 0 ? 1 : 1 - currCell;
                    }

                    this.dp[row, col] = minHealth;
                }
            }

            return this.dp[0, 0];
        }

        public class MyCircularQueue
        {
            protected int capacity;
            protected int tailIndex;
            public int[] queue;

            public MyCircularQueue(int capacity)
            {
                this.queue = new int[capacity];
                this.tailIndex = 0;
                this.capacity = capacity;
            }

            public void EnQueue(int value)
            {
                this.queue[this.tailIndex] = value;
                this.tailIndex = (this.tailIndex + 1) % this.capacity;
            }

            public int Get(int index)
            {
                return this.queue[index % this.capacity];
            }
        }
        MyCircularQueue dpCQ;

        public int GetMinHealth2(int currCell, int nextRow, int nextCol)
        {
            if (nextRow < 0 || nextCol < 0)
                return inf;

            int index = cols * nextRow + nextCol;
            int nextCell = this.dpCQ.Get(index);
            return Math.Max(1, nextCell - currCell);
        }

        public int CalculateMinimumHPDPCQ(int[][] dungeon)
        {
            this.rows = dungeon.Length;
            this.cols = dungeon[0].Length;
            this.dpCQ = new MyCircularQueue(this.cols);

            int currCell, rightHealth, downHealth, nextHealth, minHealth;
            for (int row = 0; row < this.rows; ++row)
            {
                for (int col = 0; col < this.cols; ++col)
                {
                    currCell = dungeon[rows - row - 1][cols - col - 1];

                    rightHealth = GetMinHealth2(currCell, row, col - 1);
                    downHealth = GetMinHealth2(currCell, row - 1, col);
                    nextHealth = Math.Min(rightHealth, downHealth);

                    if (nextHealth != inf)
                    {
                        minHealth = nextHealth;
                    }
                    else
                    {
                        minHealth = currCell >= 0 ? 1 : 1 - currCell;
                    }

                    this.dpCQ.EnQueue(minHealth);
                }
            }

            return this.dpCQ.Get(this.cols - 1);
        }


        /*

        2214. Minimum Health to Beat Game
        https://leetcode.com/problems/minimum-health-to-beat-game/description/	

        Approach: Greedy
        Complexity Analysis
        Here, n is the number of levels in the game.
        •	Time complexity: O(n)
        o	We iterate once through the complete array damage to compute the totalDamage and maxDamage.
        •	Space complexity: O(1)
        o	We only used two variables: maxDamage and totalDamage.


        */
        public long MinimumHealth(int[] damage, int armor)
        {
            int maxDamage = 0;
            long totalDamage = 0;

            foreach (int d in damage)
            {
                totalDamage += d;
                maxDamage = Math.Max(maxDamage, d);
            }

            return totalDamage - Math.Min(armor, maxDamage) + 1;
        }

        /*
        1921. Eliminate Maximum Number of Monsters
        https://leetcode.com/problems/eliminate-maximum-number-of-monsters/description/

        */
        public int EliminateMaximum(int[] dist, int[] speed)
        {
            /*
            
Approach 1: Sort By Arrival Time
 Complexity Analysis
Given n as the length of dist and speed,
•	Time complexity: O(n⋅logn)
Creating arrival costs O(n). Then, we sort it which costs O(n⋅logn). Finally, we iterate up to n times.
•	Space complexity: O(n)
arrival has a size of O(n). Note that we could instead modify one of the input arrays and use that as arrival. However, it is generally considered bad practice to modify the input, especially when it is something passed by reference like an array. Also, many people will argue that if you modify the input, you must include it as part of the space complexity anyway.
           
            */

            int maxNumOfMonstersEleminated = EliminateMaximumSort(dist, speed);



            /*            
 Approach 2: Heap           
 Complexity Analysis
Given n as the length of dist and speed,
•	Time complexity: O(n⋅logn)
The heap operations will cost O(logn). If all monsters can be killed, then we will perform O(n) iterations and thus use O(n⋅logn) time.
Note: an array can be converted to a heap in linear time. In fact, Python's heapq.heapify does this, as does C++ std::priority_queue constructor. Without linear time heapify, we always use O(n⋅logn) time since we need to build the heap. However, if we have linear time heapify and a monster reaches our city early, then this algorithm will have a better theoretical performance, since not many O(logn) operations will occur.
•	Space complexity: O(n)
heap uses O(n) space.

            
            */


            maxNumOfMonstersEleminated = EliminateMaximumHeap(dist, speed);

            return maxNumOfMonstersEleminated;

        }
        public int EliminateMaximumSort(int[] dist, int[] speed)
        {
            double[] arrival = new double[dist.Length];
            for (int i = 0; i < dist.Length; i++)
            {
                arrival[i] = (double)dist[i] / speed[i];
            }

            Array.Sort(arrival);
            int ans = 0;

            for (int i = 0; i < arrival.Length; i++)
            {
                if (arrival[i] <= i)
                {
                    break;
                }

                ans++;
            }

            return ans;
        }
        public int EliminateMaximumHeap(int[] dist, int[] speed)
        {
            PriorityQueue<double, double> minHeap = new PriorityQueue<double, double>();
            for (int i = 0; i < dist.Length; i++)
            {
                var arrivalTime = (double)dist[i] / speed[i];
                minHeap.Enqueue(arrivalTime, arrivalTime);
            }

            int ans = 0;
            while (minHeap.Count > 0)
            {
                if (minHeap.Dequeue() <= ans)
                {
                    break;
                }

                ans++;
            }

            return ans;
        }


    }
}