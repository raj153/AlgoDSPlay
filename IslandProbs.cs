using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class IslandProbs
    {
        /* 
        1568. Minimum Number of Days to Disconnect Island
        https://leetcode.com/problems/minimum-number-of-days-to-disconnect-island/description/
         */
        public class MinimumNumberOfDaysToDisconnectIsland
        {
            // Directions for adjacent cells: right, left, down, up
            private static readonly int[][] DIRECTIONS = {
        new int[]{ 0, 1 },
        new int[]{ 0, -1 },
        new int[]{ 1, 0 },
        new int[]{ -1, 0 },
    };
            /*
            Approach 1: Brute Force
            Complexity Analysis
            Let m be the number of rows and n be the number of columns in the grid.
            •	Time complexity: O((m⋅n)^2)
            The main operation in this algorithm is the countIslands function, which is called multiple times. countIslands in turn calls the exploreIslands method, which performs a depth-first search on the grid. The DFS in the worst case can explore all the cells in the grid, resulting in a time complexity of O(m⋅n).
            The countIslands method may be called a maximum of 1+m⋅n times.
            Thus, the overall time complexity of the algorithm is O((m⋅n)⋅(1+m⋅n)), which simplifies to O((m⋅n) ^2).
            •	Space complexity: O(m⋅n)
            The main space usage comes from the visited array in the countIslands function, which has a size of m×n.
            The recursive call stack in the DFS (exploreIsland function) can go as deep as m⋅n in the worst case.
            Therefore, the space complexity of the algorithm is O(m⋅n).

            */
            public int Naive(int[][] grid)
            {
                int rows = grid.GetLength(0);
                int cols = grid.GetLength(1);

                // Count initial islands
                int initialIslandCount = CountIslands(grid);

                // Already disconnected or no land
                if (initialIslandCount != 1)
                {
                    return 0;
                }

                // Try removing each land cell
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {
                        if (grid[row][col] == 0) continue; // Skip water

                        // Temporarily change to water
                        grid[row][col] = 0;
                        int newIslandCount = CountIslands(grid);

                        // Check if disconnected
                        if (newIslandCount != 1) return 1;

                        // Revert change
                        grid[row][col] = 1;
                    }
                }

                return 2;
            }

            private int CountIslands(int[][] grid)
            {
                int rows = grid.GetLength(0);
                int cols = grid.GetLength(1);
                bool[,] visited = new bool[rows, cols];
                int islandCount = 0;

                // Iterate through all cells
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {
                        // Found new island
                        if (!visited[row, col] && grid[row][col] == 1)
                        {
                            ExploreIsland(grid, row, col, visited);
                            islandCount++;
                        }
                    }
                }
                return islandCount;
            }

            // Helper method to explore all cells of an island
            private void ExploreIsland(int[][] grid, int row, int col, bool[,] visited)
            {
                visited[row, col] = true;

                // Check all adjacent cells
                for (int i = 0; i < DIRECTIONS.GetLength(0); i++)
                {
                    int newRow = row + DIRECTIONS[i][0];
                    int newCol = col + DIRECTIONS[i][1];
                    // Explore if valid land cell
                    if (IsValidLandCell(grid, newRow, newCol, visited))
                    {
                        ExploreIsland(grid, newRow, newCol, visited);
                    }
                }
            }

            private bool IsValidLandCell(int[][] grid, int row, int col, bool[,] visited)
            {
                int rows = grid.GetLength(0);
                int cols = grid.GetLength(1);
                // Check bounds, land, and not visited
                return (
                    row >= 0 &&
                    col >= 0 &&
                    row < rows &&
                    col < cols &&
                    grid[row][col] == 1 &&
                    !visited[row, col]
                );
            }

            /*
            Approach 2: Tarjan's Algorithm
            Complexity Analysis
Let m be the number of rows and n be the number of columns in the grid.
•	Time complexity: O(m⋅n)
Initializing the arrays discoveryTime, lowestReachable, and parentCell takes O(m⋅n) time each.
The DFS traversal by the findArticulationPoints method visits each cell exactly once, taking O(m⋅n) time.
Thus, the overall time complexity of the algorithm is O(m⋅n).
•	Space complexity: O(m⋅n)
The arrays discoveryTime, lowestReachable, and parentCell each take O(m⋅n) space.
The recursive call stack for the DFS traversal can go as deep as the number of land cells in the worst case. If all cells are land, the depth of the recursive call stack can be O(m⋅n).
Thus, the total space complexity of the algorithm is O(m⋅n)+O(m⋅n)=O(m⋅n).

            */
            public int TarjansAlgo(int[][] grid)
            {
                int rows = grid.GetLength(0), cols = grid.GetLength(1);
                ArticulationPointInfo articulationPointInfo = new ArticulationPointInfo(false, 0);
                int landCellCount = 0, islandCount = 0;

                // Arrays to store information for each cell
                int[][] discoveryTime = new int[rows][]; // Time when a cell is first discovered
                int[][] lowestReachable = new int[rows][]; // Lowest discovery time reachable from the subtree rooted at this cell
                int[][] parentCell = new int[rows][]; // Parent of each cell in DFS tree

                // Initialize arrays with default values
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        discoveryTime[i][j] = -1;
                        lowestReachable[i][j] = -1;
                        parentCell[i][j] = -1;
                    }
                }

                // Traverse the grid to find islands and articulation points
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        if (grid[i][j] == 1)
                        {
                            landCellCount++;
                            if (discoveryTime[i][j] == -1) // If not yet visited
                            {
                                // Start DFS for a new island
                                FindArticulationPoints(
                                    grid,
                                    i,
                                    j,
                                    discoveryTime,
                                    lowestReachable,
                                    parentCell,
                                    articulationPointInfo
                                );
                                islandCount++;
                            }
                        }
                    }
                }

                // Determine the minimum number of days to disconnect the grid
                if (islandCount == 0 || islandCount >= 2) return 0; // Already disconnected or no land
                if (landCellCount == 1) return 1; // Only one land cell
                if (articulationPointInfo.HasArticulationPoint) return 1; // An articulation point exists
                return 2; // Need to remove any two land cells
            }

            private void FindArticulationPoints(
                int[][] grid,
                int row,
                int col,
                int[][] discoveryTime,
                int[][] lowestReachable,
                int[][] parentCell,
                ArticulationPointInfo articulationPointInfo
            )
            {
                int rows = grid.Length, cols = grid[0].Length;
                discoveryTime[row][col] = articulationPointInfo.Time;
                articulationPointInfo.Time++;
                lowestReachable[row][col] = discoveryTime[row][col];
                int childCount = 0;

                // Explore all adjacent cells
                foreach (int[] direction in DIRECTIONS)
                {
                    int newRow = row + direction[0];
                    int newCol = col + direction[1];
                    if (IsValidLandCell(grid, newRow, newCol))
                    {
                        if (discoveryTime[newRow][newCol] == -1)
                        {
                            childCount++;
                            parentCell[newRow][newCol] = row * cols + col; // Set parent
                            FindArticulationPoints(
                                grid,
                                newRow,
                                newCol,
                                discoveryTime,
                                lowestReachable,
                                parentCell,
                                articulationPointInfo
                            );

                            // Update lowest reachable time
                            lowestReachable[row][col] = Math.Min(
                                lowestReachable[row][col],
                                lowestReachable[newRow][newCol]
                            );

                            // Check for articulation point condition
                            if (
                                lowestReachable[newRow][newCol] >=
                                    discoveryTime[row][col] &&
                                parentCell[row][col] != -1
                            )
                            {
                                articulationPointInfo.HasArticulationPoint = true;
                            }
                        }
                        else if (newRow * cols + newCol != parentCell[row][col])
                        {
                            // Update lowest reachable time for back edge
                            lowestReachable[row][col] = Math.Min(
                                lowestReachable[row][col],
                                discoveryTime[newRow][newCol]
                            );
                        }
                    }
                }

                // Root of DFS tree is an articulation point if it has more than one child
                if (parentCell[row][col] == -1 && childCount > 1)
                {
                    articulationPointInfo.HasArticulationPoint = true;
                }
            }
            // Check if the given cell is a valid land cell
            private bool IsValidLandCell(int[][] grid, int row, int col)
            {
                int rows = grid.GetLength(0), cols = grid.GetLength(1);
                return (
                    row >= 0 &&
                    col >= 0 &&
                    row < rows &&
                    col < cols &&
                    grid[row][col] == 1
                );
            }

            private class ArticulationPointInfo
            {
                public bool HasArticulationPoint;
                public int Time;

                public ArticulationPointInfo(bool hasArticulationPoint, int time)
                {
                    HasArticulationPoint = hasArticulationPoint;
                    Time = time;
                }
            }
        }
        /*
            305. Number of Islands II
            https://leetcode.com/problems/number-of-islands-ii/description/

            UNION FIND

            •	Time complexity: O(m⋅n+l)
                o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1).
                o	Initializing UnionFind takes O(m⋅n) time beacuse we are initializing the parent and rank arrays of size m∗n each.
                o	For each position in positions, we perform addLand which takes O(1) time. Furthermore, we check all four neighbors of every position and if there is land at any neighbor, we perform union of position and the neighbor. Because there can only be four union operations at a time, each union operation would take O(4)=O(1) time. It would take O(l) time for l positions.
                o	Obtaining the number of islands for each position and pushing it to answer takes O(1) per position. For l positions, it would take O(l) time.
                o	As a result, the total time required is O(m⋅n+l).
            •	Space complexity: O(m⋅n)
                o	We are using the parent and rank arrays, both of which require O(m⋅n) space.
                o	Other integers, such as count, and arrays, such as x and y take up O(1) space.

            */

        public List<int> NumIslands2(int rows, int columns, int[][] positions)
        {
            int[] xOffsets = { -1, 1, 0, 0 };
            int[] yOffsets = { 0, 0, -1, 1 };
            UnionFindExt unionFind = new UnionFindExt(rows * columns);
            List<int> result = new List<int>();

            foreach (int[] position in positions)
            {
                int landPosition = position[0] * columns + position[1];
                unionFind.AddLand(landPosition);

                for (int i = 0; i < 4; i++)
                {
                    int neighborRow = position[0] + xOffsets[i];
                    int neighborColumn = position[1] + yOffsets[i];
                    int neighborPosition = neighborRow * columns + neighborColumn;

                    // If neighborRow and neighborColumn correspond to a point in the grid and there is a
                    // land at that point, then merge it with the current land.
                    if (neighborRow >= 0 && neighborRow < rows && neighborColumn >= 0 && neighborColumn < columns &&
                            unionFind.IsLand(neighborPosition))
                    {
                        unionFind.Union(landPosition, neighborPosition);
                    }
                }
                result.Add(unionFind.NumberOfIslands());
            }
            return result;
        }

        class UnionFindExt
        {
            private int[] parent;
            private int[] rank;
            private int count;

            public UnionFindExt(int size)
            {
                parent = new int[size];
                rank = new int[size];
                for (int i = 0; i < size; i++)
                    parent[i] = -1;
                count = 0;
            }

            public void AddLand(int landIndex)
            {
                if (parent[landIndex] >= 0)
                    return;
                parent[landIndex] = landIndex;
                count++;
            }

            public bool IsLand(int landIndex)
            {
                return parent[landIndex] >= 0;
            }

            public int NumberOfIslands()
            {
                return count;
            }

            public int Find(int landIndex)
            {
                if (parent[landIndex] != landIndex)
                    parent[landIndex] = Find(parent[landIndex]);
                return parent[landIndex];
            }

            public void Union(int landIndex1, int landIndex2)
            {
                int set1 = Find(landIndex1), set2 = Find(landIndex2);
                if (set1 == set2)
                {
                    return;
                }
                else if (rank[set1] < rank[set2])
                {
                    parent[set1] = set2;
                }
                else if (rank[set1] > rank[set2])
                {
                    parent[set2] = set1;
                }
                else
                {
                    parent[set2] = set1;
                    rank[set1]++;
                }
                count--;
            }
        }
        /*
        200. Number of Islands	
        https://leetcode.com/problems/number-of-islands/description/
        */
        public int NumIslands(int[][] grid)
        {
            if (grid == null || grid.Length == 0) return 0;
            int numberOfIslands = 0;

            //1.DFS 
            /*
            Time complexity : O(M×N) where M is the number of rows and N is the number of columns.
            Space complexity : worst case O(M×N) in case that the grid map is filled with lands where DFS goes by M×N deep.
            */
            numberOfIslands = NumIslandsDFS(grid);

            //2.BFS 
            /*
            Time complexity : O(M×N) where M is the number of rows and N is the number of columns.
            Space complexity : O(min(M,N)) because in worst case where the grid is filled with lands, the size of queue can grow up to min(M,N).
            */
            numberOfIslands = NumIslandsBFS(grid);

            //3.Union Find (aka Disjoint Set)
            /*
            Time complexity : O(M×N) where M is the number of rows and N is the number of columns. Note that Union operation takes essentially constant time when UnionFind is implemented with both path compression and union by rank.
            Space complexity : O(M×N) as required by UnionFind data structure.
            */
            numberOfIslands = NumIslandsUnionFind(grid);

            return numberOfIslands;
        }

        private int NumIslandsUnionFind(int[][] grid)
        {
            if (grid == null || grid.Length == 0)
            {
                return 0;
            }

            int nr = grid.GetLength(0);
            int nc = grid.GetLength(1);
            UnionFindExt1 uf = new UnionFindExt1(grid);
            for (int r = 0; r < nr; ++r)
            {
                for (int c = 0; c < nc; ++c)
                {
                    if (grid[r][c] == 1)
                    {
                        grid[r][c] = 0;
                        if (r - 1 >= 0 && grid[r - 1][c] == 1)
                        {
                            uf.Union(r * nc + c, (r - 1) * nc + c);
                        }
                        if (r + 1 < nr && grid[r + 1][c] == 1)
                        {
                            uf.Union(r * nc + c, (r + 1) * nc + c);
                        }
                        if (c - 1 >= 0 && grid[r][c - 1] == 1)
                        {
                            uf.Union(r * nc + c, r * nc + c - 1);
                        }
                        if (c + 1 < nc && grid[r][c + 1] == 1)
                        {
                            uf.Union(r * nc + c, r * nc + c + 1);
                        }
                    }
                }
            }

            return uf.GetCount();
        }
        private class UnionFindExt1
        {
            private int count; // # of connected components
            private int[] parent;
            private int[] rank;

            public UnionFindExt1(int[][] grid)
            { // for problem 200
                count = 0;
                int m = grid.GetLength(0);
                int n = grid.GetLength(1);
                parent = new int[m * n];
                rank = new int[m * n];
                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        if (grid[i][j] == 1)
                        {
                            parent[i * n + j] = i * n + j;
                            ++count;
                        }
                        rank[i * n + j] = 0;
                    }
                }
            }

            public int Find(int i)
            { // path compression
                if (parent[i] != i) parent[i] = Find(parent[i]);
                return parent[i];
            }

            public void Union(int x, int y)
            { // union with rank
                int rootx = Find(x);
                int rooty = Find(y);
                if (rootx != rooty)
                {
                    if (rank[rootx] > rank[rooty])
                    {
                        parent[rooty] = rootx;
                    }
                    else if (rank[rootx] < rank[rooty])
                    {
                        parent[rootx] = rooty;
                    }
                    else
                    {
                        parent[rooty] = rootx;
                        rank[rootx] += 1;
                    }
                    --count;
                }
            }

            public int GetCount()
            {
                return count;
            }
        }
        private int NumIslandsBFS(int[][] grid)
        {
            if (grid == null || grid.Length == 0)
            {
                return 0;
            }

            int nr = grid.GetLength(0);
            int nc = grid.GetLength(1);
            int numIslands = 0;

            for (int r = 0; r < nr; ++r)
            {
                for (int c = 0; c < nc; ++c)
                {
                    if (grid[r][c] == 1)
                    {
                        ++numIslands;
                        grid[r][c] = 0; // mark as visited
                        Queue<int> neighbors = new Queue<int>();
                        neighbors.Enqueue(r * nc + c);
                        while (neighbors.Count > 0)
                        {
                            int id = neighbors.Dequeue();
                            int row = id / nc;
                            int col = id % nc;
                            if (row - 1 >= 0 && grid[row - 1][col] == 1)
                            {
                                neighbors.Enqueue((row - 1) * nc + col);
                                grid[row - 1][col] = 0;
                            }
                            if (row + 1 < nr && grid[row + 1][col] == 1)
                            {
                                neighbors.Enqueue((row + 1) * nc + col);
                                grid[row + 1][col] = 0;
                            }
                            if (col - 1 >= 0 && grid[row][col - 1] == 1)
                            {
                                neighbors.Enqueue(row * nc + col - 1);
                                grid[row][col - 1] = 0;
                            }
                            if (col + 1 < nc && grid[row][col + 1] == 1)
                            {
                                neighbors.Enqueue(row * nc + col + 1);
                                grid[row][col + 1] = 0;
                            }
                        }
                    }
                }
            }

            return numIslands;
        }

        private int NumIslandsDFS(int[][] grid)
        {
            int numberOfIslands = 0;
            for (int i = 0; i < grid.GetLength(0); i++)
            {
                for (int j = 0; j < grid.GetLength(1); j++)
                {
                    if (grid[i][j] == 1)
                    {
                        numberOfIslands++;
                        NumIslandsDFS(grid, i, j);
                    }
                }
            }
            return numberOfIslands;

        }

        private void NumIslandsDFS(int[][] grid, int r, int c)
        {
            if (r < 0 || r >= grid.GetLength(0) || c < 0 || c >= grid.GetLength(1) || grid[r][c] != 1) return;

            grid[r][c] = 0;

            NumIslandsDFS(grid, r - 1, c);
            NumIslandsDFS(grid, r, c + 1);
            NumIslandsDFS(grid, r, c - 1);
            NumIslandsDFS(grid, r + 1, c);

        }

        /*
        694. Number of Distinct Islands
        https://leetcode.com/problems/number-of-distinct-islands/description/
        */
        public int NumDistinctIslands(int[][] grid)
        {
            int numOFDistinctIslands = 0;
            if (grid.Length == 0) return numOFDistinctIslands;

            //1. BruteForce with DFS
            /*Its inefficient because the operation for determining whether or not an island is unique requires looping through every coordinate of every island discovered so far

            Time Complexity: O(M^2 * N^2).In the worst case, we would have a large grid, with many unique islands all of the same size, and the islands packed as closely together as possible. 
                            This would mean that for each island we discover, we'd be looping over the cells of all the other islands we've discovered so far. 
                            
            Space complexity: O(N⋅M).The seen set requires O(N⋅M) memory. Additionally, each cell with land requires O(1) space in the islands array.
            */
            numOFDistinctIslands = NumDistinctIslandsNaive(grid);

            //2. Hash By Local Coordinates with DFS
            /* 
            •	Time Complexity: O(M⋅N).
            •	Space complexity: O(M⋅N). The seen set is the biggest use of additional memory
            */

            numOFDistinctIslands = NumDistinctIslandsOptimal(grid);

            //3. Hash By Path Signature with DFS
            /* 
            •	Time Complexity: O(M⋅N).
            •	Space complexity: O(M⋅N). The seen set is the biggest use of additional memory
            */
            numOFDistinctIslands = NumDistinctIslandsOptimal2(grid);

            return numOFDistinctIslands;

        }

        private int NumDistinctIslandsOptimal2(int[][] grid)
        {
            if (grid == null || grid.Length == 0) return 0;
            HashSet<string> set = new HashSet<string>();

            for (int i = 0; i < grid.GetLength(0); i++)
            {
                for (int j = 0; j < grid.GetLength(1); j++)
                {
                    if (grid[i][j] == 1)
                    {
                        //START - X
                        // Outofbounds or Water - O
                        string path = ComputePath(grid, i, j, "X");
                        set.Add(path);
                    }
                }
            }

            return set.Count();

        }
        private string ComputePath(int[][] grid, int i, int j, string direction)
        {
            if (i < 0 || i >= grid.GetLength(0) || j < 0 || j >= grid.GetLength(1) || grid[i][j] == 0) return "O";

            grid[i][j] = 0;

            string left = ComputePath(grid, i, j - 1, "L");
            string right = ComputePath(grid, i, j + 1, "R");
            string up = ComputePath(grid, i - 1, j, "U");
            string down = ComputePath(grid, i + 1, j, "D");


            return direction + left + right + up + down;
        }

        public int NumDistinctIslandsOptimal(int[][] grid)
        {
            this.grid = grid;
            bool[][] seen = new bool[grid.Length][]; // Cells that have been explored. 

            for (int i = 0; i < grid.Length; i++)
            {
                seen[i] = new bool[grid[0].Length];
            }
            HashSet<HashSet<(int, int)>> islands = new HashSet<HashSet<(int, int)>>();
            for (int row = 0; row < grid.Length; row++)
            {
                for (int col = 0; col < grid[0].Length; col++)
                {
                    this.currentIslandSet = new HashSet<(int, int)>();
                    this.currRowOrigin = row;
                    this.currColOrigin = col;
                    NumDistinctIslandsOptimalDfs(row, col, seen);
                    if (currentIsland.Count > 0)
                    {
                        islands.Add(currentIslandSet);
                    }
                }
            }
            return islands.Count;
        }
        private void NumDistinctIslandsOptimalDfs(int row, int col, bool[][] seen)
        {
            if (row < 0 || row >= grid.Length || col < 0 || col >= grid[0].Length)
            {
                return;
            }
            if (grid[row][col] == 0 || seen[row][col])
            {
                return;
            }
            seen[row][col] = true;
            currentIslandSet.Add((row - currRowOrigin, col - currColOrigin));
            NumDistinctIslandsOptimalDfs(row + 1, col, seen);
            NumDistinctIslandsOptimalDfs(row - 1, col, seen);
            NumDistinctIslandsOptimalDfs(row, col + 1, seen);
            NumDistinctIslandsOptimalDfs(row, col - 1, seen);
        }

        private int currRowOrigin;
        private int currColOrigin;
        private List<List<int[]>> uniqueIslands = new List<List<int[]>>(); // All known unique islands.        
        private List<int[]> currentIsland = new List<int[]>(); // Current Island
        HashSet<(int, int)> currentIslandSet; // Current Island
        private int[][] grid; // Input grid

        private int NumDistinctIslandsNaive(int[][] grid)
        {
            this.grid = grid;
            bool[][] seen = new bool[grid.Length][]; // Cells that have been explored. 
            for (int i = 0; i < grid.Length; i++)
            {
                seen[i] = new bool[grid[0].Length];
            }
            for (int row = 0; row < grid.Length; row++)
            {
                for (int col = 0; col < grid[0].Length; col++)
                {
                    NumDistinctIslandsNaiveDfs(row, col, seen);
                    if (currentIsland.Count == 0)
                    {
                        continue;
                    }
                    // Translate the island we just found to the top left.
                    int minCol = grid[0].Length - 1;
                    for (int i = 0; i < currentIsland.Count; i++)
                    {
                        minCol = Math.Min(minCol, currentIsland[i][1]);
                    }
                    for (int j = 0; j < currentIsland.Count; j++)
                    {
                        currentIsland[j][0] -= row;
                        currentIsland[j][1] -= minCol;
                    }

                    // If this island is unique, add it to the list.
                    if (CurrentIslandUnique())
                    {
                        uniqueIslands.Add(new List<int[]>(currentIsland));
                    }
                    currentIsland = new List<int[]>();
                }

            }
            return uniqueIslands.Count;
        }

        void NumDistinctIslandsNaiveDfs(int row, int col, bool[][] seen)
        {
            if (row < 0 || col < 0 || row >= grid.Length || col >= grid[0].Length) return;
            if (seen[row][col] || grid[row][col] == 0) return;
            seen[row][col] = true;
            currentIsland.Add(new int[] { row, col });
            NumDistinctIslandsNaiveDfs(row + 1, col, seen);
            NumDistinctIslandsNaiveDfs(row - 1, col, seen);
            NumDistinctIslandsNaiveDfs(row, col + 1, seen);
            NumDistinctIslandsNaiveDfs(row, col - 1, seen);
        }

        private bool CurrentIslandUnique()
        {
            foreach (var otherIsland in uniqueIslands)
            {
                if (currentIsland.Count != otherIsland.Count)
                {
                    continue;
                }
                if (EqualIslands(currentIsland, otherIsland))
                {
                    return false;
                }
            }
            return true;
        }

        private bool EqualIslands(List<int[]> island1, List<int[]> island2)
        {
            for (int i = 0; i < island1.Count; i++)
            {
                if (island1[i][0] != island2[i][0] || island1[i][1] != island2[i][1])
                {
                    return false;
                }
            }
            return true;
        }

        /*
        711. Number of Distinct Islands II
        https://leetcode.com/problems/number-of-distinct-islands-ii/description/       
        
        */
        public int NumDistinctIslands2(int[][] matrix)
        {
            //1. No Rotation or Reflection Calculation But using Maths
            /*
            Time Complexity: dfs will take overall O(n * m). positions will be size of n*m, and we nest for loop it, which is (n * m) ^2
                          overall O(n * m) + O((n * m) ^ 2) -> O((n * m) ^ 2)
            Space Complexity: 
            */
            int numberOfDistinctIslands = NumDistinctIslandsWihtMaths(matrix);

            //2.using DFS +sorting+transpose/rotations to find canonical representation for each island 
            /*
            Time complexity:   O(mnlogm*n)
            Space complexity: O(m*n)
            */
            numberOfDistinctIslands = NumDistinctIslands2Optimal(matrix);

            return numberOfDistinctIslands;
        }

        private int NumDistinctIslandsWihtMaths(int[][] matrix)
        {
            HashSet<Dictionary<int, int>> allDistinctIslands = new HashSet<Dictionary<int, int>>();
            int numberOfRows = matrix.Length;
            int numberOfColumns = matrix[0].Length;

            for (int row = 0; row < numberOfRows; row++)
            {
                for (int column = 0; column < numberOfColumns; column++)
                {
                    if (matrix[row][column] == 1)
                    {
                        List<int[]> positions = new List<int[]>();
                        GetIsland(matrix, row, column, positions);
                        Dictionary<int, int> distanceCountMap = new Dictionary<int, int>();

                        for (int i = 0; i < positions.Count; i++)
                        {
                            for (int j = i + 1; j < positions.Count; j++)
                            {
                                int distance = (int)Math.Pow(positions[i][0] - positions[j][0], 2) + (int)Math.Pow(positions[i][1] - positions[j][1], 2);
                                if (distanceCountMap.ContainsKey(distance))
                                {
                                    distanceCountMap[distance]++;
                                }
                                else
                                {
                                    distanceCountMap[distance] = 1;
                                }
                            }
                        }
                        allDistinctIslands.Add(distanceCountMap);
                    }
                }
            }
            return allDistinctIslands.Count;
        }

        private void GetIsland(int[][] matrix, int row, int column, List<int[]> positions)
        {
            positions.Add(new int[] { row, column });
            matrix[row][column] = 0;

            foreach (int[] direction in directions)
            {
                int nextRow = row + direction[0];
                int nextColumn = column + direction[1];

                if (nextRow < 0 || nextRow >= matrix.Length || nextColumn < 0 || nextColumn >= matrix[0].Length || matrix[nextRow][nextColumn] == 0)
                {
                    continue;
                }
                GetIsland(matrix, nextRow, nextColumn, positions);
            }
        }
        private readonly int[][] directions = new int[][] { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 } };
        private readonly int[][] trans = new int[][] { new int[] { 1, 1 }, new int[] { 1, -1 }, new int[] { -1, 1 }, new int[] { -1, -1 } };
        public int NumDistinctIslands2Optimal(int[][] grid)
        {
            if (grid == null || grid.Length == 0 || grid[0].Length == 0) return 0;
            int rowCount = grid.Length, columnCount = grid[0].Length;
            HashSet<string> islands = new HashSet<string>();

            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    if (grid[i][j] == 1)
                    {
                        List<int[]> cells = new List<int[]>();
                        NumDistinctIslands2OptimalDfs(grid, i, j, cells);
                        string key = Normalize(cells);
                        islands.Add(key);
                    }
                }
            }
            return islands.Count;
        }
        private void NumDistinctIslands2OptimalDfs(int[][] grid, int i, int j, List<int[]> cells)
        {
            cells.Add(new int[] { i, j });
            grid[i][j] = -1;

            foreach (int[] direction in directions)
            {
                int x = i + direction[0];
                int y = j + direction[1];
                if (x >= 0 && x < grid.Length && y >= 0 && y < grid[0].Length && grid[x][y] == 1)
                    NumDistinctIslands2OptimalDfs(grid, x, y, cells);
            }
        }
        private string Normalize(List<int[]> cells)
        {
            List<string> forms = new List<string>();
            // generate the 8 different transformations
            // (x, y), (x, -y), (-x, y), (-x, -y)
            // (y, x), (-y, x), (y, -x), (-y, -x)
            foreach (int[] transformation in trans)
            {
                List<int[]> list1 = new List<int[]>();
                List<int[]> list2 = new List<int[]>();
                foreach (int[] cell in cells)
                {
                    list1.Add(new int[] { cell[0] * transformation[0], cell[1] * transformation[1] });
                    list2.Add(new int[] { cell[1] * transformation[1], cell[0] * transformation[0] });
                }
                forms.Add(GetKey(list1));
                forms.Add(GetKey(list2));
            }

            // sort the keys: take the first one as the representative key
            forms.Sort();
            return forms[0];
        }
        private string GetKey(List<int[]> cells)
        {
            // sort the cells before generating the key
            cells.Sort((a, b) =>
            {
                if (a[0] != b[0])
                {
                    return a[0] - b[0];
                }
                else
                {
                    return a[1] - b[1];
                }
            });

            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            int x = cells[0][0], y = cells[0][1];
            foreach (int[] cell in cells)
                sb.Append((cell[0] - x) + ":" + (cell[1] - y) + ":");

            return sb.ToString();
        }

        /*
        1905. Count Sub Islands
        https://leetcode.com/problems/count-sub-islands/description
        */
        public int CountSubIslands(int[][] grid1, int[][] grid2)
        {
            //1.Breadth-First Search (BFS)
            /*
            Let m and n represent the number of rows and columns, respectively.
        •	Time complexity: O(m∗n)
                We iterate on each grid cell and perform BFS to traverse all land cells of all the islands. Each land cell is only traversed once. In the worst case, we may traverse all cells of the grid.
                Thus, in the worst case time complexity will be O(m∗n).
        •	Space complexity: O(m∗n)
                We create an additional grid visited of size m∗n and push the land cells in the queue.
                Thus, in the worst case space complexity will be O(m∗n).

            */
            int countOfSubIslands = CountSubIslandsBFS(grid1, grid2);

            //2.Depth-First Search
            /*
            Let m and n represent the number of rows and columns, respectively.
        •	Time complexity: O(m∗n)
                We iterate on each grid cell and perform DFS to traverse all land cells of all the islands. Each land cell is only traversed once. In the worst case, we may traverse all cells of the grid.
                Thus, in the worst case time complexity will be O(m∗n).
        •	Space complexity: O(m∗n)
                We create an additional grid visited of size m∗n and push the land cells in the recursive stack.
                Thus, in the worst case space complexity will be O(m∗n).

            */

            countOfSubIslands = CountSubIslandsDFS(grid1, grid2);

            //3.Union-Find, or Disjoint Set Union (DSU)
            /*
            Let m and n represent the number of rows and columns, respectively.
        •	Time complexity: O(m∗n)
                We iterate on each land cell of the grid and perform union operations with its adjacent cells. In the worst case, we may traverse all cells of the grid.
                Thus, in the worst case time complexity will be O(m∗n).
        •	Space complexity: O(m∗n)
                We create an additional object uf and a bool array isSubIsland of size m∗n.
                Thus, in the worst case space complexity will be O(m∗n).

            */
            countOfSubIslands = CountSubIslandsUF(grid1, grid2);

            return countOfSubIslands;


        }


        private int CountSubIslandsDFS(int[][] grid1, int[][] grid2)
        {
            int totalRows = grid2.Length;
            int totalCols = grid2[0].Length;

            bool[][] visited = new bool[totalRows][];
            for (int i = 0; i < totalRows; i++)
            {
                visited[i] = new bool[totalCols];
            }

            int subIslandCounts = 0;

            // Iterate over each cell in 'grid2'.
            for (int x = 0; x < totalRows; ++x)
            {
                for (int y = 0; y < totalCols; ++y)
                {
                    // If the cell at position (x, y) in 'grid2' is not visited,
                    // is a land cell in 'grid2', and the island starting from this cell is a sub-island in 'grid1',
                    // then increment the count of sub-islands.
                    if (!visited[x][y] && IsCellLand(x, y, grid2))
                    {
                        visited[x][y] = true;
                        if (IsSubIslandDFS(x, y, grid1, grid2, visited))
                        {
                            subIslandCounts += 1;
                        }
                    }
                }
            }
            // Return total count of sub-islands.
            return subIslandCounts;


        }
        // Traverse all cells of island starting at position (x, y) in 'grid2',
        // and check if this island is a sub-island in 'grid1'.
        private bool IsSubIslandDFS(
            int x,
            int y,
            int[][] grid1,
            int[][] grid2,
            bool[][] visited
        )
        {
            int totalRows = grid2.Length;
            int totalCols = grid2[0].Length;
            // Traverse on all cells using the depth-first search method.
            bool isSubIsland = true;

            // If the current cell is not a land cell in 'grid1', then the current island can't be a sub-island.
            if (!IsCellLand(x, y, grid1))
            {
                isSubIsland = false;
            }

            // Traverse on all adjacent cells.
            foreach (int[] direction in directions)
            {
                int nextX = x + direction[0];
                int nextY = y + direction[1];
                // If the next cell is inside 'grid2', is not visited, and is a land cell,
                // then we traverse to the next cell.
                if (
                    nextX >= 0 &&
                    nextY >= 0 &&
                    nextX < totalRows &&
                    nextY < totalCols &&
                    !visited[nextX][nextY] &&
                    IsCellLand(nextX, nextY, grid2)
                )
                {
                    // Mark the next cell as visited.
                    visited[nextX][nextY] = true;
                    bool nextCellIsPartOfSubIsland = IsSubIslandDFS(
                        nextX,
                        nextY,
                        grid1,
                        grid2,
                        visited
                    );
                    isSubIsland = isSubIsland && nextCellIsPartOfSubIsland;
                }
            }
            return isSubIsland;
        }
        private int CountSubIslandsUF(int[][] grid1, int[][] grid2)
        {
            int totalRows = grid2.Length;
            int totalCols = grid2[0].Length;
            DSUArray uf = new DSUArray(totalRows * totalCols);
            // Traverse each land cell of 'grid2'.
            for (int x = 0; x < totalRows; ++x)
            {
                for (int y = 0; y < totalCols; ++y)
                {
                    if (IsCellLand(x, y, grid2))
                    {
                        // Union adjacent land cells with the current land cell.
                        foreach (int[] direction in directions)
                        {
                            int nextX = x + direction[0];
                            int nextY = y + direction[1];
                            if (
                                nextX >= 0 &&
                                nextY >= 0 &&
                                nextX < totalRows &&
                                nextY < totalCols &&
                                IsCellLand(nextX, nextY, grid2)
                            )
                            {
                                uf.Union(
                                    ConvertToIndex(x, y, totalCols),
                                    ConvertToIndex(nextX, nextY, totalCols)
                                );
                            }
                        }
                    }
                }
            }
            // Traverse 'grid2' land cells and mark that cell's root as not a sub-island
            // if the land cell is not present at the respective position in 'grid1'.
            bool[] isSubIsland = new bool[totalRows * totalCols];
            for (int i = 0; i < isSubIsland.Length; i++)
            {
                isSubIsland[i] = true;
            }
            for (int x = 0; x < totalRows; ++x)
            {
                for (int y = 0; y < totalCols; ++y)
                {
                    if (IsCellLand(x, y, grid2) && !IsCellLand(x, y, grid1))
                    {
                        int root = uf.Find(ConvertToIndex(x, y, totalCols));
                        isSubIsland[root] = false;
                    }
                }
            }
            // Count all the sub-islands.
            int subIslandCounts = 0;
            for (int x = 0; x < totalRows; ++x)
            {
                for (int y = 0; y < totalCols; ++y)
                {
                    if (IsCellLand(x, y, grid2))
                    {
                        int root = uf.Find(ConvertToIndex(x, y, totalCols));
                        if (isSubIsland[root])
                        {
                            subIslandCounts++;
                            // One cell can be the root of multiple land cells, so to
                            // avoid counting the same island multiple times, mark it as false.
                            isSubIsland[root] = false;
                        }
                    }
                }
            }

            return subIslandCounts;
        }
        private int ConvertToIndex(int x, int y, int totalCols)
        {
            return x * totalCols + y;
        }

        private int CountSubIslandsBFS(int[][] grid1, int[][] grid2)
        {
            int totalRows = grid2.Length;
            int totalCols = grid2[0].Length;

            bool[][] visited = new bool[totalRows][];
            for (int i = 0; i < totalRows; i++)
            {
                visited[i] = new bool[totalCols];
            }
            int subIslandCounts = 0;

            // Iterate on each cell in 'grid2'
            for (int x = 0; x < totalRows; ++x)
            {
                for (int y = 0; y < totalCols; ++y)
                {
                    // If cell at the position (x, y) in the 'grid2' is not visited,
                    // is a land cell in 'grid2', and the island
                    // starting from this cell is a sub-island in 'grid1', then we
                    // increment the count of sub-islands.
                    if (
                        !visited[x][y] &&
                        IsCellLand(x, y, grid2) &&
                        IsSubIsland(x, y, grid1, grid2, visited)
                    )
                    {
                        subIslandCounts += 1;
                    }
                }
            }
            // Return total count of sub-islands.
            return subIslandCounts;
        }
        // Helper method to check if the cell at the position (x, y) in the 'grid'
        // is a land cell.
        private bool IsCellLand(int x, int y, int[][] grid)
        {
            return grid[x][y] == 1;
        }
        // Traverse all cells of island starting at position (x, y) in 'grid2',
        // and check this island is a sub-island in 'grid1'.
        private bool IsSubIsland(
            int x,
            int y,
            int[][] grid1,
            int[][] grid2,
            bool[][] visited
        )
        {
            int totalRows = grid2.Length;
            int totalCols = grid2[0].Length;

            bool isSubIsland = true;

            Queue<int[]> pendingCells = new Queue<int[]>();
            // Push the starting cell in the queue and mark it as visited.
            pendingCells.Enqueue(new int[] { x, y });
            visited[x][y] = true;

            // Traverse on all cells using the breadth-first search method.
            while (pendingCells.Count > 0)
            {
                int[] currentCell = pendingCells.Dequeue();
                int currentX = currentCell[0];
                int currentY = currentCell[1];

                // If the current position cell is not a land cell in 'grid1',
                // then the current island can't be a sub-island.
                if (!IsCellLand(currentX, currentY, grid1))
                {
                    isSubIsland = false;
                }

                foreach (int[] direction in directions)
                {
                    int nextX = currentX + direction[0];
                    int nextY = currentY + direction[1];
                    // If the next cell is inside 'grid2', is never visited and
                    // is a land cell, then we traverse to the next cell.
                    if (
                        nextX >= 0 &&
                        nextY >= 0 &&
                        nextX < totalRows &&
                        nextY < totalCols &&
                        !visited[nextX][nextY] &&
                        IsCellLand(nextX, nextY, grid2)
                    )
                    {
                        // Push the next cell in the queue and mark it as visited.
                        pendingCells.Enqueue(new int[] { nextX, nextY });
                        visited[nextX][nextY] = true;
                    }
                }
            }

            return isSubIsland;
        }
        /*
        695. Max Area of Island
        https://leetcode.com/problems/max-area-of-island/description/	

        */
        public int MaxAreaOfIsland(int[][] grid)
        {
            //1.Depth-First Search (Iterative) 
            /*
            Time Complexity: O(R∗C), where R is the number of rows in the given grid, and C is the number of columns. We visit every square once.
            Space complexity: O(R∗C), the space used by seen to keep track of visited squares and the space used by stack.
            */
            int maxAreaOfIsland = MaxAreaOfIslandDFSRec(grid);

            //2.Depth-First Search (Recursive)
            /*
            Time Complexity: O(R∗C), where R is the number of rows in the given grid, and C is the number of columns. We visit every square once.
            Space complexity: O(R∗C), the space used by seen to keep track of visited squares and the space used by the call stack during our recursion.

            */

            maxAreaOfIsland = CountSubIslandsDFSIterative(grid);


            return maxAreaOfIsland;


        }

        private int CountSubIslandsDFSIterative(int[][] grid)
        {
            bool[][] seen = new bool[grid.Length][];
            for (int i = 0; i < grid.Length; i++)
            {
                seen[i] = new bool[grid[i].Length];
            }
            int[] rowDirections = new int[] { 1, -1, 0, 0 };
            int[] columnDirections = new int[] { 0, 0, 1, -1 };

            int maxArea = 0;
            for (int row = 0; row < grid.Length; row++)
            {
                for (int column = 0; column < grid[0].Length; column++)
                {
                    if (grid[row][column] == 1 && !seen[row][column])
                    {
                        int currentArea = 0;
                        Stack<int[]> stack = new Stack<int[]>();
                        stack.Push(new int[] { row, column });
                        seen[row][column] = true;
                        while (stack.Count > 0)
                        {
                            int[] node = stack.Pop();
                            int currentRow = node[0], currentColumn = node[1];
                            currentArea++;
                            for (int direction = 0; direction < 4; direction++)
                            {
                                int newRow = currentRow + rowDirections[direction];
                                int newColumn = currentColumn + columnDirections[direction];
                                if (0 <= newRow && newRow < grid.Length &&
                                        0 <= newColumn && newColumn < grid[0].Length &&
                                        grid[newRow][newColumn] == 1 && !seen[newRow][newColumn])
                                {
                                    stack.Push(new int[] { newRow, newColumn });
                                    seen[newRow][newColumn] = true;
                                }
                            }
                        }
                        maxArea = Math.Max(maxArea, currentArea);
                    }
                }
            }
            return maxArea;
        }
        private int MaxAreaOfIslandDFSRec(int[][] grid)
        {
            this.grid = grid;
            bool[][] seen = new bool[grid.Length][];
            int maxAreaOfIsland = 0;
            for (int r = 0; r < grid.Length; r++)
            {
                for (int c = 0; c < grid[0].Length; c++)
                {
                    maxAreaOfIsland = Math.Max(maxAreaOfIsland, MaxAreaOfIslandDFSRec(r, c, seen));
                }
            }
            return maxAreaOfIsland;
        }

        private int MaxAreaOfIslandDFSRec(int r, int c, bool[][] seen)
        {
            if (r < 0 || r >= grid.Length || c < 0 || c >= grid[0].Length ||
                    seen[r][c] || grid[r][c] == 0)
                return 0;
            seen[r][c] = true;
            return (1 + MaxAreaOfIslandDFSRec(r + 1, c, seen) + MaxAreaOfIslandDFSRec(r - 1, c, seen)
                      + MaxAreaOfIslandDFSRec(r, c - 1, seen) + MaxAreaOfIslandDFSRec(r, c + 1, seen));
        }

        /*
        463. Island Perimeter
        https://leetcode.com/problems/island-perimeter/description/

        */
        public class IslandPerimeterSol
        {
            /*
            Approach 1: Simple Counting
            Complexity Analysis
•	Time complexity : O(mn) where m is the number of rows of the grid and n is
the number of columns of the grid. Since two for loops go through all
the cells on the grid, for a two-dimensional grid of size m×n, the algorithm
would have to check mn cells.
•	Space complexity : O(1). Only the result variable is updated and there is
no other space requirement.

            */
            public int SimpleCouting(int[][] grid)
            {

                int rows = grid.Length;
                int cols = grid[0].Length;

                int up, down, left, right;
                int result = 0;

                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        if (grid[r][c] == 1)
                        {
                            if (r == 0) { up = 0; }
                            else { up = grid[r - 1][c]; }

                            if (c == 0) { left = 0; }
                            else { left = grid[r][c - 1]; }

                            if (r == rows - 1) { down = 0; }
                            else { down = grid[r + 1][c]; }

                            if (c == cols - 1) { right = 0; }
                            else { right = grid[r][c + 1]; }

                            result += 4 - (up + left + right + down);
                        }
                    }
                }

                return result;
            }
            /*
Approach 2: Better Counting

            Complexity Analysis
•	Time complexity : O(mn) where m is the number of rows of the grid and n is
the number of columns of the grid. Since two for loops go through all
the cells on the grid, for a two-dimensional grid of size m×n, the algorithm
would have to check mn cells.
•	Space complexity : O(1). Only the result variable is updated and there is
no other space requirement.
            
            */
            public int islandPerimeter(int[][] grid)
            {
                int rows = grid.Length;
                int cols = grid[0].Length;

                int result = 0;
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        if (grid[r][c] == 1)
                        {
                            result += 4;

                            if (r > 0 && grid[r - 1][c] == 1)
                            {
                                result -= 2;
                            }

                            if (c > 0 && grid[r][c - 1] == 1)
                            {
                                result -= 2;
                            }
                        }
                    }
                }

                return result;
            }
        }

        /*
        827. Making A Large Island
        https://leetcode.com/problems/making-a-large-island/description/
        */
        public class LargestIslandSol
        {
            int[] directionRows = new int[] { -1, 0, 1, 0 };
            int[] directionColumns = new int[] { 0, -1, 0, 1 };

            /*
            Approach 1: Naive Depth First Search
           Complexity Analysis
•	Time Complexity: O(N^4), where N is the length and width of the grid.
•	Space Complexity: O(N^2), the additional space used in the depth first search by stack and seen.
 
            */
            public int NaiveDFS(int[][] grid)
            {
                int size = grid.Length;

                int maximumArea = 0;
                bool hasZero = false;
                for (int row = 0; row < size; ++row)
                    for (int column = 0; column < size; ++column)
                        if (grid[row][column] == 0)
                        {
                            hasZero = true;
                            grid[row][column] = 1;
                            maximumArea = Math.Max(maximumArea, Check(grid, row, column));
                            grid[row][column] = 0;
                        }

                return hasZero ? maximumArea : size * size;
            }

            private int Check(int[][] grid, int initialRow, int initialColumn)
            {
                int size = grid.Length;
                Stack<int> positionStack = new Stack<int>();
                HashSet<int> visitedPositions = new HashSet<int>();
                positionStack.Push(initialRow * size + initialColumn);
                visitedPositions.Add(initialRow * size + initialColumn);

                while (positionStack.Count > 0)
                {
                    int code = positionStack.Pop();
                    int row = code / size, column = code % size;
                    for (int direction = 0; direction < 4; ++direction)
                    {
                        int newRow = row + directionRows[direction], newColumn = column + directionColumns[direction];
                        if (!visitedPositions.Contains(newRow * size + newColumn) &&
                            0 <= newRow && newRow < size &&
                            0 <= newColumn && newColumn < size && grid[newRow][newColumn] == 1)
                        {
                            positionStack.Push(newRow * size + newColumn);
                            visitedPositions.Add(newRow * size + newColumn);
                        }
                    }
                }

                return visitedPositions.Count;
            }
            /*
            Approach #2: Component Sizes
            Complexity Analysis
•	Time Complexity: O(N^2), where N is the length and width of the grid.
•	Space Complexity: O(N^2), the additional space used in the depth first search by area.

            */
            private int[,] grid;
            private int gridSize;

            public int UsingComponentSizes(int[,] grid)
            {
                this.grid = grid;
                gridSize = grid.GetLength(0);

                int index = 2;
                int[] area = new int[gridSize * gridSize + 2];
                for (int row = 0; row < gridSize; ++row)
                    for (int column = 0; column < gridSize; ++column)
                        if (grid[row, column] == 1)
                            area[index] = Dfs(row, column, index++);

                int maxArea = 0;
                foreach (int x in area) maxArea = Math.Max(maxArea, x);
                for (int row = 0; row < gridSize; ++row)
                    for (int column = 0; column < gridSize; ++column)
                        if (grid[row, column] == 0)
                        {
                            HashSet<int> seen = new HashSet<int>();
                            foreach (int move in GetNeighbors(row, column))
                                if (grid[move / gridSize, move % gridSize] > 1)
                                    seen.Add(grid[move / gridSize, move % gridSize]);

                            int bonus = 1;
                            foreach (int i in seen) bonus += area[i];
                            maxArea = Math.Max(maxArea, bonus);
                        }

                return maxArea;
            }

            private int Dfs(int row, int column, int index)
            {
                int areaCount = 1;
                grid[row, column] = index;
                foreach (int move in GetNeighbors(row, column))
                {
                    if (grid[move / gridSize, move % gridSize] == 1)
                    {
                        grid[move / gridSize, move % gridSize] = index;
                        areaCount += Dfs(move / gridSize, move % gridSize, index);
                    }
                }

                return areaCount;
            }

            private List<int> GetNeighbors(int row, int column)
            {
                List<int> neighbors = new List<int>();
                for (int k = 0; k < 4; ++k)
                {
                    int newRow = row + directionRows[k];
                    int newColumn = column + directionColumns[k];
                    if (0 <= newRow && newRow < gridSize && 0 <= newColumn && newColumn < gridSize)
                        neighbors.Add(newRow * gridSize + newColumn);
                }

                return neighbors;
            }

        }

        /* 1254. Number of Closed Islands
        https://leetcode.com/problems/number-of-closed-islands/description/
         */
        public class ClosedIslandSol
        {
            /*
            Approach 1: Breadth First Search
Complexity Analysis
Here, m and n are the number of rows and columns in the given grid.
•	Time complexity: O(m⋅n)
o	Initializing the visit array takes O(m⋅n) time.
o	We iterate over all the cells and find unvisited land cells to perform BFS traversal from those. This takes O(m⋅n) time.
o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can be pushed once, leading to O(m⋅n) operations for m⋅n nodes. We iterate over all the neighbors of each node that is popped out of the queue. So for every node, we would iterate four times to iterate over the neighbors, resulting in O(4⋅m⋅n)=O(m⋅n) operations total for all the nodes.
•	Space complexity: O(m⋅n)
o	The visit array takes O(m⋅n) space.
o	The BFS queue takes O(m⋅n) space in the worst-case because each node is added once.

            */
            public int BFS(int[][] grid)
            {
                int rows = grid.Length;
                int columns = grid[0].Length;
                bool[][] visited = new bool[rows][];
                for (int i = 0; i < rows; i++)
                {
                    visited[i] = new bool[columns];
                }
                int count = 0;
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 0; column < columns; column++)
                    {
                        if (grid[row][column] == 0 && !visited[row][column] && Bfs(row, column, rows, columns, grid, visited))
                        {
                            count++;
                        }
                    }
                }
                return count;
            }

            private bool Bfs(int x, int y, int rows, int columns, int[][] grid, bool[][] visited)
            {
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { x, y });
                visited[x][y] = true;
                bool isClosed = true;

                int[] directionX = { 0, 1, 0, -1 };
                int[] directionY = { -1, 0, 1, 0 };

                while (queue.Count > 0)
                {
                    int[] temp = queue.Dequeue();
                    x = temp[0];
                    y = temp[1];

                    for (int i = 0; i < 4; i++)
                    {
                        int newRow = x + directionX[i];
                        int newColumn = y + directionY[i];
                        if (newRow < 0 || newRow >= rows || newColumn < 0 || newColumn >= columns)
                        {
                            // (x, y) is a boundary cell.
                            isClosed = false;
                        }
                        else if (grid[newRow][newColumn] == 0 && !visited[newRow][newColumn])
                        {
                            queue.Enqueue(new int[] { newRow, newColumn });
                            visited[newRow][newColumn] = true;
                        }
                    }
                }

                return isClosed;
            }

            /*
            Approach 2: Depth First Search
           Complexity Analysis
Here, m and n are the number of rows and columns in the given grid.
•	Time complexity: O(m⋅n)
o	Initializing the visit array takes O(m⋅n) time.
o	We iterate over all the cells and find unvisited land cells to perform DFS traversal from those. This takes O(m⋅n) time.
o	The dfs function visits each node once, leading to O(m⋅n) operations for m⋅n nodes. We iterate over all the neighbors of each node that is popped out of the queue. So for every node, we would iterate four times to iterate over the neighbors, resulting in O(4⋅m⋅n)=O(m⋅n) operations total for all the nodes.
•	Space complexity: O(m⋅n)
o	The visit array takes O(m⋅n) space.
o	The recursion stack used by dfs can have no more than O(m⋅n) elements in the worst-case scenario. It would take up O(m⋅n) space in that case.
 
            */
            public int DFS(int[][] grid)
            {
                int rowCount = grid.Length;
                int columnCount = grid[0].Length;
                bool[][] visited = new bool[rowCount][];
                for (int i = 0; i < rowCount; i++)
                {
                    visited[i] = new bool[columnCount];
                }
                int closedIslandCount = 0;
                for (int i = 0; i < rowCount; i++)
                {
                    for (int j = 0; j < columnCount; j++)
                    {
                        if (grid[i][j] == 0 && !visited[i][j] && Dfs(i, j, rowCount, columnCount, grid, visited))
                        {
                            closedIslandCount++;
                        }
                    }
                }
                return closedIslandCount;
            }

            public bool Dfs(int x, int y, int rowCount, int columnCount, int[][] grid, bool[][] visited)
            {
                if (x < 0 || x >= grid.Length || y < 0 || y >= grid[0].Length)
                {
                    return false;
                }
                if (grid[x][y] == 1 || visited[x][y])
                {
                    return true;
                }

                visited[x][y] = true;
                bool isClosed = true;
                int[] directionX = { 0, 1, 0, -1 };
                int[] directionY = { -1, 0, 1, 0 };

                for (int i = 0; i < 4; i++)
                {
                    int newRow = x + directionX[i];
                    int newColumn = y + directionY[i];
                    if (!Dfs(newRow, newColumn, rowCount, columnCount, grid, visited))
                    {
                        isClosed = false;
                    }
                }

                return isClosed;
            }
        }
        //https://www.algoexpert.io/questions/remove-islands
        // O(wh) time | O(wh) space - where w and h
        // are the width and height of the input matrix
        public int[][] RemoveIslands1(int[][] matrix)
        {
            bool[,] onesConnectedToBorder = new bool[matrix.Length, matrix[0].Length];
            for (int i = 0; i < matrix.Length; i++)
            {
                onesConnectedToBorder[i, matrix[0].Length - 1] = false;
            }

            // Find all the 1s that are not islands
            for (int row = 0; row < matrix.Length; row++)
            {
                for (int col = 0; col < matrix[row].Length; col++)
                {
                    bool rowIsBorder = row == 0 || row == matrix.Length - 1;
                    bool colIsBorder = col == 0 || col == matrix[row].Length - 1;
                    bool isBorder = rowIsBorder || colIsBorder;

                    if (!isBorder)
                    {
                        continue;
                    }

                    if (matrix[row][col] != 1)
                    {
                        continue;
                    }

                    FindOnesConnectedToBorder(matrix, row, col, onesConnectedToBorder);
                }
            }

            for (int row = 1; row < matrix.Length - 1; row++)
            {
                for (int col = 1; col < matrix[row].Length - 1; col++)
                {
                    if (onesConnectedToBorder[row, col])
                    {
                        continue;
                    }
                    matrix[row][col] = 0;
                }
            }

            return matrix;
        }
        public void FindOnesConnectedToBorder(
          int[][] matrix, int startRow, int startCol, bool[,] onesConnectedToBorder
        )
        {
            Stack<Tuple<int, int>> stack = new Stack<Tuple<int, int>>();
            stack.Push(new Tuple<int, int>(startRow, startCol));

            while (stack.Count > 0)
            {
                var currentPosition = stack.Pop();
                int currentRow = currentPosition.Item1;
                int currentCol = currentPosition.Item2;

                bool alreadyVisited = onesConnectedToBorder[currentRow, currentCol];
                if (alreadyVisited)
                {
                    continue;
                }

                onesConnectedToBorder[currentRow, currentCol] = true;

                var neighbors = GetNeighbors(matrix, currentRow, currentCol);
                foreach (var neighbor in neighbors)
                {
                    int row = neighbor.Item1;
                    int col = neighbor.Item2;

                    if (matrix[row][col] != 1)
                    {
                        continue;
                    }
                    stack.Push(neighbor);
                }
            }
        }

        public List<Tuple<int, int>> GetNeighbors(int[][] matrix, int row, int col)
        {
            int numRows = matrix.Length;
            int numCols = matrix[row].Length;
            List<Tuple<int, int>> neighbors = new List<Tuple<int, int>>();

            if (row - 1 >= 0)
            {
                neighbors.Add(new Tuple<int, int>(row - 1, col));  // UP
            }
            if (row + 1 < numRows)
            {
                neighbors.Add(new Tuple<int, int>(row + 1, col));  // DOWN
            }
            if (col - 1 >= 0)
            {
                neighbors.Add(new Tuple<int, int>(row, col - 1));  // LEFT
            }
            if (col + 1 < numCols)
            {
                neighbors.Add(new Tuple<int, int>(row, col + 1));  // RIGHT
            }
            return neighbors;
        }
        // O(wh) time | O(wh) space - where w and h
        // are the width and height of the input matrix
        public int[][] RemoveIslands2(int[][] matrix)
        {
            for (int row = 0; row < matrix.Length; row++)
            {
                for (int col = 0; col < matrix[row].Length; col++)
                {
                    bool rowIsBorder = row == 0 || row == matrix.Length - 1;
                    bool colIsBorder = col == 0 || col == matrix[row].Length - 1;
                    bool isBorder = rowIsBorder || colIsBorder;

                    if (!isBorder)
                    {
                        continue;
                    }

                    if (matrix[row][col] != 1)
                    {
                        continue;
                    }

                    changeOnesConnectedToBorderToTwos(matrix, row, col);
                }
            }

            for (int row = 0; row < matrix.Length; row++)
            {
                for (int col = 0; col < matrix[row].Length; col++)
                {
                    int color = matrix[row][col];
                    if (color == 1)
                    {
                        matrix[row][col] = 0;
                    }
                    else if (color == 2)
                    {
                        matrix[row][col] = 1;
                    }
                }
            }

            return matrix;
        }
        public void changeOnesConnectedToBorderToTwos(
      int[][] matrix, int startRow, int startCol
    )
        {
            Stack<Tuple<int, int>> stack = new Stack<Tuple<int, int>>();
            stack.Push(new Tuple<int, int>(startRow, startCol));

            while (stack.Count > 0)
            {
                var currentPosition = stack.Pop();
                int currentRow = currentPosition.Item1;
                int currentCol = currentPosition.Item2;

                matrix[currentRow][currentCol] = 2;

                var neighbors = GetNeighbors(matrix, currentRow, currentCol);
                foreach (var neighbor in neighbors)
                {
                    int row = neighbor.Item1;
                    int col = neighbor.Item2;

                    if (matrix[row][col] != 1)
                    {
                        continue;
                    }
                    stack.Push(neighbor);
                }
            }
        }

        ////https://www.algoexpert.io/questions/largest-island

        //1.  O(w^2 * h^2) time | O(w * h) space - where w is the width of the matrix,
        // and h is the height of the matrix
        public int LargestIslandNaive(int[][] matrix)
        {
            int maxSize = 0;
            for (int row = 0; row < matrix.Length; row++)
            {
                for (int col = 0; col < matrix[row].Length; col++)
                {
                    if (matrix[row][col] == 0)
                    {
                        continue;
                    }
                    maxSize = Math.Max(maxSize, getSizeFromNode(row, col, matrix));
                }
            }

            return maxSize;
        }

        private int getSizeFromNode(int row, int col, int[][] matrix)
        {
            int size = 1;
            bool[,] visited = new bool[matrix.Length, matrix[0].Length];
            Stack<List<int>> nodesToExplore = new Stack<List<int>>();
            getLandNeighbors(row, col, matrix, nodesToExplore);

            while (nodesToExplore.Count > 0)
            {
                List<int> currentNode = nodesToExplore.Pop();
                int currentRow = currentNode[0];
                int currentCol = currentNode[1];

                if (visited[currentRow, currentCol])
                {
                    continue;
                }
                visited[currentRow, currentCol] = true;

                size++;
                getLandNeighbors(currentRow, currentCol, matrix, nodesToExplore);
            }
            return size;
        }

        private void getLandNeighbors(
          int row, int col, int[][] matrix, Stack<List<int>> nodesToExplore
        )
        {
            if (row > 0 && matrix[row - 1][col] != 1)
            {
                nodesToExplore.Push(new List<int> { row - 1, col });
            }
            if (row < matrix.Length - 1 && matrix[row + 1][col] != 1)
            {
                nodesToExplore.Push(new List<int> { row + 1, col });
            }
            if (col > 0 && matrix[row][col - 1] != 1)
            {
                nodesToExplore.Push(new List<int> { row, col - 1 });
            }
            if (col < matrix[0].Length - 1 && matrix[row][col + 1] != 1)
            {
                nodesToExplore.Push(new List<int> { row, col + 1 });
            }
        }

        //2. O(w * h) time | O(w * h) space - where w is the width of the matrix, and
        // h is the height of the matrix
        public int LargestIslandOptimal(int[][] matrix)
        {
            List<int> islandSizes = new List<int>();
            // islandNumber starts at 2 to avoid overwriting existing 0s and 1s
            int islandNumber = 2;
            for (int row = 0; row < matrix.Length; row++)
            {
                for (int col = 0; col < matrix[row].Length; col++)
                {
                    if (matrix[row][col] == 0)
                    {
                        islandSizes.Add(getSizeFromNode(row, col, matrix, islandNumber));
                        islandNumber++;
                    }
                }
            }

            int maxSize = 0;
            for (int row = 0; row < matrix.Length; row++)
            {
                for (int col = 0; col < matrix[row].Length; col++)
                {
                    if (matrix[row][col] != 1)
                    {
                        continue;
                    }

                    List<List<int>> landNeighbors = getLandNeighbors(row, col, matrix);
                    HashSet<int> islands = new HashSet<int>();
                    foreach (var neighbor in landNeighbors)
                    {
                        islands.Add(matrix[neighbor[0]][neighbor[1]]);
                    }

                    int size = 1;
                    foreach (var island in islands)
                    {
                        size += islandSizes[island - 2];
                    }
                    maxSize = Math.Max(maxSize, size);
                }
            }
            return maxSize;
        }
        private int getSizeFromNode(int row, int col, int[][] matrix, int islandNumber)
        {
            int size = 0;
            Stack<List<int>> nodesToExplore = new Stack<List<int>>();
            nodesToExplore.Push(new List<int> { row, col });

            while (nodesToExplore.Count > 0)
            {
                List<int> currentNode = nodesToExplore.Pop();
                int currentRow = currentNode[0];
                int currentCol = currentNode[1];

                if (matrix[currentRow][currentCol] != 0)
                {
                    continue;
                }
                matrix[currentRow][currentCol] = islandNumber;

                size++;
                List<List<int>> newNeighbors =
                  getLandNeighbors(currentRow, currentCol, matrix);
                foreach (var neighbor in newNeighbors)
                {
                    nodesToExplore.Push(neighbor);
                }
            }
            return size;
        }

        private List<List<int>> getLandNeighbors(int row, int col, int[][] matrix)
        {
            List<List<int>> landNeighbors = new List<List<int>>();
            if (row > 0 && matrix[row - 1][col] != 1)
            {
                landNeighbors.Add(new List<int> { row - 1, col });
            }
            if (row < matrix.Length - 1 && matrix[row + 1][col] != 1)
            {
                landNeighbors.Add(new List<int> { row + 1, col });
            }
            if (col > 0 && matrix[row][col - 1] != 1)
            {
                landNeighbors.Add(new List<int> { row, col - 1 });
            }
            if (col < matrix[0].Length - 1 && matrix[row][col + 1] != 1)
            {
                landNeighbors.Add(new List<int> { row, col + 1 });
            }
            return landNeighbors;
        }

    }
}