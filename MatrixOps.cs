using System.ComponentModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;
using System.Security.AccessControl;
using System.Data;
using System.Text.RegularExpressions;
using System.Runtime.CompilerServices;
using System.Security.Principal;

namespace AlgoDSPlay
{
    public class MatrixOps
    {




        //https://www.algoexpert.io/questions/waterfall-streams
        public double[] WaterfallStreams(double[][] array, int source)
        {
            //T:O(W^2*h) | S:O(W)
            double[] rowAbove = array[0];

            rowAbove[source] = -1; // -1 used to represent water since 1 is used for a block

            for (int row = 1; row < array.Length; row++)
            {
                double[] currentRow = array[row];

                for (int col = 0; col < rowAbove.Length; col++)
                {
                    double valueAbove = rowAbove[col];

                    bool hasWaterAbove = valueAbove < 0;
                    bool hasBlock = currentRow[col] == 1.0;

                    if (!hasWaterAbove) continue;

                    if (!hasBlock)
                    {
                        currentRow[col] += valueAbove;
                        continue;
                    }

                    double splitWatter = valueAbove / 2;

                    int rightColIdx = col;

                    while (rightColIdx + 1 < rowAbove.Length)
                    {
                        rightColIdx += 1;

                        if (rowAbove[rightColIdx] == 1.0)
                        {
                            break;
                        }

                        if (currentRow[rightColIdx] != 1.0)
                        {

                            currentRow[rightColIdx] += splitWatter;
                            break;
                        }
                    }

                    int leftColIdx = col;
                    while (leftColIdx - 1 >= 0)
                    {
                        leftColIdx -= 1;

                        if (rowAbove[leftColIdx] == 1.0) break;

                        if (currentRow[leftColIdx] != 1.0)
                        {
                            currentRow[leftColIdx] += splitWatter;
                            break;
                        }
                    }

                }
                rowAbove = currentRow;

            }

            double[] finalPercentages = new double[rowAbove.Length];
            for (int idx = 0; idx < rowAbove.Length; idx++)
            {
                double num = rowAbove[idx];
                if (num == 0)
                    finalPercentages[idx] = num;
                else
                    finalPercentages[idx] = (num * -100);
            }

            return finalPercentages;

        }
        //https://www.algoexpert.io/questions/a*-algorithm 
        public static int[,] FindShortestPathUsingAStarAlgo(int startRow, int startCol, int endRow, int endCol, int[,] graph)
        {
            //T:O(w*h*log(w*h)) | S:O(w*h)
            List<List<NodeExt>> nodes = InitializeNodes(graph);
            NodeExt startNode = nodes[startRow][startCol];
            NodeExt endNode = nodes[endRow][endCol];

            startNode.distanceFromStart = 0;
            startNode.estimatedDistanceToEnd = CalculateManhattanDistance(startNode, endNode);

            List<NodeExt> nodesToVisitList = new List<NodeExt>();
            nodesToVisitList.Add(startNode);

            MinHeapForAStarAlgo nodesToVisit = new MinHeapForAStarAlgo(nodesToVisitList);

            while (!nodesToVisit.IsEmpty())
            {
                NodeExt currentMinDistanceNode = nodesToVisit.Remove();
                if (currentMinDistanceNode == endNode) break;

                List<NodeExt> neighbors = GetNeighbors(currentMinDistanceNode, nodes);
                foreach (var neighbor in neighbors)
                {
                    if (neighbor.Value == 1) continue;

                    int tentativeDistanceToNeighbor = currentMinDistanceNode.distanceFromStart + 1;
                    if (tentativeDistanceToNeighbor >= neighbor.distanceFromStart)
                        continue;
                    neighbor.CameFrom = currentMinDistanceNode;
                    neighbor.distanceFromStart = tentativeDistanceToNeighbor;
                    neighbor.estimatedDistanceToEnd = tentativeDistanceToNeighbor + CalculateManhattanDistance(neighbor, endNode);

                    if (!nodesToVisit.ContainsNode(neighbor))
                        nodesToVisit.Insert(neighbor);
                    else
                    {
                        nodesToVisit.Update(neighbor);
                    }

                }

            }
            return ReconstructPath(endNode);
        }

        private static List<NodeExt> GetNeighbors(NodeExt node, List<List<NodeExt>> nodes)
        {
            List<NodeExt> neighbors = new List<NodeExt>();

            int numRows = nodes.Count();
            int numCols = nodes[0].Count();

            int row = node.Row;
            int col = node.Col;

            if (row < numRows - 1)//DOWN
                neighbors.Add(nodes[row + 1][col]);
            if (row > 0)//UP
                neighbors.Add(nodes[row - 1][col]);
            if (col < numCols - 1)//RIGHT
                neighbors.Add(nodes[row][col + 1]);
            if (col > 0)//LEFT
                neighbors.Add(nodes[row][col - 1]);

            return neighbors;
        }

        private static int[,] ReconstructPath(NodeExt endNode)
        {
            if (endNode.CameFrom == null)
                return new int[,] { };

            List<List<int>> path = new List<List<int>>();

            NodeExt currentNode = endNode;
            while (currentNode != null)
            {
                List<int> nodeData = new List<int>();
                nodeData.Add(currentNode.Row);
                nodeData.Add(currentNode.Col);
                path.Add(nodeData);
                currentNode = currentNode.CameFrom;
            }
            int[,] result = new int[path.Count, 2];
            for (int i = 0; i < path.Count; i++)
            {
                List<int> lst = path[path.Count - 1 - i];
                result[i, 0] = lst[0];
                result[i, 1] = lst[1];
            }
            return result;
        }

        private static int CalculateManhattanDistance(NodeExt currentNode, NodeExt endNode)
        {
            int currentRow = currentNode.Row;
            int currentCol = currentNode.Col;
            int endRow = endNode.Row;
            int endCol = endNode.Col;

            return Math.Abs(currentRow - endRow) + Math.Abs(currentCol - endCol);
        }

        private static List<List<NodeExt>> InitializeNodes(int[,] graph)
        {
            List<List<NodeExt>> nodes = new List<List<NodeExt>>();
            for (int row = 0; row < graph.GetLength(0); row++)
            {
                List<NodeExt> nodeList = new List<NodeExt>();
                for (int col = 0; col < graph.GetLength(1); col++)
                {
                    nodeList.Add(new NodeExt(row, col, graph[row, col]));
                }
                nodes.Add(nodeList);

            }
            return nodes;

        }
        //https://www.algoexpert.io/questions/river-sizes
        //Islands
        public static List<int> FindRiverSizes(int[,] matrix)
        {
            //T:O(w*h) | S:O(w*h)
            List<int> sizes = new List<int>();
            bool[,] visited = new bool[matrix.GetLength(0), matrix.GetLength(1)];
            for (int row = 0; row < matrix.GetLength(0); row++)
            {

                for (int col = 0; col < matrix.GetLength(1); col++)
                {
                    if (visited[row, col])
                        continue;
                    TraverseNode(row, col, visited, matrix, sizes);
                }
            }
            return sizes;
        }

        private static void TraverseNode(int row, int col, bool[,] visited, int[,] matrix, List<int> sizes)
        {
            int currentRiverSize = 0;

            Stack<int[]> nodesToExplore = new Stack<int[]>();
            nodesToExplore.Push(new int[] { row, col });
            while (nodesToExplore.Count > 0)
            {
                int[] currentNode = nodesToExplore.Pop();
                row = currentNode[0];
                col = currentNode[1];
                if (visited[row, col])
                    continue;

                visited[row, col] = true;
                if (matrix[row, col] == 0)
                    continue;
                currentRiverSize++;

                List<int[]> unVisitedNeighbors = GetUnVisitedNeighbors(row, col, matrix, visited);
                foreach (int[] unVisitedNeigh in unVisitedNeighbors)
                {
                    nodesToExplore.Push(unVisitedNeigh);
                }
            }
            if (currentRiverSize > 0)
                sizes.Append(currentRiverSize);

        }

        private static List<int[]> GetUnVisitedNeighbors(int row, int col, int[,] matrix, bool[,] visited)
        {
            List<int[]> unVisitedNeighbors = new List<int[]>();

            if (row > 0 && !visited[row - 1, col])
                unVisitedNeighbors.Add(new int[] { row - 1, col });

            if (row < matrix.GetLength(0) - 1 && !visited[row + 1, col])
                unVisitedNeighbors.Add(new int[] { row + 1, col });

            if (col > 0 && !visited[row, col - 1])
                unVisitedNeighbors.Add(new int[] { row, col - 1 });

            if (col < matrix.GetLength(1) - 1 && !visited[row, col + 1])
                unVisitedNeighbors.Add(new int[] { row, col + 1 });

            return unVisitedNeighbors;
        }

        //https://www.algoexpert.io/questions/spiral-traverse
        public static List<int> SpiralTraverse(int[,] array)
        {

            if (array.GetLength(0) == 0) return new List<int>();

            //1. Iterative - T:O(n) | S:O(n)
            var result = SpiralTraverseIterative(array);

            //2. Recursive - T:O(n) | S:O(n)
            SpiralTraverseRecursion(array, 0, array.GetLength(0) - 1, 0, array.GetLength(1) - 1, result);
            return result;
        }

        private static void SpiralTraverseRecursion(int[,] array, int startRow, int endRow, int startCol, int endCol, List<int> result)
        {
            if (startRow > endRow || startCol > endCol) return;

            //TOP
            for (int col = startCol; col <= endCol; col++)
            {
                result.Add(array[startRow, col]);
            }

            //Right
            for (int row = startRow + 1; row <= endRow; row++)
            {
                result.Add(array[row, endCol]);
            }

            //Bottom 
            for (int col = endCol - 1; col >= startCol; col++)
            {

                //Single Row edge case
                if (startRow == endRow) break;

                result.Add(array[endRow, col]);
            }

            //Left
            for (int row = endRow - 1; row > startRow; row++)
            {

                //Single column edge case
                if (startCol == endCol) break;

                result.Add(array[row, startCol]);
            }

            SpiralTraverseRecursion(array, startRow++, endRow--, startCol++, endCol--, result);
        }

        private static List<int> SpiralTraverseIterative(int[,] array)
        {
            List<int> result = new List<int>();

            var startRow = 0;
            var endRow = array.GetLength(0) - 1;
            var startCol = 0;
            var endCol = array.GetLength(1) - 1;

            while (startRow <= endRow && startCol <= endCol)
            {

                //Top(Left->Right)
                for (int col = startCol; col <= endCol; col++)
                {
                    result.Add(array[startRow, col]);
                }

                //Right (Top to Bottom)
                for (int row = startRow + 1; row <= endRow; row++)
                {
                    result.Add(array[row, endCol]);
                }

                //Bottom (Right -> Left)
                for (int col = endCol - 1; col >= startCol; col--)
                {

                    //Single Row edge case
                    if (startRow == endRow) break;

                    result.Add(array[endRow, col]);
                }

                //Left (Bottom to Top)
                for (int row = endRow - 1; row > startRow; row--)
                {

                    //Single Column Edge code
                    if (startCol == endCol) break;

                    result.Add(array[row, startCol]);
                }
                startRow++;
                startCol++;
                endRow--;
                endCol--;


            }

            return result;

        }

        //https://www.algoexpert.io/questions/minimum-passes-of-matrix
        public static int MinimumPassesOfMatrix(int[][] matrix)
        {
            int passes = ConvertNegatives(matrix);

            return (!ContainsNegatives(matrix)) ? passes : -1;
        }

        private static bool ContainsNegatives(int[][] matrix)
        {
            foreach (var row in matrix)
            {
                foreach (var val in row)
                {
                    if (val < 0)
                        return true;
                }
            }
            return false;
        }

        private static int ConvertNegatives(int[][] matrix)
        {
            Queue<Pos> posQ = GetAllPositivePositions(matrix);
            int passes = 0;
            int size = posQ.Count();
            while (posQ.Count > 0)
            {
                Pos curPos = posQ.Dequeue();
                size--;
                List<int[]> adjacentPos = GetAdjacentPositions(matrix, curPos);
                foreach (var pos in adjacentPos)
                {
                    int row = pos[0], col = pos[1];
                    int val = matrix[row][col];
                    if (val < 0)
                    {
                        matrix[row][col] = val * -1;
                        posQ.Enqueue(new Pos { Row = row, Col = col });
                    }

                }
                if (size == 0)
                {
                    size = posQ.Count();
                    if (size > 0)
                        passes++;

                }

            }
            return passes;
        }

        private static List<int[]> GetAdjacentPositions(int[][] matrix, Pos pos)
        {
            List<int[]> adjPos = new List<int[]>();
            int row = pos.Row;
            int col = pos.Col;

            //https://www.tutorialsteacher.com/csharp/csharp-multi-dimensional-array
            //var twoDArr = new int[,] {{1,2},{2,3}};
            //https://www.tutorialsteacher.com/csharp/csharp-jagged-array
            //var jogArr = new int[][] {new int[3]{0, 1, 2}, new int[5]{1,2,3,4,5}};

            //Top
            if (row > 0)
                adjPos.Add(new int[] { row - 1, col });

            //Bottom/Down
            if (row < matrix.Length - 1)
            {
                adjPos.Add(new int[] { row + 1, col });
            }

            //Left
            if (col > 0)
                adjPos.Add(new int[] { row, col - 1 });

            //Right
            if (col < matrix[0].Length - 1)
                adjPos.Add(new int[] { row, col + 1 });



            return adjPos;
        }

        private static Queue<Pos> GetAllPositivePositions(int[][] matrix)
        {
            Queue<Pos> positivePos = new Queue<Pos>();

            for (int row = 0; row < matrix.Length; row++)
            {
                for (int col = 0; col < matrix[row].Length; col++)
                {
                    int val = matrix[row][col];
                    if (val > 0)
                        positivePos.Enqueue(new Pos() { Row = row, Col = col });
                }
            }
            return positivePos;
        }

        struct Pos
        {
            public int Row, Col;

        }

        //https://www.algoexpert.io/questions/rectangle-mania
        static string UP = "up";
        static string RIGHT = "right";
        static string DOWN = "down";
        static string LEFT = "left";

        public static int RectangleMania(List<int[]> coords)
        {

            Dictionary<string, Dictionary<string, List<int[]>>> coordsTable;
            int rectangleCount = 0;
            //1. T: O(n^2) S:O(n^2) where n is number of co-ordinates 
            coordsTable = GetCoordsTable(coords);
            rectangleCount = GetRectangleCount(coords, coordsTable);
        //2. T: O(n^2) S:O(n) where n is number of co-ordinates 
        TODO:

            //3. T: O(n^2) S:O(n) where n is number of co-ordinates 
            HashSet<string> coordsSet = GetCoordsSet(coords);
            rectangleCount = GetRectangleCount(coords, coordsSet);
            return rectangleCount;

        }

        private static int GetRectangleCount(List<int[]> coords, HashSet<string> coordsSet)
        {
            int rectangleCount = 0;

            foreach (var coord1 in coords)
            {
                foreach (var coord2 in coords)
                {
                    if (!IsInUpperRight(coord1, coord2)) continue;
                    string upperCoordString = CoordsToString(new int[] { coord1[0], coord2[1] });
                    string bottomRightCoordString = CoordsToString(new int[] { coord2[0], coord1[1] });
                    if (coordsSet.Contains(upperCoordString) && coordsSet.Contains(bottomRightCoordString))
                        rectangleCount++;
                }
            }
            return rectangleCount;
        }

        private static bool IsInUpperRight(int[] coord1, int[] coord2)
        {
            return coord2[0] > coord1[0] && coord2[1] > coord2[1];
        }

        private static HashSet<string> GetCoordsSet(List<int[]> coords)
        {
            HashSet<string> coordsSet = new HashSet<string>();
            foreach (var coord in coords)
            {
                string coordString = CoordsToString(coord);
                coordsSet.Add(coordString);
            }

            return coordsSet;
        }

        private static int GetRectangleCount(List<int[]> coords, Dictionary<string, Dictionary<string, List<int[]>>> coordsTable)
        {
            int rectangleCount = 0;
            foreach (var coord in coords)
                rectangleCount += ClockWiseCountRectangles(coord, coordsTable, UP, coord);

            return rectangleCount;
        }

        private static int ClockWiseCountRectangles(int[] coord, Dictionary<string, Dictionary<string, List<int[]>>> coordsTable, string direction, int[] origin)
        {
            string coordString = CoordsToString(coord);

            if (direction == LEFT)
            {
                bool rectangleFound = coordsTable[coordString][LEFT].Contains(origin);
                return rectangleFound ? 1 : 0;
            }
            else
            {
                int rectangleCount = 0;
                string nextDirection = GetNextClockWiseDirection(direction);
                foreach (var nextCoord in coordsTable[coordString][direction])
                {
                    rectangleCount += ClockWiseCountRectangles(nextCoord, coordsTable, nextDirection, origin);
                }
                return rectangleCount;
            }

        }

        private static string GetNextClockWiseDirection(string direction)
        {
            if (direction == UP) return RIGHT;
            if (direction == RIGHT) return DOWN;
            if (direction == DOWN) return LEFT;

            return "";
        }

        private static Dictionary<string, Dictionary<string, List<int[]>>> GetCoordsTable(List<int[]> coords)
        {

            Dictionary<string, Dictionary<string, List<int[]>>> coordsTable = new Dictionary<string, Dictionary<string, List<int[]>>>();

            foreach (int[] coord1 in coords)
            {

                Dictionary<string, List<int[]>> coord1Directions = new Dictionary<string, List<int[]>>();
                coord1Directions[UP] = new List<int[]>();
                coord1Directions[DOWN] = new List<int[]>();
                coord1Directions[RIGHT] = new List<int[]>();
                coord1Directions[LEFT] = new List<int[]>();

                foreach (var coord2 in coords)
                {

                    string coord2Direction = GetCoordDirection(coord1, coord2);

                    if (coord1Directions.ContainsKey(coord2Direction))
                        coord1Directions[coord2Direction].Add(coord2);

                }
                string coord1String = CoordsToString(coord1);
                coordsTable[coord1String] = coord1Directions;

            }

            return coordsTable;
        }

        private static string CoordsToString(int[] coord)
        {
            return coord[0].ToString() + "-" + coord[1].ToString();
        }

        private static string GetCoordDirection(int[] coord1, int[] coord2)
        {
            if (coord2[1] == coord1[1])
            {
                if (coord2[0] > coord1[0])
                {
                    return RIGHT;
                }
                else if (coord2[0] < coord1[0])
                {
                    return LEFT;
                }
            }
            else if (coord2[0] == coord1[0])
            {
                if (coord2[1] > coord1[1])
                {
                    return UP;
                }
                else if (coord2[1] < coord1[1])
                {
                    return DOWN;
                }
            }
            return "";
        }

        //https://www.algoexpert.io/questions/tournament-winner
        public static string TournamentWinner(List<List<string>> competitions, List<int> results)
        {
            //T:O(n) | S:O(k) where n is number of competitions and k is number of teams.
            Dictionary<string, int> teamScores = new Dictionary<string, int>();
            string maxScoreTeam = "";
            for (int i = 0; i < results.Count; i++)
            {

                int winner = results[i];
                string currentWinningTeam = competitions[i][1 - winner];
                if (!teamScores.ContainsKey(currentWinningTeam))
                    teamScores[currentWinningTeam] = 0;

                teamScores[currentWinningTeam] += 3;
                if (string.IsNullOrEmpty(maxScoreTeam))
                    maxScoreTeam = currentWinningTeam;
                else if (teamScores[maxScoreTeam] < teamScores[currentWinningTeam])
                    maxScoreTeam = currentWinningTeam;


            }
            return maxScoreTeam;

        }
        //https://www.algoexpert.io/questions/square-of-zeroes
        public static bool SquareOfZeroes(List<List<int>> matrix)
        {

            //1.Naive Iterative - T:n^4 |S:O(1) where n is height and width of matrix
            bool IsSquarOfZeroesExists = SquareOfZeroesNaiveIterative(matrix);

            //2.Optimal(Precompute) Iterative - T:(n^3) |S:O(n^2) where n is height and width of matrix
            IsSquarOfZeroesExists = SquareOfZeroesOptimalIterative(matrix);

            //3.Naive Recursive without Caching - T:(n^4) |S:O(n^3) where n is height and width of matrix

            //4.Optimal Recursive with Caching - T:(n^4) |S:O(n^3) where n is height and width of matrix
            IsSquarOfZeroesExists = SquareOfZeroesOptimalRecursive(matrix);
        //5.Optimal Recursive with Caching & PreCompute - T:(n^3) |S:O(n^3) where n is height and width of matrix
        TODO:
            return IsSquarOfZeroesExists;
        }

        private static bool SquareOfZeroesOptimalRecursive(List<List<int>> matrix)
        {
            int lastIdx = matrix.Count - 1;
            Dictionary<string, bool> cache = new Dictionary<string, bool>();
            return HasSquareOfZeroes(matrix, 0, 0, lastIdx, lastIdx, cache);

        }

        private static bool HasSquareOfZeroes(List<List<int>> matrix, int topRow, int leftCol, int bottomRow, int rightCol, Dictionary<string, bool> cache)
        {
            if (topRow >= bottomRow || leftCol >= rightCol) return false;
            string key = topRow.ToString() + '-' + leftCol.ToString() + '-' + bottomRow.ToString() + '-' + rightCol.ToString();
            if (cache.ContainsKey(key)) return cache[key];

            cache[key] = IsSquareOfZeroes(matrix, topRow, leftCol, bottomRow, rightCol) ||
                         HasSquareOfZeroes(matrix, topRow + 1, leftCol + 1, bottomRow - 1, rightCol - 1, cache) ||
                         HasSquareOfZeroes(matrix, topRow, leftCol + 1, bottomRow - 1, rightCol, cache) ||
                         HasSquareOfZeroes(matrix, topRow + 1, leftCol, bottomRow, rightCol - 1, cache) ||
                         HasSquareOfZeroes(matrix, topRow + 1, leftCol + 1, bottomRow, rightCol, cache) ||
                         HasSquareOfZeroes(matrix, topRow, leftCol, bottomRow - 1, rightCol - 1, cache);

            return cache[key];
        }

        private static bool SquareOfZeroesOptimalIterative(List<List<int>> matrix)
        {
            List<List<InfoMatrixItem>> infoMatrix = PreComputeNumOfZeroes(matrix);
            int n = matrix.Count;
            for (int topRow = 0; topRow < n; topRow++)
            {
                for (int leftCol = 0; leftCol < n; leftCol++)
                {
                    int squareLen = 2;
                    while (squareLen <= n - leftCol && squareLen <= n - topRow)
                    {
                        int bottomRow = topRow + squareLen - 1;
                        int rightCol = leftCol + squareLen - 1;
                        if (IsSquareOfZeroes(infoMatrix, topRow, leftCol, bottomRow, rightCol))
                        {
                            return true;
                        }
                        squareLen++;
                    }
                }
            }
            return false;
        }

        private static bool IsSquareOfZeroes(List<List<InfoMatrixItem>> infoMatrix, int topRow, int leftCol, int bottomRow, int rightCol)
        {
            int squareLen = rightCol - leftCol + 1;
            bool hasTopBorder = infoMatrix[topRow][leftCol].NumberZeroesRight >= squareLen;
            bool hasLeftBorder = infoMatrix[topRow][leftCol].NumberZeroesBelow >= squareLen;
            bool hasBottomBorder = infoMatrix[bottomRow][leftCol].NumberZeroesRight >= squareLen;
            bool hasRightBorder = infoMatrix[topRow][rightCol].NumberZeroesBelow >= squareLen;

            return hasBottomBorder && hasLeftBorder && hasRightBorder && hasTopBorder;
        }

        private static List<List<InfoMatrixItem>> PreComputeNumOfZeroes(List<List<int>> matrix)
        {
            List<List<InfoMatrixItem>> infoMatrix = new List<List<InfoMatrixItem>>();
            for (int row = 0; row < matrix.Count; row++)
            {
                List<InfoMatrixItem> inner = new List<InfoMatrixItem>();
                for (int col = 0; col < matrix[row].Count; col++)
                {
                    int numZeroes = matrix[row][col] == 0 ? 1 : 0;
                    inner.Add(new InfoMatrixItem(numZeroes, numZeroes));
                }
                infoMatrix.Add(inner);
            }
            int lastIdx = infoMatrix.Count - 1;
            for (int row = lastIdx; row >= 0; row--)
            {
                for (int col = lastIdx; col >= 0; col--)
                {
                    if (matrix[row][col] == 1) continue;
                    if (row < lastIdx)
                    {
                        infoMatrix[row][col].NumberZeroesBelow += infoMatrix[row + 1][col].NumberZeroesBelow;
                    }
                    if (col < lastIdx)
                    {
                        infoMatrix[row][col].NumberZeroesRight += infoMatrix[row][col + 1].NumberZeroesRight;
                    }
                }
            }
            return infoMatrix;
        }
        internal class InfoMatrixItem
        {
            public int NumberZeroesBelow { get; set; }
            public int NumberZeroesRight { get; set; }

            public InfoMatrixItem(int numZeroesBelow, int numZeroesRight)
            {
                this.NumberZeroesBelow = numZeroesBelow;
                this.NumberZeroesRight = numZeroesRight;
            }
        }
        private static bool SquareOfZeroesNaiveIterative(List<List<int>> matrix)
        {
            int n = matrix.Count;
            for (int topRow = 0; topRow < n; topRow++)
            {
                for (int leftCol = 0; leftCol < n; leftCol++)
                {
                    int squareLen = 2;
                    while (squareLen <= n - leftCol && squareLen <= n - topRow)
                    {
                        int bottomRow = topRow + squareLen - 1;
                        int rightCol = leftCol + squareLen - 1;
                        if (IsSquareOfZeroes(matrix, topRow, leftCol, bottomRow, rightCol))
                            return true;
                        squareLen++;
                    }
                }
            }
            return false;
        }

        private static bool IsSquareOfZeroes(List<List<int>> matrix, int topRow, int leftCol, int bottomRow, int rightCol)
        {
            for (int row = topRow; row < bottomRow + 1; row++)
            {
                if (matrix[row][leftCol] != 0 || matrix[row][rightCol] != 0) return false;
            }
            for (int col = leftCol; col < rightCol + 1; col++)
            {

                if (matrix[topRow][col] != 0 || matrix[bottomRow][col] != 0) return false;
            }
            return true;
        }

        //https://www.algoexpert.io/questions/knapsack-problem
        public static List<List<int>> KnapsackProblem(int[,] items, int capacity)
        {
            //T:O(nc) | S:O(nc)
            int[,] knapsackValues = new int[items.GetLength(0) + 1, capacity + 1];
            for (int row = 1; row < items.GetLength(0) + 1; row++)
            {
                int currentWeigtht = items[row - 1, 1];
                int currentValue = items[row - 1, 0];
                for (int col = 0; col < capacity + 1; col++)
                {
                    if (currentWeigtht > col)
                    {
                        knapsackValues[row, col] = knapsackValues[row - 1, col];
                    }
                    else
                    {
                        knapsackValues[row, col] = Math.Max(knapsackValues[row - 1, col],
                                                            knapsackValues[row - 1, col - currentWeigtht] + currentValue);

                    }
                }
            }
            return GetKnapsackItems(knapsackValues, items, knapsackValues[items.GetLength(0), capacity]);
        }

        private static List<List<int>> GetKnapsackItems(int[,] knapsackValues, int[,] items, int weight)
        {
            List<List<int>> sequence = new List<List<int>>();

            List<int> totalWeight = new List<int>();
            sequence.Add(totalWeight);
            sequence.Add(new List<int>());
            int row = knapsackValues.GetLength(0) - 1;
            int col = knapsackValues.GetLength(1) - 1;
            while (row > 0)
            {
                if (knapsackValues[row, col] == knapsackValues[row - 1, col])
                    row--;
                else
                {
                    sequence[1].Insert(0, row - 1);
                    col -= items[row - 1, 1];
                    row--;

                }
                if (col == 0) break;
            }

            return sequence;
        }
        //https://www.algoexpert.io/questions/maximum-sum-submatrix
        public static int MaximumSumSubmatrix(int[,] matrix, int size)
        {

            //1.Naive, with pair of loops to find submatrix based on size and loop thru square values to get sum; Overlaps can occur
            //T:O(w*h*size^2) | S:O(1)


            //2.Optimal - precompute sums at each cell to avoid repeated calculations due to overlap
            //T:O(w*h) | S:O(w*h)
            int maxSumMatrixSum = MaximumSumSubmatrixOptimal(matrix, size);
            return maxSumMatrixSum;
        }

        private static int MaximumSumSubmatrixOptimal(int[,] matrix, int size)
        {
            int[,] sums = PrecomputeSumMatrix(matrix);
            int maxSubMatrixSum = Int32.MinValue;
        TODO:
            for (int row = size - 1; row < matrix.GetLength(0); row++)
            {
                for (int col = size - 1; col < matrix.GetLength(1); col++)
                {
                    int total = sums[row, col];

                    bool touchesTopBorder = (row - size < 0);
                    if (!touchesTopBorder)
                        total -= sums[row - size, col];

                    bool touchesLeftBorder = (col - size < 0);
                    if (!touchesLeftBorder)
                    {
                        total -= sums[row, col - size];
                    }

                    bool touchesTopOrLeftBorder = (touchesTopBorder || touchesLeftBorder);
                    if (!touchesTopOrLeftBorder)
                        total += sums[row - size, col - size];

                    maxSubMatrixSum = Math.Max(maxSubMatrixSum, total);

                }
            }
            return maxSubMatrixSum;
        }

        private static int[,] PrecomputeSumMatrix(int[,] matrix)
        {
            int[,] sums = new int[matrix.GetLength(0), matrix.GetLength(1)];
            sums[0, 0] = matrix[0, 0];

            //Fill first row
            for (int idx = 1; idx < matrix.GetLength(1); idx++)
            {
                sums[0, idx] = sums[0, idx - 1] + matrix[0, idx];
            }
            //Fill first column
            for (int idx = 1; idx < matrix.GetLength(0); idx++)
            {
                sums[idx, 0] = sums[idx - 1, 0] + matrix[idx, 0];
            }

            //Fill in the rest of matrix
            for (int row = 1; row < matrix.GetLength(0); row++)
            {
                for (int col = 1; col < matrix.GetLength(1); col++)
                {
                    sums[row, col] = sums[row - 1, col] + sums[row, col - 1] - sums[row - 1, col - 1] + matrix[row, col];
                }
            }
            return sums;
        }
        //https://www.algoexpert.io/questions/search-in-sorted-matrix
        public static int[] SearchInMatrix(int[,] matrix, int target)
        {
            //1.Naive - pair of loops
            //T:O(n^2) | S:O(1)

            //2. Leveraging facts that rows and columns of matrix sorted

            int row = 0;
            int col = matrix.GetLength(1) - 1;
            while (row < matrix.GetLength(0) && col >= 0)
            {

                if (matrix[row, col] > target)
                {
                    col--;
                }
                else if (matrix[row, col] < target)
                {
                    row++;
                }
                else
                {
                    return new int[] { row, col };
                }
            }
            return new int[] { -1, -1 };
        }
        //https://www.algoexpert.io/questions/minimum-area-rectangle
        public static int MinimumAreaRectangle(int[][] points)
        {

            if (points.Length < 4) return 0;
            //1.Naive - 4 pair/nested of loops to generate all possible combinations of 4 points and find minimum area among rectangles found
            //T:O(n^4) | S:O(1)

            //2.Optimal - edge pairing algo- find parallel points vertically and horizontally to see if they can form rectangles
            //T:O(n^2) | S:O(n) - n is number of points
            int minAreaRect = MinimumAreaRectangleOptimal(points);

            //3.Optimal - simplified -find two opposite end points and try to match them with any two points to see if they can form rectangles
            //T:O(n^2) | S:O(n) - n is number of points
            minAreaRect = MinimumAreaRectangleOptima2(points);

            return minAreaRect != Int32.MinValue ? minAreaRect : 0;

        }

        private static int MinimumAreaRectangleOptima2(int[][] points)
        {
            HashSet<string> pointSet = CreatePointSet(points);
            int minAreaRect = Int32.MaxValue;
            for (int curIdx = 0; curIdx < points.Length; curIdx++)
            {

                int p2x = points[curIdx][0];
                int p2y = points[curIdx][1];

                for (int prevIdx = 0; prevIdx < curIdx; prevIdx++)
                {

                    int p1x = points[prevIdx][0];
                    int p1y = points[prevIdx][1];

                    bool pointsShareValue = p1x == p2x || p2y == p1y;
                    if (pointsShareValue) continue;

                    bool point1OnOppositDirectionExists = pointSet.Contains(ConvertPointToString(p1x, p2y));
                    bool point2OnOppositDirectionExists = pointSet.Contains(ConvertPointToString(p2x, p1y));

                    bool oppositeDiagonalExists = point1OnOppositDirectionExists && point2OnOppositDirectionExists;

                    if (oppositeDiagonalExists)
                    {
                        int curArea = Math.Abs(p1x - p2x) * Math.Abs(p1y - p2y);
                        minAreaRect = Math.Min(minAreaRect, curArea);
                    }
                }

            }
            return minAreaRect;
        }

        private static string ConvertPointToString(int x, int y)
        {
            return x.ToString() + ":" + y.ToString();
        }

        private static HashSet<string> CreatePointSet(int[][] points)
        {
            HashSet<string> pointSet = new HashSet<string>();
            foreach (var point in points)
            {
                int x = point[0];
                int y = point[1];
                string pointStr = x.ToString() + "-" + y.ToString();
                pointSet.Add(pointStr);
            }
            return pointSet;
        }

        private static int MinimumAreaRectangleOptimal(int[][] points)
        {
            Dictionary<int, int[]> columns = InitializeColumns(points);
            int minAreaRect = Int32.MaxValue;
            Dictionary<string, int> edgesParallelToYAxis = new Dictionary<string, int>();
            List<int> sortedColumns = new List<int>(columns.Keys);
            sortedColumns.Sort();

            foreach (var x in sortedColumns)
            {
                int[] yValuesInCurrentColumn = columns[x];
                Array.Sort(yValuesInCurrentColumn);

                for (int curIdx = 0; curIdx < yValuesInCurrentColumn.Length; curIdx++)
                {
                    int y2 = yValuesInCurrentColumn[curIdx];
                    for (int prevIdx = 0; prevIdx < curIdx; prevIdx++)
                    {
                        int y1 = yValuesInCurrentColumn[prevIdx];
                        string pointString = y1.ToString() + ":" + y2.ToString();

                        if (edgesParallelToYAxis.ContainsKey(pointString))
                        {
                            int currArea = (x - edgesParallelToYAxis[pointString] * y2 - y1);
                            minAreaRect = Math.Min(minAreaRect, currArea);
                        }
                        edgesParallelToYAxis[pointString] = x;
                    }
                }
            }
            return minAreaRect;
        }

        private static Dictionary<int, int[]> InitializeColumns(int[][] points)
        {
            Dictionary<int, int[]> columns = new Dictionary<int, int[]>();

            foreach (var point in points)
            {
                int x = point[0];
                int y = point[1];

                if (!columns.ContainsKey(x))
                {
                    columns[x] = new int[] { };
                }
                int[] column = columns[x];
                int[] newColumn = new int[column.Length + 1];
                for (int i = 0; i < column.Length; i++)
                {
                    newColumn[i] = column[i];
                }
                newColumn[column.Length] = y;
                columns[x] = newColumn;
            }
            return columns;
        }
        //https://www.algoexpert.io/questions/number-of-ways-to-traverse-graph
        public static int NumberOfWaysToTraverseGraph(int width, int height)
        {

            //1.Naive 
            //T:O(2^(n+m)) |S:O(n+m)
            int numWaysToTraverseGraph = NumberOfWaysToTraverseGraphNaive(width, height);

            //2.Optimal - using Dynamic Programming aka memoization
            //T:O(n*m) |S:O(n*m)
            numWaysToTraverseGraph = NumberOfWaysToTraverseGraphOptimal1(width, height);

            //3.Optimal - using math formual for permutations among given sets ex: (3, 2) representing width and height
            //T:O(n+m) |S:O(1)
            numWaysToTraverseGraph = NumberOfWaysToTraverseGraphOptimal2(width, height);
            return numWaysToTraverseGraph;
        }

        private static int NumberOfWaysToTraverseGraphOptimal2(int width, int height)
        {
            int xDistinaceToCorner = width - 1;
            int yDistanceToCorner = height - 1;

            //The number of permutations of right and down movements 
            //is the number of ways to reach the bottom right corner
            //(n+r)!/n!*r!
            int numerator = Factorial(xDistinaceToCorner + yDistanceToCorner);
            int denominator = Factorial(xDistinaceToCorner) * Factorial(yDistanceToCorner);

            return numerator / denominator;
        }

        private static int Factorial(int num)
        {
            int result = 1;
            for (int i = 2; i <= num; i++)
                result *= i;

            return result;
        }

        private static int NumberOfWaysToTraverseGraphOptimal1(int width, int height)
        {
            //Heigh =>Row And Width =>Column
            int[,] numberOfWays = new int[height + 1, width + 1];

            for (int widthIdx = 1; widthIdx < width + 1; widthIdx++)
            {

                for (int heightIdx = 1; heightIdx < height + 1; heightIdx++)
                {

                    if (widthIdx == 1 || heightIdx == 1)
                    {
                        numberOfWays[heightIdx, widthIdx] = 1;
                    }
                    else
                    {
                        int waysLeft = numberOfWays[heightIdx, widthIdx - 1];
                        int waysRight = numberOfWays[heightIdx - 1, widthIdx];
                        numberOfWays[heightIdx, widthIdx] = waysLeft + waysRight;

                    }

                }
            }
            return numberOfWays[height, width];
        }

        private static int NumberOfWaysToTraverseGraphNaive(int width, int height)
        {
            if (width == 1 || height == 1) return 1;

            return NumberOfWaysToTraverseGraphNaive(width - 1, height) +
                    NumberOfWaysToTraverseGraphNaive(width, height - 1);

        }
        //https://www.algoexpert.io/questions/two-colorable
        public static bool TwoColorables(int[][] edges)
        {

            //T:O(v+e) | S:O(v)
            int[] colors = new int[edges.Length];
            colors[0] = 1;
            Stack<int> stack = new Stack<int>();
            stack.Push(0);

            while (stack.Count > 0)
            {
                int node = stack.Pop();
                foreach (var connection in edges[node])
                {
                    if (colors[connection] == 0)
                    {
                        colors[connection] = colors[node] == 1 ? 2 : 1;
                        stack.Push(connection);
                    }
                    else if (colors[connection] == colors[node])
                        return false;
                }
            }
            return true;

        }
        //https://www.algoexpert.io/questions/breadth-first-search
        // O(v + e) time | O(v) space
        public class NodeBFS
        {
            public string name;
            public List<NodeBFS> children = new List<NodeBFS>();

            public NodeBFS(string name)
            {
                this.name = name;
            }

            // O(v + e) time | O(v) space
            public List<string> BreadthFirstSearch(List<string> array)
            {
                Queue<NodeBFS> queue = new Queue<NodeBFS>();
                queue.Enqueue(this);
                while (queue.Count > 0)
                {
                    NodeBFS current = queue.Dequeue();
                    array.Add(current.name);
                    current.children.ForEach(o => queue.Enqueue(o));
                }
                return array;
            }

            public NodeBFS AddChild(string name)
            {
                NodeBFS child = new NodeBFS(name);
                children.Add(child);
                return this;
            }
        }

        //https://www.algoexpert.io/questions/dijkstra's-algorithm

        // O(v^2 + e) time | O(v) space - where v is the number of
        // vertices and e is the number of edges in the input graph
        public int[] DijkstrasAlgorithm1(int start, int[][][] edges)
        {
            int numberOfVertices = edges.Length;

            int[] minDistances = new int[edges.Length];
            Array.Fill(minDistances, Int32.MaxValue);
            minDistances[start] = 0;

            HashSet<int> visited = new HashSet<int>();

            while (visited.Count != numberOfVertices)
            {
                int[] getVertexData = getVertexWithMinDistances(minDistances, visited);
                int vertex = getVertexData[0];
                int currentMinDistance = getVertexData[1];

                if (currentMinDistance == Int32.MaxValue)
                {
                    break;
                }

                visited.Add(vertex);

                foreach (var edge in edges[vertex])
                {
                    int destination = edge[0];
                    int distanceToDestination = edge[1];

                    if (visited.Contains(destination))
                    {
                        continue;
                    }

                    int newPathDistance = currentMinDistance + distanceToDestination;
                    int currentDestinationDistance = minDistances[destination];
                    if (newPathDistance < currentDestinationDistance)
                    {
                        minDistances[destination] = newPathDistance;
                    }
                }
            }

            int[] finalDistances = new int[minDistances.Length];
            for (int i = 0; i < minDistances.Length; i++)
            {
                int distance = minDistances[i];
                if (distance == Int32.MaxValue)
                {
                    finalDistances[i] = -1;
                }
                else
                {
                    finalDistances[i] = distance;
                }
            }

            return finalDistances;
        }

        public int[] getVertexWithMinDistances(
          int[] distances, HashSet<int> visited
        )
        {
            int currentMinDistance = Int32.MaxValue;
            int vertex = -1;

            for (int vertexIdx = 0; vertexIdx < distances.Length; vertexIdx++)
            {
                int distance = distances[vertexIdx];

                if (visited.Contains(vertexIdx))
                {
                    continue;
                }

                if (distance <= currentMinDistance)
                {
                    vertex = vertexIdx;
                    currentMinDistance = distance;
                }
            }

            return new int[] { vertex, currentMinDistance };
        }


        // O((v + e) * log(v)) time | O(v) space - where v is the number
        // of vertices and e is the number of edges in the input graph
        public int[] DijkstrasAlgorithmOptimal(int start, int[][][] edges)
        {
            int numberOfVertices = edges.Length;

            int[] minDistances = new int[numberOfVertices];
            Array.Fill(minDistances, Int32.MaxValue);
            minDistances[start] = 0;

            List<Item> minDistancesPairs = new List<Item>();
            for (int i = 0; i < numberOfVertices; i++)
            {
                Item item = new Item(i, Int32.MaxValue);
                minDistancesPairs.Add(item);
            }

            MinHeap minDistancesHeap = new MinHeap(minDistancesPairs);
            minDistancesHeap.Update(start, 0);

            while (!minDistancesHeap.isEmpty())
            {
                Item heapItem = minDistancesHeap.Remove();
                int vertex = heapItem.vertex;
                int currentMinDistance = heapItem.distance;

                if (currentMinDistance == Int32.MaxValue)
                {
                    break;
                }

                foreach (var edge in edges[vertex])
                {
                    int destination = edge[0];
                    int distanceToDestination = edge[1];
                    int newPathDistance = currentMinDistance + distanceToDestination;
                    int currentDestinationDistance = minDistances[destination];
                    if (newPathDistance < currentDestinationDistance)
                    {
                        minDistances[destination] = newPathDistance;
                        minDistancesHeap.Update(destination, newPathDistance);
                    }
                }
            }

            int[] finalDistances = new int[minDistances.Length];
            for (int i = 0; i < minDistances.Length; i++)
            {
                int distance = minDistances[i];
                if (distance == Int32.MaxValue)
                {
                    finalDistances[i] = -1;
                }
                else
                {
                    finalDistances[i] = distance;
                }
            }

            return finalDistances;
        }
        public class Item
        {
            public int vertex;
            public int distance;

            public Item(int vertex, int distance)
            {
                this.vertex = vertex;
                this.distance = distance;
            }
        };

        public class MinHeap
        {
            Dictionary<int, int> vertexDictionary = new Dictionary<int, int>();
            List<Item> heap = new List<Item>();
            public MinHeap(List<Item> array)
            {
                for (int i = 0; i < array.Count; i++)
                {
                    Item item = array[i];
                    vertexDictionary[item.vertex] = item.vertex;
                }
                heap = buildHeap(array);
            }
            List<Item> buildHeap(List<Item> array)
            {
                int firstParentIdx = (array.Count - 2) / 2;
                for (int currentIdx = firstParentIdx + 1; currentIdx >= 0; currentIdx--)
                {
                    siftDown(currentIdx, array.Count - 1, array);
                }
                return array;
            }

            public bool isEmpty()
            {
                return heap.Count == 0;
            }

            void siftDown(int currentIdx, int endIdx, List<Item> heap)
            {
                int childOneIdx = currentIdx * 2 + 1;
                while (childOneIdx <= endIdx)
                {
                    int childTwoIdx =
                      currentIdx * 2 + 2 <= endIdx ? currentIdx * 2 + 2 : -1;
                    int idxToSwap;
                    if (childTwoIdx != -1 && heap[childTwoIdx].distance < heap[childOneIdx].distance)
                    {
                        idxToSwap = childTwoIdx;
                    }
                    else
                    {
                        idxToSwap = childOneIdx;
                    }
                    if (heap[idxToSwap].distance < heap[currentIdx].distance)
                    {
                        swap(currentIdx, idxToSwap);
                        currentIdx = idxToSwap;
                        childOneIdx = currentIdx * 2 + 1;
                    }
                    else
                    {
                        return;
                    }
                }
            }

            void siftUp(int currentIdx)
            {
                int parentIdx = (currentIdx - 1) / 2;
                while (currentIdx > 0 &&
                       heap[currentIdx].distance < heap[parentIdx].distance)
                {
                    swap(currentIdx, parentIdx);
                    currentIdx = parentIdx;
                    parentIdx = (currentIdx - 1) / 2;
                }
            }

            public Item Remove()
            {
                swap(0, heap.Count - 1);
                Item lastItem = heap[heap.Count - 1];
                int vertex = lastItem.vertex;
                int distance = lastItem.distance;
                heap.RemoveAt(heap.Count - 1);
                vertexDictionary.Remove(vertex);
                siftDown(0, heap.Count - 1, heap);
                return new Item(vertex, distance);
            }

            public void Update(int vertex, int value)
            {
                heap[vertexDictionary[vertex]] = new Item(vertex, value);
                siftUp(vertexDictionary[vertex]);
            }

            void swap(int i, int j)
            {
                vertexDictionary[heap[i].vertex] = j;
                vertexDictionary[heap[j].vertex] = i;
                Item temp = heap[i];
                heap[i] = heap[j];
                heap[j] = temp;
            }
        }



        //https://www.algoexpert.io/questions/transpose-matrix
        // O(w * h) time | O(w * h) space - where w is the
        // width of the matrix and h is the height
        public int[,] TransposeMatrix(int[,] matrix)
        {
            int[,] transposedMatrix = new int[matrix.GetLength(1), matrix.GetLength(0)];
            for (int col = 0; col < matrix.GetLength(1); col++)
            {
                for (int row = 0; row < matrix.GetLength(0); row++)
                {
                    transposedMatrix[col, row] = matrix[row, col];
                }
            }
            return transposedMatrix;
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

                    findOnesConnectedToBorder(matrix, row, col, onesConnectedToBorder);
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
        public void findOnesConnectedToBorder(
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

                var neighbors = getNeighbors(matrix, currentRow, currentCol);
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

        public List<Tuple<int, int>> getNeighbors(int[][] matrix, int row, int col)
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

                var neighbors = getNeighbors(matrix, currentRow, currentCol);
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

        //https://www.algoexpert.io/questions/zigzag-traverse
        // O(n) time | O(n) space - where n is the total number of elements in the
        // two-dimensional array
        public class Program
        {
            public static List<int> ZigzagTraverse(List<List<int>> array)
            {
                int height = array.Count - 1;
                int width = array[0].Count - 1;
                List<int> result = new List<int>();
                int row = 0;
                int col = 0;
                bool goingDown = true;
                while (!isOutOfBounds(row, col, height, width))
                {
                    result.Add(array[row][col]);
                    if (goingDown)
                    {
                        if (col == 0 || row == height)
                        {
                            goingDown = false;
                            if (row == height)
                            {
                                col++;
                            }
                            else
                            {
                                row++;
                            }
                        }
                        else
                        {
                            row++;
                            col--;
                        }
                    }
                    else
                    {
                        if (row == 0 || col == width)
                        {
                            goingDown = true;
                            if (col == width)
                            {
                                row++;
                            }
                            else
                            {
                                col++;
                            }
                        }
                        else
                        {
                            row--;
                            col++;
                        }
                    }
                }
                return result;
            }

            public static bool isOutOfBounds(int row, int col, int height, int width)
            {
                return row < 0 || row > height || col < 0 || col > width;
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
        //https://www.algoexpert.io/questions/line-through-points
        // O(n^2) time | O(n) space - where n is the number of points
        public int LineThroughPoints(int[][] points)
        {
            int maxNumberOfPointsOnLine = 1;

            for (int idx1 = 0; idx1 < points.Length; idx1++)
            {
                int[] p1 = points[idx1];
                Dictionary<string, int> slopes = new Dictionary<string, int>();

                for (int idx2 = idx1 + 1; idx2 < points.Length; idx2++)
                {
                    int[] p2 = points[idx2];
                    int[] slopeOfLineBetweenPoints = getSlopeOfLineBetweenPoints(p1, p2);
                    int rise = slopeOfLineBetweenPoints[0];
                    int run = slopeOfLineBetweenPoints[1];

                    string slopeKey = createHashableKeyForRational(rise, run);
                    if (!slopes.ContainsKey(slopeKey))
                    {
                        slopes[slopeKey] = 1;
                    }
                    slopes[slopeKey] = slopes[slopeKey] + 1;
                }

                int currentMaxNumberOfPointsOnLine = maxSlope(slopes);
                maxNumberOfPointsOnLine =
                  Math.Max(maxNumberOfPointsOnLine, currentMaxNumberOfPointsOnLine);
            }

            return maxNumberOfPointsOnLine;
        }
        public int[] getSlopeOfLineBetweenPoints(int[] p1, int[] p2)
        {
            int p1x = p1[0];
            int p1y = p1[1];
            int p2x = p2[0];
            int p2y = p2[1];

            int[] slope = new int[] { 1, 0 };  // slope of a vertical line

            if (p1x != p2x)
            {  // if line is not vertical
                int xDiff = p1x - p2x;
                int yDiff = p1y - p2y;
                int gcd = getGreatestCommonDivisor(Math.Abs(xDiff), Math.Abs(yDiff));
                xDiff = xDiff / gcd;
                yDiff = yDiff / gcd;
                if (xDiff < 0)
                {
                    xDiff *= -1;
                    yDiff *= -1;
                }

                slope = new int[] { yDiff, xDiff };
            }

            return slope;
        }

        public string createHashableKeyForRational(int numerator, int denominator)
        {
            return numerator.ToString() + ":" + denominator.ToString();
        }
        public int maxSlope(Dictionary<string, int> slopes)
        {
            int currentMax = 0;
            foreach (var slope in slopes)
            {
                currentMax = Math.Max(slope.Value, currentMax);
            }
            return currentMax;
        }

        public int getGreatestCommonDivisor(int num1, int num2)
        {
            int a = num1;
            int b = num2;
            while (true)
            {
                if (a == 0)
                {
                    return b;
                }
                if (b == 0)
                {
                    return a;
                }
                int temp = a;
                a = b;
                b = temp % b;
            }
        }


        //https://www.algoexpert.io/questions/two-edge-connected-graph

        // O(v + e) time | O(v) space - where v is the number of
        // vertices and e is the number of edges in the graph
        public bool TwoEdgeConnectedGraph(int[][] edges)
        {
            if (edges.Length == 0) return true;

            int[] arrivalTimes = new int[edges.Length];
            Array.Fill(arrivalTimes, -1);
            int startVertex = 0;

            if (getMinimumArrivalTimeOfAncestors(startVertex, -1, 0, arrivalTimes, edges) == -1)
            {
                return false;
            }

            return areAllVerticesVisited(arrivalTimes);
        }

        public bool areAllVerticesVisited(int[] arrivalTimes)
        {
            foreach (var time in arrivalTimes)
            {
                if (time == -1)
                {
                    return false;
                }
            }
            return true;
        }

        public int getMinimumArrivalTimeOfAncestors(
          int currentVertex,
          int parent,
          int currentTime,
          int[] arrivalTimes,
          int[][] edges
        )
        {
            arrivalTimes[currentVertex] = currentTime;

            int minimumArrivalTime = currentTime;

            foreach (var destination in edges[currentVertex])
            {
                if (arrivalTimes[destination] == -1)
                {
                    minimumArrivalTime = Math.Min(
                      minimumArrivalTime,
                      getMinimumArrivalTimeOfAncestors(
                        destination, currentVertex, currentTime + 1, arrivalTimes, edges
                      )
                    );
                }
                else if (destination != parent)
                {
                    minimumArrivalTime =
                      Math.Min(minimumArrivalTime, arrivalTimes[destination]);
                }
            }

            // A bridge was detected, which means the graph isn't two-edge-connected.
            if (minimumArrivalTime == currentTime && parent != -1)
            {
                return -1;
            }

            return minimumArrivalTime;
        }

        //547. Number of Provinces
        //https://leetcode.com/problems/number-of-provinces/        
        public static int FindNumberOfConnectedProvinces(int[,] cities)
        {
            int numOfProvinces = 0;
            bool[] visited = new bool[cities.GetLength(1)];
            Queue<int> q = new Queue<int>();
            for (int i = 0; i < cities.GetLength(0); ++i)
            {
                if (!visited[i])
                {
                    q.Enqueue(i);
                    /*1.DFS
                    Time complexity : O(n^2).The complete matrix of size n^2n 2 is traversed.
                    Space complexity : O(n)O(n). visited array of size nn is used.
                    */
                    FindNumberOfConnectedProvincesDfs(cities, visited, i);

                    /*
                     * 2.BFS
                     * 
                     */

                    FindNumberOfConnectedProvincesBfs(cities, visited, q);

                    numOfProvinces += 1;
                }
                /*3. Using UnionFind

                    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        UnionFind dsu = new UnionFind(n);
        int numberOfComponents = n;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isConnected[i][j] == 1 && dsu.find(i) != dsu.find(j)) {
                    numberOfComponents--;
                    dsu.union_set(i, j);
                }
            }
        }

        return numberOfComponents;
    }
                */
            }
            return numOfProvinces;
        }

        private static void FindNumberOfConnectedProvincesBfs(int[,] cities, bool[] visited, Queue<int> q)
        {
            while (q.Count > 0)
            {

                var i = q.Dequeue();

                visited[i] = true;
                for (int j = 0; j < cities.GetLength(1); ++j)
                {
                    if (cities[i, j] == 1 && !visited[j])
                    {
                        q.Enqueue(j);
                    }

                }


            }

        }

        private static void FindNumberOfConnectedProvincesDfs(int[,] cities, bool[] visited, int i)
        {
            for (int j = 0; j < cities.GetLength(1); j++)
            {
                if (cities[i, j] == 1 && !visited[j])
                {
                    visited[j] = true;
                    FindNumberOfConnectedProvincesDfs(cities, visited, j);
                }

            }
        }
        //684. Redundant Connection		https://leetcode.com/problems/redundant-connection
        //https://www.youtube.com/watch?v=P6tEGES63ag

        //1. Using DFS
        //Time Complexity: O(N^2) where N is the number of vertices (and also the number of edges) in the graph
        //Space Complexity: O(N)
        HashSet<int> seen = new HashSet<int>();
        int MAX_EDGE_VAL = 1000;
        public int[] FindRedundantConnectionNaive(int[][] edges)
        {
            List<int>[] graph = new List<int>[MAX_EDGE_VAL + 1];
            for (int i = 0; i <= MAX_EDGE_VAL; i++)
            {
                graph[i] = new List<int>();
            }

            foreach (var edge in edges)
            {
                seen.Clear();
                if (graph[edge[0]].Count > 0 && graph[edge[1]].Count > 0 &&
                        Dfs(graph, edge[0], edge[1]))
                {
                    return edge;
                }
                graph[edge[0]].Add(edge[1]);
                graph[edge[1]].Add(edge[0]);
            }
            throw new InvalidOperationException();
        }

        public bool Dfs(List<int>[] graph, int source, int target)
        {
            if (!seen.Contains(source))
            {
                seen.Add(source);
                if (source == target) return true;
                foreach (int nei in graph[source])
                {
                    if (Dfs(graph, nei, target)) return true;
                }
            }
            return false;
        }
        //2. Using Union Find / Disjoing Set Union (DSU)
        //Time Complexity: O(N) where N is the number of vertices (and also the number of edges) in the graph
        //Space Complexity: O(N)
        public int[] FindRedundantConnectionOptimal(int[][] edges)
        {
            //T: O(n) : S: O(n)
            int MAX_EDGE_VAL = 1000;
            DSUArray dsu = new DSUArray(MAX_EDGE_VAL + 1);
            foreach (var edge in edges)
            {
                if (!dsu.Union(edge[0], edge[1])) return edge;
            }
            throw new Exception("No redundant connection found.");

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
                We create an additional object uf and a boolean array isSubIsland of size m∗n.
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
    }

}