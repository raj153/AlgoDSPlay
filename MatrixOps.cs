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
using static AlgoDSPlay.LinkedListOps;

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
        int n = isConnected.Length;
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
        79. Word Search	
        https://leetcode.com/problems/word-search/description/

        Approach 1: Backtracking

        Complexity
            •	Time Complexity: O(N⋅3L) where N is the number of cells in the board and L is the length of the word to be matched.
            o	For the backtracking function, initially we could have at most 4 directions to explore, but further the choices are reduced into 3 (since we won't go back to where we come from).
            As a result, the execution trace after the first step could be visualized as a 3-nary tree, each of the branches represent a potential exploration in the corresponding direction. Therefore, in the worst case, the total number of invocation would be the number of nodes in a full 3-nary tree, which is about 3L.
            o	We iterate through the board for backtracking, i.e. there could be N times invocation for the backtracking function in the worst case.
            o	As a result, overall the time complexity of the algorithm would be O(N⋅3L).
        •	Space Complexity: O(L) where L is the length of the word to be matched.
            o	The main consumption of the memory lies in the recursion call of the backtracking function. The maximum length of the call stack would be the length of the word. Therefore, the space complexity of the algorithm is O(L).

        */
        public bool Exist(char[][] board, string word)
        {
            this.board = board;
            this.rows = board.Length;
            this.cols = board[0].Length;

            for (int row = 0; row < this.rows; row++)
            {
                for (int col = 0; col < this.cols; col++)
                {
                    if (Backtrack(row, col, word, 0))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        private bool Backtrack(int row, int col, string word, int index)
        {
            // Step 1: Check if the last character matches
            if (index >= word.Length)
            {
                return true;
            }

            // Step 2: Check boundaries
            if (row < 0 || row == this.rows || col < 0 || col == this.cols ||
                this.board[row][col] != word[index])
            {
                return false;
            }

            // Step 3: Explore the neighbors in DFS
            bool ret = false;
            // Mark the path before the next exploration
            char temp = this.board[row][col];
            // Mark the cell as visited by replacing it with '#'
            this.board[row][col] = '#';

            int[] rowOffsets = { 0, 1, 0, -1 };
            int[] colOffsets = { 1, 0, -1, 0 };
            for (int d = 0; d < 4; d++)
            {
                ret = Backtrack(row + rowOffsets[d], col + colOffsets[d], word,
                                index + 1);
                if (ret)      // If path is found, return true without cleanup

                    break;
            }

            // Step 4: Clean up and return 
            // Restore the original character before returning
            this.board[row][col] = temp;
            return ret;
        }

        /*
        212. Word Search II
        https://leetcode.com/problems/word-search-ii/description/

        Approach 1: Backtracking with Trie

        Complexity
        •	Time complexity: O(M(4⋅3L−1)), where M is the number of cells in the board and L is the maximum length of words.
            o	It is tricky is calculate the exact number of steps that a backtracking algorithm would perform. We provide a upper bound of steps for the worst scenario for this problem. The algorithm loops over all the cells in the board, therefore we have M as a factor in the complexity formula. It then boils down to the maximum number of steps we would need for each starting cell (i.e.4⋅3L−1).
            o	Assume the maximum length of word is L, starting from a cell, initially we would have at most 4 directions to explore. Assume each direction is valid (i.e. worst case), during the following exploration, we have at most 3 neighbor cells (excluding the cell where we come from) to explore. As a result, we would traverse at most 4⋅3L−1 cells during the backtracking exploration.
            o	One might wonder what the worst case scenario looks like. Well, here is an example. Imagine, each of the cells in the board contains the letter a, and the word dictionary contains a single word ['aaaa']. Voila. This is one of the worst scenarios that the algorithm would encounter.
            
            o	Note that, the above time complexity is estimated under the assumption that the Trie data structure would not change once built. If we apply the optimization trick to gradually remove the nodes in Trie, we could greatly improve the time complexity, since the cost of backtracking would reduced to zero once we match all the words in the dictionary, i.e. the Trie becomes empty.


        •	Space Complexity: O(N), where N is the total number of letters in the dictionary.
            o	The main space consumed by the algorithm is the Trie data structure we build. In the worst case where there is no overlapping of prefixes among the words, the Trie would have as many nodes as the letters of all words. And optionally, one might keep a copy of words in the Trie as well. As a result, we might need 2N space for the Trie.

        */
        private int rows;
        private int cols;
        private char[][] board = null;
        private List<string> result = new List<string>();

        public List<string> FindWords(char[][] board, string[] words)
        {
            // Step 1). Construct the Trie
            TrieNode root = new TrieNode();
            foreach (string word in words)
            {
                TrieNode node = root;

                foreach (char letter in word)
                {
                    if (node.Children.ContainsKey(letter))
                    {
                        node = node.Children[letter];
                    }
                    else
                    {
                        TrieNode newNode = new TrieNode();
                        node.Children[letter] = newNode;
                        node = newNode;
                    }
                }
                node.Word = word; // store words in Trie
            }

            this.board = board;
            // Step 2). Backtracking starting for each cell in the board
            for (int row = 0; row < board.Length; ++row)
            {
                for (int col = 0; col < board[row].Length; ++col)
                {
                    if (root.Children.ContainsKey(board[row][col]))
                    {
                        Backtracking(row, col, root);
                    }
                }
            }

            return this.result;
        }

        private void Backtracking(int row, int col, TrieNode parent)
        {
            char letter = this.board[row][col];
            TrieNode currNode = parent.Children[letter];

            // check if there is any match
            if (currNode.Word != null)
            {
                this.result.Add(currNode.Word);
                currNode.Word = null;
            }

            // mark the current letter before the EXPLORATION
            this.board[row][col] = '#';

            // explore neighbor cells in around-clock directions: up, right, down, left
            int[] rowOffset = { -1, 0, 1, 0 };
            int[] colOffset = { 0, 1, 0, -1 };
            for (int i = 0; i < 4; ++i)
            {
                int newRow = row + rowOffset[i];
                int newCol = col + colOffset[i];
                if (
                    newRow < 0 ||
                    newRow >= this.board.Length ||
                    newCol < 0 ||
                    newCol >= this.board[0].Length
                )
                {
                    continue;
                }
                if (currNode.Children.ContainsKey(this.board[newRow][newCol]))
                {
                    Backtracking(newRow, newCol, currNode);
                }
            }

            // End of EXPLORATION, restore the original letter in the board.
            this.board[row][col] = letter;

            // Optimization: incrementally remove the leaf nodes
            if (currNode.Children.Count == 0)
            {
                parent.Children.Remove(letter);
            }
        }
        /*
        1631. Path With Minimum Effort
        https://leetcode.com/problems/path-with-minimum-effort/description/
        */
        public int MinimumEffortPath(int[][] heights)
        {

            /*
Approach 1: Brute Force using Backtracking
Complexity Analysis
Let m be the number of rows and n be the number of columns in the matrix heights.
•	Time Complexity : O(3m⋅n). The total number of cells in the matrix is given by m⋅n. For the backtracking, there are at most 4 possible directions to explore, but further, the choices are reduced to 3 (since we won't go back to where we come from). Thus, considering 3 possibilities for every cell in the matrix the time complexity would be O(3m⋅n).
The time complexity is exponential, hence this approach is exhaustive and results in Time Limit Exceeded (TLE).
•	Space Complexity: O(m⋅n) This space will be used to store the recursion stack. As we recursively move to the adjacent cells, in the worst case there could be m⋅n cells in the recursive call stack.
            */

            int result = MinimumEffortPathNaiveBacktrack(heights);
            /*
Approach 2: Variations of Dijkstra's Algorithm
Complexity Analysis
•	Time Complexity : O(m⋅nlog(m⋅n)), where m is the number of rows and n is the number of columns in matrix heights.
It will take O(m⋅n) time to visit every cell in the matrix. The priority queue will contain at most m⋅n cells, so it will take O(log(m⋅n)) time to re-sort the queue after every adjacent cell is added to the queue.
This given as total time complexiy as O(m⋅nlog(m⋅n)).
•	Space Complexity: O(m⋅n), where m is the number of rows and n is the number of columns in matrix heights.
The maximum queue size is equal to the total number of cells in the matrix height which is given by m⋅n. Also, we use a difference matrix of size m⋅n. This gives as time complexity as O(m⋅n+m⋅n) = O(m⋅n)


            */
            result = MinimumEffortPathDijKPQ(heights);

            /*
Approach 3: Union Find - Disjoint Set
Complexity Analysis
Let m be the number of rows and n be the number of columns of the matrix height.
•	Time Complexity : O(m⋅n(log(m⋅n))). We iterate each edge in the matrix. From the above example, it is evident that for a matrix of size 3⋅3, the total number of edges are 12. Thus, for a m⋅n matrix, the total number of edges could be given by (m⋅n⋅2)−(m+n) (3*3*2) - (3+3)), which is roughly equivalent to m⋅n.
For every edge, we find the parent of each cell and perform the union (Union Find). For n elements, the time complexity of Union Find is logn. (Refer Proof Of Time Complexity Of Union Find). Thus for m⋅n cells, the time taken to perform Union Find would be logm⋅n. This gives us total time complexity as, O(m⋅n(log(m⋅n))).
•	Space Complexity : O(m⋅n) , we use arrays edgeList, parent, and rank of size m⋅n.

            
            */
            result = MinimumEffortPathUF(heights);

            /*
Approach 4: Binary Search Using BFS
Complexity Analysis
Let m be the number of rows and n be the number of columns for the matrix height.
•	Time Complexity : O(m⋅n). We do a binary search to calculate the mid values and then do Breadth First Search on the matrix for each of those values.
Binary Search:To perform Binary search on numbers in range (0..106), the time taken would be O(log106).
Breadth First Search: The time complexity for the Breadth First Search for vertices V and edges E is O(V+E) (See our Explore Card on BFS)
Thus, in the matrix of size m⋅n, with m⋅n vertices and m⋅n edges (Refer time complexity of Approach 3), the time complexity to perform Breadth First Search would be O(m⋅n+m⋅n) = O(m⋅n).
This gives us total time complexity as O(log106⋅(m⋅n)) which is equivalent to O(m⋅n).
•	Space Complexity: O(m⋅n), as we use a queue and visited array of size m⋅n

            
            */
            result = MinimumEffortPathBFSBS(heights);

            /*
Approach 5: Binary Search Using DFS
 Complexity Analysis
•	Time Complexity : O(m⋅n). As in Approach 4. The only difference is that we are using Depth First Search instead of Breadth First Search and have similar time complexity.
•	Space Complexity: O(m⋅n), As in Approach 4. In Depth First Search, we use the internal call stack (instead of the queue in Breadth First Search).
           
            */
            result = MinimumEffortPathDFSBS(heights);

            return result;

        }

        public int MinimumEffortPathNaiveBacktrack(int[][] heights)
        {
            return Backtrack(0, 0, heights, heights.Length, heights[0].Length, 0);
        }

        //int[][] directions = new int[][] { new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { -1, 0 } };
        int maxSoFar = int.MaxValue;

        int Backtrack(int x, int y, int[][] heights, int row, int col, int maxDifference)
        {
            if (x == row - 1 && y == col - 1)
            {
                maxSoFar = Math.Min(maxSoFar, maxDifference);
                return maxDifference;
            }
            int currentHeight = heights[x][y];
            heights[x][y] = 0;
            int minEffort = int.MaxValue;
            for (int i = 0; i < 4; i++)
            {
                int adjacentX = x + directions[i][0];
                int adjacentY = y + directions[i][1];
                if (IsValidCell(adjacentX, adjacentY, row, col) && heights[adjacentX][adjacentY] != 0)
                {
                    int currentDifference = Math.Abs(heights[adjacentX][adjacentY] - currentHeight);
                    int maxCurrentDifference = Math.Max(maxDifference, currentDifference);
                    if (maxCurrentDifference < maxSoFar)
                    {
                        int result = Backtrack(adjacentX, adjacentY, heights, row, col, maxCurrentDifference);
                        minEffort = Math.Min(minEffort, result);
                    }
                }
            }
            heights[x][y] = currentHeight;
            return minEffort;
        }

        bool IsValidCell(int x, int y, int rowCount, int colCount)
        {
            return x >= 0 && x <= rowCount - 1 && y >= 0 && y <= colCount - 1;
        }
        public int MinimumEffortPathDijKPQ(int[][] heights)
        {
            int rowCount = heights.Length;
            int colCount = heights[0].Length;
            int[][] differenceMatrix = new int[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                differenceMatrix[i] = new int[colCount];
                Array.Fill(differenceMatrix[i], int.MaxValue);
            }
            differenceMatrix[0][0] = 0;
            PriorityQueue<Cell, int> queue = new PriorityQueue<Cell, int>(Comparer<int>.Create((a, b) => a.CompareTo(b)));
            bool[][] visited = new bool[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                visited[i] = new bool[colCount];
            }
            queue.Enqueue(new Cell(0, 0, differenceMatrix[0][0]), differenceMatrix[0][0]);

            while (queue.Count > 0)
            {
                Cell currentCell = queue.Dequeue();
                visited[currentCell.X][currentCell.Y] = true;
                if (currentCell.X == rowCount - 1 && currentCell.Y == colCount - 1)
                    return currentCell.Difference;
                foreach (int[] direction in directions)
                {
                    int adjacentX = currentCell.X + direction[0];
                    int adjacentY = currentCell.Y + direction[1];
                    if (IsValidCell(adjacentX, adjacentY, rowCount, colCount) && !visited[adjacentX][adjacentY])
                    {
                        int currentDifference = Math.Abs(heights[adjacentX][adjacentY] - heights[currentCell.X][currentCell.Y]);
                        int maxDifference = Math.Max(currentDifference, differenceMatrix[currentCell.X][currentCell.Y]);
                        if (differenceMatrix[adjacentX][adjacentY] > maxDifference)
                        {
                            differenceMatrix[adjacentX][adjacentY] = maxDifference;
                            queue.Enqueue(new Cell(adjacentX, adjacentY, maxDifference), maxDifference);
                        }
                    }
                }
            }
            return differenceMatrix[rowCount - 1][colCount - 1];

            bool IsValidCell(int x, int y, int rowCount, int colCount)
            {
                return x >= 0 && x < rowCount && y >= 0 && y < colCount;
            }

        }

        class Cell
        {
            public int X { get; }
            public int Y { get; }
            public int Difference { get; }

            public Cell(int x, int y, int difference)
            {
                X = x;
                Y = y;
                Difference = difference;
            }

            public Cell(int x, int y)
            {
                X = x;
                Y = y;
            }

        }

        public int MinimumEffortPathUF(int[][] heights)
        {
            int rowCount = heights.Length;
            int columnCount = heights[0].Length;
            if (rowCount == 1 && columnCount == 1) return 0;
            UnionFind unionFind = new UnionFind(heights);
            List<Edge> edgeList = unionFind.EdgeList;
            edgeList.Sort((e1, e2) => e1.Difference - e2.Difference);

            for (int i = 0; i < edgeList.Count; i++)
            {
                int x = edgeList[i].X;
                int y = edgeList[i].Y;
                unionFind.Union(x, y);
                if (unionFind.Find(0) == unionFind.Find(rowCount * columnCount - 1)) return edgeList[i].Difference;
            }
            return -1;
        }


        class UnionFind
        {
            private int[] Parent;
            private int[] Rank;
            public List<Edge> EdgeList;

            public UnionFind(int[][] heights)
            {
                int rowCount = heights.Length;
                int columnCount = heights[0].Length;
                Parent = new int[rowCount * columnCount];
                EdgeList = new List<Edge>();
                Rank = new int[rowCount * columnCount];
                for (int currentRow = 0; currentRow < rowCount; currentRow++)
                {
                    for (int currentCol = 0; currentCol < columnCount; currentCol++)
                    {
                        if (currentRow > 0)
                        {
                            EdgeList.Add(new Edge(currentRow * columnCount + currentCol,
                                                   (currentRow - 1) * columnCount + currentCol,
                                                   Math.Abs(heights[currentRow][currentCol] - heights[currentRow - 1][currentCol])));
                        }
                        if (currentCol > 0)
                        {
                            EdgeList.Add(new Edge(currentRow * columnCount + currentCol,
                                                   currentRow * columnCount + currentCol - 1,
                                                   Math.Abs(heights[currentRow][currentCol] - heights[currentRow][currentCol - 1])));
                        }
                        Parent[currentRow * columnCount + currentCol] = currentRow * columnCount + currentCol;
                    }
                }
            }

            public int Find(int x)
            {
                if (Parent[x] != x) Parent[x] = Find(Parent[x]);
                return Parent[x];
            }

            public void Union(int x, int y)
            {
                int parentX = Find(x);
                int parentY = Find(y);
                if (parentX != parentY)
                {
                    if (Rank[parentX] > Rank[parentY]) Parent[parentY] = parentX;
                    else if (Rank[parentX] < Rank[parentY]) Parent[parentX] = parentY;
                    else
                    {
                        Parent[parentY] = parentX;
                        Rank[parentX] += 1;
                    }
                }
            }
        }

        class Edge
        {
            public int X;
            public int Y;
            public int Difference;

            public Edge(int x, int y, int difference)
            {
                X = x;
                Y = y;
                Difference = difference;
            }
        }
        public int MinimumEffortPathBFSBS(int[][] heights)
        {
            int leftBoundary = 0;
            int rightBoundary = 1000000;
            int result = rightBoundary;
            //Binary Search
            while (leftBoundary <= rightBoundary)
            {
                int midPoint = (leftBoundary + rightBoundary) / 2;
                if (CanReachDestination(heights, midPoint))
                {
                    result = Math.Min(result, midPoint);
                    rightBoundary = midPoint - 1;
                }
                else
                {
                    leftBoundary = midPoint + 1;
                }
            }
            return result;
        }
        // use bfs to check if we can reach destination with max absolute difference k
        bool CanReachDestination(int[][] heights, int maxDifference)
        {
            int rowCount = heights.Length;
            int columnCount = heights[0].Length;
            Queue<Cell> queue = new Queue<Cell>();
            bool[][] visitedCells = new bool[heights.Length][];
            for (int i = 0; i < heights.Length; i++)
            {
                visitedCells[i] = new bool[heights[0].Length];
            }
            queue.Enqueue(new Cell(0, 0));
            visitedCells[0][0] = true;
            while (queue.Count > 0)
            {
                Cell currentCell = queue.Dequeue();
                if (currentCell.X == rowCount - 1 && currentCell.Y == columnCount - 1)
                {
                    return true;
                }
                foreach (int[] direction in directions)
                {
                    int adjacentX = currentCell.X + direction[0];
                    int adjacentY = currentCell.Y + direction[1];
                    if (IsValidCell(adjacentX, adjacentY, rowCount, columnCount) && !visitedCells[adjacentX][adjacentY])
                    {
                        int currentDifference = Math.Abs(heights[adjacentX][adjacentY] - heights[currentCell.X][currentCell.Y]);
                        if (currentDifference <= maxDifference)
                        {
                            visitedCells[adjacentX][adjacentY] = true;
                            queue.Enqueue(new Cell(adjacentX, adjacentY));
                        }
                    }
                }
            }
            return false;
            bool IsValidCell(int x, int y, int rowCount, int columnCount)
            {
                return x >= 0 && x <= rowCount - 1 && y >= 0 && y <= columnCount - 1;
            }

        }
        public int MinimumEffortPathDFSBS(int[][] heights)
        {
            int left = 0;
            int right = 1000000;
            int result = right;
            //Binary Search
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (MinimumEffortPathDFSBS(heights, mid))
                {
                    result = Math.Min(result, mid);
                    right = mid - 1;
                }
                else
                {
                    left = mid + 1;
                }
            }
            return result;
        }
        private bool MinimumEffortPathDFSBS(int[][] heights, int mid)
        {
            int rowCount = heights.Length;
            int colCount = heights[0].Length;
            bool[][] visited = new bool[rowCount][];
            for (int i = 0; i < rowCount; i++)
            {
                visited[i] = new bool[colCount];
            }
            return MinimumEffortPathDFSBSRec(0, 0, heights, visited, rowCount, colCount, mid);
        }
        private bool MinimumEffortPathDFSBSRec(int x, int y, int[][] heights, bool[][] visited, int rowCount, int colCount, int mid)
        {
            if (x == rowCount - 1 && y == colCount - 1)
            {
                return true;
            }
            visited[x][y] = true;
            foreach (int[] direction in directions)
            {
                int adjacentX = x + direction[0];
                int adjacentY = y + direction[1];
                if (IsValidCell(adjacentX, adjacentY, rowCount, colCount) && !visited[adjacentX][adjacentY])
                {
                    int currentDifference = Math.Abs(heights[adjacentX][adjacentY] - heights[x][y]);
                    if (currentDifference <= mid)
                    {
                        if (MinimumEffortPathDFSBSRec(adjacentX, adjacentY, heights, visited, rowCount, colCount, mid))
                            return true;
                    }
                }
            }
            return false;
        }
        /*
        778. Swim in Rising Water
        https://leetcode.com/problems/swim-in-rising-water/
        */
        public int SwimInWater(int[][] grid)
        {
            int n = grid.Length;
            bool IsCell(int r, int c) => r >= 0 && r < n && c >= 0 && c < n;
            int[][] moves = new int[][]{
            new int[]{1,0},
            new int[]{-1,0},
            new int[]{0,1},
            new int[]{0,-1}
            };
            PriorityQueue<int[], int> pq = new(Comparer<int>.Create((a, b) => a.CompareTo(b)));
            pq.Enqueue(new int[] { 0, 0, grid[0][0] }, grid[0][0]);
            grid[0][0] = -1;
            while (pq.Count > 0)
            {
                int[] cur = pq.Dequeue();
                int x = cur[0], y = cur[1], val = cur[2];
                if (x == n - 1 && y == n - 1)
                    return val;
                foreach (var move in moves)
                {
                    var newCell = new int[] { x + move[0], y + move[1], val };
                    if (IsCell(newCell[0], newCell[1]) && grid[newCell[0]][newCell[1]] != -1)
                    {
                        newCell[2] = Math.Max(newCell[2], grid[newCell[0]][newCell[1]]);
                        grid[newCell[0]][newCell[1]] = -1;
                        pq.Enqueue(newCell, newCell[2]);
                    }
                }
            }
            return 0;
        }

        /*
        62. Unique Paths
        https://leetcode.com/problems/unique-paths/description/

        */
        public int UniquePaths(int m, int n)
        {
            /*
 Approach 1: Dynamic Programming           
  Complexity Analysis
•	Time complexity: O(N×M).
•	Space complexity: O(N×M).
          
            */
            int numOfUniquePaths = UniquePathsDP(m, n);

            /*
  Approach 2: Math (Python3 only)          
   Complexity Analysis
•	Time complexity: O((M+N)(log(M+N)loglog(M+N))^2).
•	Space complexity: O(1).
         
            */
            numOfUniquePaths = UniquePathsMaths(m, n);

            return numOfUniquePaths;

        }

        public int UniquePathsDP(int m, int n)
        {
            int[][] d = new int[m][];
            for (int i = 0; i < m; ++i)
            {
                d[i] = new int[n];
                for (int j = 0; j < n; ++j)
                {
                    d[i][j] = 1;
                }
            }

            for (int col = 1; col < m; ++col)
            {
                for (int row = 1; row < n; ++row)
                {
                    d[col][row] = d[col - 1][row] + d[col][row - 1];
                }
            }

            return d[m - 1][n - 1];
        }
        public int UniquePathsMaths(int m, int n)
        {
            long totalPlaces = m + n - 2;
            long minPlaces = Math.Min(m - 1, n - 1);
            long result = 1;
            for (int i = 0; i < minPlaces; i++)
            {
                result = result * (totalPlaces - i) / (i + 1);
            }

            return (int)result;

        }
        /*

        Approach 1: Dynamic Programming
        Complexity Analysis
        •	Time Complexity: O(M×N). The rectangular grid given to us is of size M×N and we process each cell just once.
        •	Space Complexity: O(1). We are utilizing the obstacleGrid as the DP array. Hence, no extra space.

        */
        public int UniquePathsWithObstacles(int[][] obstacleGrid)
        {
            int R = obstacleGrid.Length;
            int C = obstacleGrid[0].Length;
            // If the starting cell has an obstacle, then simply return as there
            // would be no paths to the destination.
            if (obstacleGrid[0][0] == 1)
            {
                return 0;
            }

            // Number of ways of reaching the starting cell = 1.
            obstacleGrid[0][0] = 1;
            // Filling the values for the first column
            for (int i = 1; i < R; i++)
            {
                obstacleGrid[i][0] =
                    (obstacleGrid[i][0] == 0 && obstacleGrid[i - 1][0] == 1) ? 1
                                                                             : 0;
            }

            // Filling the values for the first row
            for (int i = 1; i < C; i++)
            {
                obstacleGrid[0][i] =
                    (obstacleGrid[0][i] == 0 && obstacleGrid[0][i - 1] == 1) ? 1
                                                                             : 0;
            }

            // Starting from cell(1,1) fill up the values
            // No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
            // i.e. From above and left.
            for (int i = 1; i < R; i++)
            {
                for (int j = 1; j < C; j++)
                {
                    if (obstacleGrid[i][j] == 0)
                    {
                        obstacleGrid[i][j] =
                            obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
                    }
                    else
                    {
                        obstacleGrid[i][j] = 0;
                    }
                }
            }

            // Return value stored in rightmost bottommost cell. That is the
            // destination.
            return obstacleGrid[R - 1][C - 1];
        }

        /*
        980. Unique Paths III
        https://leetcode.com/problems/unique-paths-iii/description/

        Approach 1: Backtracking
        Complexity Analysis
        Let N be the total number of cells in the input grid.
        •	Time Complexity: O(3^N)
        o	Although technically we have 4 directions to explore at each step, we have at most 3 directions to try at any moment except the first step.
        The last direction is the direction where we came from, therefore we don't need to explore it, since we have been there before.
        o	In the worst case where none of the cells is an obstacle, we have to explore each cell.
        Hence, the time complexity of the algorithm is O(4∗3^(N−1))=O(3^N).
        •	Space Complexity: O(N)
        o	Thanks to the in-place technique, we did not use any additional memory to keep track of the state.
        o	On the other hand, we apply recursion in the algorithm, which could incur O(N) space in the function call stack.
        o	Hence, the overall space complexity of the algorithm is O(N).


        */
        public int UniquePathsIII(int[][] grid)
        {
            int non_obstacles = 0, start_row = 0, start_col = 0;

            this.rows = grid.Length;
            this.cols = grid[0].Length;

            // step 1). initialize the conditions for backtracking
            //   i.e. initial state and final state
            for (int row = 0; row < rows; ++row)
                for (int col = 0; col < cols; ++col)
                {
                    int cell = grid[row][col];
                    if (cell >= 0)
                        non_obstacles += 1;
                    if (cell == 1)
                    {
                        start_row = row;
                        start_col = col;
                    }
                }

            this.path_count = 0;
            this.grid = grid;

            // kick-off the backtracking
            Backtrack(start_row, start_col, non_obstacles);

            return this.path_count;
        }
        int path_count;
        protected void Backtrack(int row, int col, int remain)
        {
            // base case for the termination of backtracking
            if (this.grid[row][col] == 2 && remain == 1)
            {
                // reach the destination
                this.path_count += 1;
                return;
            }

            // mark the square as visited. case: 0, 1, 2
            int temp = grid[row][col];
            grid[row][col] = -4;
            remain -= 1; // we now have one less square to visit

            // explore the 4 potential directions around
            int[] row_offsets = { 0, 0, 1, -1 };
            int[] col_offsets = { 1, -1, 0, 0 };
            for (int i = 0; i < 4; ++i)
            {
                int next_row = row + row_offsets[i];
                int next_col = col + col_offsets[i];

                if (0 > next_row || next_row >= this.rows ||
                    0 > next_col || next_col >= this.cols)
                    // invalid coordinate
                    continue;

                if (grid[next_row][next_col] < 0)
                    // either obstacle or visited square
                    continue;

                Backtrack(next_row, next_col, remain);
            }

            // unmark the square after the visit
            grid[row][col] = temp;
        }

        /*
      64. Minimum Path Sum
    https://leetcode.com/problems/minimum-path-sum/description/

        */

        public int MinPathSum(int[][] grid)
        {

            /*
            Approach 1: Brute Force
            Complexity Analysis
    •	Time complexity : O(2^(m+n)). For every move, we have at most 2 options.
    •	Space complexity : O(m+n). Recursion of depth m+n.

            */
            int minPathSum = MinPathSumNaive(grid);
            /*
    Approach 2: Dynamic Programming 2D
    Complexity Analysis
    •	Time complexity : O(mn). We traverse the entire matrix once.
    •	Space complexity : O(mn). Another matrix of the same size is used.

            */
            minPathSum = MinPathSumDP2D(grid);
            /*
      Approach 3: Dynamic Programming 1D
      Complexity Analysis

    Time complexity : O(mn). We traverse the entire matrix once.

    Space complexity : O(n). Another array of row size is used.    
            */
            minPathSum = MinPathSumDP1D(grid);
            /*
            Approach 4: Dynamic Programming (Without Extra Space)

        Complexity Analysis

        Time complexity : O(mn). We traverse the entire matrix once.

        Space complexity : O(1). No extra space is used.
            */
            minPathSum = MinPathSumDPSpaceOptimal(grid);

            return minPathSum;

        }
        private int Calculate(int[][] grid, int i, int j)
        {
            if (i == grid.Length || j == grid[0].Length)
                return int.MaxValue;
            if (i == grid.Length - 1 && j == grid[0].Length - 1)
                return grid[i][j];
            return grid[i][j] +
                   Math.Min(Calculate(grid, i + 1, j), Calculate(grid, i, j + 1));
        }

        public int MinPathSumNaive(int[][] grid)
        {
            return Calculate(grid, 0, 0);
        }

        public int MinPathSumDP2D(int[][] grid)
        {
            int[][] dp = new int[grid.Length][];
            for (int i = 0; i < grid.Length; i++) dp[i] = new int[grid[0].Length];
            for (int i = grid.Length - 1; i >= 0; i--)
            {
                for (int j = grid[0].Length - 1; j >= 0; j--)
                {
                    if (i == grid.Length - 1 && j != grid[0].Length - 1)
                        dp[i][j] = grid[i][j] + dp[i][j + 1];
                    else if (j == grid[0].Length - 1 && i != grid.Length - 1)
                        dp[i][j] = grid[i][j] + dp[i + 1][j];
                    else if (j != grid[0].Length - 1 && i != grid.Length - 1)
                        dp[i][j] =
                            grid[i][j] + Math.Min(dp[i + 1][j], dp[i][j + 1]);
                    else
                        dp[i][j] = grid[i][j];
                }
            }

            return dp[0][0];
        }

        public int MinPathSumDP1D(int[][] grid)
        {
            int[] dp = new int[grid[0].Length];
            for (int i = grid.Length - 1; i >= 0; i--)
            {
                for (int j = grid[0].Length - 1; j >= 0; j--)
                {
                    if (i == grid.Length - 1 && j != grid[0].Length - 1)
                        dp[j] = grid[i][j] + dp[j + 1];
                    else if (j == grid[0].Length - 1 && i != grid.Length - 1)
                        dp[j] = grid[i][j] + dp[j];
                    else if (i != grid.Length - 1 && j != grid[0].Length - 1)
                        dp[j] = grid[i][j] + Math.Min(dp[j], dp[j + 1]);
                    else
                        dp[j] = grid[i][j];
                }
            }

            return dp[0];
        }
        public int MinPathSumDPSpaceOptimal(int[][] grid)
        {
            for (int i = grid.Length - 1; i >= 0; i--)
            {
                for (int j = grid[0].Length - 1; j >= 0; j--)
                {
                    if (i == grid.Length - 1 && j != grid[0].Length - 1)
                        grid[i][j] += grid[i][j + 1];
                    else if (j == grid[0].Length - 1 && i != grid.Length - 1)
                        grid[i][j] += grid[i + 1][j];
                    else if (j != grid[0].Length - 1 && i != grid.Length - 1)
                        grid[i][j] += Math.Min(grid[i + 1][j], grid[i][j + 1]);
                }
            }

            return grid[0][0];
        }

        /*
        741. Cherry Pickup
        https://leetcode.com/problems/cherry-pickup/description/

        */
        public int CherryPickup(int[][] grid)
        {
            /*
            Approach #1: Greedy [Wrong Answer]
            Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of grid. Our dynamic programming consists of two for-loops of length N.
•	Space Complexity: O(N^2), the size of dp.

            */
            int maxNumOfCherrysPicked = CherryPickupNaiveWrongAnswer(grid);
            /*
Approach #2: Dynamic Programming (Top Down) (DPTD)
Complexity Analysis
•	Time Complexity: O(N^3), where N is the length of grid. Our dynamic programming has N3 states, and each state is calculated once.
•	Space Complexity: O(N^3), the size of memo.

            */
            maxNumOfCherrysPicked = CherryPickupDPTD(grid);
            /*
Approach #3: Dynamic Programming (Bottom Up)   (DPBU)          
 Complexity Analysis
•	Time Complexity: O(N^3), where N is the length of grid. We have three for-loops of size N.
•	Space Complexity: O(N^2), the sizes of dp and dp2.

            */
            maxNumOfCherrysPicked = CherryPickupDPBU(grid);

            return maxNumOfCherrysPicked;
        }


        public int CherryPickupNaiveWrongAnswer(int[][] grid)
        {
            int ans = 0;
            int[][] path = BestPath(grid);
            if (path == null)
            {
                return 0;
            }
            foreach (int[] step in path)
            {
                ans += grid[step[0]][step[1]];
                grid[step[0]][step[1]] = 0;
            }

            foreach (int[] step in BestPath(grid))
            {
                ans += grid[step[0]][step[1]];
            }

            return ans;
        }

        public int[][] BestPath(int[][] grid)
        {
            int N = grid.Length;
            int[][] dp = new int[N][];
            foreach (int[] row in dp)
            {
                Array.Fill(row, int.MinValue);
            }
            dp[N - 1][N - 1] = grid[N - 1][N - 1];
            for (int r = N - 1; r >= 0; --r)
            {
                for (int c = N - 1; c >= 0; --c)
                {
                    if (grid[r][c] >= 0 && (r != N - 1 || c != N - 1))
                    {
                        dp[r][c] = Math.Max(r + 1 < N ? dp[r + 1][c] : int.MinValue,
                                            c + 1 < N ? dp[r][c + 1] : int.MinValue);
                        dp[r][c] += grid[r][c];
                    }
                }
            }
            if (dp[0][0] < 0)
            {
                return null;
            }
            int[][] ans = new int[2 * N - 1][];
            int i = 0, j = 0, t = 0;
            while (i != N - 1 || j != N - 1)
            {
                if (j + 1 == N || i + 1 < N && dp[i + 1][j] >= dp[i][j + 1])
                {
                    i++;
                }
                else
                {
                    j++;
                }

                ans[t][0] = i;
                ans[t][1] = j;
                t++;
            }
            return ans;
        }

        int N;
        int[][][] memo;
        public int CherryPickupDPTD(int[][] grid)
        {
            this.grid = grid;
            N = grid.Length;
            memo = new int[N][][];
            foreach (int[][] layer in memo)
            {
                foreach (int[] row in layer)
                {
                    Array.Fill(row, int.MinValue);
                }
            }
            return Math.Max(0, CherryPickupDPTDRec(0, 0, 0));
        }
        public int CherryPickupDPTDRec(int r1, int c1, int c2)
        {
            int r2 = r1 + c1 - c2;
            if (N == r1 || N == r2 || N == c1 || N == c2 ||
                    grid[r1][c1] == -1 || grid[r2][c2] == -1)
            {
                return -999999;
            }
            else if (r1 == N - 1 && c1 == N - 1)
            {
                return grid[r1][c1];
            }
            else if (memo[r1][c1][c2] != int.MinValue)
            {
                return memo[r1][c1][c2];
            }
            else
            {
                int ans = grid[r1][c1];
                if (c1 != c2)
                {
                    ans += grid[r2][c2];
                }
                ans += Math.Max(Math.Max(CherryPickupDPTDRec(r1, c1 + 1, c2 + 1), CherryPickupDPTDRec(r1 + 1, c1, c2 + 1)),
                                Math.Max(CherryPickupDPTDRec(r1, c1 + 1, c2), CherryPickupDPTDRec(r1 + 1, c1, c2)));
                memo[r1][c1][c2] = ans;
                return ans;
            }
        }
        public int CherryPickupDPBU(int[][] grid)
        {
            if (grid == null || grid.Length == 0 || grid.Any(row => row.Length != grid.Length))
            {
                throw new ArgumentException("Input must be a valid square matrix.");
            }

            int n = grid.Length;
            int[][] dp = InitializeDPArray(n);

            for (int step = 1; step <= 2 * n - 2; step++)
            {
                int[][] dpNext = InitializeDPArray(n);

                for (int row = Math.Max(0, step - (n - 1)); row <= Math.Min(n - 1, step); row++)
                {
                    for (int col = Math.Max(0, step - (n - 1)); col <= Math.Min(n - 1, step); col++)
                    {
                        if (grid[row][step - row] == -1 || grid[col][step - col] == -1)
                        {
                            continue;
                        }

                        int cherries = (row != col) ? grid[row][step - row] + grid[col][step - col] : grid[row][step - row];

                        for (int prevRow = row - 1; prevRow <= row; prevRow++)
                        {
                            for (int prevCol = col - 1; prevCol <= col; prevCol++)
                            {
                                if (prevRow >= 0 && prevCol >= 0)
                                {
                                    dpNext[row][col] = Math.Max(dpNext[row][col], dp[prevRow][prevCol] + cherries);
                                }
                            }
                        }
                    }
                }

                dp = dpNext;
            }

            return Math.Max(0, dp[n - 1][n - 1]);


        }



        private int[][] InitializeDPArray(int n)
        {
            int[][] dpArray = new int[n][];
            for (int i = 0; i < n; i++)
            {
                dpArray[i] = new int[n];
                Array.Fill(dpArray[i], int.MinValue);
            }
            dpArray[0][0] = grid[0][0];
            return dpArray;

        }

        /*
        51. N-Queens
        https://leetcode.com/problems/n-queens/description/	

        Complexity Analysis
        Given N as the number of queens (which is the same as the width and height of the board),
        •	Time complexity: O(N!)
        Unlike the brute force approach, we will only place queens on squares that aren't under attack. For the first queen, we have N options. For the next queen, we won't attempt to place it in the same column as the first queen, and there must be at least one square attacked diagonally by the first queen as well. Thus, the maximum number of squares we can consider for the second queen is N−2. For the third queen, we won't attempt to place it in 2 columns already occupied by the first 2 queens, and there must be at least two squares attacked diagonally from the first 2 queens. Thus, the maximum number of squares we can consider for the third queen is N−4. This pattern continues, resulting in an approximate time complexity of N!.
        While it costs O(N^2) to build each valid solution, the amount of valid solutions S(N) does not grow nearly as fast as N!, so O(N!+S(N)∗N^2)=O(N!)
        •	Space complexity: O(N^2)
        Extra memory used includes the 3 sets used to store board state, as well as the recursion call stack. All of this scales linearly with the number of queens. However, to keep the board state costs O(N^2), since the board is of size N * N. Space used for the output does not count towards space complexity.

        */
        public class NQueenSolution
        {
            private int size;
            private List<IList<string>> solutions = new List<IList<string>>();

            public IList<IList<string>> SolveNQueens(int n)
            {
                size = n;
                char[][] emptyBoard = new char[size][];
                for (int i = 0; i < n; i++)
                {
                    emptyBoard[i] = new char[size];
                    for (int j = 0; j < n; j++)
                    {
                        emptyBoard[i][j] = '.';
                    }
                }

                Backtrack(0, new HashSet<int>(), new HashSet<int>(), new HashSet<int>(),
                          emptyBoard);
                return solutions;
            }

            // Making use of a helper function to get the
            // solutions in the correct output format
            private List<string> CreateBoard(char[][] state)
            {
                List<string> board = new List<string>();
                for (int row = 0; row < size; row++)
                {
                    string current_row = new string(state[row]);
                    board.Add(current_row);
                }

                return board;
            }

            private void Backtrack(int row, HashSet<int> diagonals,
                                   HashSet<int> antiDiagonals, HashSet<int> cols,
                                   char[][] state)
            {
                // Base case - N queens have been placed
                if (row == size)
                {
                    solutions.Add(CreateBoard(state));
                    return;
                }

                for (int col = 0; col < size; col++)
                {
                    int currDiagonal = row - col;
                    int currAntiDiagonal = row + col;
                    // If the queen is not placeable
                    if (cols.Contains(col) || diagonals.Contains(currDiagonal) ||
                        antiDiagonals.Contains(currAntiDiagonal))
                    {
                        continue;
                    }

                    // "Add" the queen to the board
                    cols.Add(col);
                    diagonals.Add(currDiagonal);
                    antiDiagonals.Add(currAntiDiagonal);
                    state[row][col] = 'Q';
                    // Move on to the next row with the updated board state
                    Backtrack(row + 1, diagonals, antiDiagonals, cols, state);
                    // "Remove" the queen from the board since we have already
                    // explored all valid paths using the above function call
                    cols.Remove(col);
                    diagonals.Remove(currDiagonal);
                    antiDiagonals.Remove(currAntiDiagonal);
                    state[row][col] = '.';
                }
            }
        }

        /*
        52. N-Queens II
https://leetcode.com/problems/n-queens-ii/description/

        Approach: Backtracking
Complexity Analysis
•	Time complexity: O(N!), where N is the number of queens (which is the same as the width and height of the board).
Unlike the brute force approach, we place a queen only on squares that aren't attacked. For the first queen, we have N options. For the next queen, we won't attempt to place it in the same column as the first queen, and there must be at least one square attacked diagonally by the first queen as well. Thus, the maximum number of squares we can consider for the second queen is N−2. For the third queen, we won't attempt to place it in 2 columns already occupied by the first 2 queens, and there must be at least two squares attacked diagonally from the first 2 queens. Thus, the maximum number of squares we can consider for the third queen is N−4. This pattern continues, giving an approximate time complexity of N! at the end.
•	Space complexity: O(N), where N is the number of queens (which is the same as the width and height of the board).
Extra memory used includes the 3 sets used to store board state, as well as the recursion call stack. All of this scales linearly with the number of queens.

        */
        public class TotalNQueensSolution
        {
            private int size;

            public int TotalNQueens(int n)
            {
                size = n;
                return Backtrack(0, new HashSet<int>(), new HashSet<int>(),
                                 new HashSet<int>());
            }

            private int Backtrack(int row, HashSet<int> diagonals,
                                  HashSet<int> antiDiagonals, HashSet<int> cols)
            {
                // Base case - N queens have been placed
                if (row == size)
                {
                    return 1;
                }

                int solutions = 0;
                for (int col = 0; col < size; col++)
                {
                    int currDiagonal = row - col;
                    int currAntiDiagonal = row + col;
                    // If the queen is not placeable
                    if (cols.Contains(col) || diagonals.Contains(currDiagonal) ||
                        antiDiagonals.Contains(currAntiDiagonal))
                    {
                        continue;
                    }

                    // "Add" the queen to the board
                    cols.Add(col);
                    diagonals.Add(currDiagonal);
                    antiDiagonals.Add(currAntiDiagonal);
                    // Move on to the next row with the updated board state
                    solutions += Backtrack(row + 1, diagonals, antiDiagonals, cols);
                    // "Remove" the queen from the board since we have already
                    // explored all valid paths using the above function call
                    cols.Remove(col);
                    diagonals.Remove(currDiagonal);
                    antiDiagonals.Remove(currAntiDiagonal);
                }

                return solutions;
            }
        }

        /*
        54. Spiral Matrix
        https://leetcode.com/problems/spiral-matrix/description/

        */
        public IList<int> SpiralOrder(int[][] matrix)
        {
            /*
Approach 1: Set Up Boundaries (SUB)
Complexity Analysis

Let M be the number of rows and N be the number of columns.
Time complexity: O(M⋅N). This is because we visit each element once.
Space complexity: O(1). This is because we don't use other data structures. Remember that we don't include the output array in the space complexity.
            
            */

            IList<int> spiralOrder = SpiralOrderSUB(matrix);

            /*
           Approach 2: Mark Visited Elements (MVE)
 Complexity Analysis

Let M be the number of rows and N be the number of columns.
Time complexity: O(M⋅N). This is because we visit each element once.
Space complexity: O(1). This is because we don't use other data structures. Remember that we don't consider the output array or the input matrix when calculating the space complexity. However, if we were prohibited from mutating the input matrix, then this would be an O(M⋅N) space solution. This is because we would need to use a bool matrix to track all of the previously seen cells.

            */
            spiralOrder = SpiralOrderMVE(matrix);

            return spiralOrder;

        }
        public IList<int> SpiralOrderSUB(int[][] matrix)
        {
            IList<int> result = new List<int>();
            int rows = matrix.Length;
            int columns = matrix[0].Length;
            int up = 0;
            int left = 0;
            int right = columns - 1;
            int down = rows - 1;
            while (result.Count < rows * columns)
            {
                // Traverse from left to right.
                for (int col = left; col <= right; col++)
                {
                    result.Add(matrix[up][col]);
                }

                // Traverse downwards.
                for (int row = up + 1; row <= down; row++)
                {
                    result.Add(matrix[row][right]);
                }

                // Make sure we are now on a different row.
                if (up != down)
                {
                    // Traverse from right to left.
                    for (int col = right - 1; col >= left; col--)
                    {
                        result.Add(matrix[down][col]);
                    }
                }

                // Make sure we are now on a different column.
                if (left != right)
                {
                    // Traverse upwards.
                    for (int row = down - 1; row > up; row--)
                    {
                        result.Add(matrix[row][left]);
                    }
                }

                left++;
                right--;
                up++;
                down--;
            }

            return result;
        }
        public IList<int> SpiralOrderMVE(int[][] matrix)
        {
            int VISITED = 101;
            int rows = matrix.Length, columns = matrix[0].Length;
            // Four directions that we will move: right, down, left, up.
            int[][] directions =
                new int[4][] { new int[] { 0, 1 }, new int[] { 1, 0 },
                            new int[] { 0, -1 }, new int[] { -1, 0 } };
            // Initial direction: moving right.
            int currentDirection = 0;
            // The number of times we change the direction.
            int changeDirection = 0;
            // Current place that we are at is (row, col).
            // row is the row index; col is the column index.
            int row = 0, col = 0;
            // Store the first element and mark it as visited.
            List<int> result = new List<int> { matrix[0][0] };
            matrix[0][0] = VISITED;
            while (changeDirection < 2)
            {
                while (0 <= row + directions[currentDirection][0] &&
                       row + directions[currentDirection][0] < rows &&
                       0 <= col + directions[currentDirection][1] &&
                       col + directions[currentDirection][1] < columns &&
                       matrix[row + directions[currentDirection][0]]
                             [col + directions[currentDirection][1]] != VISITED)
                {
                    // Reset this to 0 since we did not break and change the
                    // direction.
                    changeDirection = 0;
                    // Calculate the next place that we will move to.
                    row += directions[currentDirection][0];
                    col += directions[currentDirection][1];
                    result.Add(matrix[row][col]);
                    matrix[row][col] = VISITED;
                }

                // Change our direction.
                currentDirection = (currentDirection + 1) % 4;
                // Increment changeDirection because we changed our direction.
                changeDirection++;
            }

            return result;
        }

        /*
        59. Spiral Matrix II
        https://leetcode.com/problems/spiral-matrix-ii/description/

        */
        public int[][] GenerateMatrix(int n)
        {
            /*
          Approach 1: Traverse Layer by Layer in Spiral Form (TLLSF)
  Complexity Analysis
•	Time Complexity: O(n^2). Here, n is given input and we are iterating over n⋅n matrix in spiral form.
•	Space Complexity: O(1) We use constant extra space for storing cnt

            */
            var matrixGenerated = GenerateMatrixTLLSF(n);
            /*
            Approach 2: Optimized spiral traversal (OST)
       Complexity Analysis
•	Time Complexity: O(n^2). Here, n is given input and we are iterating over n⋅n matrix in spiral form.
•	Space Complexity: O(1) We use constant extra space for storing cnt.
     
            */
            matrixGenerated = GenerateMatrixOST(n);

            return matrixGenerated;


        }

        public int[][] GenerateMatrixTLLSF(int n)
        {
            int[][] result = new int[n][];
            for (int i = 0; i < n; i++) result[i] = new int[n];
            int cnt = 1;
            for (int layer = 0; layer < (n + 1) / 2; layer++)
            {
                // direction 1 - traverse from left to right
                for (int ptr = layer; ptr < n - layer; ptr++)
                    result[layer][ptr] = cnt++;
                // direction 2 - traverse from top to bottom
                for (int ptr = layer + 1; ptr < n - layer; ptr++)
                    result[ptr][n - layer - 1] = cnt++;
                // direction 3 - traverse from right to left
                for (int ptr = n - layer - 2; ptr >= layer; ptr--)
                    result[n - layer - 1][ptr] = cnt++;
                // direction 4 - traverse from bottom to top
                for (int ptr = n - layer - 2; ptr > layer; ptr--)
                    result[ptr][layer] = cnt++;
            }

            return result;
        }
        public int[][] GenerateMatrixOST(int n)
        {
            int[][] res = new int[n][];
            for (int i = 0; i < n; i++) res[i] = new int[n];
            int[][] helpers =
                new int[4][] { new int[2] { 0, 1 }, new int[2] { 1, 0 },
                            new int[2] { 0, -1 }, new int[2] { -1, 0 } };
            int val = 1, d = 0, row = 0, col = 0;
            while (val <= n * n)
            {
                res[row][col] = val++;
                int r = (row + helpers[d][0] + n) % n;
                int c = (col + helpers[d][1] + n) % n;
                if (res[r][c] != 0)
                    d = (d + 1) % 4;
                row += helpers[d][0];
                col += helpers[d][1];
            }

            return res;
        }


        /*
        885. Spiral Matrix III
        https://leetcode.com/problems/spiral-matrix-iii/description/

        Complexity Analysis
        Let rows be the number of rows and cols be the number of columns in the matrix.
        •	Time complexity: O(max(rows,cols)^2)
        We fill the traversed matrix with the values on the simulated path. However, we might also move out of the matrix during traversal. The total distance covered depends on max(rows,cols)^2. 
        •	Space complexity: O(rows⋅cols)
        Apart from the traversed matrix, no additional memory is used. The traversed matrix stores all the cells of the matrix, so its size is rows×cols. Therefore, the total space complexity is O(rows⋅cols).


        */
        public int[][] SpiralMatrixIII(int rows, int cols, int rStart, int cStart)
        {
            // Store all possible directions in an array.
            int[][] dir = new int[][] { new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { -1, 0 } };
            int[][] traversed = new int[rows * cols][];
            int idx = 0;

            // Initial step size is 1, value of d represents the current direction.
            for (int step = 1, direction = 0; idx < rows * cols;)
            {
                // direction = 0 -> East, direction = 1 -> South
                // direction = 2 -> West, direction = 3 -> North
                for (int i = 0; i < 2; ++i)
                {
                    for (int j = 0; j < step; ++j)
                    {
                        // Validate the current position
                        if (
                            rStart >= 0 &&
                            rStart < rows &&
                            cStart >= 0 &&
                            cStart < cols
                        )
                        {
                            traversed[idx][0] = rStart;
                            traversed[idx][1] = cStart;
                            ++idx;
                        }
                        // Make changes to the current position.
                        rStart += dir[direction][0];
                        cStart += dir[direction][1];
                    }

                    direction = (direction + 1) % 4;
                }
                ++step;
            }
            return traversed;
        }

        /*
        2326. Spiral Matrix IV
        https://leetcode.com/problems/spiral-matrix-iv/description/

        Time: O(n) Solution
        */
        public int[][] SpiralMatrixIV(int m, int n, ListNode head)
        {
            int[][] res = new int[m][];
            for (int i = 0; i < m; i++) res[i] = new int[n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) res[i][j] = -1;
            int top = 0, bottom = m, l = 0, r = n;

            // Traverse the matrix in spiral form, and update with the values present in the head list.
            // If head reacher NULL pointer break out from the loop, and return the spiral matrix.
            while (head != null)
            {
                for (int i = l; i < r && head != null; i++)
                {
                    res[top][i] = head.Val;
                    head = head.Next;
                }
                top++;
                for (int i = top; i < bottom && head != null; i++)
                {
                    res[i][r - 1] = head.Val;
                    head = head.Next;
                }
                r--;
                for (int i = r - 1; i >= l && head != null; i--)
                {
                    res[bottom - 1][i] = head.Val;
                    head = head.Next;
                }
                bottom--;
                for (int i = bottom - 1; i >= top && head != null; i--)
                {
                    res[i][l] = head.Val;
                    head = head.Next;
                }
                l++;
            }
            return res;
        }

        /*
        57. Insert Interval
        https://leetcode.com/problems/insert-interval/description/

        */
        public class InsertIntervalSol
        {
            public int[][] InsertInterval(int[][] intervals, int[] newInterval)
            {
                /*
    Approach 1: Linear Search (LS)
     Complexity Analysis
    Let N be the number of intervals.
    •	Time complexity: O(N)
    We iterate through the intervals once, and each interval is considered and processed only once.
    •	Space complexity: O(1)
    We only use the result (res) array to store output, so this could be considered O(1).

                */
                var result = InsertIntervalLS(intervals, newInterval);

                /*
    Approach 2: Binary Search (BS)
       Complexity Analysis
    Let N be the number of intervals.
    •	Time complexity: O(N)
    The binary search for finding the position to insert the newInterval has a time complexity of O(logN). However, the insertion of the newInterval into the list may take O(N) time in the worst case, as it could involve shifting elements within the list. Consequently, the overall time complexity is O(N+logN), which simplifies to O(N).
    •	Space complexity: O(N)
    We use the additional space to store the result (res) and perform calculations using res, so it does count towards the space complexity. In the worst case, the size of res will be proportional to the number of intervals in the input list.

                */
                result = InsertIntervalBS(intervals, newInterval);

                return result;

            }

            public int[][] InsertIntervalLS(int[][] intervals, int[] newInterval)
            {
                int n = intervals.Length, i = 0;
                List<int[]> res = new List<int[]>();

                // Case 1: No overlapping before merging intervals
                while (i < n && intervals[i][1] < newInterval[0])
                {
                    res.Add(intervals[i]);
                    i++;
                }

                // Case 2: Overlapping and merging intervals
                while (i < n && newInterval[1] >= intervals[i][0])
                {
                    newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
                    newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
                    i++;
                }

                res.Add(newInterval);

                // Case 3: No overlapping after merging newInterval
                while (i < n)
                {
                    res.Add(intervals[i]);
                    i++;
                }

                return res.ToArray();
            }
            public int[][] InsertIntervalBS(int[][] intervals, int[] newInterval)
            {
                // If the intervals vector is empty, return a vector containing the
                // newInterval
                if (intervals.Length == 0)
                {
                    return new int[][] { newInterval };
                }

                int n = intervals.Length;
                int target = newInterval[0];
                int left = 0, right = n - 1;

                // Binary search to find the position to insert newInterval
                while (left <= right)
                {
                    int mid = (left + right) / 2;
                    if (intervals[mid][0] < target)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                // Insert newInterval at the found position
                List<int[]> result = new List<int[]>();
                for (int i = 0; i < left; i++)
                {
                    result.Add(intervals[i]);
                }

                result.Add(newInterval);
                for (int i = left; i < n; i++)
                {
                    result.Add(intervals[i]);
                }

                // Merge overlapping intervals
                List<int[]> merged = new List<int[]>();
                foreach (int[] interval in result)
                {
                    // If res is empty or there is no overlap, add the interval to the
                    // result
                    if (merged.Count == 0 ||
                        merged[merged.Count - 1][1] < interval[0])
                    {
                        merged.Add(interval);
                        // If there is an overlap, merge the intervals by updating the
                        // end of the last interval in res
                    }
                    else
                    {
                        merged[merged.Count - 1][1] =
                            Math.Max(merged[merged.Count - 1][1], interval[1]);
                    }
                }

                return merged.ToArray();
            }

        }



        /*
        73. Set Matrix Zeroes		
        https://leetcode.com/problems/set-matrix-zeroes/description/

        */
        public class SetZeroesSol
        {
            /*
            Approach 1: Additional Memory Approach (AM)
            Complexity Analysis
•	Time Complexity: O(M×N) where M and N are the number of rows and columns respectively.
•	Space Complexity: O(M+N).

            */
            public void SetZeroesSAM(int[][] matrix)
            {
                int R = matrix.Length;
                int C = matrix[0].Length;
                HashSet<int> rows = new HashSet<int>();
                HashSet<int> cols = new HashSet<int>();
                // Essentially, we mark the rows and columns that are to be made zero
                for (int i = 0; i < R; i++)
                {
                    for (int j = 0; j < C; j++)
                    {
                        if (matrix[i][j] == 0)
                        {
                            rows.Add(i);
                            cols.Add(j);
                        }
                    }
                }

                // Iterate over the array once again and using the rows and cols sets,
                // update the elements.
                for (int i = 0; i < R; i++)
                {
                    for (int j = 0; j < C; j++)
                    {
                        if (rows.Contains(i) || cols.Contains(j))
                        {
                            matrix[i][j] = 0;
                        }
                    }
                }
            }

            /*
            Approach 2: O(1) Space, Efficient Solution
Complexity Analysis
•	Time Complexity : O(M×N)
•	Space Complexity : O(1)

            */
            public void SetZeroesSpaceOptimal(int[][] matrix)
            {
                bool isCol = false;
                int R = matrix.Length;
                int C = matrix[0].Length;
                for (int i = 0; i < R; i++)
                {
                    if (matrix[i][0] == 0)
                    {
                        isCol = true;
                    }

                    for (int j = 1; j < C; j++)
                    {
                        if (matrix[i][j] == 0)
                        {
                            matrix[0][j] = 0;
                            matrix[i][0] = 0;
                        }
                    }
                }

                for (int i = 1; i < R; i++)
                {
                    for (int j = 1; j < C; j++)
                    {
                        if (matrix[i][0] == 0 || matrix[0][j] == 0)
                        {
                            matrix[i][j] = 0;
                        }
                    }
                }

                if (matrix[0][0] == 0)
                {
                    for (int j = 0; j < C; j++)
                    {
                        matrix[0][j] = 0;
                    }
                }

                if (isCol)
                {
                    for (int i = 0; i < R; i++)
                    {
                        matrix[i][0] = 0;
                    }
                }
            }

        }


        /*
        289. Game of Life	
        https://leetcode.com/problems/game-of-life/description/

        */
        public class GameOfLifeSol
        {
            /*
            Approach 1: O(mn) Space Solution
Complexity Analysis
•	Time Complexity: O(M×N), where M is the number of rows and N is the number of columns of the Board.
•	Space Complexity: O(M×N), where M is the number of rows and N is the number of columns of the Board. This is the space occupied by the copy board we created initially.

            */
            public void GameOfLife1(int[][] board)
            {
                // Neighbors array to find 8 neighboring cells for a given cell
                int[] neighbors = { 0, 1, -1 };

                int rows = board.Length;
                int cols = board[0].Length;

                // Create a copy of the original board
                int[][] copyBoard = new int[rows][];

                // Create a copy of the original board
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {
                        copyBoard[row][col] = board[row][col];
                    }
                }

                // Iterate through board cell by cell.
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {

                        // For each cell count the number of live neighbors.
                        int liveNeighbors = 0;

                        for (int i = 0; i < 3; i++)
                        {
                            for (int j = 0; j < 3; j++)
                            {

                                if (!(neighbors[i] == 0 && neighbors[j] == 0))
                                {
                                    int r = (row + neighbors[i]);
                                    int c = (col + neighbors[j]);

                                    // Check the validity of the neighboring cell.
                                    // and whether it was originally a live cell.
                                    // The evaluation is done against the copy, since that is never updated.
                                    if ((r < rows && r >= 0) && (c < cols && c >= 0) && (copyBoard[r][c] == 1))
                                    {
                                        liveNeighbors += 1;
                                    }
                                }
                            }
                        }

                        // Rule 1 or Rule 3
                        if ((copyBoard[row][col] == 1) && (liveNeighbors < 2 || liveNeighbors > 3))
                        {
                            board[row][col] = 0;
                        }
                        // Rule 4
                        if (copyBoard[row][col] == 0 && liveNeighbors == 3)
                        {
                            board[row][col] = 1;
                        }
                    }
                }

            }



            /*
            Approach 2: O(1) Space Solution
            Complexity Analysis
    •	Time Complexity: O(M×N), where M is the number of rows and N is the number of columns of the Board.
    •	Space Complexity: O(1)


            */
            public void GameOfLife2(int[][] board)
            {

                // Neighbors array to find 8 neighboring cells for a given cell
                int[] neighbors = { 0, 1, -1 };

                int rows = board.Length;
                int cols = board[0].Length;

                // Iterate through board cell by cell.
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {

                        // For each cell count the number of live neighbors.
                        int liveNeighbors = 0;

                        for (int i = 0; i < 3; i++)
                        {
                            for (int j = 0; j < 3; j++)
                            {

                                if (!(neighbors[i] == 0 && neighbors[j] == 0))
                                {
                                    int r = (row + neighbors[i]);
                                    int c = (col + neighbors[j]);

                                    // Check the validity of the neighboring cell.
                                    // and whether it was originally a live cell.
                                    if ((r < rows && r >= 0) && (c < cols && c >= 0) && (Math.Abs(board[r][c]) == 1))
                                    {
                                        liveNeighbors += 1;
                                    }
                                }
                            }
                        }

                        // Rule 1 or Rule 3
                        if ((board[row][col] == 1) && (liveNeighbors < 2 || liveNeighbors > 3))
                        {
                            // -1 signifies the cell is now dead but originally was live.
                            board[row][col] = -1;
                        }
                        // Rule 4
                        if (board[row][col] == 0 && liveNeighbors == 3)
                        {
                            // 2 signifies the cell is now live but was originally dead.
                            board[row][col] = 2;
                        }
                    }
                }

                // Get the final representation for the newly updated board.
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {
                        if (board[row][col] > 0)
                        {
                            board[row][col] = 1;
                        }
                        else
                        {
                            board[row][col] = 0;
                        }
                    }
                }
            }


        }

        /*
        240. Search a 2D Matrix II	
        https://leetcode.com/problems/search-a-2d-matrix-ii/

        */
        public class SearchMatrixSol
        {
            /*
            Approach 1: Brute Force

            Complexity Analysis
•	Time complexity : O(nm)
Becase we perform a constant time operation for each element of an
n×m element matrix, the overall time complexity is equal to the
size of the matrix.
•	Space complexity : O(1)
The brute force approach does not allocate more additional space than a
handful of pointers, so the memory footprint is constant.

            */


            public bool SearchMatrix(int[][] matrix, int target)
            {
                for (int i = 0; i < matrix.Length; i++)
                {
                    for (int j = 0; j < matrix[0].Length; j++)
                    {
                        if (matrix[i][j] == target)
                        {
                            return true;
                        }
                    }
                }

                return false;
            }
            /*
    Approach 2: Binary Search
    Complexity Analysis
    •	Time complexity : O(log(n!))
    It's not super obvious how O(log(n!)) time complexity arises
    from this algorithm, so let's analyze it step-by-step. The
    asymptotically-largest amount of work performed is in the main loop,
    which runs for min(m,n) iterations, where m denotes the number
    of rows and n denotes the number of columns. On each iteration, we
    perform two binary searches on array slices of length m−i and
    n−i. Therefore, each iteration of the loop runs in
    O(log(m−i)+log(n−i)) time, where i denotes the current
    iteration. We can simplify this to O(2⋅log(n−i))=O(log(n−i))
    by seeing that, in the worst case, n≈m. To see why, consider
    what happens when n≪m (without loss of generality); n will
    dominate m in the asymptotic analysis. By summing the runtimes of all
    iterations, we get the following expression:
    (1)O(log(n)+log(n−1)+log(n−2)+…+log(1))
    Then, we can leverage the log multiplication rule (log(a)+log(b)=log(ab))
    to rewrite the complexity as:
    (2)O(log(n)+log(n−1)+log(n−2)+…+log(1))=O(log(n⋅(n−1)⋅(n−2)⋅…⋅1))=O(log(1⋅…⋅(n−2)⋅(n−1)⋅n))=O(log(n!))
    Because this time complexity is fairly uncommon, it is worth thinking about
    its relation to the usual analyses. For one, log(n!)=O(nlogn).
    To see why, recall step 1 from the analysis above; there are n terms, each no
    greater than log(n). Therefore, the asymptotic runtime is certainly no worse than
    that of an O(nlogn) algorithm.
    •	Space complexity : O(1)
    Because our binary search implementation does not literally slice out
    copies of rows and columns from matrix, we can avoid allocating
    greater-than-constant memory.

            */
            public bool SearchMatrixBS(int[][] matrix, int target)
            {
                // an empty matrix obviously does not contain `target`
                if (matrix == null || matrix.Length == 0)
                {
                    return false;
                }

                // iterate over matrix diagonals
                int shorterDim = Math.Min(matrix.Length, matrix[0].Length);
                for (int i = 0; i < shorterDim; i++)
                {
                    bool verticalFound = BinarySearch(matrix, target, i, true);
                    bool horizontalFound = BinarySearch(matrix, target, i, false);
                    if (verticalFound || horizontalFound)
                    {
                        return true;
                    }
                }

                return false;
            }
            private bool BinarySearch(int[][] matrix, int target, int start, bool vertical)
            {
                int lo = start;
                int hi = vertical ? matrix[0].Length - 1 : matrix.Length - 1;

                while (hi >= lo)
                {
                    int mid = (lo + hi) / 2;
                    if (vertical)
                    { // searching a column
                        if (matrix[start][mid] < target)
                        {
                            lo = mid + 1;
                        }
                        else if (matrix[start][mid] > target)
                        {
                            hi = mid - 1;
                        }
                        else
                        {
                            return true;
                        }
                    }
                    else
                    { // searching a row
                        if (matrix[mid][start] < target)
                        {
                            lo = mid + 1;
                        }
                        else if (matrix[mid][start] > target)
                        {
                            hi = mid - 1;
                        }
                        else
                        {
                            return true;
                        }
                    }
                }

                return false;
            }


            /*
    Approach 3: Divide and Conquer (DAC)
    Complexity Analysis
    •	Time complexity : O(nlogn)
    First, for ease of analysis, assume that n≈m, as in the
    analysis of approach 2. Also, assign x=n^2=∣matrix∣; this will make
    the master method
    easier to apply. Now, let's model the runtime of the
    divide & conquer approach as a recurrence relation:
    T(x)=2⋅T(x/4)+ sqrt of x
    The first term (2⋅T(x/4)) arises from the fact that we
    recurse on two submatrices of roughly one-quarter size, while
    x comes from the time spent seeking along a O(n)-length
    column for the partition point. After binding the master method variables
    (a=2;b=4;c=0.5) we notice that logba=c. Therefore, this
    recurrence falls under case 2 of the master method, and the following
    falls out:
    T(x)=O(x^c⋅logx)=O(x^0.5⋅logx)=O(((n^2) ^0.5)⋅log(n^2))=O(n⋅log(n^2))=O(2n⋅logn)=O(n⋅logn)
    Extension: what would happen to the complexity if we binary searched for
    the partition point, rather than used a linear scan?
    •	Space complexity : O(logn)
    Although this approach does not fundamentally require
    greater-than-constant addition memory, its use of recursion means that it
    will use memory proportional to the height of its recursion tree. Because
    this approach discards half of matrix on each level of recursion (and
    makes two recursive calls), the height of the tree is bounded by logn.



            */
            private int[][] matrix;
            private int target;

            private bool SearchRec(int left, int up, int right, int down)
            {
                // this submatrix has no height or no width.
                if (left > right || up > down)
                {
                    return false;
                    // `target` is already larger than the largest element or smaller
                    // than the smallest element in this submatrix.
                }
                else if (target < matrix[up][left] || target > matrix[down][right])
                {
                    return false;
                }

                int mid = left + (right - left) / 2;

                // Locate `row` such that matrix[row-1][mid] < target < matrix[row][mid]
                int row = up;
                while (row <= down && matrix[row][mid] <= target)
                {
                    if (matrix[row][mid] == target)
                    {
                        return true;
                    }
                    row++;
                }

                return SearchRec(left, row, mid - 1, down) || SearchRec(mid + 1, up, right, row - 1);
            }

            public bool SearchMatrixDAC(int[][] mat, int targ)
            {
                // cache input values in object to avoid passing them unnecessarily
                // to `searchRec`
                matrix = mat;
                target = targ;

                // an empty matrix obviously does not contain `target`
                if (matrix == null || matrix.Length == 0)
                {
                    return false;
                }

                return SearchRec(0, 0, matrix[0].Length - 1, matrix.Length - 1);
            }

            /*
      Approach 4: Search Space Reduction (SPC)
      Complexity Analysis
•	Time complexity : O(n+m)
The key to the time complexity analysis is noticing that, on every
iteration (during which we do not return true) either row or col is
is decremented/incremented exactly once. Because row can only be
decremented m times and col can only be incremented n times
before causing the while loop to terminate, the loop cannot run for
more than n+m iterations. Because all other work is constant, the
overall time complexity is linear in the sum of the dimensions of the
matrix.
•	Space complexity : O(1)
Because this approach only manipulates a few pointers, its memory
footprint is constant.

            */
            public bool SearchMatrixSPC(int[][] matrix, int target)
            {
                // start our "pointer" in the bottom-left
                int row = matrix.Length - 1;
                int col = 0;

                while (row >= 0 && col < matrix[0].Length)
                {
                    if (matrix[row][col] > target)
                    {
                        row--;
                    }
                    else if (matrix[row][col] < target)
                    {
                        col++;
                    }
                    else
                    { // found it
                        return true;
                    }
                }

                return false;
            }

        }


        /*
        118. Pascal's Triangle
        https://leetcode.com/problems/pascals-triangle/description/

        */
        public class GeneratePascalTriangleSol
        {
            /*
            Approach 1: Dynamic Programming
            Complexity Analysis
•	Time complexity: O(numRows^2)
Although updating each value of triangle happens in constant time, it is performed O(numRows2) times. 
•	Space complexity: O(1)
While O(numRows^2) space is used to store the output, the input and output generally do not count towards the space complexity.

            */
            public IList<IList<int>> IterateDP(int numRows)
            {
                List<IList<int>> triangle = new List<IList<int>>();
                // Base case; first row is always [1].
                triangle.Add(new List<int>());
                triangle[0].Add(1);
                for (int rowNum = 1; rowNum < numRows; rowNum++)
                {
                    List<int> row = new List<int>();
                    List<int> prevRow = (List<int>)triangle[rowNum - 1];
                    // The first row element is always 1.
                    row.Add(1);
                    // Each triangle element (other than the first and last of each row)
                    // is equal to the sum of the elements above-and-to-the-left and
                    // above-and-to-the-right.
                    for (int j = 1; j < rowNum; j++)
                    {
                        row.Add(prevRow[j - 1] + prevRow[j]);
                    }

                    // The last row element is always 1.
                    row.Add(1);
                    triangle.Add(row);
                }

                return triangle;
            }
        }

        /*
        119. Pascal's Triangle II
https://leetcode.com/problems/pascals-triangle-ii/description/
        */
        public class GetNthRowOfPascalTraingleSol
        {
            /*
            Approach 1: Brute Force Recursion
            Complexity Analysis
Complexity Analysis
•	Time complexity : O(2^k). The time complexity recurrence is straightforward:
•	Space complexity : O(k)+O(k)≃O(k).
o	We need O(k) space to store the output of the kth row.
o	At worst, the recursive call stack has a maximum of k calls in memory, each call taking constant space. That's O(k) worst case recursive call stack space.

            */
            public static IList<int> Naive(int rowIndex)
            {
                IList<int> ans = new List<int>();
                for (int i = 0; i <= rowIndex; i++)
                {
                    ans.Add(GetNum(rowIndex, i));
                }

                return ans;
                int GetNum(int row, int col)
                {
                    if (row == 0 || col == 0 || row == col)
                    {
                        return 1;
                    }

                    return GetNum(row - 1, col - 1) + GetNum(row - 1, col);
                }
            }

            /*
            Approach 2: Dynamic Programming 
            Complexity Analysis
•	Time complexity : O(k^2).
o	Simple memoization would make sure that a particular element in a row is only calculated once. Assuming that our memoization cache allows constant time lookup and updation (like a hash-map), it takes constant time to calculate each element in Pascal's triangle.
o	Since calculating a row requires calculating all the previous rows as well, we end up calculating 1+2+3+...+(k+1)=((k+1)(k+2))/2≃k^2 elements for the kth row.
•	Space complexity : O(k)+O(k)≃O(k).
o	Simple memoization would need to hold all 1+2+3+...+(k+1)=((k+1)(k+2))/2 elements in the worst case. That would require O(k^2) space.
o	Saving space by keeping only the latest generated row, we need only O(k) extra space, other than the O(k) space required to store the output.

            */
            public IList<int> DP(int rowIndex)
            {
                Dictionary<(int, int), int> cache = new Dictionary<(int, int), int>();
                List<int> ans = new List<int>();
                for (int i = 0; i <= rowIndex; i++)
                {
                    ans.Add(GetNum(rowIndex, i));
                }

                return ans;

                int GetNum(int row, int col)
                {
                    if (cache.ContainsKey((row, col)))
                    {
                        return cache[(row, col)];
                    }

                    int computedVal = (row == 0 || col == 0 || row == col)
                                          ? 1
                                          : GetNum(row - 1, col - 1) + GetNum(row - 1, col);
                    cache[(row, col)] = computedVal;
                    return computedVal;
                }
            }
            /*
Simple memoization above caches results of deep recursive calls and provides significant savings on runtime.
But, it is worth noting that generating a number for a particular row requires only two numbers from the previous row. Consequently, generating a row only requires numbers from the previous row.
Thus, we could reduce our memory footprint by only keeping the latest row generated, as shown below, and use that to generate a new row.

            */
            public IList<int> DPSpaceOptimal(int rowIndex)
            {
                IList<int> prev = new List<int> { 1 };
                for (int i = 1; i <= rowIndex; i++)
                {
                    IList<int> curr = new List<int>(new int[i + 1]);
                    curr[0] = 1;
                    curr[i] = 1;
                    for (int j = 1; j < i; j++)
                    {
                        curr[j] = prev[j - 1] + prev[j];
                    }

                    prev = curr;
                }

                return prev;
            }

            /*
       Approach 3: Memory-efficient Dynamic Programming

Complexity Analysis
•	Time complexity : O(k^2). Same as the previous dynamic programming approach.
•	Space complexity : O(k). No extra space is used other than that required to hold the output.
•	Although there is no savings in theoretical computational complexity, in practice there are some minor wins:
o	We have one vector/array instead of two. So memory consumption is roughly half.
o	No time wasted in swapping references to vectors for previous and current row.
o	Locality of reference shines through here. Since every read is for consecutive memory locations in the array/vector, we get a performance boost.

       */
            public IList<int> MemoryEfficientDP(int rowIndex)
            {
                IList<int> row = new int[rowIndex + 1];
                for (int i = 0; i <= rowIndex; i++)
                {
                    row[i] = 1;
                }

                for (int i = 1; i < rowIndex; i++)
                {
                    for (int j = i; j > 0; j--)
                    {
                        row[j] += row[j - 1];
                    }
                }

                return row;
            }

            /*
            Approach 4: Math! (specifically, Combinatorics)
            Complexity Analysis
            •	Time complexity : O(k). Each term is calculated once, in constant time.
            •	Space complexity : O(k). No extra space required other than that required to hold the output.

            */
            public IList<int> Maths(int rowIndex)
            {
                List<int> row = new List<int>() { 1 };
                for (int k = 1; k <= rowIndex; k++)
                {
                    row.Add((int)((row[row.Count - 1] * (long)(rowIndex - k + 1)) / k));
                }

                return row;
            }

        }



        /*
        120. Triangle
    https://leetcode.com/problems/triangle/description/

        */
        public class MinPathSumFromTopToBottomOfTraingleSol
        {
            /*
            Approach 1: Dynamic Programming (Bottom-up: In-place)
Complexity Analysis
Let n be the number of rows in the triangle.
•	Time Complexity: O(n^2).
A triangle with n rows and n columns contains (n(n+1))/2=(n^2+n)/2 cells. Recall that in big O notaton, we ignore the less significant terms. This gives us O((n^2+n)/2)=O(n^2) cells. For each cell, we are performing a constant number of operatons, therefore giving us a total time complexity of O(n^2).
•	Space Complexity: O(1).
As we're overwriting the input, we don't need any collections to store our calculations.

            */
            public static int BottomUpInPlaceDP(IList<IList<int>> triangle)
            {
                for (int row = 1; row < triangle.Count; row++)
                {
                    for (int col = 0; col <= row; col++)
                    {
                        int smallestAbove = int.MaxValue;
                        if (col > 0)
                        {
                            smallestAbove = triangle[row - 1][col - 1];
                        }

                        if (col < row)
                        {
                            smallestAbove =
                                Math.Min(smallestAbove, triangle[row - 1][col]);
                        }

                        int path = smallestAbove + triangle[row][col];
                        triangle[row][col] = path;
                    }
                }

                return triangle[triangle.Count - 1].Min();
            }
            /*
            Approach 2: Dynamic Programming (Bottom-up: Auxiliary Space)
Complexity Analysis
•	Time Complexity: O(n^2).
Same as Approach 1. We are still performing a constant number of operations for each cell in the input triangle.
•	Space Complexity: O(n).
We're using two arrays of up to size n each: prevRow and currRow. While this is a higher space complexity than Approach 1, the advantage of this approach is that the input triangle remains unmodified.

            */
            public static int BottomUpAuxSpaceDP(IList<IList<int>> triangle)
            {
                List<int> prevRow = triangle[0].ToList();
                for (int row = 1; row < triangle.Count; row++)
                {
                    List<int> currRow = new List<int>();
                    for (int col = 0; col <= row; col++)
                    {
                        int smallestAbove = int.MaxValue;
                        if (col > 0)
                        {
                            smallestAbove = prevRow[col - 1];
                        }

                        if (col < row)
                        {
                            smallestAbove = Math.Min(smallestAbove, prevRow[col]);
                        }

                        currRow.Add(smallestAbove + triangle[row][col]);
                    }

                    prevRow = currRow;
                }

                return prevRow.Min();
            }

            /*
            Approach 3: Dynamic Programming (Bottom-up: Flip Triangle Upside Down)
Complexity Analysis
The time and space complexity for Approach 3 depends on which implementation you're looking at. 
The in-place implementation has the same complexity analysis as Approach 1, 
whereas the auxiliary space implementation has the same complexity analysis as Approach 2.


            */
            public static int BottomUpFlipTriangleUpsideDownInPlaceDP(IList<IList<int>> triangle)
            {
                for (int row = triangle.Count - 2; row >= 0; row--)
                {
                    for (int col = 0; col <= row; col++)
                    {
                        int bestBelow = Math.Min(triangle[row + 1][col],
                                                 triangle[row + 1][col + 1]);
                        triangle[row][col] += bestBelow;
                    }
                }

                return triangle[0][0];
            }
            public static int BottomUpFlipTriangleUpsideDownAuxSpaceeDP(IList<IList<int>> triangle)
            {
                IList<int> belowRow = triangle[triangle.Count - 1];
                for (int row = triangle.Count - 2; row >= 0; row--)
                {
                    IList<int> currRow = new List<int>();
                    for (int col = 0; col <= row; col++)
                    {
                        int bestBelow = Math.Min(belowRow[col], belowRow[col + 1]);
                        currRow.Add(bestBelow + triangle[row][col]);
                    }

                    belowRow = currRow;
                }

                return belowRow[0];
            }
            /*
            Approach 4: Memoization (Top-Down)
  Complexity Analysis
Let n be the number of rows in the triangle.
•	Time Complexity: O(n^2).
There are two steps to analyzing the time complexity of an algorithm that uses recursive memoization.
Firstly, determine what the cost of each call to the recursive function is. That is, how much time is spent actively executing instructions within a single call. It does not include time spent in functions called by that function.
Secondly, determine how many times the recursive function is called.
Each call to minPath is O(1), because there are no loops. The memoization table ensures that minPath is only called once for each cell. As there are n^2 cells, we get a total time complexity of O(n^2).
•	Space Complexity: O(n^2).
Each time a base case cell is reached, there will be a path of n cells on the run-time stack, going from the triangle tip, down to that base case cell. This means that there is O(n) space on the run-time stack.
Each time a subproblem is solved (a call to minPath), its result is stored in a memoization table. We determined above that there are O(n^2) such subproblems, giving a total space complexity of O(n^2) for the memoization table.
          
            */

            public static int TopDownMemo(IList<IList<int>> triangle)
            {
                Dictionary<string, int> memoTable = new Dictionary<string, int>();
                memoTable.Clear();
                return MinPath(0, 0);


                int MinPath(int row, int col)
                {
                    string paramsKey = row + ":" + col;
                    if (memoTable.ContainsKey(paramsKey))
                    {
                        return memoTable[paramsKey];
                    }

                    int path = triangle[row][col];
                    if (row < triangle.Count - 1)
                    {
                        path += Math.Min(MinPath(row + 1, col), MinPath(row + 1, col + 1));
                    }

                    memoTable[paramsKey] = path;
                    return path;
                }
            }

        }


        /*
        2473. Minimum Cost to Buy Apples
https://leetcode.com/problems/minimum-cost-to-buy-apples/description/
        */

        public class MinCostToBuyApplesSol
        {
            /*
            Approach 1: Shortest Path (modified Dijkstra’s Algo)
Complexity Analysis
Let n be the number of cities and r be the number of roads.
•	Time complexity: O(n⋅(n+r)logn)
Adding each of the r edges from the road array to the graph takes O(r).
We push and pop up to n+r vertices from the heap. Pushing and popping vertices takes logn time. So for n vertices, the shortestPath function takes O((n+r)logn).
In the main program, we call shortestPath n times, once for each city, so the time complexity of calculating the minimum cost to buy an apple for each city is O(n⋅(n+r)logn).
Therefore, the total time complexity is O(r+n⋅(n+r)logn), which we can simplify to O(n⋅(n+r)logn).
•	Space complexity: O(n+r)
The list of lists of size (n+2r) stores the graph representation, the travelCost array of size n stores the travel costs, and the heap can grow up to size n.
Therefore, the overall space complexity is O(n+r).
            */
            public long[] ShortestPathWithModifiedDijkstraAlgo(int n, int[][] roads, int[] appleCost, int k)
            {
                // Store the graph as a list of lists
                // The rows represent the cities (vertices)
                // The columns store an adjacency list of road, cost pairs (edge, weight)
                List<List<(int, int)>> graph = new List<List<(int, int)>>();
                for (int i = 0; i < n; i++)
                {
                    graph.Add(new List<(int, int)>());
                }

                // Add each road to the graph using adjacency lists
                // Store each city at graph[city - 1]
                foreach (int[] road in roads)
                {
                    int cityA = road[0] - 1, cityB = road[1] - 1, cost = road[2];
                    graph[cityA].Add((cityB, cost));
                    graph[cityB].Add((cityA, cost));
                }

                // Find the minimum cost to buy an apple starting in each city
                long[] result = new long[n];
                for (int startCity = 0; startCity < n; startCity++)
                {
                    result[startCity] = ShortestPath(startCity, graph, appleCost, k, n);
                }

                return result;
            }

            // Finds the minimum cost to buy an apple from the start city
            private long ShortestPath(int startCity, List<List<(int, int)>> graph,
                                      int[] appleCost, int k, int n)
            {
                // Stores the travel cost reach each city from the start city
                int[] travelCosts = new int[n];
                Array.Fill(travelCosts, int.MaxValue);
                travelCosts[startCity] = 0;

                // Initialize the heap (priority queue) with the starting city
                PriorityQueue<int[], int> heap = new PriorityQueue<int[], int>(Comparer<int>.Create((a, b) => a.CompareTo(b)));
                heap.Enqueue(new int[] { 0, startCity }, 0);
                long minCost = int.MaxValue;

                while (heap.Count > 0)
                {
                    // Remove the city with the minimum cost from the top of the heap
                    int[] current = heap.Dequeue();
                    int travelCost = current[0];
                    int currCity = current[1];

                    // Update the min cost if the curr city has a smaller total cost
                    minCost = Math.Min(minCost,
                                       (long)appleCost[currCity] + (k + 1) * travelCost);

                    // Add each neighboring city to the heap if an apple is cheaper
                    foreach ((int neighbor, int cost) in graph[currCity])
                    {
                        int nextCost = travelCost + cost;
                        if (nextCost < travelCosts[neighbor])
                        {
                            travelCosts[neighbor] = nextCost;
                            heap.Enqueue(new int[] { nextCost, neighbor }, nextCost);
                        }
                    }
                }
                return minCost;
            }
            /*            
Approach 2: One Pass Shortest Path
Complexity Analysis
Let n be the number of cities and r be the number of roads.
•	Time complexity: O((n+r)log(n+r))
Adding each of the r edges from the road array to the graph takes O(r).
Adding the local cost to buy an apple in each city to the result array takes O(n).
To initialize the heap, we insert n cities into the heap. Pushing vertices to the heap takes logn time, so this step takes O(nlogn)
In the main loop, we push and pop up to n+r pairs from the heap. Pushing and popping vertices from the heap takes logn time where n is the size of the heap. So for n+r pairs, the shortestPath function takes O((n+r)log(n+r)).
Therefore, the total time complexity is O(r+n+nlogn+(n+r)log(n+r)), which we can simplify to O((n+r)log(n+r)).
•	Space complexity: O(n+r)
We use a list of lists of size n+2r to store the graph.
The result array, which stores the minimum cost to buy an apple from each city, is of size n.
The heap, which stores the cities to be explored during the shortest path algorithm, can grow up to size n+r.
Therefore, the overall space complexity is O(n+r).	

            */
            public long[] OnePassShortestPath(int n, int[][] roads, int[] appleCost, int k)
            {
                // Store the graph as a list of lists
                // Each element of the outer list represents a city,
                // and each inner list contains pairs of neighboring city and its cost
                List<List<(int, int)>> graph = new List<List<(int, int)>>();
                for (int i = 0; i < n; ++i)
                {
                    graph.Add(new List<(int, int)>());
                }

                // Add each road to the graph using adjacency lists
                // Store each city at `graph[city - 1]`
                foreach (int[] road in roads)
                {
                    int cityA = road[0] - 1, cityB = road[1] - 1, cost = road[2];
                    graph[cityA].Add((cityB, cost));
                    graph[cityB].Add((cityA, cost));
                }

                // Store the cost to buy an apple in each city 
                // without traveling in the result
                long[] result = new long[n];
                for (int startCity = 0; startCity < n; startCity++)
                {
                    result[startCity] = appleCost[startCity];
                }

                // Initialize the min heap (priority queue) with each starting city
                // Each element of the heap is a pair with the cost and city
                PriorityQueue<(long, int), long> heap = new PriorityQueue<(long, int), long>(Comparer<long>.Create((a, b) => a.CompareTo(b)));

                for (int startCity = 0; startCity < n; startCity++)
                {
                    var cost = (long)appleCost[startCity];
                    heap.Enqueue((cost, startCity), cost);
                }

                // Find the minimum cost to buy an apple starting in each city
                while (heap.Count > 0)
                {
                    (long totalCost, int currCity) = heap.Dequeue();

                    // If we have already found a path to buy an apple
                    // for cheaper than the local apple cost, skip this city
                    if (result[currCity] < totalCost)
                        continue;

                    // Add each neighboring city to the heap if it is cheaper to
                    // start there, travel to the current city and buy an apple 
                    // than buy in the neighboring city
                    foreach ((int neighbor, int cost) in graph[currCity])
                    {
                        if (result[neighbor] > result[currCity] + (k + 1) * cost)
                        {
                            result[neighbor] = result[currCity] + (k + 1) * cost;
                            heap.Enqueue((result[neighbor], neighbor), result[neighbor]);
                        }
                    }
                }
                return result;
            }

        }



        /*
        490. The Maze	
        https://leetcode.com/problems/the-maze/description/
        */
        public class HasPathinMazeSol
        {
            /*
            Approach 1: Depth First Search
            Complexity Analysis
Here, m and n are the number of rows and columns in maze.
•	Time complexity: O(m⋅n⋅(m+n))
o	Initializing the visit array takes O(m⋅n) time.
o	The function dfs visits each node at most once, resulting in O(m⋅n) calls. For each call, we loop through the node's neighbors. To discover neighboring nodes for a node, we check in each direction with a while loop and it would take O(m) steps for vertical directions or O(n) steps for horizontal directions to reach a wall, resulting in O(m+n) operations. It would take O(m⋅n⋅(m+n)) in total for all the nodes.
•	Space complexity: O(m⋅n)
o	The visit array takes O(m⋅n) space.
o	The recursion stack used by dfs can have no more than O(m⋅n) elements in the worst-case scenario. It would take up O(m⋅n) space in that case.

            */
            public bool DFS(int[][] maze, int[] start, int[] destination)
            {
                int m = maze.Length;
                int n = maze[0].Length;
                bool[][] visit = new bool[m][];
                return DFSRec(m, n, maze, start, destination, visit);
            }
            private bool DFSRec(int m, int n, int[][] maze, int[] curr, int[] destination,
            bool[][] visit)
            {
                if (visit[curr[0]][curr[1]])
                {
                    return false;
                }
                if (curr[0] == destination[0] && curr[1] == destination[1])
                {
                    return true;
                }

                visit[curr[0]][curr[1]] = true;
                int[] dirX = { 0, 1, 0, -1 };
                int[] dirY = { -1, 0, 1, 0 };

                for (int i = 0; i < 4; i++)
                {
                    int r = curr[0], c = curr[1];
                    // Move the ball in the chosen direction until it can.
                    while (r >= 0 && r < m && c >= 0 && c < n && maze[r][c] == 0)
                    {
                        r += dirX[i];
                        c += dirY[i];
                    }
                    // Revert the last move to get the cell to which the ball rolls.
                    if (DFSRec(m, n, maze, new int[] { r - dirX[i], c - dirY[i] }, destination, visit))
                    {
                        return true;
                    }
                }
                return false;
            }

            /*
            Approach 2: Breadth First Search
         Complexity Analysis
    Here, m and n are the number of rows and columns in maze.
    •	Time complexity: O(m⋅n⋅(m+n))
    o	Initializing the visit array takes O(m⋅n) time.
    o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can be pushed once, leading to O(m⋅n) operations for m⋅n nodes. We iterate over all the neighbors of each node that is popped out of the queue. To discover neighboring nodes for a node, we check in each direction with a while loop and it would take O(m) steps for vertical directions or O(n) steps for horizontal directions to reach a wall, resulting in O(m+n) operations. It would take O(m⋅n⋅(m+n)) in total for all the nodes.
    •	Space complexity: O(m⋅n)
    o	The visit array takes O(m⋅n) space.
    o	The BFS queue takes O(m⋅n) space in the worst-case because each node is added once.

            */
            public bool BFS(int[][] maze, int[] start, int[] destination)
            {
                int m = maze.Length;
                int n = maze[0].Length;
                bool[][] visit = new bool[m][];
                int[] dirX = { 0, 1, 0, -1 };
                int[] dirY = { -1, 0, 1, 0 };

                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(start);
                visit[start[0]][start[1]] = true;

                while (queue.Count > 0)
                {
                    int[] curr = queue.Dequeue();
                    if (curr[0] == destination[0] && curr[1] == destination[1])
                    {
                        return true;
                    }
                    for (int i = 0; i < 4; i++)
                    {
                        int r = curr[0];
                        int c = curr[1];
                        // Move the ball in the chosen direction until it can.
                        while (r >= 0 && r < m && c >= 0 && c < n && maze[r][c] == 0)
                        {
                            r += dirX[i];
                            c += dirY[i];
                        }
                        // Revert the last move to get the cell to which the ball rolls.
                        r -= dirX[i];
                        c -= dirY[i];
                        if (!visit[r][c])
                        {
                            queue.Enqueue(new int[] { r, c });
                            visit[r][c] = true;
                        }
                    }
                }
                return false;
            }

        }


        /*
        505. The Maze II
        https://leetcode.com/problems/the-maze-ii/description/
        */
        public class HasPathinMazeIISol
        {
            /*
            Approach #1 Depth First Search 
Complexity Analysis
•	Time complexity : O(m∗n∗max(m,n)). Complete traversal of maze will be done in the worst case. Here, m and n refers to the number of rows and columns of the maze. Further, for every current node chosen, we can travel upto a maximum depth of max(m,n) in any direction.
•	Space complexity : O(mn). distance array of size m∗n is used.

            */
            public int DFS(int[][] maze, int[] start, int[] dest)
            {
                int[][] distance = new int[maze.Length][];
                for (int i = 0; i < distance.Length; i++)
                {
                    distance[i] = new int[maze[0].Length];
                    Array.Fill(distance[i], int.MaxValue);
                }
                distance[start[0]][start[1]] = 0;
                DFSRec(maze, start, distance);
                return distance[dest[0]][dest[1]] == int.MaxValue ? -1 : distance[dest[0]][dest[1]];
            }

            public void DFSRec(int[][] maze, int[] start, int[][] distance)
            {
                int[][] directions = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { -1, 0 }, new int[] { 1, 0 } };
                foreach (int[] direction in directions)
                {
                    int x = start[0] + direction[0];
                    int y = start[1] + direction[1];
                    int count = 0;
                    while (x >= 0 && y >= 0 && x < maze.Length && y < maze[0].Length && maze[x][y] == 0)
                    {
                        x += direction[0];
                        y += direction[1];
                        count++;
                    }
                    if (distance[start[0]][start[1]] + count < distance[x - direction[0]][y - direction[1]])
                    {
                        distance[x - direction[0]][y - direction[1]] = distance[start[0]][start[1]] + count;
                        DFSRec(maze, new int[] { x - direction[0], y - direction[1] }, distance);
                    }
                }
            }

            /*
            Approach #2 Using Breadth First Search 
            Complexity Analysis
•	Time complexity : O(m∗n∗max(m,n)). Time complexity : O(m∗n∗max(m,n)). Complete traversal of maze will be done in the worst case. Here, m and n refers to the number of rows and columns of the maze. Further, for every current node chosen, we can travel upto a maximum depth of max(m,n) in any direction.
•	Space complexity : O(mn). queue size can grow upto m∗n in the worst case.

            */
            public int BFS(int[][] maze, int[] start, int[] destination)
            {
                int[][] distance = new int[maze.Length][];
                for (int i = 0; i < distance.Length; i++)
                {
                    distance[i] = new int[maze[0].Length];
                    Array.Fill(distance[i], int.MaxValue);
                }

                distance[start[0]][start[1]] = 0;
                int[][] directions = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { -1, 0 }, new int[] { 1, 0 } };
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(start);

                while (queue.Count > 0)
                {
                    int[] current = queue.Dequeue();
                    foreach (int[] direction in directions)
                    {
                        int x = current[0] + direction[0];
                        int y = current[1] + direction[1];
                        int count = 0;

                        while (x >= 0 && y >= 0 && x < maze.Length && y < maze[0].Length && maze[x][y] == 0)
                        {
                            x += direction[0];
                            y += direction[1];
                            count++;
                        }

                        if (distance[current[0]][current[1]] + count < distance[x - direction[0]][y - direction[1]])
                        {
                            distance[x - direction[0]][y - direction[1]] = distance[current[0]][current[1]] + count;
                            queue.Enqueue(new int[] { x - direction[0], y - direction[1] });
                        }
                    }
                }

                return distance[destination[0]][destination[1]] == int.MaxValue ? -1 : distance[destination[0]][destination[1]];
            }
            /*
            Approach #3 Using Dijkstra Algorithm 
Complexity Analysis**
•	Time complexity : O((mn)^2). Complete traversal of maze will be done in the worst case and function minDistance takes O(mn) time.
•	Space complexity : O(mn). distance array of size m∗n is used.

            */
            public int Dijkstra(int[][] maze, int[] start, int[] dest)
            {
                int[][] distance = new int[maze.Length][];
                for (int i = 0; i < maze.Length; i++)
                {
                    distance[i] = new int[maze[0].Length];
                    Array.Fill(distance[i], int.MaxValue);
                }

                bool[][] visited = new bool[maze.Length][];
                for (int i = 0; i < maze.Length; i++)
                {
                    visited[i] = new bool[maze[0].Length];
                }

                distance[start[0]][start[1]] = 0;
                DijkstraAlgo(maze, distance, visited);
                return distance[dest[0]][dest[1]] == int.MaxValue ? -1 : distance[dest[0]][dest[1]];
            }

            public int[] MinDistance(int[][] distance, bool[][] visited)
            {
                int[] min = { -1, -1 };
                int minValue = int.MaxValue;
                for (int i = 0; i < distance.Length; i++)
                {
                    for (int j = 0; j < distance[0].Length; j++)
                    {
                        if (!visited[i][j] && distance[i][j] < minValue)
                        {
                            min = new int[] { i, j };
                            minValue = distance[i][j];
                        }
                    }
                }
                return min;
            }

            public void DijkstraAlgo(int[][] maze, int[][] distance, bool[][] visited)
            {
                int[][] directions = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { -1, 0 }, new int[] { 1, 0 } };
                while (true)
                {
                    int[] current = MinDistance(distance, visited);
                    if (current[0] < 0)
                        break;
                    visited[current[0]][current[1]] = true;
                    foreach (int[] direction in directions)
                    {
                        int x = current[0] + direction[0];
                        int y = current[1] + direction[1];
                        int count = 0;
                        while (x >= 0 && y >= 0 && x < maze.Length && y < maze[0].Length && maze[x][y] == 0)
                        {
                            x += direction[0];
                            y += direction[1];
                            count++;
                        }
                        if (distance[current[0]][current[1]] + count < distance[x - direction[0]][y - direction[1]])
                        {
                            distance[x - direction[0]][y - direction[1]] = distance[current[0]][current[1]] + count;
                        }
                    }
                }
            }
            /*
            Approach #4 Using Dijkstra Algorithm and Priority Queue
Complexity Analysis**
•	Time complexity : O(mn∗log(mn)). Complete traversal of maze will be done in the worst case giving a factor of mn. Further, poll method is a combination of heapifying(O(log(n))) and removing the top element(O(1)) from the priority queue, and it takes O(n) time for n elements. In the current case, poll introduces a factor of log(mn).
•	Space complexity : O(mn). distance array of size m∗n is used and queue size can grow upto m∗n in worst case.

            */
            public int DijkstraWithPQ(int[][] maze, int[] start, int[] dest)
            {
                int[][] distance = new int[maze.Length][];
                for (int i = 0; i < maze.Length; i++)
                {
                    distance[i] = new int[maze[0].Length];
                    Array.Fill(distance[i], int.MaxValue);
                }
                distance[start[0]][start[1]] = 0;
                Dijkstra(maze, start, distance);
                return distance[dest[0]][dest[1]] == int.MaxValue ? -1 : distance[dest[0]][dest[1]];
            }

            public void Dijkstra(int[][] maze, int[] start, int[][] distance)
            {
                int[][] directions = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { -1, 0 }, new int[] { 1, 0 } };
                PriorityQueue<int[], int> queue = new PriorityQueue<int[], int>(Comparer<int>.Create((a, b) => a.CompareTo(b)));
                queue.Enqueue(new int[] { start[0], start[1], 0 }, 0);
                while (queue.Count > 0)
                {
                    int[] current = queue.Dequeue();
                    if (distance[current[0]][current[1]] < current[2])
                        continue;
                    foreach (int[] direction in directions)
                    {
                        int x = current[0] + direction[0];
                        int y = current[1] + direction[1];
                        int count = 0;
                        while (x >= 0 && y >= 0 && x < maze.Length && y < maze[0].Length && maze[x][y] == 0)
                        {
                            x += direction[0];
                            y += direction[1];
                            count++;
                        }
                        if (distance[current[0]][current[1]] + count < distance[x - direction[0]][y - direction[1]])
                        {
                            distance[x - direction[0]][y - direction[1]] = distance[current[0]][current[1]] + count;
                            var dist = distance[x - direction[0]][y - direction[1]];
                            queue.Enqueue(new int[] { x - direction[0], y - direction[1], dist }, dist);
                        }
                    }
                }
            }
        }


        /*
        499. The Maze III
        https://leetcode.com/problems/the-maze-iii/description/

        */
        public class FindShortestWaySol
        {
            private int[][] directions = new int[][] { new int[] { 0, -1 }, new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 1, 0 } };
            private string[] textDirections = new string[] { "l", "u", "r", "d" };
            private int m;
            private int n;

            /*
            Approach: Dijkstra's Wiht Priority Queue
Complexity Analysis
Given n as the number of squares in maze,
•	Time complexity: O(n⋅logn)
In a typical graph problem, most if not all nodes are visited by the algorithm. In this problem, due to the nature of the edges, the majority of test cases will result in most squares not being visited. However, one can think of a worst-case scenario where O(n) nodes are visited. 
•	Space complexity: O(n)
The heap and seen both require up to O(n) space.
Note: this problem can also be solved using DFS or BFS, but Dijkstra's is the most intuitive option once we recognize the edges are weighted, and also more efficient as performing DFS or BFS without a heap on a weighted graph requires modifications.

            */
            public string DijkstraWithPQ(int[][] maze, int[] ball, int[] hole)
            {
                m = maze.Length;
                n = maze[0].Length;

                PriorityQueue<State,State> heap = new PriorityQueue<State,State>(Comparer<State>.Create((a, b) =>
                {
                    int distA = a.Dist;
                    int distB = b.Dist;

                    if (distA == distB)
                    {
                        return string.Compare(a.Path, b.Path);
                    }

                    return distA - distB;
                }));

                bool[,] seen = new bool[m, n];
                var state= new State(ball[0], ball[1], 0, "");
                heap.Enqueue(state, state);

                while (heap.Count > 0)
                {
                    State curr = heap.Dequeue();
                    int row = curr.Row;
                    int col = curr.Col;

                    if (seen[row, col])
                    {
                        continue;
                    }

                    if (row == hole[0] && col == hole[1])
                    {
                        return curr.Path;
                    }

                    seen[row, col] = true;

                    foreach (State nextState in GetNeighbors(row, col, maze, hole))
                    {
                        int nextRow = nextState.Row;
                        int nextCol = nextState.Col;
                        int nextDist = nextState.Dist;
                        string nextChar = nextState.Path;
                        var newState = new State(nextRow, nextCol, curr.Dist + nextDist, curr.Path + nextChar);
                        heap.Enqueue(newState, newState);
                    }
                }

                return "impossible";
            }

            private bool Valid(int row, int col, int[][] maze)
            {
                if (row < 0 || row >= m || col < 0 || col >= n)
                {
                    return false;
                }

                return maze[row][col] == 0;
            }

            private List<State> GetNeighbors(int row, int col, int[][] maze, int[] hole)
            {
                List<State> neighbors = new List<State>();

                for (int i = 0; i < 4; i++)
                {
                    int dy = directions[i][0];
                    int dx = directions[i][1];
                    string direction = textDirections[i];

                    int currRow = row;
                    int currCol = col;
                    int dist = 0;

                    while (Valid(currRow + dy, currCol + dx, maze))
                    {
                        currRow += dy;
                        currCol += dx;
                        dist++;
                        if (currRow == hole[0] && currCol == hole[1])
                        {
                            break;
                        }
                    }

                    neighbors.Add(new State(currRow, currCol, dist, direction));
                }

                return neighbors;


            }

            class State
            {
                public int Row { get; set; }
                public int Col { get; set; }
                public int Dist { get; set; }
                public string Path { get; set; }

                public State(int row, int col, int dist, string path)
                {
                    Row = row;
                    Col = col;
                    Dist = dist;
                    Path = path;
                }
            }















        }














    }
}

