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
using System.Text;

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
        /* 
        939. Minimum Area Rectangle	
        https://leetcode.com/problems/minimum-area-rectangle/description/
        /https://www.algoexpert.io/questions/minimum-area-rectangle 
        */
        public class MinimumAreaRectangleSol
        {

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
        }

        /* 963. Minimum Area Rectangle II
        https://leetcode.com/problems/minimum-area-rectangle-ii/description/
        https://algo.monster/liteproblems/963
         */
        class MinimumAreaRectangleIISol
        {
            /* Time and Space Complexity
            Time Complexity
            The time complexity of the given code can be determined by analyzing the nested loops and the operations inside them:
            •	The first loop runs over all the points, which gives us O(n).
            •	Nested within the first loop, the second loop runs over all other points except the one chosen by the first loop, contributing O(n-1) complexity.
            •	Inside the second loop, there's another loop that starts from the current index of the second loop and goes over the remaining points, which, in the worst case, contributes O(n/2) (this is an approximation since it depends on the current index of the second loop).
            Putting these together, the complexity of the loops is approximately O(n) * O(n-1) * O(n/2), which simplifies to O(n^3) for the worst case.
            Additionally, within the innermost loop, there are constant-time operations such as checking if a point exists in the set, and basic arithmetic operations. The dot product and the calculation of rectangle area also take constant time, which does not affect the overall O(n^3) time complexity.
            Space Complexity
            The space complexity can be considered by looking at the data structures used:
            •	A set s that holds all the points. In the worst case, every point is unique, so the space complexity for the set is O(n).
            •	Constant amount of extra space is used for variables such as x1, y1, x2, y2, x3, y3, x4, y4, v21, v31, w, h, and ans.
            Therefore, the overall space complexity of the algorithm is O(n) for storing the set of points.
             */
            public double MinAreaFreeRect(int[][] points)
            {
                int n = points.Length;
                // Using a HashSet to store unique representations of points
                HashSet<int> pointSet = new HashSet<int>();
                foreach (int[] point in points)
                {
                    pointSet.Add(Encode(point[0], point[1]));
                }
                // Initialize the minimum area to the maximum possible value
                double minArea = Double.MaxValue;
                // Iterate through all combinations of three points to find the fourth point
                for (int i = 0; i < n; ++i)
                {
                    int x1 = points[i][0], y1 = points[i][1];
                    for (int j = 0; j < n; ++j)
                    {
                        if (j != i)
                        {
                            int x2 = points[j][0], y2 = points[j][1];
                            for (int k = j + 1; k < n; ++k)
                            {
                                if (k != i)
                                {
                                    int x3 = points[k][0], y3 = points[k][1];
                                    // Calculate potential fourth point's coordinates
                                    int x4 = x2 - x1 + x3, y4 = y2 - y1 + y3;
                                    // Check if the fourth point exists in the set
                                    if (pointSet.Contains(Encode(x4, y4)))
                                    {
                                        // Check if the three points form a right angle
                                        if ((x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) == 0)
                                        {
                                            // Calculate the square of the rectangle's width and height
                                            int widthSquared = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
                                            int heightSquared = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1);
                                            // Update the minimum area, if smaller than the current minimum area
                                            minArea = Math.Min(minArea, Math.Sqrt((long)widthSquared * heightSquared));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Return 0 if no rectangle is found, otherwise return the minimum area found
                return minArea == Double.MaxValue ? 0 : minArea;
            }

            // Helper function to encode a 2D point into a unique integer
            private int Encode(int x, int y)
            {
                // We use 40001 as a base for encoding since the problem statement might give a max coordinate value of 40000
                return x * 40001 + y;
            }
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
        /*
        684. Redundant Connection		
        https://leetcode.com/problems/redundant-connection
        https://www.youtube.com/watch?v=P6tEGES63ag
        */
        public class FindRedundantDirectedConnectionSol
        {
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
        }

        /* 685. Redundant Connection II	
        https://leetcode.com/problems/redundant-connection-ii/description/
         */
        public class FindRedundantDirectedConnectionIISol
        {
            /* Approach #1: Depth-First Search [Accepted]
            Complexity Analysis
•	Time Complexity: O(N) where N is the number of vertices (and also the number of edges) in the graph. We perform a depth-first search.
•	Space Complexity: O(N), the size of the graph.

             */
            public int[] DFSIterative(int[][] edges)
            {
                int numberOfEdges = edges.Length;
                Dictionary<int, int> parent = new Dictionary<int, int>();
                List<int[]> candidates = new List<int[]>();

                foreach (int[] edge in edges)
                {
                    if (parent.ContainsKey(edge[1]))
                    {
                        candidates.Add(new int[] { parent[edge[1]], edge[1] });
                        candidates.Add(edge);
                    }
                    else
                    {
                        parent[edge[1]] = edge[0];
                    }
                }

                int root = Orbit(1, parent).Node;
                if (candidates.Count == 0)
                {
                    HashSet<int> cycle = Orbit(root, parent).Seen;
                    int[] result = new int[] { 0, 0 };
                    foreach (int[] edge in edges)
                    {
                        if (cycle.Contains(edge[0]) && cycle.Contains(edge[1]))
                        {
                            result = edge;
                        }
                    }
                    return result;
                }

                Dictionary<int, List<int>> children = new Dictionary<int, List<int>>();
                foreach (int vertex in parent.Keys)
                {
                    int parentVertex = parent[vertex];
                    if (!children.ContainsKey(parentVertex))
                    {
                        children[parentVertex] = new List<int>();
                    }
                    children[parentVertex].Add(vertex);
                }

                bool[] seen = new bool[numberOfEdges + 1];
                seen[0] = true;
                Stack<int> stack = new Stack<int>();
                stack.Push(root);
                while (stack.Count > 0)
                {
                    int node = stack.Pop();
                    if (!seen[node])
                    {
                        seen[node] = true;
                        if (children.ContainsKey(node))
                        {
                            foreach (int child in children[node])
                            {
                                stack.Push(child);
                            }
                        }
                    }
                }
                foreach (bool isSeen in seen)
                {
                    if (!isSeen)
                    {
                        return candidates[0];
                    }
                }
                return candidates[1];
            }

            public OrbitResult Orbit(int node, Dictionary<int, int> parent)
            {
                HashSet<int> seen = new HashSet<int>();
                while (parent.ContainsKey(node) && !seen.Contains(node))
                {
                    seen.Add(node);
                    node = parent[node];
                }
                return new OrbitResult(node, seen);
            }
            public class OrbitResult
            {
                public int Node;
                public HashSet<int> Seen;

                public OrbitResult(int n, HashSet<int> s)
                {
                    Node = n;
                    Seen = s;
                }
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
            this.rowCount = board.Length;
            this.cols = board[0].Length;

            for (int row = 0; row < this.rowCount; row++)
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
            if (row < 0 || row == this.rowCount || col < 0 || col == this.cols ||
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
        private int rowCount;
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
        private readonly int[][] directions = new int[][] { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 } };
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
        public class UniquePathsSol
        {
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
        }

        /* 63. Unique Paths II
        https://leetcode.com/problems/unique-paths-ii/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class UniquePathsIISol
        {
            /* 
            Approach 1: Dynamic Programming

            Complexity Analysis
•	Time Complexity: O(M×N). The rectangular grid given to us is of size M×N and we process each cell just once.
•	Space Complexity: O(1). We are utilizing the obstacleGrid as the DP array. Hence, no extra space.

 */
            public int UsingDP(int[][] obstacleGrid)
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

            this.rowCount = grid.Length;
            this.cols = grid[0].Length;

            // step 1). initialize the conditions for backtracking
            //   i.e. initial state and final state
            for (int row = 0; row < rowCount; ++row)
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
        private int[][] grid; // Input grid
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

                if (0 > next_row || next_row >= this.rowCount ||
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

        /* 931. Minimum Falling Path Sum
        https://leetcode.com/problems/minimum-falling-path-sum/description/
         */
        class MinFallingPathSumSol
        {
            /*
            Approach 1: Brute Force Using Depth First Search
Complexity Analysis
Let N be the length of matrix.
•	Time Complexity: O(N⋅3^N) The solution takes the form of a 3-ary recursion tree where there are 3 possibilities for every node in the tree. The time complexity can be derived as follows,
•	The maximum depth of the recursion tree is equal to the number of rows in the matrix i.e N.
•	Each level (level) of the recursion tree will contain approximately 3level nodes. For example, at level 0 there are 3^0 nodes, for level 1, 3^1 nodes, and so on. Thus, the maximum number of nodes at level N would be approximately 3^N.
•	Thus the time complexity is roughly, O(N⋅3^N).
The time complexity is exponential, hence this approach is exhaustive and results in Time Limit Exceeded (TLE).
•	Space Complexity: O(N) This space will be used to store the recursion stack. As the maximum depth of the tree is N, we will not have more than N recursive calls on the call stack at any time.

            */
            public int DFSNaive(int[][] matrix)
            {

                int minFallingSum = int.MaxValue;
                for (int startCol = 0; startCol < matrix.Length; startCol++)
                {
                    minFallingSum = Math.Min(minFallingSum, FindMinFallingPathSum(matrix, 0, startCol));
                }
                return minFallingSum;
            }

            private int FindMinFallingPathSum(int[][] matrix, int row, int col)
            {
                // check if we are out of the left or right boundary of the matrix
                if (col < 0 || col == matrix.Length)
                {
                    return int.MaxValue;
                }
                //check if we have reached the last row
                if (row == matrix.Length - 1)
                {
                    return matrix[row][col];
                }

                // calculate the minimum falling path sum starting from each possible next step
                int left = FindMinFallingPathSum(matrix, row + 1, col);
                int middle = FindMinFallingPathSum(matrix, row + 1, col + 1);
                int right = FindMinFallingPathSum(matrix, row + 1, col - 1);

                return Math.Min(left, Math.Min(middle, right)) + matrix[row][col];
            }
            /*
            Approach 2: Top Down Dynamic Programming
Complexity Analysis
Let N be the length of matrix.
•	Time Complexity: O(N^2)
For every cell in the matrix, we will compute the result only once and update the memo. For the subsequent calls, we are using the stored results that take O(1) time. There are N^2 cells in the matrix, and thus N2 dp states. So, the time complexity is O(N^2).
•	Space Complexity: O(N^2)
The recursive call stack uses O(N) space. As the maximum depth of the tree is N, we can’t have more than N recursive calls on the call stack at any time. The 2D matrix memo uses O(N^2) space. Thus, the space complexity is O(N)+O(N^2)=O(N^2).

            */
            public int TopDownDP(int[][] matrix)
            {
                int minFallingSum = int.MaxValue;
                int?[][] memo = new int?[matrix.Length][];

                for (int i = 0; i < matrix.Length; i++)
                {
                    memo[i] = new int?[matrix[0].Length];
                }

                // start a DFS (with memoization) from each cell in the top row
                for (int startCol = 0; startCol < matrix[0].Length; startCol++)
                {
                    minFallingSum = Math.Min(minFallingSum,
                        FindMinFallingPathSum(matrix, 0, startCol, memo));
                }
                return minFallingSum;
            }

            private int FindMinFallingPathSum(int[][] matrix, int row, int col, int?[][] memo)
            {
                //base cases
                if (col < 0 || col >= matrix[0].Length)
                {
                    return int.MaxValue;
                }
                //check if we have reached the last row
                if (row == matrix.Length - 1)
                {
                    return matrix[row][col];
                }
                //check if the results are calculated before
                if (memo[row][col].HasValue)
                {
                    return memo[row][col].Value;
                }

                // calculate the minimum falling path sum starting from each possible next step
                int left = FindMinFallingPathSum(matrix, row + 1, col, memo);
                int middle = FindMinFallingPathSum(matrix, row + 1, col + 1, memo);
                int right = FindMinFallingPathSum(matrix, row + 1, col - 1, memo);

                memo[row][col] = Math.Min(left, Math.Min(middle, right)) + matrix[row][col];
                return memo[row][col].Value;
            }

            /*
            Approach 3: Bottom-Up Dynamic Programming (Tabulation)
Complexity Analysis
Let N be the length of matrix.
•	Time Complexity: O(N^2)
o	The nested for loop takes (N^2) times to fill the dp array.
o	Then, takes N time to find the minimum falling path.
o	So, Time Complexity T(n)=O(N^2)+O(N)=O(N^2)
•	Space Complexity: O(N^2). The additional space is used for dp array of size N^2.

            */
            public int BottomUpDPTabulation(int[][] matrix)
            {
                int[][] dp = new int[matrix.Length + 1][];
                for (int i = 0; i < dp.Length; i++)
                {
                    dp[i] = new int[matrix.Length + 1];
                }

                for (int row = matrix.Length - 1; row >= 0; row--)
                {
                    for (int col = 0; col < matrix.Length; col++)
                    {
                        if (col == 0)
                        {
                            dp[row][col] =
                                Math.Min(dp[row + 1][col], dp[row + 1][col + 1]) + matrix[row][col];
                        }
                        else if (col == matrix.Length - 1)
                        {
                            dp[row][col] =
                                Math.Min(dp[row + 1][col], dp[row + 1][col - 1]) + matrix[row][col];
                        }
                        else
                        {
                            dp[row][col] = Math.Min(dp[row + 1][col],
                                Math.Min(dp[row + 1][col + 1], dp[row + 1][col - 1])) + matrix[row][col];
                        }
                    }
                }

                int minFallingSum = int.MaxValue;
                for (int startCol = 0; startCol < matrix.Length; startCol++)
                {
                    minFallingSum = Math.Min(minFallingSum, dp[0][startCol]);
                }
                return minFallingSum;
            }
            /*
            Approach 4: Space Optimized, Bottom-Up Dynamic Programming
            Complexity Analysis
        Let N be the length of matrix.
        •	Time Complexity: O(N^2)
        o	The nested for loop takes (N^2) time.
        o	Then, it takes Ntime to find the minimum falling path.
        o	So, Time Complexity T(n)=O(N^2)+O(N)=O(N^2)
        •	Space Complexity: O(N).
        o	We are using two 1-dimensional arrays dp and currentRow of size N.

            */
            public int BottomUpDPTabulationSpaceOptimal(int[][] matrix)
            {
                int[] dynamicProgrammingArray = new int[matrix.Length + 1];
                for (int currentRow = matrix.Length - 1; currentRow >= 0; currentRow--)
                {
                    int[] currentRowArray = new int[matrix.Length + 1];
                    for (int currentColumn = 0; currentColumn < matrix.Length; currentColumn++)
                    {
                        if (currentColumn == 0)
                        {
                            currentRowArray[currentColumn] =
                                Math.Min(dynamicProgrammingArray[currentColumn], dynamicProgrammingArray[currentColumn + 1]) + matrix[currentRow][currentColumn];
                        }
                        else if (currentColumn == matrix.Length - 1)
                        {
                            currentRowArray[currentColumn] =
                                Math.Min(dynamicProgrammingArray[currentColumn], dynamicProgrammingArray[currentColumn - 1]) + matrix[currentRow][currentColumn];
                        }
                        else
                        {
                            currentRowArray[currentColumn] = Math.Min(dynamicProgrammingArray[currentColumn],
                                Math.Min(dynamicProgrammingArray[currentColumn + 1], dynamicProgrammingArray[currentColumn - 1])) + matrix[currentRow][currentColumn];
                        }
                    }
                    dynamicProgrammingArray = currentRowArray;
                }
                int minimumFallingSum = int.MaxValue;
                for (int startingColumn = 0; startingColumn < matrix.Length; startingColumn++)
                {
                    minimumFallingSum = Math.Min(minimumFallingSum, dynamicProgrammingArray[startingColumn]);
                }
                return minimumFallingSum;
            }

        }

        /* 1289. Minimum Falling Path Sum II
        https://leetcode.com/problems/minimum-falling-path-sum-ii/description/
         */
        public class MinFallingPathSumIISol
        {
            // Initialize a dictionary to cache the result of each sub-problem
            private Dictionary<Tuple<int, int>, int> memo = new Dictionary<Tuple<int, int>, int>();

            /*
            Approach 1: Top-Down Dynamic Programming
            Complexity Analysis
            Let N be the number of rows of the square grid. Every row has N columns.
            •	Time complexity: O(N^3)
            In the main function, we are calling optimal from every element of the first row. Let's analyze every element separately.
            o	Calling optimal(0, 0). Readers can appreciate that due to the recursive nature of the function, all yellow-highlighted sub-problems will be called, and their results will be saved in memo after the first call.

            This is because every cell calls optimal for every column of the next row, except for one in the same column.
            Thus, 1+((N−1)⋅N)−1 sub-problems will be called, which is O(N^2).
            In each sub-problem call, we are traversing linearly in the next row. Thus, the time complexity of each sub-problem call is O(N).
            Hence, the time complexity of optimal(0, 0) is O(N^2⋅N), which is O(N^3).
            o	Calling optimal(0, 1). It will directly call all the cells having red dots on them. 
            o	The value of the yellow-highlighted cell will be fetched from memo. Thus, there will be no recursive call from that cell. There are N−2 such cells, and they will have constant time complexity.
            o	The value of the cell that is not yellow-highlighted will be calculated by calling optimal for N−1 columns of the third row.
            There is 1 such cell, and it will have linear time complexity.
            Hence, time complexity of optimal(0, 1) is O((N−2)⋅1+1⋅N), which is O(N).
            After the end of this call, the optimal value of all yellow-highlighted cells will be cached in memo. 
            o	We have N−2 cells remaining in first row. They will pick the minimum result of N−1 valid cells from the second row.
            Thus, for remaining cells, time complexity will be O((N−2)⋅(N−1)), which is O(N^2).
            Hence, the time complexity of the main function is O(N3+N+N2), which is O(N^3).
            •	Space complexity: O(N^2)
            o	The space complexity of a recursive function depends on the maximum number of recursive calls on the stack.
            At any point in time, there will be at most N recursive calls on the stack, as each recursive call is made from a different row. In each recursive call, we have constant space complexity independent of input size. Therefore, space complexity because of the recursive call stack will be O(N).
            o	We are using a hash map memo to cache the result of each sub-problem. There are N2 such sub-problems. Therefore, space complexity because of caching will be O(N^2).
            o	All other variables use constant space independent of input size.
            Hence, the overall space complexity will be O(N+N^2+1), which is O(N^2).

            */
            public int TopDownDP(int[][] grid)
            {
                // We can select any element from the first row. We will select
                // the element which leads to minimum sum.
                int minimumPathSum = int.MaxValue;
                for (int column = 0; column < grid.Length; column++)
                {
                    minimumPathSum = Math.Min(minimumPathSum, Optimal(0, column, grid));
                }

                // Return the minimum sum
                return minimumPathSum;
            }

            // The Optimal(row, col) function returns the minimum sum of a
            // falling path with non-zero shifts, starting from grid[row][col]
            private int Optimal(int row, int column, int[][] grid)
            {
                // If the last row, then return the value of the cell itself
                if (row == grid.Length - 1)
                {
                    return grid[row][column];
                }

                // If the result of this sub-problem is already cached
                if (memo.ContainsKey(Tuple.Create(row, column)))
                {
                    return memo[Tuple.Create(row, column)];
                }

                // Select grid[row][col], and move on to next row. For next
                // row, choose the cell that leads to the minimum sum
                int nextMinimum = int.MaxValue;
                for (int nextRowColumn = 0; nextRowColumn < grid.Length; nextRowColumn++)
                {
                    if (nextRowColumn != column)
                    {
                        nextMinimum = Math.Min(nextMinimum, Optimal(row + 1, nextRowColumn, grid));
                    }
                }

                // Minimum cost from this cell
                memo[Tuple.Create(row, column)] = grid[row][column] + nextMinimum;
                return memo[Tuple.Create(row, column)];
            }
            /*
            Approach 2: Bottom-Up Dynamic Programming
Complexity Analysis
Let N be the number of rows of the square grid. Every row has N columns.
•	Time complexity: O(N^3)
We are traversing in every cell of the memo array once.
o	For the last row, we do a constant time operation of assigning grid[row][col] to memo[row][col]. There are N such cells, and each cell will take constant time. Thus, the time complexity will be O(N).
o	For the remaining rows, we find a minimum from valid elements of the next row. There are (N−1)⋅N such cells, and each cell will take linear time. Thus, the time complexity will be O((N−1)⋅N⋅N), which is O(N^3).
At the end, we find the minimum from the first row. It will take O(N) time.
Thus, overall time complexity will be O(N+N3+N), which is O(N^3).
•	Space complexity: O(N^2)
We used a two-dimensional array memo of size N×N. Thus, space complexity will be O(N^2). All other variables use constant space independent of input size.

            */
            public int BottomUpDP(int[][] grid)
            {
                // Initialize a two-dimensional array to cache the result of each sub-problem
                int[][] memo = new int[grid.Length][];
                for (int i = 0; i < grid.Length; i++)
                {
                    memo[i] = new int[grid.Length];
                }

                // Fill the base case
                for (int column = 0; column < grid.Length; column++)
                {
                    memo[grid.Length - 1][column] = grid[grid.Length - 1][column];
                }

                // Fill the recursive cases
                for (int row = grid.Length - 2; row >= 0; row--)
                {
                    for (int column = 0; column < grid.Length; column++)
                    {
                        // Select minimum from valid cells of next row
                        int nextMinimum = int.MaxValue;
                        for (int nextRowColumn = 0; nextRowColumn < grid.Length; nextRowColumn++)
                        {
                            if (nextRowColumn != column)
                            {
                                nextMinimum = Math.Min(nextMinimum, memo[row + 1][nextRowColumn]);
                            }
                        }

                        // Minimum cost from this cell
                        memo[row][column] = grid[row][column] + nextMinimum;
                    }
                }

                // Find the minimum from the first row
                int answer = int.MaxValue;
                for (int column = 0; column < grid.Length; column++)
                {
                    answer = Math.Min(answer, memo[0][column]);
                }

                // Return the answer
                return answer;
            }
            /*
            
Approach 3: Bottom-Up Dynamic Programming. Save Minimum and Second Minimum
Complexity Analysis
Let N be the number of rows of the square grid. Every row has N columns.
•	Time complexity: O(N^2)
We are traversing in every cell of the memo array once.
For all the cells, we do two main operations
o	Computing memo[row][col]. In the base case, and even in recursive cases, the operation is constant time.
o	Ensuring loop invariant of next_min1_c and next_min2_c.
Both of these are constant time operations.
Thus, N2 cells take O(1) time. Hence, the overall time complexity will be O(N^2).
•	Space complexity: O(N^2)
We are using a two-dimensional array memo of size N⋅N. Thus, space complexity will be O(N^2). All other variables use constant space independent of input size.

            */
            public int BottomUpDPSaveMinAndSecondMin(int[][] grid)
            {
                // Initialize a two-dimensional array to cache the result of each sub-problem
                int[][] memo = new int[grid.Length][];
                for (int i = 0; i < grid.Length; i++)
                {
                    memo[i] = new int[grid.Length];
                }

                // Minimum and Second Minimum Column Index
                int nextMin1C = -1;
                int nextMin2C = -1;

                // Base Case. Fill and save the minimum and second minimum column index
                for (int col = 0; col < grid.Length; col++)
                {
                    memo[grid.Length - 1][col] = grid[grid.Length - 1][col];
                    if (nextMin1C == -1 || memo[grid.Length - 1][col] <= memo[grid.Length - 1][nextMin1C])
                    {
                        nextMin2C = nextMin1C;
                        nextMin1C = col;
                    }
                    else if (nextMin2C == -1 || memo[grid.Length - 1][col] <= memo[grid.Length - 1][nextMin2C])
                    {
                        nextMin2C = col;
                    }
                }

                // Fill the recursive cases
                for (int row = grid.Length - 2; row >= 0; row--)
                {
                    // Minimum and Second Minimum Column Index of the current row
                    int min1C = -1;
                    int min2C = -1;

                    for (int col = 0; col < grid.Length; col++)
                    {
                        // Select minimum from valid cells of the next row
                        if (col != nextMin1C)
                        {
                            memo[row][col] = grid[row][col] + memo[row + 1][nextMin1C];
                        }
                        else
                        {
                            memo[row][col] = grid[row][col] + memo[row + 1][nextMin2C];
                        }

                        // Save minimum and second minimum column index
                        if (min1C == -1 || memo[row][col] <= memo[row][min1C])
                        {
                            min2C = min1C;
                            min1C = col;
                        }
                        else if (min2C == -1 || memo[row][col] <= memo[row][min2C])
                        {
                            min2C = col;
                        }
                    }

                    // Change of row. Update nextMin1C and nextMin2C
                    nextMin1C = min1C;
                    nextMin2C = min2C;
                }

                // Return the minimum from the first row
                return memo[0][nextMin1C];
            }
            /*
Approach 4: Space-Optimized Bottom-Up Dynamic Programming
Complexity Analysis
Let N be the number of rows of the square grid. Every row has N columns.
•	Time complexity: O(N^2)
We are traversing in every cell of the grid array once.
For all the cells, we are doing two main operations
o	Computing value. It will take constant time.
o	Ensuring loop invariant of next_min1_c, next_min2_c, next_min1, and next_min2.
All these operations are constant time operations.
Thus, N2 cells take O(1) time. Hence, the overall time complexity will be O(N^2).
•	Space complexity: O(1)
We are using only a handful of variables, which are independent of input size. Thus, space complexity will be O(1).
            */
            public int BottomUpDPSpaceOptimal(int[][] grid)
            {
                // Minimum and Second Minimum Column Index
                int nextMin1C = -1;
                int nextMin2C = -1;

                // Minimum and Second Minimum Value
                int nextMin1 = -1;
                int nextMin2 = -1;

                // Find the minimum and second minimum from the last row
                for (int col = 0; col < grid.Length; col++)
                {
                    if (nextMin1 == -1 || grid[grid.Length - 1][col] <= nextMin1)
                    {
                        nextMin2 = nextMin1;
                        nextMin2C = nextMin1C;
                        nextMin1 = grid[grid.Length - 1][col];
                        nextMin1C = col;
                    }
                    else if (nextMin2 == -1 || grid[grid.Length - 1][col] <= nextMin2)
                    {
                        nextMin2 = grid[grid.Length - 1][col];
                        nextMin2C = col;
                    }
                }

                // Fill the recursive cases
                for (int row = grid.Length - 2; row >= 0; row--)
                {
                    // Minimum and Second Minimum Column Index of the current row
                    int min1C = -1;
                    int min2C = -1;

                    // Minimum and Second Minimum Value of current row
                    int min1 = -1;
                    int min2 = -1;

                    for (int col = 0; col < grid.Length; col++)
                    {
                        // Select minimum from valid cells of the next row
                        int value;
                        if (col != nextMin1C)
                        {
                            value = grid[row][col] + nextMin1;
                        }
                        else
                        {
                            value = grid[row][col] + nextMin2;
                        }

                        // Save minimum and second minimum
                        if (min1 == -1 || value <= min1)
                        {
                            min2 = min1;
                            min2C = min1C;
                            min1 = value;
                            min1C = col;
                        }
                        else if (min2 == -1 || value <= min2)
                        {
                            min2 = value;
                            min2C = col;
                        }
                    }

                    // Change of row. Update nextMin1C, nextMin2C, nextMin1, nextMin2
                    nextMin1C = min1C;
                    nextMin2C = min2C;
                    nextMin1 = min1;
                    nextMin2 = min2;
                }

                // Return the minimum from the first row
                return nextMin1;
            }

        }

        /*
        741. Cherry Pickup
        https://leetcode.com/problems/cherry-pickup/description/

        */
        public class CherryPickupSol
        {
            private int[][] grid; // Input grid
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
        }
        /* 1463. Cherry Pickup II
        https://leetcode.com/problems/cherry-pickup-ii/description/
         */
        class CherryPickupIISol
        {
            /*
            Approach #1: Dynamic Programming (Top Down)
            Complexity Analysis
Let M be the number of rows in grid and N be the number of columns in grid.
•	Time Complexity: O(M(N^2)), since our helper function have three variables as input, which have M, N, and N possible values respectively. In the worst case, we have to calculate them all once, so that would cost O(M⋅N⋅N)=O(M(N^2). Also, since we save the results after calculating, we would not have repeated calculation.
•	Space Complexity: O(M(N^2), since our helper function have three variables as input, and they have M, N, and N possible values respectively. We need a map with size of O(M⋅N⋅N)=O(M(N^2) to store the results.

            */
            public int TopDownDP(int[][] grid)
            {
                int m = grid.Length;
                int n = grid[0].Length;
                int[][][] dpCache = new int[m][][];
                // initial all elements to -1 to mark unseen
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        for (int k = 0; k < n; k++)
                        {
                            dpCache[i][j][k] = -1;
                        }
                    }
                }
                return dp(0, 0, n - 1, grid, dpCache);
            }

            private int dp(int row, int col1, int col2, int[][] grid, int[][][] dpCache)
            {
                if (col1 < 0 || col1 >= grid[0].Length || col2 < 0 || col2 >= grid[0].Length)
                {
                    return 0;
                }
                // check cache
                if (dpCache[row][col1][col2] != -1)
                {
                    return dpCache[row][col1][col2];
                }
                // current cell
                int result = 0;
                result += grid[row][col1];
                if (col1 != col2)
                {
                    result += grid[row][col2];
                }
                // transition
                if (row != grid.Length - 1)
                {
                    int max = 0;
                    for (int newCol1 = col1 - 1; newCol1 <= col1 + 1; newCol1++)
                    {
                        for (int newCol2 = col2 - 1; newCol2 <= col2 + 1; newCol2++)
                        {
                            max = Math.Max(max, dp(row + 1, newCol1, newCol2, grid, dpCache));
                        }
                    }
                    result += max;
                }

                dpCache[row][col1][col2] = result;
                return result;
            }
            /*
            Approach #2: Dynamic Programming (Bottom Up)
Complexity Analysis
Let M be the number of rows in grid and N be the number of columns in grid.
•	Time Complexity: O(M(N^2)), since our dynamic programming has three nested for-loops, which have M, N, and N iterations respectively. In total, it costs O(M⋅N⋅N)=O(M(N^2)).
•	Space Complexity: O(M(N^2)) if not use state compression, since our dp array has O(M⋅N⋅N)=O(M(N^2)) elements. O(N^2) if use state compression, since we can reuse the first dimension, and our dp array only has O(N⋅N)=O(N^2) elements.

            */
            public int BottomUpDP(int[][] grid)
            {
                int m = grid.Length;
                int n = grid[0].Length;
                int[][][] dp = new int[m][][];

                for (int row = m - 1; row >= 0; row--)
                {
                    for (int col1 = 0; col1 < n; col1++)
                    {
                        for (int col2 = 0; col2 < n; col2++)
                        {
                            int result = 0;
                            // current cell
                            result += grid[row][col1];
                            if (col1 != col2)
                            {
                                result += grid[row][col2];
                            }
                            // transition
                            if (row != m - 1)
                            {
                                int max = 0;
                                for (int newCol1 = col1 - 1; newCol1 <= col1 + 1; newCol1++)
                                {
                                    for (int newCol2 = col2 - 1; newCol2 <= col2 + 1; newCol2++)
                                    {
                                        if (newCol1 >= 0 && newCol1 < n && newCol2 >= 0 && newCol2 < n)
                                        {
                                            max = Math.Max(max, dp[row + 1][newCol1][newCol2]);
                                        }
                                    }
                                }
                                result += max;
                            }
                            dp[row][col1][col2] = result;
                        }
                    }
                }
                return dp[0][0][n - 1];
            }
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

                PriorityQueue<State, State> heap = new PriorityQueue<State, State>(Comparer<State>.Create((a, b) =>
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
                var state = new State(ball[0], ball[1], 0, "");
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
        /*         2392. Build a Matrix With Conditions
        https://leetcode.com/problems/build-a-matrix-with-conditions/description/
         */
        class BuildMatrixWithConditionsSol
        {
            /*
            Approach 1: Depth-First Search + TopoLogical Sort
            Complexity Analysis
            Let n be the size of the rowConditions and colConditions array.
            •	Time complexity: O(max(k⋅k,n))
            Since the total edges in the graph are n and all the nodes are visited exactly once, the time complexity of the depth-first search operation is O(n).
            The time complexity of creating and filling the values of a k⋅k sized matrix is O(k⋅k). Both these operations are performed independently.
            Therefore, the time complexity is given by O(max(k⋅k,n)).
            •	Space complexity: O(max(k⋅k,n))
            Since the total edges in the graph are n, the space complexity of the depth-first search operation is O(n). The space complexity of creating a k⋅k sized matrix is O(k⋅k). Both these operations are performed independently.
            Therefore, the space complexity is given by O(max(k⋅k,n)).


            */
            public int[][] DFSTopologicalSort(
                int k,
                int[][] rowConditions,
                int[][] colConditions
            )
            {
                // Store the topologically sorted sequences.
                List<int> orderRows = TopologicalSort(rowConditions, k);
                List<int> orderColumns = TopologicalSort(colConditions, k);

                // If no topological sort exists, return empty array.
                if (orderRows.Count == 0 || orderColumns.Count == 0) return new int[0][];

                int[][] matrix = new int[k][];
                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        if (orderRows[i].Equals(orderColumns[j]))
                        {
                            matrix[i][j] = orderRows[i];
                        }
                    }
                }
                return matrix;
            }

            private List<int> TopologicalSort(int[][] edges, int n)
            {
                // Build adjacency list
                List<List<int>> adj = new();
                for (int i = 0; i <= n; i++)
                {
                    adj.Add(new List<int>());
                }
                foreach (int[] edge in edges)
                {
                    adj[edge[0]].Add(edge[1]);
                }

                List<int> order = new List<int>();
                // 0: not visited, 1: visiting, 2: visited
                int[] visited = new int[n + 1];
                bool[] hasCycle = { false };

                // Perform DFS for each node
                for (int i = 1; i <= n; i++)
                {
                    if (visited[i] == 0)
                    {
                        dfs(i, adj, visited, order, hasCycle);
                        // Return empty if cycle detected
                        if (hasCycle[0]) return new List<int>();
                    }
                }

                // Reverse to get the correct order
                order.Reverse();
                return order;
            }

            private void dfs(
                int node,
                List<List<int>> adj,
                int[] visited,
                List<int> order,
                bool[] hasCycle
            )
            {
                visited[node] = 1; // Mark node as visiting
                foreach (int neighbor in adj[node])
                {
                    if (visited[neighbor] == 0)
                    {
                        dfs(neighbor, adj, visited, order, hasCycle);
                        // Early exit if a cycle is detected
                        if (hasCycle[0]) return;
                    }
                    else if (visited[neighbor] == 1)
                    {
                        // Cycle detected
                        hasCycle[0] = true;
                        return;
                    }
                }
                // Mark node as visited
                visited[node] = 2;
                // Add node to the order
                order.Add(node);
            }

            /*
            Approach 2: Kahn's Algorithm
Complexity Analysis
Let n be the size of the rowConditions and colConditions array.
•	Time complexity: O(max(k⋅k,n))
Since the total edges in the graph are n and all the nodes are visited exactly once, the time complexity of the breadth-first search operation is O(n).
The time complexity of creating and filling the values of a k⋅k sized matrix is O(k⋅k). Both these operations are performed independently.
Therefore, the time complexity is given by O(max(k⋅k,n)).
•	Space complexity: O(max(k⋅k,n))
Since the total edges in the graph are n, the space complexity of the breadth-first search operation is O(n).
The space complexity of creating a k⋅k sized matrix is O(k⋅k). Both these operations are performed independently.
Therefore, the space complexity is given by O(max(k⋅k,n)).

            */

            public int[][] KahnsAlgo(
                int k,
                int[][] rowConditions,
                int[][] colConditions
            )
            {
                int[] orderRows = TopologicalSort(rowConditions, k);
                int[] orderColumns = TopologicalSort(colConditions, k);
                if (
                    orderRows.Length == 0 || orderColumns.Length == 0
                ) return new int[0][];
                int[][] matrix = new int[k][];
                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        if (orderRows[i] == orderColumns[j])
                        {
                            matrix[i][j] = orderRows[i];
                        }
                    }
                }
                return matrix;

                int[] TopologicalSort(int[][] edges, int n)
                {
                    List<int>[] adj = new List<int>[n + 1];
                    for (int i = 0; i <= n; i++)
                    {
                        adj[i] = new List<int>();
                    }
                    int[] deg = new int[n + 1], order = new int[n];
                    int idx = 0;
                    foreach (int[] x in edges)
                    {
                        adj[x[0]].Add(x[1]);
                        deg[x[1]]++;
                    }
                    Queue<int> q = new Queue<int>();
                    for (int i = 1; i <= n; i++)
                    {
                        if (deg[i] == 0) q.Enqueue(i);
                    }
                    while (q.Count > 0)
                    {
                        int f = q.Dequeue();
                        order[idx++] = f;
                        n--;
                        foreach (int v in adj[f])
                        {
                            if (--deg[v] == 0) q.Enqueue(v);
                        }
                    }
                    if (n != 0) return new int[0];
                    return order;

                }
            }


        }


        /* 995. Minimum Number of K Consecutive Bit Flips
        https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/description/
         */
        class MinNumberKConsecutiveBitFlipsSol
        {
            /*
            Approach 1: Using an Auxiliary Array
            Complexity Analysis
            Let n be the size of the input array.
            •	Time Complexity: O(n)
            The time complexity is O(n) because we iterate through the input array once, performing constant-time operations inside the loop.
            •	Space Complexity: O(n)
            The space complexity is O(n) because it creates a flipped array of size n to track element states.	

            */
            public int UsingAuxiliaryArray(int[] nums, int k)
            {
                // Keeps track of flipped states
                bool[] flipped = new bool[nums.Length];
                // Tracks valid flips within the past window
                int validFlipsFromPastWindow = 0;
                // Counts total flips needed
                int flipCount = 0;

                for (int i = 0; i < nums.Length; i++)
                {
                    if (i >= k)
                    {
                        // Decrease count of valid flips from the past window if needed
                        if (flipped[i - k])
                        {
                            validFlipsFromPastWindow--;
                        }
                    }

                    // Check if current bit needs to be flipped
                    if (validFlipsFromPastWindow % 2 == nums[i])
                    {
                        // If flipping the window extends beyond the array length, return -1
                        if (i + k > nums.Length)
                        {
                            return -1;
                        }
                        // Increment the count of valid flips and mark current as flipped
                        validFlipsFromPastWindow++;
                        flipped[i] = true;
                        flipCount++;
                    }
                }

                return flipCount;
            }
            /*
            Approach 2: Using a Deque
Complexity Analysis
Let n be the size of the input array.
•	Time complexity: O(n)
The time complexity is O(n) because we make a single linear pass through the input array, performing constant-time operations inside the loop.
•	Space complexity: O(k)
The space complexity is O(k) because it uses a deque flipQueue to track flips within the window size k, resulting in maximum size k.

            */
            public int UsingDeque(int[] nums, int k)
            {
                int n = nums.Length; // Length of the input 
                //Replace below Queue with actual Deque
                Queue<int> flipQueue = new Queue<int>(); // Queue to keep track of flips
                int flipped = 0; // Current flip state
                int result = 0; // Total number of flips

                for (int i = 0; i < n; i++)
                {
                    // Remove the effect of the oldest flip if it's out of the current window
                    if (i >= k)
                    {
                        flipped ^= flipQueue.Dequeue();
                    }

                    // If the current bit is 0 (i.e., it needs to be flipped)
                    if (flipped == nums[i])
                    {
                        // If we cannot flip a subarray starting at index i
                        if (i + k > n)
                        {
                            return -1;
                        }
                        // Add a flip at this position
                        flipQueue.Enqueue(1);
                        flipped ^= 1; // Toggle the flipped state
                        result += 1; // Increment the flip count
                    }
                    else
                    {
                        flipQueue.Enqueue(0);
                    }
                }

                return result;
            }
            /*
            Approach 3: In Constant Space
Complexity Analysis
Let n be the size of input array.
•	Time complexity: O(n)
The algorithm iterates through the input array once with constant time operations inside the loop (comparisons, increments/decrements, and array access). This results in a linear time complexity.
•	Space complexity: O(1)
The algorithm uses constant additional space for variables like currentFlips and totalFlips. It doesn't create any data structures that scale with the input size (n or k). Therefore, the space complexity is constant.

            */
            public int WithConstantSpace(int[] nums, int k)
            {
                int currentFlips = 0; // Tracks the current number of flips
                int totalFlips = 0; // Tracks the total number of flips

                for (int i = 0; i < nums.Length; ++i)
                {
                    // If the window slides out of the range and the leftmost element is
                    // marked as flipped (2), decrement currentFlips
                    if (i >= k && nums[i - k] == 2)
                    {
                        currentFlips--;
                    }

                    // Check if the current bit needs to be flipped
                    if ((currentFlips % 2) == nums[i])
                    {
                        // If flipping would exceed array bounds, return -1
                        if (i + k > nums.Length)
                        {
                            return -1;
                        }
                        // Mark the current bit as flipped
                        nums[i] = 2;
                        currentFlips++;
                        totalFlips++;
                    }
                }

                return totalFlips;
            }

        }

        /* 2167. Minimum Time to Remove All Cars Containing Illegal Goods
        https://leetcode.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/description/
         */
        public class MinimumTimeToRemoveAllCarsContainIllegalGoodsSol
        {
            /*
            Complexity
Time O(n)
Space O(1)
            */
            public int MinimumTime(string s)
            {
                // "left" is the minimum cost to move all the illegal cars from s[0] to s[i] (i + 1 cars in total), this can be done by 
                // either removing all of the cars in this range consecutively starting from the left OR 
                // removing some of them from the left and some of them in the middle. 
                // "res" stands for the minimum cost of moving all the illegal cars, and it is important to understand that it has an upper bound of n, 
                // which equals to the total cost if we remove all the cars consecutively from one end. 
                // The reason we initialize it with n is that we seek to minimize it with other possible ways of removing cars. 
                int n = s.Length, left = 0, res = n;
                for (int i = 0; i < n; ++i)
                {
                    // As explained in the original post, each time when s[i] is illegal, we have the option to either 
                    // remove it in a consecutive fashion starting from the left OR to remove it as if it is picked from the middle. 
                    left = Math.Min(left + (s[i] - '0') * 2, i + 1);

                    // Here is the key part. "left + n - 1 - i" means the total cost with the cars from s[i + 1] to s[n - 1] to be removed 
                    // starting from the right consecutively, and we compare it with the current minimum res. 
                    // An alternative way to look at it is: imagine if we have maintained a dp array,
                    // where dp[i] := cost of removing illegal cars from s[0] to s[i] in the optimal fasion + 
                    // cost of removing illegal cars from s[i + 1] to s[n - 1] consecutively from the right.
                    // As the dp array is filled from 0 to n - 1 using the rules defined above, it covers all possible min cost at each index, 
                    // while avoids optimizing the costs associated with the right portion of the input array. 
                    // And then we find the minimum cost in the dp array, which is the answer we are looking for. 
                    res = Math.Min(res, left + n - 1 - i);
                }
                return res;

            }
        }


        /* 1381. Design a Stack With Increment Operation
        https://leetcode.com/problems/design-a-stack-with-increment-operation/description
         */
        public class StackWithIncrementOperationSol
        {

            /*            Approach 1: Array
            Implementation
           Complexity Analysis
           •	Time complexity: O(1) for push and pop, O(k) for increment
           The push and pop methods both perform a single comparison and at most one array operation, all of which are constant time operations.
           The increment method iterates over k elements in the worst case, thus having a O(k) time complexity.
           •	Space complexity: O(maxSize)
           The overall space complexity is O(maxSize), due to the stackArray which can store at most maxSize elements.

            */
            class CustomStacUsingArray
            {

                // Array to store stack elements
                private int[] stackArray;
                // Index of the top element in the stack
                private int topIndex;

                public CustomStacUsingArray(int maxSize)
                {
                    stackArray = new int[maxSize];
                    topIndex = -1;
                }

                public void Push(int x)
                {
                    if (topIndex < stackArray.Length - 1)
                    {
                        stackArray[++topIndex] = x;
                    }
                }

                public int Pop()
                {
                    if (topIndex >= 0)
                    {
                        return stackArray[topIndex--];
                    }
                    return -1; // Return -1 if the stack is empty
                }

                public void Increment(int k, int val)
                {
                    int limit = Math.Min(k, topIndex + 1);
                    for (int i = 0; i < limit; i++)
                    {
                        stackArray[i] += val;
                    }
                }
            }
            /*
            Approach 2: Linked List
Complexity Analysis
•	Time complexity: O(1) for push and pop, O(k) for increment
The push and pop operations modify the last node in the list, both taking constant time.
In the worst case, the increment method updates k elements, taking O(k) time.
•	Space complexity: O(maxSize)
The stack can store maxSize elements in the worst case.

            */
            class CustomStackUsingLinkedList
            {

                private LinkedList<int> stack;
                private int maxSize;

                public CustomStackUsingLinkedList(int maxSize)
                {
                    // Initialize the stack as a LinkedList for efficient add/remove operations
                    stack = new LinkedList<int>();
                    this.maxSize = maxSize;
                }

                public void Push(int x)
                {
                    // Add the element to the top of the stack if it hasn't reached maxSize
                    if (stack.Count < maxSize)
                    {
                        stack.AddLast(x);
                    }
                }

                public int Pop()
                {
                    // Return -1 if the stack is empty, otherwise remove and return the top element
                    if (stack.Count == 0) return -1;

                    var lastElem = stack.Last();
                    stack.RemoveLast();
                    return lastElem;
                }

                public void Increment(int k, int val)
                {
                    // Increment the bottom k elements (or all elements if k > stack size)
                    var iterator = stack.First;

                    while (iterator != null && k > 0)
                    {
                        iterator.Value += val;
                        iterator = iterator.Next;
                        k--;
                    }
                }
            }
            /*
            Approach 3: Array using Lazy Propagation
Complexity Analysis
•	Time complexity: O(1) for all operations
The push, pop, and increment methods perform only constant time operations (comparisons and array operations).
•	Space complexity: O(maxSize)
The stackArray and the incrementArray arrays both have a size of maxSize. Thus, the overall space complexity of the algorithm is O(2⋅maxSize)=O(maxSize)	

            */
            class CustomStackUsingArrayWithLazyPropogation
            {

                // Array to store stack elements
                private int[] stackArray;

                // Array to store increments for lazy propagation
                private int[] incrementArray;

                // Current top index of the stack
                private int topIndex;

                public CustomStackUsingArrayWithLazyPropogation(int maxSize)
                {
                    stackArray = new int[maxSize];
                    incrementArray = new int[maxSize];
                    topIndex = -1;
                }

                public void Push(int x)
                {
                    if (topIndex < stackArray.Length - 1)
                    {
                        stackArray[++topIndex] = x;
                    }
                }

                public int Pop()
                {
                    if (topIndex < 0)
                    {
                        return -1;
                    }

                    // Calculate the actual value with increment
                    int result = stackArray[topIndex] + incrementArray[topIndex];

                    // Propagate the increment to the element below
                    if (topIndex > 0)
                    {
                        incrementArray[topIndex - 1] += incrementArray[topIndex];
                    }

                    // Reset the increment for this position
                    incrementArray[topIndex] = 0;

                    topIndex--;
                    return result;
                }

                public void Increment(int k, int val)
                {
                    if (topIndex >= 0)
                    {
                        // Apply increment to the topmost element of the range
                        int incrementIndex = Math.Min(topIndex, k - 1);
                        incrementArray[incrementIndex] += val;
                    }
                }
            }

        }

        /* 2334. Subarray With Elements Greater Than Varying Threshold
        https://leetcode.com/problems/subarray-with-elements-greater-than-varying-threshold/description/
         */
        class ValidSubarraySizeSol
        {
            /*
Approach: Monotonic Stack
            */
            public int UsingMonotonicStack(int[] numbers, int threshold)
            {
                //the thing is. The next_smaller and the prev_smaller is helping us to find hoow long we can expand our element .
                //Expanding element is in both side how long we have the element that have eigther smae as our value or greater than our value;

                int length = numbers.Length;
                int[] nextSmall = new int[length];
                int[] prevSmall = new int[length];
                Stack<int> stack = new Stack<int>();
                stack.Push(0);
                Array.Fill(nextSmall, length);
                Array.Fill(prevSmall, -1);

                for (int i = 1; i < length; i++)
                {
                    while (stack.Count > 0 && numbers[stack.Peek()] >= numbers[i])
                    {
                        stack.Pop();
                    }
                    if (stack.Count != 0)
                    {
                        prevSmall[i] = stack.Peek();
                    }
                    stack.Push(i);
                }

                stack = new Stack<int>();
                stack.Push(length - 1);

                for (int i = length - 2; i >= 0; i--)
                {
                    while (stack.Count > 0 && numbers[stack.Peek()] >= numbers[i])
                    {
                        stack.Pop();
                    }
                    if (stack.Count != 0)
                    {
                        nextSmall[i] = stack.Peek();
                    }
                    stack.Push(i);
                }

                for (int i = 0; i < length; i++)
                {
                    int subarrayLength = nextSmall[i] - prevSmall[i] - 1; // representing the lenth of our sope. in 
                                                                          // representing the lenth of our sope. in 
                                                                          // test case 1 lets talk about 3 it prev smaller index=0 and next smaller index =4;
                                                                          // so it can be expandex 4-0-1=3 --> and those 3 element are 3 4 3.
                    if (threshold / (double)subarrayLength < numbers[i])
                    {
                        // here if we have 3 element in our array and our threshold is 6 then, all the elemet 
                        //must have 6/3 =2  greater than 2 as value;
                        // now testcase:1 ,we have seen that the max expanding length for 3 ie(nums[1] or nums[3]) is 3 and that reprsent 
                        //that 3 element have 3 or more than 3 value. so they are definitly grater than 2.
                        // 6/3 <3 --> thats a valid case. 

                        return subarrayLength;
                    }
                }
                return -1;
            }
        }

        /* 514. Freedom Trail
        https://leetcode.com/problems/freedom-trail/description/
         */
        class FindRotateStepsSol
        {
            private const int MAX = int.MaxValue;

            /*
            Approach 0: Top-Down Dynamic Programming - TLE

            */
            public int TopDownDPNaive(String ring, String key)
            {
                return TryLock(0, 0, ring, key, MAX);
            }

            // Find the minimum steps between two indexes of ring
            private int CountSteps(int curr, int next, int ringLength)
            {
                int stepsBetween = Math.Abs(curr - next);
                int stepsAround = ringLength - stepsBetween;
                return Math.Min(stepsBetween, stepsAround);
            }

            // Find the minimum number of steps to spell the keyword
            public int TryLock(int ringIndex, int keyIndex, String ring, String key, int minSteps)
            {
                // If we reach the end of the key, it has been spelled
                if (keyIndex == key.Length)
                {
                    return 0;
                }
                // For each occurrence of the character at key_index of key in ring
                // Calculate the minimum steps to that character from the ringIndex of ring
                for (int i = 0; i < ring.Length; i++)
                {
                    if (ring[i] == key[keyIndex])
                    {
                        int totalSteps = CountSteps(ringIndex, i, ring.Length) + 1 +
                                                    TryLock(i, keyIndex + 1, ring, key, MAX);
                        minSteps = Math.Min(minSteps, totalSteps);
                    }
                }
                return minSteps;
            }

            /*
     Approach 1: Top-Down Dynamic Programming - Optimized
Complexity Analysis
Let R be the length of ring and K be the length of key.
•	Time Complexity: O(K⋅R^2).
When every character in ring is unique, K recursive calls are made, one for each letter in the keyword.
At worst, when every character of ring is the same, we initially call trylock R times. For each of these R recursive calls, tryLock is called for each occurrence of the character in ring for each character in the keyword. This means the trylock function is called a total of R⋅K⋅R times.
Therefore, the overall time complexity is O(K⋅R^2).
•	Space Complexity: O(K⋅R)
O(K⋅R) space is used for the map. The call stack can grow as deep as K since a recursive call is made for each character in key. This makes the overall space complexity O(K⋅R).

     */
            public int TopDownDPOptimal(string ring, string key)
            {
                Dictionary<(int, int), int> bestSteps = new Dictionary<(int, int), int>();
                return TryLock(0, 0, ring, key, MAX, bestSteps);
            }
            public int TryLock(int ringIndex, int keyIndex, string ring, string key, int minSteps,
                Dictionary<(int, int), int> bestSteps)
            {
                // If we have already calculated this sub-problem, return the result
                if (bestSteps.ContainsKey((ringIndex, keyIndex)))
                {
                    return bestSteps[(ringIndex, keyIndex)];
                }
                // If we reach the end of the key, it has been spelled
                if (keyIndex == key.Length)
                {
                    return 0;
                }
                // For each occurrence of the character at keyIndex of key in ring
                // Calculate and save the minimum steps to that character from the ringIndex of ring
                for (int charIndex = 0; charIndex < ring.Length; charIndex++)
                {
                    if (ring[charIndex] == key[keyIndex])
                    {
                        int totalSteps = CountSteps(ringIndex, charIndex, ring.Length) + 1
                                                + TryLock(charIndex, keyIndex + 1, ring, key, MAX, bestSteps);
                        minSteps = Math.Min(minSteps, totalSteps);
                        bestSteps[(ringIndex, keyIndex)] = minSteps;
                    }
                }
                return minSteps;
            }

            /*
            Approach 2: Bottom-Up Dynamic Programming
            Complexity Analysis
Let R be the length of ring and K be the length of key.
•	Time Complexity: O(K⋅R^2)
We use nested loops iterating K times through key and R times through ring for all R characters in ring. This gives an overall time complexity of O(K⋅R^2).
•	Space Complexity: O(KR)
We use a 2D array with the dimensions K+1 and R.

            */
            public int BottomUpDP(String ring, String key)
            {
                int ringLen = ring.Length;
                int keyLen = key.Length;
                int[][] bestSteps = new int[ringLen][];
                // Initialize values of best_steps to largest integer
                foreach (int[] row in bestSteps)
                {
                    Array.Fill(row, int.MaxValue);
                }
                // Initialize last column to zero to represent the word has been spelled 
                for (int i = 0; i < ring.Length; i++)
                {
                    bestSteps[i][keyLen] = 0;
                }
                // For each occurrence of the character at keyIndex of key in ring
                // Stores minimum steps to the character from ringIndex of ring
                for (int keyIndex = keyLen - 1; keyIndex >= 0; keyIndex--)
                {
                    for (int ringIndex = 0; ringIndex < ringLen; ringIndex++)
                    {
                        for (int charIndex = 0; charIndex < ringLen; charIndex++)
                        {
                            if (ring[charIndex] == key[keyIndex])
                            {
                                bestSteps[ringIndex][keyIndex] = Math.Min(bestSteps[ringIndex][keyIndex],
                                        1 + CountSteps(ringIndex, charIndex, ringLen)
                                        + bestSteps[charIndex][keyIndex + 1]);
                            }
                        }
                    }
                }
                return bestSteps[0][0];
            }
            /*
            Approach 3: Space-Optimized Bottom-Up Dynamic Programming
            Complexity Analysis
Let R be the length of ring and K be the length of key.
•	Time Complexity: O(K⋅R^2)
We use nested loops iterating K times through key and R times through ring for all R characters in ring. This gives an overall time complexity of O(K⋅R^2).
•	Space Complexity: O(R).
We used two arrays of length R to store the minimum steps between the characters. This gives an overall space complexity of O(R).

            */
            public int BottomUpDPSpaceOptimal(String ring, String key)
            {
                int ringLen = ring.Length;
                int keyLen = key.Length;
                int[] curr = new int[ringLen];
                int[] prev = new int[ringLen];
                Array.Fill(prev, 0);
                // For each occurrence of the character at key_index of key in ring
                // Stores minimum steps to the character from ringIndex of ring
                for (int keyIndex = keyLen - 1; keyIndex >= 0; keyIndex--)
                {
                    Array.Fill(curr, int.MaxValue);
                    for (int ringIndex = 0; ringIndex < ringLen; ringIndex++)
                    {
                        for (int charIndex = 0; charIndex < ringLen; charIndex++)
                        {
                            if (ring[charIndex] == key[keyIndex])
                            {
                                curr[ringIndex] = Math.Min(curr[ringIndex],
                                        1 + CountSteps(ringIndex, charIndex, ringLen) + prev[charIndex]);
                            }
                        }
                    }
                    prev = (int[])curr.Clone();
                }
                return prev[0];
            }
            /*
            Approach 4: Shortest Path
            Complexity Analysis
Let R be the length of ring and K be the length of key.
•	Time complexity: O(RK⋅log(RK))
Building the characterIndices hashmap takes O(R) time as we add an entry for each character in ring.
The main loop will run once for each pair that we visit. We use the seen set, so we never visit the same (keyIndex, ringIndex) pair more than once. The maximum number of pairs we visit is the number of unique possible pairs, which is R⋅K.
Looking up a pair in seen takes O(1) time in the average case.
It takes the priority queue O(RK⋅log(RK)) time to push or pop R⋅K elements from the queue.
Therefore, the overall time complexity is O(RK⋅log(RK)).
•	Space complexity: O(R⋅K)
The characterIndices hashmap is size R because it stores a total of R (character, index) mappings.
The main space used is by the priority queue, which can store up to R⋅K pairs.
We also use the seen hash set, which can grow up to size R⋅K.	

            */
            public int UsingShortestPath(string ring, string key)
            {
                int ringLength = ring.Length;
                int keyLength = key.Length;

                // Dictionary to store the indices of occurrences of each character in the ring
                Dictionary<char, List<int>> characterIndices = new Dictionary<char, List<int>>();
                for (int i = 0; i < ring.Length; i++)
                {
                    char character = ring[i];
                    if (!characterIndices.ContainsKey(character))
                    {
                        characterIndices[character] = new List<int>();
                    }
                    characterIndices[character].Add(i);
                }

                // Initialize the min heap (priority queue) with the starting point
                // Each element of the heap is an array of integers representing:
                // totalSteps, ringIndex, keyIndex
                PriorityQueue<int[], int[]> heap = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[0].CompareTo(b[0])));

                heap.Enqueue(new int[] { 0, 0, 0 }, new int[] { 0, 0, 0 });

                // HashSet to keep track of visited states
                HashSet<(int, int)> seen = new HashSet<(int, int)>();

                // Spell the keyword using the metal dial
                int totalSteps = 0;
                while (heap.Count > 0)
                {
                    // Pop the element with the smallest total steps from the heap
                    int[] state = heap.Dequeue();
                    totalSteps = state[0];
                    int ringIndex = state[1];
                    int keyIndex = state[2];

                    // We have spelled the keyword
                    if (keyIndex == keyLength)
                    {
                        break;
                    }

                    // Continue if we have visited this character from this position in ring before
                    var currentState = (ringIndex, keyIndex);
                    if (seen.Contains(currentState))
                    {
                        continue;
                    }

                    // Otherwise, add this pair to the visited list
                    seen.Add(currentState);

                    // Add the rest of the occurrences of this character in ring to the heap
                    foreach (int nextIndex in characterIndices[key[keyIndex]])
                    {
                        var newVal = new int[] {
                totalSteps + CountSteps(ringIndex, nextIndex, ringLength),
                nextIndex,
                keyIndex + 1
            };
                        heap.Enqueue(newVal, newVal);
                    }
                }

                // Return the total steps and add keyLength to account for 
                // pressing the center button for each character in the keyword
                return totalSteps + keyLength;
            }

        }


        /* 329. Longest Increasing Path in a Matrix
        https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/
         */
        public class LongestIncreasingPathSol
        {
            private static readonly int[,] directions = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
            private static readonly int[][] Directions = new int[][] { new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { -1, 0 } };

            private int rowCount, columnCount;

            /* Approach #1 (Naive DFS) [Time Limit Exceeded]
            Complexity Analysis
            •	Time complexity : O(2^(m+n)). The search is repeated for each valid increasing path. In the worst case we can have O(2^(m+n)) calls.
            •	Space complexity : O(mn). For each DFS we need O(h) space used by the system stack, where h is the maximum depth of the recursion. In the worst case, O(h)=O(mn).

             */
            public int DFSNaive(int[][] matrix)
            {
                if (matrix.Length == 0) return 0;
                rowCount = matrix.Length;
                columnCount = matrix[0].Length;
                int result = 0;
                for (int row = 0; row < rowCount; ++row)
                    for (int col = 0; col < columnCount; ++col)
                        result = Math.Max(result, DepthFirstSearch(matrix, row, col));
                return result;
            }

            private int DepthFirstSearch(int[][] matrix, int row, int col)
            {
                int result = 0;
                for (int d = 0; d < directions.GetLength(0); d++)
                {
                    int x = row + directions[d, 0], y = col + directions[d, 1];
                    if (0 <= x && x < rowCount && 0 <= y && y < columnCount && matrix[x][y] > matrix[row][col])
                        result = Math.Max(result, DepthFirstSearch(matrix, x, y));
                }
                return ++result;
            }
            /*             Approach #2 (DFS + Memoization) [Accepted]
Complexity Analysis
•	Time complexity : O(mn). Each vertex/cell will be calculated once and only once, and each edge will be visited once and only once. The total time complexity is then O(V+E). V is the total number of vertices and E is the total number of edges. In our problem, O(V)=O(mn), O(E)=O(4V)=O(mn).
•	Space complexity : O(mn). The cache dominates the space complexity.

             */
            public int DFSWithMemo(int[][] matrix)
            {
                if (matrix.Length == 0) return 0;
                rowCount = matrix.Length; columnCount = matrix[0].Length;
                int[][] cache = new int[rowCount][];
                for (int i = 0; i < rowCount; i++)
                {
                    cache[i] = new int[columnCount];
                }
                int longestPath = 0;
                for (int i = 0; i < rowCount; ++i)
                    for (int j = 0; j < columnCount; ++j)
                        longestPath = Math.Max(longestPath, DepthFirstSearch(matrix, i, j, cache));
                return longestPath;
            }

            private int DepthFirstSearch(int[][] matrix, int i, int j, int[][] cache)
            {
                if (cache[i][j] != 0) return cache[i][j];
                foreach (int[] direction in Directions)
                {
                    int newRow = i + direction[0], newColumn = j + direction[1];
                    if (0 <= newRow && newRow < rowCount && 0 <= newColumn && newColumn < columnCount && matrix[newRow][newColumn] > matrix[i][j])
                        cache[i][j] = Math.Max(cache[i][j], DepthFirstSearch(matrix, newRow, newColumn, cache));
                }
                return ++cache[i][j];
            }
            /* Approach #3 (Peeling Onion) 
            Complexity Analysis
            •	Time complexity : O(mn). The the topological sort is O(V+E)=O(mn).
            Here, V is the total number of vertices and E is the total number of edges. In our problem, O(V)=O(mn), O(E)=O(4V)=O(mn).
            •	Space complexity : O(mn). We need to store the out degrees and each level of leaves.

             */
            public int PeelingOnion(int[,] grid)
            {
                int rows = grid.GetLength(0);
                if (rows == 0) return 0;
                int columns = grid.GetLength(1);
                // padding the matrix with zero as boundaries
                // assuming all positive integers, otherwise use int.MinValue as boundaries
                int[,] matrix = new int[rows + 2, columns + 2];
                for (int i = 0; i < rows; ++i)
                    Array.Copy(grid, i * columns, matrix, (i + 1) * (columns + 2) + 1, columns);

                // calculate outdegrees
                int[,] outdegree = new int[rows + 2, columns + 2];
                for (int i = 1; i <= rows; ++i)
                    for (int j = 1; j <= columns; ++j)
                        for (int d = 0; d < directions.GetLength(0); d++)
                            if (matrix[i, j] < matrix[i + directions[d, 0], j + directions[d, 1]])
                                outdegree[i, j]++;

                // find leaves who have zero out degree as the initial level
                columns += 2;
                rows += 2;
                List<int[]> leaves = new List<int[]>();
                for (int i = 1; i < rows - 1; ++i)
                    for (int j = 1; j < columns - 1; ++j)
                        if (outdegree[i, j] == 0) leaves.Add(new int[] { i, j });

                // remove leaves level by level in topological order
                int height = 0;
                while (leaves.Count > 0)
                {
                    height++;
                    List<int[]> newLeaves = new List<int[]>();
                    foreach (int[] node in leaves)
                    {
                        for (int d = 0; d < directions.GetLength(0); d++)
                        {
                            int x = node[0] + directions[d, 0], y = node[1] + directions[d, 1];
                            if (matrix[node[0], node[1]] > matrix[x, y])
                                if (--outdegree[x, y] == 0)
                                    newLeaves.Add(new int[] { x, y });
                        }
                    }
                    leaves = newLeaves;
                }
                return height;
            }

        }

        /* 2814. Minimum Time Takes to Reach Destination Without Drowning
        https://leetcode.com/problems/minimum-time-takes-to-reach-destination-without-drowning/description/
         */
        public class MinTimeTakesToReachDestinationWithoutDrowningSol
        {
            /* Simple BFS - 2 Queues 
Complexity
Time complexity: O(M∗N)
Space complexity: O(M∗N)

             */
            public int BFSAnd2Queues(IList<IList<string>> land)
            {
                int m = land.Count, n = land[0].Count;
                var flood = new Queue<(int, int)>();
                var move = new Queue<(int, int)>();
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (land[i][j] == "S") move.Enqueue((i, j));
                        if (land[i][j] == "*") flood.Enqueue((i, j));
                    }
                }
                var dirs = new[] { (1, 0), (-1, 0), (0, 1), (0, -1) };
                int secs = 0;
                while (move.Count > 0)
                {
                    int l1 = flood.Count, l2 = move.Count;
                    for (int _ = 0; _ < l1; _++)
                    {
                        var (x, y) = flood.Dequeue();
                        foreach (var (dx, dy) in dirs)
                        {
                            int cx = x + dx, cy = y + dy;
                            if (0 <= cx && cx < m && 0 <= cy && cy < n && land[cx][cy] == ".")
                            {
                                land[cx][cy] = "*";
                                flood.Enqueue((cx, cy));
                            }
                        }
                    }
                    for (int _ = 0; _ < l2; _++)
                    {
                        var (x, y) = move.Dequeue();
                        if (land[x][y] == "D") return secs;
                        foreach (var (dx, dy) in dirs)
                        {
                            int cx = x + dx, cy = y + dy;
                            if (0 <= cx && cx < m && 0 <= cy && cy < n && (land[cx][cy] == "." || land[cx][cy] == "D"))
                            {
                                if (land[cx][cy] != "D") land[cx][cy] = "*";
                                move.Enqueue((cx, cy));
                            }
                        }
                    }
                    secs++;
                }
                return -1;
            }
        }


        /* 302. Smallest Rectangle Enclosing Black Pixels
        https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/description/
         */
        public class SmallestRectangleEnclosingBlackPixelsSol
        {
            /* 
            Approach 1: Naive Linear Search
Complexity Analysis
•	Time complexity : O(mn). m and n are the height and width of the image.
•	Space complexity : O(1). All we need to store are the four boundaries.

             */
            public int Naive(char[][] image, int x, int y)
            {
                int top = x, bottom = x;
                int left = y, right = y;
                for (x = 0; x < image.Length; ++x)
                {
                    for (y = 0; y < image[0].Length; ++y)
                    {
                        if (image[x][y] == '1')
                        {
                            top = Math.Min(top, x);
                            bottom = Math.Max(bottom, x + 1);
                            left = Math.Min(left, y);
                            right = Math.Max(right, y + 1);
                        }
                    }
                }
                return (right - left) * (bottom - top);
            }


            /*
             Approach 2: DFS or BFS
             Complexity Analysis
            •	Time complexity : O(E)=O(B)=O(mn).
            Here E is the number of edges in the traversed graph. B is the total number of black pixels. Since each pixel have four edges at most, O(E)=O(B). In the worst case, O(B)=O(mn).
            •	Space complexity : O(V)=O(B)=O(mn).
            The space complexity is O(V) where V is the number of vertices in the traversed graph. In this problem O(V)=O(B). Again, in the worst case, O(B)=O(mn).
            Comment
            Although this approach is better than naive approach when B is much smaller than mn, it is asymptotically the same as approach #1 when B is comparable to mn. And it costs a lot more auxiliary space.

             */
            private int top, bottom, left, right;
            public int DFS(char[][] image, int x, int y)
            {
                if (image.Length == 0 || image[0].Length == 0) return 0;
                top = bottom = x;
                left = right = y;
                Dfs(image, x, y);
                return (right - left) * (bottom - top);
            }
            private void Dfs(char[][] image, int x, int y)
            {
                if (x < 0 || y < 0 || x >= image.Length || y >= image[0].Length ||
                  image[x][y] == '0')
                    return;
                image[x][y] = '0'; // mark visited black pixel as white
                top = Math.Min(top, x);
                bottom = Math.Max(bottom, x + 1);
                left = Math.Min(left, y);
                right = Math.Max(right, y + 1);
                Dfs(image, x + 1, y);
                Dfs(image, x - 1, y);
                Dfs(image, x, y - 1);
                Dfs(image, x, y + 1);
            }

            /*  
                       Approach 3: Binary Search 
                       Complexity Analysis
            •	Time complexity : O(mlogn+nlogm).
            Here, m and n are the height and width of the image. We embedded a linear search for every iteration of binary search. See previous sections for details.
            •	Space complexity : O(1).
            Both binary search and linear search used only constant extra space.

                       */
            public int UsingBinarySearch(char[][] image, int x, int y)
            {
                int m = image.Length, n = image[0].Length;
                int left = SearchColumns(image, 0, y, 0, m, true);
                int right = SearchColumns(image, y + 1, n, 0, m, false);
                int top = SearchRows(image, 0, x, left, right, true);
                int bottom = SearchRows(image, x + 1, m, left, right, false);
                return (right - left) * (bottom - top);
            }
            private int SearchColumns(char[][] image, int i, int j, int top, int bottom, bool whiteToBlack)
            {
                while (i != j)
                {
                    int k = top, mid = (i + j) / 2;
                    while (k < bottom && image[k][mid] == '0') ++k;
                    if (k < bottom == whiteToBlack) // k < bottom means the column mid has black pixel
                        j = mid; //search the boundary in the smaller half
                    else
                        i = mid + 1; //search the boundary in the greater half
                }
                return i;
            }
            private int SearchRows(char[][] image, int i, int j, int left, int right, bool whiteToBlack)
            {
                while (i != j)
                {
                    int k = left, mid = (i + j) / 2;
                    while (k < right && image[mid][k] == '0') ++k;
                    if (k < right == whiteToBlack) // k < right means the row mid has black pixel
                        j = mid;
                    else
                        i = mid + 1;
                }
                return i;
            }
        }


        /* 2699. Modify Graph Edge Weights
        https://leetcode.com/problems/modify-graph-edge-weights/description/
         */
        public class ModifiedGraphEdgeWeightsSol
        {
            private const int INF = (int)2e9;

            /* 
            Approach 1: Traditional Dijkstra's algorithm
            Complexity Analysis
            Let V be the number of nodes and E be the number of edges.
            •	Time complexity: O(E×V^2)
            Dijkstra's algorithm runs in O(V^2) time, due to the adjacency matrix representation.
            The overall complexity is O(E×V^2) because we potentially run Dijkstra's algorithm for each modifiable edge.
            •	Space complexity: O(V^2)
            The space complexity is O(V^2) due to the adjacency matrix, with additional space for the distance and visited arrays.

             */

            public int[][] UsingDijkstraAlgo(int n, int[][] edges, int source, int destination, int target)
            {
                // Step 1: Compute the initial shortest distance from source to destination
                long currentShortestDistance = RunDijkstra(edges, n, source, destination);

                // If the current shortest distance is less than the target, return an empty result
                if (currentShortestDistance < target) return new int[0][];

                bool matchesTarget = (currentShortestDistance == target);

                // Step 2: Iterate through each edge to adjust its weight if necessary
                foreach (int[] edge in edges)
                {
                    // Skip edges that already have a positive weight
                    if (edge[2] > 0) continue;

                    // Set edge weight to a large value if current distance matches target, else set to 1
                    edge[2] = matchesTarget ? INF : 1;

                    // Step 3: If current shortest distance does not match target
                    if (!matchesTarget)
                    {
                        // Compute the new shortest distance with the updated edge weight
                        long newDistance = RunDijkstra(edges, n, source, destination);
                        // If the new distance is within the target range, update edge weight to match target
                        if (newDistance <= target)
                        {
                            matchesTarget = true;
                            edge[2] += (int)(target - newDistance);
                        }
                    }
                }

                // Return modified edges if the target distance is achieved, otherwise return an empty result
                return matchesTarget ? edges : new int[0][];
            }

            // Dijkstra's algorithm to find the shortest path distance
            private long RunDijkstra(int[][] edges, int n, int source, int destination)
            {
                // Step 1: Initialize adjacency matrix and distance arrays
                long[][] adjMatrix = new long[n][];
                for (int i = 0; i < n; i++)
                    adjMatrix[i] = new long[n];

                long[] minDistance = new long[n];
                bool[] visited = new bool[n];

                Array.Fill(minDistance, INF);
                for (int i = 0; i < n; i++)
                    Array.Fill(adjMatrix[i], INF);

                // Set the distance to the source node as 0
                minDistance[source] = 0;

                // Step 2: Fill the adjacency matrix with edge weights
                foreach (int[] edge in edges)
                {
                    if (edge[2] != -1)
                    {
                        adjMatrix[edge[0]][edge[1]] = edge[2];
                        adjMatrix[edge[1]][edge[0]] = edge[2];
                    }
                }

                // Step 3: Perform Dijkstra's algorithm
                for (int i = 0; i < n; ++i)
                {
                    // Find the nearest unvisited node
                    int nearestUnvisitedNode = -1;
                    for (int j = 0; j < n; ++j)
                    {
                        if (!visited[j] && (nearestUnvisitedNode == -1 || minDistance[j] < minDistance[nearestUnvisitedNode]))
                        {
                            nearestUnvisitedNode = j;
                        }
                    }
                    // Mark the nearest node as visited
                    visited[nearestUnvisitedNode] = true;

                    // Update the minimum distance for each adjacent node
                    for (int v = 0; v < n; ++v)
                    {
                        minDistance[v] = Math.Min(minDistance[v], minDistance[nearestUnvisitedNode] + adjMatrix[nearestUnvisitedNode][v]);
                    }
                }

                // Return the shortest distance to the destination node
                return minDistance[destination];
            }
            /* 
            Approach 2: Dijkstra's Algorithm with Min-Heap 
            Complexity Analysis
            Let V be the number of nodes and E be the number of edges.
            •	Time complexity: O(E×(V+E)logV)
            Dijkstra's algorithm operates with a time complexity of O((V+E)logV) when using a priority queue (min-heap). This is because each vertex and edge is processed at most once, and each priority queue operation (insertion and extraction) takes O(logV) time.
            Dijkstra's algorithm once executes the shortest path from the source to the destination with the current weights. Then, for each edge that weights -1, Dijkstra's algorithm is rerun after modifying the edge weight. In the worst-case scenario, where all edges weigh -1, this results in running Dijkstra's up to E times.
            Thus, the overall time complexity for handling all possible edge modifications is O(E×(V+E)logV).
            •	Space complexity: O(V+E)
            The adjacency list representation of the graph requires O(V+E) space. Each vertex has a list of its adjacent vertices and their corresponding edge weights.
            Dijkstra’s algorithm uses an array to store the shortest distance from the source to each vertex, which requires O(V) space.
            The priority queue used during Dijkstra's algorithm can hold up to V elements, which also requires O(V) space.
            Summing up these components, the total space complexity is O(V+E).

            */
            List<int[]>[] graph;


            public int[][] DijkstraAlgoWithMinHeap(int n, int[][] edges, int source, int destination, int target)
            {
                // Step 1: Build the graph, excluding edges with -1 weights
                graph = new List<int[]>[n];
                for (int i = 0; i < n; i++)
                {
                    graph[i] = new List<int[]>();
                }

                foreach (int[] edge in edges)
                {
                    if (edge[2] != -1)
                    {
                        graph[edge[0]].Add(new int[] { edge[1], edge[2] });
                        graph[edge[1]].Add(new int[] { edge[0], edge[2] });
                    }
                }

                // Step 2: Compute the initial shortest distance from source to destination
                int currentShortestDistance = RunDijkstra(n, source, destination);
                if (currentShortestDistance < target)
                {
                    return new int[0][];
                }

                bool matchesTarget = (currentShortestDistance == target);

                // Step 3: Iterate through each edge to adjust its weight if necessary
                foreach (int[] edge in edges)
                {
                    if (edge[2] != -1) continue; // Skip edges with already known weights

                    // Set edge weight to a large value if current distance matches target, else set
                    // to 1
                    edge[2] = matchesTarget ? INF : 1;
                    graph[edge[0]].Add(new int[] { edge[1], edge[2] });
                    graph[edge[1]].Add(new int[] { edge[0], edge[2] });

                    // Step 4: If current shortest distance does not match target
                    if (!matchesTarget)
                    {
                        // Compute the new shortest distance with the updated edge weight
                        int newDistance = RunDijkstra(n, source, destination);
                        // If the new distance is within the target range, update edge weight to match
                        // target
                        if (newDistance <= target)
                        {
                            matchesTarget = true;
                            edge[2] += target - newDistance;
                        }
                    }
                }

                // Return modified edges if the target distance is achieved, otherwise return an
                // empty result
                return matchesTarget ? edges : new int[0][];
            }

            // Dijkstra's algorithm to find the shortest path distance
            private int RunDijkstra(int n, int source, int destination)
            {
                int[] minDistance = new int[n];
                PriorityQueue<int[], int[]> minHeap = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[1] - b[1]));

                Array.Fill(minDistance, INF);
                minDistance[source] = 0;
                minHeap.Enqueue(new int[] { source, 0 }, new int[] { source, 0 });

                while (minHeap.Count > 0)
                {
                    int[] curr = minHeap.Dequeue();
                    int u = curr[0];
                    int d = curr[1];

                    if (d > minDistance[u]) continue;

                    foreach (int[] neighbor in graph[u])
                    {
                        int v = neighbor[0];
                        int weight = neighbor[1];

                        if (d + weight < minDistance[v])
                        {
                            minDistance[v] = d + weight;
                            minHeap.Enqueue(new int[] { v, minDistance[v] }, new int[] { v, minDistance[v] });
                        }
                    }
                }

                return minDistance[destination];
            }

        }

        /* 363. Max Sum of Rectangle No Larger Than K
        https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/description/
         */
        class MaxSumSubmatrixSol
        {
            int maximumResult = int.MinValue;

            /* 
            Approach 1: Prefix Sum on 1D Array using Sorted Container
            Complexity Analysis
            Let m be the number of rows and n be the number of columns.
            •	Time complexity: O((m^2)nlogn). We iterate over each i and j where 0≤i≤j<m, within this we iterate over each i where 0≤i<n and perform a binary search on the same number of elements.
            •	Space complexity: O(n). We create a separate array of size n representing the 2D matrix and also store prefix sums for all indices.

             */
            public int PrefixSum1DArrayUsingSortedSet(int[][] matrix, int target)
            {
                // Stores the 1D representation of the matrix.
                int[] oneDimensionalRowSum = new int[matrix[0].Length];
                for (int rowIndex = 0; rowIndex < matrix.Length; rowIndex++)
                {
                    // Initialize the 1D representation with 0s.
                    Array.Fill(oneDimensionalRowSum, 0);
                    // We convert the matrix between rows rowIndex..rowInclusive to 1D array
                    for (int currentRow = rowIndex; currentRow < matrix.Length; currentRow++)
                    {
                        // Add the current row to the previous row.
                        // This converts the matrix between rowIndex..currentRow to 1D array
                        for (int columnIndex = 0; columnIndex < matrix[0].Length; columnIndex++)
                            oneDimensionalRowSum[columnIndex] += matrix[currentRow][columnIndex];

                        // Run the 1D algorithm for `oneDimensionalRowSum`
                        UpdateMaximumResult(oneDimensionalRowSum, target);

                        // If maximumResult is target, this is the best possible answer, so return.
                        if (maximumResult == target)
                            return maximumResult;
                    }
                }
                return maximumResult;
            }
            void UpdateMaximumResult(int[] numbers, int target)
            {
                int currentSum = 0;

                // Container to store sorted prefix sums.
                SortedSet<int> sortedPrefixSums = new SortedSet<int>();

                // Add 0 as the prefix sum for an empty sub-array.
                sortedPrefixSums.Add(0);
                foreach (int number in numbers)
                {
                    // Running Sum.
                    currentSum += number;

                    // Get X where Running sum - X <= target such that currentSum - X is closest to target.
                    int? x = null;
                    foreach (int prefixSum in sortedPrefixSums)
                    {
                        if (prefixSum >= currentSum - target)
                        {
                            x = prefixSum;
                            break;
                        }
                    }

                    // If such X is found in the prefix sums.
                    // Get the sum of that sub array and update the global maximum result.
                    if (x.HasValue)
                        maximumResult = Math.Max(maximumResult, currentSum - x.Value);

                    // Insert the current running sum to the prefix sums container.
                    sortedPrefixSums.Add(currentSum);
                }
            }
            /* 
                        Approach 2: Follow-up - Larger Number of Rows than Columns 
                     Complexity Analysis
Let m be the number of rows and n be the number of columns.
•	Time complexity: O(min((m,n)^2)max(m,n)log max(m,n)). Using the same thought process as approach 1.
•	Space complexity: O(max(m,n)). Using the same thought process as approach 1.


                        */
            int result = int.MinValue;

            public int WhenRowsLargerThanColumns(int[][] matrix, int k)
            {
                if (matrix[0].Length > matrix.Length)
                {
                    // Stores the 1D representation of the matrix.
                    int[] rowSum = new int[matrix[0].Length];
                    for (int i = 0; i < matrix.Length; i++)
                    {
                        // Initialize the 1D representation with 0s.
                        Array.Fill(rowSum, 0);
                        // We convert the matrix between rows i..row inclusive to 1D array
                        for (int row = i; row < matrix.Length; row++)
                        {
                            // Add the current row to the previous row.
                            // This converts the matrix between i..j to 1D array
                            for (int col = 0; col < matrix[0].Length; col++)
                                rowSum[col] += matrix[row][col];

                            // Run the 1D algorithm for `rowSum`
                            UpdateResult(rowSum, k);

                            // If result is k, this is the best possible answer, so return.
                            if (result == k)
                                return result;
                        }
                    }
                }
                else
                {
                    // Stores the 1D representation of the matrix column wise.
                    int[] colSum = new int[matrix.Length];
                    for (int i = 0; i < matrix[0].Length; i++)
                    {
                        // Initialize the 1D representation with 0s.
                        Array.Fill(colSum, 0);

                        // We convert the matrix between columns i..col inclusive to 1D array
                        for (int col = i; col < matrix[0].Length; col++)
                        {
                            // Add the current column to the previous column.
                            for (int row = 0; row < matrix.Length; row++)
                                colSum[row] += matrix[row][col];

                            // Run the 1D algorithm for `colSum`
                            UpdateResult(colSum, k);

                            // If result is k, this is the best possible answer, so return.
                            if (result == k)
                                return result;
                        }
                    }
                }
                return result;
            }
            private void UpdateResult(int[] nums, int k)
            {
                int sum = 0;

                // Container to store sorted prefix sums.
                SortedSet<int> sortedSum = new SortedSet<int>();

                // Add 0 as the prefix sum for an empty sub-array.
                sortedSum.Add(0);
                foreach (int num in nums)
                {
                    // Running Sum.
                    sum += num;

                    // Get X where Running sum - X <= k such that sum - X is closest to k.
                    int? x = null;
                    foreach (int prefixSum in sortedSum)
                    {
                        if (prefixSum >= sum - k)
                        {
                            x = prefixSum;
                            break;
                        }
                    }

                    // If such X is found in the prefix sums.
                    // Get the sum of that sub array and update the global maximum result.
                    if (x.HasValue)
                        result = Math.Max(result, sum - x.Value);

                    // Insert the current running sum to the prefix sums container.
                    sortedSum.Add(sum);
                }
            }

            /* 
            Approach 3: Combining it with Kadane's Algorithm 
            Complexity Analysis
Let m be the number of rows and n be the number of columns.
•	Time complexity: O((min(m,n)^2)max(m,n)log max(m,n)). Using the same thought process as approach 1 as in the worst case we end up running the algorithm from approach 1 for all 1D arrays.
•	Space complexity: O(max(m,n)). Using the same thought process as approach 1.

            */

            public int UsingKadanesAlgo(int[][] matrix, int k)
            {
                if (matrix[0].Length > matrix.Length)
                {
                    // Stores the 1D representation of the matrix.
                    int[] rowSum = new int[matrix[0].Length];
                    for (int i = 0; i < matrix.Length; i++)
                    {
                        // Initialize the 1D representation with 0s.
                        Array.Fill(rowSum, 0);
                        // We convert the matrix between rows i..row inclusive to 1D array
                        for (int row = i; row < matrix.Length; row++)
                        {
                            // Add the current row to the previous row.
                            // This converts the matrix between i..j to 1D array
                            for (int col = 0; col < matrix[0].Length; col++)
                                rowSum[col] += matrix[row][col];

                            // Run the 1D algorithm for `rowSum`
                            UpdateResultExt(rowSum, k);

                            // If result is k, this is the best possible answer, so return.
                            if (result == k)
                                return result;
                        }
                    }
                }
                else
                {
                    // Stores the 1D representation of the matrix column-wise.
                    int[] colSum = new int[matrix.Length];
                    for (int i = 0; i < matrix[0].Length; i++)
                    {
                        // Initialize the 1D representation with 0s.
                        Array.Fill(colSum, 0);

                        // We convert the matrix between columns i..col inclusive to 1D array
                        for (int col = i; col < matrix[0].Length; col++)
                        {
                            // Add the current column to the previous column.
                            for (int row = 0; row < matrix.Length; row++)
                                colSum[row] += matrix[row][col];

                            // Run the 1D algorithm for `colSum`
                            UpdateResultExt(colSum, k);

                            // If result is k, this is the best possible answer, so return.
                            if (result == k)
                                return result;
                        }
                    }
                }
                return result;
            }
            // Standard Kadane's algorithm.
            private int GetMaxKadane(int[] nums)
            {
                int maxKadane = int.MinValue, currentMaxSum = 0;
                foreach (int num in nums)
                {
                    currentMaxSum = Math.Max(currentMaxSum + num, num);
                    maxKadane = Math.Max(maxKadane, currentMaxSum);
                }
                return maxKadane;
            }

            private void UpdateResultExt(int[] nums, int k)
            {
                int kadaneSum = GetMaxKadane(nums);

                // If max possible sum of any subarray of nums is <= k
                // use that result to compare with global maximum result and return
                if (kadaneSum <= k)
                {
                    result = Math.Max(result, kadaneSum);
                    return;
                }
                int sum = 0;

                // Container to store sorted prefix sums.
                SortedSet<int> sortedSum = new SortedSet<int>();

                // Add 0 as the prefix sum for an empty sub-array.
                sortedSum.Add(0);
                foreach (int num in nums)
                {
                    // Running Sum.
                    sum += num;

                    // Get X where Running sum - X <= k such that sum - X is closest to k.
                    int x = sortedSum.FirstOrDefault(value => value >= sum - k);

                    // If such X is found in the prefix sums.
                    // Get the sum of that subarray and update the global maximum result.
                    if (x != 0)
                        result = Math.Max(result, sum - x);

                    // Insert the current running sum to the prefix sums container.
                    sortedSum.Add(sum);
                }
            }

        }


        /* 864. Shortest Path to Get All Keys
        https://leetcode.com/problems/shortest-path-to-get-all-keys/
         */
        class ShortestPathToGetAllKeysSol
        {
            int INF = int.MaxValue;
            string[] grid;
            int R, C;
            Dictionary<char, Point> location;
            int[] dr = new int[] { -1, 0, 1, 0 };
            int[] dc = new int[] { 0, -1, 0, 1 };

            /* 
            Approach 1: Brute Force + Permutations 
            Complexity Analysis
            •	Time Complexity: O(R∗C∗A∗A!), where R,C are the dimensions of the grid, and A is the maximum number of keys (A because it is the "size of the alphabet".) Each bfs is performed up to A∗A! times.
            •	Space Complexity: O(R∗C+A!), the space for the bfs and to store the candidate key permutations.

            */
            public int NaiveWithPermutations(string[] grid)
            {
                this.grid = grid;
                R = grid.Length;
                C = grid[0].Length;

                // location['a'] = the coordinates of 'a' on the grid, etc.
                location = new Dictionary<char, Point>();
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c)
                    {
                        char v = grid[r][c];
                        if (v != '.' && v != '#')
                            location[v] = new Point(r, c);
                    }

                int ans = INF;
                int numKeys = location.Count / 2;
                string[] alphabet = new string[numKeys];
                for (int i = 0; i < numKeys; ++i)
                    alphabet[i] = ((char)('a' + i)).ToString();
                // alphabet = ["a", "b", "c"], if there were 3 keys

                //TODO: Fix below commented code
                /* search: foreach (string candidate in Permutations(alphabet, 0, numKeys))
                {
                    // bns : the built candidate answer, consisting of the sum
                    // of distances of the segments from '@' to candidate[0] to candidate[1] etc.
                    int bns = 0;
                    for (int i = 0; i < numKeys; ++i)
                    {
                        char source = i > 0 ? candidate[i - 1] : '@';
                        char target = candidate[i];

                        // keymask : an integer with the 0-th bit set if we picked up
                        // key 'a', the 1-th bit set if we picked up key 'b', etc.
                        int keymask = 0;
                        for (int j = 0; j < i; ++j)
                            keymask |= 1 << (candidate[j] - 'a');
                        int distance = Bfs(source, target, keymask);
                        if (distance == INF) continue search;
                        bns += distance;
                        if (bns >= ans) continue search;
                    }
                    ans = bns;
                } */

                return ans < INF ? ans : -1;
            }

            private int Bfs(char source, char target, int keymask)
            {
                int sr = location[source].X;
                int sc = location[source].Y;
                int tr = location[target].X;
                int tc = location[target].Y;
                bool[,] seen = new bool[R, C];
                seen[sr, sc] = true;
                int curDepth = 0;
                Queue<Point> queue = new Queue<Point>();
                queue.Enqueue(new Point(sr, sc));
                queue.Enqueue(new Point(-1, -1)); // Use a sentinel value to indicate depth increment

                while (queue.Count > 0)
                {
                    Point p = queue.Dequeue();
                    if (p.X == -1 && p.Y == -1)
                    {
                        curDepth++;
                        if (queue.Count > 0)
                            queue.Enqueue(new Point(-1, -1));
                        continue;
                    }
                    int r = p.X, c = p.Y;
                    if (r == tr && c == tc) return curDepth;
                    for (int i = 0; i < 4; ++i)
                    {
                        int cr = r + dr[i];
                        int cc = c + dc[i];
                        if (0 <= cr && cr < R && 0 <= cc && cc < C && !seen[cr, cc])
                        {
                            char cur = grid[cr][cc];
                            if (cur != '#')
                            {
                                if (char.IsUpper(cur) && (((1 << (cur - 'A')) & keymask) <= 0))
                                    continue; // at lock and don't have key

                                queue.Enqueue(new Point(cr, cc));
                                seen[cr, cc] = true;
                            }
                        }
                    }
                }

                return INF;
            }

            private List<string> Permutations(string[] alphabet, int used, int size)
            {
                List<string> ans = new List<string>();
                if (size == 0)
                {
                    ans.Add("");
                    return ans;
                }

                for (int b = 0; b < alphabet.Length; ++b)
                    if (((used >> b) & 1) == 0)
                        foreach (string rest in Permutations(alphabet, used | (1 << b), size - 1))
                            ans.Add(alphabet[b] + rest);
                return ans;
            }
            internal class Point
            {
                public int X;
                public int Y;

                public Point(int sr, int sc)
                {
                    this.X = sr;
                    this.Y = sc;
                }
            }

            /* 
            Approach 2: Points of Interest + Dijkstra 
            Complexity Analysis
•	Time Complexity: O(RC(2A+1)+ElogN), where R,C are the dimensions of the grid, and A is the maximum number of keys, N=(2A+1)∗2A is the number of nodes when we perform Dijkstra's, and E=N∗(2A+1) is the maximum number of edges.
•	Space Complexity: O(N).

            */
            public int UsingDijkstraAlgo(string[] grid)
            {
                this.grid = grid;
                R = grid.Length;
                C = grid[0].Length;

                //location : the points of interest
                location = new Dictionary<char, Point>();
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c < C; ++c)
                    {
                        char v = grid[r][c];
                        if (v != '.' && v != '#')
                            location[v] = new Point(r, c);
                    }

                int targetState = (1 << (location.Count / 2)) - 1;
                Dictionary<char, Dictionary<char, int>> dists = new Dictionary<char, Dictionary<char, int>>();
                foreach (char place in location.Keys)
                    dists[place] = BfsFrom(place);

                //Dijkstra
                PriorityQueue<ANode, ANode> pq = new PriorityQueue<ANode, ANode>(Comparer<ANode>.Create((a, b) =>
                        a.dist.CompareTo(b.dist)));
                pq.Enqueue(new ANode(new Node('@', 0), 0), new ANode(new Node('@', 0), 0));
                Dictionary<Node, int> finalDist = new Dictionary<Node, int>();
                finalDist[new Node('@', 0)] = 0;

                while (pq.Count > 0)
                {
                    ANode anode = pq.Dequeue();
                    Node node = anode.node;
                    int d = anode.dist;
                    if (finalDist.GetValueOrDefault(node, int.MaxValue) < d) continue;
                    if (node.state == targetState) return d;

                    foreach (char destination in dists[node.place].Keys)
                    {
                        int d2 = dists[node.place][destination];
                        int state2 = node.state;
                        if (char.IsLower(destination)) //key
                            state2 |= (1 << (destination - 'a'));
                        if (char.IsUpper(destination)) //lock
                            if ((node.state & (1 << (destination - 'A'))) == 0) // no key
                                continue;

                        if (d + d2 < finalDist.GetValueOrDefault(new Node(destination, state2), int.MaxValue))
                        {
                            finalDist[new Node(destination, state2)] = d + d2;
                            pq.Enqueue(new ANode(new Node(destination, state2), d + d2), new ANode(new Node(destination, state2), d + d2));
                        }
                    }
                }

                return -1;
            }

            private Dictionary<char, int> BfsFrom(char source)
            {
                int sr = location[source].X;
                int sc = location[source].Y;
                bool[,] seen = new bool[R, C];
                seen[sr, sc] = true;
                int curDepth = 0;
                Queue<Point?> queue = new Queue<Point?>();
                queue.Enqueue(new Point(sr, sc));
                queue.Enqueue(null);
                Dictionary<char, int> dist = new Dictionary<char, int>();

                while (queue.Count > 0)
                {
                    Point? p = queue.Dequeue();
                    if (p == null)
                    {
                        curDepth++;
                        if (queue.Count > 0)
                            queue.Enqueue(null);
                        continue;
                    }

                    int r = p.X, c = p.Y;
                    if (grid[r][c] != source && grid[r][c] != '.')
                    {
                        dist[grid[r][c]] = curDepth;
                        continue; // Stop walking from here if we reach a point of interest
                    }

                    for (int i = 0; i < 4; ++i)
                    {
                        int cr = r + dr[i];
                        int cc = c + dc[i];
                        if (0 <= cr && cr < R && 0 <= cc && cc < C && !seen[cr, cc])
                        {
                            if (grid[cr][cc] != '#')
                            {
                                queue.Enqueue(new Point(cr, cc));
                                seen[cr, cc] = true;
                            }
                        }
                    }
                }

                return dist;
            }
            // ANode: Annotated Node
            class ANode
            {
                public Node node;
                public int dist;

                public ANode(Node n, int d)
                {
                    node = n;
                    dist = d;
                }
            }

            class Node
            {
                public char place;
                public int state;

                public Node(char p, int s)
                {
                    place = p;
                    state = s;
                }

                public override bool Equals(object o)
                {
                    if (this == o) return true;
                    if (!(o is Node)) return false;
                    Node other = (Node)o;
                    return (place == other.place && state == other.state);
                }

                public override int GetHashCode()
                {
                    return 256 * state + place;
                }
            }

        }

        /* 1074. Number of Submatrices That Sum to Target
        https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description/
         */
        class NumSubmatrixSumToTargetSol
        {
            /*             Approach 1: Number of Subarrays that Sum to Target: Horizontal 1D Prefix Sum
            Complexity Analysis
            •	Time complexity: O(R^2C), where R is the number of rows and C is the number of columns.
            •	Space complexity: O(RC) to store 2D prefix sum.

             */
            public int Horizontal1DPrefixSum(int[][] matrix, int target)
            {
                int r = matrix.Length, c = matrix[0].Length;

                // compute 2D prefix sum
                int[][] ps = new int[r + 1][];
                for (int i = 1; i < r + 1; ++i)
                {
                    for (int j = 1; j < c + 1; ++j)
                    {
                        ps[i][j] = ps[i - 1][j] + ps[i][j - 1] - ps[i - 1][j - 1] + matrix[i - 1][j - 1];
                    }
                }

                int count = 0, currSum;
                Dictionary<int, int> h = new();
                // reduce 2D problem to 1D one
                // by fixing two rows r1 and r2 and 
                // computing 1D prefix sum for all matrices using [r1..r2] rows
                for (int r1 = 1; r1 < r + 1; ++r1)
                {
                    for (int r2 = r1; r2 < r + 1; ++r2)
                    {
                        h.Clear();
                        h.Add(0, 1);
                        for (int col = 1; col < c + 1; ++col)
                        {
                            // current 1D prefix sum
                            currSum = ps[r2][col] - ps[r1 - 1][col];

                            // add subarrays which sum up to (currSum - target)
                            count += h.GetValueOrDefault(currSum - target, 0);

                            // save current prefix sum
                            h[currSum] = h.GetValueOrDefault(currSum, 0) + 1;
                        }
                    }
                }

                return count;
            }
            /*
             Approach 2: Number of Subarrays that Sum to Target: Vertical 1D Prefix Sum 
             Complexity Analysis
•	Time complexity: O(RC^2), where R is the number of rows and C is the number of columns.
•	Space complexity: O(RC) to store 2D prefix sum.

            */
            public int Vertical1DPrefixSum(int[][] matrix, int target)
            {
                int rowCount = matrix.Length, columnCount = matrix[0].Length;

                // compute 2D prefix sum
                int[][] prefixSum = new int[rowCount + 1][];
                for (int i = 0; i < prefixSum.Length; i++)
                {
                    prefixSum[i] = new int[columnCount + 1];
                }

                for (int i = 1; i < rowCount + 1; ++i)
                {
                    for (int j = 1; j < columnCount + 1; ++j)
                    {
                        prefixSum[i][j] = prefixSum[i - 1][j] + prefixSum[i][j - 1] - prefixSum[i - 1][j - 1] + matrix[i - 1][j - 1];
                    }
                }

                int count = 0, currentSum;
                Dictionary<int, int> hashMap = new Dictionary<int, int>();
                // reduce 2D problem to 1D one
                // by fixing two columns c1 and c2 and 
                // computing 1D prefix sum for all matrices using [c1..c2] columns
                for (int columnStart = 1; columnStart < columnCount + 1; ++columnStart)
                {
                    for (int columnEnd = columnStart; columnEnd < columnCount + 1; ++columnEnd)
                    {
                        hashMap.Clear();
                        hashMap[0] = 1;
                        for (int row = 1; row < rowCount + 1; ++row)
                        {
                            // current 1D prefix sum 
                            currentSum = prefixSum[row][columnEnd] - prefixSum[row][columnStart - 1];

                            // add subarrays which sum up to (currentSum - target)
                            count += hashMap.TryGetValue(currentSum - target, out int value) ? value : 0;

                            // save current prefix sum
                            if (hashMap.ContainsKey(currentSum))
                            {
                                hashMap[currentSum]++;
                            }
                            else
                            {
                                hashMap[currentSum] = 1;
                            }
                        }
                    }
                }

                return count;
            }
        }



        /* 2421. Number of Good Paths
        https://leetcode.com/problems/number-of-good-paths/description/
         */
        public class NumberOfGoodPathsSol
        {
            /*
                         Approach: Union-Find
                         Complexity Analysis
            Here, n is the number of nodes.
            •	Time complexity: O(n⋅log(n))
            o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1). We iterate over each edge once from the larger value node to the smaller one, or if the neighbors (nodes that share an edge) have equal value, we iterate that edge twice, which is also linear. To iterate over n−1 edges, we have to perform O(n) operations which needs O(n) time.
            o	We also need a map valuesToNodes having sorted keys. Each operation in such a data structure comes with a log factor. We push all the n nodes into the map and iterate over all of them, which further adds O(n⋅log(n))) time.
            o	The group map has unsorted keys, and each of its operation takes O(1) time on average. We need O(n) time to iterate through all of the nodes to find the set size using it.
            o	Additionally, we need O(n) time each to initialize the adj, parent and rank arrays.
            •	Space complexity: O(n)
            o	We require O(n) space each for the adj, parent and rank arrays.
            o	We also require O(n) space for the valuesToNodes and the group maps.

             */

            public int UsingUnionFind(int[] values, int[][] edges)
            {
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] edge in edges)
                {
                    int firstNode = edge[0];
                    int secondNode = edge[1];
                    if (!adjacencyList.ContainsKey(firstNode))
                    {
                        adjacencyList[firstNode] = new List<int>();
                    }
                    adjacencyList[firstNode].Add(secondNode);
                    if (!adjacencyList.ContainsKey(secondNode))
                    {
                        adjacencyList[secondNode] = new List<int>();
                    }
                    adjacencyList[secondNode].Add(firstNode);
                }

                int numberOfNodes = values.Length;
                // Mapping from value to all the nodes having the same value in sorted order of values.
                SortedDictionary<int, List<int>> valueToNodesMap = new SortedDictionary<int, List<int>>();
                for (int index = 0; index < numberOfNodes; index++)
                {
                    if (!valueToNodesMap.ContainsKey(values[index]))
                    {
                        valueToNodesMap[values[index]] = new List<int>();
                    }
                    valueToNodesMap[values[index]].Add(index);
                }

                UnionFind unionFind = new UnionFind(numberOfNodes);
                int totalGoodPaths = 0;

                // Iterate over all the nodes with the same value in sorted order, starting from the lowest value.
                foreach (int value in valueToNodesMap.Keys)
                {
                    // For every node in nodes, combine the sets of the node and its neighbors into one set.
                    foreach (int node in valueToNodesMap[value])
                    {
                        if (!adjacencyList.ContainsKey(node))
                        {
                            continue;
                        }
                        foreach (int neighbor in adjacencyList[node])
                        {
                            // Only choose neighbors with a smaller value, as there is no point in traversing to other neighbors.
                            if (values[node] >= values[neighbor])
                            {
                                unionFind.UnionSet(node, neighbor);
                            }
                        }
                    }
                    // Map to compute the number of nodes under observation (with the same values) in each of the sets.
                    Dictionary<int, int> groupCount = new Dictionary<int, int>();
                    // Iterate over all the nodes. Get the set of each node and increase the count of the set by 1.
                    foreach (int node in valueToNodesMap[value])
                    {
                        int root = unionFind.Find(node);
                        if (!groupCount.ContainsKey(root))
                        {
                            groupCount[root] = 0;
                        }
                        groupCount[root]++;
                    }
                    // For each set of "size", add size * (size + 1) / 2 to the number of goodPaths.
                    foreach (int key in groupCount.Keys)
                    {
                        int size = groupCount[key];
                        totalGoodPaths += size * (size + 1) / 2;
                    }
                }
                return totalGoodPaths;
            }
            public class UnionFind
            {
                private int[] parent;
                private int[] rank;

                public UnionFind(int size)
                {
                    parent = new int[size];
                    for (int index = 0; index < size; index++)
                    {
                        parent[index] = index;
                    }
                    rank = new int[size];
                }

                public int Find(int element)
                {
                    if (parent[element] != element)
                    {
                        parent[element] = Find(parent[element]);
                    }
                    return parent[element];
                }

                public void UnionSet(int firstElement, int secondElement)
                {
                    int firstSet = Find(firstElement);
                    int secondSet = Find(secondElement);

                    if (firstSet == secondSet)
                    {
                        return;
                    }
                    else if (rank[firstSet] < rank[secondSet])
                    {
                        parent[firstSet] = secondSet;
                    }
                    else if (rank[firstSet] > rank[secondSet])
                    {
                        parent[secondSet] = firstSet;
                    }
                    else
                    {
                        parent[secondSet] = firstSet;
                        rank[firstSet]++;
                    }
                }
            }
        }


        /* 847. Shortest Path Visiting All Nodes
        https://leetcode.com/problems/shortest-path-visiting-all-nodes/description/
         */
        class ShortestPathVisitAllNodesLengthSolution
        {
            private int[][] cache;
            private int endingMask;

            /* Approach 1: DFS + Memoization (Top-Down DP)
            Complexity Analysis
            Given N as the number of nodes in the graph:
            •	Time complexity: O((2^N)⋅(N^2))
            The total number of possible states is O((2^N)⋅N), because there are 2^N possibilities for mask, each of which can be paired with one of N nodes.
            At each state, we perform a for loop that loops through all the edges the given node has. In the worst case scenario, every node in the graph is connected to every other node, so this for loop will cost O(N).
            •	Space complexity: O((2^N)⋅N)
            Depending on the implementation, cache will either be the same size as the number of states when it is initialized or it will eventually grow to that size by the end of the algorithm in the worst-case scenario.

             */
            public int TopDownDPWithMemo(int[][] graph)
            {
                int n = graph.Length;
                endingMask = (1 << n) - 1;
                cache = new int[n + 1][];

                int best = int.MaxValue;
                for (int node = 0; node < n; node++)
                {
                    best = Math.Min(best, dp(node, endingMask, graph));
                }

                return best;
            }
            private int dp(int node, int mask, int[][] graph)
            {
                if (cache[node][mask] != 0)
                {
                    return cache[node][mask];
                }
                if ((mask & (mask - 1)) == 0)
                {
                    // Base case - mask only has a single "1", which means
                    // that only one node has been visited (the current node)
                    return 0;
                }

                cache[node][mask] = int.MaxValue - 1; // Avoid infinite loop in recursion
                foreach (int neighbor in graph[node])
                {
                    if ((mask & (1 << neighbor)) != 0)
                    {
                        int alreadyVisited = dp(neighbor, mask, graph);
                        int notVisited = dp(neighbor, mask ^ (1 << node), graph);
                        int betterOption = Math.Min(alreadyVisited, notVisited);
                        cache[node][mask] = Math.Min(cache[node][mask], 1 + betterOption);
                    }
                }

                return cache[node][mask];
            }
            /* Approach 2: Breadth-First Search (BFS)
            Complexity Analysis
            Given N as the number of nodes in the graph:
            •	Time complexity: O(2^N⋅N^2)
            The total number of possible states that can be put in our queue is O(2^N⋅N), because there are 2^N possibilities for mask, each of which can be paired with one of N nodes.
            At each state, we use a for loop to loop through all the edges the given node has. In the worst case, when the graph is fully connected, each node will have N−1 neighbors, so this step costs O(N) as the work done inside the for-loop is O(1).
            Despite having the same time complexity as the first approach, in most cases, this algorithm will outperform the first one for the reasons we talked about in the intuition section, particularly because this algorithm will exit early as soon as it finds a solution.
            •	Space complexity: O(2^N⋅N)
            By the end of the algorithm, most of our extra space will be occupied by seen. Same as in the previous approach, depending on the implementation, seen will either be the same size as the number of states when it is initialized or it will eventually grow to that size by the end of the algorithm in the worst-case scenario.

             */
            public int BFS(int[][] graph)
            {
                if (graph.Length == 1)
                {
                    return 0;
                }

                int nodeCount = graph.Length;
                int endingMask = (1 << nodeCount) - 1;
                bool[,] seen = new bool[nodeCount, endingMask];
                List<int[]> queue = new List<int[]>();

                for (int i = 0; i < nodeCount; i++)
                {
                    queue.Add(new int[] { i, 1 << i });
                    seen[i, 1 << i] = true;
                }

                int steps = 0;
                while (queue.Count > 0)
                {
                    List<int[]> nextQueue = new List<int[]>();
                    for (int i = 0; i < queue.Count; i++)
                    {
                        int[] currentPair = queue[i];
                        int currentNode = currentPair[0];
                        int currentMask = currentPair[1];
                        foreach (int neighbor in graph[currentNode])
                        {
                            int nextMask = currentMask | (1 << neighbor);
                            if (nextMask == endingMask)
                            {
                                return 1 + steps;
                            }

                            if (!seen[neighbor, nextMask])
                            {
                                seen[neighbor, nextMask] = true;
                                nextQueue.Add(new int[] { neighbor, nextMask });
                            }
                        }
                    }
                    steps++;
                    queue = nextQueue;
                }

                return -1;
            }

        }

        /* 668. Kth Smallest Number in Multiplication Table
        https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/description/
         */

        class FindKthSmallestNumInMultiplyTableSol
        {
            /* Approach #1: Brute Force [Memory Limit Exceeded]
Complexity Analysis
•	Time Complexity: O(m∗n) to create the table, and O(m∗nlog(m∗n)) to sort it.
•	Space Complexity: O(m∗n) to store the table.

             */
            public int Naive(int m, int n, int k)
            {
                int[] table = new int[m * n];
                for (int i = 1; i <= m; i++)
                {
                    for (int j = 1; j <= n; j++)
                    {
                        table[(i - 1) * n + j - 1] = i * j;
                    }
                }
                Array.Sort(table);
                return table[k - 1];
            }
            /* Approach #2: Next Heap [Time Limit Exceeded]
            Complexity Analysis
•	Time Complexity: O(k∗mlogm)=O((m^2)nlogm). Our initial heapify operation is O(m). Afterwards, each pop and push is O(mlogm), and our outer loop is O(k)=O(m∗n)
•	Space Complexity: O(m). Our heap is implemented as an array with m elements

             */
            public int UsingMinHeap(int rows, int columns, int k)
            {
                PriorityQueue<Node, Node> minHeap = new PriorityQueue<Node, Node>(Comparer<Node>.Create((a, b) => a.Value.CompareTo(b.Value)));

                for (int i = 1; i <= rows; i++)
                {
                    minHeap.Enqueue(new Node(i, i), new Node(i, i));
                }

                Node currentNode = null;
                for (int i = 0; i < k; i++)
                {
                    currentNode = minHeap.Dequeue();
                    int nextValue = currentNode.Value + currentNode.Root;
                    if (nextValue <= currentNode.Root * columns)
                    {
                        minHeap.Enqueue(new Node(nextValue, currentNode.Root), new Node(nextValue, currentNode.Root));
                    }
                }
                return currentNode.Value;
            }
            class Node
            {
                public int Value { get; set; }
                public int Root { get; set; }
                public Node(int value, int root)
                {
                    Value = value;
                    Root = root;
                }
            }

            class NodeComparer : IComparer<Node>
            {
                public int Compare(Node x, Node y)
                {
                    return x.Value.CompareTo(y.Value);
                }
            }

            /* Approach #3: Binary Search [Accepted]
            Complexity Analysis
•	Time Complexity: O(m∗log(m∗n)). Our binary search divides the interval [lo, hi] into half at each step. At each step, we call enough which requires O(m) time.
•	Space Complexity: O(1). We only keep integers in memory during our intermediate calculations.

             */
            public int UsingBinarySearch(int m, int n, int k)
            {
                int lo = 1, hi = m * n;
                while (lo < hi)
                {
                    int mi = lo + (hi - lo) / 2;
                    if (!Enough(mi, m, n, k)) lo = mi + 1;
                    else hi = mi;
                }
                return lo;
            }
            private bool Enough(int x, int m, int n, int k)
            {
                int count = 0;
                for (int i = 1; i <= m; i++)
                {
                    count += Math.Min(x / i, n);
                }
                return count >= k;
            }
        }


        /* 2360. Longest Cycle in a Graph
        https://leetcode.com/problems/longest-cycle-in-a-graph/description/
         */
        public class LongestCycleInGraphSol
        {
            int answer = -1;
            /* 
            Approach 1: Depth First Search
            Complexity Analysis
Here n is the number of nodes.
•	Time complexity: O(n).
o	Initializing the visit array takes O(n) time.
o	The dfs function visits each node once, which takes O(n) time in total. Because we have directed edges, each edge will be iterated once, resulting in O(n) operations in total while visiting all the nodes.
o	Each operation on the dist map takes O(1) time. Because we insert a distance for each node when it is visited, it will take O(n) time to insert distances for all of the nodes. It is also used to check the formation of a cycle when a previously visited node is encountered again. Because there are n nodes, it can be checked at most n times. It would also take O(n) time in that case.
•	Space complexity: O(n).
o	The visit array takes O(n) space.
o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.
o	The dist map can also have no more than n elements and hence it would take up O(n) space as well.

             */
            public int DFS(int[] edges)
            {
                int n = edges.Length;
                bool[] visit = new bool[n];

                for (int i = 0; i < n; i++)
                {
                    if (!visit[i])
                    {
                        Dictionary<int, int> dist = new Dictionary<int, int>();
                        dist[i] = 1;
                        Dfs(i, edges, dist, visit);
                    }
                }
                return answer;
            }
            public void Dfs(int node, int[] edges, Dictionary<int, int> dist, bool[] visit)
            {
                visit[node] = true;
                int neighbor = edges[node];

                if (neighbor != -1 && !visit[neighbor])
                {
                    dist[neighbor] = dist[node] + 1;
                    Dfs(neighbor, edges, dist, visit);
                }
                else if (neighbor != -1 && dist.ContainsKey(neighbor))
                {
                    answer = Math.Max(answer, dist[node] - dist[neighbor] + 1);
                }
            }

            /* Approach 2: Kahn's Algorithm
            Complexity Analysis
            Here n is the number of nodes.
            •	Time complexity: O(n).
            o	Initializing the visit and indegree arrays take O(n) time each.
            o	Each queue operation takes O(1) time, and a single node will be pushed once, leading to O(n) operations for n nodes. We iterate over the neighbor of each node that is popped out of the queue iterating over all the edges once. Since there are n edges at most, it would take O(n) time in total.
            o	We iterate over all the nodes that are in the cycles. There cannot be more than n nodes in all the cycles combined, so it would take O(n) time.
            •	Space complexity: O(n).
            o	The visit and indegree arrays takes O(n) space each.
            o	The queue can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

             */
            public int KahnsAlgor(int[] edges)
            {
                int n = edges.Length;
                bool[] visit = new bool[n];
                int[] indegree = new int[n];

                // Count indegree of each node.
                foreach (int edge in edges)
                {
                    if (edge != -1)
                    {
                        indegree[edge]++;
                    }
                }

                // Kahn's algorithm starts.
                Queue<int> q = new Queue<int>();
                for (int i = 0; i < n; i++)
                {
                    if (indegree[i] == 0)
                    {
                        q.Enqueue(i);
                    }
                }

                while (q.Count != 0)
                {
                    int node = q.Dequeue();
                    visit[node] = true;
                    int neighbor = edges[node];
                    if (neighbor != -1)
                    {
                        indegree[neighbor]--;
                        if (indegree[neighbor] == 0)
                        {
                            q.Enqueue(neighbor);
                        }
                    }
                }
                // Kahn's algorithm ends.

                int answer = -1;
                for (int i = 0; i < n; i++)
                {
                    if (!visit[i])
                    {
                        int neighbor = edges[i];
                        int count = 1;
                        visit[i] = true;
                        // Iterate in the cycle.
                        while (neighbor != i)
                        {
                            visit[neighbor] = true;
                            count++;
                            neighbor = edges[neighbor];
                        }
                        answer = Math.Max(answer, count);
                    }
                }
                return answer;
            }


        }


        /* 2328. Number of Increasing Paths in a Grid
        https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/description/
         */

        class CountPathsSol
        {
            /* Approach 1: Sorting + DP
Complexity Analysis
Let m×n be the size of the input array grid.
•	Time complexity: O(m⋅n⋅log(m⋅n))
o	We sort all cells by value, it takes O(klogk) to sort an array of size O(k), so it takes O(m⋅n⋅log(m⋅n)) time.
o	The iteration over sorted cells has O(m⋅n) steps, each step consists of checking at most four neighbor cells, thus it takes O(m⋅n) time.
o	For initialization of dp and the calculation of answer we iterate over all the cells of the dp array, which also takes O(m⋅n) time.
o	To sum up, the overall time complexity is O(m⋅n⋅log(m⋅n)).
•	Space complexity: O(m⋅n)
o	We used two arrays, cellList and dp, they both contain O(m⋅n) elements.

             */
            public int SortingAndDP(int[][] grid)
            {
                int[][] directions = new int[][] { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 } };
                int rowCount = grid.Length;
                int columnCount = grid[0].Length;
                int mod = 1_000_000_007;

                // Initialize dp, 1 stands for the path made by a cell itself.
                int[][] dp = new int[rowCount][];
                for (int i = 0; i < rowCount; i++)
                {
                    dp[i] = new int[columnCount];
                    Array.Fill(dp[i], 1);
                }

                // Sort all cells by value.
                int[][] cellList = new int[rowCount * columnCount][];
                for (int i = 0; i < rowCount; ++i)
                {
                    for (int j = 0; j < columnCount; ++j)
                    {
                        int index = i * columnCount + j;
                        cellList[index] = new int[] { i, j };
                    }
                }
                Array.Sort(cellList, (a, b) => grid[a[0]][a[1]].CompareTo(grid[b[0]][b[1]]));

                // Iterate over the sorted cells, for each cell grid[i][j]: 
                foreach (int[] cell in cellList)
                {
                    int i = cell[0], j = cell[1];

                    // Check its four neighbor cells, if a neighbor cell grid[currI][currJ] has a
                    // larger value, increment dp[currI][currJ] by dp[i][j]
                    foreach (int[] d in directions)
                    {
                        int currI = i + d[0], currJ = j + d[1];
                        if (0 <= currI && currI < rowCount && 0 <= currJ && currJ < columnCount
                           && grid[currI][currJ] > grid[i][j])
                        {
                            dp[currI][currJ] += dp[i][j];
                            dp[currI][currJ] %= mod;
                        }
                    }
                }

                // Sum over dp[i][j].
                int answer = 0;
                for (int i = 0; i < rowCount; ++i)
                {
                    for (int j = 0; j < columnCount; ++j)
                    {
                        answer += dp[i][j];
                        answer %= mod;
                    }
                }
                return answer;
            }

            /* Approach 2: DFS with Memoization
            Complexity Analysis
Let m×n be the size of the input array grid.
•	Time complexity: O(m⋅n)
o	We used dp as memory to avoid repeated computation, so each cell is only visited and calculated once.
o	Initialization of the dp array also takes O(m⋅n) time.
•	Space complexity: O(m⋅n)
o	We build the auxiliary array dp of the same size as grid.
o	The space complexity of recursive algorithm is proportional to the maximum depth of the recursion tree generated. There are at most m⋅n recursive call of dfs in the stack simultaneously, thus the stack takes O(m⋅n) space.
o	To sum up, the space complexity is O(m⋅n).

             */
            private int[,] dp;
            private readonly int[,] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
            private const int mod = 1_000_000_007;

            public int DFSWithMemo(int[,] grid)
            {
                int m = grid.GetLength(0), n = grid.GetLength(1);
                dp = new int[m, n];

                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        dp[i, j] = -1;
                    }
                }

                // Iterate over all cells grid[i, j] and sum over Dfs(i, j).
                int totalPaths = 0;
                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        totalPaths = (totalPaths + Dfs(grid, i, j)) % mod;
                    }
                }

                return totalPaths;
            }
            private int Dfs(int[,] grid, int i, int j)
            {
                // If dp[i, j] is non-zero, it means we have got the value of Dfs(i, j),
                // so just return dp[i, j].
                if (dp[i, j] != -1)
                    return dp[i, j];

                // Otherwise, set answer = 1, the path made of grid[i, j] itself.
                int answer = 1;

                // Check its four neighbor cells, if a neighbor cell grid[prevI, prevJ] has a
                // smaller value, we move to this cell and solve the subproblem: Dfs(prevI, prevJ).
                for (int d = 0; d < directions.GetLength(0); d++)
                {
                    int prevI = i + directions[d, 0], prevJ = j + directions[d, 1];
                    if (0 <= prevI && prevI < grid.GetLength(0) && 0 <= prevJ &&
                        prevJ < grid.GetLength(1) && grid[prevI, prevJ] < grid[i, j])
                    {
                        answer += Dfs(grid, prevI, prevJ);
                        answer %= mod;
                    }
                }

                // Update dp[i, j], so that we don't recalculate its value later.
                dp[i, j] = answer;
                return answer;
            }


        }


        /* 675. Cut Off Trees for Golf Event
        https://leetcode.com/problems/cut-off-trees-for-golf-event/description/
         */
        class CutOffTreeForGolfEventSol
        {
            int[] dr = { -1, 1, 0, 0 };
            int[] dc = { 0, 0, -1, 1 };

            /* Approach #1: BFS [Accepted]
            Complexity Analysis
            All three algorithms have similar worst-case complexities, but in practice, each successive algorithm presented performs faster on random data.
            •	Time Complexity: O((RC)^2) where there are R rows and C columns in the given forest. We walk to R∗C trees, and each walk could spend O(R∗C) time searching for the tree.
            •	Space Complexity: O(R∗C), the maximum size of the data structures used.

             */
            public int BFS(List<List<int>> forest)
            {
                List<int[]> trees = new();
                for (int r = 0; r < forest.Count; ++r)
                {
                    for (int c = 0; c < forest[0].Count; ++c)
                    {
                        int v = forest[r][c];
                        if (v > 1) trees.Add(new int[] { v, r, c });
                    }
                }
                trees.Sort((a, b) => a[0].CompareTo(b[0]));

                int ans = 0, sr = 0, sc = 0;
                foreach (int[] tree in trees)
                {
                    int d = Bfs(forest, sr, sc, tree[1], tree[2]);
                    if (d < 0) return -1;
                    ans += d;
                    sr = tree[1]; sc = tree[2];
                }
                return ans;
            }
            private int Bfs(List<List<int>> forest, int startRow, int startCol, int targetRow, int targetCol)
            {
                int rowCount = forest.Count, colCount = forest[0].Count;
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { startRow, startCol, 0 });
                bool[,] visited = new bool[rowCount, colCount];
                visited[startRow, startCol] = true;

                while (queue.Count > 0)
                {
                    int[] current = queue.Dequeue();
                    if (current[0] == targetRow && current[1] == targetCol) return current[2];

                    for (int directionIndex = 0; directionIndex < 4; ++directionIndex)
                    {
                        int newRow = current[0] + dr[directionIndex];
                        int newCol = current[1] + dc[directionIndex];
                        if (0 <= newRow && newRow < rowCount && 0 <= newCol && newCol < colCount &&
                            !visited[newRow, newCol] && forest[newRow][newCol] > 0)
                        {
                            visited[newRow, newCol] = true;
                            queue.Enqueue(new int[] { newRow, newCol, current[2] + 1 });
                        }
                    }
                }
                return -1;
            }
            /*             Approach #2: A* Search [Accepted]
   Complexity Analysis
            All three algorithms have similar worst-case complexities, but in practice, each successive algorithm presented performs faster on random data.
            •	Time Complexity: O((RC)^2) where there are R rows and C columns in the given forest. We walk to R∗C trees, and each walk could spend O(R∗C) time searching for the tree.
            •	Space Complexity: O(R∗C), the maximum size of the data structures used.

             */
            public int UsingAStarAlgo(List<List<int>> forest)
            {
                List<int[]> trees = new();
                for (int r = 0; r < forest.Count; ++r)
                {
                    for (int c = 0; c < forest[0].Count; ++c)
                    {
                        int v = forest[r][c];
                        if (v > 1) trees.Add(new int[] { v, r, c });
                    }
                }
                trees.Sort((a, b) => a[0].CompareTo(b[0]));

                int ans = 0, sr = 0, sc = 0;
                foreach (int[] tree in trees)
                {
                    int d = AStarAlgo(forest, sr, sc, tree[1], tree[2]);
                    if (d < 0) return -1;
                    ans += d;
                    sr = tree[1]; sc = tree[2];
                }
                return ans;
            }
            private int AStarAlgo(List<List<int>> forest, int sr, int sc, int tr, int tc)
            {
                int R = forest.Count, C = forest[0].Count;
                PriorityQueue<int[], int[]> heap = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => a[0].CompareTo(b[0])));

                heap.Enqueue(new int[] { 0, 0, sr, sc }, new int[] { 0, 0, sr, sc });

                Dictionary<int, int> cost = new();
                cost.Add(sr * C + sc, 0);

                while (heap.Count > 0)
                {
                    int[] cur = heap.Dequeue();
                    int g = cur[1], r = cur[2], c = cur[3];
                    if (r == tr && c == tc) return g;
                    for (int di = 0; di < 4; ++di)
                    {
                        int nr = r + dr[di], nc = c + dc[di];
                        if (0 <= nr && nr < R && 0 <= nc && nc < C && forest[nr][nc] > 0)
                        {
                            int ncost = g + 1 + Math.Abs(nr - tr) + Math.Abs(nc - tr);
                            if (ncost < cost.GetValueOrDefault(nr * C + nc, 9999))
                            {
                                cost[nr * C + nc] = ncost;
                                heap.Enqueue(new int[] { ncost, g + 1, nr, nc }, new int[] { ncost, g + 1, nr, nc });
                            }
                        }
                    }
                }
                return -1;
            }

            /*             Approach #3: Hadlock's Algorithm [Accepted]
               Complexity Analysis
                        All three algorithms have similar worst-case complexities, but in practice, each successive algorithm presented performs faster on random data.
                        •	Time Complexity: O((RC)^2) where there are R rows and C columns in the given forest. We walk to R∗C trees, and each walk could spend O(R∗C) time searching for the tree.
                        •	Space Complexity: O(R∗C), the maximum size of the data structures used.
             */
            public int UsingHadlocksAlgo(List<List<int>> forest)
            {
                List<int[]> trees = new();
                for (int r = 0; r < forest.Count; ++r)
                {
                    for (int c = 0; c < forest[0].Count; ++c)
                    {
                        int v = forest[r][c];
                        if (v > 1) trees.Add(new int[] { v, r, c });
                    }
                }
                trees.Sort((a, b) => a[0].CompareTo(b[0]));

                int ans = 0, sr = 0, sc = 0;
                foreach (int[] tree in trees)
                {
                    int d = CalculateHadlocks(forest, sr, sc, tree[1], tree[2]);
                    if (d < 0) return -1;
                    ans += d;
                    sr = tree[1]; sc = tree[2];
                }
                return ans;
            }

            private int CalculateHadlocks(List<List<int>> forest, int startRow, int startCol, int targetRow, int targetCol)
            {
                int rows = forest.Count, columns = forest[0].Count;
                HashSet<int> processed = new HashSet<int>();
                LinkedList<int[]> deque = new LinkedList<int[]>();
                deque.AddFirst(new int[] { 0, startRow, startCol });

                while (deque.Count > 0)
                {
                    int[] current = deque.First.Value;
                    deque.RemoveFirst();
                    int detours = current[0], row = current[1], col = current[2];

                    if (!processed.Contains(row * columns + col))
                    {
                        processed.Add(row * columns + col);
                        if (row == targetRow && col == targetCol)
                        {
                            return Math.Abs(startRow - targetRow) + Math.Abs(startCol - targetCol) + 2 * detours;
                        }
                        for (int directionIndex = 0; directionIndex < 4; ++directionIndex)
                        {
                            int newRow = row + dr[directionIndex];
                            int newCol = col + dc[directionIndex];
                            bool isCloser;
                            if (directionIndex <= 1)
                                isCloser = directionIndex == 0 ? row > targetRow : row < targetRow;
                            else
                                isCloser = directionIndex == 2 ? col > targetCol : col < targetCol;

                            if (0 <= newRow && newRow < rows && 0 <= newCol && newCol < columns && forest[newRow][newCol] > 0)
                            {
                                if (isCloser)
                                    deque.AddFirst(new int[] { detours, newRow, newCol });
                                else
                                    deque.AddLast(new int[] { detours + 1, newRow, newCol });
                            }
                        }
                    }
                }
                return -1;
            }


        }



        /* 1515. Best Position for a Service Centre
        https://leetcode.com/problems/best-position-for-a-service-centre/description/
         */
        class GetMinDistSumSol
        {
            /*
            Time and Space Complexity
Time Complexity
The given code snippet is an iterative method designed to find the position that minimizes the sum of distances to all points in the array positions. Let's analyze the time complexity step by step:
•	The initialization step calculates the centroid by averaging the x and y coordinates. This loop runs n times, where n is the number of positions. Hence, it has a time complexity of O(n).
•	The while loop does not have a fixed number of iterations, as it continues until the changes in x and y are smaller than eps. However, within this loop, every iteration involves a loop through all n positions to compute gradients and distances.
•	Within this nested loop, the time complexity of the operations (calculations for a, b, c, grad_x, grad_y, and dist) is constant, O(1).
•	The update of x and y and the condition check are also constant time operations.
Therefore, if we denote the number of iterations the while loop runs as k, the total time complexity would be O(nk), where k depends on the initial positions, the decay factor alpha, and the threshold eps.
Space Complexity
The space complexity is determined by the extra space used:
•	Variables x, y, grad_x, grad_y, dist, dx, and dy use constant space.
•	The code does not use any additional data structures that grow with the input. Only fixed amount of extra space is needed to store intermediate calculational variables.
Consequently, the space complexity of the algorithm is O(1).

            */
            public double GetMinDistSum(int[][] positions)
            {
                int n = positions.Length; // Number of positions
                double centerX = 0, centerY = 0; // Initialize center x and y with 0

                // Calculate initial centroid by averaging all positions
                foreach (int[] position in positions)
                {
                    centerX += position[0];
                    centerY += position[1];
                }
                centerX /= n;
                centerY /= n;

                // Set decay factor for the learning rate and an epsilon for convergence condition
                double decayFactor = 0.999;
                double convergenceThreshold = 1e-6;

                // Start with an initial learning rate
                double learningRate = 0.5;

                // Use Gradient Descent to minimize the distance sum
                while (true)
                {
                    double gradientX = 0, gradientY = 0; // Initialize gradients
                    double totalDistance = 0; // Initialize total distance

                    // Compute gradients for X and Y with respect to the objective function
                    foreach (int[] position in positions)
                    {
                        double deltaX = centerX - position[0], deltaY = centerY - position[1];
                        double distance = Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
                        gradientX += deltaX / (distance + 1e-8); // Add small value to avoid division by zero
                        gradientY += deltaY / (distance + 1e-8);
                        totalDistance += distance; // Sum up total distance
                    }

                    // Scale the gradient by the learning rate
                    double stepX = gradientX * learningRate;
                    double stepY = gradientY * learningRate;

                    // Check for convergence
                    if (Math.Abs(stepX) <= convergenceThreshold && Math.Abs(stepY) <= convergenceThreshold)
                    {
                        return totalDistance; // Return the minimized total distance
                    }

                    // Update the center position by taking a step against the gradient direction
                    centerX -= stepX;
                    centerY -= stepY;

                    // Reduce the learning rate by the decay factor
                    learningRate *= decayFactor;
                }
            }
        }


        /* 1889. Minimum Space Wasted From Packaging	
        https://leetcode.com/problems/minimum-space-wasted-from-packaging/description/
         */

        public class MinWastedSpaceSol
        {
            /* 
            Approach: Sorting+Binary Search
           Time and Space Complexity
Time Complexity
The time complexity of the given code can be analyzed as follows:
•	Sorting the packages list takes O(NlogN) time, with N being the length of the packages.
•	The for loop iterates through each box in boxes. Let's say there are M boxes.
•	Each box is also sorted, taking O(BlogB) time, where B is the maximum number of items in a single box.
•	The inner for loop iterates through each item b in the box. The bisect_right function is also called inside this loop, which works in O(logN) time complexity for each b because it uses a binary search algorithm.
o	In the worst case, every call to bisect_right can iterate over all elements of packages, thus the combined complexity of the loop with the bisect_right is O(BlogN). However, since i = j assigns the new starting index after every bisect, it ensures that each package is considered only once across all boxes. Hence, the complexity for all boxes together should be O(BlogN), not O(M*B*logN).
•	The overall time complexity is O(NlogN + M*B*logN) since the sorting of the packages and the boxes are the dominating factors.
Space Complexity
The space complexity of the code can be determined as follows:
•	The sorted packages list and box list require additional space, which contributes to the space complexity. This could be, in the worst case, O(N + B) respective space for sorted packages and the largest box.
•	The variables ans, s, i, j, and mod use constant space, which does not depend on the input size, hence contributing O(1) space.
•	Consequently, the total space complexity is O(N + B).
Note: inf and mod are constants defined in the global namespace, and their space is considered as O(1).

*/
            public int SortingAndBinarySearch(int[] packages, int[][] boxes)
            {
                Array.Sort(packages);        // Sort the packages for binary search
                // Define a high value for initial comparison
                long infinity = (long)1e11;
                long result = infinity, modulo = (long)1e9 + 7, sumArrayA = 0L;

                // Calculate the total size of all packages
                foreach (int elementA in packages)
                    sumArrayA += elementA;

                foreach (int[] box in boxes)
                {
                    // Sort each type of box since we need to handle them sequentially                    
                    Array.Sort(box);
                    // Skip the box type if the largest box cannot hold the largest package
                    if (box[box.Length - 1] < packages[packages.Length - 1]) continue;

                    long currentWastedSpace = 0, indexA = 0, indexB;

                    // Calculate the waste for this box type
                    foreach (int elementB in box)
                    {
                        indexB = BinarySearch(packages, elementB + 1);
                        currentWastedSpace += elementB * (indexB - indexA);
                        indexA = indexB;
                    }
                    // Update the minimum waste if the current one is smaller
                    result = Math.Min(result, currentWastedSpace);
                }

                // Return -1 if no box type can accommodate all packages
                // Modulo for the final result to avoid number overflow
                return result < infinity ? (int)((result - sumArrayA) % modulo) : -1;
            }

            // Custom binary search function to find the upper bound
            private int BinarySearch(int[] arrayA, int valueB)
            {
                int left = 0, right = arrayA.Length;
                while (left < right)
                {
                    int middle = (left + right) / 2;
                    if (arrayA[middle] < valueB)
                        left = middle + 1;
                    else
                        right = middle;
                }
                return left;
            }
        }

        /* 2435. Paths in Matrix Whose Sum Is Divisible by K
        https://leetcode.com/problems/paths-in-matrix-whose-sum-is-divisible-by-k/description/
         */
        class NumPathsSumDivisibleByKSol
        {
            /* 
            1. Dynamic Programming, 

Time and Space Complexity
The provided Python code defines a method to calculate the number of paths on a 2D grid where the sum of the values along the path is divisible by k. It uses dynamic programming to store the counts for intermediate paths where the sum of the values modulo k is a specific remainder.
Time Complexity:
The time complexity of the given code can be analyzed by considering the three nested loops:
1.	The outermost loop runs for m iterations, where m is the number of rows in the grid.
2.	The middle loop runs for n iterations for each i, where n is the number of columns in the grid.
3.	The innermost loop runs for k iterations for each combination of i and j.
Combining these, we get m * n * k iterations in total. Within the innermost loop, all operations are constant time. Hence, the time complexity is O(m * n * k).
Space Complexity:
The space complexity is determined by the size of the dp array, which stores intermediate counts for each cell and each possible remainder modulo k:
•	The dp array is a 3-dimensional array with dimensions m, n, and k.
•	This results in a space requirement for m * n * k integers.
Hence, the space complexity of the code is also O(m * n * k).
            */
            // Define the modulus constant for preventing integer overflow
            private const int MOD = (int)1e9 + 7;

            public int UsingDP(int[][] grid, int k)
            {
                // m and n represent the dimensions of the grid
                int numRows = grid.Length;
                int numCols = grid[0].Length;

                // 3D dp array to store the number of ways to reach a cell (i, j) 
                // such that the path sum modulo k is s
                int[][][] dp = new int[numRows][][];

                // Base case: start at the top-left corner of the grid
                dp[0][0][grid[0][0] % k] = 1;

                // Iterate over all cells of the grid
                for (int i = 0; i < numRows; ++i)
                {
                    for (int j = 0; j < numCols; ++j)
                    {
                        // Try all possible sums modulo k
                        for (int sumModK = 0; sumModK < k; ++sumModK)
                        {
                            // Calculate the modulo to identify how the current value of grid contributes to the new sum
                            int remainder = ((sumModK - grid[i][j] % k) + k) % k;

                            // If not in the first row, add paths from the cell above
                            if (i > 0)
                            {
                                dp[i][j][sumModK] += dp[i - 1][j][remainder];
                            }
                            // If not in the first column, add paths from the cell on the left
                            if (j > 0)
                            {
                                dp[i][j][sumModK] += dp[i][j - 1][remainder];
                            }

                            // Use modulus operation to prevent integer overflow
                            dp[i][j][sumModK] %= MOD;
                        }
                    }
                }

                // The result is the number of ways to reach the bottom-right corner such that path sum modulo k is 0
                return dp[numRows - 1][numCols - 1][0];
            }
        }


        /* 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
        https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/description/
         */
        public class FindCriticalAndPseudoCriticalEdgesSol
        {
            /* Approach 1: Kruskal's Algorithm
Complexity Analysis
•	Time complexity of this algorithm is O(m^2⋅α(n)), where m is the number of edges, n is the number of nodes and α is the inverse Ackermann function.
o	Sorting the edges. The first operation in this algorithm is sorting the edges. We perform this operation once in O(mlogm) time.
o	Constructing the MST by ignoring/forcing an edge. For each edge in our sorted list, we construct two MSTs – one where we force include the edge in the MST and one where we ignore it. To do this, we use the Union-Find data structure, performing union operations to connect the nodes in the graph. The time complexity of these union operations with union by rank and path compression optimization is nearly a constant time operation, represented as O(α(n)), where α is the inverse Ackermann function. You do not have to know what exactly this function is. It suffices to know that this function grows extremely slowly, so much so that for any conceivable practical input, it does not exceed 5. Hence for each edge, it would take O(m⋅α(n)) time to construct the MST.
o	Iterating through all edges. The previous step is repeated for each edge in the graph, meaning we perform it m times. This results in a total time complexity of O(m^2⋅α(n)) for constructing all the MSTs.
Adding these all together, we find that the total time complexity of this algorithm is O(mlogm+m^2⋅α(n)), which simplifies to O(m^2⋅α(n)).
•	Space complexity is O(m).
o	Storing the edges. We need to store all the edges and their information in our program, which requires O(m) space.
o	Union-Find data structure. The Union-Find data structure uses an array to keep track of the parent of each node and another array to keep track of the size of each tree in the forest. It requires O(n) space, where n is the number of nodes in the graph.
When we add these components together, we find that the total space complexity of this algorithm is O(m+n). Since the graph is connected, thus m≥n−1 and O(m+n)=O(m).

             */
            public List<List<int>> UsingKruskalsAlgo(int numberOfVertices, int[][] edges)
            {
                // Add index to edges for tracking
                int numberOfEdges = edges.Length;
                int[][] newEdges = new int[numberOfEdges][];
                for (int i = 0; i < numberOfEdges; i++)
                {
                    newEdges[i] = new int[4];
                    for (int j = 0; j < 3; j++)
                    {
                        newEdges[i][j] = edges[i][j];
                    }
                    newEdges[i][3] = i;
                }

                // Sort edges based on weight
                Array.Sort(newEdges, (edge1, edge2) => edge1[2].CompareTo(edge2[2]));

                // Find MST weight using union-find
                UnionFind unionFindStandard = new UnionFind(numberOfVertices);
                int standardWeight = 0;
                foreach (int[] edge in newEdges)
                {
                    if (unionFindStandard.Union(edge[0], edge[1]))
                    {
                        standardWeight += edge[2];
                    }
                }

                List<List<int>> result = new List<List<int>> {
            new List<int>(),
            new List<int>()
        };

                // Check each edge for critical and pseudo-critical
                for (int i = 0; i < numberOfEdges; i++)
                {
                    // Ignore this edge and calculate MST weight
                    UnionFind unionFindIgnore = new UnionFind(numberOfVertices);
                    int ignoreWeight = 0;
                    for (int j = 0; j < numberOfEdges; j++)
                    {
                        if (i != j && unionFindIgnore.Union(newEdges[j][0], newEdges[j][1]))
                        {
                            ignoreWeight += newEdges[j][2];
                        }
                    }
                    // If the graph is disconnected or the total weight is greater, 
                    // the edge is critical
                    if (unionFindIgnore.MaxSize < numberOfVertices || ignoreWeight > standardWeight)
                    {
                        result[0].Add(newEdges[i][3]);
                    }
                    else
                    {
                        // Force this edge and calculate MST weight
                        UnionFind unionFindForce = new UnionFind(numberOfVertices);
                        unionFindForce.Union(newEdges[i][0], newEdges[i][1]);
                        int forceWeight = newEdges[i][2];
                        for (int j = 0; j < numberOfEdges; j++)
                        {
                            if (i != j && unionFindForce.Union(newEdges[j][0], newEdges[j][1]))
                            {
                                forceWeight += newEdges[j][2];
                            }
                        }
                        // If total weight is the same, the edge is pseudo-critical
                        if (forceWeight == standardWeight)
                        {
                            result[1].Add(newEdges[i][3]);
                        }
                    }
                }

                return result;
            }

            private class UnionFind
            {
                private int[] parent;
                private int[] size;
                public int MaxSize;

                public UnionFind(int numberOfVertices)
                {
                    parent = new int[numberOfVertices];
                    size = new int[numberOfVertices];
                    MaxSize = 1;
                    for (int i = 0; i < numberOfVertices; i++)
                    {
                        parent[i] = i;
                        size[i] = 1;
                    }
                }

                public int Find(int vertex)
                {
                    // Finds the root of vertex
                    if (vertex != parent[vertex])
                    {
                        parent[vertex] = Find(parent[vertex]);
                    }
                    return parent[vertex];
                }

                public bool Union(int vertexX, int vertexY)
                {
                    // Connects vertexX and vertexY
                    int rootX = Find(vertexX);
                    int rootY = Find(vertexY);
                    if (rootX != rootY)
                    {
                        if (size[rootX] < size[rootY])
                        {
                            int temp = rootX;
                            rootX = rootY;
                            rootY = temp;
                        }
                        parent[rootY] = rootX;
                        size[rootX] += size[rootY];
                        MaxSize = Math.Max(MaxSize, size[rootX]);
                        return true;
                    }
                    return false;
                }
            }
        }

        /* 
        1857. Largest Color Value in a Directed Graph
        https://leetcode.com/problems/largest-color-value-in-a-directed-graph/description/
         */
        public class LargestPathValueSol
        {

            /* Approach 1: Topological Sort Using Kahn's Algorithm
            Complexity Analysis
            Here, n be the number of nodes and m be the number of edges in the graph.
            •	Time complexity: O(26⋅m+26⋅n)=O(m+n).
            o	Initializing the adj takes O(m) time as we go through all the edges. The indegree array take O(n) time and the count array takes O(26⋅n) time.
            o	Each queue operation takes O(1) time, and a single node will be pushed once, leading to O(n) operations for n nodes. We iterate over the neighbor of each node that is popped out of the queue iterating over all the edges once. Since there are m edges at most and while iterating over each edge we try to update the frequencies of all the 26 colors, it would take O(26⋅m) time.
            •	Space complexity: O(m+26⋅n)=O(m+n).
            o	The adj arrays takes O(m) space. The count array takes O(26⋅n) space.
            o	The queue can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

             */
            public int TopologicalSortUsingKahnsAlgo(string colors, int[][] edges)
            {
                int numberOfNodes = colors.Length;
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                int[] indegree = new int[numberOfNodes];

                foreach (int[] edge in edges)
                {
                    if (!adjacencyList.ContainsKey(edge[0]))
                    {
                        adjacencyList[edge[0]] = new List<int>();
                    }
                    adjacencyList[edge[0]].Add(edge[1]);
                    indegree[edge[1]]++;
                }

                int[,] colorCount = new int[numberOfNodes, 26];
                Queue<int> queue = new Queue<int>();

                // Push all the nodes with indegree zero in the queue.
                for (int i = 0; i < numberOfNodes; i++)
                {
                    if (indegree[i] == 0)
                    {
                        queue.Enqueue(i);
                    }
                }

                int maxColorValue = 1, nodesSeen = 0;
                while (queue.Count > 0)
                {
                    int currentNode = queue.Dequeue();
                    maxColorValue = Math.Max(maxColorValue, ++colorCount[currentNode, colors[currentNode] - 'a']);
                    nodesSeen++;

                    if (!adjacencyList.ContainsKey(currentNode))
                    {
                        continue;
                    }

                    foreach (int neighbor in adjacencyList[currentNode])
                    {
                        for (int i = 0; i < 26; i++)
                        {
                            // Try to update the frequency of colors for the neighbor to include paths
                            // that use currentNode->neighbor edge.
                            colorCount[neighbor, i] = Math.Max(colorCount[neighbor, i], colorCount[currentNode, i]);
                        }

                        indegree[neighbor]--;
                        if (indegree[neighbor] == 0)
                        {
                            queue.Enqueue(neighbor);
                        }
                    }
                }

                return nodesSeen < numberOfNodes ? -1 : maxColorValue;
            }

            /* Approach 2: Depth First Search
            Complexity Analysis
Here, n be the number of nodes and m be the number of edges in the graph.
•	Time complexity: O(26⋅m+26⋅n)=O(m+n).
o	Initializing the adj takes O(m) time as we go through all the edges. The count array takes O(26⋅n) time.
o	The dfs function visits each node once, which takes O(n) time in total. Since there are m edges at most and while iterating over each edge we try to update the frequencies of all the 26 colors, it would take O(26⋅m) time.
•	Space complexity: O(m+26⋅n)=O(m+n).
o	The adj arrays takes O(m) space. The count array takes O(26⋅n) space.
o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

             */
            public int DFS(string colors, int[,] edges)
            {
                int n = colors.Length;
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();

                for (int i = 0; i < edges.GetLength(0); i++)
                {
                    int from = edges[i, 0];
                    int to = edges[i, 1];
                    if (!adjacencyList.ContainsKey(from))
                    {
                        adjacencyList[from] = new List<int>();
                    }
                    adjacencyList[from].Add(to);
                }

                int[,] count = new int[n, 26];
                bool[] visited = new bool[n];
                bool[] inStack = new bool[n];
                int answer = 0;
                for (int i = 0; i < n; i++)
                {
                    answer = Math.Max(answer, Dfs(i, colors, adjacencyList, count, visited, inStack));
                }

                return answer == int.MaxValue ? -1 : answer;
            }
            private int Dfs(int node, string colors, Dictionary<int, List<int>> adjacencyList, int[,] count,
        bool[] visited, bool[] inStack)
            {
                // If the node is already in the stack, we have a cycle.
                if (inStack[node])
                {
                    return int.MaxValue;
                }
                if (visited[node])
                {
                    return count[node, colors[node] - 'a'];
                }
                // Mark the current node as visited and part of current recursion stack.
                visited[node] = true;
                inStack[node] = true;

                if (adjacencyList.ContainsKey(node))
                {
                    foreach (int neighbor in adjacencyList[node])
                    {
                        if (Dfs(neighbor, colors, adjacencyList, count, visited, inStack) == int.MaxValue)
                        {
                            return int.MaxValue;
                        }
                        for (int i = 0; i < 26; i++)
                        {
                            count[node, i] = Math.Max(count[node, i], count[neighbor, i]);
                        }
                    }
                }

                // After all the incoming edges to the node are processed,
                // we count the color on the node itself.
                count[node, colors[node] - 'a']++;
                // Remove the node from the stack.
                inStack[node] = false;
                return count[node, colors[node] - 'a'];
            }
        }

        /* 1091. Shortest Path in Binary Matrix
        https://leetcode.com/problems/shortest-path-in-binary-matrix/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM

         */
        public class ShortestPathBinaryMatrixSol
        {
            private static readonly int[][] directions =
                new int[][] { new int[] { -1, -1 }, new int[] { -1, 0 }, new int[] { -1, 1 }, new int[] { 0, -1 }, new int[] { 0, 1 }, new int[] { 1, -1 }, new int[] { 1, 0 }, new int[] { 1, 1 } };

            /* Approach 1: Breadth-first Search (BFS), Overwriting Input
            Complexity Analysis
            Let N be the number of cells in the grid.
            •	Time complexity : O(N).
            Each cell was guaranteed to be enqueued at most once. This is because a condition for a cell to be enqueued was that it had a zero in the grid, and when enqueuing, we also permanently changed the cell's grid value to be non-zero.
            The outer loop ran as long as there were still cells in the queue, dequeuing one each time. Therefore, it ran at most N times, giving a time complexity of O(N).
            The inner loop iterated over the unvisited neighbors of the cell that was dequeued by the outer loop. There were at most 8 neighbors. Identifying the unvisited neighbors is an O(1) operation because we treat the 8 as a constant.
            Therefore, we have a time complexity of O(N).
            •	Space complexity : O(N).
            The only additional space we used was the queue. We determined above that at most, we enqueued N cells. Therefore, an upper bound on the worst-case space complexity is O(N).
            Given that BFS will have nodes of at most two unique distances on the queue at any one time, it would be reasonable to wonder if the worst-case space complexity is actually lower. But actually, it turns out that there are cases with massive grids where the number of cells at a single distance is proportional to N. So even with cells of a single distance on the queue, in the worst case, the space needed is O(N).

             */
            public int BFSWithOverwritingInput(int[][] grid)
            {
                // Firstly, we need to check that the start and target cells are open.
                if (grid[0][0] != 0 || grid[grid.Length - 1][grid[0].Length - 1] != 0)
                {
                    return -1;
                }

                // Set up the BFS.
                Queue<int[]> queue = new Queue<int[]>();
                grid[0][0] = 1;
                queue.Enqueue(new int[] { 0, 0 });

                // Carry out the BFS
                while (queue.Count > 0)
                {
                    int[] cell = queue.Dequeue();
                    int row = cell[0];
                    int col = cell[1];
                    int distance = grid[row][col];
                    if (row == grid.Length - 1 && col == grid[0].Length - 1)
                    {
                        return distance;
                    }
                    foreach (int[] neighbour in GetNeighbours(row, col, grid))
                    {
                        int neighbourRow = neighbour[0];
                        int neighbourCol = neighbour[1];
                        queue.Enqueue(new int[] { neighbourRow, neighbourCol });
                        grid[neighbourRow][neighbourCol] = distance + 1;
                    }
                }

                // The target was unreachable.
                return -1;
            }

            private List<int[]> GetNeighbours(int row, int col, int[][] grid)
            {
                List<int[]> neighbours = new List<int[]>();
                for (int i = 0; i < directions.Length; i++)
                {
                    int newRow = row + directions[i][0];
                    int newCol = col + directions[i][1];
                    if (newRow < 0 || newCol < 0 || newRow >= grid.Length
                            || newCol >= grid[0].Length
                            || grid[newRow][newCol] != 0)
                    {
                        continue;
                    }
                    neighbours.Add(new int[] { newRow, newCol });
                }
                return neighbours;
            }
            /* Approach 2: Breadth-first Search (Without Overwriting the Input)
Complexity Analysis
Let N be the number of cells in the grid.
•	Time complexity : O(N).
Same as approach 1. Processing a cell is O(1), and each of the N cells is processed at most once, giving a total of O(N).
•	Space complexity : O(N).
Same as approach 1. The visited set also requires O(N) space; in the worst case, it will hold the row and column of each of the N cells.

             */
            public int BFSWithoutOverwritingInput(int[][] grid)
            {

                // Firstly, we need to check that the start and target cells are open.
                if (grid[0][0] != 0 || grid[grid.Length - 1][grid[0].Length - 1] != 0)
                {
                    return -1;
                }

                // Set up the BFS.
                Queue<int[]> queue = new();
                queue.Enqueue(new int[] { 0, 0, 1 }); // Put distance on the queue
                bool[][] visited = new bool[grid.Length][]; // Used as visited set.
                visited[0][0] = true;

                // Carry out the BFS
                while (queue.Count > 0)
                {
                    int[] cell = queue.Dequeue();
                    int row = cell[0];
                    int col = cell[1];
                    int distance = cell[2];
                    // Check if this is the target cell.
                    if (row == grid.Length - 1 && col == grid[0].Length - 1)
                    {
                        return distance;
                    }
                    foreach (int[] neighbour in GetNeighbours(row, col, grid))
                    {
                        int neighbourRow = neighbour[0];
                        int neighbourCol = neighbour[1];
                        if (visited[neighbourRow][neighbourCol])
                        {
                            continue;
                        }
                        visited[neighbourRow][neighbourCol] = true;
                        queue.Enqueue(new int[] { neighbourRow, neighbourCol, distance + 1 });
                    }
                }

                // The target was unreachable.
                return -1;
            }
            /* Approach 3: A* (Advanced)
Complexity Analysis
Let N be the number of cells in the grid.
•	Time complexity : O(NlogN).
The difference between this approach and the previous one is that adding and removing items from a priority queue is O(logN), as opposed to O(1). Given that we are adding/ removing O(N) items, this gives a time complexity of O(NlogN).
•	Space complexity : O(N).
Interestingly, there are ways to reduce the time complexity back down to O(N). The simplest is to recognize that there will be at most 3 unique estimates on the priority queue at any one time, and so to maintain 3 lists instead of a priority queue. Adding and removing from lists is O(1), bringing the total time complexity back down to O(N).
             */
            public int UsingAStarAlgo(int[][] grid)
            {

                // Firstly, we need to check that the start and target cells are open.
                if (grid[0][0] != 0 || grid[grid.Length - 1][grid[0].Length - 1] != 0)
                {
                    return -1;
                }

                // Set up the A* search.
                PriorityQueue<Candidate, Candidate> pq = new PriorityQueue<Candidate, Candidate>(
                    Comparer<Candidate>.Create((a, b) => a.TotalEstimate.CompareTo(b.TotalEstimate)));
                pq.Enqueue(new Candidate(0, 0, 1, Estimate(0, 0, grid)), new Candidate(0, 0, 1, Estimate(0, 0, grid)));

                bool[][] visited = new bool[grid.Length][];

                // Carry out the A* search.
                while (pq.Count > 0)
                {

                    Candidate best = pq.Dequeue();

                    // Is this for the target cell?
                    if (best.Row == grid.Length - 1 && best.Col == grid[0].Length - 1)
                    {
                        return best.DistanceSoFar;
                    }

                    // Have we already found the best path to this cell?
                    if (visited[best.Row][best.Col])
                    {
                        continue;
                    }

                    visited[best.Row][best.Col] = true;

                    foreach (int[] neighbour in GetNeighbours(best.Row, best.Col, grid))
                    {
                        int neighbourRow = neighbour[0];
                        int neighbourCol = neighbour[1];

                        // This check isn't necessary for correctness, but it greatly
                        // improves performance.
                        if (visited[neighbourRow][neighbourCol])
                        {
                            continue;
                        }

                        // Otherwise, we need to create a Candidate object for the option
                        // of going to neighbor through the current cell.
                        int newDistance = best.DistanceSoFar + 1;
                        int totalEstimate = newDistance + Estimate(neighbourRow, neighbourCol, grid);
                        Candidate candidate =
                            new Candidate(neighbourRow, neighbourCol, newDistance, totalEstimate);
                        pq.Enqueue(candidate, candidate);
                    }
                }
                // The target was unreachable.
                return -1;
            }
            // Get the best case estimate of how much further it is to the bottom-right cell.
            private int Estimate(int row, int col, int[][] grid)
            {
                int remainingRows = grid.Length - row - 1;
                int remainingCols = grid[0].Length - col - 1;
                return Math.Max(remainingRows, remainingCols);
            }
            // Candidate represents a possible option for travelling to the cell
            // at (row, col).
            class Candidate
            {

                public int Row;
                public int Col;
                public int DistanceSoFar;
                public int TotalEstimate;

                public Candidate(int row, int col, int distanceSoFar, int totalEstimate)
                {
                    this.Row = row;
                    this.Col = col;
                    this.DistanceSoFar = distanceSoFar;
                    this.TotalEstimate = totalEstimate;
                }
            }

        }


        /* 133. Clone Graph
        https://leetcode.com/problems/clone-graph/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        public class CloneGraphSol
        {
            private Dictionary<Node, Node> visited = new Dictionary<Node, Node>();

            /* 
            Approach 1: Depth First Search

            Complexity Analysis
            •	Time Complexity: O(N+M), where N is a number of nodes (vertices) and M is a number of edges.
            •	Space Complexity: O(N). This space is occupied by the visited hash map and in addition to that, space would also be occupied by the recursion stack since we are adopting a recursive approach here. The space occupied by the recursion stack would be equal to O(H) where H is the height of the graph. Overall, the space complexity would be O(N).
             */
            public Node DFS(Node node)
            {
                if (node == null)
                {
                    return node;
                }

                // If the node was already visited before.
                // Return the clone from the visited dictionary.
                if (visited.ContainsKey(node))
                {
                    return visited[node];
                }

                // Create a clone for the given node.
                // Note that we don't have cloned neighbors as of now, hence [].
                Node cloneNode = new Node(node.val, new List<Node>());
                // The key is original node and value being the clone node.
                visited[node] = cloneNode;
                // Iterate through the neighbors to generate their clones
                // and prepare a list of cloned neighbors to be added to the cloned
                // node.
                foreach (Node neighbor in node.neighbors)
                {
                    cloneNode.neighbors.Add(DFS(neighbor));
                }

                return cloneNode;
            }
            // Definition for a Node.
            public class Node
            {
                public int val;
                public IList<Node> neighbors;
                public Node()
                {
                    val = 0;
                    neighbors = new List<Node>();
                }
                public Node(int _val)
                {
                    val = _val;
                    neighbors = new List<Node>();
                }
                public Node(int _val, List<Node> _neighbors)
                {
                    val = _val;
                    neighbors = _neighbors;
                }
            }
            /*             Approach 2: Breadth First Search
            Complexity Analysis
            •	Time Complexity : O(N+M), where N is a number of nodes (vertices) and M is a number of edges.
            •	Space Complexity : O(N). This space is occupied by the visited dictionary and in addition to that, space would also be occupied by the queue since we are adopting the BFS approach here. The space occupied by the queue would be equal to O(W) where W is the width of the graph. Overall, the space complexity would be O(N).

             */
            public Node BFS(Node node)
            {
                if (node == null)
                {
                    return node;
                }

                // Hash map to save the visited node and it's respective clone
                // as key and value respectively. This helps to avoid cycles.
                Dictionary<Node, Node> visited = new Dictionary<Node, Node>();
                // Put the first node in the queue
                Queue<Node> queue = new Queue<Node>();
                queue.Enqueue(node);
                // Clone the node and put it in the visited dictionary.
                visited[node] = new Node(node.val, new List<Node>());
                // Start BFS traversal
                while (queue.Count > 0)
                {
                    // Pop a node say "n" from the from the front of the queue.
                    Node n = queue.Dequeue();
                    // Iterate through all the neighbors of the node "n"
                    foreach (Node neighbor in n.neighbors)
                    {
                        if (!visited.ContainsKey(neighbor))
                        {
                            // Clone the neighbor and put in the visited, if not present
                            // already
                            visited[neighbor] =
                                new Node(neighbor.val, new List<Node>());
                            // Add the newly encountered node to the queue.
                            queue.Enqueue(neighbor);
                        }

                        // Add the clone of the neighbor to the neighbors of the clone
                        // node "n".
                        visited[n].neighbors.Add(visited[neighbor]);
                    }
                }

                // Return the clone of the node from visited.
                return visited[node];
            }

        }

        /* 498. Diagonal Traverse
        https://leetcode.com/problems/diagonal-traverse/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class FindDiagonalOrderSol
        {
            /* Approach 1: Diagonal Iteration and Reversal
Complexity Analysis
•	Time Complexity: O(N⋅M) considering the array has N rows and M columns. An important thing to remember is that for all the odd numbered diagonals, we will be processing the elements twice since we have to reverse the elements before adding to the result array. Additionally, to save space, we have to clear the intermediate array before we process a new diagonal. That operation also takes O(K) where K is the size of that array. So, we will be processing all the elements of the array at least twice. But, as far as the asymptotic complexity is concerned, it remains the same.
•	Space Complexity: O(min(N,M)) since the extra space is occupied by the intermediate arrays we use for storing diagonal elements and the maximum it can occupy is the equal to the minimum of N and M. Remember, the diagonal can only extend till one of its indices goes out of scope.

             */
            public int[] DiagonalIterationAndReversal(int[][] matrix)
            {

                // Check for empty matrices
                if (matrix == null || matrix.Length == 0)
                {
                    return new int[0];
                }

                // Variables to track the size of the matrix
                int N = matrix.Length;
                int M = matrix[0].Length;

                // The two arrays as explained in the algorithm
                int[] result = new int[N * M];
                int k = 0;
                List<int> intermediate = new();

                // We have to go over all the elements in the first
                // row and the last column to cover all possible diagonals
                for (int d = 0; d < N + M - 1; d++)
                {

                    // Clear the intermediate array every time we start
                    // to process another diagonal
                    intermediate.Clear();

                    // We need to figure out the "head" of this diagonal
                    // The elements in the first row and the last column
                    // are the respective heads.
                    int r = d < M ? 0 : d - M + 1;
                    int c = d < M ? d : M - 1;

                    // Iterate until one of the indices goes out of scope
                    // Take note of the index math to go down the diagonal
                    while (r < N && c > -1)
                    {

                        intermediate.Add(matrix[r][c]);
                        ++r;
                        --c;
                    }

                    // Reverse even numbered diagonals. The
                    // article says we have to reverse odd 
                    // numbered articles but here, the numbering
                    // is starting from 0 :P
                    if (d % 2 == 0)
                    {
                        intermediate.Reverse();
                    }

                    for (int i = 0; i < intermediate.Count; i++)
                    {
                        result[k++] = intermediate[i];
                    }
                }
                return result;
            }/* 
            Approach 2: Simulation
            Complexity Analysis
•	Time Complexity: O(N⋅M) since we process each element of the matrix exactly once.
•	Space Complexity: O(1) since we don't make use of any additional data structure. Note that the space occupied by the output array doesn't count towards the space complexity since that is a requirement of the problem itself. Space complexity comprises any additional space that we may have used to get to build the final array. For the previous solution, it was the intermediate arrays. In this solution, we don't have any additional space apart from a couple of variables.

             */
            public int[] UsingSimulation(int[][] matrix)
            {

                // Check for empty matrices
                if (matrix == null || matrix.Length == 0)
                {
                    return new int[0];
                }

                // Variables to track the size of the matrix
                int N = matrix.Length;
                int M = matrix[0].Length;

                // Incides that will help us progress through 
                // the matrix, one element at a time.
                int row = 0, column = 0;

                // As explained in the article, this is the variable
                // that helps us keep track of what direction we are
                // processing the current diaonal
                int direction = 1;

                // The final result array
                int[] result = new int[N * M];
                int r = 0;

                // The uber while loop which will help us iterate over all
                // the elements in the array.
                while (row < N && column < M)
                {

                    // First and foremost, add the current element to 
                    // the result matrix. 
                    result[r++] = matrix[row][column];

                    // Move along in the current diagonal depending upon
                    // the current direction.[i, j] -> [i - 1, j + 1] if 
                    // going up and [i, j] -> [i + 1][j - 1] if going down.
                    int new_row = row + (direction == 1 ? -1 : 1);
                    int new_column = column + (direction == 1 ? 1 : -1);

                    // Checking if the next element in the diagonal is within the
                    // bounds of the matrix or not. If it's not within the bounds,
                    // we have to find the next head. 
                    if (new_row < 0 || new_row == N || new_column < 0 || new_column == M)
                    {

                        // If the current diagonal was going in the upwards
                        // direction.
                        if (direction == 1)
                        {

                            // For an upwards going diagonal having [i, j] as its tail
                            // If [i, j + 1] is within bounds, then it becomes
                            // the next head. Otherwise, the element directly below
                            // i.e. the element [i + 1, j] becomes the next head
                            row += (column == M - 1 ? 1 : 0);
                            column += (column < M - 1 ? 1 : 0);

                        }
                        else
                        {

                            // For a downwards going diagonal having [i, j] as its tail
                            // if [i + 1, j] is within bounds, then it becomes
                            // the next head. Otherwise, the element directly below
                            // i.e. the element [i, j + 1] becomes the next head
                            column += (row == N - 1 ? 1 : 0);
                            row += (row < N - 1 ? 1 : 0);
                        }

                        // Flip the direction
                        direction = 1 - direction;

                    }
                    else
                    {

                        row = new_row;
                        column = new_column;
                    }
                }
                return result;
            }
        }

        /* 1424. Diagonal Traverse II
        https://leetcode.com/problems/diagonal-traverse-ii/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class FindDiagonalOrderIISol
        {
            /* Approach 1: Group Elements by the Sum of Row and Column Indices
Complexity Analysis
Given n as the number of integers in grid,
•	Time complexity: O(n)
We iterate over each of the n integers to populate groups, then we iterate over them again to populate ans.
•	Space complexity: O(n)
The values of groups are lists that together will store exactly n integers, thus using O(n) space.

             */
            public int[] GroupElementsBySumOfRowAndColumnIndices(List<List<int>> nums)
            {
                Dictionary<int, List<int>> groups = new();
                int n = 0;
                for (int row = nums.Count - 1; row >= 0; row--)
                {
                    for (int col = 0; col < nums[row].Count; col++)
                    {
                        int diagonal = row + col;
                        if (!groups.ContainsKey(diagonal))
                        {
                            groups[diagonal] = new List<int>();
                        }

                        groups[diagonal].Add(nums[row][col]);
                        n++;
                    }
                }

                int[] ans = new int[n];
                int i = 0;
                int curr = 0;

                while (groups.ContainsKey(curr))
                {
                    foreach (int num in groups[curr])
                    {
                        ans[i] = num;
                        i++;
                    }

                    curr++;
                }

                return ans;
            }
            /* Approach 2: Breadth First Search
            Complexity Analysis
Given n as the number of integers in grid,
•	Time complexity: O(n)
During the BFS, we visit each square once, performing O(1) work at each iteration.
•	Space complexity: O(n)
The extra space we use is for queue. The largest size queue will be is proportional to the size of the largest diagonal.
Let's say you had a diagonal with a size of k starting from the bottom left of the input and going to the top right. What are the fewest squares possible that could support such a diagonal existing? The first square in the diagonal can be the only square in its row. The second square in the diagonal needs one square to its left. The third square in the diagonal needs two squares to its left, and so on.
 

As you can see in the above image, the green diagonal requires many squares to its left to support its existence. In fact, we can notice that if we extended the image to a square, we would have a grid of size k∗k. That means to support a diagonal of size k, we require k^2/2=O(k^2) squares.
The conclusion is that a grid of size O(k^2) can only support a diagonal of size k. In our problem, we defined the input grid to have a size of n. Thus, the largest diagonal it could support would be O(n).

             */
            public int[] BFS(List<List<int>> nums)
            {
                Queue<Tuple<int, int>> queue = new();
                queue.Enqueue(new Tuple<int, int>(0, 0));
                List<int> ans = new();

                while (queue.Count > 0)
                {
                    Tuple<int, int> p = queue.Dequeue();
                    int row = p.Item1;
                    int col = p.Item2;
                    ans.Add(nums[row][col]);

                    if (col == 0 && row + 1 < nums.Count)
                    {
                        queue.Enqueue(new Tuple<int, int>(row + 1, col));
                    }

                    if (col + 1 < nums[row].Count)
                    {
                        queue.Enqueue(new Tuple<int, int>(row, col + 1));
                    }
                }

                // Java needs conversion
                int[] result = new int[ans.Count];
                int i = 0;
                foreach (int num in ans)
                {
                    result[i] = num;
                    i++;
                }

                return result;
            }
        }

        /* 304. Range Sum Query 2D - Immutable
        https://leetcode.com/problems/range-sum-query-2d-immutable/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class NumMatrixSol
        {
            /* Approach 1: Brute Force
            Complexity Analysis
            •	Time complexity: O(mn) time per query.
            Assume that m and n represents the number of rows and columns respectively, each sumRegion query can go through at most m×n elements.
            •	Space complexity: O(1). Note that data is a reference to matrix and is not a copy of it.

             */
            class NaiveSol
            {
                private int[][] data;

                public NaiveSol(int[][] matrix)
                {
                    data = matrix;
                }

                public int SumRegion(int row1, int col1, int row2, int col2)
                {
                    int sum = 0;
                    for (int r = row1; r <= row2; r++)
                    {
                        for (int c = col1; c <= col2; c++)
                        {
                            sum += data[r][c];
                        }
                    }
                    return sum;
                }
            }
            /* Approach 2: Caching [Memory Limit Exceeded]
            Complexity Analysis
•	Time complexity: O(1) time per query, O(m^2n^2) time pre-computation.
Each sumRegion query takes O(1) time as the hash table lookup's time complexity is constant. The pre-computation will take O(m^2n^2) time as there are a total of m^2×n^2 possibilities need to be cached.
•	Space complexity: O(m^2n^2).
Since there are mn different possibilities for both top left and bottom right points of the rectangular region, the extra space required is O(m^2n^2).

             */
            /*              Approach 3: Caching Rows
            Complexity Analysis
            •	Time complexity: O(m) time per query, O(mn) time pre-computation.
            The pre-computation in the constructor takes O(mn) time. The sumRegion query takes O(m) time.
            •	Space complexity: O(mn).
            The algorithm uses O(mn) space to store the cumulative sum of all rows.

             */
            class CachingRowsSol
            {
                private int[][] dp;

                public CachingRowsSol(int[][] matrix)
                {
                    if (matrix.Length == 0 || matrix[0].Length == 0) return;
                    dp = new int[matrix.Length][];
                    for (int r = 0; r < matrix.Length; r++)
                    {
                        for (int c = 0; c < matrix[0].Length; c++)
                        {
                            dp[r][c + 1] = dp[r][c] + matrix[r][c];
                        }
                    }
                }

                public int SumRegion(int row1, int col1, int row2, int col2)
                {
                    int sum = 0;
                    for (int row = row1; row <= row2; row++)
                    {
                        sum += dp[row][col2 + 1] - dp[row][col1];
                    }
                    return sum;
                }

                /*                 Approach 4: Caching Smarter
                Complexity Analysis
                •	Time complexity: O(1) time per query, O(mn) time pre-computation.
                The pre-computation in the constructor takes O(mn) time. Each sumRegion query takes O(1) time.
                •	Space complexity: O(mn).
                The algorithm uses O(mn) space to store the cumulative region sum.

                 */
                class CachingSmartSol
                {
                    private int[][] dp;

                    public CachingSmartSol(int[][] matrix)
                    {
                        if (matrix.Length == 0 || matrix[0].Length == 0) return;
                        dp = new int[matrix.Length + 1][];
                        for (int r = 0; r < matrix.Length; r++)
                        {
                            for (int c = 0; c < matrix[0].Length; c++)
                            {
                                dp[r + 1][c + 1] = dp[r + 1][c] + dp[r][c + 1] + matrix[r][c] - dp[r][c];
                            }
                        }
                    }

                    public int SumRegion(int row1, int col1, int row2, int col2)
                    {
                        return dp[row2 + 1][col2 + 1] - dp[row1][col2 + 1] - dp[row2 + 1][col1] + dp[row1][col1];
                    }
                }

            }
        }

        /* 378. Kth Smallest Element in a Sorted Matrix
        https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class KthSmallestElementInASortedMatrixSol
        {
            /* Approach 1: Min-Heap approach	
            Complexity Analysis
            •	Time Complexity: let X=min(K,N);X+Klog(X)
            o	Well the heap construction takes O(X) time.
            o	After that, we perform K iterations and each iteration has two operations. We extract the minimum element from a heap containing X elements. Then we add a new element to this heap. Both the operations will take O(log(X)) time.
            o	Thus, the total time complexity for this algorithm comes down to be O(X+Klog(X)) where X is min(K,N).
            •	Space Complexity: O(X) which is occupied by the heap.

             */
            public int UsingMinHeap(int[][] matrix, int k)
            {
                int N = matrix.Length;

                PriorityQueue<MyHeapNode, MyHeapNode> minHeap =
                    new PriorityQueue<MyHeapNode, MyHeapNode>(new MyHeapComparator());

                // Preparing our min-heap
                for (int r = 0; r < Math.Min(N, k); r++)
                {
                    // We add triplets of information for each cell
                    minHeap.Enqueue(new MyHeapNode(matrix[r][0], r, 0), new MyHeapNode(matrix[r][0], r, 0));
                }

                MyHeapNode element = minHeap.Peek();
                while (k-- > 0)
                {
                    // Extract-Min
                    element = minHeap.Dequeue();
                    int r = element.GetRow(), c = element.GetColumn();

                    // If we have any new elements in the current row, add them
                    if (c < N - 1)
                    {
                        minHeap.Enqueue(new MyHeapNode(matrix[r][c + 1], r, c + 1), new MyHeapNode(matrix[r][c + 1], r, c + 1));
                    }
                }

                return element.GetValue();
            }
            class MyHeapNode
            {
                public int Row;
                public int Column;
                public int Value;

                public MyHeapNode(int v, int r, int c)
                {
                    this.Value = v;
                    this.Row = r;
                    this.Column = c;
                }

                public int GetValue()
                {
                    return this.Value;
                }

                public int GetRow()
                {
                    return this.Row;
                }

                public int GetColumn()
                {
                    return this.Column;
                }
            }

            class MyHeapComparator : IComparer<MyHeapNode>
            {
                public int Compare(MyHeapNode x, MyHeapNode y)
                {
                    return x.Value - y.Value;
                }
            }

            /* Approach 2: Binary Search	
            Complexity Analysis
            •	Time Complexity: O(N×log(Max−Min))
            o	Let's think about the time complexity in terms of the normal binary search algorithm. For a one-dimensional binary search over an array with N elements, the complexity comes out to be O(log(N)).
            o	For our scenario, we are kind of defining our binary search space in terms of the minimum and the maximum numbers in the array. Going by this idea, the complexity for our binary search should be O(log(Max−Min)) where Max is the maximum element in the array and likewise, Min is the minimum element.
            o	However, we update our search space after each iteration. So, even if the maximum element is super large as compared to the remaining elements in the matrix, we will bring down the search space considerably in the next iterations. But, going purely by the extremes for our search space, the complexity of our binary search in search of Kth smallest element will be O(log(Max−Min)).
            o	In each iteration of our binary search approach, we iterate over the matrix trying to determine the size of the left-half as explained before. That takes O(N).
            o	Thus, the overall time complexity is O(N×log(Max−Min))
            •	Space Complexity: O(1) since we don't use any additional space for performing our binary search.

             */
            public int UsingBinarySearch(int[][] matrix, int k)
            {

                int n = matrix.Length;
                int start = matrix[0][0], end = matrix[n - 1][n - 1];
                while (start < end)
                {

                    int mid = start + (end - start) / 2;
                    // first number is the smallest and the second number is the largest
                    int[] smallLargePair = { matrix[0][0], matrix[n - 1][n - 1] };

                    int count = this.CountLessEqual(matrix, mid, smallLargePair);

                    if (count == k) return smallLargePair[0];

                    if (count < k) start = smallLargePair[1]; // search higher
                    else end = smallLargePair[0]; // search lower
                }
                return start;
            }

            private int CountLessEqual(int[][] matrix, int mid, int[] smallLargePair)
            {

                int count = 0;
                int n = matrix.Length, row = n - 1, col = 0;

                while (row >= 0 && col < n)
                {

                    if (matrix[row][col] > mid)
                    {

                        // as matrix[row][col] is bigger than the mid, let's keep track of the
                        // smallest number greater than the mid
                        smallLargePair[1] = Math.Min(smallLargePair[1], matrix[row][col]);
                        row--;

                    }
                    else
                    {

                        // as matrix[row][col] is less than or equal to the mid, let's keep track of the
                        // biggest number less than or equal to the mid
                        smallLargePair[0] = Math.Max(smallLargePair[0], matrix[row][col]);
                        count += row + 1;
                        col++;
                    }
                }

                return count;
            }
        }


        /* 311. Sparse Matrix Multiplication
         https://leetcode.com/problems/sparse-matrix-multiplication/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
          */
        class MultiplySparseMatrixSol
        {

            /* Approach 1: Naive Iteration
            Complexity Analysis
Let m and k represent the number of rows and columns in mat1, respectively. Likewise, let k and n represent the number of rows and columns in mat2, respectively.
•	Time complexity: O(m⋅k⋅n).
o	We iterate over all m⋅k elements of the matrix mat1.
o	For each element of matrix mat1, we iterate over all n columns of the matrix mat2.
o	Thus, it leads to a time complexity of m⋅k⋅n.
•	Space complexity: O(1).
We use a matrix ans of size m×n to output the multiplication result which is not included in auxiliary space.

             */
            public int[][] NaiveIteration(int[][] mat1, int[][] mat2)
            {
                int n = mat1.Length;
                int k = mat1[0].Length;
                int m = mat2[0].Length;

                // Product matrix will have 'n x m' size.
                int[][] ans = new int[n][];

                for (int rowIndex = 0; rowIndex < n; ++rowIndex)
                {
                    for (int elementIndex = 0; elementIndex < k; ++elementIndex)
                    {
                        // If current element of mat1 is non-zero then iterate over all columns of mat2.
                        if (mat1[rowIndex][elementIndex] != 0)
                        {
                            for (int colIndex = 0; colIndex < m; ++colIndex)
                            {
                                ans[rowIndex][colIndex] += mat1[rowIndex][elementIndex] * mat2[elementIndex][colIndex];
                            }
                        }
                    }
                }

                return ans;
            }

            /*             Approach 2: List of Lists
            Complexity Analysis
            Let m and k represent the number of rows and columns in mat1, respectively. Likewise, let k and n represent the number of rows and columns in mat2, respectively.
            •	Time complexity: O(m⋅k⋅n).
            o	We iterate over all non-zero elements of the matrix mat1. And for each non-zero element, we iterate over one row of the matrix mat2.
            o	In the worst-case, mat1 can have m⋅k elements and mat2 can have n elements in each row.
            o	Thus, it leads to the time complexity of m⋅k⋅n.
            •	Space complexity: O(m⋅k+k⋅n).
            o	We use a data structure (an array of arrays) to efficiently store elements of both matrices.
            In the worst-case, we will store all m⋅k elements of mat1 and k⋅n elements of mat2 in our data structures.
            o	We use a matrix ans of size m×n to output the multiplication result which is not included in auxiliary space.

             */
            public int[][] UsingListOfLists(int[][] mat1, int[][] mat2)
            {
                int m = mat1.Length;
                int k = mat1[0].Length;
                int n = mat2[0].Length;

                // Store the non-zero values of each matrix.
                List<List<KeyValuePair<int, int>>> A = CompressMatrix(mat1);
                List<List<KeyValuePair<int, int>>> B = CompressMatrix(mat2);

                int[][] ans = new int[m][];
                for (int i = 0; i < m; i++)
                {
                    ans[i] = new int[n];
                }

                for (int mat1Row = 0; mat1Row < m; ++mat1Row)
                {
                    // Iterate on all current 'row' non-zero elements of mat1.
                    foreach (var mat1Element in A[mat1Row])
                    {
                        int element1 = mat1Element.Key;
                        int mat1Col = mat1Element.Value;

                        // Multiply and add all non-zero elements of mat2
                        // where the row is equal to col of current element of mat1.
                        foreach (var mat2Element in B[mat1Col])
                        {
                            int element2 = mat2Element.Key;
                            int mat2Col = mat2Element.Value;
                            ans[mat1Row][mat2Col] += element1 * element2;
                        }
                    }
                }

                return ans;
            }
            private List<List<KeyValuePair<int, int>>> CompressMatrix(int[][] matrix)
            {
                int rows = matrix.Length;
                int cols = matrix[0].Length;

                List<List<KeyValuePair<int, int>>> compressedMatrix = new List<List<KeyValuePair<int, int>>>();

                for (int row = 0; row < rows; ++row)
                {
                    List<KeyValuePair<int, int>> currRow = new List<KeyValuePair<int, int>>();
                    for (int col = 0; col < cols; ++col)
                    {
                        if (matrix[row][col] != 0)
                        {
                            currRow.Add(new KeyValuePair<int, int>(matrix[row][col], col));
                        }
                    }
                    compressedMatrix.Add(currRow);
                }
                return compressedMatrix;
            }

            /* Approach 3: Yale Format
            Complexity Analysis
Let m and k represent the number of rows and columns in mat1, respectively. Likewise, let k and n represent the number of rows and columns in mat2, respectively.
•	Time complexity: O(m⋅n⋅k).
o	We iterate over all m⋅n elements of ans matrix.
o	For each element we iterate over k row elements in mat1, and k column elements in mat2. In the worst-case scenario, none of the elements in either matrix is zero.
o	Thus, the time complexity is O(m⋅n⋅k).
•	Space complexity: O(m⋅k+k⋅n).
o	We use a data structure to efficiently store the non-zero elements of both matrices.
In the worst-case scenario we will store all m⋅k elements of mat1 and all k⋅n elements of mat2 in one dimensional arrays.
o	We use a matrix ans of size m×n to output the multiplication result which is not included in auxiliary space.

             */
            public int[][] UsingYaleFormat(int[][] mat1, int[][] mat2)
            {
                SparseMatrix A = new SparseMatrix(mat1);
                SparseMatrix B = new SparseMatrix(mat2, true);

                int[][] ans = new int[mat1.Length][];
                for (int i = 0; i < ans.Length; i++)
                {
                    ans[i] = new int[mat2[0].Length];
                }

                for (int row = 0; row < ans.Length; ++row)
                {
                    for (int col = 0; col < ans[0].Length; ++col)
                    {

                        // Row element range indices
                        int matrixOneRowStart = A.rowIndex[row];
                        int matrixOneRowEnd = A.rowIndex[row + 1];

                        // Column element range indices
                        int matrixTwoColStart = B.colIndex[col];
                        int matrixTwoColEnd = B.colIndex[col + 1];

                        // Iterate over both row and column.
                        while (matrixOneRowStart < matrixOneRowEnd && matrixTwoColStart < matrixTwoColEnd)
                        {
                            if (A.colIndex[matrixOneRowStart] < B.rowIndex[matrixTwoColStart])
                            {
                                matrixOneRowStart++;
                            }
                            else if (A.colIndex[matrixOneRowStart] > B.rowIndex[matrixTwoColStart])
                            {
                                matrixTwoColStart++;
                            }
                            else
                            {
                                // Row index and col index are same so we can multiply these elements.
                                ans[row][col] += A.values[matrixOneRowStart] * B.values[matrixTwoColStart];
                                matrixOneRowStart++;
                                matrixTwoColStart++;
                            }
                        }
                    }
                }

                return ans;
            }
            class SparseMatrix
            {
                public int cols = 0, rows = 0;
                public List<int> values = new List<int>();
                public List<int> colIndex = new List<int>();
                public List<int> rowIndex = new List<int>();

                // Compressed Sparse Row
                public SparseMatrix(int[][] matrix)
                {
                    rows = matrix.Length;
                    cols = matrix[0].Length;
                    rowIndex.Add(0);

                    for (int row = 0; row < rows; ++row)
                    {
                        for (int col = 0; col < cols; ++col)
                        {
                            if (matrix[row][col] != 0)
                            {
                                values.Add(matrix[row][col]);
                                colIndex.Add(col);
                            }
                        }
                        rowIndex.Add(values.Count);
                    }
                }

                // Compressed Sparse Column
                public SparseMatrix(int[][] matrix, bool colWise)
                {
                    rows = matrix.Length;
                    cols = matrix[0].Length;
                    colIndex.Add(0);

                    for (int col = 0; col < cols; ++col)
                    {
                        for (int row = 0; row < rows; ++row)
                        {
                            if (matrix[row][col] != 0)
                            {
                                values.Add(matrix[row][col]);
                                rowIndex.Add(row);
                            }
                        }
                        colIndex.Add(values.Count);
                    }
                }
            };

        }
        /* 
        1778. Shortest Path in a Hidden Grid
        https://leetcode.com/problems/shortest-path-in-a-hidden-grid/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
        https://algo.monster/liteproblems/1778
         */

        class FindShortestPathInHiddenGridSol
        {
            // Delta arrays to allow movement in grid: up, right, down, left
            private static readonly char[] DIRECTIONS = { 'U', 'R', 'D', 'L' };
            // Opposite directions for backtracking: down, left, up, right
            private static readonly char[] OPPOSITE_DIRECTIONS = { 'D', 'L', 'U', 'R' };
            // Row and column increments for corresponding directions
            private static readonly int[] DELTAS = { -1, 0, 1, 0, -1 };
            // Grid dimension for marking visited coordinates
            private static readonly int GRID_SIZE = 1010;
            // Set to keep track of visited coordinates
            private HashSet<int> visited;
            // Coordinate pair for the target location
            private int[] targetPosition;
            /* 
            Time and Space Complexity
            The given Python code represents an algorithm to find the shortest path from a starting point to a target in a grid-like structure, using depth-first search (DFS) to explore the grid and Breadth-first search (BFS) to find the shortest path. Analyzing the code segment provided:
            Time Complexity:
            The DFS part of the code explores each cell at most once. For each cell, the DFS function performs a constant amount of work. Therefore, if there are N cells that can be visited, the time complexity for the DFS portion is O(N).
            Subsequently, the BFS portion of the code also visits each cell at most once. Similar to DFS, since BFS performs a constant amount of work for each cell, the time complexity for the BFS portion is also O(N).
            Hence, considering both DFS and BFS combined, the overall time complexity of the code is O(N), where N is the number of cells that can be visited.
            Space Complexity:
            The space complexity is influenced by both the recursive call stack for DFS and the storage of cells in the set s, and the queue q.
            For DFS:
            •	The call stack can grow up to O(N) in the worst case when the grid forms a deep structure with only one path and no branching.
            •	The set s stores all visited cells. In the worst case, this could be all N cells in the grid, which also contributes to O(N).
            For BFS:
            •	The queue q can store at most N cells during the layer-by-layer exploration.
            Combining DFS and BFS, the space required is the maximum space used by either algorithm since they are not run simultaneously, which leads to a combined space complexity of O(N).
            Thus, the overall space complexity of the code is O(N), where N is the number of cells that can be visited.
             */
            // Method to find the shortest path using BFS and DFS traversals in a grid
            public int FindShortestPath(GridMaster master)
            {
                targetPosition = null;
                visited = new HashSet<int>();
                // Initial position as origin (0, 0)
                visited.Add(0);
                // Use DFS to traverse through the grid and mark visited paths
                Dfs(0, 0, master);
                if (targetPosition == null)
                {
                    // Target not found, return -1
                    return -1;
                }
                // Remove origin from visited as we will use BFS from origin to find the shortest path
                visited.Remove(0);
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { 0, 0 });
                // Distance counter
                int steps = -1;
                // BFS to find the shortest path
                while (queue.Count > 0)
                {
                    ++steps;
                    for (int size = queue.Count; size > 0; --size)
                    {
                        int[] position = queue.Dequeue();
                        int row = position[0], col = position[1];
                        // Check if current position is the target
                        if (targetPosition[0] == row && targetPosition[1] == col)
                        {
                            return steps;
                        }
                        // Explore 4-directionally adjacent cells
                        for (int k = 0; k < 4; ++k)
                        {
                            int newRow = row + DELTAS[k], newCol = col + DELTAS[k + 1];
                            int newPosKey = newRow * GRID_SIZE + newCol;
                            if (visited.Contains(newPosKey))
                            {
                                visited.Remove(newPosKey);
                                queue.Enqueue(new int[] { newRow, newCol });
                            }
                        }
                    }
                }
                return -1;
            }

            // DFS to explore the grid and mark paths that have been visited
            private void Dfs(int row, int col, GridMaster master)
            {
                if (master.IsTarget())
                {
                    targetPosition = new int[] { row, col };
                }
                for (int k = 0; k < 4; ++k)
                {
                    char d = DIRECTIONS[k], oppositeD = OPPOSITE_DIRECTIONS[k];
                    int newRow = row + DELTAS[k], newCol = col + DELTAS[k + 1];
                    int newPosKey = newRow * GRID_SIZE + newCol;
                    if (master.CanMove(d) && !visited.Contains(newPosKey))
                    {
                        visited.Add(newPosKey);
                        master.Move(d);
                        Dfs(newRow, newCol, master);
                        master.Move(oppositeD);
                    }
                }
            }
            internal class GridMaster
            {
                internal bool CanMove(char d)
                {
                    throw new NotImplementedException();
                }

                internal bool IsTarget()
                {
                    throw new NotImplementedException();
                }

                internal void Move(char d)
                {
                    throw new NotImplementedException();
                }
            }
        }
        /* 
        785. Is Graph Bipartite?
        https://leetcode.com/problems/is-graph-bipartite/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class IsGraphBipartiteSol
        {
            /*
            Approach #1: Coloring by Depth-First Search [Accepted]
                        Complexity Analysis
            •	Time Complexity: O(N+E), where N is the number of nodes in the graph, and E is the number of edges. We explore each node once when we transform it from uncolored to colored, traversing all its edges in the process.
            •	Space Complexity: O(N), the space used to store the color.
             */
            public bool UsingColoringByDFS(int[][] graph)
            {
                int n = graph.Length;
                int[] color = new int[n];
                Array.Fill(color, -1);

                for (int start = 0; start < n; ++start)
                {
                    if (color[start] == -1)
                    {
                        Stack<int> stack = new();
                        stack.Push(start);
                        color[start] = 0;

                        while (stack.Count > 0)
                        {
                            int node = stack.Pop();
                            foreach (int nei in graph[node])
                            {
                                if (color[nei] == -1)
                                {
                                    stack.Push(nei);
                                    color[nei] = color[node] ^ 1;
                                }
                                else if (color[nei] == color[node])
                                {
                                    return false;
                                }
                            }
                        }
                    }
                }

                return true;
            }
        }

        /* 1605. Find Valid Matrix Given Row and Column Sums
        https://leetcode.com/problems/find-valid-matrix-given-row-and-column-sums/description/
         */
        class RestoreMatrixSol
        {

            /* Approach 1: Greedy
            Complexity Analysis
            Here,N is the number size of the list rowSum and M is the size of the list colSum.
            •	Time complexity: O(N×M).
            Initializing the answer matrix origMatrix takes O(N×M) time. Also, we iterate over each of the N×M cells to find the values. Hence, the total time complexity is equal to O(N×M).
            •	Space complexity: O(N+M).
            The space required to store the answer is not considered part of the space complexity. Therefore, the space required for this approach is the two lists to store the current sum of rows and columns. Hence, the total space complexity is equal to O(N+M).

             */
            public int[][] UsingGreedy(int[] rowSum, int[] colSum)
            {
                int N = rowSum.Length;
                int M = colSum.Length;

                int[] currRowSum = new int[N];
                int[] currColSum = new int[M];

                int[][] origMatrix = new int[N][];
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < M; j++)
                    {
                        origMatrix[i][j] = Math.Min(
                            rowSum[i] - currRowSum[i],
                            colSum[j] - currColSum[j]
                        );

                        currRowSum[i] += origMatrix[i][j];
                        currColSum[j] += origMatrix[i][j];
                    }
                }
                return origMatrix;
            }

            /* Approach 2: Space Optimized Greedy
Complexity Analysis
Here, N is the number size of the list rowSum and M is the size of the list colSum.
•	Time complexity: O(N∗M).
Initializing the answer matrix origMatrix takes O(N×M) time. Also, we iterate over each of the N×M cells to find the values. Hence, the total time complexity is equal to O(N×M).
•	Space complexity: O(1).
The space required to store the answer is not considered part of the space complexity. We don't require any extra space other than the matrix to store the answer. Hence, the total space complexity is constant.

             */
            public int[][] UsingGreedyWithSpaceOptimal(int[] rowSum, int[] colSum)
            {
                int N = rowSum.Length;
                int M = colSum.Length;

                int[][] origMatrix = new int[N][];
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < M; j++)
                    {
                        origMatrix[i][j] = Math.Min(rowSum[i], colSum[j]);

                        rowSum[i] -= origMatrix[i][j];
                        colSum[j] -= origMatrix[i][j];
                    }
                }

                return origMatrix;
            }
            /* Approach 3: Time + Space Optimized Greedy
            Complexity Analysis
Here, N is the number size of the list rowSum and M is the size of the list colSum.
•	Time complexity: O(N×M).
Initializing the answer matrix origMatrix takes O(N×M) time. To store the values in the answer matrix we performed O(N+M) operations as we skipped either the row or column at each iteration. Hence, the total time complexity is equal to O(N×M).
•	Space complexity: O(1).
The space required to store the answer is not considered part of the space complexity. We don't require any extra space other than the matrix to store the answer. Hence, the total space complexity is constant.

             */
            public int[][] UsingGreedyWithTimeAndSpaceOptimal(int[] rowSum, int[] colSum)
            {
                int N = rowSum.Length;
                int M = colSum.Length;

                int[][] origMatrix = new int[N][];
                int i = 0, j = 0;

                while (i < N && j < M)
                {
                    origMatrix[i][j] = Math.Min(rowSum[i], colSum[j]);

                    rowSum[i] -= origMatrix[i][j];
                    colSum[j] -= origMatrix[i][j];

                    if (rowSum[i] == 0)
                    {
                        i++;
                    }
                    else
                    {
                        j++;
                    }
                }

                return origMatrix;
            }
        }

        /* 959. Regions Cut By Slashes
        https://leetcode.com/problems/regions-cut-by-slashes/description/
         */
        class RegionsCutBySlashesSol
        {

            /* Approach 1: Expanded Grid
            Complexity Analysis
            Let n be the height and width of the grid.
            •	Time complexity: O(n^2)
            The algorithm populates the expanded grid by iterating over the original grid, which takes O(n^2) time.
            In the worst case, the flood fill algorithm will visit every cell in the expanded grid once. The expanded grid is 3n×3n, resulting in O((3n) ^2)=O(9n^2)=O(n^2) operations.
            Thus, the overall time complexity of the algorithm is 2⋅O(n^2), which simplifies to O(n^2).
            •	Space complexity: O(n^2)
            The expanded grid has dimensions 3n×3n, which requires O(n^2) space.
            In the flood fill algorithm, the queue can store all 9n^2 cells of the expanded grid in the worst case. This results in a space complexity of O(9n^2)=O(n^2).
            Thus, the total time complexity of the algorithm is O(n^2)+O(n^2)=O(n^2).

             */
            // Directions for traversal: right, left, down, up
            private static readonly int[][] DIRECTIONS = {    new int[]{ 0, 1 },
        new int[]{ 0, -1 },new int[]{ 1, 0 },new int[]{ -1, 0 },    };

            public int UsingEpandedGrid(String[] grid)
            {
                int gridSize = grid.Length;
                // Create a 3x3 matrix for each cell in the original grid
                int[][] expandedGrid = new int[gridSize * 3][];

                // Populate the expanded grid based on the original grid
                // 1 represents a barrier in the expanded grid
                for (int i = 0; i < gridSize; i++)
                {
                    for (int j = 0; j < gridSize; j++)
                    {
                        int baseRow = i * 3;
                        int baseCol = j * 3;
                        // Check the character in the original grid
                        if (grid[i][j] == '\\')
                        {
                            // Mark diagonal for backslash
                            expandedGrid[baseRow][baseCol] = 1;
                            expandedGrid[baseRow + 1][baseCol + 1] = 1;
                            expandedGrid[baseRow + 2][baseCol + 2] = 1;
                        }
                        else if (grid[i][j] == '/')
                        {
                            // Mark diagonal for forward slash
                            expandedGrid[baseRow][baseCol + 2] = 1;
                            expandedGrid[baseRow + 1][baseCol + 1] = 1;
                            expandedGrid[baseRow + 2][baseCol] = 1;
                        }
                    }
                }

                int regionCount = 0;
                // Count regions using flood fill
                for (int i = 0; i < gridSize * 3; i++)
                {
                    for (int j = 0; j < gridSize * 3; j++)
                    {
                        // If we find an unvisited cell (0), it's a new region
                        if (expandedGrid[i][j] == 0)
                        {
                            // Fill that region
                            FloodFill(expandedGrid, i, j);
                            regionCount++;
                        }
                    }
                }
                return regionCount;
            }

            // Flood fill algorithm to mark all cells in a region
            private void FloodFill(int[][] expandedGrid, int row, int col)
            {
                Queue<int[]> queue = new();
                expandedGrid[row][col] = 1;
                queue.Enqueue(new int[] { row, col });

                while (queue.Count > 0)
                {
                    int[] currentCell = queue.Dequeue();
                    // Check all four directions from the current cell
                    foreach (int[] direction in DIRECTIONS)
                    {
                        int newRow = direction[0] + currentCell[0];
                        int newCol = direction[1] + currentCell[1];
                        // If the new cell is valid and unvisited, mark it and add to queue
                        if (IsValidCell(expandedGrid, newRow, newCol))
                        {
                            expandedGrid[newRow][newCol] = 1;
                            queue.Enqueue(new int[] { newRow, newCol });
                        }
                    }
                }
            }

            // Check if a cell is within bounds and unvisited
            private bool IsValidCell(int[][] expandedGrid, int row, int col)
            {
                int n = expandedGrid.Length;
                return (
                    row >= 0 &&
                    col >= 0 &&
                    row < n &&
                    col < n &&
                    expandedGrid[row][col] == 0
                );
            }
            /* Approach 2: Disjoint Set Union (Triangles)
            Complexity Analysis
            Let n be the height and width of the grid.
            •	Time complexity: O(n^2⋅α(n))
            Initializing the parentArray takes O(4⋅n^2) time.
            The main loop iterates over all n^2 cells in the grid. In each iteration, it calls the unionTriangles method which includes findPath operations. With path compression, the amortized time complexity of findPath is denoted as α(n), where α is the inverse Ackermann function. Thus, the time complexity of the loop comes out to be O(n^2⋅α(n)).
            Thus, the overall time complexity of the algorithm is O(4⋅n^2)+O(n^2⋅α(n))=O(n^2⋅α(n))
            •	Space complexity: O(n^2)
            The only additional data structure used by the algorithm is the parentArray, which takes O(n^2) space.
            The recursive find operation can have a call stack of size O(logn) in the worst case.
            Thus, the overall space complexity is O(n^2).

             */
            public int UsingDisjointSetUnionTraingles(String[] grid)
            {
                int gridSize = grid.Length;
                int totalTriangles = gridSize * gridSize * 4;
                int[] parentArray = new int[totalTriangles];
                Array.Fill(parentArray, -1);

                // Initially, each small triangle is a separate region
                int regionCount = totalTriangles;

                for (int row = 0; row < gridSize; row++)
                {
                    for (int col = 0; col < gridSize; col++)
                    {
                        // Connect with the cell above
                        if (row > 0)
                        {
                            regionCount -=
                            UnionTriangles(
                                parentArray,
                                GetTriangleIndex(gridSize, row - 1, col, 2),
                                GetTriangleIndex(gridSize, row, col, 0)
                            );
                        }
                        // Connect with the cell to the left
                        if (col > 0)
                        {
                            regionCount -=
                            UnionTriangles(
                                parentArray,
                                GetTriangleIndex(gridSize, row, col - 1, 1),
                                GetTriangleIndex(gridSize, row, col, 3)
                            );
                        }

                        // If not '/', connect triangles 0-1 and 2-3
                        if (grid[row][col] != '/')
                        {
                            regionCount -=
                            UnionTriangles(
                                parentArray,
                                GetTriangleIndex(gridSize, row, col, 0),
                                GetTriangleIndex(gridSize, row, col, 1)
                            );
                            regionCount -=
                            UnionTriangles(
                                parentArray,
                                GetTriangleIndex(gridSize, row, col, 2),
                                GetTriangleIndex(gridSize, row, col, 3)
                            );
                        }

                        // If not '\', connect triangles 0-3 and 1-2
                        if (grid[row][col] != '\\')
                        {
                            regionCount -=
                            UnionTriangles(
                                parentArray,
                                GetTriangleIndex(gridSize, row, col, 0),
                                GetTriangleIndex(gridSize, row, col, 3)
                            );
                            regionCount -=
                            UnionTriangles(
                                parentArray,
                                GetTriangleIndex(gridSize, row, col, 2),
                                GetTriangleIndex(gridSize, row, col, 1)
                            );
                        }
                    }
                }
                return regionCount;
            }

            // Calculate the index of a triangle in the flattened array
            // Each cell is divided into 4 triangles, numbered 0 to 3 clockwise from the top
            private int GetTriangleIndex(
                int gridSize,
                int row,
                int col,
                int triangleNum
            )
            {
                return (gridSize * row + col) * 4 + triangleNum;
            }

            // Union two triangles and return 1 if they were not already connected, 0 otherwise
            private int UnionTriangles(int[] parentArray, int x, int y)
            {
                int parentX = FindParent(parentArray, x);
                int parentY = FindParent(parentArray, y);

                if (parentX != parentY)
                {
                    parentArray[parentX] = parentY;
                    return 1; // Regions were merged, so count decreases by 1
                }

                return 0; // Regions were already connected
            }

            // Find the parent (root) of a set
            private int FindParent(int[] parentArray, int x)
            {
                if (parentArray[x] == -1) return x;

                return parentArray[x] = FindParent(parentArray, parentArray[x]);
            }

            /* Approach 3: Disjoint Set Union (Graph)
            Complexity Analysis
Let n be the height and width of the grid.
•	Time complexity: O(n^2⋅α(n^2))
Filling the parent array requires O((n+1)⋅(n+1)) time, which can be simplified to O(n^2). Connecting the border points requires another O(n^2) time.
As the algorithm iterates over the grid, it potentially performs two union operations for each cell. The time complexity of a single find/union operation is O(α(n^2)), where α is the inverse Ackermann function. We perform at most O(n^2) union operations, making the complexity of this part O(n^2⋅α(n^2)).
Thus, the overall time complexity is 2⋅O(n^2)+O(n^2⋅2α(n^2))=O(n^2⋅α(n^2))
•	Space complexity: O(n^2)
The algorithm creates an array of size (n+1)2, which is O(n^2).
The recursive call stack for find operation is O(logn) in the worst case.
Thus, the total time complexity of the algorithm is O(n^2).

             */
            public int UsingDisjointSetUnionGraph(String[] grid)
            {
                int gridSize = grid.Length;
                int pointsPerSide = gridSize + 1;
                int totalPoints = pointsPerSide * pointsPerSide;

                // Initialize disjoint set data structure
                int[] parentArray = new int[totalPoints];
                Array.Fill(parentArray, -1);

                // Connect border points
                for (int i = 0; i < pointsPerSide; i++)
                {
                    for (int j = 0; j < pointsPerSide; j++)
                    {
                        if (
                            i == 0 ||
                            j == 0 ||
                            i == pointsPerSide - 1 ||
                            j == pointsPerSide - 1
                        )
                        {
                            int point = i * pointsPerSide + j;
                            parentArray[point] = 0;
                        }
                    }
                }

                // Set the parent of the top-left corner to itself
                parentArray[0] = -1;
                int regionCount = 1; // Start with one region (the border)

                // Process each cell in the grid
                for (int i = 0; i < gridSize; i++)
                {
                    for (int j = 0; j < gridSize; j++)
                    {
                        // Treat each slash as an edge connecting two points
                        if (grid[i][j] == '/')
                        {
                            int topRight = i * pointsPerSide + (j + 1);
                            int bottomLeft = (i + 1) * pointsPerSide + j;
                            regionCount += Union(parentArray, topRight, bottomLeft);
                        }
                        else if (grid[i][j] == '\\')
                        {
                            int topLeft = i * pointsPerSide + j;
                            int bottomRight = (i + 1) * pointsPerSide + (j + 1);
                            regionCount += Union(parentArray, topLeft, bottomRight);
                        }
                    }
                }

                return regionCount;
            }

            // Find the parent of a set
            private int Find(int[] parentArray, int node)
            {
                if (parentArray[node] == -1) return node;

                return parentArray[node] = Find(parentArray, parentArray[node]);
            }

            // Union two sets and return 1 if a new region is formed, 0 otherwise
            private int Union(int[] parentArray, int node1, int node2)
            {
                int parent1 = Find(parentArray, node1);
                int parent2 = Find(parentArray, node2);

                if (parent1 == parent2)
                {
                    return 1; // Nodes are already in the same set, new region formed
                }

                parentArray[parent2] = parent1; // Union the sets
                return 0; // No new region formed
            }

        }


        /* 840. Magic Squares In Grid
        https://leetcode.com/problems/magic-squares-in-grid/description/
         */
        class NumMagicSquaresInGridSol
        {
            /* Approach 1: Manual Scan
Time Complexity
Let M and N be the number of rows and columns of grid, respectively.
•	Time Complexity: O(M⋅N)
The number of 3 x 3 subarrays to check for grid is linearly proportional to the size of grid, which is M⋅N. For each 3 x 3 subarray of grid, we iterate through all its values to check that they are distinct and within range, which takes constant time. We also perform the sum calculations that involve additional array indexing into a grid, which also takes constant time. Thus, the total time complexity is O(M⋅N).
•	Space Complexity: O(1)
isMagicSquare uses an array to keep track of which values the current subarray of grid contains. However, this array has a constant size of 10, so the space complexity is O(1)

             */
            public int UsingManualScan(int[][] grid)
            {
                int ans = 0;
                int m = grid.Length;
                int n = grid[0].Length;
                for (int row = 0; row + 2 < m; row++)
                {
                    for (int col = 0; col + 2 < n; col++)
                    {
                        if (IsMagicSquare(grid, row, col))
                        {
                            ans++;
                        }
                    }
                }
                return ans;
            }

            private bool IsMagicSquare(int[][] grid, int row, int col)
            {
                bool[] seen = new bool[10];
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int num = grid[row + i][col + j];
                        if (num < 1 || num > 9) return false;
                        if (seen[num]) return false;
                        seen[num] = true;
                    }
                }

                // check if diagonal sums are same value
                int diagonal1 =
                    grid[row][col] + grid[row + 1][col + 1] + grid[row + 2][col + 2];
                int diagonal2 =
                    grid[row + 2][col] + grid[row + 1][col + 1] + grid[row][col + 2];

                if (diagonal1 != diagonal2) return false;

                // check if all row sums share the same value as the diagonal sums
                int row1 = grid[row][col] + grid[row][col + 1] + grid[row][col + 2];
                int row2 =
                    grid[row + 1][col] +
                    grid[row + 1][col + 1] +
                    grid[row + 1][col + 2];
                int row3 =
                    grid[row + 2][col] +
                    grid[row + 2][col + 1] +
                    grid[row + 2][col + 2];

                if (!(row1 == diagonal1 && row2 == diagonal1 && row3 == diagonal1))
                {
                    return false;
                }

                // check if all column sums share same value as the diagonal sums
                int col1 = grid[row][col] + grid[row + 1][col] + grid[row + 2][col];
                int col2 =
                    grid[row][col + 1] +
                    grid[row + 1][col + 1] +
                    grid[row + 2][col + 1];
                int col3 =
                    grid[row][col + 2] +
                    grid[row + 1][col + 2] +
                    grid[row + 2][col + 2];

                if (!(col1 == diagonal1 && col2 == diagonal1 && col3 == diagonal2))
                {
                    return false;
                }

                return true;
            }
            /* Approach 2: Check Unique Properties of Magic Square
            Time Complexity
Let M and N be the number of rows and columns of grid, respectively.
•	Time Complexity: O(M⋅N)
Similar to Approach 1, the pattern checking in isMagicSquare is done in constant time. This function is called O(M⋅N) times, so the total time complexity is O(M⋅N).
•	Space Complexity: O(1)
The only auxiliary data structure used is a string storing our bordering pattern, which is a constant size. Thus, the space complexity is O(1).

             */
            public int UsingUniquePropsOfMagicSquare(int[][] grid)
            {
                int ans = 0;
                int m = grid.Length;
                int n = grid[0].Length;
                for (int row = 0; row + 2 < m; row++)
                {
                    for (int col = 0; col + 2 < n; col++)
                    {
                        if (IsMagicSquare(grid, row, col))
                        {
                            ans++;
                        }
                    }
                }
                return ans;

                bool IsMagicSquare(int[][] grid, int row, int col)
                {
                    // The sequences are each repeated twice to account for
                    // the different possible starting points of the sequence
                    // in the magic square
                    String sequence = "2943816729438167";
                    String sequenceReversed = "7618349276183492";

                    StringBuilder border = new StringBuilder();
                    // Flattened indices for bordering elements of 3x3 grid
                    int[] borderIndices = new int[] { 0, 1, 2, 5, 8, 7, 6, 3 };
                    foreach (int i in borderIndices)
                    {
                        int num = grid[row + i / 3][col + (i % 3)];
                        border.Append(num);
                    }

                    String borderConverted = border.ToString();

                    // Make sure the sequence starts at one of the corners
                    return (
                        grid[row][col] % 2 == 0 &&
                        (sequence.Contains(borderConverted) ||
                            sequenceReversed.Contains(borderConverted)) &&
                        grid[row + 1][col + 1] == 5
                    );
                }
            }


        }


        /* 1937. Maximum Number of Points with Cost
        https://leetcode.com/problems/maximum-number-of-points-with-cost/description/
         */
        class MaxPointsWithCostSol
        {
            /* Approach 1: Dynamic Programming
            Complexity Analysis
            Let m and n be the height and width of points.
            •	Time complexity: O(m⋅n)
            The outer loop runs m−1 times. Inside it, two inner loops run n−1 times and another one runs n times. Thus, the overall time complexity is O((m−1)⋅(n−1+n−1+n)), which simplifies to O(m⋅n).
            •	Space complexity: O(n)
            We use four additional arrays, each of which takes n space. All other variables take constant space.
            Thus, the space complexity is O(4⋅n)=O(n).

             */
            public long UsingDP(int[][] points)
            {
                int rows = points.Length, cols = points[0].Length;
                long[] previousRow = new long[cols];

                // Initialize the first row
                for (int col = 0; col < cols; ++col)
                {
                    previousRow[col] = points[0][col];
                }

                // Process each row
                for (int row = 0; row < rows - 1; ++row)
                {
                    long[] leftMax = new long[cols];
                    long[] rightMax = new long[cols];
                    long[] currentRow = new long[cols];

                    // Calculate left-to-right maximum
                    leftMax[0] = previousRow[0];
                    for (int col = 1; col < cols; ++col)
                    {
                        leftMax[col] = Math.Max(leftMax[col - 1] - 1, previousRow[col]);
                    }

                    // Calculate right-to-left maximum
                    rightMax[cols - 1] = previousRow[cols - 1];
                    for (int col = cols - 2; col >= 0; --col)
                    {
                        rightMax[col] = Math.Max(
                            rightMax[col + 1] - 1,
                            previousRow[col]
                        );
                    }

                    // Calculate the current row's maximum points
                    for (int col = 0; col < cols; ++col)
                    {
                        currentRow[col] = points[row + 1][col] +
                        Math.Max(leftMax[col], rightMax[col]);
                    }

                    // Update previousRow for the next iteration
                    previousRow = currentRow;
                }

                // Find the maximum value in the last processed row
                long maxPoints = 0;
                for (int col = 0; col < cols; ++col)
                {
                    maxPoints = Math.Max(maxPoints, previousRow[col]);
                }

                return maxPoints;
            }

            /* Approach 2: Dynamic Programming (Optimized)
            Complexity Analysis
Let m and n be the height and width of points.
•	Time complexity: O(m⋅n)
The main loop iterates through each row of points. Inside this loop, the algorithm uses two nested loops, each iterating n times. Overall, this takes O(m⋅n) time.
The final loop to find the maximum points also iterates n times.
Thus, the total time complexity of the algorithm is O(m⋅n)+O(n)=O(m⋅n).
•	Space complexity: O(n)
The algorithm uses an array previousRow of length n. Thus, the space complexity of the algorithm is O(n).

             */
            public long DPOptimal(int[][] points)
            {
                int cols = points[0].Length;
                long[] previousRow = new long[cols];

                foreach (int[] row in points)
                {
                    // runningMax holds the maximum value generated in the previous iteration of each loop
                    long runningMax = 0;

                    // Left to right pass
                    for (int col = 0; col < cols; ++col)
                    {
                        runningMax = Math.Max(runningMax - 1, previousRow[col]);
                        previousRow[col] = runningMax;
                    }

                    runningMax = 0;
                    // Right to left pass
                    for (int col = cols - 1; col >= 0; --col)
                    {
                        runningMax = Math.Max(runningMax - 1, previousRow[col]);
                        previousRow[col] = Math.Max(previousRow[col], runningMax) +
                        row[col];
                    }
                }

                // Find maximum points in the last row
                long maxPoints = 0;
                for (int col = 0; col < cols; ++col)
                {
                    maxPoints = Math.Max(maxPoints, previousRow[col]);
                }

                return maxPoints;
            }
        }


        /* 947. Most Stones Removed with Same Row or Column
        https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/description/
         */
        class MostStonesRemovedWithSameRowORColSol
        {

            /* Approach 1: Depth First Search
            Complexity Analysis
Let n be the length of the stones array.
•	Time complexity: O(n^2)
The graph is built by iterating over all pairs of stones (i,j) to check if they share the same row or column, resulting in O(n^2) time complexity.
In the worst case, the depth-first search will traverse all nodes and edges. Since each stone can be connected to every other stone, the algorithm can visit all O(n2) edges across all DFS calls.
Thus, the overall time complexity of the algorithm is 2⋅O(n^2)=O(n^2).
•	Space complexity: O(n^2)
In the worst case, any two stones could share the same row or column. So, the adjacencyList could store up to n2 edges, taking O(n^2) space.
The visited array takes an additional linear space.
The recursive DFS call stack can go as deep as the number of stones in a single connected component. In the worst case, this depth could be n, leading to O(n) additional space for the stack.
Thus, the space complexity of the algorithm is 2⋅O(n)+O(n^2)=O(n^2).
             */
            public int RemoveStones(int[][] stones)
            {
                int n = stones.Length;

                // Adjacency list to store graph connections
                List<int>[] adjacencyList = new List<int>[n];
                for (int i = 0; i < n; i++)
                {
                    adjacencyList[i] = new();
                }

                // Build the graph: Connect stones that share the same row or column
                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (
                            stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1]
                        )
                        {
                            adjacencyList[i].Add(j);
                            adjacencyList[j].Add(i);
                        }
                    }
                }

                int numOfConnectedComponents = 0;
                bool[] visited = new bool[n];

                // Traverse all stones using DFS to count connected components
                for (int i = 0; i < n; i++)
                {
                    if (!visited[i])
                    {
                        DepthFirstSearch(adjacencyList, visited, i);
                        numOfConnectedComponents++;
                    }
                }

                // Maximum stones that can be removed is total stones minus number of connected components
                return n - numOfConnectedComponents;
            }

            // DFS to visit all stones in a connected component
            private void DepthFirstSearch(
                List<int>[] adjacencyList,
                bool[] visited,
                int stone
            )
            {
                visited[stone] = true;

                foreach (int neighbor in adjacencyList[stone])
                {
                    if (!visited[neighbor])
                    {
                        DepthFirstSearch(adjacencyList, visited, neighbor);
                    }
                }
            }

            /* Approach 2: Disjoint Set Union
            Complexity Analysis
Let n be the length of the stones array.
•	Time complexity: O(n^2⋅α(n))
Initializing the parent array with -1 takes O(n) time.
The nested loops iterate through each pair of stones (i, j). The number of pairs is (n(n−1))/2, which is O(n^2).
For each pair, if the stones share the same row or column, the union operation is performed. The union (and subsequent find) operation takes O(α(n)), where α is the inverse Ackermann function.
Thus, the overall time complexity of the algorithm is O(n^2⋅α(n)).
•	Space complexity: O(n)
The only additional space used by the algorithm is the parent array, which takes O(n) space.

             */
            public int UsingDisjointSetUnion(int[][] stones)
            {
                int n = stones.Length;
                UnionFind uf = new UnionFind(n);

                // Populate uf by connecting stones that share the same row or column
                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        if (
                            stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1]
                        )
                        {
                            uf.Union(i, j);
                        }
                    }
                }

                return n - uf.Count;
            }

            // Union-Find data structure for tracking connected components
            public class UnionFind
            {

                public int[] Parent; // Array to track the parent of each node
                public int Count; // Number of connected components

                public UnionFind(int n)
                {
                    Parent = new int[n];
                    Array.Fill(Parent, -1); // Initialize all nodes as their own parent
                    Count = n; // Initially, each stone is its own connected component
                }

                // Find the root of a node with path compression
                public int Find(int node)
                {
                    if (Parent[node] == -1)
                    {
                        return node;
                    }
                    return Parent[node] = Find(Parent[node]);
                }

                // Union two nodes, reducing the number of connected components
                public void Union(int n1, int n2)
                {
                    int root1 = Find(n1);
                    int root2 = Find(n2);

                    if (root1 == root2)
                    {
                        return; // If they are already in the same component, do nothing
                    }

                    // Merge the components and reduce the count of connected components
                    Count--;
                    Parent[root1] = root2;
                }
            }
            /* Approach 3: Disjoint Set Union (Optimized)
Complexity Analysis
Let n be the length of the stones array.
•	Time complexity: O(n)
Since the size of the parent array is constant (20002), initializing it takes constant time.
The union operation is called n times, once for each stone. All union and find operations take O(α(20002))=O(1) time, where α is the inverse Ackermann function.
Thus, the overall time complexity is O(n).
•	Space complexity: O(n+20002)
The parent array takes a constant space of 20002.
The uniqueNodes set can have at most 2⋅n elements, corresponding to all unique x and y coordinates. The space complexity of this set is O(n).
Thus, the overall space complexity of the approach is O(n+20002).
While constants are typically excluded from complexity analysis, we've included it here due to its substantial size.

             */
            public int UsingDisjointSetUnionOptimal(int[][] stones)
            {
                int n = stones.Length;
                UnionFindExt uf = new UnionFindExt(20002); // Initialize UnionFind with a large enough range to handle coordinates

                // Union stones that share the same row or column
                for (int i = 0; i < n; i++)
                {
                    uf.Union(stones[i][0], stones[i][1] + 10001); // Offset y-coordinates to avoid conflict with x-coordinates
                }

                return n - uf.componentCount;
            }
            // Union-Find data structure for tracking connected components
            class UnionFindExt
            {

                public int[] parent; // Array to track the parent of each node
                public int componentCount; // Number of connected components
                public HashSet<int> uniqueNodes; // Set to track unique nodes

                public UnionFindExt(int n)
                {
                    parent = new int[n];
                    Array.Fill(parent, -1); // Initialize all nodes as their own parent
                    componentCount = 0;
                    uniqueNodes = new();
                }

                // Find the root of a node with path compression
                public int Find(int node)
                {
                    // If node is not marked, increase the component count
                    if (!uniqueNodes.Contains(node))
                    {
                        componentCount++;
                        uniqueNodes.Add(node);
                    }

                    if (parent[node] == -1)
                    {
                        return node;
                    }
                    return parent[node] = Find(parent[node]);
                }

                // Union two nodes, reducing the number of connected components
                public void Union(int node1, int node2)
                {
                    int root1 = Find(node1);
                    int root2 = Find(node2);

                    if (root1 == root2)
                    {
                        return; // If they are already in the same component, do nothing
                    }

                    // Merge the components and reduce the component count
                    parent[root1] = root2;
                    componentCount--;
                }
            }

        }


        /* 2018. Check if Word Can Be Placed In Crossword
        https://leetcode.com/problems/check-if-word-can-be-placed-in-crossword/description/	
        https://algo.monster/liteproblems/2018
         */
        class PlaceWordInCrosswordSol
        {
            private int rows;
            private int cols;
            private char[][] board;
            private String word;
            private int wordLength;

            /* Time and Space Complexity
            Time Complexity
            The time complexity of the given code is O(m * n * k), where m is the number of rows in the board, n is the number of columns in the board, and k is the length of the word to be placed. This complexity arises because the code iterates over all cells of the board (m * n) and for each cell, it attempts to place the word in all four directions. The check function, which is called for each direction, runs a loop up to the length of the word (k).
            Additionally, when evaluating the board for possible placements, the algorithm checks the perpendicular cells to ensure placement is at the beginning or end of a word sequence (which is a constant time check). Due to these operations being constant in time, they do not impact the linear relationship between the time complexity and the number of cells times the length of the word.
            Space Complexity
            The space complexity of the code is O(1) (constant space complexity). This is because the algorithm only uses a fixed amount of extra space for variables that store the dimensions of the board and indices during the checks regardless of the input size. No additional space proportional to the input size is required beyond what is used to store the board and word, which are inputs and not counted towards the space complexity.
             */
            // Method to check if the word can be placed in the crossword
            public bool PlaceWordInCrossword(char[][] board, String word)
            {
                rows = board.Length;
                cols = board[0].Length;
                this.board = board;
                this.word = word;
                wordLength = word.Length;

                // Traverse the board to check every potential starting point
                for (int i = 0; i < rows; ++i)
                {
                    for (int j = 0; j < cols; ++j)
                    {
                        // Check four possible directions from the current cell
                        // Left to right
                        bool leftToRight = (j == 0 || board[i][j - 1] == '#') && CanPlaceWord(i, j, 0, 1);
                        // Right to left
                        bool rightToLeft = (j == cols - 1 || board[i][j + 1] == '#') && CanPlaceWord(i, j, 0, -1);
                        // Up to down
                        bool upToDown = (i == 0 || board[i - 1][j] == '#') && CanPlaceWord(i, j, 1, 0);
                        // Down to up
                        bool downToUp = (i == rows - 1 || board[i + 1][j] == '#') && CanPlaceWord(i, j, -1, 0);

                        // If any direction is possible, return true
                        if (leftToRight || rightToLeft || upToDown || downToUp)
                        {
                            return true;
                        }
                    }
                }
                // If no direction is possible, return false
                return false;
            }

            // Helper method to check if the word can be placed starting from (i, j) in the specified direction (a, b)
            private bool CanPlaceWord(int i, int j, int rowIncrement, int colIncrement)
            {
                int endRow = i + rowIncrement * wordLength;
                int endCol = j + colIncrement * wordLength;

                // Check if the word goes out of bounds or is not terminated properly
                if (endRow < 0 || endRow >= rows || endCol < 0 || endCol >= cols || board[endRow][endCol] != '#')
                {
                    return false;
                }

                // Check each character to see if the word fits
                for (int p = 0; p < wordLength; ++p)
                {
                    if (i < 0 || i >= rows || j < 0 || j >= cols
                        || (board[i][j] != ' ' && board[i][j] != word[p]))
                    {
                        return false;
                    }
                    i += rowIncrement;
                    j += colIncrement;
                }
                return true;
            }
        }


        /* 3071. Minimum Operations to Write the Letter Y on a Grid
        https://leetcode.com/problems/minimum-operations-to-write-the-letter-y-on-a-grid/description/
        https://algo.monster/liteproblems/3071
         */
        class MinimumOperationsToWriteYInGridSol
        {

            /* Time and Space Complexity
            Time Complexity
            The time complexity of the given code is O(n^2). This is because the nested for-loops iterate over the entire grid, which is of size n x n. The operations inside the for-loops are constant time operations, causing the overall time to be proportional to the number of elements in the grid.
            Space Complexity
            The space complexity of the code is higher than O(1) mentioned in the reference answer. It actually depends on the range of the elements in the grid. The two counters cnt1 and cnt2 are used to count occurrences of elements that satisfy certain conditions. Because Python's Counter is essentially a hash map, the space used by these counters will increase with the variety of elements in the grid. Assuming the range of numbers in the grid is k, the space complexity would be O(k). If we were to consider k as being a constant because the problem might define a limit on the values of grid elements, then we could consider the space complexity to be O(1). Without such a constraint being specified, the space complexity is not strictly constant.
             */
            // Method to calculate the minimum number of operations needed to write 'Y' on the grid
            public int MinimumOperationsToWriteY(int[][] grid)
            {
                int gridSize = grid.Length; // Dimension of the grid (since it's n x n)

                // Arrays to store the count of each number (0, 1, 2) in the regions that form 'Y'
                int[] countRegionY = new int[3];
                int[] countOutsideRegionY = new int[3];

                // Loop through each cell in the grid to separate the cells that will form 'Y'
                // and those that will not
                for (int i = 0; i < gridSize; ++i)
                {
                    for (int j = 0; j < gridSize; ++j)
                    {
                        // Conditions to detect cells that are part of 'Y'
                        bool isDiagonalFromTopLeft = i == j && i <= gridSize / 2;
                        bool isDiagonalFromTopRight = i + j == gridSize - 1 && i <= gridSize / 2;
                        bool isVerticalMiddleLine = j == gridSize / 2 && i >= gridSize / 2;

                        if (isDiagonalFromTopLeft || isDiagonalFromTopRight || isVerticalMiddleLine)
                        {
                            // Increment the count for the region that forms 'Y'
                            ++countRegionY[grid[i][j]];
                        }
                        else
                        {
                            // Increment the count for the region outside 'Y'
                            ++countOutsideRegionY[grid[i][j]];
                        }
                    }
                }

                // We'll assume the worst case where we need to change every cell initially
                int minOperations = gridSize * gridSize;

                // Evaluate all combinations where the number for region 'Y' and outside are different
                for (int numberForY = 0; numberForY < 3; ++numberForY)
                {
                    for (int numberOutsideY = 0; numberOutsideY < 3; ++numberOutsideY)
                    {
                        if (numberForY != numberOutsideY)
                        {
                            // Calculate the number of operations needed if these two numbers were used
                            int operations = gridSize * gridSize - countRegionY[numberForY] - countOutsideRegionY[numberOutsideY];
                            // Check if the current operation count is lower than the minimum found so far
                            minOperations = Math.Min(minOperations, operations);
                        }
                    }
                }

                // Return the minimum number of operations found
                return minOperations;
            }
        }

        /* 1219. Path with Maximum Gold
        https://leetcode.com/problems/path-with-maximum-gold/description/	
         */

        class GetPathWithMaxGoldSol
        {
            private readonly int[] DIRECTIONS = new int[] { 0, 1, 0, -1, 0 };

            /* Approach 1: Depth-First Search with Backtracking 
Complexity Analysis
Let n be the number of rows in the grid, m be the number of columns, and g be the number of gold cells.
•	Time complexity: O(m⋅n−g+g⋅3^g)
We search for the path with maximum gold from each starting cell that contains gold using the backtrack function, which recursively calls itself. From the starting cell, we explore paths in 4 directions, but for each additional cell in the path, we explore paths in 3 directions because we already collected gold from the direction we came from. That means the backtrack function can be called up to 3^g times for a given starting cell, and it takes O(g⋅3^g) to search for the maximum gold from all the gold cells.
In the getMaximumGold function, we iterate through each cell in the matrix, checking whether each has gold. We've already accounted for the gold cells, so this takes O(m⋅n−g) for the cells that do not contain gold.
Therefore, the overall time complexity is O(m⋅n−g+g⋅3^g)
•	Space complexity: O(g)
Since the length of a path through gold cells can be g, the recursive call stack can grow up to size g.

            */
            public int getMaximumGold(int[][] grid)
            {
                int rows = grid.Length;
                int cols = grid[0].Length;
                int maxGold = 0;

                // Search for the path with the maximum gold starting from each cell
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {
                        maxGold = Math.Max(maxGold, DfsBacktrack(grid, rows, cols, row, col));
                    }
                }
                return maxGold;
            }

            private int DfsBacktrack(int[][] grid, int rows, int cols, int row, int col)
            {
                // Base case: this cell is not in the matrix or this cell has no gold
                if (row < 0 || col < 0 || row == rows || col == cols || grid[row][col] == 0)
                {
                    return 0;
                }
                int maxGold = 0;

                // Mark the cell as visited and save the value
                int originalVal = grid[row][col];
                grid[row][col] = 0;

                // Backtrack in each of the four directions
                for (int direction = 0; direction < 4; direction++)
                {
                    maxGold = Math.Max(maxGold,
                            DfsBacktrack(grid, rows, cols, DIRECTIONS[direction] + row,
                                    DIRECTIONS[direction + 1] + col));
                }

                // Set the cell back to its original value
                grid[row][col] = originalVal;
                return maxGold + originalVal;
            }

            /* Approach 2: Breadth-First Search with Backtracking
            Complexity Analysis
Let n be the number of rows in the grid, m be the number of columns, and g be the number of gold cells.
•	Time complexity: O(m⋅n−g+g⋅3^g)
We search for the path with the maximum gold starting from each gold cell. We search in three directions for each cell along the path because we have already collected the gold on the current path. This means we push up to 3^g entries to the queue. We stop the BFS when the queue is empty, so this process takes O(g⋅3^g).
In the UsingBFSWithBacktracking function, we check whether each cell contains gold. The gold cells have already been accounted for, so this takes m⋅n−g for the cells with no gold.
Therefore, the overall time complexity is O(m⋅n−g+g⋅3^g).
•	Space complexity: O(g⋅3^g) (Java and Python3) or O(3^g+m⋅n) (C++)
Java and Python3: A visited set of size g is created for each entry in the queue. The queue can grow to size 3g, so the currVis sets can use up to O(g⋅3^g) space.
C++: The queue may use up to 3^g space. We initialize the visited bitset to size 1024 since the constraints limit m and n to 100, ensuring the bitset is large enough to store all 1000 possible coordinates. Therefore, the space complexity is O(3^g+m⋅n).
	
             */
            public int UsingBFSWithBacktracking(int[][] grid)
            {
                int rows = grid.Length;
                int cols = grid[0].Length;

                // Find the total amount of gold in the grid
                int totalGold = 0;
                foreach (int[] row in grid)
                {
                    foreach (int cell in row)
                    {
                        totalGold += cell;
                    }
                }

                // Search for the path with the maximum gold starting from each cell
                int maxGold = 0;
                for (int row = 0; row < rows; row++)
                {
                    for (int col = 0; col < cols; col++)
                    {
                        if (grid[row][col] != 0)
                        {
                            maxGold = Math.Max(maxGold, BfsBacktrack(grid, rows, cols, row, col));
                            // If we found a path with the total gold, it's the max gold
                            if (maxGold == totalGold)
                            {
                                return totalGold;
                            }
                        }
                    }
                }
                return maxGold;
            }

            // Helper function to perform BFS
            private int BfsBacktrack(int[][] grid, int rows, int cols, int row, int col)
            {
                Queue<(int[], HashSet<String>)> queue = new();
                HashSet<String> visited = new HashSet<string>();
                int maxGold = 0;
                visited.Add(row + "," + col);
                queue.Enqueue((new int[] { row, col, grid[row][col] }, visited));

                while (queue.Count > 0)
                {
                    (int[] cell, HashSet<String> currVis) current = queue.Dequeue();
                    int currRow = current.cell[0];
                    int currCol = current.cell[1];
                    int currGold = current.cell[2];

                    maxGold = Math.Max(maxGold, currGold);

                    // Search for gold in each of the 4 neighbor cells
                    for (int direction = 0; direction < 4; direction++)
                    {
                        int nextRow = currRow + DIRECTIONS[direction];
                        int nextCol = currCol + DIRECTIONS[direction + 1];

                        // If the next cell is in the matrix, has gold,
                        // and has not been visited, add it to the queue
                        if (nextRow >= 0 && nextRow < rows && nextCol >= 0 && nextCol < cols &&
                                grid[nextRow][nextCol] != 0 &&
                                !current.currVis.Contains(nextRow + "," + nextCol))
                        {

                            current.currVis.Add(nextRow + "," + nextCol);
                            HashSet<string> copyVis = new HashSet<string>(current.currVis);
                            queue.Enqueue((new int[] { nextRow, nextCol,
                                           currGold + grid[nextRow][nextCol]}, copyVis));
                            current.currVis.Remove(nextRow + "," + nextCol);

                        }
                    }
                }
                return maxGold;
            }
        }


        /* 861. Score After Flipping Matrix
        https://leetcode.com/problems/score-after-flipping-matrix/description/
         */

        class MatrixScoreAfterFlippingSol
        {

            /* Approach 1: Greedy Way (Modifying Input)
Complexity Analysis
Let m and n be the number of rows and columns of the matrix, respectively.
•	Time complexity: O(m⋅n)
In the worst case, we traverse the entire matrix twice. The total number of cells in the matrix is m⋅n. Thus, the time complexity is O(2⋅m⋅n), which simplifies to O(m⋅n).
•	Space complexity: O(1)
We do not use any additional data structures in our implementation. Therefore, the space complexity remains O(1).

             */
            public int UsingGreedyWithModifyInput(int[][] grid)
            {
                int m = grid.Length;
                int n = grid[0].Length;

                // Set first column
                for (int i = 0; i < m; i++)
                {
                    if (grid[i][0] == 0)
                    {
                        // Flip row
                        for (int j = 0; j < n; j++)
                        {
                            grid[i][j] = 1 - grid[i][j];
                        }
                    }
                }

                // Optimize columns except first column
                for (int j = 1; j < n; j++)
                {
                    int countZero = 0;
                    // Count zeros
                    for (int i = 0; i < m; i++)
                    {
                        if (grid[i][j] == 0)
                        {
                            countZero++;
                        }
                    }
                    // Flip the column if there are more zeros for better score
                    if (countZero > m - countZero)
                    {
                        for (int i = 0; i < m; i++)
                        {
                            grid[i][j] = 1 - grid[i][j];
                        }
                    }
                }

                // Calculate the final score considering bit positions
                int score = 0;
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        // Left shift bit by place value of column to find column contribution
                        int columnScore = grid[i][j] << (n - j - 1);
                        // Add contribution to score
                        score += columnScore;
                    }
                }

                // return final result
                return score;
            }
            /* Approach 2: Greedy Way (Without Modifying Input)
            Complexity Analysis
Let m and n be the number of rows and columns of the matrix, respectively.
•	Time complexity: O(m⋅n)
We traverse the entire matrix only once. The total number of cells in the matrix is m⋅n, resulting in a time complexity of O(m⋅n).
•	Space complexity: O(1)
We do not use any additional space in our implementation. Therefore, our space complexity remains O(1).	

             */
            public int UsingGreedyWithoutModifyInput(int[][] grid)
            {
                int m = grid.Length;
                int n = grid[0].Length;

                // Set score to summation of first column
                int score = (1 << (n - 1)) * m;

                // Loop over all other columns
                for (int j = 1; j < n; j++)
                {
                    int countSameBits = 0;
                    for (int i = 0; i < m; i++)
                    {
                        // Count elements matching first in row
                        if (grid[i][j] == grid[i][0])
                        {
                            countSameBits++;
                        }
                    }
                    // Calculate score based on the number of same bits in a column
                    countSameBits = Math.Max(countSameBits, m - countSameBits);
                    // Calculate the score contribution for the current column
                    int columnScore = (1 << (n - j - 1)) * countSameBits;
                    // Add contribution to score
                    score += columnScore;
                }

                return score;
            }

        }
        /* 
        542. 01 Matrix
        https://leetcode.com/problems/01-matrix/description/
         */
        class UpdateMatrixSol
        {
            int m;
            int n;
            int[][] directions = new int[][] { new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { -1, 0 } };

            /* Approach 1: Breadth-First Search (BFS)

            Complexity Analysis
            Given m as the number of rows and n as the number of columns,
            •	Time complexity: O(m⋅n)
            The BFS never visits a node more than once due to seen. Each node has at most 4 neighbors, so the work done at each node is O(1). This gives us a time complexity of O(m⋅n), the number of nodes.
            •	Space complexity: O(m⋅n)
            Note: some people may choose to modify the input mat instead of creating a copy matrix and using seen.
            It is generally not considered good practice to modify the input, especially if it's an array as they are passed by reference. Even then, you would only be saving on auxiliary space - if you modify the input as part of your algorithm, you still need to count it towards the space complexity.
            We could also elect to not count matrix as part of the space complexity as it serves only as the output and the output does not count towards the space complexity if it is not used in any logic during the algorithm.
            There is a lot of nuance when it comes to these decisions and you should always clarify your decisions with the interviewer.
            In our implementation, seen and queue uses O(m⋅n) space regardless of interpretation, so that is our space complexity.	
             */
            public int[][] UsingBFS(int[][] mat)
            {
                m = mat.Length;
                n = mat[0].Length;

                int[][] matrix = new int[m][];
                bool[][] seen = new bool[m][];
                Queue<State> queue = new();

                for (int row = 0; row < m; row++)
                {
                    for (int col = 0; col < n; col++)
                    {
                        matrix[row][col] = mat[row][col];
                        if (mat[row][col] == 0)
                        {
                            queue.Enqueue(new State(row, col, 0));
                            seen[row][col] = true;
                        }
                    }
                }

                while (queue.Count > 0)
                {
                    State state = queue.Dequeue();
                    int row = state.Row, col = state.Col, steps = state.Steps;

                    foreach (int[] direction in directions)
                    {
                        int nextRow = row + direction[0], nextCol = col + direction[1];
                        if (IsValid(nextRow, nextCol) && !seen[nextRow][nextCol])
                        {
                            seen[nextRow][nextCol] = true;
                            queue.Enqueue(new State(nextRow, nextCol, steps + 1));
                            matrix[nextRow][nextCol] = steps + 1;
                        }
                    }
                }

                return matrix;
            }

            public bool IsValid(int row, int col)
            {
                return 0 <= row && row < m && 0 <= col && col < n;
            }
            public class State
            {
                public int Row;
                public int Col;
                public int Steps;
                public State(int row, int col, int steps)
                {
                    this.Row = row;
                    this.Col = col;
                    this.Steps = steps;
                }
            }
            /* Approach 2: Dynamic Programming
            Complexity Analysis
Given m as the number of rows and n as the number of columns,
•	Time complexity: O(m⋅n)
We iterate over m⋅n cells twice, performing constant work at each iteration.
•	Space complexity: O(m⋅n)
As discussed above, some people may choose to reuse mat as dp.
A common misconception is that it would be O(1) space. It would only be O(1) auxiliary space, but as we are modifying the input and using it in the algorithm, it must be included in the space complexity. Again, it is not considered good practice to modify the input anyways, which is why we are creating dp, which uses O(m⋅n) space.

            */
            public int[][] updateMatrix(int[][] mat)
            {
                int m = mat.Length;
                int n = mat[0].Length;
                int[][] dp = new int[m][];

                for (int row = 0; row < m; row++)
                {
                    for (int col = 0; col < n; col++)
                    {
                        dp[row][col] = mat[row][col];
                    }
                }

                for (int row = 0; row < m; row++)
                {
                    for (int col = 0; col < n; col++)
                    {
                        if (dp[row][col] == 0)
                        {
                            continue;
                        }

                        int minNeighbor = m * n;
                        if (row > 0)
                        {
                            minNeighbor = Math.Min(minNeighbor, dp[row - 1][col]);
                        }

                        if (col > 0)
                        {
                            minNeighbor = Math.Min(minNeighbor, dp[row][col - 1]);
                        }

                        dp[row][col] = minNeighbor + 1;
                    }
                }

                for (int row = m - 1; row >= 0; row--)
                {
                    for (int col = n - 1; col >= 0; col--)
                    {
                        if (dp[row][col] == 0)
                        {
                            continue;
                        }

                        int minNeighbor = m * n;
                        if (row < m - 1)
                        {
                            minNeighbor = Math.Min(minNeighbor, dp[row + 1][col]);
                        }

                        if (col < n - 1)
                        {
                            minNeighbor = Math.Min(minNeighbor, dp[row][col + 1]);
                        }

                        dp[row][col] = Math.Min(dp[row][col], minNeighbor + 1);
                    }
                }

                return dp;
            }

        }

        /* 2497. Maximum Star Sum of a Graph
        https://leetcode.com/problems/maximum-star-sum-of-a-graph/description/
        https://algo.monster/liteproblems/2497
         */
        class MaxStarSumOfGraphSol
        {
            /* Time and Space Complexity
Time Complexity
The time complexity of the code is associated with several operations performed:
1.	Construction of the graph g: The for loop that iterates over the edges has a time complexity of O(E) where E is the number of edges because each edge is visited once.
2.	Sorting the adjacency lists: Each adjacency list in graph g is sorted in reverse order. In the worst case, if all nodes are connected to all other nodes, it will take O(N log N) time per node to sort, where N is the number of nodes. However, typically the number of edges connected to a node (degree of a node) is much lower than N, so it will more often be O(D log D) where D is the average degree of a node.
3.	Calculating the max star sum: The maximum star sum is computed in a single for loop that iterates over the nodes once. The complexity for this part is primarily from the summation operation which takes O(k) time for each node, as it sums up to k elements from the sorted adjacency list. Since it iterates over all N nodes, this results in a O(Nk) time complexity.
The overall worst-case time complexity can be approximated by adding these components together: O(E + N log N + Nk).
However, the sorting step's time complexity will often be dominated by the number of edges (sorting smaller adjacency lists): O(E + D log D + Nk).
Space Complexity
The space complexity involves:
1.	Storing the graph g: The graph is stored using adjacency lists, which will require O(N + E) space: O(N) for storing N keys and O(E) for storing the list of neighbor values.
2.	Auxiliary space for sorting: Sorting the adjacency lists will require additional space which depends on the sorting algorithm used by Python's .sort() method (Timsort), but since in-place sorting is used, the impact on space complexity should be minimal.
3.	Storing sums: The space for storing the sums is not significant as it only stores the sums for each node which is O(1) per node resulting in O(N) in total.
The overall space complexity is O(N + E).
 */

            // Function to calculate the maximum star sum
            public int MaxStarSum(int[] values, int[][] edges, int k)
            {
                int numberOfValues = values.Length;

                // Create a list of lists to represent the graph
                List<int>[] graph = new List<int>[numberOfValues];
                for (int i = 0; i < numberOfValues; i++)
                {
                    graph[i] = new List<int>();
                }

                // Building an adjacency list for each node containing values
                foreach (int[] edge in edges)
                {
                    int from = edge[0], to = edge[1];

                    // Only add the positive values to the adjacency list
                    if (values[to] > 0)
                    {
                        graph[from].Add(values[to]);
                    }
                    if (values[from] > 0)
                    {
                        graph[to].Add(values[from]);
                    }
                }

                // Sort the adjacency lists in descending order
                foreach (List<int> adjacentValues in graph)
                {
                    adjacentValues.Sort((a, b) => b.CompareTo(a));
                }

                // Initialize the answer with the lowest possible value
                int maxStarSum = int.MinValue;

                // Iterate through each node to calculate the sum of the stars
                for (int i = 0; i < numberOfValues; ++i)
                {
                    int nodeValue = values[i];

                    // Add up the k highest values connected to the node
                    for (int j = 0; j < Math.Min(graph[i].Count, k); ++j)
                    {
                        nodeValue += graph[i][j];
                    }

                    // Update the answer with the maximum sum found so far
                    maxStarSum = Math.Max(maxStarSum, nodeValue);
                }

                // Return the maximum star sum
                return maxStarSum;
            }
        }

        /* 1514. Path with Maximum Probability
        https://leetcode.com/problems/path-with-maximum-probability/description/
         */
        public class PathWithMaxProbabilitySol
        {

            /* Approach 1: Bellman-Ford Algorithm 
            Complexity Analysis
Let n be the number of nodes and m be the number of edges.
•	Time complexity: O(n⋅m)
o	The algorithm relaxes all edges in the graph n - 1 times, each round contains an iteration over all m edges.
•	Space complexity: O(n)
o	We only need an array of size n to update the maximum probability to reach each node from the starting node.

            */
            public double UsingBellmanFordAlgo(
                int n,
                int[][] edges,
                double[] succProb,
                int start,
                int end
            )
            {
                double[] maxProb = new double[n];
                maxProb[start] = 1.0;

                for (int i = 0; i < n - 1; i++)
                {
                    bool hasUpdate = false;
                    for (int j = 0; j < edges.Length; j++)
                    {
                        int u = edges[j][0];
                        int v = edges[j][1];
                        double pathProb = succProb[j];
                        if (maxProb[u] * pathProb > maxProb[v])
                        {
                            maxProb[v] = maxProb[u] * pathProb;
                            hasUpdate = true;
                        }
                        if (maxProb[v] * pathProb > maxProb[u])
                        {
                            maxProb[u] = maxProb[v] * pathProb;
                            hasUpdate = true;
                        }
                    }
                    if (!hasUpdate)
                    {
                        break;
                    }
                }

                return maxProb[end];
            }

            /* Approach 2: Shortest Path Faster Algorithm
Complexity Analysis
Let n be the number of nodes and m be the number of edges.
•	Time complexity: O(n⋅m)
o	The worst-case running of SPFA is O(∣V∣⋅∣E∣). However, this is only the worst-case scenario, and the average runtime of SPFA is better than in Bellman-Ford.
•	Space complexity: O(n+m)
o	We build a hash map graph based on all edges, which takes O(m) space.
o	The algorithm stores the probability array max_prob of size O(n) and a queue of vertices queue. In the worst-case scenario, there are O(m) nodes in queue at the same time.

             */
            public double UsingShortestPathFasterAlgo(
              int nodeCount,
              int[][] edges,
              double[] successProbabilities,
              int startNode,
              int endNode
          )
            {
                Dictionary<int, List<KeyValuePair<int, double>>> graph = new Dictionary<int, List<KeyValuePair<int, double>>>();
                for (int i = 0; i < edges.Length; i++)
                {
                    int startEdge = edges[i][0], endEdge = edges[i][1];
                    double pathProbability = successProbabilities[i];
                    if (!graph.ContainsKey(startEdge))
                    {
                        graph[startEdge] = new List<KeyValuePair<int, double>>();
                    }
                    graph[startEdge].Add(new KeyValuePair<int, double>(endEdge, pathProbability));
                    if (!graph.ContainsKey(endEdge))
                    {
                        graph[endEdge] = new List<KeyValuePair<int, double>>();
                    }
                    graph[endEdge].Add(new KeyValuePair<int, double>(startEdge, pathProbability));
                }

                double[] maxProbability = new double[nodeCount];
                maxProbability[startNode] = 1d;

                Queue<int> queue = new Queue<int>();
                queue.Enqueue(startNode);
                while (queue.Count > 0)
                {
                    int currentNode = queue.Dequeue();
                    foreach (KeyValuePair<int, double> neighbor in graph.GetValueOrDefault(currentNode, new List<KeyValuePair<int, double>>()))
                    {
                        int nextNode = neighbor.Key;
                        double pathProbability = neighbor.Value;

                        // Only update maxProbability[nextNode] if the current path increases
                        // the probability of reaching nextNode.
                        if (maxProbability[currentNode] * pathProbability > maxProbability[nextNode])
                        {
                            maxProbability[nextNode] = maxProbability[currentNode] * pathProbability;
                            queue.Enqueue(nextNode);
                        }
                    }
                }

                return maxProbability[endNode];
            }
            /* Approach 3: Dijkstra's Algorithm
Complexity Analysis
Let n be the number of nodes and m be the number of edges.
•	Time Complexity: O((n+m)⋅logn)
o	We build an adjacency list graph based on all edges, which takes O(m) time.
o	In the worst case, each node could be pushed into the priority queue exactly once, which results in O(n⋅logn) operations.
o	Each edge is considered exactly once when its corresponding node is dequeued from the priority queue. This takes O(m⋅logn) time in total, due to the priority queue's logn complexity for insertion and deletion operations.
You can also refer to our Dijkstra's Algorithm Explore Card for details on the complexity analysis.
•	Space Complexity: O(n+m)
o	We build an adjacency list graph based on all edges, which takes O(m) space.
o	The algorithm stores the maxProb array, which uses O(n) space.
o	We use a priority queue to keep track of nodes to be visited, and there are at most n nodes in the queue.
o	To sum up, the overall space complexity is O(n+m).

             */
            public double UsingDijkstrasAlgo(int n, int[][] edges, double[] succProb, int start, int end)
            {
                var graph = new Dictionary<int, List<(int, double)>>();
                for (int i = 0; i < edges.Length; i++)
                {
                    int u = edges[i][0], v = edges[i][1];
                    double pathProb = succProb[i];
                    if (!graph.ContainsKey(u))
                        graph[u] = new List<(int, double)>();
                    graph[u].Add((v, pathProb));
                    if (!graph.ContainsKey(v))
                        graph[v] = new List<(int, double)>();
                    graph[v].Add((u, pathProb));
                }

                double[] maxProb = new double[n];
                maxProb[start] = 1d;

                var pq = new PriorityQueue<(double, int), double>();
                pq.Enqueue((1.0, start), -1.0); //MaxHeap
                while (pq.Count > 0)
                {
                    var cur = pq.Dequeue();
                    double curProb = cur.Item1;
                    int curNode = cur.Item2;
                    if (curNode == end)
                    {
                        return curProb;
                    }
                    if (graph.ContainsKey(curNode)) // Check if the node has been processed
                    {
                        foreach (var nxt in graph[curNode])
                        {
                            int nxtNode = nxt.Item1;
                            double pathProb = nxt.Item2;
                            if (curProb * pathProb > maxProb[nxtNode])
                            {
                                maxProb[nxtNode] = curProb * pathProb;
                                pq.Enqueue((maxProb[nxtNode], nxtNode), -maxProb[nxtNode]);
                            }
                        }
                        graph.Remove(curNode); // Clear the adjacency list by removing the entry
                    }
                }

                return 0d;
            }


        }


        /* 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
        https://leetcode.com/problems/maximum-side-length-of-a-square-with-sum-less-than-or-equal-to-threshold/description/
        https://algo.monster/liteproblems/1292        
         */

        class MaxSideLengthOfSquareWithSumLessThanOrEqualToThresholdSol
        {
            private int numRows;           // Number of rows in the matrix
            private int numCols;           // Number of columns in the matrix
            private int threshold;         // Threshold for the maximum sum of the square sub-matrix
            private int[][] prefixSum;     // Prefix sum matrix

            /* Time and Space Complexity
            Time Complexity
            The provided code operates in several steps. The computation of the prefix sum matrix, s, takes O(m * n) time where m is the number of rows and n is the number of columns in the input matrix, mat. This is because it needs to iterate over every element in mat to compute the sum.
            The check function, which checks if a square with side k has a sum less than or equal to the threshold, is O(m * n) as well for each individual k, since, in the worst case, it has to iterate over the entire matrix again, checking each possible top-left corner of a square.
            Finally, the binary search performed in the last part of the code runs in O(log(min(m, n))) time. Since the binary search calls the check function at each iteration, the combined time complexity for the search routine with subsequent checks is O(m * n * log(min(m, n))).
            Hence, the overall time complexity of the code is O(m * n + m * n * log(min(m, n))) which simplifies to O(m * n * log(min(m, n))).
            Space Complexity
            Regarding space complexity, the additional space is used for the prefix sum matrix, s, which has dimensions (m + 1) x (n + 1). Therefore, the space complexity is O((m + 1) * (n + 1)), which simplifies to O(m * n) since the +1 is relatively negligible for large m and n.
             */
            public int MaxSideLength(int[][] mat, int threshold)
            {
                this.numRows = mat.Length;
                this.numCols = mat[0].Length;
                this.threshold = threshold;
                this.prefixSum = new int[numRows + 1][];

                // Build the prefix sum matrix
                for (int i = 1; i <= numRows; ++i)
                {
                    for (int j = 1; j <= numCols; ++j)
                    {
                        prefixSum[i][j] = prefixSum[i - 1][j] + prefixSum[i][j - 1]
                                          - prefixSum[i - 1][j - 1] + mat[i - 1][j - 1];
                    }
                }

                int left = 0, right = Math.Min(numRows, numCols);
                // Binary search for the maximum side length of a square sub-matrix
                while (left < right)
                {
                    int mid = (left + right + 1) >> 1;
                    if (CanFindSquareWithMaxSum(mid))
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }
                return left;
            }

            // Check if there's a square sub-matrix of side length k with a sum less than or equal to the threshold
            private bool CanFindSquareWithMaxSum(int sideLength)
            {
                for (int i = 0; i <= numRows - sideLength; ++i)
                {
                    for (int j = 0; j <= numCols - sideLength; ++j)
                    {
                        int sum = prefixSum[i + sideLength][j + sideLength]
                                - prefixSum[i][j + sideLength]
                                - prefixSum[i + sideLength][j] + prefixSum[i][j];
                        if (sum <= threshold)
                        {
                            return true;
                        }
                    }
                }
                return false;
            }


        }

        /* 2316. Count Unreachable Pairs of Nodes in an Undirected Graph
        https://leetcode.com/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/description/
         */

        class CountUnreachablePairsOfNodesInUndirectedGraphSol
        {

            /* Approach 1: Depth First Search
Complexity Analysis
Here n is the number of nodes and e is the total number of edges.
•	Time complexity: O(n+e).
o	We need O(e) time to initialize the adjacency list and O(n) time to initialize the visit array.
o	The dfs function visits each node once, which takes O(n) time in total. Because we have undirected edges, each edge can only be iterated twice (by nodes at the end), resulting in O(e) operations total while visiting all nodes.
o	As a result, the total time required is O(n+e)
•	Space complexity: O(n+e).
o	Building the adjacency list takes O(e) space.
o	The visit array takes O(n) space.
o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.
s
             */
            public long UsingDFS(int totalNodes, int[][] edges)
            {
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] edge in edges)
                {
                    if (!adjacencyList.ContainsKey(edge[0]))
                    {
                        adjacencyList[edge[0]] = new List<int>();
                    }
                    adjacencyList[edge[0]].Add(edge[1]);
                    if (!adjacencyList.ContainsKey(edge[1]))
                    {
                        adjacencyList[edge[1]] = new List<int>();
                    }
                    adjacencyList[edge[1]].Add(edge[0]);
                }

                long totalPairs = 0;
                long componentSize = 0;
                long remainingNodes = totalNodes;
                bool[] visitedNodes = new bool[totalNodes];
                for (int i = 0; i < totalNodes; i++)
                {
                    if (!visitedNodes[i])
                    {
                        componentSize = DepthFirstSearch(i, adjacencyList, visitedNodes);
                        totalPairs += componentSize * (remainingNodes - componentSize);
                        remainingNodes -= componentSize;
                    }
                }
                return totalPairs;
            }
            private int DepthFirstSearch(int node, Dictionary<int, List<int>> adjacencyList, bool[] visitedNodes)
            {
                int componentSize = 1;
                visitedNodes[node] = true;
                if (!adjacencyList.ContainsKey(node))
                {
                    return componentSize;
                }
                foreach (int neighbor in adjacencyList[node])
                {
                    if (!visitedNodes[neighbor])
                    {
                        componentSize += DepthFirstSearch(neighbor, adjacencyList, visitedNodes);
                    }
                }
                return componentSize;
            }
            /* 
Approach 2: Breadth First Search
Complexity Analysis
Here n is the number of nodes and e is the total number of edges.
•	Time complexity: O(n+e).
o	We need O(e) time to initialize the adjacency list and O(n) to initialize the visit array.
o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can only be pushed once, leading to O(n) operations for n nodes. We iterate over all the neighbors of each node that is popped out of the queue, so for an undirected edge, a given edge could be iterated at most twice (by nodes at both ends), resulting in O(e) operations total for all the nodes.
o	As a result, the total time required is O(n+e).
•	Space complexity: O(n+e).
o	Building the adjacency list takes O(e) space.
o	The visit array takes O(n) space as well.
o	The BFS queue takes O(n) space in the worst-case because each node is added once.

 */
            public long UsingBFS(int numberOfNodes, int[][] edges)
            {
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] edge in edges)
                {
                    if (!adjacencyList.ContainsKey(edge[0]))
                    {
                        adjacencyList[edge[0]] = new List<int>();
                    }
                    adjacencyList[edge[0]].Add(edge[1]);

                    if (!adjacencyList.ContainsKey(edge[1]))
                    {
                        adjacencyList[edge[1]] = new List<int>();
                    }
                    adjacencyList[edge[1]].Add(edge[0]);
                }

                long numberOfPairs = 0;
                long sizeOfComponent = 0;
                long remainingNodes = numberOfNodes;
                bool[] visited = new bool[numberOfNodes];
                for (int i = 0; i < numberOfNodes; i++)
                {
                    if (!visited[i])
                    {
                        sizeOfComponent = Bfs(i, adjacencyList, visited);
                        numberOfPairs += sizeOfComponent * (remainingNodes - sizeOfComponent);
                        remainingNodes -= sizeOfComponent;
                    }
                }
                return numberOfPairs;
            }
            public int Bfs(int node, Dictionary<int, List<int>> adjacencyList, bool[] visited)
            {
                Queue<int> queue = new Queue<int>();
                queue.Enqueue(node);
                int count = 1;
                visited[node] = true;

                while (queue.Count > 0)
                {
                    node = queue.Dequeue();
                    if (!adjacencyList.ContainsKey(node))
                    {
                        continue;
                    }
                    foreach (int neighbor in adjacencyList[node])
                    {
                        if (!visited[neighbor])
                        {
                            visited[neighbor] = true;
                            count++;
                            queue.Enqueue(neighbor);
                        }
                    }
                }
                return count;
            }
            /* Approach 3: Union-find
            Complexity Analysis
Here n is the number of nodes and e is the total number edges.
•	Time complexity: O(n+e).
o	For T operations, the amortized time complexity of the union-find algorithm (using path compression and union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1).
o	Initializing UnionFind takes O(n) time because we are initializing the parent and rank arrays of size n each.
o	We iterate through every edge and perform union of the nodes connected by the edge which takes O(1) time per operation. It takes O(e) time for e edges.
o	We then iterate through all the nodes and use the find operation to find their components. The find operation takes O(1) amortized time for each operation.
o	We iterate through the componentSize map. There can be no more than n components because there are n nodes. In the worst-case scenario, determining the size of all the components would take O(n).
o	As a result, the total time required is O(n+e).
•	Space complexity: O(n).
o	We are using the parent and rank arrays, both of which require O(n) space each.
o	The componentSize map would require O(n) space as well in the worst-case.

             */
            public long UsingUnionFind(int n, int[][] edges)
            {
                DSUArray dsu = new DSUArray(n);
                foreach (int[] edge in edges)
                {
                    dsu.Union(edge[0], edge[1]);
                }

                Dictionary<int, int> componentSize = new();
                for (int i = 0; i < n; i++)
                {
                    int parent = dsu.Find(i);
                    if (componentSize.ContainsKey(parent))
                    {
                        componentSize[parent] = componentSize[parent] + 1;
                    }
                    else
                    {
                        componentSize[parent] = 1;
                    }
                }

                long numberOfPaths = 0;
                long remainingNodes = n;
                foreach (int size in componentSize.Values)
                {
                    numberOfPaths += size * (remainingNodes - size);
                    remainingNodes -= size;
                }
                return numberOfPaths;
            }

        }










    }


}

