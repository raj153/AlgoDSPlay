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
        public double[] WaterfallStreams(double[][] array, int source){
            //T:O(W^2*h) | S:O(W)
            double[] rowAbove =array[0];

            rowAbove[source] = -1; // -1 used to represent water since 1 is used for a block

            for(int row=1; row < array.Length; row++){
                double[] currentRow = array[row];

                for(int col=0; col < rowAbove.Length; col++){
                    double valueAbove = rowAbove[col];

                    bool hasWaterAbove = valueAbove < 0;
                    bool hasBlock = currentRow[col] == 1.0;

                    if(!hasWaterAbove) continue;

                    if(!hasBlock){
                        currentRow[col]+=valueAbove;
                        continue;
                    }

                    double splitWatter = valueAbove/2;

                    int rightColIdx = col;

                    while(rightColIdx+1 < rowAbove.Length){
                        rightColIdx +=1;

                        if(rowAbove[rightColIdx] == 1.0) {
                            break;
                        }

                        if(currentRow[rightColIdx] != 1.0){

                            currentRow[rightColIdx] += splitWatter;
                            break;
                        }
                    }

                    int leftColIdx = col;
                    while(leftColIdx -1 >=0){
                        leftColIdx -=1;

                        if(rowAbove[leftColIdx] == 1.0) break;

                        if(currentRow[leftColIdx] != 1.0){
                            currentRow[leftColIdx]+= splitWatter;
                            break;
                        }
                    }
                   
                }
                rowAbove = currentRow;

            }

            double[] finalPercentages = new double[rowAbove.Length];
            for(int idx=0; idx< rowAbove.Length; idx++){
                double num = rowAbove[idx];
                if(num ==0)
                    finalPercentages[idx] = num;
                else
                    finalPercentages[idx] = (num*-100);
            }

            return finalPercentages;
        
        }
        //https://www.algoexpert.io/questions/a*-algorithm 
        public static int[,] FindShortestPathUsingAStarAlgo(int startRow, int startCol, int endRow, int endCol, int[,] graph){
            //T:O(w*h*log(w*h)) | S:O(w*h)
            List<List<Node>> nodes = InitializeNodes(graph);
            Node startNode = nodes[startRow][startCol];
            Node endNode = nodes[endRow][endCol];

            startNode.distanceFromStart = 0;
            startNode.estimatedDistanceToEnd = CalculateManhattanDistance(startNode, endNode);
            
            List<Node> nodesToVisitList = new List<Node>();
            nodesToVisitList.Add(startNode);
                    
            MinHeapForAStarAlgo nodesToVisit = new MinHeapForAStarAlgo(nodesToVisitList);

            while(!nodesToVisit.IsEmpty()){
                Node currentMinDistanceNode = nodesToVisit.Remove();
                if(currentMinDistanceNode == endNode) break;

                List<Node> neighbors= GetNeighbors(currentMinDistanceNode, nodes);
                foreach(var neighbor in neighbors){
                    if(neighbor.Value == 1) continue;

                    int tentativeDistanceToNeighbor = currentMinDistanceNode.distanceFromStart+1;
                    if(tentativeDistanceToNeighbor >= neighbor.distanceFromStart)
                        continue;
                    neighbor.CameFrom = currentMinDistanceNode;
                    neighbor.distanceFromStart = tentativeDistanceToNeighbor;
                    neighbor.estimatedDistanceToEnd= tentativeDistanceToNeighbor+ CalculateManhattanDistance(neighbor,endNode);
                    
                    if(!nodesToVisit.ContainsNode(neighbor))
                        nodesToVisit.Insert(neighbor);
                    else{
                        nodesToVisit.Update(neighbor);
                    }

                }
                
            }
            return ReconstructPath(endNode);
        }

        private static List<Node> GetNeighbors(Node node, List<List<Node>> nodes)
        {
            List<Node> neighbors = new List<Node>();

            int numRows = nodes.Count();
            int numCols = nodes[0].Count();

            int row = node.Row;
            int col = node.Col;

            if(row < numRows -1)//DOWN
                neighbors.Add(nodes[row+1][col]);
            if(row > 0)//UP
                neighbors.Add(nodes[row-1][col]);
            if(col < numCols -1)//RIGHT
                neighbors.Add(nodes[row][col+1]);
            if(col >0)//LEFT
                neighbors.Add(nodes[row][col-1]);

            return neighbors;
        }

        private static int[,] ReconstructPath(Node endNode)
        {
            if(endNode.CameFrom == null)
                return new int[,]{};
            
            List<List<int>>  path = new List<List<int>>();

            Node currentNode = endNode;
            while(currentNode != null){
                List<int> nodeData = new List<int>();
                nodeData.Add(currentNode.Row);
                nodeData.Add(currentNode.Col);
                path.Add(nodeData);
                currentNode = currentNode.CameFrom;
            }
            int[,] result = new int[path.Count,2];
            for(int i=0;i<path.Count; i++){
                List<int> lst = path[path.Count-1-i];
                result[i,0]= lst[0];
                result[i,1] = lst[1];
            }
            return result;
        }

        private static int CalculateManhattanDistance(Node currentNode, Node endNode)
        {
            int currentRow = currentNode.Row;
            int currentCol = currentNode.Col;
            int endRow = endNode.Row;
            int endCol = endNode.Col;

            return Math.Abs(currentRow-endRow)+Math.Abs(currentCol-endCol);
        }

        private static List<List<Node>> InitializeNodes(int[,] graph)
        {
            List<List<Node>> nodes= new List<List<Node>>();
            for(int row=0; row< graph.GetLength(0); row++){
                List<Node> nodeList = new List<Node>();
                for(int col=0; col< graph.GetLength(1); col++){
                    nodeList.Add(new Node(row, col,graph[row,col]));
                }
                nodes.Add(nodeList);

            }
            return nodes;

        }
        //https://www.algoexpert.io/questions/river-sizes
        public static List<int> FindRiverSizes(int[,] matrix){
            //T:O(w*h) | S:O(w*h)
            List<int> sizes = new List<int>();
            bool[,] visited = new bool[matrix.GetLength(0), matrix.GetLength(1)];
            for(int row=0; row<matrix.GetLength(0); row++){

                for(int col=0; col<matrix.GetLength(1); col++){
                    if(visited[row,col])
                        continue;
                    TraverseNode(row, col, visited, matrix,sizes);
                }
            }
            return sizes;
        }

        private static void TraverseNode(int row, int col, bool[,] visited, int[,] matrix, List<int> sizes)
        {
            int currentRiverSize =0;

            Stack<int[]> nodesToExplore = new Stack<int[]>();
            nodesToExplore.Push(new int[]{row, col});
            while(nodesToExplore.Count > 0){
                int[] currentNode = nodesToExplore.Pop();
                row = currentNode[0];
                col = currentNode[1];
                if(visited[row,col])
                    continue;
                
                visited[row, col]= true;
                if(matrix[row, col] == 0)
                    continue;
                currentRiverSize++;

                List<int[]> unVisitedNeighbors = GetUnVisitedNeighbors(row, col,matrix,visited);
                foreach(int[] unVisitedNeigh in unVisitedNeighbors){
                    nodesToExplore.Push(unVisitedNeigh);
                }
            }
            if(currentRiverSize >0 )
                sizes.Append(currentRiverSize);

        }

        private static List<int[]> GetUnVisitedNeighbors(int row, int col, int[,] matrix, bool[,] visited)
        {
            List<int[]> unVisitedNeighbors = new List<int[]>();

            if(row >0 && !visited[row-1, col])
                unVisitedNeighbors.Add(new int[]{row-1, col});
            
            if( row < matrix.GetLength(0)-1 && !visited[row+1, col])
                unVisitedNeighbors.Add(new int[]{row+1, col});

            if(col >0 && !visited[row, col-1])
                unVisitedNeighbors.Add(new int[]{row, col-1});
            
            if( col < matrix.GetLength(1)-1 && !visited[row, col+1])
                unVisitedNeighbors.Add(new int[]{row, col+1});

            return unVisitedNeighbors;
        }

        //https://www.algoexpert.io/questions/spiral-traverse
        public static List<int> SpiralTraverse(int[,] array){
            
            if(array.GetLength(0) ==0) return new List<int>();            
            
            //1. Iterative - T:O(n) | S:O(n)
            var result = SpiralTraverseIterative(array);

            //2. Recursive - T:O(n) | S:O(n)
            SpiralTraverseRecursion(array, 0, array.GetLength(0)-1, 0, array.GetLength(1)-1, result);
            return result;
        }

        private static void SpiralTraverseRecursion(int[,] array, int startRow, int endRow, int startCol, int endCol, List<int> result)
        {
            if(startRow > endRow || startCol > endCol) return;

            //TOP
            for(int col=startCol; col<=endCol; col++){
                result.Add(array[startRow, col]);
            }

            //Right
            for(int row=startRow+1; row<=endRow; row++){
                result.Add(array[row, endCol]);
            }

            //Bottom 
            for(int col=endCol-1; col>=startCol; col++){

                //Single Row edge case
                if(startRow == endRow) break;

                result.Add(array[endRow, col]);
            }

            //Left
            for(int row=endRow-1; row > startRow; row++){

                //Single column edge case
                if(startCol == endCol) break;                

                result.Add(array[row,startCol]);
            }

            SpiralTraverseRecursion(array,startRow++, endRow--, startCol++, endCol--, result);
        }

        private static List<int> SpiralTraverseIterative(int[,] array)
        {   
            List<int> result = new List<int>();
            
            var startRow =0;
            var endRow = array.GetLength(0)-1;
            var startCol=0;
            var endCol = array.GetLength(1)-1;

            while(startRow <=endRow && startCol <= endCol){
                
                //Top(Left->Right)
                for(int col=startCol; col<=endCol; col++){
                    result.Add(array[startRow, col]);                    
                }

                //Right (Top to Bottom)
                for(int row=startRow+1; row<=endRow; row++){
                    result.Add(array[row, endCol]);
                }

                //Bottom (Right -> Left)
                for(int col=endCol-1; col>=startCol; col--){

                    //Single Row edge case
                    if(startRow == endRow) break;
                    
                    result.Add(array[endRow, col]);
                }

                //Left (Bottom to Top)
                for(int row =endRow-1; row > startRow; row--){

                    //Single Column Edge code
                    if(startCol == endCol) break;

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
        public static int MinimumPassesOfMatrix(int[][] matrix){
            int passes = ConvertNegatives(matrix);

            return (!ContainsNegatives(matrix))?passes:-1;
        }

        private static bool ContainsNegatives(int[][] matrix)
        {
            foreach(var row in matrix){
                foreach(var val in row){
                    if (val <0)
                    return true;
                }
            }
            return false;
        }

        private static int ConvertNegatives(int[][] matrix)
        {
            Queue<Pos> posQ = GetAllPositivePositions(matrix);
            int passes=0;
            int size = posQ.Count();
            while(posQ.Count>0){
                Pos curPos = posQ.Dequeue();
                size--;
                List<int[]> adjacentPos = GetAdjacentPositions(matrix, curPos);
                foreach(var pos in adjacentPos){
                   int row = pos[0], col=pos[1];
                   int val = matrix[row][col];
                   if(val < 0) {
                        matrix[row][col] = val*-1;
                        posQ.Enqueue(new Pos{Row=row, Col=col});     
                   }
                   
                }
                if(size == 0){
                    size =posQ.Count();
                    if(size > 0)
                        passes++;
                    
                }

            }
            return passes;
        }

        private static List<int[]> GetAdjacentPositions(int[][] matrix, Pos pos)
        {
            List<int[]> adjPos = new List<int[]>();
            int row= pos.Row;
            int col=pos.Col;

            //https://www.tutorialsteacher.com/csharp/csharp-multi-dimensional-array
            //var twoDArr = new int[,] {{1,2},{2,3}};
            //https://www.tutorialsteacher.com/csharp/csharp-jagged-array
            //var jogArr = new int[][] {new int[3]{0, 1, 2}, new int[5]{1,2,3,4,5}};
           
           //Top
            if(row> 0)
                adjPos.Add(new int[]{row-1, col});

            //Bottom/Down
            if(row < matrix.Length-1){
                adjPos.Add(new int[]{row+1, col});
            }
            
            //Left
            if(col >0)
                adjPos.Add(new int[]{row, col-1});

            //Right
            if(col < matrix[0].Length-1)
                adjPos.Add(new int[]{row, col+1});



            return adjPos;
        }

        private static Queue<Pos> GetAllPositivePositions(int[][] matrix)
        {
            Queue<Pos> positivePos= new Queue<Pos>();

            for(int row=0; row < matrix.Length; row++){
                for(int col=0; col<matrix[row].Length; col++){
                    int val= matrix[row][col];
                    if(val>0)
                        positivePos.Enqueue(new Pos(){Row=row, Col=col});
                }
            }
            return positivePos;
        }

        struct Pos{
            public int Row, Col;

        }

        //https://www.algoexpert.io/questions/rectangle-mania
        static string UP ="up";
        static string RIGHT = "right";
        static string DOWN = "down";
        static string LEFT = "left";
        
        public static int RectangleMania(List<int[]> coords){

            Dictionary<string, Dictionary<string, List<int[]>>> coordsTable; 
            int rectangleCount=0;
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
            int rectangleCount =0;

            foreach(var coord1 in coords){
                foreach(var coord2 in coords){
                    if(!IsInUpperRight(coord1, coord2)) continue;
                    string upperCoordString = CoordsToString(new int[]{coord1[0], coord2[1]});
                    string bottomRightCoordString = CoordsToString(new int[]{coord2[0],coord1[1]});
                    if(coordsSet.Contains(upperCoordString) && coordsSet.Contains(bottomRightCoordString))
                        rectangleCount++;
                }
            }
            return rectangleCount;
        }

        private static bool IsInUpperRight(int[] coord1, int[] coord2)
        {
            return coord2[0]>coord1[0] && coord2[1]> coord2[1];
        }

        private static HashSet<string> GetCoordsSet(List<int[]> coords)
        {
            HashSet<string> coordsSet = new HashSet<string>();
            foreach(var coord in coords){
                string coordString = CoordsToString(coord);
                coordsSet.Add(coordString);
            }

            return coordsSet;
        }

        private static int GetRectangleCount(List<int[]> coords, Dictionary<string, Dictionary<string, List<int[]>>> coordsTable)
        {
            int rectangleCount =0;
            foreach(var coord in coords)
                rectangleCount+= ClockWiseCountRectangles(coord,coordsTable,UP, coord);

            return rectangleCount;
        }

        private static int ClockWiseCountRectangles(int[] coord, Dictionary<string, Dictionary<string, List<int[]>>> coordsTable, string direction, int[] origin)
        {
            string coordString = CoordsToString(coord);

            if(direction == LEFT){
                bool rectangleFound = coordsTable[coordString][LEFT].Contains(origin);
                return rectangleFound?1:0;
            }else{
                int rectangleCount =0;
                string nextDirection = GetNextClockWiseDirection(direction);
                foreach(var nextCoord in coordsTable[coordString][direction]){
                    rectangleCount += ClockWiseCountRectangles(nextCoord, coordsTable, nextDirection, origin);
                }
                return rectangleCount;
            }            
            
        }

        private static string GetNextClockWiseDirection(string direction)
        {
            if(direction == UP) return RIGHT;
            if(direction == RIGHT) return DOWN;
            if(direction == DOWN) return LEFT;

            return "";
        }

        private static Dictionary<string, Dictionary<string, List<int[]>>> GetCoordsTable(List<int[]> coords)
        {
            
            Dictionary<string, Dictionary<string, List<int[]>>> coordsTable = new Dictionary<string, Dictionary<string, List<int[]>>>();
            
            foreach(int[] coord1 in coords){
                
                Dictionary<string, List<int[]>> coord1Directions = new Dictionary<string, List<int[]>>();
                coord1Directions[UP] = new List<int[]>();
                coord1Directions[DOWN] = new List<int[]>();
                coord1Directions[RIGHT] = new List<int[]>();
                coord1Directions[LEFT] = new List<int[]>();

                foreach(var coord2 in coords){

                    string coord2Direction = GetCoordDirection(coord1, coord2);
                    
                    if(coord1Directions.ContainsKey(coord2Direction))
                        coord1Directions[coord2Direction].Add(coord2);
                    
                }
                string coord1String = CoordsToString(coord1);
                coordsTable[coord1String] = coord1Directions;

            }

            return coordsTable;
        }

        private static string CoordsToString(int[] coord)
        {
            return coord[0].ToString() +"-"+coord[1].ToString();
        }

        private static string GetCoordDirection(int[] coord1, int[] coord2)
        {
            if(coord2[1] == coord1[1]){
                if(coord2[0]> coord1[0]){
                    return RIGHT;
                }else if (coord2[0]<coord1[0]){
                    return LEFT;
                }
            }else if (coord2[0] == coord1[0]){
                if(coord2[1] > coord1[1]){
                    return UP;
                }
                else if (coord2[1] < coord1[1]){
                    return DOWN;
                }
            }
            return "";
        }

        //https://www.algoexpert.io/questions/tournament-winner
        public static string TournamentWinner(List<List<string>> competitions, List<int> results){
            //T:O(n) | S:O(k) where n is number of competitions and k is number of teams.
            Dictionary<string,int> teamScores = new Dictionary<string, int>();
            string maxScoreTeam="";
            for(int i=0; i< results.Count; i++){
                
                int winner = results[i];
                string currentWinningTeam = competitions[i][1-winner];
                if(!teamScores.ContainsKey(currentWinningTeam))
                    teamScores[currentWinningTeam]=0;
                
                teamScores[currentWinningTeam]+=3;
                if(string.IsNullOrEmpty(maxScoreTeam))
                    maxScoreTeam = currentWinningTeam;
                else if(teamScores[maxScoreTeam]< teamScores[currentWinningTeam])
                    maxScoreTeam = currentWinningTeam;


            }
            return maxScoreTeam;

        }
        //https://www.algoexpert.io/questions/square-of-zeroes
        public static bool SquareOfZeroes(List<List<int>> matrix){

            //1.Naive Iterative - T:n^4 |S:O(1) where n is height and width of matrix
            bool IsSquarOfZeroesExists = SquareOfZeroesNaiveIterative (matrix); 

            //2.Optimal(Precompute) Iterative - T:(n^3) |S:O(n^2) where n is height and width of matrix
            IsSquarOfZeroesExists = SquareOfZeroesOptimalIterative (matrix); 

            //3.Naive Recursive without Caching - T:(n^4) |S:O(n^3) where n is height and width of matrix
            
            //4.Optimal Recursive with Caching - T:(n^4) |S:O(n^3) where n is height and width of matrix
            IsSquarOfZeroesExists = SquareOfZeroesOptimalRecursive (matrix); 
            //5.Optimal Recursive with Caching & PreCompute - T:(n^3) |S:O(n^3) where n is height and width of matrix
            TODO:
            return IsSquarOfZeroesExists;
        }

        private static bool SquareOfZeroesOptimalRecursive(List<List<int>> matrix)
        {
            int lastIdx = matrix.Count-1;
            Dictionary<string, bool> cache = new Dictionary<string, bool>();
            return HasSquareOfZeroes(matrix, 0, 0, lastIdx, lastIdx, cache);

        }

        private static bool HasSquareOfZeroes(List<List<int>> matrix, int topRow, int leftCol, int bottomRow, int rightCol, Dictionary<string, bool> cache)
        {
            if(topRow >= bottomRow || leftCol >= rightCol) return false;
            string key = topRow.ToString() +'-'+ leftCol.ToString() +'-'+ bottomRow.ToString() +'-'+ rightCol.ToString();
            if(cache.ContainsKey(key)) return cache[key];

            cache[key] = IsSquareOfZeroes(matrix, topRow, leftCol,bottomRow, rightCol) || 
                         HasSquareOfZeroes(matrix, topRow+1, leftCol+1, bottomRow-1, rightCol-1, cache) ||
                         HasSquareOfZeroes(matrix,topRow,leftCol+1,bottomRow-1,rightCol, cache) ||
                         HasSquareOfZeroes(matrix, topRow+1, leftCol, bottomRow, rightCol-1,cache) ||
                         HasSquareOfZeroes(matrix, topRow+1,leftCol+1, bottomRow, rightCol, cache) ||
                         HasSquareOfZeroes(matrix, topRow,leftCol, bottomRow-1, rightCol-1, cache);

            return cache[key];
        }

        private static bool SquareOfZeroesOptimalIterative(List<List<int>> matrix)
        {
            List<List<InfoMatrixItem>> infoMatrix = PreComputeNumOfZeroes(matrix);
            int n= matrix.Count;
            for(int topRow=0; topRow < n; topRow++){
                for(int leftCol=0; leftCol<n; leftCol++){
                    int squareLen = 2;
                    while(squareLen <=n-leftCol && squareLen <=n-topRow){
                        int bottomRow = topRow+squareLen-1;
                        int rightCol = leftCol+squareLen-1;
                        if(IsSquareOfZeroes(infoMatrix, topRow, leftCol, bottomRow, rightCol)){
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
            int squareLen = rightCol-leftCol+1;
            bool hasTopBorder = infoMatrix[topRow][leftCol].NumberZeroesRight >= squareLen;
            bool hasLeftBorder = infoMatrix[topRow][leftCol].NumberZeroesBelow >= squareLen;
            bool hasBottomBorder = infoMatrix[bottomRow][leftCol].NumberZeroesRight >= squareLen;
            bool hasRightBorder = infoMatrix[topRow][rightCol].NumberZeroesBelow >= squareLen;

            return hasBottomBorder && hasLeftBorder && hasRightBorder && hasTopBorder;
        }

        private static List<List<InfoMatrixItem>> PreComputeNumOfZeroes(List<List<int>> matrix)
        {
            List<List<InfoMatrixItem>> infoMatrix = new List<List<InfoMatrixItem>>();
            for(int row=0; row < matrix.Count; row++){
                List<InfoMatrixItem> inner = new List<InfoMatrixItem>();
                for (int col=0; col < matrix[row].Count; col++){
                    int numZeroes = matrix[row][col] ==0 ?1:0;
                    inner.Add(new InfoMatrixItem(numZeroes, numZeroes));
                }
                infoMatrix.Add(inner);
            }
            int lastIdx= infoMatrix.Count-1;
            for(int row=lastIdx; row >=0; row--){
                for(int col=lastIdx; col>=0; col--){
                    if(matrix[row][col] == 1) continue;
                    if(row < lastIdx){
                        infoMatrix[row][col].NumberZeroesBelow += infoMatrix[row+1][col].NumberZeroesBelow;
                    }
                    if(col < lastIdx){
                        infoMatrix[row][col].NumberZeroesRight += infoMatrix[row][col+1].NumberZeroesRight;
                    }
                }
            }
            return infoMatrix;
        }
        internal class InfoMatrixItem
        {
            public int NumberZeroesBelow{get;set;}
            public int NumberZeroesRight{get;set;}

            public InfoMatrixItem(int numZeroesBelow, int numZeroesRight){
                this.NumberZeroesBelow = numZeroesBelow;
                this.NumberZeroesRight = numZeroesRight;
            }
        }
        private static bool SquareOfZeroesNaiveIterative(List<List<int>> matrix)
        {
            int n=matrix.Count;
            for(int topRow=0; topRow < n; topRow++){
                for(int leftCol=0; leftCol<n; leftCol++){
                    int squareLen =2;
                    while(squareLen <= n-leftCol && squareLen <=n-topRow){
                        int bottomRow = topRow+squareLen-1;
                        int rightCol = leftCol+squareLen-1;
                        if(IsSquareOfZeroes(matrix,topRow,leftCol,bottomRow, rightCol))
                            return true;
                        squareLen++;
                    }
                }
            }
            return false;
        }

        private static bool IsSquareOfZeroes(List<List<int>> matrix, int topRow, int leftCol, int bottomRow, int rightCol)
        {
            for(int row=topRow; row<bottomRow+1; row++){
                if(matrix[row][leftCol] !=0 || matrix[row][rightCol] != 0) return false;
            }
            for(int col=leftCol; col<rightCol+1; col++){
            
                if(matrix[topRow][col] != 0 || matrix[bottomRow][col] != 0) return false;
            }
            return true;
        }      

        //https://www.algoexpert.io/questions/knapsack-problem
        public static List<List<int>> KnapsackProblem(int[,] items, int capacity){
            //T:O(nc) | S:O(nc)
            int[,] knapsackValues = new int[items.GetLength(0)+1, capacity+1];
            for(int row=1; row<items.GetLength(0)+1; row++){
                int currentWeigtht = items[row-1,1];
                int currentValue = items[row-1,0];
                for(int col=0; col< capacity+1; col++){
                    if(currentWeigtht > col){
                        knapsackValues[row, col] = knapsackValues[row-1, col];
                    }else{
                        knapsackValues[row, col]= Math.Max(knapsackValues[row-1,col],
                                                            knapsackValues[row-1,col-currentWeigtht]+currentValue);
                                                        
                    }
                }                
            }
            return GetKnapsackItems(knapsackValues, items, knapsackValues[items.GetLength(0),capacity]);
        }

        private static List<List<int>> GetKnapsackItems(int[,] knapsackValues, int[,] items, int weight)
        {
            List<List<int>> sequence = new List<List<int>>();

            List<int> totalWeight = new List<int>();
            sequence.Add(totalWeight);
            sequence.Add(new List<int>());
            int row=knapsackValues.GetLength(0)-1;
            int col= knapsackValues.GetLength(1)-1;
            while(row>0){
                if(knapsackValues[row,col] == knapsackValues[row-1, col])
                    row--;
                else{
                    sequence[1].Insert(0,row-1);
                    col -= items[row-1, 1];
                    row--;

                }
                if(col == 0) break;
            }

            return sequence;
        }
        //https://www.algoexpert.io/questions/maximum-sum-submatrix
        public static int MaximumSumSubmatrix(int[,] matrix, int size){

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
            for(int row=size-1; row< matrix.GetLength(0); row++ ){
                for(int col=size-1; col< matrix.GetLength(1); col++)
                {
                    int total = sums[row, col];

                    bool touchesTopBorder = (row -size < 0);
                    if(!touchesTopBorder)
                        total -= sums[row-size, col];
                    
                    bool touchesLeftBorder = (col-size <0);
                    if(!touchesLeftBorder){
                        total -=sums[row, col-size];
                    }

                    bool touchesTopOrLeftBorder =(touchesTopBorder || touchesLeftBorder);
                    if(!touchesTopOrLeftBorder)
                        total += sums[row-size, col-size];
                    
                    maxSubMatrixSum = Math.Max(maxSubMatrixSum , total);

                }
            }
            return maxSubMatrixSum;
        }

        private static int[,] PrecomputeSumMatrix(int[,] matrix)
        {
            int[,] sums = new int[matrix.GetLength(0), matrix.GetLength(1)];
            sums[0,0] = matrix[0,0];

            //Fill first row
            for(int idx=1; idx< matrix.GetLength(1); idx++){
                sums[0,idx]= sums[0, idx-1]+matrix[0,idx];
            }
            //Fill first column
            for(int idx=1; idx< matrix.GetLength(0); idx++ ){
                sums[idx,0] = sums[idx-1,0]+matrix[idx,0];                
            }

            //Fill in the rest of matrix
            for(int row=1; row < matrix.GetLength(0); row++){
                for(int col=1; col< matrix.GetLength(1); col++)
                {
                    sums[row, col]= sums[row-1,col]+sums[row, col-1]-sums[row-1, col-1]+matrix[row, col];
                }
            }
            return sums;
        }
        //https://www.algoexpert.io/questions/search-in-sorted-matrix
        public static int[] SearchInMatrix(int[,] matrix, int target){
            //1.Naive - pair of loops
            //T:O(n^2) | S:O(1)

            //2. Leveraging facts that rows and columns of matrix sorted

            int row =0;
            int col= matrix.GetLength(1)-1;
            while(row < matrix.GetLength(0) && col >=0){

                if(matrix [row, col] > target){
                    col--;
                }else if(matrix[row, col] < target){
                    row++;
                }else{
                    return new int[]{row, col};
                }
            }
            return new int[]{-1, -1};
        }
        //https://www.algoexpert.io/questions/minimum-area-rectangle
        public static int MinimumAreaRectangle(int[][] points){

            if(points.Length < 4) return 0;
            //1.Naive - 4 pair/nested of loops to generate all possible combinations of 4 points and find minimum area among rectangles found
            //T:O(n^4) | S:O(1)

            //2.Optimal - edge pairing algo- find parallel points vertically and horizontally to see if they can form rectangles
            //T:O(n^2) | S:O(n) - n is number of points
            int minAreaRect = MinimumAreaRectangleOptimal(points);
            
            //3.Optimal - simplified -find two opposite end points and try to match them with any two points to see if they can form rectangles
            //T:O(n^2) | S:O(n) - n is number of points
            minAreaRect = MinimumAreaRectangleOptima2(points);

            return minAreaRect != Int32.MinValue ? minAreaRect :0;
            
        }

        private static int MinimumAreaRectangleOptima2(int[][] points)
        {
            HashSet<string> pointSet =CreatePointSet(points);
            int minAreaRect = Int32.MaxValue;
            for(int curIdx=0; curIdx < points.Length; curIdx++){

                int p2x = points[curIdx][0];
                int p2y = points[curIdx][1];
                
                for(int prevIdx=0; prevIdx < curIdx; prevIdx++){
    
                    int p1x = points[prevIdx][0];
                    int p1y = points[prevIdx][1];
                
                    bool pointsShareValue = p1x == p2x || p2y == p1y;
                    if(pointsShareValue) continue;

                    bool point1OnOppositDirectionExists = pointSet.Contains(ConvertPointToString(p1x,p2y));
                    bool point2OnOppositDirectionExists = pointSet.Contains(ConvertPointToString(p2x,p1y));

                    bool oppositeDiagonalExists = point1OnOppositDirectionExists && point2OnOppositDirectionExists;

                    if(oppositeDiagonalExists){
                        int curArea = Math.Abs(p1x-p2x) * Math.Abs(p1y-p2y);
                        minAreaRect = Math.Min(minAreaRect, curArea);
                    }
                }
                
            }
            return minAreaRect;
        }

        private static string ConvertPointToString(int x, int y)
        {
            return x.ToString()+":"+y.ToString();
        }

        private static HashSet<string> CreatePointSet(int[][] points)
        {
            HashSet<string> pointSet = new HashSet<string>();
            foreach(var point in points){
                int x = point[0];
                int y= point[1];
                string pointStr = x.ToString()+"-"+y.ToString();
                pointSet.Add(pointStr);
            }
            return pointSet;
        }

        private static int MinimumAreaRectangleOptimal(int[][] points)
        {
            Dictionary<int, int[]> columns = InitializeColumns(points);
            int minAreaRect= Int32.MaxValue;
            Dictionary<string, int> edgesParallelToYAxis = new Dictionary<string, int>();
            List<int> sortedColumns = new List<int>(columns.Keys);
            sortedColumns.Sort();

            foreach(var x in sortedColumns){
                int[] yValuesInCurrentColumn = columns[x];
                Array.Sort(yValuesInCurrentColumn);

                for(int curIdx=0; curIdx < yValuesInCurrentColumn.Length; curIdx++){
                    int y2 = yValuesInCurrentColumn[curIdx];
                    for(int prevIdx=0; prevIdx< curIdx; prevIdx++){
                        int y1= yValuesInCurrentColumn[prevIdx];
                        string pointString = y1.ToString()+":"+y2.ToString();

                        if(edgesParallelToYAxis.ContainsKey(pointString)){
                            int currArea = (x - edgesParallelToYAxis[pointString] * y2-y1);
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
            
            foreach(var point in points){
                int x = point[0];
                int y = point[1];

                if(!columns.ContainsKey(x)){
                    columns[x]= new int[]{};
                }
                int[] column = columns[x];
                int[] newColumn = new int[column.Length+1];
                for(int i=0; i< column.Length; i++){
                    newColumn[i] = column[i];
                }
                newColumn[column.Length]=y;
                columns[x] = newColumn;
            }
            return columns;
        }
        //https://www.algoexpert.io/questions/number-of-ways-to-traverse-graph
        public static int NumberOfWaysToTraverseGraph(int width, int height){
            
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
            int xDistinaceToCorner = width -1;
            int yDistanceToCorner = height-1;

            //The number of permutations of right and down movements 
                                            //is the number of ways to reach the bottom right corner
            //(n+r)!/n!*r!
            int numerator = Factorial(xDistinaceToCorner+yDistanceToCorner);
            int denominator = Factorial(xDistinaceToCorner) * Factorial (yDistanceToCorner);

            return numerator/denominator;
        }

        private static int Factorial(int num)
        {
            int result =1;
            for(int i=2; i<=num; i++)
                result *=i;
            
            return result;
        }   

        private static int NumberOfWaysToTraverseGraphOptimal1(int width, int height)
        {   
            //Heigh =>Row And Width =>Column
            int[,] numberOfWays = new int[height+1, width+1];

            for(int widthIdx=1; widthIdx < width+1; widthIdx++){

                for(int heightIdx =1; heightIdx < height+1; heightIdx++){

                    if(widthIdx == 1 || heightIdx == 1){
                        numberOfWays[heightIdx, widthIdx] = 1;
                    }
                    else{
                        int waysLeft = numberOfWays[heightIdx, widthIdx-1];
                        int waysRight = numberOfWays[heightIdx-1, widthIdx];
                        numberOfWays[heightIdx, widthIdx]= waysLeft + waysRight;

                    }

                }
            }
            return numberOfWays[height, width];
        }

        private static int NumberOfWaysToTraverseGraphNaive(int width, int height)
        {
            if(width == 1 || height ==1) return 1;

            return NumberOfWaysToTraverseGraphNaive(width-1, height)+
                    NumberOfWaysToTraverseGraphNaive(width, height-1);
                
        }
        //https://www.algoexpert.io/questions/two-colorable
        public static bool TwoColorables(int[][] edges){

            //T:O(v+e) | S:O(v)
            int[] colors = new int[edges.Length];
            colors[0]=1;
            Stack<int> stack = new Stack<int>();
            stack.Push(0);

            while(stack.Count > 0){
                int node = stack.Pop();
                foreach(var connection in edges[node]){
                    if(colors[connection] ==0){
                        colors[connection] = colors[node]==1?2:1;
                        stack.Push(connection);
                    }else if(colors[connection] == colors[node])
                        return false;
                }
            }
            return true;

        }
    }

   
}