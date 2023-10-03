using System.ComponentModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;
using System.Security.AccessControl;
using System.Data;
using System.Text.RegularExpressions;

namespace AlgoDSPlay
{
    public class MatrixOps
    {


        //https://www.algoexpert.io/questions/boggle-board
        public static List<string> BoggleBoard(char[,] board, string[] words){
            //T:O(WS+MN*8^S) | S:O(WS+MN)
            Trie trie = new Trie();
            foreach(var word in words){
                trie.Insert(word);
            }
            HashSet<string> finalWords = new HashSet<string>();
            bool[,] visited = new bool[board.GetLength(0), board.GetLength(1)];
            for(int row=0; row< board.GetLength(0); row++){
                for(int col=0; col < board.GetLength(1); col++){
                    Explore(row, col, board, trie.root, visited, finalWords);
                }
            }
            List<string> finalWordsArray = new List<string>();
            foreach(string key in finalWords){
                finalWordsArray.Add(key);
            }
            return finalWordsArray;

        }

        private static void Explore(int row, int col, char[,] board, TrieNode trieNode, bool[,] visited, HashSet<string> finalWords)
        {
            if(visited[row, col]) return;

            char letter = board[row, col];
            if(!trieNode.children.ContainsKey(letter)) return;
            visited[row, col]= true;

            trieNode = trieNode.children[letter];
            if(trieNode.children.ContainsKey('*')){  //endSymbol checking
                finalWords.Add(trieNode.word);
            }

            List<int[]> neighbors = GetNeighbors(row, col,board);
            foreach(int[] neighbor in neighbors){
                Explore(neighbor[0], neighbor[1], board, trieNode,visited, finalWords);
            }
            visited[row, col]=false;

        }

        public static List<int[]> GetNeighbors(int row, int col, char[,] board){
            
            List<int[]> neighbors = new List<int[]>();

            //Top-Left Diagonal 
            if(row >0 && col > 0)
                neighbors.Add(new int[]{row-1, col-1});
            //Top
            if(row >0)
                neighbors.Add(new int[]{row-1, col});
            //Top-Right Diagonal
            if(row >0 && col < board.GetLength(1)-1)
                neighbors.Add(new int[]{row-1, col+1});
            //Right
            if(col < board.GetLength(1)-1)
                neighbors.Add(new int[]{row, col+1});
            //Down-Right Diagonal
            if(row > board.GetLength(0)-1 && col < board.GetLength(1)-1)
                neighbors.Add(new int[]{row+1, col+1});
            //Down
            if(row > board.GetLength(0)-1)
                neighbors.Add(new int[]{row+1, col});
            //Down-Left Diagonal
            if(row > board.GetLength(0)-1 && col >0)
                neighbors.Add(new int[]{row+1, col-1});
            //Left
            if(col>0)
                neighbors.Add(new int[]{row, col-1});

            return neighbors;
        }

        //https://www.algoexpert.io/questions/waterfall-streams
        public double[] WaterfallStreams(double[][] array, int source){
            //T:O(W^2*h) | S:O(W)
            double[] rowAbove =array[0];

            rowAbove[source] = -1;

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
            
            ////T:O(n) | S:O(n)
            
            if(array.GetLength(0) ==0) return new List<int>();            
            
            var result = SpiralTraverseIterative(array);
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
    }
}