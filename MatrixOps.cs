using System.ComponentModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

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
            if(trieNode.children.ContainsKey('*')){
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
                    
            MinHeapForAStarAlgo nodesToVisit = new MinHeapForAStarAlg(nodesToVisitList);

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
    }
}