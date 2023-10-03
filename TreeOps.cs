using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class TreeOps
    {

        //https://www.algoexpert.io/questions/find-nodes-distance-k

        public static List<int> FindNodesDistanceK(BinaryTree tree, int target, int k){
            //1.BFS - T:O(n) | S:O(n)
            List<int> nodesWithDistanceK= FindNodesDistanceKBFS(tree, target,k);

            //2.DFS - T:O(n) | S:O(n)
            FindNodesDistanceKDFS(tree, target,k, nodesWithDistanceK);
            return nodesWithDistanceK;
        }

        private static int FindNodesDistanceKDFS(BinaryTree node, int target, int k, List<int> nodeDistanceK)
        {
            if(node == null) return -1;
            if(node.Value == target){
                AddSubtreeNodesAtDistanceK(node, 0, k, nodeDistanceK);
                return 1;
            }

            int leftDistance = FindNodesDistanceKDFS(node.Left, target, k, nodeDistanceK);
            int rightDistance = FindNodesDistanceKDFS(node.Right, target,k, nodeDistanceK);

            if(leftDistance == k || rightDistance == k) nodeDistanceK.Add(node.Value);

            if(leftDistance != -1){
                AddSubtreeNodesAtDistanceK(node.Right, leftDistance+1, k, nodeDistanceK);
                return leftDistance+1;
            }
            if(rightDistance != -1){
                AddSubtreeNodesAtDistanceK(node.Left, rightDistance+1, k, nodeDistanceK);
                return rightDistance+1;
            }

            return -1;
            
        }

        private static void AddSubtreeNodesAtDistanceK(BinaryTree node, int distance, int k, List<int> nodeDistanceK)
        {
            if(node == null) return;
            if(distance == k){
                nodeDistanceK.Add(node.Value);
            }else{
                AddSubtreeNodesAtDistanceK(node.Left, distance+1, k, nodeDistanceK);
                AddSubtreeNodesAtDistanceK(node.Right, distance+1, k, nodeDistanceK);
            }
        }

        private static List<int> FindNodesDistanceKBFS(BinaryTree tree, int target, int k)
        {
            Dictionary<int, BinaryTree> nodesToParents = new Dictionary<int, BinaryTree>();
            PopulateNodesToParents(tree, nodesToParents, null);
            BinaryTree targetNode = GetTargetNodeFromValue(target, tree, nodesToParents);
            return FindNodesDistanceKBFS(targetNode, nodesToParents, k);
            
        }

        private static List<int> FindNodesDistanceKBFS(BinaryTree targetNode, Dictionary<int, BinaryTree> nodesToParents, int k)
        {
            Queue<Tuple<BinaryTree, int>> queue = new Queue<Tuple<BinaryTree, int>>();
            queue.Enqueue(new Tuple<BinaryTree, int>(targetNode,0));

            HashSet<int> seen = new HashSet<int>();
            seen.Add(targetNode.Value);

            while(queue.Count >0){
                Tuple<BinaryTree, int> vals = queue.Dequeue();
                BinaryTree currentNode = vals.Item1;
                int distanceFromTarget = vals.Item2;

                if(distanceFromTarget == k){
                    List<int> nodeDistnaceK = new List<int>();
                    foreach(var pair in queue){
                        nodeDistnaceK.Add(pair.Item1.Value);
                    }
                    nodeDistnaceK.Add(currentNode.Value);
                    return nodeDistnaceK;
                }

                List<BinaryTree> connectedNodes = new List<BinaryTree>();
                connectedNodes.Add(currentNode.Left);
                connectedNodes.Add(currentNode.Right);
                connectedNodes.Add(nodesToParents[currentNode.Value]);

                foreach(var node in connectedNodes){
                    if(node == null) continue;

                    if(seen.Contains(node.Value)) continue;

                    seen.Add(node.Value);
                    queue.Enqueue(new Tuple<BinaryTree, int>(node, distanceFromTarget+1));
                }

            }
            return new List<int>();

        }

        private static BinaryTree GetTargetNodeFromValue(int target,BinaryTree tree, Dictionary<int, BinaryTree> nodesToParents){
            if(tree.Value == target) return tree;

            BinaryTree nodeParent = nodesToParents[target];
            if(nodeParent.Left !=null && nodeParent.Left.Value == target) return nodeParent.Left;

            return nodeParent.Right;

        }
        private static void PopulateNodesToParents(BinaryTree node, Dictionary<int, BinaryTree> nodesToParents, BinaryTree parent)
        {
            if(node != null){
                nodesToParents[node.Value]= parent;
                PopulateNodesToParents(node.Left, nodesToParents, node);
                PopulateNodesToParents(node.Right, nodesToParents, node);
            }
            
        }

        //https://www.algoexpert.io/questions/find-successor
        public static BinaryTree FindSuccessor(BinaryTree tree, BinaryTree node){

            //1- T:O(n) | S:O(n) 
            BinaryTree successorNode = FindSuccessorUsingInOrderTraversalOrder(tree, node);

            //2: T:O(h) | S:O(1)
             successorNode = FindSuccessorUsingParentPointer(tree, node);

             return successorNode;
        }

        private static BinaryTree FindSuccessorUsingParentPointer(BinaryTree tree, BinaryTree node)
        {
            if(node.Right != null) return GetLeftMostChild(node.Right);
            return GetRightMostParent(node);
            
        }

        private static BinaryTree GetLeftMostChild(BinaryTree node){
            BinaryTree currentNode = node;
            while(currentNode.Left != null)
                currentNode = currentNode.Left;
            return currentNode;

        }
        private static BinaryTree GetRightMostParent(BinaryTree node){
            BinaryTree currentNode = node;

            while(currentNode.Parent != null && currentNode.Parent.Right == currentNode)
                currentNode = currentNode.Parent;
            
            return currentNode.Parent;
        }
        private static BinaryTree FindSuccessorUsingInOrderTraversalOrder(BinaryTree tree, BinaryTree node)
        {
            List<BinaryTree> inOrderTraversalOrder = new List<BinaryTree>();
            GetInOrdderTraversalOrder(tree, inOrderTraversalOrder);
            for(int i=0;i< inOrderTraversalOrder.Count; i++){
                BinaryTree currentNode = inOrderTraversalOrder[i];
                
                if(currentNode != node) continue;

                if(i == inOrderTraversalOrder.Count - 1)
                    return null;
                
                return inOrderTraversalOrder[i+1];
            }
            return null;
        }

        private static void GetInOrdderTraversalOrder(BinaryTree node, List<BinaryTree> inOrderTraversalOrder){
            if(node == null) return;

            GetInOrdderTraversalOrder(node.Left, inOrderTraversalOrder);
            inOrderTraversalOrder.Add(node);
            GetInOrdderTraversalOrder(node.Right, inOrderTraversalOrder);
        }

        public class BinaryTree{
            public int Value {get;set;}
            public BinaryTree? Left {get;set;} = null;
            public BinaryTree? Right{get;set;} = null;
            public BinaryTree? Parent{get;set;} = null;


            public BinaryTree(int value){
                this.Value = value;
            }
        }
        //https://www.algoexpert.io/questions/min-height-bst
        public static Tree? MinHeightBST(List<int> array){
            
            //1.Naive- T:O(n(log(n))) | S:O(n)
            Tree? tree = ConstructMinHeightBSTNaive(array, null, 0, array.Count-1);

            //2.Optimal- T:O(n) | S:O(n)
            tree = ConstructMinHeightBSTOptimal(array, null, 0, array.Count-1);

            //2.Optimal2- T:O(n) | S:O(n)
            tree = ConstructMinHeightBSTOptimal2(array,  0, array.Count-1);

            return tree;
        }

        private static Tree? ConstructMinHeightBSTOptimal2(List<int> array, int startIdx, int endIdx)
        {
            if(endIdx < startIdx) return null;

            int midIdx = (endIdx+startIdx)/2;

            Tree bst = new Tree(array[midIdx]);

            bst.Left = ConstructMinHeightBSTOptimal2(array, startIdx,midIdx-1);
            bst.Right = ConstructMinHeightBSTOptimal2(array, midIdx+1, endIdx);
            
            return bst;
            
        }

        private static Tree? ConstructMinHeightBSTOptimal(List<int> array, Tree? bst, int startIdx, int endIdx)
        {
            if(endIdx < startIdx) return null;
            int midIdx = (endIdx+startIdx)/2;
            Tree newBSTNode = new Tree(array[midIdx]);
            if(bst == null){
                bst = newBSTNode;
            }else{
                if(array[midIdx] < bst.Value){
                    bst.Left = newBSTNode;
                    bst = bst.Left;
                }
                else {
                    bst.Right = newBSTNode;
                    bst = bst.Right;
                } 
            }
            ConstructMinHeightBSTOptimal(array,bst, startIdx, midIdx-1);
            ConstructMinHeightBSTOptimal(array,bst, midIdx+1, endIdx);
            return bst;
        }

        private static Tree? ConstructMinHeightBSTNaive(List<int> array, Tree? bst, int startIdx, int endIdx)
        {
            if(endIdx < startIdx) return null;
            //0+5/2=>2
            int midIdx = (endIdx+startIdx)/2;
            int valueToAdd = array[midIdx];
            if(bst == null)
                bst = new Tree(valueToAdd);
            else 
                bst.InsertBST(valueToAdd);
            
            ConstructMinHeightBSTNaive(array,bst, startIdx, midIdx-1);
            ConstructMinHeightBSTNaive(array, bst, midIdx+1, endIdx);
            return bst;
        }

        //https://www.algoexpert.io/questions/find-closest-value-in-bst
        public static int FindClosestValueInBst(Tree tree, int target){

            //AVG: T:O(log(n)) | S:O(log(n))
            //Worst:T: O(n) | S: O(N)
            int closest= FindClosestValueInBstRec(tree, target, tree.Value);

            //AVG: T:O(log(n)) | S:O(1)
            //Worst:T: O(n) | S: O(1)
            closest= FindClosestValueInBstIterative(tree, target, tree.Value);

            return closest;
        }

        private static int FindClosestValueInBstIterative(Tree tree, int target, int closest)
        {
            Tree currentNode= tree;

            while(currentNode!=null){
                if(Math.Abs(target-closest) > Math.Abs(target-currentNode.Value))
                    closest=currentNode.Value;
                
                if(target < currentNode.Value)
                    currentNode = currentNode.Left;
                else if (target > currentNode.Value)
                    currentNode = currentNode.Right;
                else 
                    break; 

            }
            return closest;
        }

        private static int FindClosestValueInBstRec(Tree tree, int target, int closest)
        {
            if(Math.Abs(target-closest) > Math.Abs(target-tree.Value))
                closest = tree.Value;

            if(target < tree.Value && tree.Left != null)
                return FindClosestValueInBstRec(tree.Left, target, closest);
            else if(target > tree.Value && tree.Right != null)
                return FindClosestValueInBstRec(tree.Right, target, closest);
            else return closest;


        }

        //https://www.algoexpert.io/questions/height-balanced-binary-tree
        public class TreeInfo{
            public bool IsBalanced{get;set;}
            public int Height {get;set;}

            public TreeInfo(bool isBalanced, int height){
                this.IsBalanced = isBalanced;
                this.Height=height;
            }
        }
        public bool IsHeightBalancedBinaryTree(Tree tree){
            //T:O(n) | S:O(h)
            TreeInfo treeInfo = GetTreeInfo(tree);
            return  treeInfo.IsBalanced;
        }

        private TreeInfo GetTreeInfo(Tree node)
        {
            if(node == null) return new TreeInfo(true, -1);

            TreeInfo leftSubTreeInfo = GetTreeInfo(node.Left);
            TreeInfo rightSubTreeInfo = GetTreeInfo(node.Right);

            bool isBalanced= leftSubTreeInfo.IsBalanced && rightSubTreeInfo.IsBalanced
                            && Math.Abs(leftSubTreeInfo.Height-rightSubTreeInfo.Height) <=1;
            int height = Math.Max(leftSubTreeInfo.Height, rightSubTreeInfo.Height)+1;
            return new TreeInfo(isBalanced, height);


        }

        //https://www.algoexpert.io/questions/depth-first-search
        class Node{
            public string Name {get;set;}
            public List<Node> Children = new List<Node>();

            public Node(string name){
                this.Name= name;
            }
            public List<string> DepthFirstSearch(List<string> array){
                array.Add(this.Name);
                for(int i=0; i < Children.Count; i++){
                    Children[i].DepthFirstSearch(array);
                }

                return array;
            }
            public Node AddChild(string name){
                Node child = new Node(name);
                Children.Add(child);
                return this;
            }

        }

        //https://www.algoexpert.io/questions/validate-bst
        public static bool ValidateBST(Tree tree){
            //T:O(n) | S:O(d) d->height/distance of tree calls in call stack
            return ValidateBST(tree, Int32.MinValue, Int32.MaxValue);
            
        }

        private static bool ValidateBST(Tree node, int minValue, int maxValue)
        {
            if(node.Value < minValue || node.Value >= maxValue)
                return false;
            
            if(node.Left != null && !ValidateBST(node.Left,minValue,node.Value)){
                return false;
            }

            if(node.Right!=null && !ValidateBST(node.Right,node.Value, maxValue))
                return false;

            return true;

        }
        //https://www.algoexpert.io/questions/right-smaller-than
        public static List<int> RightSmallerThan(List<int> array){
            
            List<int> rightSmallerCounts = new List<int>();

            if(array.Count ==0 ) return rightSmallerCounts;
            
            //1. T: O(n2) | O(n)
            rightSmallerCounts = RightSmallerThanNaive(array);
        
            //2. Average case: when the created BST is balanced
                        // O(nlog(n)) time | O(n) space - where n is the length of the array
                // Worst case: when the created BST is like a linked list
                        // O(n^2) time | O(n) space
            TODO:

            //3.Average case: when the created BST is balanced
                        // O(nlog(n)) time | O(n) space - where n is the length of the array
                // Worst case: when the created BST is like a linked list
                        // O(n^2) time | O(n) space
            int lastIdx = array.Count-1;
            rightSmallerCounts[lastIdx]=0;
            SpecialBST bst = new SpecialBST(array[lastIdx]);            
            for(int i=array.Count-2; i >=0; i--){
                bst.Insert(array[i],i,rightSmallerCounts);
            }
            return rightSmallerCounts;

        }

        private static List<int> RightSmallerThanNaive(List<int> array)
        {
            List<int> rightSmallerCounts = new List<int>();
            for(int i=0; i< array.Count; i++){
                int rightSmallerCount =0;

                for(int j=i+1; j<array.Count; j++){

                    if(array[i] > array[j])
                        rightSmallerCount++;
                }

                rightSmallerCounts.Add(rightSmallerCount);
            }
            return rightSmallerCounts;
        }
    }
}