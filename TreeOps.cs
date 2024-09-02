using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class TreeOps
    {

        //https://www.algoexpert.io/questions/find-nodes-distance-k
        public static List<int> FindNodesDistanceK(BinaryTree tree, int target, int k)
        {
            //1.BFS - T:O(n) | S:O(n)
            List<int> nodesWithDistanceK = FindNodesDistanceKBFS(tree, target, k);

            //2.DFS - T:O(n) | S:O(n)
            FindNodesDistanceKDFS(tree, target, k, nodesWithDistanceK);
            return nodesWithDistanceK;
        }

        private static int FindNodesDistanceKDFS(BinaryTree node, int target, int k, List<int> nodeDistanceK)
        {
            if (node == null) return -1;
            if (node.Value == target)
            {
                AddSubtreeNodesAtDistanceK(node, 0, k, nodeDistanceK);
                return 1;
            }

            int leftDistance = FindNodesDistanceKDFS(node.Left, target, k, nodeDistanceK);
            int rightDistance = FindNodesDistanceKDFS(node.Right, target, k, nodeDistanceK);

            if (leftDistance == k || rightDistance == k) nodeDistanceK.Add(node.Value);

            if (leftDistance != -1)
            {
                AddSubtreeNodesAtDistanceK(node.Right, leftDistance + 1, k, nodeDistanceK);
                return leftDistance + 1;
            }
            if (rightDistance != -1)
            {
                AddSubtreeNodesAtDistanceK(node.Left, rightDistance + 1, k, nodeDistanceK);
                return rightDistance + 1;
            }

            return -1;

        }

        private static void AddSubtreeNodesAtDistanceK(BinaryTree node, int distance, int k, List<int> nodeDistanceK)
        {
            if (node == null) return;
            if (distance == k)
            {
                nodeDistanceK.Add(node.Value);
            }
            else
            {
                AddSubtreeNodesAtDistanceK(node.Left, distance + 1, k, nodeDistanceK);
                AddSubtreeNodesAtDistanceK(node.Right, distance + 1, k, nodeDistanceK);
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
            queue.Enqueue(new Tuple<BinaryTree, int>(targetNode, 0));

            HashSet<int> seen = new HashSet<int>();
            seen.Add(targetNode.Value);

            while (queue.Count > 0)
            {
                Tuple<BinaryTree, int> vals = queue.Dequeue();
                BinaryTree currentNode = vals.Item1;
                int distanceFromTarget = vals.Item2;

                if (distanceFromTarget == k)
                {
                    List<int> nodeDistnaceK = new List<int>();
                    foreach (var pair in queue)
                    {
                        nodeDistnaceK.Add(pair.Item1.Value);
                    }
                    nodeDistnaceK.Add(currentNode.Value);
                    return nodeDistnaceK;
                }

                List<BinaryTree> connectedNodes = new List<BinaryTree>();
                connectedNodes.Add(currentNode.Left);
                connectedNodes.Add(currentNode.Right);
                connectedNodes.Add(nodesToParents[currentNode.Value]);

                foreach (var node in connectedNodes)
                {
                    if (node == null) continue;

                    if (seen.Contains(node.Value)) continue;

                    seen.Add(node.Value);
                    queue.Enqueue(new Tuple<BinaryTree, int>(node, distanceFromTarget + 1));
                }

            }
            return new List<int>();

        }

        private static BinaryTree GetTargetNodeFromValue(int target, BinaryTree tree, Dictionary<int, BinaryTree> nodesToParents)
        {
            if (tree.Value == target) return tree;

            BinaryTree nodeParent = nodesToParents[target];
            if (nodeParent.Left != null && nodeParent.Left.Value == target) return nodeParent.Left;

            return nodeParent.Right;

        }
        private static void PopulateNodesToParents(BinaryTree node, Dictionary<int, BinaryTree> nodesToParents, BinaryTree parent)
        {
            if (node != null)
            {
                nodesToParents[node.Value] = parent;
                PopulateNodesToParents(node.Left, nodesToParents, node);
                PopulateNodesToParents(node.Right, nodesToParents, node);
            }

        }

        //https://www.algoexpert.io/questions/find-successor
        public static BinaryTree FindSuccessor(BinaryTree tree, BinaryTree node)
        {

            //1- T:O(n) | S:O(n) 
            BinaryTree successorNode = FindSuccessorUsingInOrderTraversalOrder(tree, node);

            //2: T:O(h) | S:O(1)
            successorNode = FindSuccessorUsingParentPointer(tree, node);

            return successorNode;
        }

        private static BinaryTree FindSuccessorUsingParentPointer(BinaryTree tree, BinaryTree node)
        {
            if (node.Right != null) return GetLeftMostChild(node.Right);
            return GetRightMostParent(node);

        }

        private static BinaryTree GetLeftMostChild(BinaryTree node)
        {
            BinaryTree currentNode = node;
            while (currentNode.Left != null)
                currentNode = currentNode.Left;
            return currentNode;

        }
        private static BinaryTree GetRightMostParent(BinaryTree node)
        {
            BinaryTree currentNode = node;

            while (currentNode.Parent != null && currentNode.Parent.Right == currentNode)
                currentNode = currentNode.Parent;

            return currentNode.Parent;
        }
        private static BinaryTree FindSuccessorUsingInOrderTraversalOrder(BinaryTree tree, BinaryTree node)
        {
            List<BinaryTree> inOrderTraversalOrder = new List<BinaryTree>();
            GetInOrdderTraversalOrder(tree, inOrderTraversalOrder);
            for (int i = 0; i < inOrderTraversalOrder.Count; i++)
            {
                BinaryTree currentNode = inOrderTraversalOrder[i];

                if (currentNode != node) continue;

                if (i == inOrderTraversalOrder.Count - 1)
                    return null;

                return inOrderTraversalOrder[i + 1];
            }
            return null;
        }

        private static void GetInOrdderTraversalOrder(BinaryTree node, List<BinaryTree> inOrderTraversalOrder)
        {
            if (node == null) return;

            GetInOrdderTraversalOrder(node.Left, inOrderTraversalOrder);
            inOrderTraversalOrder.Add(node);
            GetInOrdderTraversalOrder(node.Right, inOrderTraversalOrder);
        }

        public class BinaryTree
        {
            public int Value { get; set; }
            public BinaryTree? Left { get; set; } = null;
            public BinaryTree? Right { get; set; } = null;
            public BinaryTree? Parent { get; set; } = null;


            public BinaryTree(int value)
            {
                this.Value = value;
            }
            public BinaryTree(int value, BinaryTree parent)
            {
                this.Value = value;
                this.Parent = parent;
            }

        }
        //https://www.algoexpert.io/questions/min-height-bst
        public static Tree? MinHeightBST(List<int> array)
        {

            //1.Naive- T:O(n(log(n))) | S:O(n)
            Tree? tree = ConstructMinHeightBSTNaive(array, null, 0, array.Count - 1);

            //2.Optimal- T:O(n) | S:O(n)
            tree = ConstructMinHeightBSTOptimal(array, null, 0, array.Count - 1);

            //2.Optimal2- T:O(n) | S:O(n)
            tree = ConstructMinHeightBSTOptimal2(array, 0, array.Count - 1);

            return tree;
        }

        private static Tree? ConstructMinHeightBSTOptimal2(List<int> array, int startIdx, int endIdx)
        {
            if (endIdx < startIdx) return null;

            int midIdx = (endIdx + startIdx) / 2;

            Tree bst = new Tree(array[midIdx]);

            bst.Left = ConstructMinHeightBSTOptimal2(array, startIdx, midIdx - 1);
            bst.Right = ConstructMinHeightBSTOptimal2(array, midIdx + 1, endIdx);

            return bst;

        }

        private static Tree? ConstructMinHeightBSTOptimal(List<int> array, Tree? bst, int startIdx, int endIdx)
        {
            if (endIdx < startIdx) return null;
            int midIdx = (endIdx + startIdx) / 2;
            Tree newBSTNode = new Tree(array[midIdx]);
            if (bst == null)
            {
                bst = newBSTNode;
            }
            else
            {
                if (array[midIdx] < bst.Value)
                {
                    bst.Left = newBSTNode;
                    bst = bst.Left;
                }
                else
                {
                    bst.Right = newBSTNode;
                    bst = bst.Right;
                }
            }
            ConstructMinHeightBSTOptimal(array, bst, startIdx, midIdx - 1);
            ConstructMinHeightBSTOptimal(array, bst, midIdx + 1, endIdx);
            return bst;
        }

        private static Tree? ConstructMinHeightBSTNaive(List<int> array, Tree? bst, int startIdx, int endIdx)
        {
            if (endIdx < startIdx) return null;
            //0+5/2=>2
            int midIdx = (endIdx + startIdx) / 2;
            int valueToAdd = array[midIdx];
            if (bst == null)
                bst = new Tree(valueToAdd);
            else
                bst.InsertBST(valueToAdd);

            ConstructMinHeightBSTNaive(array, bst, startIdx, midIdx - 1);
            ConstructMinHeightBSTNaive(array, bst, midIdx + 1, endIdx);
            return bst;
        }

        //https://www.algoexpert.io/questions/find-closest-value-in-bst
        public static int FindClosestValueInBst(Tree tree, int target)
        {

            //AVG: T:O(log(n)) | S:O(log(n))
            //Worst:T: O(n) | S: O(N)
            int closest = FindClosestValueInBstRec(tree, target, tree.Value);

            //AVG: T:O(log(n)) | S:O(1)
            //Worst:T: O(n) | S: O(1)
            closest = FindClosestValueInBstIterative(tree, target, tree.Value);

            return closest;
        }

        private static int FindClosestValueInBstIterative(Tree tree, int target, int closest)
        {
            Tree currentNode = tree;

            while (currentNode != null)
            {
                if (Math.Abs(target - closest) > Math.Abs(target - currentNode.Value))
                    closest = currentNode.Value;

                if (target < currentNode.Value)
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
            if (Math.Abs(target - closest) > Math.Abs(target - tree.Value))
                closest = tree.Value;

            if (target < tree.Value && tree.Left != null)
                return FindClosestValueInBstRec(tree.Left, target, closest);
            else if (target > tree.Value && tree.Right != null)
                return FindClosestValueInBstRec(tree.Right, target, closest);
            else return closest;


        }

        //https://www.algoexpert.io/questions/height-balanced-binary-tree
        public class TreeInfo
        {
            public bool IsBalanced { get; set; }
            public int Height { get; set; }
            public int RootIdx;

            public TreeInfo(bool isBalanced, int height)
            {
                this.IsBalanced = isBalanced;
                this.Height = height;
            }
            public TreeInfo(int rootIDx)
            {
                this.RootIdx = rootIDx;
            }
        }
        public bool IsHeightBalancedBinaryTree(Tree tree)
        {
            //T:O(n) | S:O(h)
            TreeInfo treeInfo = GetTreeInfo(tree);
            return treeInfo.IsBalanced;
        }

        private TreeInfo GetTreeInfo(Tree node)
        {
            if (node == null) return new TreeInfo(true, -1);

            TreeInfo leftSubTreeInfo = GetTreeInfo(node.Left);
            TreeInfo rightSubTreeInfo = GetTreeInfo(node.Right);

            bool isBalanced = leftSubTreeInfo.IsBalanced && rightSubTreeInfo.IsBalanced
                            && Math.Abs(leftSubTreeInfo.Height - rightSubTreeInfo.Height) <= 1;
            int height = Math.Max(leftSubTreeInfo.Height, rightSubTreeInfo.Height) + 1;
            return new TreeInfo(isBalanced, height);


        }

        //https://www.algoexpert.io/questions/depth-first-search
        class Node
        {
            public string Name { get; set; }
            public List<Node> Children = new List<Node>();

            public Node(string name)
            {
                this.Name = name;
            }
            public List<string> DepthFirstSearch(List<string> array)
            {
                array.Add(this.Name);
                for (int i = 0; i < Children.Count; i++)
                {
                    Children[i].DepthFirstSearch(array);
                }

                return array;
            }
            public Node AddChild(string name)
            {
                Node child = new Node(name);
                Children.Add(child);
                return this;
            }

        }

        //https://www.algoexpert.io/questions/validate-bst
        public static bool ValidateBST(Tree tree)
        {
            //T:O(n) | S:O(d) d->height/distance of tree calls in call stack
            return ValidateBST(tree, Int32.MinValue, Int32.MaxValue);

        }

        private static bool ValidateBST(Tree node, int minValue, int maxValue)
        {
            if (node.Value < minValue || node.Value >= maxValue)
                return false;

            if (node.Left != null && !ValidateBST(node.Left, minValue, node.Value))
            {
                return false;
            }

            if (node.Right != null && !ValidateBST(node.Right, node.Value, maxValue))
                return false;

            return true;

        }
        //https://www.algoexpert.io/questions/right-smaller-than
        public static List<int> RightSmallerThan(List<int> array)
        {

            List<int> rightSmallerCounts = new List<int>();

            if (array.Count == 0) return rightSmallerCounts;

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
            int lastIdx = array.Count - 1;
            rightSmallerCounts[lastIdx] = 0;
            SpecialBST bst = new SpecialBST(array[lastIdx]);
            for (int i = array.Count - 2; i >= 0; i--)
            {
                bst.Insert(array[i], i, rightSmallerCounts);
            }
            return rightSmallerCounts;

        }

        private static List<int> RightSmallerThanNaive(List<int> array)
        {
            List<int> rightSmallerCounts = new List<int>();
            for (int i = 0; i < array.Count; i++)
            {
                int rightSmallerCount = 0;

                for (int j = i + 1; j < array.Count; j++)
                {

                    if (array[i] > array[j])
                        rightSmallerCount++;
                }

                rightSmallerCounts.Add(rightSmallerCount);
            }
            return rightSmallerCounts;
        }

        //https://www.algoexpert.io/questions/youngest-common-ancestor
        public static AncestralTree GetYoungCommonAncestor(
                                    AncestralTree topAncestor, AncestralTree descedantOne,
                                    AncestralTree descedantTwo)
        {
            //T:O(d) | O(1) where d is depth/hieght of ancestal tree
            int depthOne = GetDescendantDepth(descedantOne, topAncestor);
            int depthTwo = GetDescendantDepth(descedantTwo, topAncestor);

            if (depthOne > depthTwo)
            {

                return BacktrackAncestralTree(descedantOne, descedantTwo, depthOne - depthTwo);
            }
            else
            {
                return BacktrackAncestralTree(descedantTwo, descedantOne, depthTwo - depthOne);
            }

        }

        private static AncestralTree BacktrackAncestralTree(AncestralTree lowerDescedant, AncestralTree higherDescendant, int diff)
        {
            while (diff > 0)
            {
                lowerDescedant = lowerDescedant.Ancestor;
                diff--;
            }
            while (lowerDescedant != higherDescendant)
            {
                lowerDescedant = lowerDescedant.Ancestor;
                higherDescendant = higherDescendant.Ancestor;
            }
            return lowerDescedant;
        }

        private static int GetDescendantDepth(AncestralTree descedant, AncestralTree topAncestor)
        {
            int depth = 0;

            while (descedant != topAncestor)
            {
                depth++;
                descedant = descedant.Ancestor;

            }
            return depth;
        }
        //https://www.algoexpert.io/questions/binary-tree-diameter
        public static int BinaryTreeDiameter(BinaryTree binaryTree)
        {
        TODO:
            // Average case: when the tree is balanced
            // O(n) time | O(h) space - where n is the number of nodes in
            // the Binary Tree and h is the height of the Binary Tree
            return GetTreeinfo(binaryTree).Diameter;

        }

        private static TreeInfoDet GetTreeinfo(BinaryTree tree)
        {
            if (tree == null) return new TreeInfoDet(0, 0);

            TreeInfoDet leftTreeInfo = GetTreeinfo(tree.Left);
            TreeInfoDet rightTreeInfo = GetTreeinfo(tree.Right);

            int longestPathThroughRoot = leftTreeInfo.Height + rightTreeInfo.Height;

            int maxDiameterSoFar = Math.Max(leftTreeInfo.Diameter, rightTreeInfo.Diameter);

            int curretDiameter = Math.Max(longestPathThroughRoot, maxDiameterSoFar);
            int currentHeight = 1 + Math.Max(leftTreeInfo.Height, rightTreeInfo.Height);

            return new TreeInfoDet(curretDiameter, currentHeight);

        }
        public class TreeInfoDet
        {
            public int Diameter { get; set; }
            public int Height { get; set; }

            public TreeInfoDet(int diameter, int height)
            {
                this.Diameter = diameter;
                this.Height = height;
            }

        }

        public class AncestralTree
        {
            public char Name { get; set; }
            public AncestralTree Ancestor { get; set; }

            public AncestralTree(char name)
            {
                this.Name = name;
                this.Ancestor = null;
            }
            public void AddAsAncestor(AncestralTree[] descedants)
            {
                foreach (AncestralTree descendant in descedants)
                {
                    descendant.Ancestor = this;
                }
            }
        }
        //https://www.algoexpert.io/questions/compare-leaf-traversal
        public static bool CompareLeafTraversal(BinaryTree tree1, BinaryTree tree2)
        {

            //NOTE: tree1 & tree2 are root NODES and not complete tree!
            //1.Naive using PreOrder traversal and two arrays to store leaf nodes to compare later
            //T: O(n+m) | S:O(m+1)/2 + O(n+1)/2

            //2.Optimal using Stack to compare leafs on spot
            //T: O(n+m) | S:O(h1) + O(h2)
            bool result = CompareLeafTraversalStackIterative(tree1, tree2);

            //3.Optimal using  LinkedList compare leafs on spot
            //T: O(n+m) | S:O(Max(h1,h2))
            result = CompareLeafTraversalLinkedListRecursive(tree1, tree2);

            return result;
        }

        private static bool CompareLeafTraversalLinkedListRecursive(BinaryTree tree1, BinaryTree tree2)
        {
        TODO:
            BinaryTree tree1NodesLinkedList = ConnectedLeafNodes(tree1, null, null)[0];
            BinaryTree tree2NodesLinkedList = ConnectedLeafNodes(tree2, null, null)[0];

            BinaryTree list1CurrentNode = tree1NodesLinkedList;
            BinaryTree list2CurrentNode = tree2NodesLinkedList;
            while (list1CurrentNode != null && list2CurrentNode != null)
            {
                if (list1CurrentNode.Value != list2CurrentNode.Value) return false;

                list1CurrentNode = list1CurrentNode.Right;
                list2CurrentNode = list2CurrentNode.Right;
            }
            return list1CurrentNode == null && list2CurrentNode == null;
        }

        private static BinaryTree[] ConnectedLeafNodes(BinaryTree currentNode, BinaryTree head, BinaryTree prevNode)
        {
            if (currentNode == null) return new BinaryTree[] { head, prevNode };

            if (IsLeafNode(currentNode))
            {

                if (prevNode == null)
                {
                    head = currentNode;
                }
                else
                {
                    prevNode.Right = currentNode;
                }

                prevNode = currentNode;
            }
            BinaryTree[] nodes = ConnectedLeafNodes(currentNode.Left, head, prevNode);
            BinaryTree leafHead = nodes[0];
            BinaryTree leftPrevNode = nodes[1];

            return ConnectedLeafNodes(currentNode.Right, leafHead, leftPrevNode);


        }

        private static bool CompareLeafTraversalStackIterative(BinaryTree tree1, BinaryTree tree2)
        {
            Stack<BinaryTree> tree1TraversalStack = new Stack<BinaryTree>();
            tree1TraversalStack.Push(tree1);
            Stack<BinaryTree> tree2TraversalStack = new Stack<BinaryTree>();
            tree2TraversalStack.Push(tree2);

            while (tree1TraversalStack.Count > 0 && tree2TraversalStack.Count > 0)
            {
                BinaryTree tree1Leaf = GetNextLeafNode(tree1TraversalStack);
                BinaryTree tree2Leaf = GetNextLeafNode(tree2TraversalStack);

                if (tree1.Value != tree2Leaf.Value) return false;

            }
            return tree1TraversalStack.Count == 0 && tree2TraversalStack.Count == 0;
        }

        private static BinaryTree GetNextLeafNode(Stack<BinaryTree> traversalStack)
        {
            BinaryTree currentNode = traversalStack.Pop();

            while (!IsLeafNode(currentNode))
            {

                if (currentNode.Right != null)
                {
                    traversalStack.Push(currentNode.Right);
                }

                if (currentNode.Left != null)
                    traversalStack.Push(currentNode.Left);


                currentNode = traversalStack.Pop();

            }
            return currentNode;

        }

        private static bool IsLeafNode(BinaryTree node)
        {
            return (node.Left == null) && (node.Right == null);
        }

        //https://www.algoexpert.io/questions/find-kth-largest-value-in-bst
        public static int FindKthLargestValueInBST(BST tree, int k)
        {

            //1.Naive- Using In-order traversal
            //T:O(n) | O(n)
            int kthLargestNodeValue = FindKthLargestValueInBSTUsingInOrderTraversal(tree, k);


            //2.Optimal - Leverging BST property and using Reverse In-order traversal as ask is to find largest node value
            //T:O(h+k) | O(h) -  h is height of tree and k is input parameter
            kthLargestNodeValue = FindKthLargestValueInBSTUsingReverseInOrderTraversal(tree, k);

            return kthLargestNodeValue;
        }

        private static int FindKthLargestValueInBSTUsingReverseInOrderTraversal(BST tree, int k)
        {
            BSTreeInfo bSTreeInfo = new BSTreeInfo(0, -1);
            SortedNodeValuesUsingReverseInOrderTraversal(tree, k, bSTreeInfo);
            return bSTreeInfo.LatestVisitedNodeValue;
        }

        private static void SortedNodeValuesUsingReverseInOrderTraversal(BST node, int k, BSTreeInfo bSTreeInfo)
        {
            if (node == null || bSTreeInfo.NumberOfNodesVisited >= k) return;

            SortedNodeValuesUsingReverseInOrderTraversal(node.Right, k, bSTreeInfo);
            if (bSTreeInfo.NumberOfNodesVisited < k)
            {

                bSTreeInfo.LatestVisitedNodeValue = node.Value;
                bSTreeInfo.NumberOfNodesVisited += +1;
                SortedNodeValuesUsingReverseInOrderTraversal(node.Left, k, bSTreeInfo);
            }
        }

        public class BSTreeInfo
        {
            public int NumberOfNodesVisited;
            public int LatestVisitedNodeValue;

            public BSTreeInfo(int numberOfNodesVisited, int latestVisitedNodeValue)
            {
                this.NumberOfNodesVisited = numberOfNodesVisited;
                this.LatestVisitedNodeValue = latestVisitedNodeValue;
            }
        }
        private static int FindKthLargestValueInBSTUsingInOrderTraversal(BST tree, int k)
        {
            List<int> sortedNodeValues = new List<int>();
            SortedNodeValuesUsingInOrderTraversal(tree, sortedNodeValues);

            return sortedNodeValues[sortedNodeValues.Count - k];

        }

        private static void SortedNodeValuesUsingInOrderTraversal(BST node, List<int> sortedNodeValues)
        {
            if (node == null) return;
            //Left-Visit-Right
            SortedNodeValuesUsingInOrderTraversal(node.Left, sortedNodeValues);
            sortedNodeValues.Add(node.Value); //Visit
            SortedNodeValuesUsingInOrderTraversal(node.Right, sortedNodeValues);

        }
        //https://www.algoexpert.io/questions/same-bsts
        public static bool SameBSTs(List<int> arrayOne, List<int> arrayTwo)
        {


            //1.Using Extra space - T:O(n^2) | S:O(n^2)
            bool areSameBst = AreSameBSTs(arrayOne, arrayTwo);

        //2.w/o using Extra space - T:O(n^2) | S:O(1)
        TODO:

            return areSameBst;

        }

        private static bool AreSameBSTs(List<int> arrayOne, List<int> arrayTwo)
        {

            if (arrayOne.Count != arrayTwo.Count) return false;
            if (arrayOne.Count == 0 && arrayTwo.Count == 0) return true;
            if (arrayOne[0] != arrayTwo[0]) return false;

            List<int> leftOne = GetSmaller(arrayOne); //O(n)
            List<int> leftTwo = GetSmaller(arrayTwo); //O(n)

            List<int> rightOne = GetBiggerOrEqual(arrayOne); //O(n)
            List<int> rightTwo = GetBiggerOrEqual(arrayTwo);//O(n)

            return AreSameBSTs(leftOne, leftTwo) && AreSameBSTs(rightOne, rightTwo);

        }

        private static List<int> GetBiggerOrEqual(List<int> array)
        {
            List<int> biggerOrEqual = new List<int>();
            for (int i = 1; i < array.Count; i++)
            {
                if (array[i] >= array[0]) biggerOrEqual.Add(array[i]);
            }
            return biggerOrEqual;
        }

        private static List<int> GetSmaller(List<int> array)
        {
            List<int> smaller = new List<int>();
            for (int i = 1; i < array.Count; i++)
            {
                if (array[i] < array[0]) smaller.Add(array[i]);
            }
            return smaller;
        }
        //https://www.algoexpert.io/questions/validate-three-nodes
        public static bool ValidateThreeNodes(BST nodeOne, BST nodeTwo, BST nodeThree)
        {

            //1.Recursion with callstack space 
            //T:O(h) | S:O(h) where h is height of tree
            bool result = ValidateThreeNodesRecur(nodeOne, nodeTwo, nodeThree);

            //2.Iterative with no extra space
            //T:O(h) | S:O(1) where h is height of tree
            result = ValidateThreeNodesIterative(nodeOne, nodeTwo, nodeThree);

        //3.Optimal - T:O(d) | S:O(1) where d is distance beween node one and three.
        TODO:
            result = ValidateThreeNodesOptimal(nodeOne, nodeTwo, nodeThree);

            return result;

        }

        private static bool ValidateThreeNodesOptimal(BST nodeOne, BST nodeTwo, BST nodeThree)
        {
            BST searchOne = nodeOne;
            BST searchTwo = nodeThree;

            while (true)
            {

                bool foundThreeFromOne = searchOne == nodeThree;
                bool foundOneFromThree = searchTwo == nodeOne;
                bool foundNodeTwo = (searchOne == nodeTwo || searchTwo == nodeTwo);
                bool finishedSearching = (searchOne == null) & searchTwo == null;
                if (foundThreeFromOne || foundOneFromThree || foundNodeTwo || finishedSearching)
                    break;

                if (searchOne != null)
                    searchOne = searchOne.Value > nodeTwo.Value ? searchOne.Left : searchOne.Right;

                if (searchTwo != null)
                    searchTwo = searchTwo.Value > nodeTwo.Value ? searchTwo.Left : searchTwo.Right;

            }
            bool foundNodeFromOther = (searchOne == nodeThree || searchTwo == nodeOne);

            bool foundNodeTwoFinal = searchOne == nodeTwo || searchTwo == nodeTwo;

            if (!foundNodeTwoFinal || foundNodeFromOther)
                return false;

            return searchForTarget(nodeTwo, searchOne == nodeTwo ? nodeThree : nodeOne);

        }

        private static bool searchForTarget(BST node, BST target)
        {
            while (node != null && node != target)
            {
                node = target.Value < node.Value ? node.Left : node.Right;
            }

            return node == target;

        }

        private static bool ValidateThreeNodesIterative(BST nodeOne, BST nodeTwo, BST nodeThree)
        {
            if (IsDescendantIterative(nodeTwo, nodeOne))
            { // Is nodeOne descendant of NodeTwo?
                return IsDescendantIterative(nodeThree, nodeTwo); //Is nodeThree ancestor of nodeTwo?
            }
            if (IsDescendantIterative(nodeTwo, nodeThree))
            { // Is nodeThree descendant of NodeTwo?
                return IsDescendantIterative(nodeOne, nodeTwo); //Is nodeOne ancestor of nodeTwo?
            }
            return false;
        }

        private static bool IsDescendantIterative(BST node, BST target)
        {
            while (node != null && node != target)
            {
                node = target.Value < node.Value ? node.Left : node.Right;
            }
            return node == target;
        }

        private static bool ValidateThreeNodesRecur(BST nodeOne, BST nodeTwo, BST nodeThree)
        {
            if (IsDescendantRecur(nodeTwo, nodeOne))
            {
                return IsDescendantRecur(nodeThree, nodeTwo);
            }
            if (IsDescendantRecur(nodeTwo, nodeThree))
            {
                return IsDescendantRecur(nodeOne, nodeTwo);
            }
            return false;

        }

        private static bool IsDescendantRecur(BST node, BST target)
        {
            if (node == null) return false;
            if (node == target) return true;

            return (target.Value < node.Value) ? IsDescendantRecur(node.Left, target)
                                            : IsDescendantRecur(node.Right, target);
        }
        //https://www.algoexpert.io/questions/flatten-binary-tree
        public static BinaryTree FlattenBinaryTree(BinaryTree root)
        {

            //1.Using n additional space
            //T:O(n) | S:O(n)
            BinaryTree result = FlattenBinaryTree1(root);

        //2. Using no additional space apart recursive stack
        //T:O(n) | S:O(d) where d is height of tree
        TODO:
            result = FlattenBinaryTree2(root)[0];

            return result;
        }

        private static BinaryTree[] FlattenBinaryTree2(BinaryTree node)
        {
            BinaryTree leftMost;
            BinaryTree rightMost;

            if (node.Left == null)
            {
                leftMost = node;
            }
            else
            {
                BinaryTree[] leftAndRightMostNodes = FlattenBinaryTree2(node.Left);
                ConnectNodes(leftAndRightMostNodes[1], node);
                leftMost = leftAndRightMostNodes[0];
            }

            if (node.Right == null)
            {
                rightMost = node;
            }
            else
            {
                BinaryTree[] leftAndRightMostNodes = FlattenBinaryTree2(node.Right);
                ConnectNodes(node, leftAndRightMostNodes[0]);
                rightMost = leftAndRightMostNodes[1];
            }
            return new BinaryTree[] { leftMost, rightMost };

        }

        private static void ConnectNodes(BinaryTree left, BinaryTree right)
        {
            left.Right = right;
            right.Left = left;

        }

        private static BinaryTree FlattenBinaryTree1(BinaryTree root)
        {
            List<BinaryTree> inOrderNodes = new List<BinaryTree>();
            GetNodesInOrder(root, inOrderNodes);
            for (int i = 0; i < inOrderNodes.Count; i++)
            {
                BinaryTree leftNode = inOrderNodes[i];
                BinaryTree rightNode = inOrderNodes[i + 1];
                leftNode.Right = rightNode;
                rightNode.Left = leftNode;
            }
            return inOrderNodes[0];
        }

        private static void GetNodesInOrder(BinaryTree tree, List<BinaryTree> inOrderNodes)
        {
            if (tree != null)
            {
                GetNodesInOrder(tree.Left, inOrderNodes);
                inOrderNodes.Add(tree);
                GetNodesInOrder(tree.Right, inOrderNodes);

            }
        }

        //https://www.algoexpert.io/questions/split-binary-tree
        public static int SplitBinaryTree(BinaryTree tree)
        {
            //T:(n) | S:O(h) where he is height of the tree
            int treeSum = GetTreeSum(tree);

            if (treeSum % 2 != 0)
            { //Not an even sum tree to split
                return 0;
            }
            int desiredSubTreeSum = treeSum / 2;
            bool canBeSplit = TrySubTrees(tree, desiredSubTreeSum).CanBeSplit;

            return canBeSplit == true ? desiredSubTreeSum : 0;

        }

        private static ResultPair TrySubTrees(BinaryTree tree, int desiredSubTreeSum)
        {
            if (tree == null) return new ResultPair(0, false);

            ResultPair leftResultPair = TrySubTrees(tree.Left, desiredSubTreeSum);
            ResultPair rightResultPair = TrySubTrees(tree.Right, desiredSubTreeSum);

            int currentTreeSum = tree.Value + leftResultPair.CurrentTreeSum + rightResultPair.CurrentTreeSum;

            bool canBeSplit = leftResultPair.CanBeSplit || rightResultPair.CanBeSplit || currentTreeSum == desiredSubTreeSum;

            return new ResultPair(currentTreeSum, canBeSplit);
        }

        private static int GetTreeSum(BinaryTree tree)
        {
            if (tree == null) return 0;

            return tree.Value + GetTreeSum(tree.Left) + GetTreeSum(tree.Right);
        }

        public class ResultPair
        {
            public int CurrentTreeSum;
            public bool CanBeSplit;

            public ResultPair(int currentTreeSum, bool canBeSplit)
            {
                this.CurrentTreeSum = currentTreeSum;
                this.CanBeSplit = canBeSplit;
            }
        }
        //https://www.algoexpert.io/questions/merge-binary-trees
        public static BinaryTree MergeBinaryTrees(BinaryTree tree1, BinaryTree tree2)
        {

            //1. Reursion
            //T:(n) | O(h) - n is number of nodes in smaller of two trees and h is height of of shorter tree
            BinaryTree mergedBinaryTree = MergeBinaryTreesRec(tree1, tree2);

            //2. Iterative
            //T:(n) | O(h) - n is number of nodes in smaller of two trees and h is height of of shorter tree            
            mergedBinaryTree = MergeBinaryTreesIterative(tree1, tree2);
            return mergedBinaryTree;
        }

        private static BinaryTree MergeBinaryTreesIterative(BinaryTree tree1, BinaryTree tree2)
        {
            if (tree1 == null) return tree2;

            Stack<BinaryTree> tree1Stack = new Stack<BinaryTree>();
            Stack<BinaryTree> tree2Stack = new Stack<BinaryTree>();
            tree1Stack.Push(tree1);
            tree2Stack.Push(tree2);

            while (tree1Stack.Count > 0)
            {
                BinaryTree tree1Node = tree1Stack.Pop();
                BinaryTree tree2Node = tree2Stack.Pop();

                if (tree2Node == null) continue;

                tree1Node.Value += tree2Node.Value;

                if (tree1Node.Left == null)
                {
                    tree1Node.Left = tree2Node.Left;
                }
                else
                {
                    tree1Stack.Push(tree1Node.Left);
                    tree2Stack.Push(tree2Node.Left);
                }

                if (tree1Node.Right == null)
                {
                    tree1Node.Right = tree2Node.Right;
                }
                else
                {
                    tree1Stack.Push(tree1Node.Right);
                    tree2Stack.Push(tree2Node.Right);
                }


            }
            return tree1;

        }

        private static BinaryTree MergeBinaryTreesRec(BinaryTree tree1, BinaryTree tree2)
        {
            if (tree1 == null) return tree2;
            if (tree2 == null) return tree1;

            tree1.Value += tree2.Value;
            tree1.Left = MergeBinaryTrees(tree1.Left, tree2.Left);
            tree1.Right = MergeBinaryTrees(tree1.Right, tree2.Right);

            return tree1;
        }

        //https://www.algoexpert.io/questions/reconstruct-bst
        public static BST ReconstructBST(List<int> preOrderTraversalValues)
        {

            //1. Naive
            //T:O(n^2) | O(n)
            BST bst = ReconstructBSTNaive(preOrderTraversalValues);

            //1. Optimal
            //T:O(n) | O(n)
            bst = ReconstructBSTOptimal(preOrderTraversalValues);

            return bst;

        }

        private static BST ReconstructBSTOptimal(List<int> preOrderTraversalValues)
        {
            TreeInfo treeInfo = new TreeInfo(0);
            return ReconstructBSTFromRange(Int32.MinValue, Int32.MaxValue, preOrderTraversalValues, treeInfo);
        }

        private static BST ReconstructBSTFromRange(int lowerBound, int upperBound, List<int> preOrderTraversalValues, TreeInfo currentSubtreeInfo)
        {
            if (currentSubtreeInfo.RootIdx == preOrderTraversalValues.Count) return null;

            int rootValue = preOrderTraversalValues[currentSubtreeInfo.RootIdx];
            if (rootValue < lowerBound || rootValue >= upperBound)
            {
                return null;
            }

            currentSubtreeInfo.RootIdx += 1;

            BST leftSubtree = ReconstructBSTFromRange(lowerBound, rootValue, preOrderTraversalValues, currentSubtreeInfo);
            BST rightSubtree = ReconstructBSTFromRange(rootValue, upperBound, preOrderTraversalValues, currentSubtreeInfo);

            BST bst = new BST(rootValue);
            bst.Left = leftSubtree;
            bst.Right = rightSubtree;
            return bst;

        }

        private static BST ReconstructBSTNaive(List<int> preOrderTraversalValues)
        {
            if (preOrderTraversalValues.Count == 0) return null;

            int currVal = preOrderTraversalValues[0];
            int rightSubtreeRootIdx = preOrderTraversalValues.Count;

            for (int idx = 1; idx < preOrderTraversalValues.Count; idx++)
            {

                int value = preOrderTraversalValues[idx];
                if (value >= currVal)
                {
                    rightSubtreeRootIdx = idx;
                    break;
                }
            }
            BST leftSubtree = ReconstructBSTNaive(preOrderTraversalValues.GetRange(1, rightSubtreeRootIdx - 1));

            BST rightSubtree = ReconstructBSTNaive(preOrderTraversalValues.GetRange(rightSubtreeRootIdx, preOrderTraversalValues.Count - rightSubtreeRootIdx));

            BST bst = new BST(currVal);
            bst.Left = leftSubtree;
            bst.Right = rightSubtree;

            return bst;
        }
        //https://www.algoexpert.io/questions/lowest-common-manager
        public static OrgChart GetLowestCommonManager(OrgChart topManager, OrgChart reportOne, OrgChart reportTwo)
        {
            //T:O(n) | S:O(d) - n is number of employees in org and d is height/depth
            return GetOrgInfo(topManager, reportOne, reportTwo).LowestCommonManager;
        }

        private static OrgInfo GetOrgInfo(OrgChart manager, OrgChart reportOne, OrgChart reportTwo)
        {
            int numImportantReports = 0;
            foreach (OrgChart directReport in manager.DirectReports)
            {
                OrgInfo orgInfo = GetOrgInfo(directReport, reportOne, reportTwo);
                if (orgInfo.LowestCommonManager != null) return orgInfo;
                numImportantReports += orgInfo.NumImportantReports;
            }
            if (manager == reportOne || manager == reportTwo) numImportantReports++;
            OrgChart lowestCommonManager = numImportantReports == 2 ? manager : null;
            OrgInfo newOrgInfo = new OrgInfo(lowestCommonManager, numImportantReports);
            return newOrgInfo;
        }

        public class OrgChart
        {
            public char Name;
            public List<OrgChart> DirectReports;

            public OrgChart(OrgChart[] directReports)
            {
                foreach (OrgChart directReport in directReports)
                {
                    this.DirectReports.Add(directReport);
                }
            }
        }
        public class OrgInfo
        {
            public OrgChart LowestCommonManager;
            public int NumImportantReports;

            public OrgInfo(OrgChart lowestCommonManager, int numImportantReports)
            {
                this.LowestCommonManager = lowestCommonManager;
                this.NumImportantReports = numImportantReports;
            }
        }
        //https://www.algoexpert.io/questions/branch-sums
        public static List<int> BranchSums(BinaryTree root)
        {
            //T:O(n) | S:O(n)
            List<int> sums = new List<int>();
            CalculateBranchSums(root, 0, sums);
            return sums;
        }

        private static void CalculateBranchSums(BinaryTree node, int runningSum, List<int> sums)
        {
            if (node == null) return;

            int newRunningSum = runningSum + node.Value;
            if (node.Right == null && node.Left == null)
            { //Leaf node
                sums.Add(newRunningSum);
                return;
            }
            CalculateBranchSums(node.Left, newRunningSum, sums);
            CalculateBranchSums(node.Right, newRunningSum, sums);

        }
        //https://www.algoexpert.io/questions/repair-bst
        BST nodeOne = null, nodeTwo = null, previousNode = null;
        // O(n) time | O(h) space - where n is the number of nodes in the
        // tree and h is the height of the tree
        public BST RepairBst(BST tree)
        {
            this.InOrderTraversal(tree);
            int tempNodeOneValue = nodeOne.Value;
            nodeOne.Value = nodeTwo.Value;
            nodeTwo.Value = tempNodeOneValue;

            return tree;
        }

        private void InOrderTraversal(BST node)
        {
            if (node == null)
            {
                return;
            }

            InOrderTraversal(node.Left);

            if (this.previousNode != null && this.previousNode.Value > node.Value)
            {
                {
                    if (this.nodeOne == null)
                        this.nodeOne = this.previousNode;
                }
                this.nodeTwo = node;
            }

            this.previousNode = node;
            InOrderTraversal(node.Right);
        }
        // O(n) time | O(h) space - where n is the number of nodes in the
        // tree and h is the height of the tree
        public BST RepairBstIterative(BST tree)
        {
            BST nodeOne = null, nodeTwo = null, previousNode = null;

            Stack<BST> stack = new Stack<BST>();
            BST currentNode = tree;
            while (currentNode != null || stack.Count > 0)
            {
                while (currentNode != null)
                {
                    stack.Push(currentNode);
                    currentNode = currentNode.Left;
                }
                currentNode = stack.Pop();

                if (previousNode != null && previousNode.Value > currentNode.Value)
                {
                    if (nodeOne == null)
                    {
                        nodeOne = previousNode;
                    }
                    nodeTwo = currentNode;
                }

                previousNode = currentNode;
                currentNode = currentNode.Right;
            }

            int tempNodeOneValue = nodeOne.Value;
            nodeOne.Value = nodeTwo.Value;
            nodeTwo.Value = tempNodeOneValue;

            return tree;
        }

        //https://www.algoexpert.io/questions/number-of-binary-tree-topologies
        public static int NumberOfBinaryTreeTopologies(int n)
        {
            // Upper Bound: O((n*(2n)!)/(n!(n+1)!)) time | O(n) space
            int result = NumberOfBinaryTreeTopologiesNaive(n);

            // O(n^2) time | O(n) space 
            result = NumberOfBinaryTreeTopologiesOptimalRec(n);

            // O(n^2) time | O(n) space 
            result = NumberOfBinaryTreeTopologiesOptimalIterative(n);
            return result;
        }

        public static int NumberOfBinaryTreeTopologiesOptimalIterative(int n)
        {
            List<int> cache = new List<int>();
            cache.Add(1);
            for (int m = 1; m < n + 1; m++)
            {
                int numberOfTrees = 0;
                for (int leftTreeSize = 0; leftTreeSize < m; leftTreeSize++)
                {
                    int rightTreeSize = m - 1 - leftTreeSize;
                    int numberOfLeftTrees = cache[leftTreeSize];
                    int numberOfRightTrees = cache[rightTreeSize];
                    numberOfTrees += numberOfLeftTrees * numberOfRightTrees;
                }
                cache.Add(numberOfTrees);
            }
            return cache[n];
        }
        public static int NumberOfBinaryTreeTopologiesOptimalRec(int n)
        {
            Dictionary<int, int> cache = new Dictionary<int, int>();
            cache.Add(0, 1);
            return NumberOfBinaryTreeTopologiesOptimalRec(n, cache);
        }

        public static int NumberOfBinaryTreeTopologiesOptimalRec(
            int n, Dictionary<int, int> cache
        )
        {
            if (cache.ContainsKey(n))
            {
                return cache[n];
            }
            int numberOfTrees = 0;
            for (int leftTreeSize = 0; leftTreeSize < n; leftTreeSize++)
            {
                int rightTreeSize = n - 1 - leftTreeSize;
                int numberOfLeftTrees = NumberOfBinaryTreeTopologiesOptimalRec(leftTreeSize, cache);
                int numberOfRightTrees =
                    NumberOfBinaryTreeTopologiesOptimalRec(rightTreeSize, cache);
                numberOfTrees += numberOfLeftTrees * numberOfRightTrees;
            }
            cache.Add(n, numberOfTrees);
            return numberOfTrees;
        }
        public static int NumberOfBinaryTreeTopologiesNaive(int n)
        {
            if (n == 0)
            {
                return 1;
            }
            int numberOfTrees = 0;
            for (int leftTreeSize = 0; leftTreeSize < n; leftTreeSize++)
            {
                int rightTreeSize = n - 1 - leftTreeSize;
                int numberOfLeftTrees = NumberOfBinaryTreeTopologiesNaive(leftTreeSize);
                int numberOfRightTrees = NumberOfBinaryTreeTopologiesNaive(rightTreeSize);
                numberOfTrees += numberOfLeftTrees * numberOfRightTrees;
            }
            return numberOfTrees;
        }
        //https://www.algoexpert.io/questions/invert-binary-tree

        private static void swapLeftAndRight(BinaryTree tree)
        {
            BinaryTree left = tree.Left;
            tree.Left = tree.Right;
            tree.Right = left;
        }

        // O(n) time | O(n) space
        public static void InvertBinaryTreeNaive(BinaryTree tree)
        {
            List<BinaryTree> queue = new List<BinaryTree>();
            queue.Add(tree);
            var index = 0;
            while (index < queue.Count)
            {
                BinaryTree current = queue[index];
                index += 1;
                if (current == null)
                {
                    continue;
                }
                swapLeftAndRight(current);
                if (current.Left != null)
                {
                    queue.Add(current.Left);
                }
                if (current.Right != null)
                {
                    queue.Add(current.Right);
                }
            }
        }
        // O(n) time | O(d) space
        public static void InvertBinaryTreeOptimal(BinaryTree tree)
        {
            if (tree == null)
            {
                return;
            }
            swapLeftAndRight(tree);
            InvertBinaryTreeOptimal(tree.Left);
            InvertBinaryTreeOptimal(tree.Right);
        }



        //https://www.algoexpert.io/questions/symmetrical-tree
        // O(n) time | O(h) space - where n is the number of nodes in the tree
        // and h is the height of the tree.
        public bool SymmetricalTreeIterative(BinaryTree tree)
        {
            Stack<BinaryTree> stackLeft = new Stack<BinaryTree>();
            stackLeft.Push(tree.Left);
            Stack<BinaryTree> stackRight = new Stack<BinaryTree>();
            stackRight.Push(tree.Right);

            while (stackLeft.Count != 0 && stackRight.Count != 0)
            {
                BinaryTree left = stackLeft.Pop();
                BinaryTree right = stackRight.Pop();

                if (left == null && right == null)
                {
                    continue;
                }

                if (left == null || right == null || left.Value != right.Value)
                {
                    return false;
                }

                stackLeft.Push(left.Left);
                stackLeft.Push(left.Right);
                stackRight.Push(right.Right);
                stackRight.Push(right.Left);
            }

            return stackLeft.Count == 0 && stackRight.Count == 0;
        }
        // O(n) time | O(h) space - where n is the number of nodes in the tree
        // and h is the height of the tree.
        public bool SymmetricalTreeRec(BinaryTree tree)
        {
            return treesAreMirrored(tree.Left, tree.Right);
        }

        private bool treesAreMirrored(BinaryTree left, BinaryTree right)
        {
            if (left != null && right != null && left.Value == right.Value)
            {
                return treesAreMirrored(left.Left, right.Right) &&
                       treesAreMirrored(left.Right, right.Left);
            }

            return left == right;
        }

        //https://www.algoexpert.io/questions/node-depths

        // Average case: when the tree is balanced
        // O(n) time | O(h) space - where n is the number of nodes in
        // the Binary Tree and h is the height of the Binary Tree
        public static int NodeDepths1(BinaryTree root)
        {
            int sumOfDepths = 0;
            Stack<Level> stack = new Stack<Level>();
            stack.Push(new Level(root, 0));
            while (stack.Count > 0)
            {
                Level top = stack.Pop();

                BinaryTree node = top.root;
                int depth = top.depth;
                if (node == null) continue;

                sumOfDepths += depth;
                stack.Push(new Level(node.Left, depth + 1));
                stack.Push(new Level(node.Right, depth + 1));
            }
            return sumOfDepths;
        }

        public class Level
        {
            public BinaryTree root;
            public int depth;

            public Level(BinaryTree root, int depth)
            {
                this.root = root;
                this.depth = depth;
            }
        }
        // Average case: when the tree is balanced
        // O(n) time | O(h) space - where n is the number of nodes in
        // the Binary Tree and h is the height of the Binary Tree
        public static int NodeDepths2(BinaryTree root)
        {
            return nodeDepthsHelper(root, 0);
        }

        public static int nodeDepthsHelper(BinaryTree root, int depth)
        {
            if (root == null) return 0;
            return depth + nodeDepthsHelper(root.Left, depth + 1) +
                   nodeDepthsHelper(root.Right, depth + 1);
        }

        //https://www.algoexpert.io/questions/sum-bsts
        // O(n) time | O(h) space - where n is the number of nodes in the
        // tree and h is the height of the tree
        public int SumBsts(BinaryTree tree)
        {
            return getTreeInfo(tree).totalSumBstNodes;
        }

        public static TreeInfoBst getTreeInfo(BinaryTree tree)
        {
            if (tree == null)
            {
                return new TreeInfoBst(true, Int32.MinValue, Int32.MaxValue, 0, 0, 0);
            }

            TreeInfoBst leftTreeInfo = getTreeInfo(tree.Left);
            TreeInfoBst rightTreeInfo = getTreeInfo(tree.Right);

            bool satisfiesBstProp = tree.Value > leftTreeInfo.maxValue &&
                                    tree.Value <= rightTreeInfo.minValue;
            bool isBst = satisfiesBstProp && leftTreeInfo.isBst && rightTreeInfo.isBst;

            int maxValue = Math.Max(
              tree.Value, Math.Max(leftTreeInfo.maxValue, rightTreeInfo.maxValue)
            );
            int minValue = Math.Min(
              tree.Value, Math.Min(leftTreeInfo.minValue, rightTreeInfo.minValue)
            );

            int bstSum = 0;
            int bstSize = 0;

            int totalSumBstNodes =
              leftTreeInfo.totalSumBstNodes + rightTreeInfo.totalSumBstNodes;

            if (isBst)
            {
                bstSum = tree.Value + leftTreeInfo.bstSum + rightTreeInfo.bstSum;
                bstSize = 1 + leftTreeInfo.bstSize + rightTreeInfo.bstSize;

                if (bstSize >= 3)
                {
                    totalSumBstNodes = bstSum;
                }
            }

            return new TreeInfoBst(
              isBst, maxValue, minValue, bstSum, bstSize, totalSumBstNodes
            );
        }
        public class TreeInfoBst
        {
            public bool isBst;
            public int maxValue;
            public int minValue;
            public int bstSum;
            public int bstSize;
            public int totalSumBstNodes;

            public TreeInfoBst(
              bool isBst,
              int maxValue,
              int minValue,
              int bstSum,
              int bstSize,
              int totalSumBstNodes
            )
            {
                this.isBst = isBst;
                this.maxValue = maxValue;
                this.minValue = minValue;
                this.bstSum = bstSum;
                this.bstSize = bstSize;
                this.totalSumBstNodes = totalSumBstNodes;
            }

            //https://www.algoexpert.io/questions/max-path-sum-in-binary-tree
            // O(n) time | O(log(n)) space
            public static int MaxPathSum(BinaryTree tree)
            {
                List<int> maxSumArray = findMaxSum(tree);
                return maxSumArray[1];
            }

            public static List<int> findMaxSum(BinaryTree tree)
            {
                if (tree == null)
                {
                    return new List<int>() { 0, Int32.MinValue };
                }
                List<int> leftMaxSumArray = findMaxSum(tree.Left);
                int leftMaxSumAsBranch = leftMaxSumArray[0];
                int leftMaxPathSum = leftMaxSumArray[1];

                List<int> rightMaxSumArray = findMaxSum(tree.Right);
                int rightMaxSumAsBranch = rightMaxSumArray[0];
                int rightMaxPathSum = rightMaxSumArray[1];

                int maxChildSumAsBranch = Math.Max(leftMaxSumAsBranch, rightMaxSumAsBranch);
                int maxSumAsBranch = Math.Max(maxChildSumAsBranch + tree.Value, tree.Value);
                int maxSumAsRootNode = Math.Max(
                  leftMaxSumAsBranch + tree.Value + rightMaxSumAsBranch, maxSumAsBranch
                );
                int maxPathSum =
                  Math.Max(leftMaxPathSum, Math.Max(rightMaxPathSum, maxSumAsRootNode));

                return new List<int>() { maxSumAsBranch, maxPathSum };
            }

            //https://www.algoexpert.io/questions/iterative-in-order-traversal
            // O(n) time | O(1) space
            public static void IterativeInOrderTraversal(
              BinaryTree tree, Action<BinaryTree> callback
            )
            {
                BinaryTree previousNode = null;
                BinaryTree currentNode = tree;
                while (currentNode != null)
                {
                    BinaryTree nextNode;
                    if (previousNode == null || previousNode == currentNode.Parent)
                    {
                        if (currentNode.Left != null)
                        {
                            nextNode = currentNode.Left;
                        }
                        else
                        {
                            callback(currentNode);
                            nextNode =
                              currentNode.Right != null ? currentNode.Right : currentNode.Parent;
                        }
                    }
                    else if (previousNode == currentNode.Left)
                    {
                        callback(currentNode);
                        nextNode =
                          currentNode.Right != null ? currentNode.Right : currentNode.Parent;
                    }
                    else
                    {
                        nextNode = currentNode.Parent;
                    }
                    previousNode = currentNode;
                    currentNode = nextNode;
                }
            }

            //https://www.algoexpert.io/questions/right-sibling-tree

            // O(n) time | O(d) space - where n is the number of nodes in
            // the Binary Tree and d is the depth (height) of the Binary Tree
            public static BinaryTree RightSiblingTree(BinaryTree root)
            {
                mutate(root, null, false);
                return root;
            }

            public static void mutate(
              BinaryTree node, BinaryTree parent, bool isLeftChild
            )
            {
                if (node == null) return;

                var left = node.Left;
                var right = node.Right;
                mutate(left, node, true);
                if (parent == null)
                {
                    node.Right = null;
                }
                else if (isLeftChild)
                {
                    node.Right = parent.Right;
                }
                else
                {
                    if (parent.Right == null)
                    {
                        node.Right = null;
                    }
                    else
                    {
                        node.Right = parent.Right.Left;
                    }
                }
                mutate(right, node, false);
            }

            //https://www.algoexpert.io/questions/all-kinds-of-node-depths
            //1.
            // Average case: when the tree is balanced
            // O(nlog(n)) time | O(h) space - where n is the number of nodes in
            // the Binary Tree and h is the height of the Binary Tree
            public static int AllKindsOfNodeDepthsNaive(BinaryTree root)
            {
                int sumOfAllDepths = 0;
                Stack<BinaryTree> stack = new Stack<BinaryTree>();
                stack.Push(root);
                while (stack.Count > 0)
                {
                    BinaryTree node = stack.Pop();
                    if (node == null) continue;

                    sumOfAllDepths += nodeDepths(node, 0);
                    stack.Push(node.Left);
                    stack.Push(node.Right);
                }
                return sumOfAllDepths;
            }

            public static int nodeDepths(BinaryTree node, int depth)
            {
                if (node == null) return 0;
                return depth + nodeDepths(node.Left, depth + 1) +
                       nodeDepths(node.Right, depth + 1);
            }

            //2.
            // Average case: when the tree is balanced
            // O(nlog(n)) time | O(h) space - where n is the number of nodes in
            // the Binary Tree and h is the height of the Binary Tree
            public static int AllKindsOfNodeDepthsRec(BinaryTree root)
            {
                if (root == null) return 0;
                return AllKindsOfNodeDepthsRec(root.Left) + AllKindsOfNodeDepthsRec(root.Right) +
                       nodeDepths(root, 0);
            }

            //3. 
            // Average case: when the tree is balanced
            // O(n) time | O(n) space - where n is the number of nodes in the Binary Tree
            public static int AllKindsOfNodeDepths3(BinaryTree root)
            {
                Dictionary<BinaryTree, int> nodeCounts = new Dictionary<BinaryTree, int>();
                Dictionary<BinaryTree, int> nodeDepths = new Dictionary<BinaryTree, int>();
                addNodeCounts(root, nodeCounts);
                addNodeDepths(root, nodeDepths, nodeCounts);
                return sumAllNodeDepths(root, nodeDepths);
            }

            public static int sumAllNodeDepths(
              BinaryTree node, Dictionary<BinaryTree, int> nodeDepths
            )
            {
                if (node == null) return 0;
                return sumAllNodeDepths(node.Left, nodeDepths) +
                       sumAllNodeDepths(node.Right, nodeDepths) + nodeDepths[node];
            }

            public static void addNodeDepths(
    BinaryTree node,
    Dictionary<BinaryTree, int> nodeDepths,
    Dictionary<BinaryTree, int> nodeCounts
  )
            {
                nodeDepths[node] = 0;
                if (node.Left != null)
                {
                    addNodeDepths(node.Left, nodeDepths, nodeCounts);
                    nodeDepths[node] =
                      nodeDepths[node] + nodeDepths[node.Left] + nodeCounts[node.Left];
                }
                if (node.Right != null)
                {
                    addNodeDepths(node.Right, nodeDepths, nodeCounts);
                    nodeDepths[node] =
                      nodeDepths[node] + nodeDepths[node.Right] + nodeCounts[node.Right];
                }
            }

            public static void addNodeCounts(
              BinaryTree node, Dictionary<BinaryTree, int> nodeCounts
            )
            {
                nodeCounts[node] = 1;
                if (node.Left != null)
                {
                    addNodeCounts(node.Left, nodeCounts);
                    nodeCounts[node] = nodeCounts[node] + nodeCounts[node.Left];
                }
                if (node.Right != null)
                {
                    addNodeCounts(node.Right, nodeCounts);
                    nodeCounts[node] = nodeCounts[node] + nodeCounts[node.Right];
                }
            }

            //4.
            // Average case: when the tree is balanced
            // O(n) time | O(h) space - where n is the number of nodes in
            // the Binary Tree and h is the height of the Binary Tree
            public static int AllKindsOfNodeDepths4(BinaryTree root)
            {
                return getTreeInfo(root).sumOfAllDepths;
            }
            public static TreeInfoExt getTreeInfo(BinaryTree tree)
            {
                if (tree == null)
                {
                    return new TreeInfoExt(0, 0, 0);
                }

                TreeInfoExt leftTreeInfo = getTreeInfo(tree.Left);
                TreeInfoExt rightTreeInfo = getTreeInfo(tree.Right);

                int sumOfLeftDepths =
                  leftTreeInfo.sumOfDepths + leftTreeInfo.numNodesInTree;
                int sumOfRightDepths =
                  rightTreeInfo.sumOfDepths + rightTreeInfo.numNodesInTree;

                int numNodesInTree =
                  1 + leftTreeInfo.numNodesInTree + rightTreeInfo.numNodesInTree;
                int sumOfDepths = sumOfLeftDepths + sumOfRightDepths;
                int sumOfAllDepths =
                  sumOfDepths + leftTreeInfo.sumOfAllDepths + rightTreeInfo.sumOfAllDepths;

                return new TreeInfoExt(numNodesInTree, sumOfDepths, sumOfAllDepths);
            }

            public class TreeInfoExt
            {
                public int numNodesInTree;
                public int sumOfDepths;
                public int sumOfAllDepths;

                public TreeInfoExt(int numNodesInTree, int sumOfDepths, int sumOfAllDepths)
                {
                    this.numNodesInTree = numNodesInTree;
                    this.sumOfDepths = sumOfDepths;
                    this.sumOfAllDepths = sumOfAllDepths;
                }
            }

            //5.
            // Average case: when the tree is balanced
            // O(n) time | O(h) space - where n is the number of nodes in
            // the Binary Tree and h is the height of the Binary Tree
            public static int AllKindsOfNodeDepths5(BinaryTree root)
            {
                return allKindsOfNodeDepthsHelper(root, 0, 0);
            }

            public static int allKindsOfNodeDepthsHelper(
              BinaryTree root, int depthSum, int depth
            )
            {
                if (root == null) return 0;

                depthSum += depth;
                return depthSum +
                       allKindsOfNodeDepthsHelper(root.Left, depthSum, depth + 1) +
                       allKindsOfNodeDepthsHelper(root.Right, depthSum, depth + 1);
            }
            //6.
            // Average case: when the tree is balanced
            // O(n) time | O(h) space - where n is the number of nodes in
            // the Binary Tree and h is the height of the Binary Tree
            public static int AllKindsOfNodeDepths6(BinaryTree root)
            {
                return allKindsOfNodeDepthsHelper1(root, 0);
            }

            public static int allKindsOfNodeDepthsHelper1(BinaryTree root, int depth)
            {
                if (root == null) return 0;

                // Formula to calculate 1 + 2 + 3 + ... + depth - 1 + depth
                var depthSum = (depth * (depth + 1)) / 2;
                return depthSum + allKindsOfNodeDepthsHelper1(root.Left, depth + 1) +
                       allKindsOfNodeDepthsHelper1(root.Right, depth + 1);
            }


        }


    }
}