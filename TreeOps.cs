using System;
using System.Collections.Generic;
using System.Formats.Asn1;
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


        /*
        110. Balanced Binary Tree
        https://leetcode.com/problems/balanced-binary-tree/description/

        https://www.algoexpert.io/questions/height-balanced-binary-tree

        */
        public class IsHeightBalancedBTSolution
        {
            /*
            Approach 1: Top-down recursion
            Complexity Analysis
•	Time complexity : O(nlogn)
o	For a node p at depth d, GetTreeInfo(p) to get Height will be called d times.
o	We first need to obtain a bound on the height of a balanced tree and With this bound we can guarantee that
height will be called on each node O(logn) times.
•	Space complexity : O(n). The recursion stack may contain all nodes if the tree is skewed.

            */

            public static bool TopDownRec(DataStructures.TreeNode root)
            {

                // An empty tree satisfies the definition of a balanced tree
                if (root == null)
                {
                    return true;
                }

                // Check if subtrees have height within 1. If they do, check if the
                // subtrees are balanced
                return Math.Abs(Height(root.Left) - Height(root.Right)) < 2 &&
                       TopDownRec(root.Left) && TopDownRec(root.Right);

                // Compute the tree's height via recursion
                int Height(DataStructures.TreeNode root)
                {
                    // An empty tree has height -1
                    if (root == null)
                    {
                        return -1;
                    }

                    return 1 + Math.Max(Height(root.Left), Height(root.Right));
                }


            }
            /*
            Approach 2: Bottom-up recursion
            Complexity Analysis
•	Time complexity : O(n)
For every subtree, we compute its height in constant time as well as
compare the height of its children.
•	Space complexity : O(n). The recursion stack may go up to O(n) if the tree is unbalanced.

            */

            public static bool BottomUpRec(DataStructures.TreeNode root)
            {
                return IsBalancedTreeHelper(root).IsBalanced;

                // Returns whether the tree at root is balanced, along with the tree's
                // height.
                TreeInfo IsBalancedTreeHelper(DataStructures.TreeNode root)
                {
                    // An empty tree is both balanced and has a height -1.
                    if (root == null)
                    {
                        return new TreeInfo(true, -1);
                    }

                    // Checks whether the subtrees are balanced or not.
                    var left = IsBalancedTreeHelper(root.Left);
                    if (!left.IsBalanced)
                    {
                        return new TreeInfo(false, -1);
                    }

                    var right = IsBalancedTreeHelper(root.Right);
                    if (!right.IsBalanced)
                    {
                        return new TreeInfo(false, -1);
                    }

                    // The obtained height from recursive calls can also determine
                    // that the current node is balanced.
                    if (Math.Abs(left.Height - right.Height) < 2)
                    {
                        return new TreeInfo(true, Math.Max(left.Height, right.Height) + 1);
                    }

                    return new TreeInfo(false, -1);
                }
            }

        }
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


        /*
         //https://www.algoexpert.io/questions/flatten-binary-tree

        */
        public class FlattenBinaryTreeSol
        {
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
        }


        /*
        114. Flatten Binary Tree to Linked List
    https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/

        */
        public class FlattenBinaryTreeToLinkedListSol
        {
            /*
            Approach 1: Recursion
Complexity Analysis
•	Time Complexity: O(N) since we process each node of the tree exactly once.
•	Space Complexity: O(N) which is occupied by the recursion stack. The problem statement doesn't mention anything about the tree being balanced or not and hence, the tree could be e.g. left skewed and in that case the longest branch (and hence the number of nodes in the recursion stack) would be N.

   */
            public static void Rec(DataStructures.TreeNode root)
            {
                FlattenTree(root);
                DataStructures.TreeNode FlattenTree(DataStructures.TreeNode node)
                {
                    // Handle the null scenario
                    if (node == null)
                    {
                        return null;
                    }

                    // For a leaf node, we simply return the
                    // node as is.
                    if (node.Left == null && node.Right == null)
                    {
                        return node;
                    }

                    // Recursively flatten the left subtree
                    DataStructures.TreeNode leftTail = FlattenTree(node.Left);
                    // Recursively flatten the right subtree
                    DataStructures.TreeNode rightTail = FlattenTree(node.Right);
                    // If there was a left subtree, we shuffle the connections
                    // around so that there is nothing on the left side
                    // anymore.
                    if (leftTail != null)
                    {
                        leftTail.Right = node.Right;
                        node.Right = node.Left;
                        node.Left = null;
                    }

                    // We need to return the "rightmost" node after we are
                    // done wiring the new connections.
                    return rightTail == null ? leftTail : rightTail;
                }
            }


            /*
            Approach 2: Iterative Solution using Stack
Complexity Analysis
•	Time Complexity: O(N) since we process each node of the tree exactly once.
•	Space Complexity: O(N) which is occupied by the stack. The problem statement doesn't mention anything about the tree being balanced or not and hence, the tree could be e.g. left skewed and in that case the longest branch (and hence the number of nodes in the recursion stack) would be N.

            */
            public static void IterateUsingStack(DataStructures.TreeNode root)
            {
                if (root == null)
                {
                    return;
                }

                int START = 1, END = 2;
                DataStructures.TreeNode tail = null;
                Stack<(TreeNode, int)> stack = new Stack<(TreeNode, int)>();
                stack.Push(GetPair(root, START));
                while (stack.Count > 0)
                {
                    var nodeData = stack.Pop();
                    DataStructures.TreeNode node = nodeData.Item1;
                    int state = nodeData.Item2;
                    if (node.Left == null && node.Right == null)
                    {
                        tail = node;
                        continue;
                    }

                    if (state == START)
                    {
                        if (node.Left != null)
                        {
                            stack.Push(GetPair(node, END));
                            stack.Push(GetPair(node.Left, START));
                        }
                        else if (node.Right != null)
                        {
                            stack.Push(GetPair(node.Right, START));
                        }
                    }
                    else
                    {
                        DataStructures.TreeNode rightNode = node.Right;
                        if (tail != null)
                        {
                            tail.Right = node.Right;
                            node.Right = node.Left;
                            node.Left = null;
                            rightNode = tail.Right;
                        }

                        if (rightNode != null)
                        {
                            stack.Push(GetPair(rightNode, START));
                        }
                    }
                }
                (TreeNode, int) GetPair(TreeNode n, int v)
                {
                    return (n, v);
                }
            }
            /*
            Approach 3: O(1) Iterative Solution
Complexity Analysis
•	Time Complexity: O(N) since we process each node of the tree at most twice. If you think about it, we process the nodes once when we actually run our algorithm on them as the currentNode. The second time when we come across the nodes is when we are trying to find our rightmost node. Sure, this algorithm is slower than the previous two approaches but it doesn't use any additional space which is a big win.
•	Space Complexity: O(1) 

            */
            public void IterateWithConstantSpace(DataStructures.TreeNode root)
            {
                // Handle the null scenario
                if (root == null)
                    return;
                DataStructures.TreeNode node = root;
                while (node != null)
                {
                    // If the node has a left child
                    if (node.Left != null)
                    {
                        // Find the rightmost node
                        DataStructures.TreeNode rightmost = node.Left;
                        while (rightmost.Right != null)
                        {
                            rightmost = rightmost.Right;
                        }

                        // rewire the connections
                        rightmost.Right = node.Right;
                        node.Right = node.Left;
                        node.Left = null;
                    }

                    // move on to the right side of the tree
                    node = node.Right;
                }
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
        /*
        112. Path Sum
        https://leetcode.com/problems/path-sum/description/	
        */
        public bool HasPathSum(DataStructures.TreeNode root, int targetSum)
        {
            /*

Approach 1: Recursion
Complexity Analysis
•	Time complexity : we visit each node exactly once, thus the time complexity is O(N), where N is the number of nodes.
•	Space complexity : in the worst case, the tree is completely unbalanced, e.g. each node has only one child node, the recursion call would occur N times (the height of the tree), therefore the storage to keep the call stack would be O(N). But in the best case (the tree is completely balanced), the height of the tree would be log(N). Therefore, the space complexity in this case would be O(log(N)).


            */
            bool hasPathSum = HasPathSumRec(root, targetSum);
            /*

Approach 2: Iterations
Complexity Analysis
•	Time complexity: the same as the recursion approach O(N).
•	Space complexity: O(N) since in the worst case, when the tree is completely unbalanced, e.g. each node has only one child node, we would keep all N nodes in the stack. But in the best case (the tree is balanced), the height of the tree would be log(N). Therefore, the space complexity in this case would be O(log(N)).

            */

            hasPathSum = HasPathSumIterative(root, targetSum);

            return hasPathSum;

        }

        public bool HasPathSumRec(DataStructures.TreeNode root, int sum)
        {
            if (root == null)
                return false;
            sum -= root.Val;
            if ((root.Left == null) && (root.Right == null))  // if reach a leaf
                return sum == 0;
            return HasPathSum(root.Left, sum) || HasPathSum(root.Right, sum);
        }
        public bool HasPathSumIterative(DataStructures.TreeNode root, int sum)
        {
            if (root == null)
                return false;
            Stack<DataStructures.TreeNode> nodeStack = new Stack<DataStructures.TreeNode>();
            Stack<int> sumStack = new Stack<int>();
            nodeStack.Push(root);
            sumStack.Push(sum - root.Val);
            while (nodeStack.Count > 0)
            {
                DataStructures.TreeNode node = nodeStack.Pop();
                int currSum = sumStack.Pop();
                if (node.Left == null && node.Right == null && currSum == 0)
                    return true;
                if (node.Left != null)
                {
                    nodeStack.Push(node.Left);
                    sumStack.Push(currSum - node.Left.Val);
                }

                if (node.Right != null)
                {
                    nodeStack.Push(node.Right);
                    sumStack.Push(currSum - node.Right.Val);
                }
            }

            return false;
        }

        /*
        113. Path Sum II
        https://leetcode.com/problems/path-sum-ii/description/

        Approach: Depth First Traversal | Recursion
        Complexity Analysis
        •	Time Complexity: O(N^2) where N are the number of nodes in a tree. In the worst case, we could have a complete binary tree and if that is the case, then there would be N/2 leafs. For every leaf, we perform a potential O(N) operation of copying over the pathNodes nodes to a new list to be added to the final pathsList. Hence, the complexity in the worst case could be O(N^2).
        •	Space Complexity: O(N). The space complexity, like many other problems is debatable here. I personally choose not to consider the space occupied by the output in the space complexity. So, all the new lists that we create for the paths are actually a part of the output and hence, don't count towards the final space complexity. The only additional space that we use is the pathNodes list to keep track of nodes along a branch.
        We could include the space occupied by the new lists (and hence the output) in the space complexity and in that case the space would be O(N2). There's a great answer on Stack Overflow about whether to consider input and output space in the space complexity or not. I prefer not to include them.

        */
        public IList<IList<int>> PathSum(DataStructures.TreeNode root, int targetSum)
        {
            List<IList<int>> pathsList = new List<IList<int>>();
            List<int> pathNodes = new List<int>();
            this.RecurseTree(root, 0, pathNodes, pathsList);
            return pathsList;

        }
        private void RecurseTree(DataStructures.TreeNode node, int remainingSum,
                             List<int> pathNodes, IList<IList<int>> pathsList)
        {
            if (node == null)
            {
                return;
            }

            // Add the current node to the path's list
            pathNodes.Add(node.Val);
            // Check if the current node is a leaf and also, if it
            // equals our remaining sum. If it does, we add the path to
            // our list of paths
            if (remainingSum == node.Val && node.Left == null &&
                node.Right == null)
            {
                pathsList.Add(new List<int>(pathNodes));
            }
            else
            {
                // Else, we will recurse on the left and the right children
                this.RecurseTree(node.Left, remainingSum - node.Val, pathNodes,
                                 pathsList);
                this.RecurseTree(node.Right, remainingSum - node.Val, pathNodes,
                                 pathsList);
            }

            // We need to pop the node once we are done processing ALL of it's
            // subtrees.
            pathNodes.RemoveAt(pathNodes.Count - 1);
        }

        /*
        437. Path Sum III
        https://leetcode.com/problems/path-sum-iii/description/

        Approach 1: Prefix Sum
        Complexity Analysis
        •	Time complexity: O(N), where N is a number of nodes. During preorder traversal, each node is visited once.
        •	Space complexity: up to O(N) to keep the hashmap of prefix sums, where N is a number of nodes

        */
        public int PathSumIII(DataStructures.TreeNode root, int targetSum)
        {
            k = targetSum;
            Preorder(root, 0L);
            return count;
        }
        int count = 0;
        int k;
        Dictionary<long, int> hashMap = new Dictionary<long, int>();

        public void Preorder(DataStructures.TreeNode node, long currSum)
        {
            if (node == null)
                return;

            // The current prefix sum
            currSum += node.Val;

            // Here is the sum we're looking for
            if (currSum == k)
                count++;

            // The number of times the curr_sum − k has occurred already, 
            // determines the number of times a path with sum k 
            // has occurred up to the current node
            count += hashMap.GetValueOrDefault(currSum - k, 0);

            //Add the current sum into the hashmap
            // to use it during the child node's processing
            hashMap.Add(currSum, hashMap.GetValueOrDefault(currSum, 0) + 1);

            // Process the left subtree
            Preorder(node.Left, currSum);

            // Process the right subtree
            Preorder(node.Right, currSum);

            // Remove the current sum from the hashmap
            // in order not to use it during 
            // the parallel subtree processing
            hashMap.Add(currSum, hashMap[currSum] - 1);
        }
        /*
        666. Path Sum IV		
        https://leetcode.com/problems/path-sum-iv/description/

        */
        public int PathSum(int[] nums)
        {
            /*
Approach 1: Depth First Search
Complexity Analysis
Let n be the number of nodes in the tree.
•	Time complexity: O(n)
All hashmap insertion and search operations take constant time. Apart from this, in the dfs function, we visit all the nodes of the tree exactly once. Therefore, the time complexity is given by O(n).
•	Space complexity: O(n)
We perform exactly n insertion operations in the hashmap. For the dfs function, the stack space can go up to n in the worst case. Therefore, the total space complexity is given by O(n).

            */
            int pathSum = PathSumDFS(nums);

            /*
 Approach 2: Breadth First Search           
 Complexity Analysis
Let n be the number of nodes in the tree.
•	Time complexity: O(n)
All hashmap insertion and search operations take constant time. Apart from this, in the breadth-first search, we visit all the nodes of the tree exactly once. Therefore, the time complexity is given by O(n).
•	Space complexity: O(n)
We perform exactly n insertion operations in the hashmap. For the breadth-first search, the queue q stores all the elements exactly once. Therefore, the total space complexity is given by O(n).

            */

            pathSum = PathSumBFS(nums);

            return pathSum;

        }

        Dictionary<int, int> map = new Dictionary<int, int>();
        public int PathSumDFS(int[] nums)
        {
            if (nums == null || nums.Length == 0) return 0;

            // Store the data in a hashmap, with the coordinates as the key and the
            // node value as the value
            foreach (int num in nums)
            {
                int key = num / 10;
                int value = num % 10;
                map.Add(key, value);
            }

            return PathSumDFS(nums[0] / 10, 0);
        }

        private int PathSumDFS(int root, int preSum)
        {
            // Find the level and position values from the coordinates
            int level = root / 10;
            int pos = root % 10;

            //the left child and right child position in the tree
            int left = (level + 1) * 10 + pos * 2 - 1;
            int right = (level + 1) * 10 + pos * 2;
            int currSum = preSum + map[root];

            // If the node is a leaf node, return its root to leaf path sum.
            if (!map.ContainsKey(left) && !map.ContainsKey(right))
            {
                return currSum;
            }

            // Otherwise iterate through the left and right children recursively
            // using depth first search
            int leftSum = map.ContainsKey(left) ? PathSumDFS(left, currSum) : 0;
            int rightSum = map.ContainsKey(right) ? PathSumDFS(right, currSum) : 0;

            //return the total path sum of the tree rooted at the current node
            return leftSum + rightSum;
        }

        public int PathSumBFS(int[] nums)
        {
            if (nums.Length == 0)
            {
                return 0;
            }

            // Store the node values in a hashmap, using coordinates as the key.
            Dictionary<int, int> map = new Dictionary<int, int>();
            foreach (int element in nums)
            {
                int coordinates = element / 10;
                int value = element % 10;
                map.Add(coordinates, value);
            }

            // Initialize the BFS queue and start with the root node.
            Queue<(int, int)> q = new Queue<(int, int)>();
            int totalSum = 0;

            int rootCoordinates = nums[0] / 10;
            q.Enqueue(
                (
                    rootCoordinates,
                    map[rootCoordinates]
                )
            );

            while (q.Count > 0)
            {
                (int coordinates, int currentSum) = q.Dequeue();

                int level = coordinates / 10;
                int position = coordinates % 10;

                // Find the left and right child coordinates.
                int left = (level + 1) * 10 + position * 2 - 1;
                int right = (level + 1) * 10 + position * 2;

                // If it's a leaf node (no left and right children), add currentSum to totalSum.
                if (!map.ContainsKey(left) && !map.ContainsKey(right))
                {
                    totalSum += currentSum;
                }

                // Add the left child to the queue if it exists.
                if (map.ContainsKey(left))
                {
                    q.Enqueue(
                        (left, currentSum + map[left])
                    );
                }

                // Add the right child to the queue if it exists.
                if (map.ContainsKey(right))
                {
                    q.Enqueue(
                        (
                            right,
                            currentSum + map[right]
                        )
                    );
                }
            }

            return totalSum;
        }
        /*
        124. Binary Tree Maximum Path Sum
        https://leetcode.com/problems/binary-tree-maximum-path-sum/description/

        Approach: Post Order DFS (PODFS)
        Complexity Analysis
        Let n be the number of nodes in the tree.
        •	Time complexity: O(n)
        Each node in the tree is visited only once. During a visit, we perform constant time operations, including two recursive calls and calculating the max path sum for the current node. So the time complexity is O(n).
        •	Space complexity: O(n)
        We don't use any auxiliary data structure, but the recursive call stack can go as deep as the tree's height. In the worst case, the tree is a linked list, so the height is n. Therefore, the space complexity is O(n).

        */
        private int maxSum = int.MinValue;
        public int BinaryTreeMaxPathSum(DataStructures.TreeNode root)
        {
            GainFromSubtree(root);
            return maxSum;

        }

        // post order traversal of subtree rooted at `root`
        private int GainFromSubtree(DataStructures.TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }

            // add the path sum from left subtree. Note that if the path
            // sum is negative, we can ignore it, or count it as 0.
            // This is the reason we use `Math.Max` here.
            int gainFromLeft = Math.Max(GainFromSubtree(root.Left), 0);
            // add the path sum from right subtree. 0 if negative
            int gainFromRight = Math.Max(GainFromSubtree(root.Right), 0);
            maxSum = Math.Max(maxSum, gainFromLeft + gainFromRight + root.Val);
            // return the max sum for a path starting at the root of subtree
            return Math.Max(gainFromLeft + root.Val, gainFromRight + root.Val);
        }
        /*
    1376. Time Needed to Inform All Employees
    https://leetcode.com/problems/time-needed-to-inform-all-employees/description/

        */

        public int NumOfMinutesToInform(int n, int headID, int[] manager, int[] informTime)
        {
            /*
   Approach 1: Depth-First Search (DFS)         
   Complexity Analysis
Here, N is the number of employees.
•	Time complexity: O(N).
We first iterate over the employees to create the adjacency list; then, we perform the DFS, where we iterate over each node once to find when they get the information from headID.
•	Space complexity: O(N).
The size of the adjacency list is N, and there will be only N−1 edges in the tree. There will be some stack space needed for DFS. The maximum active stack calls would equal the number of nodes for a skewed tree. Hence the total space complexity would be O(N).

            */
            int numOfMinutesToInform = NumOfMinutesToInformDFS(n, headID, manager, informTime);
            /*
 Approach 2: Breadth-First Search (BFS)           
  Complexity Analysis
Here, N is the number of employees.
•	Time complexity: O(N).
We first iterate over the employees to create the adjacency list; then, we perform the BFS, where we iterate over each node once to find when they get the information from headID.
•	Space complexity: O(N).
The size of the adjacency list is N, and there will be only N−1 edges in the tree. Also, the size of the queue could be at max O(N). Hence the total space complexity would be O(N).

            */
            numOfMinutesToInform = NumOfMinutesToInformBFS(n, headID, manager, informTime);

            return numOfMinutesToInform;

        }

        int maxTime = int.MinValue;

        void NumOfMinutesToInformDFSRec(List<List<int>> adjacencyList, int[] informTime, int currentEmployee, int totalTime)
        {
            // Maximum time for an employee to get the news.
            maxTime = Math.Max(maxTime, totalTime);

            foreach (int subordinate in adjacencyList[currentEmployee])
            {
                // Visit the subordinate employee who gets the news after informTime[currentEmployee] unit time.
                NumOfMinutesToInformDFSRec(adjacencyList, informTime, subordinate, totalTime + informTime[currentEmployee]);
            }
        }

        public int NumOfMinutesToInformDFS(int numberOfEmployees, int headID, int[] manager, int[] informTime)
        {
            List<List<int>> adjacencyList = new List<List<int>>(numberOfEmployees);

            for (int i = 0; i < numberOfEmployees; i++)
            {
                adjacencyList.Add(new List<int>());
            }

            // Making an adjacent list, each index stores the Ids of subordinate employees.
            for (int i = 0; i < numberOfEmployees; i++)
            {
                if (manager[i] != -1)
                {
                    adjacencyList[manager[i]].Add(i);
                }
            }

            NumOfMinutesToInformDFSRec(adjacencyList, informTime, headID, 0);
            return maxTime;
        }

        public int NumOfMinutesToInformBFS(int numberOfEmployees, int headID, int[] manager, int[] informTime)
        {
            List<List<int>> adjacencyList = new List<List<int>>(numberOfEmployees);

            for (int i = 0; i < numberOfEmployees; i++)
            {
                adjacencyList.Add(new List<int>());
            }

            // Making an adjacent list, each index stores the Ids of subordinate employees.
            for (int i = 0; i < numberOfEmployees; i++)
            {
                if (manager[i] != -1)
                {
                    adjacencyList[manager[i]].Add(i);
                }
            }

            Queue<Tuple<int, int>> queue = new Queue<Tuple<int, int>>();
            queue.Enqueue(Tuple.Create(headID, 0));

            while (queue.Count > 0)
            {
                Tuple<int, int> employeeTuple = queue.Dequeue();

                int parent = employeeTuple.Item1;
                int time = employeeTuple.Item2;
                // Maximum time for an employee to get the news.
                maxTime = Math.Max(maxTime, time);

                foreach (int subordinate in adjacencyList[parent])
                {
                    queue.Enqueue(Tuple.Create(subordinate, time + informTime[parent]));
                }
            }

            return maxTime;
        }

        /*
104. Maximum Depth of Binary Tree
https://leetcode.com/problems/maximum-depth-of-binary-tree/description/
        */
        public class MaxDepthOfBTSol
        {
            public int MaxDepth(DataStructures.TreeNode root)
            {
                /*

    Approach 1: Recursion
    Complexity analysis
    •	Time complexity: we visit each node exactly once, thus the time complexity is O(N), where N is the number of nodes.
    •	Space complexity: in the worst case, the tree is completely unbalanced, e.g. each node has only left child node, the recursion call would occur N times (the height of the tree), therefore the storage to keep the call stack would be O(N). But in the best case (the tree is completely balanced), the height of the tree would be log(N). Therefore, the space complexity in this case would be O(log(N)).


                */
                int maxDepth = MaxDepthRec(root);
                /*

    Approach 2: Tail Recursion + BFS (BFSTC)
    Complexity analysis
    •	Time complexity: O(N), still we visit each node once and only once.
    •	Space complexity: O(2^(log2N−1))=O(N/2)=O(N), i.e. the maximum number of nodes at the same level (the number of leaf nodes in a full binary tree), since we traverse the tree in the BFS manner.

                */
                maxDepth = MaxDepthBFSTC(root);
                /*

            Approach 3: Iteration
            Complexity analysis
            •	Time complexity: O(N).
            •	Space complexity: in the worst case, the tree is completely unbalanced, e.g. each node has only the left child node, the recursion call would occur N times (the height of the tree), the storage to keep the call stack would be O(N). But in the average case (the tree is balanced), the height of the tree would be log(N). Therefore, the space complexity in this case would be O(log(N)).

                */
                maxDepth = MaxDepthBFS(root);

                return maxDepth;

            }

            public int MaxDepthRec(DataStructures.TreeNode root)
            {
                if (root == null)
                {
                    return 0;
                }
                else
                {
                    int left_height = MaxDepth(root.Left);
                    int right_height = MaxDepth(root.Right);
                    return 1 + Math.Max(left_height, right_height);
                }
            }
            private Queue<Tuple<DataStructures.TreeNode, int>> next_items =
        new Queue<Tuple<DataStructures.TreeNode, int>>();

            private int max_depth = 0;

            private int NextMaxDepth()
            {
                if (next_items.Count == 0)
                {
                    return max_depth;
                }

                Tuple<DataStructures.TreeNode, int> next_item = next_items.Dequeue();
                DataStructures.TreeNode next_node = next_item.Item1;
                int next_level = next_item.Item2 + 1;
                max_depth = Math.Max(max_depth, next_level);
                // Add the nodes to visit in the following recursive calls.
                if (next_node.Left != null)
                {
                    next_items.Enqueue(
                        new Tuple<DataStructures.TreeNode, int>(next_node.Left, next_level));
                }

                if (next_node.Right != null)
                {
                    next_items.Enqueue(
                        new Tuple<DataStructures.TreeNode, int>(next_node.Right, next_level));
                }

                return NextMaxDepth();
            }

            public int MaxDepthBFSTC(DataStructures.TreeNode root)
            {
                if (root == null)
                    return 0;
                // Clear the previous queue.
                next_items.Clear();
                max_depth = 0;
                // Push the root node into the queue to kick off the next visit.
                next_items.Enqueue(new Tuple<DataStructures.TreeNode, int>(root, 0));
                return NextMaxDepth();
            }

            public int MaxDepthBFS(DataStructures.TreeNode root)
            {
                if (root == null)
                {
                    return 0;
                }

                var stack = new Stack<(TreeNode, int)>();
                stack.Push((root, 1));
                int depth = 0;
                while (stack.Count != 0)
                {
                    var current = stack.Pop();
                    depth = Math.Max(depth, current.Item2);
                    if (current.Item1.Left != null)
                    {
                        stack.Push((current.Item1.Left, current.Item2 + 1));
                    }

                    if (current.Item1.Right != null)
                    {
                        stack.Push((current.Item1.Right, current.Item2 + 1));
                    }
                }

                return depth;
            }
        }

        /*
        111. Minimum Depth of Binary Tree
        https://leetcode.com/problems/minimum-depth-of-binary-tree/description/

        */
        public class MinDepthOfBTSol
        {
            /*
            Approach 1: Depth-First Search (DFS)
            Complexity Analysis
Here, N is the number of nodes in the binary tree.
•	Time complexity: O(N)
We will traverse each node in the tree only once; hence, the total time complexity would be O(N).
•	Space complexity: O(N)
The only space required is the stack space; the maximum number of active stack calls would equal the maximum depth of the tree, which could equal the total number of nodes in the tree. Hence, the space complexity would equal O(N).

            */

            public int DFS(DataStructures.TreeNode root)
            {
                return DFSRec(root);

                int DFSRec(DataStructures.TreeNode root)
                {
                    if (root == null)
                    {
                        return 0;
                    }

                    // If only one of child is non-null, then go into that recursion.
                    if (root.Left == null)
                    {
                        return 1 + DFSRec(root.Right);
                    }
                    else if (root.Right == null)
                    {
                        return 1 + DFSRec(root.Left);
                    }

                    // Both children are non-null, hence call for both children.
                    return 1 + Math.Min(DFSRec(root.Left), DFSRec(root.Right));
                }

            }
            /*
Approach 2: Breadth-First Search (BFS)

            */
            public int BFS(DataStructures.TreeNode root)
            {
                if (root == null)
                {
                    return 0;
                }

                Queue<DataStructures.TreeNode> q = new Queue<DataStructures.TreeNode>();
                q.Enqueue(root);
                int depth = 1;
                while (q.Count != 0)
                {
                    int qSize = q.Count;
                    for (int i = 0; i < qSize; i++)
                    {
                        DataStructures.TreeNode node = q.Dequeue();
                        // Since we added nodes without checking null, we need to skip
                        // them here.
                        if (node == null)
                        {
                            continue;
                        }

                        // The first leaf would be at minimum depth, hence return it.
                        if (node.Left == null && node.Right == null)
                        {
                            return depth;
                        }

                        q.Enqueue(node.Left);
                        q.Enqueue(node.Right);
                    }

                    depth++;
                }

                return -1;
            }
        }

        /*
        2385. Amount of Time for Binary Tree to Be Infected
        https://leetcode.com/problems/amount-of-time-for-binary-tree-to-be-infected/

        */
        public int AmountOfTimeToGetInfected(TreeNodeExt root, int start)
        {
            /*
            Approach 1: Convert to Graph and Breadth-First Search (CGBFS)
            Complexity Analysis
Let n be the number of nodes in the tree.
•	Time complexity: O(n)
Converting the tree to a graph using a preorder traversal costs O(n). We then perform BFS, which also costs O(n) because we don't visit a node more than once.
•	Space complexity: O(n)
When converting the tree to a graph, we require O(n) extra space for the map. We also require O(n) space for the queue and O(n) space for the visited set during the BFS.

            */
            int amountOfTimeToGetInfected = AmountOfTimeToGetInfectedCGBFS(root, start);

            /*            
Approach 2: One-Pass Depth-First Search (OPDFS)
Complexity
Let n be the number of nodes in the tree.
•	Time complexity: O(n)
Traversing the tree with a DFS costs O(n) as we visit each node exactly once.
•	Space complexity: O(n)
The space complexity of DFS is determined by the maximum depth of the call stack, which corresponds to the height of the tree (or the graph in our case). In the worst case, if the tree is completely unbalanced (e.g., a linked list), the call stack can grow as deep as the number of nodes, resulting in a space complexity of O(n).

            */
            amountOfTimeToGetInfected = AmountOfTimeToGetInfectedOPDFS(root, start);

            return amountOfTimeToGetInfected;
        }

        public int AmountOfTimeToGetInfectedCGBFS(TreeNodeExt root, int start)
        {
            Dictionary<int, HashSet<int>> adjacencyMap = new Dictionary<int, HashSet<int>>();
            Convert(root, 0, adjacencyMap);
            Queue<int> queue = new Queue<int>();
            queue.Enqueue(start);
            int minute = 0;
            HashSet<int> visitedNodes = new HashSet<int>();
            visitedNodes.Add(start);

            while (queue.Count > 0)
            {
                int levelSize = queue.Count;
                while (levelSize > 0)
                {
                    int currentNode = queue.Dequeue();
                    foreach (int adjacentNode in adjacencyMap[currentNode])
                    {
                        if (!visitedNodes.Contains(adjacentNode))
                        {
                            visitedNodes.Add(adjacentNode);
                            queue.Enqueue(adjacentNode);
                        }
                    }
                    levelSize--;
                }
                minute++;
            }
            return minute - 1;
        }

        public void Convert(TreeNodeExt currentNode, int parentValue, Dictionary<int, HashSet<int>> adjacencyMap)
        {
            if (currentNode == null)
            {
                return;
            }
            if (!adjacencyMap.ContainsKey(currentNode.Val))
            {
                adjacencyMap[currentNode.Val] = new HashSet<int>();
            }
            HashSet<int> adjacentList = adjacencyMap[currentNode.Val];
            if (parentValue != 0)
            {
                adjacentList.Add(parentValue);
            }
            if (currentNode.Left != null)
            {
                adjacentList.Add(currentNode.Left.Val);
            }
            if (currentNode.Right != null)
            {
                adjacentList.Add(currentNode.Right.Val);
            }
            Convert(currentNode.Left, currentNode.Val, adjacencyMap);
            Convert(currentNode.Right, currentNode.Val, adjacencyMap);
        }
        public class TreeNodeExt
        {
            public int Val;
            public TreeNodeExt Left;
            public TreeNodeExt Right;
            public TreeNodeExt Parent;
            public TreeNodeExt() { }

            public TreeNodeExt(int value)
            {
                this.Val = value;
            }

            public TreeNodeExt(int value, TreeNodeExt left, TreeNodeExt right)
            {
                this.Val = value;
                this.Left = left;
                this.Right = right;
            }

            public TreeNodeExt(int val = 0, TreeNodeExt left = null, TreeNodeExt right = null, TreeNodeExt parent = null)
            {
                this.Val = val;
                this.Left = left;
                this.Right = right;
                this.Parent = parent;
            }
        }

        private int maxDistance = 0;

        public int AmountOfTimeToGetInfectedOPDFS(TreeNodeExt root, int start)
        {
            Traverse(root, start);
            return maxDistance;
        }

        public int Traverse(TreeNodeExt root, int start)
        {
            int depth = 0;
            if (root == null)
            {
                return depth;
            }

            int leftDepth = Traverse(root.Left, start);
            int rightDepth = Traverse(root.Right, start);

            if (root.Val == start)
            {
                maxDistance = Math.Max(leftDepth, rightDepth);
                depth = -1;
            }
            else if (leftDepth >= 0 && rightDepth >= 0)
            {
                depth = Math.Max(leftDepth, rightDepth) + 1;
            }
            else
            {
                int distance = Math.Abs(leftDepth) + Math.Abs(rightDepth);
                maxDistance = Math.Max(maxDistance, distance);
                depth = Math.Min(leftDepth, rightDepth) - 1;
            }

            return depth;
        }

        /*
        863. All Nodes Distance K in Binary Tree
        https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/description/	

        */
        public IList<int> AllNodesDistanceK(TreeNodeExt root, TreeNodeExt target, int k)
        {
            /*
Approach 1: Implementing Parent Pointers (IPP)

Complexity Analysis
Let n be the number of nodes in the binary tree.
•	Time complexity: O(n)
o	Both add_parent and dfs recursively call themselves to process the left and right subtrees of the current node cur. Each node is visited once by each function.
•	Space complexity: O(n)
o	visited stores a maximum of O(n) visited nodes.
o	The recursive solution uses the call stack to keep track of the current subtree being processed. The maximum depth of the call stack is equal to the height of the given tree. In the worst-case scenario, the given binary tree may be a degenerate binary tree and the stack can hold up to n calls, resulting in a space complexity of O(n).

            */
            IList<int> allNodesDistanceK = AllNodesDistanceKIPP(root, target, k);
            /*
 Approach 2: Depth-First Search on Equivalent Graph      (DFSEG)
  Complexity Analysis
Let n be the number of nodes in the binary tree.
•	Time complexity: O(n)
o	build_graph recursively calls itself to process the left and right subtrees of the current node cur. Each node is visited once.
o	dfs recursively calls itself to process the unvisited neighbors of the current node cur. Each node is visited once.
•	Space complexity: O(n)
o	We use a hash map graph to store all edges, which requires O(n) space for n−1 edges.
o	We use a hash set visited to record the visited nodes, which takes O(n) space.
o	The recursive solution uses the call stack to keep track of the current subtree being processed. The maximum depth of the call stack is equal to the height of the given tree. In the worst-case scenario, it may be a degenerate binary tree and the stack can hold up to n calls, resulting in a space complexity of O(n).

            */
            allNodesDistanceK = AllNodesDistanceKDFSEG(root, target, k);

            /*
Approach 3: Breadth-First Search on Equivalent Graph (BFSEG)

 Complexity Analysis
Let n be the number of nodes.
•	Time complexity: O(n)
o	build_graph recursively calls itself to process the left and right subtrees of the current node cur. Each node is visited once.
o	In a typical BFS search, the time complexity is O(V+E) where V is the number of vertices and E is the number of edges. There are n nodes and n−1 edges in this problem. Each node is added to the queue and popped from the queue once, it takes O(n) to handle all nodes.
•	Space complexity: O(n)
o	We use a hash map graph to store all edges, which requires O(n) space for n−1 edges.
o	We use a hash set visited to record the visited nodes, which takes O(n) space.
o	There may be up to n nodes stored in queue and O(n) space is required.
o	Therefore, the space complexity is O(n).            

            */
            allNodesDistanceK = AllNodesDistanceKBFSEG(root, target, k);

            return allNodesDistanceK;


        }

        public IList<int> AllNodesDistanceKIPP(TreeNodeExt root, TreeNodeExt target, int k)
        {
            // Recursively add a parent pointer to each node.
            void AddParent(TreeNodeExt cur, TreeNodeExt parent)
            {
                if (cur != null)
                {
                    cur.Parent = parent;
                    AddParent(cur.Left, cur);
                    AddParent(cur.Right, cur);
                }
            }
            AddParent(root, null);

            var answer = new List<int>();
            var visited = new HashSet<TreeNodeExt>();

            void Dfs(TreeNodeExt cur, int distance)
            {
                if (cur == null || visited.Contains(cur))
                    return;

                visited.Add(cur);

                if (distance == 0)
                {
                    answer.Add(cur.Val);
                    return;
                }

                Dfs(cur.Parent, distance - 1);
                Dfs(cur.Left, distance - 1);
                Dfs(cur.Right, distance - 1);
            }

            Dfs(target, k);

            return answer;
        }

        private Dictionary<int, List<int>> graph;
        private List<int> answer;
        private HashSet<int> visited;

        public List<int> AllNodesDistanceKDFSEG(TreeNodeExt root, TreeNodeExt target, int k)
        {
            graph = new Dictionary<int, List<int>>();
            BuildGraph(root, null);

            answer = new List<int>();
            visited = new HashSet<int>();
            visited.Add(target.Val);

            Dfs(target.Val, 0, k);

            return answer;
        }

        // Recursively build the undirected graph from the given binary tree.
        private void BuildGraph(TreeNodeExt current, TreeNodeExt parent)
        {
            if (current != null && parent != null)
            {
                if (!graph.ContainsKey(current.Val))
                {
                    graph[current.Val] = new List<int>();
                }
                graph[current.Val].Add(parent.Val);

                if (!graph.ContainsKey(parent.Val))
                {
                    graph[parent.Val] = new List<int>();
                }
                graph[parent.Val].Add(current.Val);
            }
            if (current.Left != null)
            {
                BuildGraph(current.Left, current);
            }
            if (current.Right != null)
            {
                BuildGraph(current.Right, current);
            }
        }

        private void Dfs(int current, int distance, int k)
        {
            if (distance == k)
            {
                answer.Add(current);
                return;
            }
            foreach (int neighbor in graph.ContainsKey(current) ? graph[current] : new List<int>())
            {
                if (!visited.Contains(neighbor))
                {
                    visited.Add(neighbor);
                    Dfs(neighbor, distance + 1, k);
                }
            }
        }

        public IList<int> AllNodesDistanceKBFSEG(TreeNodeExt root, TreeNodeExt target, int k)
        {
            Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
            DfsBuild(root, null, graph);

            List<int> answer = new List<int>();
            HashSet<int> visited = new HashSet<int>();
            Queue<(int node, int distance)> queue = new Queue<(int, int)>();

            // Add the target node to the queue with a distance of 0
            queue.Enqueue((target.Val, 0));
            visited.Add(target.Val);

            while (queue.Count > 0)
            {
                var (node, distance) = queue.Dequeue();

                // If the current node is at distance k from target,
                // add it to the answer list and continue to the next node.
                if (distance == k)
                {
                    answer.Add(node);
                    continue;
                }

                // Add all unvisited neighbors of the current node to the queue.
                if (graph.ContainsKey(node))
                {
                    foreach (int neighbor in graph[node])
                    {
                        if (!visited.Contains(neighbor))
                        {
                            visited.Add(neighbor);
                            queue.Enqueue((neighbor, distance + 1));
                        }
                    }
                }
            }

            return answer;
        }

        // Recursively build the undirected graph from the given binary tree.
        private void DfsBuild(TreeNodeExt cur, TreeNodeExt parent, Dictionary<int, List<int>> graph)
        {
            if (cur != null && parent != null)
            {
                int curVal = cur.Val, parentVal = parent.Val;
                if (!graph.ContainsKey(curVal)) graph[curVal] = new List<int>();
                if (!graph.ContainsKey(parentVal)) graph[parentVal] = new List<int>();
                graph[curVal].Add(parentVal);
                graph[parentVal].Add(curVal);
            }

            if (cur != null && cur.Left != null)
            {
                DfsBuild(cur.Left, cur, graph);
            }

            if (cur != null && cur.Right != null)
            {
                DfsBuild(cur.Right, cur, graph);
            }
        }

        /*
        129. Sum Root to Leaf Numbers
        https://leetcode.com/problems/sum-root-to-leaf-numbers/description/

        */
        public class SumRootToLeafNumbersSol
        {
            public int SumRootToLeafNumbers(DataStructures.TreeNode root)
            {
                /*
            Approach 1: Iterative Preorder Traversal. (IPT)   
            Complexity Analysis
    •	Time complexity: O(N) since one has to visit each node.
    •	Space complexity: up to O(H) to keep the stack, where H is a tree height.

                */
                int sumRootToLeafNumbers = SumRootToLeafNumbersIPT(root);
                /*
       Approach 2: Recursive Preorder Traversal  (RPT)       
    Complexity Analysis
    •	Time complexity: O(N) since one has to visit each node.
    •	Space complexity: up to O(H) to keep the recursion stack, where H is a tree height.

                */
                sumRootToLeafNumbers = SumRootToLeafNumbersRPT(root);

                /*
      Approach 3: Morris Preorder Traversal (MPT)          
      Complexity Analysis
    •	Time complexity: O(N).
    •	Space complexity: O(1).

                */
                sumRootToLeafNumbers = SumRootToLeafNumbersMPT(root);

                return sumRootToLeafNumbers;


            }
            public int SumRootToLeafNumbersIPT(DataStructures.TreeNode root)
            {
                int rootToLeaf = 0, currNumber = 0;
                Stack<KeyValuePair<DataStructures.TreeNode, int>> stack =
                    new Stack<KeyValuePair<DataStructures.TreeNode, int>>();
                stack.Push(new KeyValuePair<DataStructures.TreeNode, int>(root, 0));
                while (stack.Count > 0)
                {
                    KeyValuePair<DataStructures.TreeNode, int> p = stack.Pop();
                    root = p.Key;
                    currNumber = p.Value;
                    if (root != null)
                    {
                        currNumber = currNumber * 10 + root.Val;
                        // if it's a leaf, update root-to-leaf sum
                        if (root.Left == null && root.Right == null)
                        {
                            rootToLeaf += currNumber;
                        }
                        else
                        {
                            stack.Push(new KeyValuePair<DataStructures.TreeNode, int>(root.Right,
                                                                       currNumber));
                            stack.Push(
                                new KeyValuePair<DataStructures.TreeNode, int>(root.Left, currNumber));
                        }
                    }
                }

                return rootToLeaf;
            }
            int rootToLeaf = 0;

            public void Preorder(DataStructures.TreeNode r, int currNumber)
            {
                if (r != null)
                {
                    currNumber = currNumber * 10 + r.Val;
                    // if it's a leaf, update root-to-leaf sum
                    if (r.Left == null && r.Right == null)
                        rootToLeaf += currNumber;
                    Preorder(r.Left, currNumber);
                    Preorder(r.Right, currNumber);
                }
            }

            public int SumRootToLeafNumbersRPT(DataStructures.TreeNode root)
            {
                Preorder(root, 0);
                return rootToLeaf;
            }

            public int SumRootToLeafNumbersMPT(DataStructures.TreeNode root)
            {
                int rootToLeaf = 0, currNumber = 0;
                int steps;
                DataStructures.TreeNode predecessor;
                while (root != null)
                {
                    if (root.Left != null)
                    {
                        predecessor = root.Left;
                        steps = 1;
                        while (predecessor.Right != null && predecessor.Right != root)
                        {
                            predecessor = predecessor.Right;
                            ++steps;
                        }

                        if (predecessor.Right == null)
                        {
                            currNumber = currNumber * 10 + root.Val;
                            predecessor.Right = root;
                            root = root.Left;
                        }
                        else
                        {
                            if (predecessor.Left == null)
                            {
                                rootToLeaf += currNumber;
                            }

                            for (int i = 0; i < steps; ++i)
                            {
                                currNumber /= 10;
                            }

                            predecessor.Right = null;
                            root = root.Right;
                        }
                    }
                    else
                    {
                        currNumber = currNumber * 10 + root.Val;
                        if (root.Right == null)
                        {
                            rootToLeaf += currNumber;
                        }

                        root = root.Right;
                    }
                }

                return rootToLeaf;
            }

        }

        /*
        100. Same Tree
https://leetcode.com/problems/same-tree/editorial/
        */

        public class SameTreeSol
        {
            /*
            Approach 1: Recursion
Complexity Analysis
•	Time complexity : O(N),
where N is a number of nodes in the tree, since one visits
each node exactly once.
•	Space complexity : O(N) in the worst case of completely unbalanced tree, to keep a recursion stack

            */
            public static bool Rec(DataStructures.TreeNode p, DataStructures.TreeNode q)
            {
                // p and q are both null
                if (p == null && q == null)
                    return true;
                // one of p and q is null
                if (q == null || p == null)
                    return false;
                if (p.Val != q.Val)
                    return false;
                return Rec(p.Right, q.Right) && Rec(p.Left, q.Left);

            }
            /*
       Approach 2: Iteration
Complexity Analysis
•	Time complexity : O(N) since each node is visited
exactly once.
•	Space complexity : O(N) in the worst case, where the tree is a perfect fully balanced binary tree, since BFS will have to store at least an entire level of the tree in the queue, and the last level has O(N) nodes.

       */

            public static bool Iterative(DataStructures.TreeNode p, DataStructures.TreeNode q)
            {
                Queue<(TreeNode, TreeNode)> deq = new Queue<(TreeNode, TreeNode)>();
                deq.Enqueue((p, q));
                while (deq.Count != 0)
                {
                    (p, q) = deq.Dequeue();
                    if (p == null && q == null)
                        continue;
                    if (q == null || p == null)
                        return false;
                    if (p.Val != q.Val)
                        return false;
                    deq.Enqueue((p.Left, q.Left));
                    deq.Enqueue((p.Right, q.Right));
                }

                return true;
            }
        }

        /*
        101. Symmetric Tree
    https://leetcode.com/problems/symmetric-tree/description/

        */
        public class SymmetricTreeSol
        {
            /*
            Approach 1: Recursive
           Complexity Analysis
•	Time complexity: O(n). Because we traverse the entire input tree once, the total run time is O(n), where n is the total number of nodes in the tree.
•	Space complexity: The number of recursive calls is bound by the height of the tree. In the worst case, the tree is linear and the height is in O(n). Therefore, space complexity due to recursive calls on the stack is O(n) in the worst case.

            */
            public static bool Rec(DataStructures.TreeNode root)
            {
                return IsMirror(root, root);

                bool IsMirror(DataStructures.TreeNode t1, DataStructures.TreeNode t2)
                {
                    if (t1 == null && t2 == null)
                        return true;
                    if (t1 == null || t2 == null)
                        return false;
                    return (t1.Val == t2.Val) && IsMirror(t1.Right, t2.Left) &&
                           IsMirror(t1.Left, t2.Right);
                }

            }

            /*            
Approach 2: Iterative
Complexity Analysis
•	Time complexity: O(n). Because we traverse the entire input tree once, the total run time is O(n), where n is the total number of nodes in the tree.
•	Space complexity: There is additional space required for the search queue. In the worst case, we have to insert O(n) nodes in the queue. Therefore, space complexity is O(n).

            */
            public static bool Iterate(DataStructures.TreeNode root)
            {
                Queue<DataStructures.TreeNode> q = new Queue<DataStructures.TreeNode>();
                q.Enqueue(root);
                q.Enqueue(root);
                while (q.Count != 0)
                {
                    DataStructures.TreeNode t1 = q.Dequeue();
                    DataStructures.TreeNode t2 = q.Dequeue();
                    if (t1 == null && t2 == null)
                        continue;
                    if (t1 == null || t2 == null)
                        return false;
                    if (t1.Val != t2.Val)
                        return false;
                    q.Enqueue(t1.Left);
                    q.Enqueue(t2.Right);
                    q.Enqueue(t1.Right);
                    q.Enqueue(t2.Left);
                }

                return true;
            }

        }


        /*
        102. Binary Tree Level Order Traversal or Top Down Level Order Traverse
        https://leetcode.com/problems/binary-tree-level-order-traversal/description/
        s
        */
        public class LevelOrderTraverseSol //TopDownLevelOrderTraversal
        {
            /*
            Approach 1: Recursion
            Complexity Analysis
•	Time complexity: O(N) since each node is processed exactly once.
•	Space complexity: O(N) to keep the output structure which contains N node values.

            */
            public static IList<IList<int>> Rec(DataStructures.TreeNode root)
            {
                IList<IList<int>> levels = new List<IList<int>>();
                if (root == null)
                    return levels;
                Helper(root, 0);
                return levels;


                void Helper(DataStructures.TreeNode node, int level)
                {
                    if (levels.Count == level)
                        levels.Add(new List<int>());
                    levels[level].Add(node.Val);
                    if (node.Left != null)
                        Helper(node.Left, level + 1);
                    if (node.Right != null)
                        Helper(node.Right, level + 1);
                }
            }

            /*
            Approach 2: Iteration
    Complexity Analysis
    •	Time complexity: O(N) since each node is processed exactly once.
    •	Space complexity: O(N) to keep the output structure which contains N node values.

            */
            public IList<IList<int>> Iterate(DataStructures.TreeNode root)
            {
                List<IList<int>> levels = new List<IList<int>>();
                if (root == null)
                    return levels;
                Queue<DataStructures.TreeNode> queue = new Queue<DataStructures.TreeNode>();
                queue.Enqueue(root);
                int level = 0;
                while (queue.Count > 0)
                {
                    // start the current level
                    levels.Add(new List<int>());
                    // number of elements in the current level
                    int level_length = queue.Count;
                    for (int i = 0; i < level_length; ++i)
                    {
                        DataStructures.TreeNode node = queue.Dequeue();
                        // fulfill the current level
                        levels[level].Add(node.Val);
                        // add child nodes of the current level
                        // in the queue for the next level
                        if (node.Left != null)
                            queue.Enqueue(node.Left);
                        if (node.Right != null)
                            queue.Enqueue(node.Right);
                    }

                    // go to the next level
                    level++;
                }

                return levels;
            }

        }


        /*
           107. Binary Tree Level Order Traversal II or Bottom Up Level Order Traverse
       https://leetcode.com/problems/binary-tree-level-order-traversal-ii/description/
           */

        public class BottomUpLevelOrderTraverseSol
        {
            /*
            Approach 1: Recursion: DFS Preorder Traversal
            Complexity Analysis
•	Time complexity: O(N) since each node is processed exactly once.
•	Space complexity: O(N) to keep the output structure which contains N node values.

            */
            public static IList<IList<int>> DFSReco(DataStructures.TreeNode root)
            {

                List<IList<int>> levels = new List<IList<int>>();
                if (root == null)
                    return levels;
                Helper(root, 0);
                levels.Reverse();
                return levels;

                void Helper(DataStructures.TreeNode node, int level)
                {
                    if (levels.Count == level)
                        levels.Add(new List<int>());
                    levels[level].Add(node.Val);
                    if (node.Left != null)
                        Helper(node.Left, level + 1);
                    if (node.Right != null)
                        Helper(node.Right, level + 1);
                }

            }
            /*
            Approach 2: Iteration: BFS Traversal
            Complexity Analysis
•	Time complexity: O(N) since each node is processed exactly once.
•	Space complexity: O(N) to keep the output structure which contains N node values.
            */
            public IList<IList<int>> Iterate(DataStructures.TreeNode root)
            {
                IList<IList<int>> levels = new List<IList<int>>();
                if (root == null)
                    return levels;
                Queue<DataStructures.TreeNode> nextLevel = new Queue<DataStructures.TreeNode>();
                nextLevel.Enqueue(root);
                while (nextLevel.Count > 0)
                {
                    Queue<DataStructures.TreeNode> currLevel = new Queue<DataStructures.TreeNode>(nextLevel);
                    nextLevel.Clear();
                    levels.Add(new List<int>());
                    foreach (DataStructures.TreeNode node in currLevel)
                    {
                        // append the current node value
                        levels[levels.Count - 1].Add(node.Val);
                        // process child nodes for the next level
                        if (node.Left != null)
                            nextLevel.Enqueue(node.Left);
                        if (node.Right != null)
                            nextLevel.Enqueue(node.Right);
                    }
                }

               ((List<IList<int>>)levels).Reverse();
                return levels;
            }

        }
        /*
        103. Binary Tree Zigzag Level Order Traversal
https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/
        */
        public class ZigzagLevelOrderTraverseSol
        {
            /*
            Approach 1: BFS (Breadth-First Search)
  Complexity Analysis
•	Time Complexity: O(N), where N is the number of nodes in the tree.
o	We visit each node once and only once.
o	In addition, the insertion operation on either end of the deque takes a constant time, rather than using the array/list data structure where the inserting at the head could take the O(K) time where K is the length of the list.
•	Space Complexity: O(N) where N is the number of nodes in the tree.
o	The main memory consumption of the algorithm is the node_queue that we use for the loop, apart from the array that we use to keep the final output.
o	As one can see, at any given moment, the node_queue would hold the nodes that are at most across two levels. Therefore, at most, the size of the queue would be no more than 2⋅L, assuming L is the maximum number of nodes that might reside on the same level. Since we have a binary tree, the level that contains the most nodes could occur to consist of all the leave nodes in a full binary tree, which is roughly L=2N. As a result, we have the space complexity of 2⋅2N=N in the worst case.

            */
            public static IList<IList<int>> BFSIterate(DataStructures.TreeNode root)
            {
                List<IList<int>> result = new List<IList<int>>();
                if (root == null)
                    return result;
                Queue<DataStructures.TreeNode> nodeQueue = new Queue<DataStructures.TreeNode>();
                nodeQueue.Enqueue(root);
                nodeQueue.Enqueue(null);
                LinkedList<int> levelList = new LinkedList<int>();
                bool isOrderLeft = true;
                while (nodeQueue.Count > 0)
                {
                    DataStructures.TreeNode currentNode = nodeQueue.Dequeue();
                    if (currentNode != null)
                    {
                        if (isOrderLeft)
                            levelList.AddLast(currentNode.Val);
                        else
                            levelList.AddFirst(currentNode.Val);
                        if (currentNode.Left != null)
                            nodeQueue.Enqueue(currentNode.Left);
                        if (currentNode.Right != null)
                            nodeQueue.Enqueue(currentNode.Right);
                    }
                    else
                    {
                        result.Add(new List<int>(levelList));
                        levelList.Clear();
                        if (nodeQueue.Count > 0)
                            nodeQueue.Enqueue(null);
                        isOrderLeft = !isOrderLeft;
                    }
                }

                return result;
            }
            /*
            Approach 2: DFS (Depth-First Search)
    Complexity Analysis
•	Time Complexity: O(N), where N is the number of nodes in the tree.
o	Same as the previous BFS approach, we visit each node once and only once.
•	Space Complexity: O(N).
o	Unlike the BFS approach, in the DFS approach, we do not need to maintain the node_queue data structure for the traversal.
o	However, the function recursion will incur additional memory consumption on the function call stack. As we can see, the size of the call stack for any invocation of DFS(node, level) will be exactly the number of level that the current node resides on. Therefore, the space complexity of our DFS algorithm is O(H), where H is the height of the tree. In the worst-case scenario, when the tree is very skewed, the tree height could be N. Thus the space complexity is also O(N).
o	Note that if the tree were guaranteed to be balanced, then the maximum height of the tree would be logN which would result in a better space complexity than the BFS approach.

            */
            public static IList<IList<int>> DFSReco(DataStructures.TreeNode root)
            {
                if (root == null)
                {
                    return new List<IList<int>>();
                }

                List<List<int>> results = new List<List<int>>();
                Action<DataStructures.TreeNode, int> DFSReco = null;
                DFSReco = (node, level) =>
                {
                    if (level >= results.Count)
                    {
                        results.Add(new List<int>() { node.Val });
                    }
                    else
                    {
                        if (level % 2 == 0)
                            results[level].Add(node.Val);
                        else
                            results[level].Insert(0, node.Val);
                    }

                    if (node.Left != null)
                        DFSReco(node.Left, level + 1);
                    if (node.Right != null)
                        DFSReco(node.Right, level + 1);
                };
                DFSReco(root, 0);
                return results.ToArray();
            }
        }


        /*
        314. Binary Tree Vertical Order Traversal
    https://leetcode.com/problems/binary-tree-vertical-order-traversal/description/
        */
        public class VerticalOrderTraverseSol
        {
            /*
            Approach 1: Breadth-First Search (BFS) with Sort
            Complexity Analysis
•	Time Complexity: O(NlogN) where N is the number of nodes in the tree.
In the first part of the algorithm, we do the BFS traversal, whose time complexity is O(N) since we traversed each node once and only once.
In the second part, in order to return the ordered results, we then sort the obtained hash table by its keys, which could result in the O(NlogN) time complexity in the worst case scenario where the binary tree is extremely imbalanced (for instance, each node has only left child node.)
As a result, the overall time complexity of the algorithm would be O(NlogN).
•	Space Complexity: O(N) where N is the number of nodes in the tree.
First of all, we use a hash table to group the nodes with the same column index. The hash table consists of keys and values. In any case, the values would consume O(N) memory. While the space for the keys could vary, in the worst case, each node has a unique column index, i.e. there would be as many keys as the values. Hence, the total space complexity for the hash table would still be O(N).
During the BFS traversal, we use a queue data structure to keep track of the next nodes to visit. At any given moment, the queue would hold no more two levels of nodes. For a binary tree, the maximum number of nodes at a level would be 2N+1 which is also the number of leafs in a full binary tree. As a result, in the worst case, our queue would consume at most O(((N+1)/2)⋅2)=O(N) space.
Lastly, we also need some space to hold the results, which is basically a reordered hash table of size O(N) as we discussed before.
To sum up, the overall space complexity of our algorithm would be O(N).

            */
            public static IList<IList<int>> BFSIterateWithSort(DataStructures.TreeNode root)
            {

                IList<IList<int>> output = new List<IList<int>>();
                if (root == null)
                {
                    return output;
                }

                Dictionary<int, List<int>> columnTable = new Dictionary<int, List<int>>();
                Queue<(TreeNode, int)> queue = new Queue<(TreeNode, int)>();
                int column = 0;
                queue.Enqueue((root, column));

                while (queue.Count > 0)
                {
                    (root, column) = queue.Dequeue();


                    if (root != null)
                    {
                        if (!columnTable.ContainsKey(column))
                        {
                            columnTable.Add(column, new List<int>());
                        }
                        columnTable[column].Add(root.Val);

                        queue.Enqueue((root.Left, column - 1));
                        queue.Enqueue((root.Right, column + 1));
                    }
                }

                List<int> sortedKeys = new List<int>(columnTable.Keys);
                sortedKeys.Sort();
                foreach (int k in sortedKeys)
                {
                    output.Add(columnTable[k]);
                }

                return output;
            }
            /*
            Approach 2: BFS without Sorting
Complexity Analysis
•	Time Complexity: O(N) where N is the number of nodes in the tree.
Following the same analysis in the previous BFS approach, the only difference is that this time we don't need the costy sorting operation (i.e. O(NlogN)).
•	Space Complexity: O(N) where N is the number of nodes in the tree. The analysis follows the same logic as in the previous BFS approach.

            */
            public List<List<int>> BFSIterate(DataStructures.TreeNode root)
            {
                List<List<int>> output = new List<List<int>>();
                if (root == null)
                {
                    return output;
                }

                Dictionary<int, List<int>> columnTable = new Dictionary<int, List<int>>();
                // Pair of node and its column offset
                Queue<KeyValuePair<DataStructures.TreeNode, int>> queue = new Queue<KeyValuePair<DataStructures.TreeNode, int>>();
                int column = 0;
                queue.Enqueue(new KeyValuePair<DataStructures.TreeNode, int>(root, column));

                int minColumn = 0, maxColumn = 0;

                while (queue.Count > 0)
                {
                    KeyValuePair<DataStructures.TreeNode, int> p = queue.Dequeue();
                    DataStructures.TreeNode currentNode = p.Key;
                    column = p.Value;

                    if (currentNode != null)
                    {
                        if (!columnTable.ContainsKey(column))
                        {
                            columnTable[column] = new List<int>();
                        }
                        columnTable[column].Add(currentNode.Val);
                        minColumn = Math.Min(minColumn, column);
                        maxColumn = Math.Max(maxColumn, column);

                        queue.Enqueue(new KeyValuePair<DataStructures.TreeNode, int>(currentNode.Left, column - 1));
                        queue.Enqueue(new KeyValuePair<DataStructures.TreeNode, int>(currentNode.Right, column + 1));
                    }
                }

                for (int i = minColumn; i <= maxColumn; ++i)
                {
                    if (columnTable.ContainsKey(i))
                    {
                        output.Add(columnTable[i]);
                    }
                }

                return output;
            }

            /*
            Approach 3: Depth-First Search (DFS)
Complexity Analysis
•	Time Complexity: O(W⋅HlogH)) where W is the width of the binary tree (i.e. the number of columns in the result) and H is the height of the tree.
In the first part of the algorithm, we traverse the tree in DFS, which results in O(N) time complexity.
Once we build the columnTable, we then have to sort it column by column.
Let us assume the time complexity of the sorting algorithm to be O(KlogK) where K is the length of the input. The maximal number of nodes in a column would be 2H where H is the height of the tree, due to the zigzag nature of the node distribution. 
As a result, the upper bound of time complexity to sort a column in a binary tree would be O(2Hlog2H).
Since we need to sort W columns, the total time complexity of the sorting operation would then be O(W⋅(2Hlog2H))=O(W⋅HlogH). Note that, the total number of nodes N in a tree is bounded by W⋅H, i.e. N<W⋅H. 
As a result, the time complexity of O(W⋅HlogH) will dominate the O(N) of the DFS traversal in the first part.
At the end of the DFS traversal, we have to iterate through the columnTable in order to retrieve the values, which will take another O(N) time.
To sum up, the overall time complexity of the algorithm would be O(W⋅HlogH).
An interesting thing to note is that in the case where the binary tree is completely imbalanced (e.g. node has only left child.), this DFS approach would have the O(N) time complexity, since the sorting takes no time on columns that contains only a single node. While the time complexity for our first BFS approach would be O(NlogN), since we have to sort the N keys in the columnTable.
•	Space Complexity: O(N) where N is the number of nodes in the tree.
We kept the columnTable which contains all the node values in the binary tree. Together with the keys, it would consume O(N) space as we discussed in previous approaches.
Since we apply the recursion for our DFS traversal, it would incur additional space consumption on the function call stack. In the worst case where the tree is completely imbalanced, we would have the size of call stack up to O(N).
Finally, we have the output which contains all the values in the binary tree, thus O(N) space.
So in total, the overall space complexity of this algorithm remains O(N).

            */
            private Dictionary<int, List<KeyValuePair<int, int>>> columnTable = new Dictionary<int, List<KeyValuePair<int, int>>>();
            private int minColumn = 0, maxColumn = 0;

            public IList<IList<int>> DFSReco(DataStructures.TreeNode root)
            {
                List<IList<int>> output = new List<IList<int>>();
                if (root == null)
                {
                    return output;
                }

                DFSReco(root, 0, 0);

                // Retrieve the results, by ordering by column and sorting by row
                for (int i = minColumn; i <= maxColumn; ++i)
                {
                    columnTable[i].Sort((p1, p2) => p1.Key.CompareTo(p2.Key));

                    List<int> sortedColumn = new List<int>();
                    foreach (var p in columnTable[i])
                    {
                        sortedColumn.Add(p.Value);
                    }
                    output.Add(sortedColumn);
                }

                return output;

                void DFSReco(DataStructures.TreeNode node, int row, int column)
                {
                    if (node == null)
                        return;

                    if (!columnTable.ContainsKey(column))
                    {
                        columnTable[column] = new List<KeyValuePair<int, int>>();
                    }

                    columnTable[column].Add(new KeyValuePair<int, int>(row, node.Val));
                    minColumn = Math.Min(minColumn, column);
                    maxColumn = Math.Max(maxColumn, column);
                    // preorder DFS traversal
                    DFSReco(node.Left, row + 1, column - 1);
                    DFSReco(node.Right, row + 1, column + 1);
                }

            }

        }


        /*
        993. Cousins in Binary Tree
https://leetcode.com/problems/cousins-in-binary-tree/description/
        */

        public class AreNodesCousinsSol
        {
            /*
            Approach 1: Depth First Search with Branch Pruning
           Complexity Analysis
•	Time Complexity: O(N), where N is the number of nodes in the binary tree. In the worst case, we might have to visit all the nodes of the binary tree.
Let's look into one such scenario. When both Node x and Node y are the leaf nodes and at the last level of the tree, the algorithm has no reasons to prune the recursion. It can only come to a conclusion once it visits both the nodes. If one of these nodes is the last node to be discovered the algorithm inevitably goes through each and every node in the tree.
•	Space Complexity: O(N). This is because the maximum amount of space utilized by the recursion stack would be N, as the height of a skewed binary tree could be, at worst, N. For a left skewed or a right skewed binary tree, where the desired nodes are lying at the maximum depth possible, the algorithm would have to maintain a recursion stack of the height of the tree.

            */

            public static bool DFSRecWithBranchPruning(DataStructures.TreeNode root, int x, int y)
            {

                bool isCousin = false;
                int recordedDepth = -1; // To save the depth of the first node.
                                        // Recurse the tree to find x and y
                DFSReco(root, 0, x, y, isCousin, recordedDepth);
                return isCousin;

                bool DFSReco(DataStructures.TreeNode node, int depth, int x, int y, bool isCousin, int recordedDepth)
                {

                    if (node == null)
                    {
                        return false;
                    }

                    // Don't go beyond the depth restricted by the first node found.
                    if (recordedDepth != -1 && depth > recordedDepth)
                    {
                        return false;
                    }

                    if (node.Val == x || node.Val == y)
                    {
                        if (recordedDepth == -1)
                        {
                            // Save depth for the first node found.
                            recordedDepth = depth;
                        }
                        // Return true, if the second node is found at the same depth.
                        return recordedDepth == depth;
                    }

                    bool left = DFSReco(node.Left, depth + 1, x, y, isCousin, recordedDepth);
                    bool right = DFSReco(node.Right, depth + 1, x, y, isCousin, recordedDepth);

                    // this.recordedDepth != depth + 1 would ensure node x and y are not
                    // immediate child nodes, otherwise they would become siblings.
                    if (left && right && recordedDepth != depth + 1)
                    {
                        isCousin = true;
                    }
                    return left || right;
                }

            }

            /*
            Approach 2: Breadth First Search with Early Stopping
Complexity Analysis
•	Time Complexity: O(N), where N is the number of nodes in the binary tree. In the worst case, we might have to visit all the nodes of the binary tree. Similar to approach 1 this approach would also have a complexity of O(N) when the Node x and Node y are present at the last level of the binary tree. The algorithm would follow the standard BFS approach and end up in checking each node before discovering the desired nodes.
•	Space Complexity: O(N). In the worst case, we need to store all the nodes of the last level in the queue. The last level of a binary tree can have a maximum of N/2 nodes. Not to forget we would also need space for N/4 null markers, one for each pair of siblings. That results in a space complexity of O((3N)/4) = O(N) (You are right Big-O notation doesn't care about constants).

            */
            public bool BFSWithEarlyStopping(DataStructures.TreeNode root, int x, int y)
            {

                // Queue for BFS
                Queue<DataStructures.TreeNode> queue = new Queue<DataStructures.TreeNode>();
                queue.Enqueue(root);

                while (queue.Count > 0)
                {
                    bool siblings = false;
                    bool cousins = false;

                    int nodesAtDepth = queue.Count;

                    for (int i = 0; i < nodesAtDepth; i++)
                    {
                        // FIFO
                        DataStructures.TreeNode node = queue.Dequeue();

                        // Encountered the marker.
                        // Siblings should be set to false as we are crossing the boundary.
                        if (node == null)
                        {
                            siblings = false;
                        }
                        else
                        {
                            if (node.Val == x || node.Val == y)
                            {
                                // Set both the siblings and cousins flag to true
                                // for a potential first sibling/cousin found.
                                if (!cousins)
                                {
                                    siblings = cousins = true;
                                }
                                else
                                {
                                    // If the siblings flag is still true this means we are still
                                    // within the siblings boundary and hence the nodes are not cousins.
                                    return !siblings;
                                }
                            }

                            if (node.Left != null) queue.Enqueue(node.Left);
                            if (node.Right != null) queue.Enqueue(node.Right);
                            // Adding the null marker for the siblings
                            queue.Enqueue(null);
                        }
                    }
                    // After the end of a level if `cousins` is set to true
                    // This means we found only one node at this level
                    if (cousins) return false;
                }
                return false;
            }


        }

        /*
        2641. Cousins in Binary Tree II	
        https://leetcode.com/problems/cousins-in-binary-tree-ii/description/
        */
        public class ReplaceValueWithCousinsSumSol
        {
            /*
            Complexity
•	Time complexity:O(N)
•	Space complexity:O(LOGN)
            */
            public static DataStructures.TreeNode DFSRec(DataStructures.TreeNode root)
            {
                GetSumOfNextLevel(root);
                GetSumOfCousin(root);
                UpdateValue(root);
                return root;

                int GetSumOfNextLevel(DataStructures.TreeNode root)
                {
                    if (root == null)
                        return 0;
                    int v = root.Val;
                    root.Val = GetSumOfNextLevel(root.Left) + GetSumOfNextLevel(root.Right);
                    return v;
                }
                void GetSumOfCousin(DataStructures.TreeNode root)
                {
                    List<DataStructures.TreeNode> ns = new() { root };

                    while (ns.Count > 0)
                    {
                        List<DataStructures.TreeNode> t = new();
                        int s = ns.Select(n => n.Val).Sum();
                        foreach (var n in ns)
                        {
                            n.Val = s - n.Val;
                            if (n.Left != null)
                                t.Add(n.Left);
                            if (n.Right != null)
                                t.Add(n.Right);
                        }
                        ns = t;
                    }

                }

                void UpdateValue(DataStructures.TreeNode root)
                {
                    if (root == null) return;
                    if (root.Left != null)
                    {
                        UpdateValue(root.Left);
                        root.Left.Val = root.Val;
                    }
                    if (root.Right != null)
                    {
                        UpdateValue(root.Right);
                        root.Right.Val = root.Val;
                    }
                }
            }




        }


        /*
        2196. Create Binary Tree From Descriptions
    https://leetcode.com/problems/create-binary-tree-from-descriptions/description/
        */
        public class CreateBinaryTreeFromDescriptionsSol
        {
            /*
            Approach 1: Convert to Graph with Breadth First Search
           Complexity Analysis
Let n be the number of entries in descriptions.
•	Time complexity: O(n)
Building the parentToChildren map and the children and parents sets takes O(n) time.
Finding the root node involves iterating through the parents set, which is O(n) in the worst case.
Constructing the binary tree using BFS also takes O(n) time since each node is processed once. Therefore, the overall time complexity is O(n).
•	Space complexity: O(n)
The parentToChildren map can store up to n entries. The children and parents sets can each store up to n elements. The BFS queue can store up to n nodes in the worst case. Therefore, the overall space complexity is O(n).

            */
            public DataStructures.TreeNode ConvertToGraphWithBFS(int[][] descriptions)
            {
                // Sets to track unique children and parents
                HashSet<int> uniqueChildren = new HashSet<int>(), uniqueParents = new HashSet<int>();
                // Map to store parent to children relationships
                Dictionary<int, List<int[]>> parentToChildrenMap = new Dictionary<int, List<int[]>>();

                // Build graph from parent to child, and add nodes to HashSets
                foreach (int[] description in descriptions)
                {
                    int parent = description[0], child = description[1], isLeft = description[2];
                    uniqueParents.Add(parent);
                    uniqueParents.Add(child);
                    uniqueChildren.Add(child);
                    if (!parentToChildrenMap.ContainsKey(parent))
                    {
                        parentToChildrenMap[parent] = new List<int[]>();
                    }
                    parentToChildrenMap[parent].Add(new int[] { child, isLeft });
                }

                // Find the root node by checking which node is in parents but not in children
                foreach (int child in uniqueChildren)
                {
                    uniqueParents.Remove(child);
                }
                DataStructures.TreeNode rootNode = new DataStructures.TreeNode(uniqueParents.GetEnumerator().Current);

                // Starting from root, use BFS to construct binary tree
                Queue<DataStructures.TreeNode> treeNodeQueue = new Queue<DataStructures.TreeNode>();
                treeNodeQueue.Enqueue(rootNode);

                while (treeNodeQueue.Count > 0)
                {
                    DataStructures.TreeNode parentNode = treeNodeQueue.Dequeue();
                    // Iterate over children of current parent
                    if (parentToChildrenMap.TryGetValue(parentNode.Val, out List<int[]> childInfoList))
                    {
                        foreach (int[] childInfo in childInfoList)
                        {
                            int childValue = childInfo[0], isLeft = childInfo[1];
                            DataStructures.TreeNode childNode = new DataStructures.TreeNode(childValue);
                            treeNodeQueue.Enqueue(childNode);
                            // Attach child node to its parent based on isLeft flag
                            if (isLeft == 1)
                            {
                                parentNode.Left = childNode;
                            }
                            else
                            {
                                parentNode.Right = childNode;
                            }
                        }
                    }
                }

                return rootNode;
            }
            /*
            Approach 2: Convert to Graph with Depth First Search
Complexity Analysis
Let n be the number of entries in descriptions.
•	Time complexity: O(n)
Building the parentToChildren map and the allNodes and children sets takes O(n) time. Finding the root node involves iterating through the allNodes set, which is O(n) in the worst case.
Constructing the binary tree using DFS also takes O(n) time since each node is processed once. Therefore, the overall time complexity is O(n).
•	Space complexity: O(n)
The parentToChildren map can store up to n entries. The allNodes and children sets can each store up to n elements. The recursive DFS stack can store up to n nodes in the worst case. Therefore, the overall space complexity is O(n).

            */
            public static DataStructures.TreeNode ConvertToGraphWithDFS(int[][] descriptions)
            {
                // Step 1: Organize data
                Dictionary<int, List<int[]>> parentToChildren = new Dictionary<int, List<int[]>>();
                HashSet<int> allNodes = new HashSet<int>();
                HashSet<int> children = new HashSet<int>();

                foreach (int[] desc in descriptions)
                {
                    int parent = desc[0];
                    int child = desc[1];
                    int isLeft = desc[2];

                    // Store child information under parent node
                    if (!parentToChildren.ContainsKey(parent))
                    {
                        parentToChildren[parent] = new List<int[]>();
                    }
                    parentToChildren[parent].Add(new int[] { child, isLeft });
                    allNodes.Add(parent);
                    allNodes.Add(child);
                    children.Add(child);
                }

                // Step 2: Find the root
                int rootVal = 0;
                foreach (int node in allNodes)
                {
                    if (!children.Contains(node))
                    {
                        rootVal = node;
                        break;
                    }
                }

                // Step 3 & 4: Build the tree using DFS
                return ConvertToGraphWithDFSRec(parentToChildren, rootVal);

                // DFS function to recursively build binary tree
                DataStructures.TreeNode ConvertToGraphWithDFSRec(Dictionary<int, List<int[]>> parentToChildren, int val)
                {
                    // Create new TreeNode for current value
                    DataStructures.TreeNode node = new DataStructures.TreeNode(val);

                    // If current node has children, recursively build them
                    if (parentToChildren.ContainsKey(val))
                    {
                        foreach (int[] childInfo in parentToChildren[val])
                        {
                            int child = childInfo[0];
                            int isLeft = childInfo[1];

                            // Attach child node based on isLeft flag
                            if (isLeft == 1)
                            {
                                node.Left = ConvertToGraphWithDFSRec(parentToChildren, child);
                            }
                            else
                            {
                                node.Right = ConvertToGraphWithDFSRec(parentToChildren, child);
                            }
                        }
                    }

                    return node;
                }
            }

            /*
          Approach 3: Constructing Tree From Directly Map and TreeNode Object
Complexity Analysis
Let n be the number of nodes created in the binary tree.
•	Time complexity: O(n)
The algorithm iterates through each description exactly once, and for each description, it performs constant-time operations:
o	Checking and adding nodes to nodeMap.
o	Updating node connections (left or right child assignments).
o	Adding child values to the children set.
The final loop iterates through the nodeMap, which contains all created nodes, to find the root node. The loop's runtime is linear in relation to the number of nodes created, resulting in a time complexity of O(n).
•	Space complexity: O(n)
The algorithm uses nodeMap to store references to all created nodes. In the worst case, this map contains all nodes, so it takes up O(n) space. The children set also takes O(n) space to store child values.
Additional space is used for the TreeNode objects themselves, but that's accounted for within the O(n) space complexity due to the nodes being stored in nodeMap

            */
            public DataStructures.TreeNode ConstructTreeFromDirectlyMapAndTreeNode(int[][] descriptions)
            {
                // Maps values to TreeNode pointers
                Dictionary<int, DataStructures.TreeNode> nodeDictionary = new Dictionary<int, DataStructures.TreeNode>();

                // Stores values which are children in the descriptions
                HashSet<int> childrenSet = new HashSet<int>();

                // Iterate through descriptions to create nodes and set up tree structure
                foreach (int[] description in descriptions)
                {
                    // Extract parent value, child value, and whether it is a
                    // left child (1) or right child (0)
                    int parentValue = description[0];
                    int childValue = description[1];
                    bool isLeft = description[2] == 1;

                    // Create parent and child nodes if not already created
                    if (!nodeDictionary.ContainsKey(parentValue))
                    {
                        nodeDictionary[parentValue] = new DataStructures.TreeNode(parentValue);
                    }
                    if (!nodeDictionary.ContainsKey(childValue))
                    {
                        nodeDictionary[childValue] = new DataStructures.TreeNode(childValue);
                    }

                    // Attach child node to parent's left or right branch
                    if (isLeft)
                    {
                        nodeDictionary[parentValue].Left = nodeDictionary[childValue];
                    }
                    else
                    {
                        nodeDictionary[parentValue].Right = nodeDictionary[childValue];
                    }

                    // Mark child as a child in the set
                    childrenSet.Add(childValue);
                }

                // Find and return the root node
                foreach (DataStructures.TreeNode node in nodeDictionary.Values)
                {
                    if (!childrenSet.Contains(node.Val))
                    {
                        return node; // Root node found
                    }
                }

                return null; // Should not occur according to problem statement
            }
        }


        /*
        1719. Number Of Ways To Reconstruct A Tree
        https://leetcode.com/problems/number-of-ways-to-reconstruct-a-tree/description/

        Time Complexity: this is the most interesting part. Imagine, that we have n nodes in our graph. Then on each step we split our data into several parts and run helper function recursively on each part, such that sum of sizes is equal to n-1 on the first step and so on. So, each element in helper(nodes) can be used no more than n times, and we can estimate it as O(n^2). Also, we need to consider for node in g[root]: g[node].remove(root) line, which for each run of helper can be estimated as O(E), where E is total number of edges in g. Finally, there is part, where we look for connected components, which can be estimated as O(E*n), because we will have no more than n levels of recursion and on each level we use each edge no more than 1 time. So, final time complexity is O(E*n). I think, this estimate can be improved to O(n^2), but I do not know at the moment, how to do it. 
        Space complexity is O(n^2) to keep all stack of recursion.

        */
        public class WaysToReconstructATreeSol
        {
            private const int CANNOT = 0;
            private const int ONE = 1;
            private const int MULTI = 2;

            public int CheckWays(int[][] pairs)
            {
                Dictionary<int, HashSet<int>> linkMap = new Dictionary<int, HashSet<int>>();
                foreach (int[] pair in pairs)
                {
                    if (!linkMap.ContainsKey(pair[0]))
                    {
                        linkMap[pair[0]] = new HashSet<int>();
                    }
                    linkMap[pair[0]].Add(pair[1]);

                    if (!linkMap.ContainsKey(pair[1]))
                    {
                        linkMap[pair[1]] = new HashSet<int>();
                    }
                    linkMap[pair[1]].Add(pair[0]);
                }

                return Helper(linkMap, new HashSet<int>(linkMap.Keys));
            }

            private int Helper(Dictionary<int, HashSet<int>> linkMap, HashSet<int> nodes)
            {
                Dictionary<int, List<int>> lenMap = new Dictionary<int, List<int>>();
                foreach (int node in nodes)
                {
                    int size = linkMap[node].Count;
                    if (!lenMap.ContainsKey(size))
                    {
                        lenMap[size] = new List<int>();
                    }
                    lenMap[size].Add(node);
                }
                if (!lenMap.ContainsKey(nodes.Count - 1))
                {
                    return CANNOT;
                }
                int root = lenMap[nodes.Count - 1][0];

                foreach (int node in linkMap[root])
                {
                    linkMap[node].Remove(root);
                }

                HashSet<int> visited = new HashSet<int>();
                Dictionary<int, HashSet<int>> comps = new Dictionary<int, HashSet<int>>();
                int comp = 0;
                foreach (int node in nodes)
                {
                    if (node != root && !visited.Contains(node))
                    {
                        Dfs(node, comp++, comps, visited, linkMap);
                    }
                }

                int ans = lenMap[nodes.Count - 1].Count >= 2 ? MULTI : ONE;
                foreach (var compEntry in comps)
                {
                    int ret = Helper(linkMap, compEntry.Value);
                    if (ret == CANNOT) return CANNOT;
                    else if (ret == MULTI) ans = MULTI;
                }
                return ans;
            }

            private void Dfs(int node, int comp, Dictionary<int, HashSet<int>> comps, HashSet<int> visited, Dictionary<int, HashSet<int>> linkMap)
            {
                if (!comps.ContainsKey(comp))
                {
                    comps[comp] = new HashSet<int>();
                }
                comps[comp].Add(node);
                visited.Add(node);
                foreach (int child in linkMap[node])
                {
                    if (!visited.Contains(child))
                    {
                        Dfs(child, comp, comps, visited, linkMap);
                    }
                }
            }
        }


        /* 834. Sum of Distances in Tree
        https://leetcode.com/problems/sum-of-distances-in-tree/description/
         */
        public class SumOfDistancesInTreeSol
        {
            private int[] answerArray, nodeCount;
            private List<HashSet<int>> adjacencyList;
            private int totalNodes;

            /*
            Approach #1: Subtree Sum and Count [Accepted]
            Complexity Analysis
            •	Time Complexity: O(N), where N is the number of nodes in the graph.
            •	Space Complexity: O(N).

            */
            public int[] SubtreeSumAndCount(int nodeCount, int[][] edges)
            {
                this.totalNodes = nodeCount;
                adjacencyList = new List<HashSet<int>>();
                answerArray = new int[nodeCount];
                this.nodeCount = new int[nodeCount];
                Array.Fill(this.nodeCount, 1);

                for (int i = 0; i < nodeCount; ++i)
                    adjacencyList.Add(new HashSet<int>());
                foreach (int[] edge in edges)
                {
                    adjacencyList[edge[0]].Add(edge[1]);
                    adjacencyList[edge[1]].Add(edge[0]);
                }
                DepthFirstSearch(0, -1);
                DepthFirstSearchSecondPass(0, -1);
                return answerArray;
            }

            private void DepthFirstSearch(int currentNode, int parentNode)
            {
                foreach (int childNode in adjacencyList[currentNode])
                {
                    if (childNode != parentNode)
                    {
                        DepthFirstSearch(childNode, currentNode);
                        nodeCount[currentNode] += nodeCount[childNode];
                        answerArray[currentNode] += answerArray[childNode] + nodeCount[childNode];
                    }
                }
            }

            private void DepthFirstSearchSecondPass(int currentNode, int parentNode)
            {
                foreach (int childNode in adjacencyList[currentNode])
                {
                    if (childNode != parentNode)
                    {
                        answerArray[childNode] = answerArray[currentNode] - nodeCount[childNode] + totalNodes - nodeCount[childNode];
                        DepthFirstSearchSecondPass(childNode, currentNode);
                    }
                }
            }
        }


        /* 3229. Minimum Operations to Make Array Equal to Target
        https://leetcode.com/problems/minimum-operations-to-make-array-equal-to-target/description/
         */
        public class MinimumOperationsToMakeArrayEqualToTargetSol
        {
            /*
            Appraoch1: keep track of how many increments & decrements are done till now
            Complexity
Time complexity: O(n)
Space complexity: O(1)

            */
            public long MinimumOperations(int[] nums, int[] target)
            {
                var n = nums.Length;
                long incr = 0, decr = 0, ops = 0;

                for (var i = 0; i < n; i++)
                {
                    var diff = target[i] - nums[i];

                    if (diff > 0)
                    {
                        if (incr < diff)
                            ops += diff - incr;
                        incr = diff;
                        decr = 0;
                    }
                    else if (diff < 0)
                    {
                        if (diff < decr)
                            ops += decr - diff;
                        decr = diff;
                        incr = 0;
                    }
                    else
                    {
                        incr = decr = 0;
                    }
                }

                return ops;
            }
        }

        /* 843. Guess the Word
        https://leetcode.com/problems/guess-the-word/description/
         */
        class GuessTheWordSol
        {
            /*
            4. Complexity
Time Complexity: O(10n) = O(n), beucase the for loop runs 10 or less times and in each iteration, we traverse the wordlist,
Space Complexity: O(10n) = O(n)


            */
            public class Master
            {
                public int Guess(string word) { return -1; } //Dummy code
            }
            public void FindSecretWord(string[] wordList, Master master)
            {
                Random randomGenerator = new Random();
                for (int attempt = 0, matchCount = 0; attempt < 10 && matchCount != 6; attempt++)
                {
                    string guess = wordList[randomGenerator.Next(wordList.Length)];
                    matchCount = master.Guess(guess);
                    List<string> candidateWords = new List<string>();
                    foreach (string word in wordList)
                    {
                        if (matchCount == GetMatches(guess, word))
                        {
                            candidateWords.Add(word);
                        }
                    }

                    wordList = candidateWords.ToArray();
                }
            }

            private int GetMatches(string firstWord, string secondWord)
            {
                int matchCount = 0;
                for (int index = 0; index < firstWord.Length; index++)
                {
                    if (firstWord[index] == secondWord[index])
                    {
                        matchCount++;
                    }
                }

                return matchCount;
            }
        }

        /* 2035. Partition Array Into Two Arrays to Minimize Sum Difference
        https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/description/
         */
        public class MinimumDifferenceSol
        {
            /*
            Approach 1: Meet In Middle
            TC
O(2^n * log(2^n))

            */
            public int MeetInMiddle(int[] nums)
            {
                // Total sum.
                int totalSum = 0;
                foreach (int number in nums) totalSum += number;

                // You can enumerate all sets using binary mask.
                int halfLength = nums.Length / 2;
                // 2^n * n
                int[][] leftSums = SubsetSums(nums, halfLength, 0);
                int[][] rightSums = SubsetSums(nums, halfLength, halfLength);

                // Enumerate each left sum, find the best right sum
                // which mins abs diff.
                // n * 2^n, n ~= 15.
                int minimumDifference = int.MaxValue;
                for (int leftLength = 0; leftLength <= halfLength; leftLength++)
                {
                    int rightLength = halfLength - leftLength;
                    int[] rightArray = rightSums[rightLength];
                    foreach (int leftSum in leftSums[leftLength])
                    {
                        // search for closest r in terms of abs(r - target).
                        int target = (totalSum - 2 * leftSum) / 2;
                        int closestRightValue = BinarySearch(rightArray, target);

                        minimumDifference = Math.Min(minimumDifference,
                                  Math.Abs(2 * (leftSum + closestRightValue) - totalSum));
                    }
                }

                return minimumDifference;
            }

            public int BinarySearch(int[] array, int target)
            {
                // searching for val closest to target.
                // abs(*)
                int leftIndex = 0;
                int rightIndex = array.Length - 1;
                while (leftIndex + 1 < rightIndex)
                {
                    int middleIndex = leftIndex + (rightIndex - leftIndex) / 2;
                    if (array[middleIndex] <= target)
                    {
                        // -> last target
                        leftIndex = middleIndex;
                    }
                    else
                    {
                        rightIndex = middleIndex;
                    }
                }

                // leftIndex + 1 == rightIndex.
                // return the closest one to target.
                return Math.Abs(array[leftIndex] - target) > Math.Abs(array[rightIndex] - target) ?
                    array[rightIndex] : array[leftIndex];
            }

            public int[][] SubsetSums(int[] nums, int size, int offset)
            {
                // picking subsets from n size sub array
                // picking from: offset -> offset + n (excl).

                // The bit mask stands for a subset, 1 means selected.
                // We can pick masks upto this value.
                // Any binary mask below this value stands for a valid subset.
                int maxState = (1 << size) - 1;
                // Len can be 0 -> n (incl)
                // init res with prefixed size, determined by binomial (i, n);
                int[][] result = new int[size + 1][];
                int binomialCoefficient = 1;
                result[0] = new int[binomialCoefficient];
                for (int i = 1; i < size + 1; i++)
                {
                    binomialCoefficient = binomialCoefficient * (size - i + 1) / i;
                    result[i] = new int[binomialCoefficient];
                }

                int[] lengthCounter = new int[size + 1];
                for (int state = 0; state <= maxState; state++)
                {
                    // cur subset is s state.
                    int length = CountBits(state);
                    int sum = 0;
                    for (int i = 0; i < size; i++)
                    {
                        // offset + i -> ele in nums.
                        if ((state & (1 << i)) > 0) sum += nums[offset + i];
                    }

                    result[length][lengthCounter[length]] = sum;
                    lengthCounter[length] += 1;
                }

                foreach (int[] array in result) Array.Sort(array);
                return result;
            }

            private int CountBits(int number)
            {
                int count = 0;
                while (number > 0)
                {
                    count += (number & 1);
                    number >>= 1;
                }
                return count;
            }
        }


        /* 987. Vertical Order Traversal of a Binary Tree
        https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/description/
         */

        class VerticalOrderTraverseOfBinaryTreeSol
        {
            List<Triplet<int, int, int>> nodeList = new List<Triplet<int, int, int>>();

            /*
            Approach 1: BFS with Global Sorting
            Complexity Analysis
            Let N be the number of nodes in the input tree.
            •	Time Complexity: O(NlogN), which applies to both the BFS and DFS approaches.
            o	In the first step of the algorithm, we traverse the input tree with either BFS or DFS, which would take O(N) time.
            o	Secondly, we sort the obtained list of coordinates which contains N elements. The sorting operation would take O(NlogN) time.
            o	Finally, we extract the results from the sorted list, which would take another O(N) time.
            o	To summarize, the overall time complexity of the algorithm would be O(NlogN), which is dominated by the sorting operation in the second step.
            •	Space Complexity: O(N). Again this applies to both the BFS and DFS approaches.
            o	In the first step of the algorithm, we build a list that contains the coordinates of all the nodes. Hence, we need O(N) space for this list.
            o	Additionally, for the BFS approach, we used a queue data structure to maintain the order of visits. At any given moment, the queue contains no more than two levels of nodes in the tree. The maximal number of nodes at one level is 2N, which is the number of the leaf nodes in a balanced binary tree. As a result, the space needed for the queue would be O(2N⋅2)=O(N).
            o	Although we don't need the queue data structure for the DFS approach, the recursion in the DFS approach incurs some additional memory consumption on the function call stack. In the worst case, the input tree might be completely imbalanced, e.g. each node has only the left child node. In this case, the recursion would occur up to N times, which in turn would consume O(N) space in the function call stack.
            o	To summarize, the space complexity for the BFS approach would be O(N)+O(N)=O(N). And the same applies to the DFS approach.

            */
            public List<List<int>> BFSWithGlobalSorting(DataStructures.TreeNode root)
            {
                List<List<int>> output = new List<List<int>>();
                if (root == null)
                {
                    return output;
                }

                // step 1). BFS traversal
                BFS(root);

                // step 2). sort the global list by <column, row, value>
                nodeList.Sort((t1, t2) =>
                {
                    if (t1.First.Equals(t2.First))
                        if (t1.Second.Equals(t2.Second))
                            return t1.Third.CompareTo(t2.Third);
                        else
                            return t1.Second.CompareTo(t2.Second);
                    else
                        return t1.First.CompareTo(t2.First);
                });

                // step 3). extract the values, partitioned by the column index.
                List<int> currColumn = new List<int>();
                int currColumnIndex = this.nodeList[0].First;

                foreach (Triplet<int, int, int> triplet in this.nodeList)
                {
                    int column = triplet.First, value = triplet.Third;
                    if (column == currColumnIndex)
                    {
                        currColumn.Add(value);
                    }
                    else
                    {
                        output.Add(currColumn);
                        currColumnIndex = column;
                        currColumn = new List<int>();
                        currColumn.Add(value);
                    }
                }
                output.Add(currColumn);

                return output;
            }
            private void BFS(DataStructures.TreeNode root)
            {
                Queue<Triplet<DataStructures.TreeNode, int, int>> queue = new Queue<Triplet<DataStructures.TreeNode, int, int>>();
                int row = 0, column = 0;
                queue.Enqueue(new Triplet<DataStructures.TreeNode, int, int>(root, row, column));

                while (queue.Count > 0)
                {
                    Triplet<DataStructures.TreeNode, int, int> triplet = queue.Dequeue();
                    root = triplet.First;
                    row = triplet.Second;
                    column = triplet.Third;

                    if (root != null)
                    {
                        this.nodeList.Add(new Triplet<int, int, int>(column, row, root.Val));
                        queue.Enqueue(new Triplet<DataStructures.TreeNode, int, int>(root.Left, row + 1, column - 1));
                        queue.Enqueue(new Triplet<DataStructures.TreeNode, int, int>(root.Right, row + 1, column + 1));
                    }
                }
            }


            class Triplet<F, S, T>
            {
                public readonly F First;
                public readonly S Second;
                public readonly T Third;

                public Triplet(F first, S second, T third)
                {
                    First = first;
                    Second = second;
                    Third = third;
                }
            }
            /*
Approach 2s: DFS with Global Sorting
Complexity Analysis
Let N be the number of nodes in the input tree.
•	Time Complexity: O(NlogN), which applies to both the BFS and DFS approaches.
o	In the first step of the algorithm, we traverse the input tree with either BFS or DFS, which would take O(N) time.
o	Secondly, we sort the obtained list of coordinates which contains N elements. The sorting operation would take O(NlogN) time.
o	Finally, we extract the results from the sorted list, which would take another O(N) time.
o	To summarize, the overall time complexity of the algorithm would be O(NlogN), which is dominated by the sorting operation in the second step.
•	Space Complexity: O(N). Again this applies to both the BFS and DFS approaches.
o	In the first step of the algorithm, we build a list that contains the coordinates of all the nodes. Hence, we need O(N) space for this list.
o	Additionally, for the BFS approach, we used a queue data structure to maintain the order of visits. At any given moment, the queue contains no more than two levels of nodes in the tree. The maximal number of nodes at one level is 2N, which is the number of the leaf nodes in a balanced binary tree. As a result, the space needed for the queue would be O(2N⋅2)=O(N).
o	Although we don't need the queue data structure for the DFS approach, the recursion in the DFS approach incurs some additional memory consumption on the function call stack. In the worst case, the input tree might be completely imbalanced, e.g. each node has only the left child node. In this case, the recursion would occur up to N times, which in turn would consume O(N) space in the function call stack.
o	To summarize, the space complexity for the BFS approach would be O(N)+O(N)=O(N). And the same applies to the DFS approach.

*/
            private void DFS(DataStructures.TreeNode node, int row, int column)
            {
                if (node == null)
                    return;

                nodeList.Add(new Triplet<int, int, int>(column, row, node.Val));
                // preorder DFS traversal
                DFS(node.Left, row + 1, column - 1);
                DFS(node.Right, row + 1, column + 1);
            }

            public List<List<int>> DFSWithGlobalSorting(DataStructures.TreeNode root)
            {
                List<List<int>> output = new List<List<int>>();
                if (root == null)
                {
                    return output;
                }

                // step 1). DFS traversal
                DFS(root, 0, 0);

                // step 2). sort the list by <column, row, value>
                nodeList.Sort((t1, t2) =>
                {
                    if (t1.First.Equals(t2.First))
                    {
                        if (t1.Second.Equals(t2.Second))
                            return t1.Third.CompareTo(t2.Third);
                        else
                            return t1.Second.CompareTo(t2.Second);
                    }
                    else
                        return t1.First.CompareTo(t2.First);
                });

                // step 3). extract the values, grouped by the column index.
                List<int> currentColumn = new List<int>();
                int currentColumnIndex = nodeList[0].First;

                foreach (var triplet in nodeList)
                {
                    int column = triplet.First, value = triplet.Third;
                    if (column == currentColumnIndex)
                    {
                        currentColumn.Add(value);
                    }
                    else
                    {
                        output.Add(currentColumn);
                        currentColumnIndex = column;
                        currentColumn = new List<int> { value };
                    }
                }
                output.Add(currentColumn);

                return output;
            }

            /* Approach 3: BFS with Partition Sorting 
            Complexity Analysis
Let N be the number of nodes in the tree.
•	Time Complexity: O(NlogkN) where k is the width of the tree, i.e. k is also the number of columns in the result.
o	In the first step, it takes O(N) time complexity for both the BFS and DFS traversal.
o	In the second step, we need to sort the hashmap entry by entry. As we shown in the intuition section, the time complexity of sorting k equal-sized subgroups of with total N elements would be O(k⋅kNlogkN)=O(NlogkN). If we assume that the nodes are evenly aligned in the columns, then this would be the time complexity of sorting the obtained hashmap.
o	Finally, it takes another O(N) time complexity to extract the results from the hashmap.
o	As a result, the overall time complexity is O(NlogkN).
o	Although the sorting operation in the second step still dominates, it is more optimized compared to the previous approach of sorting the entire coordinates.
Let us look at one particular example. In the case where the tree is complete imbalanced (e.g. a node has only left node), the tree would be partitioned into exactly N groups. Each group contains a single element. It would take no time to sort each group. As a result, the overall time complexity of this approach becomes N⋅O(1)=O(N).
While for the previous approach, its overall time complexity remains O(NlogN).
•	Space Complexity: O(N). Again this applies to both the BFS and DFS approaches. The analysis is the same as the previous approach.

            */
            // key: column; value: <row, node_value>
            Dictionary<int, List<KeyValuePair<int, int>>> columnTable = new Dictionary<int, List<KeyValuePair<int, int>>>();
            int minColumn = 0, maxColumn = 0;


            public List<List<int>> BFSWithPartitionSorting(DataStructures.TreeNode root)
            {
                List<List<int>> output = new List<List<int>>();
                if (root == null)
                {
                    return output;
                }

                // step 1). BFS traversal
                BFS(root);

                // step 2). retrieve the value from the columnTable
                for (int i = minColumn; i <= maxColumn; ++i)
                {
                    // order by both "row" and "value"
                    columnTable[i].Sort((p1, p2) =>
                    {
                        if (p1.Key == p2.Key)
                            return p1.Value.CompareTo(p2.Value);
                        else
                            return p1.Key.CompareTo(p2.Key);
                    });

                    List<int> sortedColumn = new List<int>();
                    foreach (var p in columnTable[i])
                    {
                        sortedColumn.Add(p.Value);
                    }
                    output.Add(sortedColumn);
                }

                return output;
                void BFS(DataStructures.TreeNode root)
                {
                    // tuples of <column, <row, value>>
                    Queue<KeyValuePair<DataStructures.TreeNode, KeyValuePair<int, int>>> queue = new Queue<KeyValuePair<DataStructures.TreeNode, KeyValuePair<int, int>>>();
                    int row = 0, column = 0;
                    queue.Enqueue(new KeyValuePair<DataStructures.TreeNode, KeyValuePair<int, int>>(root, new KeyValuePair<int, int>(row, column)));

                    while (queue.Count > 0)
                    {
                        KeyValuePair<DataStructures.TreeNode, KeyValuePair<int, int>> p = queue.Dequeue();
                        root = p.Key;
                        row = p.Value.Key;
                        column = p.Value.Value;

                        if (root != null)
                        {
                            if (!columnTable.ContainsKey(column))
                            {
                                columnTable[column] = new List<KeyValuePair<int, int>>();
                            }
                            columnTable[column].Add(new KeyValuePair<int, int>(row, root.Val));
                            minColumn = Math.Min(minColumn, column);
                            maxColumn = Math.Max(maxColumn, column);

                            queue.Enqueue(new KeyValuePair<DataStructures.TreeNode, KeyValuePair<int, int>>(root.Left, new KeyValuePair<int, int>(row + 1, column - 1)));
                            queue.Enqueue(new KeyValuePair<DataStructures.TreeNode, KeyValuePair<int, int>>(root.Right, new KeyValuePair<int, int>(row + 1, column + 1)));
                        }
                    }
                }
            }

            /* Approach 4: DFS with Partition Sorting 
          Complexity Analysis
    Let N be the number of nodes in the tree.
    •	Time Complexity: O(NlogkN) where k is the width of the tree, i.e. k is also the number of columns in the result.
    o	In the first step, it takes O(N) time complexity for both the BFS and DFS traversal.
    o	In the second step, we need to sort the hashmap entry by entry. As we shown in the intuition section, the time complexity of sorting k equal-sized subgroups of with total N elements would be O(k⋅kNlogkN)=O(NlogkN). If we assume that the nodes are evenly aligned in the columns, then this would be the time complexity of sorting the obtained hashmap.
    o	Finally, it takes another O(N) time complexity to extract the results from the hashmap.
    o	As a result, the overall time complexity is O(NlogkN).
    o	Although the sorting operation in the second step still dominates, it is more optimized compared to the previous approach of sorting the entire coordinates.
    Let us look at one particular example. In the case where the tree is complete imbalanced (e.g. a node has only left node), the tree would be partitioned into exactly N groups. Each group contains a single element. It would take no time to sort each group. As a result, the overall time complexity of this approach becomes N⋅O(1)=O(N).
    While for the previous approach, its overall time complexity remains O(NlogN).
    •	Space Complexity: O(N). Again this applies to both the BFS and DFS approaches. The analysis is the same as the previous approach.

          */
            public IList<IList<int>> DFSWithPartitionSorting(DataStructures.TreeNode root)
            {
                IList<IList<int>> output = new List<IList<int>>();
                if (root == null)
                {
                    return output;
                }

                // step 1). DFS traversal
                DFS(root, 0, 0);

                // step 2). retrieve the value from the columnTable
                for (int i = minColumn; i <= maxColumn; ++i)
                {
                    // order by both "row" and "value"
                    columnTable[i].Sort((p1, p2) =>
                    {
                        if (p1.Key == p2.Key)
                            return p1.Value.CompareTo(p2.Value);
                        else
                            return p1.Key.CompareTo(p2.Key);
                    });

                    List<int> sortedColumn = new List<int>();
                    foreach (var p in columnTable[i])
                    {
                        sortedColumn.Add(p.Value);
                    }
                    output.Add(sortedColumn);
                }

                return output;
                void DFS(DataStructures.TreeNode node, int row, int column)
                {
                    if (node == null)
                        return;

                    if (!columnTable.ContainsKey(column))
                    {
                        columnTable[column] = new List<KeyValuePair<int, int>>();
                    }

                    columnTable[column].Add(new KeyValuePair<int, int>(row, node.Val));
                    minColumn = Math.Min(minColumn, column);
                    maxColumn = Math.Max(maxColumn, column);
                    // preorder DFS traversal
                    DFS(node.Left, row + 1, column - 1);
                    DFS(node.Right, row + 1, column + 1);
                }

            }



        }


        /* 3068. Find the Maximum Sum of Node Values
        https://leetcode.com/problems/find-the-maximum-sum-of-node-values/description/
         */
        class FindMaxSumOfNodeValuesSol
        {
            /*             Approach 1: Top-Down Dynamic Programming - Memoization
            Complexity Analysis
            Let n be the number of nodes in the tree.
            •	Time complexity: O(n)
            The time complexity of the maxSumOfNodes function can be analyzed by considering the number of unique subproblems that need to be solved. There are at most n⋅2 unique subproblems, indexed by index and isEven values, because the number of possible values for index is n and isEven is 2 (parity).
            Here, each subproblem is computed only once (due to memoization). So, the time complexity is bounded by the number of unique subproblems.
            Therefore, the time complexity can be stated as O(n).
            •	Space complexity: O(n)
            The space complexity of the algorithm is primarily determined by two factors: the auxiliary space used for memoization and the recursion stack space. The memoization table, denoted as memo, consumes O(n) space due to its size being proportional to the length of the input node list.
            Additionally, the recursion stack space can grow up to O(n) in the worst case, constrained by the length of the input node list, as each recursive call may add a frame to the stack.
            Therefore, the overall space complexity is the sum of these two components, resulting in O(n)+O(n), which simplifies to O(n).

             */
            public long TopDownDPRecWithMemo(int[] nums, int k, int[][] edges)
            {
                long[][] memo = new long[nums.Length][];
                for (int i = 0; i < memo.Length; i++)
                {
                    memo[i] = new long[2];
                    Array.Fill(memo[i], -1);
                }
                return MaxSumOfNodes(0, 1, nums, k, memo);
            }

            private long MaxSumOfNodes(int index, int isEven, int[] nums, int k, long[][] memo)
            {
                if (index == nums.Length)
                {
                    // If the operation is performed on an odd number of elements return
                    // INT_MIN
                    return isEven == 1 ? 0 : int.MinValue;
                }
                if (memo[index][isEven] != -1)
                {
                    return memo[index][isEven];
                }
                // No operation performed on the element
                long noXorDone = nums[index] + MaxSumOfNodes(index + 1, isEven, nums, k, memo);
                // XOR operation is performed on the element
                long xorDone = (nums[index] ^ k) + MaxSumOfNodes(index + 1, isEven ^ 1, nums, k, memo);

                // Memoize and return the result
                return memo[index][isEven] = Math.Max(xorDone, noXorDone);
            }
            /*             Approach 2: Bottom-up Dynamic Programming (Tabulation) 
            Complexity Analysis
            Let n be the number of elements in the node value list.
            •	Time complexity: O(n)
            We iterate through a nested loop where the total number of iterations is given by n⋅2. Inside the nested loops, we perform constant time operations. Therefore, time complexity is given by O(n).
            •	Space complexity: O(n)
            Since we create a new dp matrix of size n⋅2, the total additional space becomes n⋅2. So, the net space complexity is O(n).

            */
            public long BottomUpDPTabulation(int[] nums, int k, int[][] edges)
            {
                int n = nums.Length;
                long[][] dp = new long[n + 1][];
                dp[n][1] = 0;
                dp[n][0] = int.MinValue;

                for (int index = n - 1; index >= 0; index--)
                {
                    for (int isEven = 0; isEven <= 1; isEven++)
                    {
                        // Case 1: we perform the operation on this element.
                        long performOperation = dp[index + 1][isEven ^ 1] + (nums[index] ^ k);
                        // Case 2: we don't perform operation on this element.
                        long dontPerformOperation = dp[index + 1][isEven] + nums[index];

                        dp[index][isEven] = Math.Max(performOperation, dontPerformOperation);
                    }
                }

                return dp[0][1];
            }

            /* Approach 3: Greedy (Sorting based approach)
            Complexity Analysis
            Let n be the number of elements in the node value list.
            •	Time complexity: O(n⋅logn)
            Other than the sort invocation, we perform simple linear operations on the list, so the runtime is dominated by the O(n⋅logn) complexity of sorting.
            •	Space complexity: O(n)
            Since we create a new netChange array of size n and sort it, the additional space becomes O(n) for netChange array and O(logn) or O(n) for sorting it (depending on the sorting algorithm used). So, the net space complexity is O(n).

             */
            public long WithGreedySorting(int[] nums, int k, int[][] edges)
            {
                int n = nums.Length;
                int[] netChange = new int[n];
                long nodeSum = 0;

                for (int i = 0; i < n; i++)
                {
                    netChange[i] = (nums[i] ^ k) - nums[i];
                    nodeSum += nums[i];
                }

                Array.Sort(netChange);
                // Reverse the sorted array
                for (int i = 0; i < n / 2; i++)
                {
                    int temp = netChange[i];
                    netChange[i] = netChange[n - 1 - i];
                    netChange[n - 1 - i] = temp;
                }

                for (int i = 0; i < n; i += 2)
                {
                    // If netChange contains odd number of elements break the loop
                    if (i + 1 == n)
                    {
                        break;
                    }
                    long pairSum = netChange[i] + netChange[i + 1];
                    // Include in nodeSum if pairSum is positive
                    if (pairSum > 0)
                    {
                        nodeSum += pairSum;
                    }
                }
                return nodeSum;
            }


            /* Approach 4: Greedy (Finding local maxima and minima)
            Complexity Analysis
            Let n be the number of elements in the node value list.
            •	Time complexity: O(n)
            We perform a single pass linear scan on the list which takes O(n) time. All other operations are performed in constant time. This makes the net time complexity as O(n).
            •	Space complexity: O(1)
            We do not allocate any additional auxiliary memory proportional to the size of the given node list. Therefore, overall space complexity is given by O(1).

             */
            public long WithGreedyLocalMaixmAndMinima(int[] nums, int k, int[][] edges)
            {
                long sum = 0;
                int count = 0, positiveMinimum = (1 << 30), negativeMaximum = -1 * (1 << 30);

                foreach (int nodeValue in nums)
                {
                    int operatedNodeValue = nodeValue ^ k;
                    sum += nodeValue;
                    int netChange = operatedNodeValue - nodeValue;
                    if (netChange > 0)
                    {
                        positiveMinimum = Math.Min(positiveMinimum, netChange);
                        sum += netChange;
                        count++;
                    }
                    else
                    {
                        negativeMaximum = Math.Max(negativeMaximum, netChange);
                    }
                }

                // If the number of positive netChange values is even, return the sum.
                if (count % 2 == 0)
                {
                    return sum;
                }

                // Otherwise return the maximum of both discussed cases.
                return Math.Max(sum - positiveMinimum, sum + negativeMaximum);
            }

        }

        /* 236. Lowest Common Ancestor of a Binary Tree
        https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class LowestCommonAncestorSol
        {
            private DataStructures.TreeNode lowestCommonAncestorNode;

            public LowestCommonAncestorSol()
            {
                // Variable to store LCA node.
                this.lowestCommonAncestorNode = null;
            }

            private bool RecurseTree(DataStructures.TreeNode currentNode, DataStructures.TreeNode nodeP, DataStructures.TreeNode nodeQ)
            {
                // If reached the end of a branch, return false.
                if (currentNode == null)
                {
                    return false;
                }

                // Left Recursion. If left recursion returns true, set left = 1 else 0
                int leftFlag = RecurseTree(currentNode.Left, nodeP, nodeQ) ? 1 : 0;

                // Right Recursion
                int rightFlag = RecurseTree(currentNode.Right, nodeP, nodeQ) ? 1 : 0;

                // If the current node is one of nodeP or nodeQ
                int midFlag = (currentNode == nodeP || currentNode == nodeQ) ? 1 : 0;

                // If any two of the flags left, right or mid become True
                if (midFlag + leftFlag + rightFlag >= 2)
                {
                    this.lowestCommonAncestorNode = currentNode;
                }

                // Return true if any one of the three bool values is True.
                return (midFlag + leftFlag + rightFlag > 0);
            }
            /* Approach 1: Recursive Approach
            Complexity Analysis
            •	Time Complexity: O(N), where N is the number of nodes in the binary tree. In the worst case we might be visiting all the nodes of the binary tree.
            •	Space Complexity: O(N). This is because the maximum amount of space utilized by the recursion stack would be N since the height of a skewed binary tree could be N.

             */
            public DataStructures.TreeNode Recur(DataStructures.TreeNode root, DataStructures.TreeNode nodeP, DataStructures.TreeNode nodeQ)
            {
                // Traverse the tree
                RecurseTree(root, nodeP, nodeQ);
                return this.lowestCommonAncestorNode;
            }
            /* Approach 2: Iterative using parent pointers 
            Complexity Analysis
•	Time Complexity : O(N), where N is the number of nodes in the binary tree. In the worst case we might be visiting all the nodes of the binary tree.
•	Space Complexity : O(N). In the worst case space utilized by the stack, the parent pointer dictionary and the ancestor set, would be N each, since the height of a skewed binary tree could be N.

            */
            public DataStructures.TreeNode IterativeUsingParentPointers(DataStructures.TreeNode root, DataStructures.TreeNode nodeP, DataStructures.TreeNode nodeQ)
            {

                // Stack for tree traversal
                Stack<DataStructures.TreeNode> traversalStack = new Stack<DataStructures.TreeNode>();

                // Dictionary for parent pointers
                Dictionary<DataStructures.TreeNode, DataStructures.TreeNode> parentPointers = new Dictionary<DataStructures.TreeNode, DataStructures.TreeNode>();

                parentPointers[root] = null;
                traversalStack.Push(root);

                // Iterate until we find both the nodes nodeP and nodeQ
                while (!parentPointers.ContainsKey(nodeP) || !parentPointers.ContainsKey(nodeQ))
                {

                    DataStructures.TreeNode currentNode = traversalStack.Pop();

                    // While traversing the tree, keep saving the parent pointers.
                    if (currentNode.Left != null)
                    {
                        parentPointers[currentNode.Left] = currentNode;
                        traversalStack.Push(currentNode.Left);
                    }
                    if (currentNode.Right != null)
                    {
                        parentPointers[currentNode.Right] = currentNode;
                        traversalStack.Push(currentNode.Right);
                    }
                }

                // Ancestors set for nodeP.
                HashSet<DataStructures.TreeNode> ancestorSet = new HashSet<DataStructures.TreeNode>();

                // Process all ancestors for nodeP using parent pointers.
                while (nodeP != null)
                {
                    ancestorSet.Add(nodeP);
                    nodeP = parentPointers[nodeP];
                }

                // The first ancestor of nodeQ which appears in
                // nodeP's ancestor set is their lowest common ancestor.
                while (!ancestorSet.Contains(nodeQ))
                    nodeQ = parentPointers[nodeQ];
                return nodeQ;
            }
            /* Approach 3: Iterative without parent pointers 
            Complexity Analysis
•	Time Complexity : O(N), where N is the number of nodes in the binary tree. In the worst case we might be visiting all the nodes of the binary tree. The advantage of this approach is that we can prune backtracking. We simply return once both the nodes are found.
•	Space Complexity : O(N). In the worst case the space utilized by stack would be N since the height of a skewed binary tree could be N.

            */
            // Three static flags to keep track of post-order traversal.

            // Both left and right traversal pending for a node.
            // Indicates the nodes children are yet to be traversed.
            private static int BOTH_PENDING = 2;

            // Left traversal done.
            private static int LEFT_DONE = 1;

            // Both left and right traversal done for a node.
            // Indicates the node can be popped off the stack.
            private static int BOTH_DONE = 0;

            public DataStructures.TreeNode IterativeWithoutParentPointers(DataStructures.TreeNode root, DataStructures.TreeNode p, DataStructures.TreeNode q)
            {

                Stack<KeyValuePair<DataStructures.TreeNode, int>> stack = new Stack<KeyValuePair<DataStructures.TreeNode, int>>();

                // Initialize the stack with the root node.
                stack.Push(new KeyValuePair<DataStructures.TreeNode, int>(root, BOTH_PENDING));

                // This flag is set when either one of p or q is found.
                bool oneNodeFound = false;

                // This is used to keep track of the LCA.
                DataStructures.TreeNode lowestCommonAncestor = null;

                // Child node
                DataStructures.TreeNode childNode = null;

                // We do a post order traversal of the binary tree using stack
                while (stack.Count > 0)
                {

                    KeyValuePair<DataStructures.TreeNode, int> top = stack.Peek();
                    DataStructures.TreeNode parentNode = top.Key;
                    int parentState = top.Value;

                    // If the parentState is not equal to BOTH_DONE,
                    // this means the parentNode can't be popped off yet.
                    if (parentState != BOTH_DONE)
                    {

                        // If both child traversals are pending
                        if (parentState == BOTH_PENDING)
                        {

                            // Check if the current parentNode is either p or q.
                            if (parentNode == p || parentNode == q)
                            {

                                // If oneNodeFound was set already, this means we have found
                                // both the nodes.
                                if (oneNodeFound)
                                {
                                    return lowestCommonAncestor;
                                }
                                else
                                {
                                    // Otherwise, set oneNodeFound to true,
                                    // to mark one of p and q is found.
                                    oneNodeFound = true;

                                    // Save the current top element of stack as the LCA.
                                    lowestCommonAncestor = stack.Peek().Key;
                                }
                            }

                            // If both pending, traverse the left child first
                            childNode = parentNode.Left;
                        }
                        else
                        {
                            // traverse right child
                            childNode = parentNode.Right;
                        }

                        // Update the node state at the top of the stack
                        // Since we have visited one more child.
                        stack.Pop();
                        stack.Push(new KeyValuePair<DataStructures.TreeNode, int>(parentNode, parentState - 1));

                        // Add the child node to the stack for traversal.
                        if (childNode != null)
                        {
                            stack.Push(new KeyValuePair<DataStructures.TreeNode, int>(childNode, BOTH_PENDING));
                        }
                    }
                    else
                    {

                        // If the parentState of the node is both done,
                        // the top node could be popped off the stack.
                        // Update the LCA node to be the next top node.
                        if (lowestCommonAncestor == stack.Pop().Key && oneNodeFound)
                        {
                            lowestCommonAncestor = stack.Peek().Key;
                        }

                    }
                }

                return null;
            }

        }


        /* 1650. Lowest Common Ancestor of a Binary Tree III
        https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
        https://algo.monster/liteproblems/1650
         */

        class LowestCommonAncestorOfBinaryTreeIIISol
        {
            /*Time and Space Complexity
The given Python function finds the lowest common ancestor (LCA) of two nodes in a binary tree where nodes have a pointer to their parent.
Time Complexity:
The time complexity of the code is O(h) where h is the height of the tree. This is because, in the worst case, both nodes p and q could be at the bottom of the tree, and we would traverse from each node up to the root before finding the LCA. Since we are moving at most up to the height of the tree for both p and q, the time complexity remains O(h).
Space Complexity:
The space complexity of the code is O(1). This is due to the fact that we are only using a fixed number of pointers (a and b) regardless of the size of the input tree, and no additional data structures or recursive stack space are used.
  */
            // This method finds the lowest common ancestor (LCA) of two nodes in a binary tree where nodes have parent pointers.
            public Node LowestCommonAncestor(Node firstNode, Node secondNode)
            {
                // Initialize two pointers for traversing the ancestors of the given nodes.
                Node pointerA = firstNode;
                Node pointerB = secondNode;

                // Traverse the ancestor chain of both nodes until they meet.
                while (pointerA != pointerB)
                {
                    // If pointerA has reached the root (parent is null), start it at secondNode,
                    // otherwise, move it to its parent.
                    pointerA = pointerA.parent == null ? secondNode : pointerA.parent;

                    // If pointerB has reached the root (parent is null), start it at firstNode,
                    // otherwise, move it to its parent.
                    pointerB = pointerB.parent == null ? firstNode : pointerB.parent;
                }

                // When pointerA and pointerB meet, we have found the LCA.
                return pointerA;
            }
            public class Node
            {
                public int val;
                public Node left;
                public Node right;
                public Node parent;
            }
        }


        /* 199. Binary Tree Right Side View
        https://leetcode.com/problems/binary-tree-right-side-view/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class RightSideViewOfBinaryTreeSol
        {

            /* Approach 1: BFS: Two Queues
            Complexity Analysis
            •	Time complexity: O(N) since one has to visit each node.
            •	Space complexity: O(D) to keep the queues, where D is a tree diameter. Let's use the last level to estimate the queue size. This level could contain up to N/2 tree nodes in the case of complete binary tree.

             */
            public IList<int> BFSWithTwoQueues(DataStructures.TreeNode root)
            {
                if (root == null) return new List<int>();

                Queue<DataStructures.TreeNode> nextLevel = new Queue<DataStructures.TreeNode>();
                nextLevel.Enqueue(root);
                Queue<DataStructures.TreeNode> currLevel = new Queue<DataStructures.TreeNode>();
                List<int> rightSide = new List<int>();

                DataStructures.TreeNode node = null;
                while (nextLevel.Count > 0)
                {
                    // prepare for the next level
                    currLevel = new Queue<DataStructures.TreeNode>(nextLevel);
                    nextLevel.Clear();

                    while (currLevel.Count > 0)
                    {
                        node = currLevel.Dequeue();

                        // add child nodes of the current level
                        // in the queue for the next level
                        if (node.Left != null) nextLevel.Enqueue(node.Left);
                        if (node.Right != null) nextLevel.Enqueue(node.Right);
                    }

                    // The current level is finished.
                    // Its last element is the rightmost one.
                    if (currLevel.Count == 0) rightSide.Add(node.Val);
                }
                return rightSide;
            }/* 
Approach 2: BFS: One Queue + Sentinel
Complexity Analysis
•	Time complexity: O(N) since one has to visit each node.
•	Space complexity: O(D) to keep the queues, where D is a tree diameter. Let's use the last level to estimate the queue size. This level could contain up to N/2 tree nodes in the case of complete binary tree.

 */
            public List<int> BFSWithOneQueueAndSentinelNode(DataStructures.TreeNode root)
            {
                if (root == null) return new List<int>();

                Queue<DataStructures.TreeNode> treeNodeQueue = new Queue<DataStructures.TreeNode>();
                treeNodeQueue.Enqueue(root);
                treeNodeQueue.Enqueue(null);

                DataStructures.TreeNode previousNode = null;
                DataStructures.TreeNode currentNode = root;
                List<int> rightSideViewList = new List<int>();

                while (treeNodeQueue.Count > 0)
                {
                    previousNode = currentNode;
                    currentNode = treeNodeQueue.Dequeue();

                    while (currentNode != null)
                    {
                        // add child nodes in the queue
                        if (currentNode.Left != null)
                        {
                            treeNodeQueue.Enqueue(currentNode.Left);
                        }
                        if (currentNode.Right != null)
                        {
                            treeNodeQueue.Enqueue(currentNode.Right);
                        }

                        previousNode = currentNode;
                        currentNode = treeNodeQueue.Dequeue();
                    }

                    // the current level is finished
                    // and previousNode is its rightmost element
                    rightSideViewList.Add(previousNode.Val);

                    // add a sentinel to mark the end
                    // of the next level
                    if (treeNodeQueue.Count > 0) treeNodeQueue.Enqueue(null);
                }

                return rightSideViewList;
            }
            /* Approach 3: BFS: One Queue + Level Size Measurements 
            Complexity Analysis
•	Time complexity: O(N) since one has to visit each node.
•	Space complexity: O(D) to keep the queues, where D is a tree diameter. Let's use the last level to estimate the queue size. This level could contain up to N/2 tree nodes in the case of complete binary tree.

            */
            public IList<int> BFSWithOneQueueAndLevelSizeMeasure(DataStructures.TreeNode root)
            {
                if (root == null) return new List<int>();

                Queue<DataStructures.TreeNode> queue = new Queue<DataStructures.TreeNode>();
                queue.Enqueue(root);
                List<int> rightSide = new List<int>();

                while (queue.Count > 0)
                {
                    int levelLength = queue.Count;

                    for (int i = 0; i < levelLength; ++i)
                    {
                        DataStructures.TreeNode currentNode = queue.Dequeue();

                        // if it's the rightmost element
                        if (i == levelLength - 1)
                        {
                            rightSide.Add(currentNode.Val);
                        }

                        // add child nodes in the queue
                        if (currentNode.Left != null)
                        {
                            queue.Enqueue(currentNode.Left);
                        }
                        if (currentNode.Right != null)
                        {
                            queue.Enqueue(currentNode.Right);
                        }
                    }
                }
                return rightSide;
            }
            /* Approach 4: Recursive DFS
Complexity Analysis
•	Time complexity: O(N) since one has to visit each node.
•	Space complexity: O(H) to keep the recursion stack, where H is a tree height. The worst-case situation is a skewed tree when H=N.

             */
            List<int> rightside = new();
            public List<int> DFSRec(DataStructures.TreeNode root)
            {
                if (root == null) return rightside;

                Helper(root, 0);
                return rightside;
            }
            private void Helper(DataStructures.TreeNode node, int level)
            {
                if (level == rightside.Count) rightside.Add(node.Val);

                if (node.Right != null) Helper(node.Right, level + 1);
                if (node.Left != null) Helper(node.Left, level + 1);
            }



        }

   


    }

}