using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;
using static AlgoDSPlay.LinkedListOps;

namespace AlgoDSPlay
{
    public class BinarySearchTreeProbs
    {
        /*
        96. Unique Binary Search Trees
https://leetcode.com/problems/unique-binary-search-trees/description/

        */
        public class NumUniqueBSTSol
        {
            /*
            
Approach 1: Dynamic Programming
Complexity Analysis
•	Time complexity : the main computation of the algorithm is done at the statement with G[i].
So the time complexity is essentially the number of iterations for the statement,
which is ∑i=2 to n i=((2+n)(n−1))/2, to be exact, therefore the time complexity is O(N^2)
•	Space complexity : The space complexity of the above algorithm is mainly the storage to
keep all the intermediate solutions, therefore O(N).

            */
            public int NumUniqueBSTDP(int n)
            {
                int[] G = new int[n + 1];
                G[0] = 1;
                G[1] = 1;
                for (int i = 2; i <= n; ++i)
                {
                    for (int j = 1; j <= i; ++j)
                    {
                        G[i] += G[j - 1] * G[i - j];
                    }
                }

                return G[n];

            }
        }
        /*
        
Approach 2: Mathematical Deduction (MD)
Complexity Analysis
•	Time complexity : O(N), as one can see, there is one single loop in the algorithm.
•	Space complexity : O(1), we use only one variable to store all the intermediate results and the final one.

        */
        public int NumUniqueBSTMD(int n)
        {
            // Note: we should use long here instead of int, otherwise overflow
            long C = 1;
            for (int i = 0; i < n; ++i)
            {
                C = C * 2 * (2 * i + 1) / (i + 2);
            }

            return (int)C;
        }


        /*
95. Unique Binary Search Trees II
https://leetcode.com/problems/unique-binary-search-trees-ii/description/

TODO: Revisit on Time N Space Complexities
        */
        public class GenerateUniqueBSTreesSol
        {
            /*
           
Approach 1: Recursive Dynamic Programming (DPRec)

            */

            public IList<TreeNode> DPRec(int n)
            {
                var memo = new Dictionary<(int, int), IList<TreeNode>>();
                return AllPossibleBST(1, n, memo);
            }
            public IList<TreeNode> AllPossibleBST(
    int start, int end, Dictionary<(int, int), IList<TreeNode>> memo)
            {
                List<TreeNode> res = new List<TreeNode>();
                if (start > end)
                {
                    res.Add(null);
                    return res;
                }

                var key = (start, end);
                if (memo.ContainsKey(key))
                {
                    return memo[key];
                }

                // Iterate through all values from start to end to construct left and
                // right subtree recursively.
                for (int i = start; i <= end; ++i)
                {
                    IList<TreeNode> leftSubTrees = AllPossibleBST(start, i - 1, memo);
                    IList<TreeNode> rightSubTrees = AllPossibleBST(i + 1, end, memo);
                    // Loop through all left and right subtrees and connect them to ith
                    // root.
                    foreach (TreeNode left in leftSubTrees)
                    {
                        foreach (TreeNode right in rightSubTrees)
                        {
                            TreeNode root = new TreeNode(i, left, right);
                            res.Add(root);
                        }
                    }
                }

                memo[key] = res;
                return res;
            }
            /*Approach 2: Iterative Dynamic Programming*/

            public IList<TreeNode> GenerateTreesDPIterative(int n)
            {
                var dp = new List<List<List<TreeNode>>>(n + 1);
                for (int i = 0; i <= n; i++)
                {
                    dp.Add(new List<List<TreeNode>>(n + 1));
                    for (int j = 0; j <= n; j++)
                    {
                        dp[i].Add(new List<TreeNode>());
                    }
                }

                for (int i = 1; i <= n; i++)
                {
                    dp[i][i].Add(new TreeNode(i));
                }

                for (int numOfNodes = 2; numOfNodes <= n; numOfNodes++)
                {
                    for (int start = 1; start <= n - numOfNodes + 1; start++)
                    {
                        int end = start + numOfNodes - 1;
                        for (int i = start; i <= end; i++)
                        {
                            List<TreeNode> leftSubtrees =
                                (i != start) ? dp[start][i - 1] : new List<TreeNode>();
                            if (leftSubtrees.Count == 0)
                                leftSubtrees.Add(null);
                            List<TreeNode> rightSubtrees =
                                (i != end) ? dp[i + 1][end] : new List<TreeNode>();
                            if (rightSubtrees.Count == 0)
                                rightSubtrees.Add(null);
                            foreach (TreeNode left in leftSubtrees)
                            {
                                foreach (TreeNode right in rightSubtrees)
                                {
                                    dp[start][end].Add(new TreeNode(i, left, right));
                                }
                            }
                        }
                    }
                }

                return dp[1][n];
            }
            /*            
Approach 3: Dynamic Programming with Space Optimization (DPSO)

            */
            public IList<TreeNode> DPSO(int n)
            {
                List<List<TreeNode>> dp = new List<List<TreeNode>>(n + 1);
                for (int i = 0; i <= n; i++)
                {
                    dp.Add(new List<TreeNode>());
                }

                dp[0].Add(null);
                for (int numberOfNodes = 1; numberOfNodes <= n; numberOfNodes++)
                {
                    for (int i = 1; i <= numberOfNodes; i++)
                    {
                        int j = numberOfNodes - i;
                        foreach (TreeNode left in dp[i - 1])
                        {
                            foreach (TreeNode right in dp[j])
                            {
                                TreeNode root = new TreeNode(i, left, Clone(right, i));
                                dp[numberOfNodes].Add(root);
                            }
                        }
                    }
                }

                return dp[n];
            }

            private TreeNode Clone(TreeNode node, int offset)
            {
                if (node == null)
                {
                    return null;
                }

                TreeNode clonedNode = new TreeNode(node.Val + offset);
                clonedNode.Left = Clone(node.Left, offset);
                clonedNode.Right = Clone(node.Right, offset);
                return clonedNode;
            }

        }


        /*
        98. Validate Binary Search Tree
https://leetcode.com/problems/validate-binary-search-tree/description/
//https://www.algoexpert.io/questions/validate-bst

        */
        public class ValidateBSTSol
        {

            /*
           
Approach 1: Recursive Traversal with Valid Range
Complexity Analysis
•	Time complexity: O(N) since we visit each node exactly once.
•	Space complexity: O(N) since we keep up to the entire tree.


            */
            private bool Validate(TreeNode root, int? low, int? high)
            {
                // Empty trees are valid BSTs.
                if (root == null)
                {
                    return true;
                }

                // The current node's value must be between low and high.
                if ((low.HasValue && root.Val <= low.Value) ||
                    (high.HasValue && root.Val >= high.Value))
                {
                    return false;
                }

                // The left and right subtree must also be valid.
                return Validate(root.Right, root.Val, high) &&
                       Validate(root.Left, low, root.Val);

            }
            public bool RecTraverseWithValidRange(TreeNode root)
            {
                return Validate(root, null, null);
            }

            /*
            
            Approach 2: Iterative Traversal with Valid Range
Complexity Analysis
•	Time complexity: O(N) since we visit each node exactly once.
•	Space complexity: O(N) since we keep up to the entire tree.

            */
            private Stack<TreeNode> stack = new Stack<TreeNode>();
            private Stack<int?> lowerLimits = new Stack<int?>();
            private Stack<int?> upperLimits = new Stack<int?>();

            public void Update(TreeNode root, int? low, int? high)
            {
                stack.Push(root);
                lowerLimits.Push(low);
                upperLimits.Push(high);
            }

            public bool IterTraverseWithValidRange(TreeNode root)
            {
                int? low = null, high = null;
                Update(root, low, high);
                while (stack.Count > 0)
                {
                    root = stack.Pop();
                    low = lowerLimits.Pop();
                    high = upperLimits.Pop();
                    if (root == null)
                        continue;
                    int val = root.Val;
                    if (low != null && val <= low)
                    {
                        return false;
                    }

                    if (high != null && val >= high)
                    {
                        return false;
                    }

                    Update(root.Right, val, high);
                    Update(root.Left, low, val);
                }

                return true;
            }

            /*
            
Approach 3: Recursive Inorder Traversal
Complexity Analysis
•	Time complexity: O(N) in the worst case when the tree is a BST or the "bad" element is a rightmost leaf.
•	Space complexity: O(N) for the space on the run-time stack.


            */
            private int? prev;

            public bool RecInOrderTraverse(TreeNode root)
            {
                prev = null;
                return Inorder(root);
            }

            private bool Inorder(TreeNode root)
            {
                if (root == null)
                {
                    return true;
                }

                if (!Inorder(root.Left))
                {
                    return false;
                }

                if (prev != null && root.Val <= prev)
                {
                    return false;
                }

                prev = root.Val;
                return Inorder(root.Right);
            }


            /*
    Approach 4: Iterative Inorder Traversal
    Complexity Analysis
    •	Time complexity: O(N) in the worst case when the tree is BST or the "bad" element is the rightmost leaf.
    •	Space complexity: O(N) to keep stack.

    */
            public bool IterInorderTraverse(TreeNode root)
            {
                Stack<TreeNode> stack = new Stack<TreeNode>();
                TreeNode prev = null;
                while (stack.Count > 0 || root != null)
                {
                    while (root != null)
                    {
                        stack.Push(root);
                        root = root.Left;
                    }

                    root = stack.Pop();
                    // If next element in inorder traversal
                    // is smaller than the previous one
                    // that's not BST.
                    if (prev != null && root.Val <= prev.Val)
                    {
                        return false;
                    }

                    prev = root;
                    root = root.Right;
                }

                return true;
            }




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
        }


        /*
        99. Recover Binary Search Tree
        https://leetcode.com/problems/recover-binary-search-tree/description/

        */
        public class RecoverBSTSol
        {
            /*          
Approach 1: Sort an Almost Sorted Array Where Two Elements Are Swapped
Complexity Analysis
•	Time complexity: O(N). To compute inorder traversal takes O(N) time, to identify and to swap back swapped nodes O(N) in the worst case.
•	Space complexity: O(N) since we keep inorder traversal nums with N elements.

            */
            public static void InorderTwoSwap(TreeNode root)
            {
                List<int> nums = new List<int>();
                Inorder(root, nums);
                int[] swapped = FindTwoSwapped(nums);
                Recover(root, 2, swapped[0], swapped[1]);

                void Inorder(TreeNode root, List<int> nums)
                {
                    if (root == null)
                        return;
                    Inorder(root.Left, nums);
                    nums.Add(root.Val);
                    Inorder(root.Right, nums);
                }
                int[] FindTwoSwapped(List<int> nums)
                {
                    int n = nums.Count;
                    int x = -1, y = -1;
                    bool swappedFirstOccurrence = false;
                    for (int i = 0; i < n - 1; ++i)
                    {
                        if (nums[i + 1] < nums[i])
                        {
                            y = nums[i + 1];
                            if (!swappedFirstOccurrence)
                            {
                                // The first swap occurrence
                                x = nums[i];
                                swappedFirstOccurrence = true;
                            }
                            else
                            {
                                // The second swap occurrence
                                break;
                            }
                        }
                    }

                    return new int[] { x, y };
                }

                void Recover(TreeNode r, int count, int x, int y)
                {
                    if (r != null)
                    {
                        if (r.Val == x || r.Val == y)
                        {
                            r.Val = r.Val == x ? y : x;
                            if (--count == 0)
                                return;
                        }

                        Recover(r.Left, count, x, y);
                        Recover(r.Right, count, x, y);
                    }
                }


            }


            /*

 Approach 2: Iterative Inorder Traversal
 Complexity Analysis
 •	Time complexity: O(N) in the worst case when one of the swapped nodes is a rightmost leaf.
 •	Space complexity : up to O(N) to keep the stack in the worst case when the tree is completely lean.

            */


            public static void InorderIterative(TreeNode root)
            {
                Stack<TreeNode> stack = new Stack<TreeNode>();
                TreeNode x = null, y = null, pred = null;
                while (stack.Count != 0 || root != null)
                {
                    while (root != null)
                    {
                        stack.Push(root);
                        root = root.Left;
                    }

                    root = stack.Pop();
                    if (pred != null && root.Val < pred.Val)
                    {
                        y = root;
                        if (x == null)
                            x = pred;
                        else
                            break;
                    }

                    pred = root;
                    root = root.Right;
                }

                Swap(ref x, ref y);

                void Swap(ref TreeNode a, ref TreeNode b)
                {
                    int tmp = a.Val;
                    a.Val = b.Val;
                    b.Val = tmp;
                }
            }

            /*
            Approach 3: Recursive Inorder Traversal
        Complexity Analysis
        •	Time complexity: O(N) in the worst case when one of the swapped nodes is a rightmost leaf.
        •	Space complexity : up to O(N) to keep the stack in the worst case when the tree is completely lean.

            */
            public static void InorderRec(TreeNode root)
            {
                TreeNode x = null, y = null, pred = null;
                FindTwoSwapped(root);
                int tmp = x.Val;
                x.Val = y.Val;
                y.Val = tmp;

                void FindTwoSwapped(TreeNode root)
                {
                    if (root == null)
                        return;
                    FindTwoSwapped(root.Left);
                    if (pred != null && root.Val < pred.Val)
                    {
                        y = root;
                        if (x == null)
                            x = pred;
                        else
                            return;
                    }

                    pred = root;
                    FindTwoSwapped(root.Right);
                }
            }


            /*
            Approach 4: Morris Inorder Traversal
            Complexity Analysis
    •	Time complexity : O(N) since we visit each node up to two times.
    •	Space complexity : O(1).

            */

            public void MorrisInorder(TreeNode root)
            {
                // predecessor is a Morris predecessor.
                // In the 'loop' cases it could be equal to the node itself predecessor
                // == root. pred is a 'true' predecessor, the previous node in the
                // inorder traversal.
                TreeNode x = null, y = null, pred = null, predecessor = null;
                while (root != null)
                {
                    // If there is a left child
                    // then compute the predecessor.
                    // If there is no link predecessor.Right = root --> set it.
                    // If there is a link predecessor.Right = root --> break it.
                    if (root.Left != null)
                    {
                        // Predecessor node is one step left
                        // and then right till you can.
                        predecessor = root.Left;
                        while (predecessor.Right != null && predecessor.Right != root)
                            predecessor = predecessor.Right;
                        // Set the link predecessor.Right = root
                        // and go to explore left subtree
                        if (predecessor.Right == null)
                        {
                            predecessor.Right = root;
                            root = root.Left;
                        }
                        // Break the link predecessor.Right = root
                        // link is broken : time to change subtree and go right
                        else
                        {
                            // Check for the swapped nodes
                            if (pred != null && root.Val < pred.Val)
                            {
                                y = root;
                                if (x == null)
                                    x = pred;
                            }

                            pred = root;
                            predecessor.Right = null;
                            root = root.Right;
                        }
                    }
                    // If there is no left child
                    // then just go right.
                    else
                    {
                        // Check for the swapped nodes
                        if (pred != null && root.Val < pred.Val)
                        {
                            y = root;
                            if (x == null)
                                x = pred;
                        }

                        pred = root;
                        root = root.Right;
                    }
                }

                Swap(x, y);

                void Swap(TreeNode a, TreeNode b)
                {
                    int tmp = a.Val;
                    a.Val = b.Val;
                    b.Val = tmp;
                }
            }



        }


        /*
        109. Convert Sorted List to Binary Search Tree
        https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/
        */

        public class SortedListToBSTSol
        {
            /*
            Approach 1: 
            Complexity Analysis
•	Time Complexity: O(NlogN). Suppose our linked list consists of N elements. For every list we pass to our recursive function, we have to calculate the middle element for that list. For a list of size N, it takes N/2 steps to find the middle element i.e. O(N) to find the mid. We do this for every half of the original linked list. From the looks of it, this seems to be an O(N^2) algorithm. However, on closer analysis, it turns out to be a bit more efficient than O(N^2).
•	Space Complexity: O(logN). Since we are resorting to recursion, there is always the added space complexity of the recursion stack that comes into picture. This could have been O(N) for a skewed tree, but the question clearly states that we need to maintain the height balanced property. This ensures the height of the tree to be bounded by O(logN). Hence, the space complexity is O(logN).
            */
            public TreeNode Rec(ListNode head)
            {
                if (head == null)
                    return null;
                ListNode mid = FindMiddleElement(head);
                TreeNode node = new TreeNode(mid.Val);
                if (head == mid)
                    return node;
                node.Left = this.Rec(head);
                node.Right = this.Rec(mid.Next);
                return node;

                ListNode FindMiddleElement(ListNode head)
                {
                    ListNode prevPtr = null;
                    ListNode slowPtr = head;
                    ListNode fastPtr = head;
                    while (fastPtr != null && fastPtr.Next != null)
                    {
                        prevPtr = slowPtr;
                        slowPtr = slowPtr.Next;
                        fastPtr = fastPtr.Next.Next;
                    }

                    if (prevPtr != null)
                        prevPtr.Next = null;
                    return slowPtr;
                }
            }
            /*
            Approach 2: Recursion + Conversion to Array
Complexity Analysis
•	Time Complexity: The time complexity comes down to just O(N) now since we convert the linked list to an array initially and then we convert the array into a BST. Accessing the middle element now takes O(1) time and hence the time complexity comes down.
•	Space Complexity: Since we used extra space to bring down the time complexity, the space complexity now goes up to O(N) as opposed to just O(logN) in the previous solution. This is due to the array we construct initially.

            */
            public static TreeNode RecWithArrayConvertion(ListNode head)
            {
                List<int> values = new List<int>();
                // Form an array out of the given linked list and then
                // use the array to form the BST.
                MapListToValues(head, values);
                // Convert the array to
                return RecWithArrayConvertion(0, values.Count - 1, values);

                void MapListToValues(ListNode head, List<int> values)
                {
                    while (head != null)
                    {
                        values.Add(head.Val);
                        head = head.Next;
                    }
                }
                TreeNode RecWithArrayConvertion(int left, int right, List<int> values)

                {
                    // Invalid case
                    if (left > right)
                    {
                        return null;
                    }

                    // Middle element forms the root.
                    int mid = (left + right) / 2;
                    TreeNode node = new TreeNode(values[mid]);
                    // Base case for when there is only one element left in the array
                    if (left == right)
                    {
                        return node;
                    }

                    // Recursively form BST on the two halves
                    node.Left = RecWithArrayConvertion(left, mid - 1, values);
                    node.Right = RecWithArrayConvertion(mid + 1, right, values);
                    return node;
                }
            }

            /*
            Approach 3: Inorder Simulation
            Complexity Analysis
•	Time Complexity: The time complexity is still O(N) since we still have to process each of the nodes in the linked list once and form corresponding BST nodes.
•	Space Complexity: O(logN) since now the only extra space is used by the recursion stack and since we are building a height balanced BST, the height is bounded by logN.

            */

            public static TreeNode InorderSimulation(ListNode head)
            {
                int size = FindSize(head);
                return ConvertListToBST(0, size - 1, head);

                int FindSize(ListNode head)
                {
                    ListNode ptr = head;
                    int c = 0;
                    while (ptr != null)
                    {
                        ptr = ptr.Next;
                        c += 1;
                    }

                    return c;
                }

                TreeNode ConvertListToBST(int l, int r, ListNode head)
                {
                    if (l > r)
                        return null;
                    int mid = (l + r) / 2;
                    TreeNode left = ConvertListToBST(l, mid - 1, head);
                    TreeNode node = new TreeNode(head.Val);
                    node.Left = left;
                    head = head.Next;
                    node.Right = ConvertListToBST(mid + 1, r, head);
                    return node;
                }
            }
        }

        /*
        108. Convert Sorted Array to Binary Search Tree
https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/
        */
        public class SortedArrayToBSTSol
        {
            /*            
Approach 1: Preorder Traversal: Always Choose Left Middle Node as a Root
Complexity Analysis
•	Time complexity: O(N) since we visit each node exactly once.
•	Space complexity: O(logN).
The recursion stack requires O(logN) space because the tree is height-balanced. Note that the O(N) space used to store the output does not count as auxiliary space, so it is not included in the space complexity.

            */
            public static TreeNode PreorderTraverseWithLeftMidNodeAsRoot(int[] nums)
            {
                return Helper(nums, 0, nums.Length - 1);

                TreeNode Helper(int[] nums, int left, int right)
                {
                    if (left > right)
                    {
                        return null;
                    }

                    int p = (left + right) / 2;
                    TreeNode root = new TreeNode(nums[p]);
                    root.Left = Helper(nums, left, p - 1);
                    root.Right = Helper(nums, p + 1, right);
                    return root;
                }

            }

            /*
            Approach 2: Preorder Traversal: Always Choose Right Middle Node as a Root
Complexity Analysis
•	Time complexity: O(N) since we visit each node exactly once.
•	Space complexity: O(logN).
The recursion stack requires O(logN) space because the tree is height-balanced. Note that the O(N) space used to store the output does not count as auxiliary space, so it is not included in the space complexity.

            */
            public static TreeNode PreorderTraverseWithRighttMidNodeAsRoot(int[] nums)
            {

                return Helper(0, nums.Length - 1, nums);

                TreeNode Helper(int left, int right, int[] nums)
                {

                    if (left > right)
                        return null;
                    // always choose right middle node as a root
                    int p = (left + right) / 2;
                    if ((left + right) % 2 == 1)
                        ++p;
                    // preorder traversal: node -> left -> right
                    TreeNode root = new TreeNode(nums[p]);
                    root.Left = Helper(left, p - 1, nums);
                    root.Right = Helper(p + 1, right, nums);
                    return root;
                }
            }

            /*
            Approach 3: Preorder Traversal: Choose a Random Middle Node as a Root
           Complexity Analysis
           •	Time complexity: O(N) since we visit each node exactly once.
           •	Space complexity: O(logN).
           The recursion stack requires O(logN) space because the tree is height-balanced. Note that the O(N) space used to store the output does not count as auxiliary space, so it is not included in the space complexity.

                       */
            public static TreeNode PreorderTraverseWithRandomtMidNodeAsRoot(int[] nums)
            {
                Random rand = new Random();
                return Helper(nums, 0, nums.Length - 1);
                TreeNode Helper(int[] nums, int left, int right)
                {
                    if (left > right)
                        return null;
                    int p = (left + right) / 2;
                    if ((left + right) % 2 == 1)
                        p += rand.Next(2);
                    TreeNode root = new TreeNode(nums[p]);
                    root.Left = Helper(nums, left, p - 1);
                    root.Right = Helper(nums, p + 1, right);
                    return root;
                }
            }




        }


        /* 272. Closest Binary Search Tree Value II
        https://leetcode.com/problems/closest-binary-search-tree-value-ii/description/
         */
        public class ClosestKValuesSol
        {

            /* 
            Approach 1: Sort With Custom Comparator 
            Complexity Analysis
            Given n as the number of nodes in the tree,
            •	Time complexity: O(n⋅logn)
            We traverse the tree and collect all values in O(n). Then, we sort the values which costs O(n⋅logn).
            •	Space complexity: O(n)
            Both arr and the recursion call stack use O(n) space. Depending on the language, some space is also used for sorting, but not more than O(n).

            */
            public IList<int> SortWithCustomCompare(TreeNode root, double target, int k)
            {
                List<int> valuesList = new List<int>();
                Dfs(root, valuesList);

                valuesList.Sort((value1, value2) => Math.Abs(value1 - target) <= Math.Abs(value2 - target) ? -1 : 1);

                return valuesList.GetRange(0, k);
            }

            private void Dfs(TreeNode node, List<int> valuesList)
            {
                if (node == null)
                {
                    return;
                }

                valuesList.Add(node.Val);
                Dfs(node.Left, valuesList);
                Dfs(node.Right, valuesList);
            }

            /* 
                        Approach 2: Traverse With Heap 
Complexity Analysis
Given n as the number of nodes in the tree,
•	Time complexity: O(n⋅logk)
A heap operation's cost is a function of the size of the heap. We are limiting the size of our heap to k, so heap operations will cost O(logk).
We visit each node once. At each node, we perform up to two heap operations. Therefore, we perform a maximum of 2n heap operations, giving us a time complexity of O(n⋅logk).
•	Space complexity: O(n+k)
We need O(n) space for the recursion call stack, and O(k) space for the heap.

                        */

            public List<int> TraverseWithHeap(TreeNode root, double target, int k)
            {
                //TODO: Test below heap comparator 
                PriorityQueue<int, int> heap = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => Math.Abs(a - target) > Math.Abs(b - target) ? -1 : 1));
                Dfs(root, heap, k);
                List<int> results = new List<int>();
                while (heap.Count > 0)
                    results.Add(heap.Dequeue());
                return results;
            }

            private void Dfs(TreeNode node, PriorityQueue<int, int> heap, int k)
            {
                if (node == null)
                {
                    return;
                }

                heap.Enqueue(node.Val, node.Val);
                if (heap.Count > k)
                {
                    heap.Dequeue();
                }

                Dfs(node.Left, heap, k);
                Dfs(node.Right, heap, k);
            }

            /* 
            Approach 3: Inorder Traversal + Sliding Window 
            Complexity Analysis
Given n as the number of nodes in the tree,
•	Time complexity: O(n+k)
First, we perform a DFS on the tree to build arr which costs O(n).
Next, we perform either a binary search or linear scan on arr which costs O(logn) or O(n). Neither will change the complexity.
Finally, we perform a sliding window process that costs O(k) since we add an element to the window at each iteration and stop when the window has a size of k.
•	Space complexity: O(n)
Both arr and the recursion call stack use O(n) space.

            */
            public IList<int> InorderTraverseWithSlidingWindow(TreeNode root, double target, int k)
            {
                List<int> valuesList = new List<int>();
                DepthFirstSearch(root, valuesList);

                int startIndex = 0;
                double minimumDifference = double.MaxValue;

                for (int i = 0; i < valuesList.Count; i++)
                {
                    if (Math.Abs(valuesList[i] - target) < minimumDifference)
                    {
                        minimumDifference = Math.Abs(valuesList[i] - target);
                        startIndex = i;
                    }
                }

                int leftIndex = startIndex;
                int rightIndex = startIndex + 1;

                while (rightIndex - leftIndex - 1 < k)
                {
                    // Be careful to not go out of bounds
                    if (leftIndex < 0)
                    {
                        rightIndex += 1;
                        continue;
                    }

                    if (rightIndex == valuesList.Count || Math.Abs(valuesList[leftIndex] - target) <= Math.Abs(valuesList[rightIndex] - target))
                    {
                        leftIndex -= 1;
                    }
                    else
                    {
                        rightIndex += 1;
                    }
                }

                return valuesList.GetRange(leftIndex + 1, rightIndex - (leftIndex + 1));
            }

            private void DepthFirstSearch(TreeNode node, List<int> valuesList)
            {
                if (node == null)
                {
                    return;
                }

                DepthFirstSearch(node.Left, valuesList);
                valuesList.Add(node.Val);
                DepthFirstSearch(node.Right, valuesList);
            }

            /* 
                        Approach 4: Binary Search The Left Bound 
                        Complexity Analysis
            Given n as the number of nodes in the tree,
            •	Time complexity: O(n) in Java, O(n+k) in Python
            First, we perform a DFS on the tree to build arr which costs O(n).
            Next, we perform a binary search on arr which costs O(log(n−k)).
            Finally, we return the answer. In Java, arr.subList() is an O(1) operation. In Python, we spend O(k) to create the answer.
            Note that an interviewer may find it reasonable to ignore the O(k) to build the answer, thus giving this algorithm a time complexity of O(n).
            •	Space complexity: O(n)
            Both arr and the recursion call stack use O(n) space.

                        */
            public List<int> BinarySearchTheLeftBound(TreeNode root, double target, int k)
            {
                List<int> valuesList = new List<int>();
                DepthFirstSearch(root, valuesList);

                int leftIndex = 0;
                int rightIndex = valuesList.Count - k;

                while (leftIndex < rightIndex)
                {
                    int middleIndex = (leftIndex + rightIndex) / 2;
                    if (Math.Abs(target - valuesList[middleIndex + k]) < Math.Abs(target - valuesList[middleIndex]))
                    {
                        leftIndex = middleIndex + 1;
                    }
                    else
                    {
                        rightIndex = middleIndex;
                    }
                }

                return valuesList.GetRange(leftIndex, k);
            }

            /* 
            Approach 5: Build The Window With Deque 
            Complexity Analysis
Given n as the number of nodes in the tree,
•	Time complexity: O(n)
We visit each node at most once during the traversal. With an efficient deque implementation, the work done at each node is O(1).
•	Space complexity: O(n+k)
We use O(n) space for the recursion call stack and O(k) space for queue.

            */
            public IList<int> WindowWithDeque(TreeNode root, double target, int k)
            {
                Deque<int> queue = new Deque<int>();
                Dfs(root, queue, k, target);
                return new List<int>(queue);
            }

            public void Dfs(TreeNode node, Deque<int> queue, int k, double target)
            {
                if (node == null)
                {
                    return;
                }

                Dfs(node.Left, queue, k, target);
                queue.Add(node.Val);
                if (queue.Count > k)
                {
                    if (Math.Abs(target - queue.PeekFirst()) <= Math.Abs(target - queue.PeekLast()))
                    {
                        queue.RemoveLast();
                        return;
                    }
                    else
                    {
                        queue.RemoveFirst();
                    }
                }

                Dfs(node.Right, queue, k, target);
            }


            public class Deque<T> : LinkedList<T>
            {
                public void Add(T value)
                {
                    this.AddLast(value);
                }

                public T PeekFirst()
                {
                    return this.First.Value;
                }

                public T PeekLast()
                {
                    return this.Last.Value;
                }

                public void RemoveFirst()
                {
                    this.RemoveFirst();
                }

                public void RemoveLast()
                {
                    this.RemoveLast();
                }
            }

        }

        /* 1569. Number of Ways to Reorder Array to Get Same BST
        https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/description/
         */
        public class NumOfWaysToReorderArrayToGetSameBSTSol
        {
            private long mod = (long)1e9 + 7;
            private long[,] table;
            /* Approach: Recursion 
            Complexity Analysis
            Let m be the size of nums.
            •	Time complexity: O(m^2)
            o	In Java or C++, a table of Pascal's triangle of size m×m is built, which takes O(m^2) time.
            o	dfs(nums) recursively calls itself to process the left and right subtrees of the current node nums[0]. Since the total size of the subtrees decreases by 1 at each level of the recursion, the maximum height of the recursion tree is m. Thus the total time complexity of the recursive solution is O(m1^2) because in each call we are doing O(m) work creating the subsequences.
            •	Space complexity: O(m^2) or O(m)
            o	In Java or C++, a table of Pascal's triangle of size m×m is built.
            o	The recursive solution uses the call stack to keep track of the current subtree being processed. The maximum depth of the call stack is equal to the height of the BST constructed from the input array. In the worst case, nums may form a degenerate BST (e.g., a sorted array), which has a height of m−1, and the stack can hold up to m−1 calls, resulting in a space complexity of O(m).

            */
            public int DFSRecur(int[] nums)
            {
                int length = nums.Length;

                // Table of Pascal's triangle
                table = new long[length, length];
                for (int i = 0; i < length; ++i)
                {
                    table[i, 0] = table[i, i] = 1;
                }
                for (int i = 2; i < length; i++)
                {
                    for (int j = 1; j < i; j++)
                    {
                        table[i, j] = (table[i - 1, j - 1] + table[i - 1, j]) % mod;
                    }
                }
                List<int> arrList = nums.ToList();
                return (int)((Dfs(arrList) - 1) % mod);
            }

            private long Dfs(List<int> nums)
            {
                int length = nums.Count;
                if (length < 3)
                {
                    return 1;
                }

                List<int> leftNodes = new List<int>();
                List<int> rightNodes = new List<int>();
                for (int i = 1; i < length; ++i)
                {
                    if (nums[i] < nums[0])
                    {
                        leftNodes.Add(nums[i]);
                    }
                    else
                    {
                        rightNodes.Add(nums[i]);
                    }
                }
                long leftWays = Dfs(leftNodes) % mod;
                long rightWays = Dfs(rightNodes) % mod;

                return (((leftWays * rightWays) % mod) * table[length - 1, leftNodes.Count]) % mod;
            }
        }



        /* 426. Convert Binary Search Tree to Sorted Doubly Linked List
        https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM

         */
        class BSTToDoublyListSol
        {
            // the smallest (first) and the largest (last) nodes
            TreeNode first = null;
            TreeNode last = null;
            /* 
            Approach 1: Recursion
            Complexity Analysis
            •	Time complexity : O(N) since each node is processed exactly once.
            •	Space complexity : O(N). We have to keep a recursion stack of the size of the tree height, which is O(logN) for the best case of a completely balanced tree and O(N) for the worst case of a completely unbalanced tree.

             */
            public TreeNode TreeToDoublyList(TreeNode root)
            {
                if (root == null) return null;

                Helper(root);

                // close DLL
                last.Right = first;
                first.Left = last;
                return first;
            }
            public void Helper(TreeNode node)
            {
                if (node != null)
                {
                    // left
                    Helper(node.Left);

                    // node 
                    if (last != null)
                    {
                        // link the previous node (last)
                        // with the current one (node)
                        last.Right = node;
                        node.Left = last;
                    }
                    else
                    {
                        // keep the smallest node
                        // to close DLL later on
                        first = node;
                    }
                    last = node;

                    // right
                    Helper(node.Right);
                }
            }


        }


        /* 173. Binary Search Tree Iterator
        https://leetcode.com/problems/binary-search-tree-iterator/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class BSTIteratorSol
        {

            /* Approach 1: Flattening the BST
            Complexity analysis
            •	Time complexity : O(N) is the time taken by the constructor for the iterator. The problem statement only asks us to analyze the complexity of the two functions, however, when implementing a class, it's important to also note the time it takes to initialize a new object of the class and in this case it would be linear in terms of the number of nodes in the BST. In addition to the space occupied by the new array we initialized, the recursion stack for the inorder traversal also occupies space but that is limited to O(h) where h is the height of the tree.
            o	next() would take O(1)
            o	hasNext() would take O(1)
            •	Space complexity : O(N) since we create a new array to contain all the nodes of the BST. This doesn't comply with the requirement specified in the problem statement that the maximum space complexity of either of the functions should be O(h) where h is the height of the tree and for a well balanced BST, the height is usually logN. So, we get great time complexities but we had to compromise on the space. Note that the new array is used for both the function calls and hence the space complexity for both the calls is O(N).

             */
            class FlatBSTSol
            {
                List<int> nodesSorted;
                int index;

                public FlatBSTSol(TreeNode root)
                {
                    // Array containing all the nodes in the sorted order
                    this.nodesSorted = new();

                    // Pointer to the next smallest element in the BST
                    this.index = -1;

                    // Call to flatten the input binary search tree
                    this._inorder(root);
                }

                private void _inorder(TreeNode root)
                {
                    if (root == null)
                    {
                        return;
                    }

                    this._inorder(root.Left);
                    this.nodesSorted.Add(root.Val);
                    this._inorder(root.Right);
                }

                /**
                 * @return the next smallest number
                 */
                public int Next()
                {
                    return this.nodesSorted[++this.index];
                }

                /**
                 * @return whether we have a next smallest number
                 */
                public bool HasNext()
                {
                    return this.index + 1 < this.nodesSorted.Count;
                }
            }
            /*             Approach 2: Controlled Recursion
Complexity analysis
•	Time complexity : The time complexity for this approach is very interesting to analyze. Let's look at the complexities for both the functions in the class:
o	hasNext is the easier of the lot since all we do in this is to return true if there are any elements left in the stack. Otherwise, we return false. So clearly, this is an O(1) operation every time. Let's look at the more complicated function now to see if we satisfy all the requirements in the problem statement
o	next involves two major operations. One is where we pop an element from the stack which becomes the next smallest element to return. This is a O(1) operation. However, we then make a call to our helper function _inorder_left which iterates over a bunch of nodes. This is clearly a linear time operation i.e. O(N) in the worst case. This is true.
However, the important thing to note here is that we only make such a call for nodes which have a right child. Otherwise, we simply return. Also, even if we end up calling the helper function, it won't always process N nodes. They will be much lesser. Only if we have a skewed tree would there be N nodes for the root. But that is the only node for which we would call the helper function.
Thus, the amortized (average) time complexity for this function would still be O(1) which is what the question asks for. We don't need to have a solution which gives constant time operations for every call. We need that complexity on average and that is what we get.
•	Space complexity: The space complexity is O(N) (N is the number of nodes in the tree), which is occupied by our custom stack for simulating the inorder traversal. Again, we satisfy the space requirements as well as specified in the problem statement.

             */
            class ControlledRecursionSol
            {
                Stack<TreeNode> stack;

                public ControlledRecursionSol(TreeNode root)
                {
                    // Stack for the recursion simulation
                    this.stack = new Stack<TreeNode>();

                    // Remember that the algorithm starts with a call to the helper function
                    // with the root node as the input
                    this.LeftmostInorder(root);
                }

                private void LeftmostInorder(TreeNode root)
                {
                    // For a given node, add all the elements in the leftmost branch of the tree
                    // under it to the stack.
                    while (root != null)
                    {
                        this.stack.Push(root);
                        root = root.Left;
                    }
                }

                /**
                 * @return the next smallest number
                 */
                public int Next()
                {
                    // Node at the top of the stack is the next smallest element
                    TreeNode topmostNode = this.stack.Pop();

                    // Need to maintain the invariant. If the node has a right child, call the
                    // helper function for the right child
                    if (topmostNode.Right != null)
                    {
                        this.LeftmostInorder(topmostNode.Right);
                    }

                    return topmostNode.Val;
                }

                /**
                 * @return whether we have a next smallest number
                 */
                public bool HasNext()
                {
                    return this.stack.Count > 0;
                }
            }
        }





    }
}