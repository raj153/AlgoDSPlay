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













    }
}