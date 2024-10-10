using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class NAryTreeProbs
    {

        /*         428. Serialize and Deserialize N-ary Tree
        https://leetcode.com/problems/serialize-and-deserialize-n-ary-tree/description/
         */
        public class CodecSol
        {
            /* Approach 1: Parent Child relationships
            Complexity Analysis
            Time Complexity
            •	Serialization: O(N) where N are the number of nodes in the tree. For every node, we add 3 different values to the final string and every node is processed exactly once.
            •	Deserialization: Well technically, it is 3N for the first for loop and N for the second one. However, constants are ignored in asymptotic complexity analysis. So, the overall time complexity for deserialization is O(N).
            Space Complexity
            •	Serialization: The space occupied by the serialization helper function is through recursion stack and the final string that is produced. Usually, we don't take into consideration the space of the output. However, in this case, the output is something which is not fixed. For all we know, someone might be able to generate a string of size N/2. We don't know! So, the size of the final string is a part of the space complexity here. Overall, the space is 4N = O(N).
            •	Deserialization: The space occupied by the deserialization helper function is through the hash map. For each entry, we have 3 values. Thus, we can say the space is 3N. But again, the constants don't really matter in asymptotic complexity. So, the overall space is O(N).
            Note that for this particular problem, the asymptotic time and space will remain the same across all the approaches. The only thing that will change are the constants and that does impact the runtime in a major way. So, we will be focusing on the constants rather than the final complexity in all these approaches.

             */

            class ParentChildRelationshipsCodec
            {
                class WrappableInt
                {
                    private int value;
                    public WrappableInt(int x)
                    {
                        this.value = x;
                    }
                    public int GetValue()
                    {
                        return this.value;
                    }
                    public void Increment()
                    {
                        this.value++;
                    }
                }

                // Represents a map of deserialized objects where the key is an integer and the value is a pair of integers and a node
                class DeserializedObject : Dictionary<int, KeyValuePair<int, KeyValuePair<int, NAryNode>>> { }

                // Encodes a tree to a single string.
                public string Serialize(NAryNode root)
                {
                    StringBuilder stringBuilder = new StringBuilder();
                    this.SerializeHelper(root, stringBuilder, new WrappableInt(1), null);
                    return stringBuilder.ToString();
                }

                private void SerializeHelper(NAryNode root, StringBuilder stringBuilder, WrappableInt identity, int? parentId)
                {
                    if (root == null)
                    {
                        return;
                    }

                    // Own identity
                    stringBuilder.Append((char)(identity.GetValue() + '0'));

                    // Actual value
                    stringBuilder.Append((char)(root.Val + '0'));

                    // Parent's identity
                    stringBuilder.Append((char)(parentId == null ? 'N' : parentId + '0'));

                    parentId = identity.GetValue();
                    foreach (NAryNode child in root.Children)
                    {
                        identity.Increment();
                        this.SerializeHelper(child, stringBuilder, identity, parentId);
                    }
                }

                // Decodes your encoded data to tree.
                public NAryNode Deserialize(string data)
                {
                    if (string.IsNullOrEmpty(data))
                        return null;

                    return this.DeserializeHelper(data);
                }

                private NAryNode DeserializeHelper(string data)
                {
                    // HashMap explained in the algorithm
                    DeserializedObject nodesAndParents = new DeserializedObject();

                    // Constructing the hashmap using the input string
                    for (int i = 0; i < data.Length; i += 3)
                    {
                        int id = data[i] - '0';
                        int originalValue = data[i + 1] - '0';
                        int parentId = data[i + 2] - '0';
                        var node = new KeyValuePair<int, KeyValuePair<int, NAryNode>>(originalValue,
                            new KeyValuePair<int, NAryNode>(parentId,
                            new NAryNode(originalValue, new List<NAryNode>())));
                        nodesAndParents[id] = node;
                    }

                    // A second pass for tying up the proper child connections
                    for (int i = 3; i < data.Length; i += 3)
                    {
                        // Current node
                        int id = data[i] - '0';
                        NAryNode node = nodesAndParents[id].Value.Value;

                        // Parent node
                        int parentId = data[i + 2] - '0';
                        NAryNode parentNode = nodesAndParents[parentId].Value.Value;

                        // Attach!
                        parentNode.Children.Add(node);
                    }

                    // Return the root node.
                    return nodesAndParents[data[0] - '0'].Value.Value;
                }
            }


            /* Approach 2: Depth First Search with Children Sizes!
            Complexity Analysis
            Time Complexity
            •	Serialization: O(N) where N are the number of nodes in the tree. For every node, we add 2 different values to the final string and every node is processed exactly once.
            •	Deserialization: For deserialization, we process the entire string, one character at a time and also construct the tree along the way. So, the overall time complexity for deserialization is 2N = O(N)
            Space Complexity
            •	Serialization: The space occupied by the serialization helper function is through recursion stack and the final string that is produced. We know the size of the final string to be 2N. So, that is one part of the space complexity. The other part is the one occupied by the recursion stack which is O(N). Overall, the space is O(N).
            •	Deserialization: For deserialization, the space occupied is by the recursion stack only. We don't use any other intermediate data structures like we did in the previous approach and simply rely on the information in the string and recursion to work it's magic. So, the space complexity would be O(N) since this is not a balanced tree of any sort. It's not even binary.
            This is one of the simplest algorithms for solving this problem. The serialization and deserialization have a very similar format and the overall space and time complexity are also very low. Also, what's nice is that it's easy to code up quickly in an interview!

             */
            class DFSWithChildrenSizesCodec
            {
                class WrappableInt
                {
                    private int value;
                    public WrappableInt(int x)
                    {
                        this.value = x;
                    }
                    public int GetValue()
                    {
                        return this.value;
                    }
                    public void Increment()
                    {
                        this.value++;
                    }
                }

                // Encodes a tree to a single string.
                public string Serialize(NAryNode root)
                {
                    System.Text.StringBuilder stringBuilder = new System.Text.StringBuilder();
                    this._SerializeHelper(root, stringBuilder);
                    return stringBuilder.ToString();
                }

                private void _SerializeHelper(NAryNode root, System.Text.StringBuilder stringBuilder)
                {
                    if (root == null)
                    {
                        return;
                    }

                    // Add the value of the node
                    stringBuilder.Append((char)(root.Val + '0'));

                    // Add the number of children
                    stringBuilder.Append((char)(root.Children.Count + '0'));

                    // Recurse on the subtrees and build the 
                    // string accordingly
                    foreach (NAryNode child in root.Children)
                    {
                        this._SerializeHelper(child, stringBuilder);
                    }
                }

                // Decodes your encoded data to tree.
                public NAryNode Deserialize(string data)
                {
                    if (string.IsNullOrEmpty(data))
                        return null;

                    return this._DeserializeHelper(data, new WrappableInt(0));
                }

                private NAryNode _DeserializeHelper(string data, WrappableInt index)
                {
                    if (index.GetValue() == data.Length)
                    {
                        return null;
                    }

                    // The invariant here is that the "index" always
                    // points to a node and the value next to it 
                    // represents the number of children it has.
                    NAryNode node = new NAryNode(data[index.GetValue()] - '0', new List<NAryNode>());
                    index.Increment();
                    int numChildren = data[index.GetValue()] - '0';
                    for (int i = 0; i < numChildren; i++)
                    {
                        index.Increment();
                        node.Children.Add(this._DeserializeHelper(data, index));
                    }

                    return node;
                }
                /* Approach 3: Depth First Search with a Sentinel
                Complexity Analysis
                Time Complexity
                •	Serialization: O(N) where N are the number of nodes in the tree. For every node, we add 2 different values to the final string and every node is processed exactly once.
                •	Deserialization: For deserialization, we process the entire string, one character at a time and also construct the tree along the way. So, the overall time complexity for deserialization is 2N = O(N)
                Space Complexity
                •	Serialization: The space occupied by the serialization helper function is through recursion stack and the final string that is produced. We know the size of the final string to be 2N. So, that is one part of the space complexity. The other part is the one occupied by the recursion stack which is O(N). Overall, the space is O(N).
                •	Deserialization: For deserialization, the space occupied is by the recursion stack only. We don't use any other intermediate data structures like we did in the previous approach and simply rely on the information in the string and recursion to work it's magic. So, the overall space complexity would be O(N).

                 */
                class DFSWithASentinelCodec
                {

                    class WrappableInt
                    {
                        private int value;
                        public WrappableInt(int x)
                        {
                            this.value = x;
                        }
                        public int GetValue()
                        {
                            return this.value;
                        }
                        public void Increment()
                        {
                            this.value++;
                        }
                    }

                    // Encodes a tree to a single string.
                    public string Serialize(NAryNode root)
                    {
                        System.Text.StringBuilder stringBuilder = new System.Text.StringBuilder();
                        this.SerializeHelper(root, stringBuilder);
                        return stringBuilder.ToString();
                    }

                    private void SerializeHelper(NAryNode root, System.Text.StringBuilder stringBuilder)
                    {
                        if (root == null)
                        {
                            return;
                        }

                        // Add the value of the node
                        stringBuilder.Append((char)(root.Val + '0'));

                        // Recurse on the subtrees and build the 
                        // string accordingly
                        foreach (NAryNode child in root.Children)
                        {
                            this.SerializeHelper(child, stringBuilder);
                        }

                        // Add the sentinel to indicate that all the children
                        // for the current node have been processed
                        stringBuilder.Append('#');
                    }

                    // Decodes your encoded data to tree.
                    public NAryNode Deserialize(string data)
                    {
                        if (string.IsNullOrEmpty(data))
                            return null;

                        return this.DeserializeHelper(data, new WrappableInt(0));
                    }

                    private NAryNode DeserializeHelper(string data, WrappableInt index)
                    {
                        if (index.GetValue() == data.Length)
                        {
                            return null;
                        }

                        NAryNode node = new NAryNode(data[index.GetValue()] - '0', new List<NAryNode>());
                        index.Increment();
                        while (data[index.GetValue()] != '#')
                        {
                            node.Children.Add(this.DeserializeHelper(data, index));
                        }

                        // Discard the sentinel. Note that this also moves us
                        // forward in the input string. So, we don't have the index
                        // progressing inside the above while loop!
                        index.Increment();

                        return node;
                    }


                    /* Approach 4: Level order traversal
                    Complexity Analysis
                    Time Complexity
                    •	Serialization: O(N) where N are the number of nodes in the tree. For every node, we add 2 different values to the final string and every node is processed exactly once. We add the value of the node itself and we also add the child switch sentinel. Also, for the nodes that end a particular level, we add the level end sentinel.
                    •	Deserialization: For deserialization, we process the entire string, one character at a time and also construct the tree along the way. So, the overall time complexity for deserialization is 2N = O(N)
                    Space Complexity
                    •	Serialization: The space occupied by the serialization helper function is through the queue and the final string that is produced. We know the size of the final string to be 2N. So that is one part of the space complexity. The other part is the one occupied by the queue which is O(N). Overall, the space is O(N).
                    •	Deserialization: For deserialization, the space is mostly occupied by the two lists that we use. The space complexity there is O(N). Note that when we re-initialize a list, the memory that was allocated earlier is deallocated by the garbage collector and it's essentially equal to a single list of size O(N).

                     */
                    public class LevelOrderTraversalCodec
                    {
                        // Encodes a tree to a single string.
                        public string Serialize(NAryNode root)
                        {
                            if (root == null)
                            {
                                return "";
                            }

                            StringBuilder stringBuilder = new StringBuilder();
                            SerializeHelper(root, stringBuilder);
                            return stringBuilder.ToString();
                        }

                        private void SerializeHelper(NAryNode root, StringBuilder stringBuilder)
                        {
                            // Queue to perform a level order traversal of the tree
                            Queue<NAryNode> queue = new Queue<NAryNode>();

                            // Two dummy nodes that will help us in serialization string formation.
                            // We insert the "endNode" whenever a level ends and the "childNode"
                            // whenever a node's children are added to the queue and we are about
                            // to switch over to the next node.
                            NAryNode endNode = new NAryNode();
                            NAryNode childNode = new NAryNode();
                            queue.Enqueue(root);
                            queue.Enqueue(endNode);

                            while (queue.Count > 0)
                            {
                                // Pop a node
                                NAryNode node = queue.Dequeue();

                                // If this is an "endNode", we need to add another one
                                // to mark the end of the current level unless this
                                // was the last level.
                                if (node == endNode)
                                {
                                    // We add a sentinel value of "#" here
                                    stringBuilder.Append('#');
                                    if (queue.Count > 0)
                                    {
                                        queue.Enqueue(endNode);
                                    }
                                }
                                else if (node == childNode)
                                {
                                    // Add a sentinel value of "$" here to mark the switch to a
                                    // different parent.
                                    stringBuilder.Append('$');
                                }
                                else
                                {
                                    // Add value of the current node and add all of its
                                    // children nodes to the queue. Note how we convert
                                    // the integers to their corresponding ASCII counterparts.
                                    stringBuilder.Append((char)(node.Val + '0'));
                                    foreach (NAryNode child in node.Children)
                                    {
                                        queue.Enqueue(child);
                                    }

                                    // If this node is NOT the last one on the current level, 
                                    // add a childNode as well since we move on to processing
                                    // the next node.
                                    if (queue.Peek() != endNode)
                                    {
                                        queue.Enqueue(childNode);
                                    }
                                }
                            }
                        }

                        // Decodes your encoded data to tree.
                        public NAryNode Deserialize(string data)
                        {
                            if (data.Length == 0)
                            {
                                return null;
                            }

                            NAryNode rootNode = new NAryNode(data[0] - '0', new List<NAryNode>());
                            DeserializeHelper(data, rootNode);
                            return rootNode;
                        }

                        private void DeserializeHelper(string data, NAryNode rootNode)
                        {
                            // We move one level at a time and at every level, we need access
                            // to the nodes on the previous level as well so that we can form
                            // the children arrays properly. Hence two lists.
                            LinkedList<NAryNode> currentLevel = new LinkedList<NAryNode>();
                            LinkedList<NAryNode> prevLevel = new LinkedList<NAryNode>();
                            currentLevel.AddLast(rootNode);
                            NAryNode parentNode = rootNode;

                            // Process the characters in the string one at a time.
                            for (int i = 1; i < data.Length; i++)
                            {
                                char d = data[i];
                                if (d == '#')
                                {
                                    // Special processing for end of level. We need to swap the
                                    // lists. Here, we simply re-initialize the "currentLevel"
                                    // list rather than clearing it.
                                    prevLevel = currentLevel;
                                    currentLevel = new LinkedList<NAryNode>();

                                    // Since we move one level down, we take the parent as the first
                                    // node on the current level.
                                    parentNode = prevLevel.First.Value;
                                    prevLevel.RemoveFirst();
                                }
                                else
                                {
                                    if (d == '$')
                                    {
                                        // Special handling for change in parent on the same level
                                        parentNode = prevLevel.First.Value;
                                        prevLevel.RemoveFirst();
                                    }
                                    else
                                    {
                                        NAryNode childNode = new NAryNode(d - '0', new List<NAryNode>());
                                        currentLevel.AddLast(childNode);
                                        parentNode.Children.Add(childNode);
                                    }
                                }
                            }
                        }
                    }

                }




























































































            }
        }

        /*     431. Encode N-ary Tree to Binary Tree
        https://leetcode.com/problems/encode-n-ary-tree-to-binary-tree/description/
         */
        public class NAryToBinaryTreeCodecSol
        {
            /*             Approach 1: BFS (Breadth-First Search) Traversal 
            Complexity Analysis
            •	Time Complexity: O(N) where N is the number of nodes in the N-ary tree. We traverse each node in the tree once and only once.
            •	Space Complexity: O(L) where L is the maximum number of nodes that reside at the same level.
            Since L is proportional to N in the worst case, we could further generalize the time complexity to O(N).
            o	We use a queue data structure to do BFS traversal, i.e. visiting the nodes level by level.
            o	At any given moment, the queue contains nodes that are at most spread into two levels. As a result, assuming the maximum number of nodes at one level is L, the size of the queue would be less than 2L at any time.
            o	Therefore, the space complexity of both encode() and decode() functions is O(L).

            */

            class UsingBFSCodec
            {
                // Encodes an n-ary tree to a binary tree.
                public TreeNode Encode(TreeNode root)
                {
                    if (root == null)
                    {
                        return null;
                    }
                    TreeNode newRoot = new TreeNode(root.Val);
                    Tuple<TreeNode, TreeNode> head = new Tuple<TreeNode, TreeNode>(newRoot, root);

                    // Add the first element to kickoff the loop
                    Queue<Tuple<TreeNode, TreeNode>> queue = new Queue<Tuple<TreeNode, TreeNode>>();
                    queue.Enqueue(head);

                    while (queue.Count > 0)
                    {
                        Tuple<TreeNode, TreeNode> pair = queue.Dequeue();
                        TreeNode binaryNode = pair.Item1;
                        TreeNode naryNode = pair.Item2;

                        // Encoding the children nodes into a list of TreeNode.
                        TreeNode previousBinaryNode = null, headBinaryNode = null;
                        foreach (TreeNode nChild in naryNode.Children)
                        {
                            TreeNode newBinaryNode = new TreeNode(nChild.Val);
                            if (previousBinaryNode == null)
                            {
                                headBinaryNode = newBinaryNode;
                            }
                            else
                            {
                                previousBinaryNode.Right = newBinaryNode;
                            }
                            previousBinaryNode = newBinaryNode;

                            Tuple<TreeNode, TreeNode> nextEntry = new Tuple<TreeNode, TreeNode>(newBinaryNode, nChild);
                            queue.Enqueue(nextEntry);
                        }

                        // Attach the list of children to the left node.
                        binaryNode.Left = headBinaryNode;
                    }

                    return newRoot;
                }

                // Decodes your binary tree to an n-ary tree.
                public TreeNode Decode(TreeNode root)
                {
                    if (root == null)
                    {
                        return null;
                    }
                    TreeNode newRoot = new TreeNode(root.Val, new List<TreeNode>());

                    // adding the first element to kickoff the loop
                    Queue<Tuple<TreeNode, TreeNode>> queue = new Queue<Tuple<TreeNode, TreeNode>>();
                    Tuple<TreeNode, TreeNode> head = new Tuple<TreeNode, TreeNode>(newRoot, root);
                    queue.Enqueue(head);

                    while (queue.Count > 0)
                    {
                        Tuple<TreeNode, TreeNode> entry = queue.Dequeue();
                        TreeNode nNode = entry.Item1;
                        TreeNode binaryNode = entry.Item2;

                        // Decoding the children list
                        TreeNode firstChild = binaryNode.Left;
                        TreeNode sibling = firstChild;
                        while (sibling != null)
                        {
                            TreeNode nChild = new TreeNode(sibling.Val, new List<TreeNode>());
                            nNode.Children.Add(nChild);

                            // prepare the decoding the children of the child, by standing in the queue.
                            Tuple<TreeNode, TreeNode> nextEntry = new Tuple<TreeNode, TreeNode>(nChild, sibling);
                            queue.Enqueue(nextEntry);

                            sibling = sibling.Right;
                        }
                    }

                    return newRoot;
                }
            }
            /* Approach 2: DFS (Depth-First Search) Traversal 
            Complexity Analysis
•	Time Complexity: O(N) where N is the number of nodes in the N-ary tree. We traverse each node in the tree once and only once.
•	Space Complexity: O(D) where D is the depth of the N-ary tree.
Since D is proportional to N in the worst case, we could further generalize the time complexity to O(N).
o	Unlike the BFS algorithm, we don't use the queue data structure in the DFS algorithm. However, implicitly the algorithm would consume more space in the function call stack due to the recursive function calls.
o	And this consumption of call stack space is the main space complexity for our DFS algorithm. As we can see, the size of the call stack at any moment is exactly the number of level where the currently visited node resides, e.g. for the root node (level 0), the recursive call stack is empty.

            */
            public class UsingDFSCodec
            {

                // Encodes an n-ary tree to a binary tree.
                public TreeNode Encode(TreeNode root)
                {
                    if (root == null)
                    {
                        return null;
                    }

                    TreeNode newRoot = new TreeNode(root.Val);

                    // Encode the first child of n-ary node to the left node of binary tree.
                    if (root.Children.Count > 0)
                    {
                        TreeNode firstChild = root.Children[0];
                        newRoot.Left = this.Encode(firstChild);
                    }

                    // Encoding the rest of the sibling nodes.
                    TreeNode sibling = newRoot.Left;
                    for (int i = 1; i < root.Children.Count; ++i)
                    {
                        sibling.Right = this.Encode(root.Children[i]);
                        sibling = sibling.Right;
                    }

                    return newRoot;
                }

                // Decodes your binary tree to an n-ary tree.
                public TreeNode Decode(TreeNode root)
                {
                    if (root == null)
                    {
                        return null;
                    }

                    TreeNode newRoot = new TreeNode(root.Val, new List<TreeNode>());

                    // Decoding all the children nodes
                    TreeNode sibling = root.Left;
                    while (sibling != null)
                    {
                        newRoot.Children.Add(this.Decode(sibling));
                        sibling = sibling.Right;
                    }

                    return newRoot;
                }
            }

        }







































    }
}