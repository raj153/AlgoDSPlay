using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    297. Serialize and Deserialize Binary Tree
    https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/

    
    Complexity Analysis

    Time complexity: in both serialization and deserialization functions, we visit each node exactly once, thus the time complexity is O(N), where N is the number of nodes, i.e. the size of the tree.

    Space complexity: in both serialization and deserialization functions, we keep the entire tree, either at the beginning or at the end, therefore, the space complexity is O(N).

    The solutions with BFS or other DFS strategies normally will have the same time and space complexity.


    */
    public class BInaryTreeCodec
    {
        //Approach 1: Depth First Search (DFS)
        public class BInaryTreeCodecDfs
        {
            public string RSerialize(TreeNode root, string str)
            {
                // Recursive serialization.
                if (root == null)
                {
                    str += "null,";
                }
                else
                {
                    str += root.val.ToString() + ",";
                    str = RSerialize(root.left, str);
                    str = RSerialize(root.right, str);
                }
                return str;
            }

            // Encodes a tree to a single string.
            public string Serialize(TreeNode root)
            {
                return RSerialize(root, "");
            }

            public TreeNode RDeserialize(List<string> nodeList)
            {
                // Recursive deserialization.
                if (nodeList[0].Equals("null"))
                {
                    nodeList.RemoveAt(0);
                    return null;
                }

                TreeNode root = new TreeNode(int.Parse(nodeList[0]));
                nodeList.RemoveAt(0);
                root.left = RDeserialize(nodeList);
                root.right = RDeserialize(nodeList);

                return root;
            }

            // Decodes your encoded data to tree.
            public TreeNode Deserialize(string data)
            {
                string[] dataArray = data.Split(',');
                List<string> dataList = new List<string>(dataArray);
                return RDeserialize(dataList);
            }
        }

        ////Approach 2: Breadth First Search (BFS)
        public class BInaryTreeCodecBfs
        {

            // Encodes a tree to a single string.
            public string serialize(TreeNode root)
            {
                if (root == null) return "null";
                return root.val + " " + serialize(root.left) + " " + serialize(root.right);
            }

            // Decodes your encoded data to tree.
            public TreeNode deserialize(string data)
            {
                List<TreeNode> list = new List<TreeNode>();

                if (data == "null") return null;

                string[] words = data.Split(' ');
                TreeNode root = new TreeNode(Convert.ToInt32(words[0]));
                list.Add(root);

                bool goLeft = true;
                for (int i = 1; i < words.Count(); ++i)
                {
                    if (words[i] == "null")
                    {
                        if (goLeft) goLeft = false;
                        else list.RemoveAt(list.Count() - 1);
                    }
                    else
                    {
                        TreeNode node = new TreeNode(Convert.ToInt32(words[i]));
                        if (goLeft)
                        {
                            list[list.Count() - 1].left = node;
                        }
                        else
                        {
                            list[list.Count() - 1].right = node;
                            list.RemoveAt(list.Count() - 1);
                        }
                        list.Add(node);
                        goLeft = true;
                    }
                }

                return root;
            }
        }


        public class TreeNode
        {
            public int val;
            public TreeNode left;
            public TreeNode right;

            public TreeNode(int x)
            {
                val = x;
            }
        }

    }
}