using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class TreeNode
    {
        public int Val { get; set; }
        public TreeNode? Left;
        public TreeNode? Right;
        public string Name;
        public List<TreeNode> Children = new List<TreeNode>();
        public TreeNode(int value, TreeNode left = null, TreeNode right = null)
        {
            this.Val = value;
            this.Left = left;
            this.Right = right;
        }
        public TreeNode(int _val, List<TreeNode> _children)
        {
            Val = _val;
            Children = _children;
        }
        public TreeNode(string name)
        {
            this.Name = name;
        }

        public TreeNode AddChild(string name)
        {
            TreeNode child = new TreeNode(name);
            Children.Add(child);
            return this;
        }

    }
}