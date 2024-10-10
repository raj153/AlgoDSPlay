using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class Node
    {
        public string name;
        public int val;
        public List<Node> children = new List<Node>();
        internal Node next;
        private Node curr;

        public Node(string name)
        {
            this.name = name;
        }

        public Node(int _val)
        {
            val = _val;
        }

        public Node(int _val, List<Node> _children)
        {
            val = _val;
            children = _children;
        }

        public Node(int _val, Node curr) : this(_val)
        {
            this.curr = curr;
        }

        public Node AddChild(string name)
        {
            Node child = new Node(name);
            children.Add(child);
            return this;
        }
    }
}
