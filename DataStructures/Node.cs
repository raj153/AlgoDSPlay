using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
        public class Node
        {
            public string name;
            public List<Node> children = new List<Node>();

            public Node(string name)
            {
                this.name = name;
            }

            public Node AddChild(string name)
            {
                Node child = new Node(name);
                children.Add(child);
                return this;
            }

        }
        

    
}