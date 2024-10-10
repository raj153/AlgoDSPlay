using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class NAryNode
    {
        public int Val;
        public IList<NAryNode> Children;

        public NAryNode() { }

        public NAryNode(int _val)
        {
            Val = _val;
        }

        public NAryNode(int _val, IList<NAryNode> _children)
        {
            Val = _val;
            Children = _children;
        }
    }
}