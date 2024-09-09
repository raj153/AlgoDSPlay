using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
     public class TreeNode
    {
        public int Val{get;set;}
        public TreeNode? Left;
        public TreeNode? Right;

        public TreeNode(int value, TreeNode left=null, TreeNode right=null){
            this.Val = value;
            this.Left=left;
            this.Right =right;
        }
        
    }
}