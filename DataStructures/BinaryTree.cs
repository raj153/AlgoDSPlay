using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
     public class BinaryTree
    {
        public int Value{get;set;}
        public BinaryTree? Left;
        public BinaryTree? Right;

        public BinaryTree(int value){
            this.Value = value;
            this.Left=null;
            this.Right =null;
        }
    }
}