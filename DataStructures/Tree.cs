using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class Tree
    {
        public int Value{get;set;}
        public Tree? Left;
        public Tree? Right;

        public Tree(int value){
            this.Value = value;
            this.Left=null;
            this.Right =null;
        }
        
        public void InsertBST(int value){
            if(this.Value < value){
                if(Right== null)
                    Right = new Tree(value);
                else 
                    Right.InsertBST(value);
            }else {
                if(Left == null)
                    Left = new Tree(value);
                else 
                    Left.InsertBST(value);
            }

        }
    }
 	
}