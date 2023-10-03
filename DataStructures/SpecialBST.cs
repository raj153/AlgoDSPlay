using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Threading.Tasks;
using System.Xml.Schema;

namespace AlgoDSPlay.DataStructures
{
    public class SpecialBST
    {
        public int Value {get;set;}
        public SpecialBST Left {get;set;}

        public SpecialBST Right {get;set;}

        public int LeftSubtreeSize {get;set;}

        public SpecialBST(int value){
            Value = value;
            Left = null;
            Right = null;
        }

        public void Insert(int value, int idx, List<int> rightSmallerCounts){
            InsertHelper(value, idx, rightSmallerCounts, 0);
            
        }

        private void InsertHelper(int value, int idx, List<int> rightSmallerCounts, int numSmallerAtInsertTime)
        {
            if(value < this.Value){ //Left Subtree- BST            
                LeftSubtreeSize++;
                if(Left == null){
                    Left = new SpecialBST(value);
                    rightSmallerCounts[idx] = numSmallerAtInsertTime;   
                }else{
                    Left.InsertHelper(value, idx, rightSmallerCounts, numSmallerAtInsertTime);
                }
            }
            else
            {
                numSmallerAtInsertTime +=LeftSubtreeSize;
                if(value > this.Value) numSmallerAtInsertTime++;
                if(Right == null){
                    Right = new SpecialBST(value);
                    rightSmallerCounts[idx] = numSmallerAtInsertTime;
                }else{
                    Right.InsertHelper(value, idx, rightSmallerCounts, numSmallerAtInsertTime);
                }
            }
            
        }           
        
    }
}