using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class BST
    {
        
        public int Value {get;set;}
        public BST Left {get;set;}

        public BST Right {get;set;}

        public BST(int value){
            this.Value = value;
        }

        //https://www.algoexpert.io/questions/bst-traversal
        public static List<int> InOrderTraverse(BST tree, List<int> array){
            //T:O(n)| S:O(n)
            if(tree.Left != null){
                InOrderTraverse(tree.Left, array);
            }
            array.Add(tree.Value);
            if(tree.Right != null){
                InOrderTraverse(tree.Right, array);
            }
            return array;
        }

        public static List<int> PreOrderTraverse(BST tree, List<int> array){
            //T:O(n)| S:O(n)
            array.Add(tree.Value);
            if(tree.Left != null){
                InOrderTraverse(tree.Left, array);
            }            
            if(tree.Right != null){
                InOrderTraverse(tree.Right, array);
            }
            return array;
        }

        public static List<int> PostOrderTraverse(BST tree, List<int> array){
            //T:O(n)| S:O(n)
            
            if(tree.Left != null){
                InOrderTraverse(tree.Left, array);
            }            
            if(tree.Right != null){
                InOrderTraverse(tree.Right, array);
            }
            array.Add(tree.Value);
            return array;
        }

    }
    
}