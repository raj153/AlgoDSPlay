using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    //https://www.algoexpert.io/questions/union-find
    //https://algodaily.com/lessons/what-to-know-about-the-union-find-algorithm
    public class UnionFind
    {
        private Dictionary<int, int> parents = new Dictionary<int, int>();
        private Dictionary<int, int> ranks = new Dictionary<int, int>();
        public void CreateSet(int value){
            parents[value] = value;
            ranks[value] =0;
        }

        /*
        //T:O(n) | S:O(1)
        public int? Find(int value){
            if(!parents.ContainsKey(value))
                return null;
            
            int currentParent=value;
            while(currentParent != parents[currentParent]){
                currentParent = parents[currentParent];
            }
            return currentParent;
        }
        
         
        //T:O(log(n)) | S:O(1)
        //Union uses Ranking to keep childrens one stpe reahble to parents
        public int? Find(int value){
            if(!parents.ContainsKey(value))
                return null;
            
            int currentParent=value;
            while(currentParent != parents[currentParent]){
                currentParent = parents[currentParent];
            }
            return currentParent;
        }
        */
        //T:O(alpha(n)) | S:O(1)
        //O(alpha(n)), approximately O(1)
        //Union uses Ranking to attach children to a right parent to keep tree balanced
        //Furtherly optimized to attach a children directly to root parent aka Path Compression
        public int? Find(int value){
            if(!parents.ContainsKey(value))
                return null;
            
            if(value != parents[value]){
                value = (int)Find(parents[value]); //Path Compression
            }
            return parents[value];
        }
        /*
        //T:O(n) | S:O(1)        
        public void Union(int valueOne, int valueTwo){
            if(!parents.ContainsKey(valueTwo) || !parents.ContainsKey(valueTwo))
                return;
            
            int valueOneRoot = (int)Find(valueOne);
            int valueTwoRoot = (int)Find(valueTwo);
            parents[valueTwoRoot] = valueOneRoot;
        }
 
        //T:O(log(n)) | S:O(1)
        //Leaveraging Ranks
        public void Union(int valueOne, int valueTwo){
            if(!parents.ContainsKey(valueTwo) || !parents.ContainsKey(valueTwo))
                return;
            
            int valueOneRoot = (int)Find(valueOne);
            int valueTwoRoot = (int)Find(valueTwo);

            if(ranks[valueOneRoot] < ranks[valueTwoRoot])
                parents[valueOneRoot] = valueTwoRoot;
            else if(ranks[valueOneRoot] > ranks[valueTwoRoot])
                parents[valueTwoRoot] = valueOneRoot;
            else{
                parents[valueTwoRoot] = valueOneRoot;
                ranks[valueOneRoot] = ranks[valueOneRoot]+1;
            }
            parents[valueTwoRoot] = valueOneRoot;
        }
        */
        //T:O(alpha(n)), approximately O(1) | S:O(alpha(n)), approximately O(1)
        //Leaveraging Ranks
        public void Union(int valueOne, int valueTwo){
            if(!parents.ContainsKey(valueTwo) || !parents.ContainsKey(valueTwo))
                return;
            
            int valueOneRoot = (int)Find(valueOne);
            int valueTwoRoot = (int)Find(valueTwo);

            if(ranks[valueOneRoot] < ranks[valueTwoRoot])
                parents[valueOneRoot] = valueTwoRoot;
            else if(ranks[valueOneRoot] > ranks[valueTwoRoot])
                parents[valueTwoRoot] = valueOneRoot;
            else{
                parents[valueTwoRoot] = valueOneRoot;
                ranks[valueOneRoot] = ranks[valueOneRoot]+1;
            }
            parents[valueTwoRoot] = valueOneRoot;
        }
 
 
    }
    
  
}