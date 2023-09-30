using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class StringOps
    {


        //https://www.algoexpert.io/questions/multi-string-search
        public static List<bool> MultiStringSearch(string bigString, string[] smallStrings){
            List<bool> solution = new List<bool>();
            //1.Naive - T:O(bns) | S:O(n)
            foreach(string smallString in smallStrings){
                solution.Add(isInBigString(bigString, smallString));

            }
            //2.ModifiedSuffixTrie - T:O(b^2+ns) | S:O(b^2+n)
            ModifiedSuffixTrie modifiedSuffixTrie = new ModifiedSuffixTrie(bigString);
            solution.Clear();
            foreach(string smallString in smallStrings){
                solution.Add(modifiedSuffixTrie.Contains(smallString));
            }

            solution.Clear();            
            //3.Trie - T:O(ns+bs) | S:O(ns)
            Trie trie = new Trie();
            foreach(string smallString in smallStrings){
                trie.Insert(smallString);
            }
            HashSet<string> containedStrings = new HashSet<string>();
            for(int i=0; i<bigString.Length; ++i){
                FindSmallStringsIn(bigString, i, trie, containedStrings);
            }
            foreach(string smallString in smallStrings){
                solution.Add(containedStrings.Contains(smallString));
            }
            return solution;

        }

        private static void FindSmallStringsIn(string bigString, int startIdx, Trie trie, HashSet<string> containedStrings)
        {
            TrieNode currentNode = trie.root;
            for(int i=startIdx; i<bigString.Length; ++i){
                char currentChar = bigString[i];
                if(!currentNode.children.ContainsKey(currentChar)){
                    break;
                }
                currentNode = currentNode.children[currentChar];
                if(currentNode.children.ContainsKey(trie.endSymbol)){
                    containedStrings.Add(currentNode.word);
                }
            }

            
        }

        private static bool isInBigString(string bigString, string smallString)
        {
            for(int i=0; i< bigString.Length; ++i){
                if(i+ smallString.Length > bigString.Length){
                    break;
                }
                if(isInBigString(bigString, smallString, i)){
                    return true;
                }
            }
            return false;
        }

        private static bool isInBigString (string bigString, string smallString, int startIdx)
        {
            //big bigger
            //egg
            int leftBigIndex= startIdx;
            int rightBigIndex= startIdx+ smallString.Length-1;
            int leftSmallIndex = 0;
            int rightSmallIndex=smallString.Length-1;
            while(leftBigIndex <= rightBigIndex){

                if(bigString[leftBigIndex] != smallString[leftSmallIndex]
                    || bigString[rightBigIndex] != smallString[rightSmallIndex])
                    return false;
                
                leftBigIndex++;
                rightSmallIndex--;
                leftSmallIndex++;
                rightBigIndex--;
            } 
            return true;
 
      }
    }
}