using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class SuffixTrie
    {
        //https://www.algoexpert.io/questions/suffix-trie-construction
        public TrieNode root = new TrieNode();
        
        public char endSymbol ='*';
        public SuffixTrie(string str){
            PopulateSuffixTreeFrom(str);
            
        }

        //T:O(n^2) | S:O(n^2)
        private void PopulateSuffixTreeFrom(string str)
        {
            for(int i=0; i<str.Length; i++){
                InsertSubstringStartingAt(i, str);
            }
        }

        private void InsertSubstringStartingAt(int i, string str)
        {
            TrieNode node = root;
            for(int j=i; j<str.Length; j++){
                char letter = str[j];
                if(!node.Children.ContainsKey(letter)){
                    TrieNode trieNode = new TrieNode();                    
                    node.Children.Add(letter, trieNode);
                }
                node = node.Children[letter];

            }
            node.Children[endSymbol] = null;
        }

        //T:O(m) | O(1) where m is length of input string
        public bool Contains(string str){
            TrieNode node =root;
            foreach(char c in str){
                if(!node.Children.ContainsKey(c)) return false;
                node = node.Children[c];
            }
            return node.Children.ContainsKey(endSymbol);
        }
        
    }
}