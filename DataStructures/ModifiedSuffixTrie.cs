using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    
    public class ModifiedSuffixTrie
    {
        class TrieNode {
            public Dictionary<char, TrieNode> children = new Dictionary<char, TrieNode>();
        }
        TrieNode root = new TrieNode();

        public ModifiedSuffixTrie(string str){
            PopulateModifiedSuffixTrieFrom(str);
        }

        private void PopulateModifiedSuffixTrieFrom(string str)
        {
            for(int i=0; i< str.Length; ++i)
                InsertSubstringStartingAt(i, str);
        }

        private void InsertSubstringStartingAt(int i, string str)
        {
            TrieNode node = root;

            for(int j=i; j<str.Length; ++j){
                char letter = str[j];

                if(!node.children.ContainsKey(letter)){
                    TrieNode newNode = new TrieNode();
                    node.children.Add(letter, newNode);
                }
                node = node.children[letter];
            }
        }
        public bool Contains(string str){
            TrieNode node = root;
            for(int i=0; i< str.Length; ++i){
                char letter = str[i];
                if(!node.children.ContainsKey(letter))
                    return false;
                node = node.children[letter];
            }
            return true;
        }
    }
}