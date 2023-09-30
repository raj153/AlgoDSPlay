using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class TrieNode{
            public Dictionary<char, TrieNode> children = new Dictionary<char, TrieNode>();
            public string word="";
    }
    public class Trie
    {
        
        public char endSymbol;

        public TrieNode root;

        public Trie(){
            this.root = new TrieNode();
            this.endSymbol='*';
        }
        public void Insert(string str){
            TrieNode node = root;

            for(int i=0; i< str.Length; ++i){
                char letter = str[i];
                if(!node.children.ContainsKey(letter)){
                    node.children.Add(letter, new TrieNode());
                }
                node = node.children[letter];
            }
            node.children[endSymbol]=null;
            node.word=str;
        }


    }
}