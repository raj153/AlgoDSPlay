using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class TrieNode{
            public Dictionary<char, TrieNode> Children = new Dictionary<char, TrieNode>();
            public string Word="";
            public bool IsEndOfString= false; //Added for StringsMadeUpOfSubstring problem
            
            public int Count =0; //Added for Longest Prefix string problem
    }
    public class Trie
    {        
        public char EndSymbol;
        public TrieNode Root;
        public int MaxPrefixCount =0; //Added for Longest Prefix string problem
        public int MaxPrefixLen = 0; //Added for Longest Prefix string problem
        public string MaxPrefixFullString = ""; //Added for Longest Prefix string problem
        

        public Trie(){
            this.Root = new TrieNode();
            this.EndSymbol='*';
        }
        public void Insert(string str){
            TrieNode node = Root;

            for(int i=0; i< str.Length; ++i){
                char letter = str[i];
                if(!node.Children.ContainsKey(letter)){
                    node.Children.Add(letter, new TrieNode());
                }
                node = node.Children[letter];
                
                //Added for Longest Prefix string problem
                    node.Count++;                
                    if(node.Count > this.MaxPrefixCount){
                        this.MaxPrefixCount = node.Count;
                        this.MaxPrefixLen = i+1;
                        this.MaxPrefixFullString =str;
                    }else if(node.Count == this.MaxPrefixCount && i+1 > this.MaxPrefixLen){
                        this.MaxPrefixLen = i+1;
                        this.MaxPrefixFullString = str;
                    }
            }
            node.Children[EndSymbol]=null;
            node.Word=str;
            node.IsEndOfString = true;            

        }


    }
}