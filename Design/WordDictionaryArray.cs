using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{

    class WordDictionaryArray
    {
        private TrieNode root;

        /** Initialize your data structure here. */
        public WordDictionaryArray()
        {
            root = new TrieNode();
        }

        // Adds a word into the data structure
        public void AddWord(String word)
        {
            TrieNode node = root;
            foreach (char c in word.ToCharArray())
            {
                int index = c - 'a';
                if (node.Children[index] == null)
                {
                    // If there is no TrieNode for this letter, create a new TrieNode
                    node.Children[index] = new TrieNode();
                }
                // Move to the next node
                node = node.Children[index];
            }
            // Mark this node as the end of a word
            node.isEndOfWord = true;
        }

        // Searches for a word in the data structure and can handle '.' as a wildcard character
        public bool search(String word)
        {
            return searchInNode(word, root);
        }

        private bool searchInNode(String word, TrieNode node)
        {
            for (int i = 0; i < word.Length; ++i)
            {
                char c = word[i];
                if (c == '.')
                {
                    // If it's a wildcard, we need to check all possible paths
                    foreach (TrieNode child in node.Children)
                    {
                        if (child != null && searchInNode(word.Substring(i + 1), child))
                        {
                            return true;
                        }
                    }
                    return false; // If no paths find a match, return false
                }
                else
                {
                    int index = c - 'a';
                    // If the specific child TrieNode does not exist, the word does not exist
                    if (node.Children[index] == null)
                    {
                        return false;
                    }
                    // Move to the next node
                    node = node.Children[index];
                }
            }
            // After processing all characters, check if it is the end of a word
            return node.isEndOfWord;
        }
        class TrieNode
        {
            // Each TrieNode contains an array of children TrieNodes to represent each letter of the alphabet
            public TrieNode[] Children = new TrieNode[26];
            // Indicates if a word ends at this node
            public bool isEndOfWord;
        }
    }

    /**
     * Your WordDictionary object will be instantiated and called as such:
     * WordDictionary obj = new WordDictionary();
     * obj.addWord(word);
     * bool param_2 = obj.search(word);
     */

}