using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    208. Implement Trie (Prefix Tree)
    https://leetcode.com/problems/implement-trie-prefix-tree/

    */
    public class Trie
    {
        private TrieNode root;

        public Trie()
        {
            root = new TrieNode();
        }

        /*
        **Complexity Analysis**
            Time complexity : O(m), where m is the key length.
            In each iteration of the algorithm, we either examine or create a node in the trie till we reach the end of the key. This takes only m operations.

        Space complexity : O(m).
            In the worst case newly inserted key doesn't share a prefix with the the keys already inserted in the trie. We have to add m
            new nodes, which takes us O(m) space.


        */
        // Inserts a word into the trie.
        public void Insert(String word)
        {
            TrieNode node = root;
            for (int i = 0; i < word.Length; i++)
            {
                char currentChar = word[i];
                if (!node.ContainsKey(currentChar))
                {
                    node.Put(currentChar, new TrieNode());
                }
                node = node.Get(currentChar);
            }
            node.SetEnd();
        }

        /*
        **Complexity Analysis**
        Time complexity : O(m)
        In each step of the algorithm we search for the next key character. In the worst case the algorithm performs m operations.

        Space complexity : O(1)
        */
        // search a prefix or whole key in trie and
        // returns the node where search ends
        private TrieNode SearchPrefix(String word)
        {
            TrieNode node = root;
            for (int i = 0; i < word.Length; i++)
            {
                char curLetter = word[i];
                if (node.ContainsKey(curLetter))
                {
                    node = node.Get(curLetter);
                }
                else
                {
                    return null;
                }
            }
            return node;
        }

        // Returns if the word is in the trie.
        public bool Search(String word)
        {
            TrieNode node = SearchPrefix(word);
            return node != null && node.IsEnd();
        }

        /*
        **Complexity Analysis**
        Time complexity : O(m)
        Space complexity : O(1)
        */
        // Returns if there is any word in the trie
        // that starts with the given prefix.
        public bool StartsWith(String prefix)
        {
            TrieNode node = SearchPrefix(prefix);
            return node != null;
        }

    }
    class TrieNode
    {

        // R links to node children
        private TrieNode[] links;

        private readonly int R = 26;

        private bool isEnd;

        public TrieNode()
        {
            links = new TrieNode[R];
        }

        public bool ContainsKey(char ch)
        {
            return links[ch - 'a'] != null;
        }
        public TrieNode Get(char ch)
        {
            return links[ch - 'a'];
        }
        public void Put(char ch, TrieNode node)
        {
            links[ch - 'a'] = node;
        }
        public void SetEnd()
        {
            isEnd = true;
        }
        public bool IsEnd()
        {
            return isEnd;
        }
    }
    /**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.Insert(word);
 * bool param_2 = obj.Search(word);
 * bool param_3 = obj.StartsWith(prefix);
 */
}