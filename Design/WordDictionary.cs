using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{

/*
211. Design Add and Search Words Data Structure
https://leetcode.com/problems/design-add-and-search-words-data-structure/	

*/

    public class WordDictionary
    {
        TrieNode trie;

        /** Initialize your data structure here. */
        public WordDictionary()
        {
            trie = new TrieNode();
        }

/*
Complexity Analysis
•	Time complexity: O(M), where M is the key length. At each step, we either examine or create a node in the trie. That takes only M operations.
•	Space complexity: O(M). In the worst-case newly inserted key doesn't share a prefix with the keys already inserted in the trie. We have to add M new nodes, which takes O(M) space.

*/
        /** Adds a word into the data structure. */
        public void AddWord(String word)
        {
            TrieNode node = trie;

            foreach (char ch in word.ToCharArray())
            {
                if (!node.Children.ContainsKey(ch))
                {
                    node.Children.Add(ch, new TrieNode());
                }
                node = node.Children[ch];
            }
            node.Word = true;
        }

/*
Complexity Analysis
•	Time complexity: O(M) for the "well-defined" words without dots, where M is the key length, and N is a number of keys, and O(N⋅26^M) for the "undefined" words. That corresponds to the worst-case situation of searching an undefined word M times......... which is one character longer than all inserted keys.
•	Space complexity: O(1) for the search of "well-defined" words without dots, and up to O(M) for the "undefined" words, to keep the recursion stack.

*/
        /** Returns if the word is in the node. */
        public bool SearchInNode(String word, TrieNode node)
        {
            for (int i = 0; i < word.Length; ++i)
            {
                char ch = word[i];
                if (!node.Children.ContainsKey(ch))
                {
                    // if the current character is '.'
                    // check all possible nodes at this level
                    if (ch == '.')
                    {
                        foreach (char x in node.Children.Keys)
                        {
                            TrieNode child = node.Children[x];
                            if (SearchInNode(word.Substring(i + 1), child))
                            {
                                return true;
                            }
                        }
                    }

                    // if no nodes lead to answer
                    // or the current character != '.'
                    return false;
                }
                else
                {
                    // if the character is found
                    // go down to the next level in trie
                    node = node.Children[ch];
                }
            }
            return node.Word;
        }

        /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
        public bool Search(String word)
        {
            return SearchInNode(word, trie);
        }
        public class TrieNode
        {
            public Dictionary<char, TrieNode> Children = new Dictionary<char, TrieNode>();
            public bool Word = false;

            public TrieNode() { }
        }
    }
}
