using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    642. Design Search Autocomplete System
    https://leetcode.com/problems/design-search-autocomplete-system/description/

    */
    public class AutocompleteSystem
    {

        /*
        Approach 1: Trie

        Complexity Analysis:
        Given n as the length of sentences, k as the average length of all sentences, and m as the number of times input is called,
        •	Time complexity: O(n⋅k+m⋅(n+km)⋅log(n+km))
                constructor:
                    o	We initialize the trie, which costs O(n⋅k) as we iterate over each character in each sentence.
                input:
                    o	We add a character to currSentence and the trie, both cost O(1). Next, we fetch and sort the sentences in the current node. Initially, a node could hold O(n) sentences. After we call input m times, we could add km new sentences. Overall, there could be up to O(n+km) sentences, so a sort would cost O((n+km)⋅log(n+km)).
                    o	The work done in the other cases (like adding a new sentence to the trie) will be dominated by this sort.
                    o	input is called m times, which gives us a total of O(m⋅(n+km)⋅log(n+km))
        •	Space complexity: O(k⋅(n⋅k+m))
                    The worst-case scenario for the trie size is when no two sentences share any prefix. The trie will initially have a size of n⋅k. Then, each call to input would create a new node.
                    Each of these trie nodes has children and sentences hash maps. The size of children is limited to 26, so we will ignore it. The size of sentences is variable, but in the case described, each node will only have 1 entry (because no two sentences share any prefix, so no trie node is visited by more than one sentence). This 1 entry will have a size of O(k).


        */

        class TrieNode
        {
            public Dictionary<char, TrieNode> Children { get; set; }
            public Dictionary<string, int> Sentences { get; set; }

            public TrieNode()
            {
                Children = new Dictionary<char, TrieNode>();
                Sentences = new Dictionary<string, int>();
            }
        }

        class AutocompleteSystemTrie
        {
            private TrieNode root;
            private TrieNode currentNode;
            private TrieNode deadNode;
            private StringBuilder currentSentence;

            public AutocompleteSystemTrie(string[] sentences, int[] times)
            {
                root = new TrieNode();
                for (int i = 0; i < sentences.Length; i++)
                {
                    AddToTrie(sentences[i], times[i]);
                }

                currentSentence = new StringBuilder();
                currentNode = root;
                deadNode = new TrieNode();
            }

            public List<string> Input(char c)
            {
                if (c == '#')
                {
                    AddToTrie(currentSentence.ToString(), 1);
                    currentSentence.Clear();
                    currentNode = root;
                    return new List<string>();
                }

                currentSentence.Append(c);
                if (!currentNode.Children.ContainsKey(c))
                {
                    currentNode = deadNode;
                    return new List<string>();
                }

                currentNode = currentNode.Children[c];
                List<string> sentences = new List<string>(currentNode.Sentences.Keys);
                sentences.Sort((a, b) =>
                {
                    int hotA = currentNode.Sentences[a];
                    int hotB = currentNode.Sentences[b];
                    if (hotA == hotB)
                    {
                        return string.Compare(a, b);
                    }

                    return hotB - hotA;
                });

                List<string> result = new List<string>();
                for (int i = 0; i < Math.Min(3, sentences.Count); i++)
                {
                    result.Add(sentences[i]);
                }

                return result;
            }

            private void AddToTrie(string sentence, int count)
            {
                TrieNode node = root;
                foreach (char c in sentence)
                {
                    if (!node.Children.ContainsKey(c))
                    {
                        node.Children[c] = new TrieNode();
                    }

                    node = node.Children[c];
                    node.Sentences[sentence] = node.Sentences.GetValueOrDefault(sentence, 0) + count;
                }
            }
        }

        /*
        Approach 2: Trie Optimized with Heap

        Complexity Analysis
        Given n as the length of sentences, k as the average length of all sentences, and m as the number of times input is called,
        This analysis will assume that you have access to a linear time heapify method, like in the Python implementation.
            •	Time complexity: O(n⋅k+m⋅(n+km))
                constructor:
                    o	We initialize the trie, which costs O(n⋅k) as we iterate over each character in each sentence.
                input:  
                    o	We add a character to currSentence and the trie, both cost O(1). Next, we fetch the sentences in the current node. Initially, a node could hold O(n) sentences. After we call input m times, we could add km new sentences. Overall, there could be up to O(n+km) sentences. We heapify these sentences and find the best 3 in linear time, which costs O(n+km).
                    o	The work done in the other cases (like adding a new sentence to the trie) will be dominated by this.
                    o	input is called m times, which gives us a total of O(m⋅(n+km)).
        •	Space complexity: O(k⋅(n⋅k+m))
                    The worst-case scenario for the trie size is when no two sentences share any prefix. The trie will initially have a size of n⋅k. Then, each call to input would create a new node.
                    Each of these trie nodes has children and sentences hash maps. The size of children is limited to 26, so we will ignore it. The size of sentences is variable, but in the case described, each node will only have 1 entry (because no two sentences share any prefix, 
                    so no trie node is visited by more than one sentence). This 1 entry will have a size of O(k).
        */
        public class AutocompleteSystemTrieHeap
        {
            private TrieNode root;
            private TrieNode currentNode;
            private TrieNode deadNode;
            private StringBuilder currentSentence;

            public AutocompleteSystemTrieHeap(string[] sentences, int[] times)
            {
                root = new TrieNode();
                for (int i = 0; i < sentences.Length; i++)
                {
                    AddToTrie(sentences[i], times[i]);
                }

                currentSentence = new StringBuilder();
                currentNode = root;
                deadNode = new TrieNode();
            }

            public IList<string> Input(char c)
            {
                if (c == '#')
                {
                    AddToTrie(currentSentence.ToString(), 1);
                    currentSentence.Clear();
                    currentNode = root;
                    return new List<string>();
                }

                currentSentence.Append(c);
                if (!currentNode.Children.ContainsKey(c))
                {
                    currentNode = deadNode;
                    return new List<string>();
                }

                currentNode = currentNode.Children[c];
                SortedSet<string> heap = new SortedSet<string>(Comparer<string>.Create((a, b) => //Heap
                {
                    int hotA = currentNode.Sentences[a];
                    int hotB = currentNode.Sentences[b];
                    if (hotA == hotB)
                    {
                        return string.Compare(b, a); // Reverse order for tie-breaking
                    }

                    return hotA - hotB;
                }));

                PriorityQueue<string, string> heap1 = new PriorityQueue<string, string>(Comparer<string>.Create((a, b) => //TODO: Replace above SortedSet with this PQ
                {
                    int hotA = currentNode.Sentences[a];
                    int hotB = currentNode.Sentences[b];
                    if (hotA == hotB)
                    {
                        return string.Compare(b, a); // Reverse order for tie-breaking
                    }

                    return hotA - hotB;
                }));

                foreach (var sentence in currentNode.Sentences.Keys)
                {
                    heap.Add(sentence);
                    if (heap.Count > 3)
                    {
                        heap.Remove(heap.Min);
                    }
                }

                List<string> result = new List<string>(heap);
                result.Reverse();
                return result;
            }

            private void AddToTrie(string sentence, int count)
            {
                TrieNode node = root;
                foreach (char c in sentence)
                {
                    if (!node.Children.ContainsKey(c))
                    {
                        node.Children[c] = new TrieNode();
                    }

                    node = node.Children[c];
                    if (!node.Sentences.ContainsKey(sentence))
                    {
                        node.Sentences[sentence] = 0;
                    }
                    node.Sentences[sentence] += count;
                }
            }
        }
    }
}