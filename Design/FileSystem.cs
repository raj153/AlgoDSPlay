using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*     1166. Design File System
        https://leetcode.com/problems/design-file-system/description/
     */
    public class FileSystem
    {
        /*
        Approach 1: Dictionary for storing paths
        Complexity Analysis
•	Time Complexity: O(M), where M is the length of path. All the time is actually consumed by the operation that gives us the parent path. We first spend O(M) on finding the last "/" of the path and then another O(M) to obtain the parent string. Searching and addition into a HashMap/dictionary takes an ammortized O(1) time.
•	Space Complexity: O(K) where K represents the number of unique paths that we add.

        */
        class FileSystemUsingDict
        {
            private Dictionary<string, int> paths;

            public FileSystemUsingDict()
            {
                this.paths = new Dictionary<string, int>();
            }

            public bool CreatePath(string path, int value)
            {
                // Step-1: basic path validations
                if (string.IsNullOrEmpty(path) || (path.Length == 1 && path.Equals("/")) || this.paths.ContainsKey(path))
                {
                    return false;
                }

                int delimIndex = path.LastIndexOf("/");
                string parent = path.Substring(0, delimIndex);

                // Step-2: if the parent doesn't exist. Note that "/" is a valid parent.
                if (parent.Length > 1 && !this.paths.ContainsKey(parent))
                {
                    return false;
                }

                // Step-3: add this new path and return true.
                this.paths[path] = value;
                return true;
            }

            public int Get(string path)
            {
                return this.paths.TryGetValue(path, out int value) ? value : -1;
            }
        }

        /*
        Approach 2: Trie based approach
Complexity Analysis
Before we get into the complexity analysis, let's see why one might prefer the Trie approach. The main advantage of the trie based approach is that we are able to save on space. All the paths sharing common prefixes can be represented by a common branch in the tree. The disadvantage however is that the get operation no longer remains O(1).
•	Time Complexity:
o	create ~ It takes O(T) to add a path to the trie if it contains T components.
o	get ~ It takes O(T) to find a path in the trie if it contains T components.
•	Space Complexity:
o	create ~ Lets look at the worst case space complexity. In the worst case, none of the paths will have any common prefixes. We are not considering the ancestors of a larger path here. In such a case, each unique path will end up taking a different branch in the trie. Also, for a path containing T components, there will be T nodes in the trie.
o	get ~ O(1).

        */
        public class FileSystemUsingTrie
        {
            // The TrieNode data structure.
            private class TrieNode
            {
                public string Name { get; }
                public int Value { get; set; } = -1;
                public Dictionary<string, TrieNode> Map { get; } = new Dictionary<string, TrieNode>();

                public TrieNode(string name)
                {
                    Name = name;
                }
            }

            private TrieNode root;

            // Root node contains the empty string.
            public FileSystemUsingTrie()
            {
                root = new TrieNode("");
            }

            public bool CreatePath(string path, int value)
            {
                // Obtain all the components
                string[] components = path.Split('/');

                // Start "current" from the root node.
                TrieNode current = root;

                // Iterate over all the components.
                for (int i = 1; i < components.Length; i++)
                {
                    string currentComponent = components[i];

                    // For each component, we check if it exists in the current node's dictionary.
                    if (!current.Map.ContainsKey(currentComponent))
                    {
                        // If it doesn't and it is the last node, add it to the Trie.
                        if (i == components.Length - 1)
                        {
                            current.Map[currentComponent] = new TrieNode(currentComponent);
                        }
                        else
                        {
                            return false;
                        }
                    }

                    current = current.Map[currentComponent];
                }

                // Value not equal to -1 means the path already exists in the trie. 
                if (current.Value != -1)
                {
                    return false;
                }

                current.Value = value;
                return true;
            }

            public int Get(string path)
            {
                // Obtain all the components
                string[] components = path.Split('/');

                // Start "current" from the root node.
                TrieNode current = root;

                // Iterate over all the components.
                for (int i = 1; i < components.Length; i++)
                {
                    string currentComponent = components[i];

                    // For each component, we check if it exists in the current node's dictionary.
                    if (!current.Map.ContainsKey(currentComponent))
                    {
                        return -1;
                    }

                    current = current.Map[currentComponent];
                }

                return current.Value;
            }
        }

    }
}