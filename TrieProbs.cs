using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class TrieProbs
    {
        //https://www.algoexpert.io/questions/strings-made-up-of-strings
        public static string[] StringMadeUpOfSubstrings(string[] strings, string[] substrings)
        {
            //T:O(s2*m+s1*n^2) | S:O(s2*m+s1*n) - where s2 is number of substrings and m is length of longest substring.
            //s1 is number of strings and n is legnth of longest string
            Trie trie = new Trie();

            foreach (string str in substrings)
            {
                trie.Insert(str);
            }

            List<string> solutions = new List<string>();
            foreach (var str in strings)
            {
                if (IsMadeUpOfSubstrings(str, 0, trie, new Dictionary<int, bool>()))
                {
                    solutions.Add(str);
                }
            }
            return solutions.ToArray();
        }

        private static bool IsMadeUpOfSubstrings(string str, int startIdx, Trie trie, Dictionary<int, bool> memo)
        {
            if (startIdx == str.Length) return true; //Base case

            if (memo.ContainsKey(startIdx)) return memo[startIdx]; //Base case

            TrieNode currentTrieNode = trie.Root;

            for (int currentCharIdx = startIdx; currentCharIdx < str.Length; currentCharIdx++)
            {

                char curChar = str[currentCharIdx];

                if (!currentTrieNode.Children.ContainsKey(curChar)) break;

                currentTrieNode = currentTrieNode.Children[curChar];
                if (currentTrieNode.IsEndOfString && IsMadeUpOfSubstrings(str, currentCharIdx + 1, trie, memo))
                {
                    memo[startIdx] = true;
                    return true;
                }
            }
            memo[startIdx] = false;
            return false;

        }
        //https://www.algoexpert.io/questions/boggle-board
        public static List<string> BoggleBoard(char[,] board, string[] words)
        {
            //T:O(WS+MN*8^S) | S:O(WS+MN)
            Trie trie = new Trie();
            foreach (var word in words)
            {
                trie.Insert(word);
            }
            HashSet<string> finalWords = new HashSet<string>();
            bool[,] visited = new bool[board.GetLength(0), board.GetLength(1)];
            for (int row = 0; row < board.GetLength(0); row++)
            {
                for (int col = 0; col < board.GetLength(1); col++)
                {
                    Explore(row, col, board, trie.Root, visited, finalWords);
                }
            }
            List<string> finalWordsArray = new List<string>();
            foreach (string key in finalWords)
            {
                finalWordsArray.Add(key);
            }
            return finalWordsArray;

        }

        private static void Explore(int row, int col, char[,] board, TrieNode trieNode, bool[,] visited, HashSet<string> finalWords)
        {
            if (visited[row, col]) return;

            char letter = board[row, col];
            if (!trieNode.Children.ContainsKey(letter)) return;
            visited[row, col] = true;

            trieNode = trieNode.Children[letter];
            if (trieNode.Children.ContainsKey('*'))
            {  //endSymbol checking
                finalWords.Add(trieNode.Word);
            }

            List<int[]> neighbors = GetNeighbors(row, col, board);
            foreach (int[] neighbor in neighbors)
            {
                Explore(neighbor[0], neighbor[1], board, trieNode, visited, finalWords);
            }
            visited[row, col] = false;

        }

        public static List<int[]> GetNeighbors(int row, int col, char[,] board)
        {

            List<int[]> neighbors = new List<int[]>();

            //Top-Left Diagonal 
            if (row > 0 && col > 0)
                neighbors.Add(new int[] { row - 1, col - 1 });
            //Top
            if (row > 0)
                neighbors.Add(new int[] { row - 1, col });
            //Top-Right Diagonal
            if (row > 0 && col < board.GetLength(1) - 1)
                neighbors.Add(new int[] { row - 1, col + 1 });
            //Right
            if (col < board.GetLength(1) - 1)
                neighbors.Add(new int[] { row, col + 1 });
            //Down-Right Diagonal
            if (row > board.GetLength(0) - 1 && col < board.GetLength(1) - 1)
                neighbors.Add(new int[] { row + 1, col + 1 });
            //Down
            if (row > board.GetLength(0) - 1)
                neighbors.Add(new int[] { row + 1, col });
            //Down-Left Diagonal
            if (row > board.GetLength(0) - 1 && col > 0)
                neighbors.Add(new int[] { row + 1, col - 1 });
            //Left
            if (col > 0)
                neighbors.Add(new int[] { row, col - 1 });

            return neighbors;
        }

        //https://www.algoexpert.io/questions/multi-string-search
        public static List<bool> MultiStringSearch(string bigString, string[] smallStrings)
        {
            List<bool> solution = new List<bool>();
            //1.Naive - T:O(bns) | S:O(n) where b is length of big string, s is length of biggest small string and n is length of small string array
            foreach (string smallString in smallStrings)
            {
                solution.Add(IsInBigString(bigString, smallString));

            }
            //2.ModifiedSuffixTrie - T:O(b^2+ns) | S:O(b^2+n)
            ModifiedSuffixTrie modifiedSuffixTrie = new ModifiedSuffixTrie(bigString);
            solution.Clear();
            foreach (string smallString in smallStrings)
            {
                solution.Add(modifiedSuffixTrie.Contains(smallString));
            }

            solution.Clear();
            //3.Trie - T:O(ns+bs) | S:O(ns)
            Trie trie = new Trie();
            foreach (string smallString in smallStrings)
            {
                trie.Insert(smallString);
            }
            HashSet<string> containedStrings = new HashSet<string>();
            for (int i = 0; i < bigString.Length; ++i)
            {
                FindSmallStringsIn(bigString, i, trie, containedStrings);
            }
            foreach (string smallString in smallStrings)
            {
                solution.Add(containedStrings.Contains(smallString));
            }
            return solution;

        }

        private static void FindSmallStringsIn(string bigString, int startIdx, Trie trie, HashSet<string> containedStrings)
        {
            TrieNode currentNode = trie.Root;
            for (int i = startIdx; i < bigString.Length; ++i)
            {
                char currentChar = bigString[i];
                if (!currentNode.Children.ContainsKey(currentChar))
                {
                    break;
                }
                currentNode = currentNode.Children[currentChar];
                if (currentNode.Children.ContainsKey(trie.EndSymbol))
                {
                    containedStrings.Add(currentNode.Word);
                }
            }


        }

        private static bool IsInBigString(string bigString, string smallString)
        {
            for (int i = 0; i < bigString.Length; ++i)
            {
                if (i + smallString.Length > bigString.Length)
                {
                    break;
                }
                if (IsInBigString(bigString, smallString, i))
                {
                    return true;
                }
            }
            return false;
        }

        private static bool IsInBigString(string bigString, string smallString, int startIdx)
        {
            //big bigger
            //egg
            int leftBigIndex = startIdx;
            int rightBigIndex = startIdx + smallString.Length - 1;
            int leftSmallIndex = 0;
            int rightSmallIndex = smallString.Length - 1;
            while (leftBigIndex <= rightBigIndex)
            {

                if (bigString[leftBigIndex] != smallString[leftSmallIndex]
                    || bigString[rightBigIndex] != smallString[rightSmallIndex])
                    return false;

                leftBigIndex++;
                rightSmallIndex--;
                leftSmallIndex++;
                rightBigIndex--;
            }
            return true;

        }
        //https://www.algoexpert.io/questions/longest-most-frequent-prefix
        public static string LongestMostFrequentPrefix(string[] strings)
        {

            //T:O(n*m) | S:O(n*m) where n is number of strings and m is length of longest string in strings array
            Trie trie = new Trie();
            foreach (string str in strings)
            {
                trie.Insert(str);
            }
            return trie.MaxPrefixFullString.Substring(0, trie.MaxPrefixLen);

        }
        //https://www.algoexpert.io/questions/shortest-unique-prefixes

        // O(n * m) time | O(n * m) space - where n is the length of strings, and m
        // is the length of the longest string
        public string[] ShortestUniquePrefixes(string[] strings)
        {
            Trie trie = new Trie();
            foreach (var str in strings)
            {
                trie.Insert(str);
            }

            string[] prefixes = new string[strings.Length];
            for (int idx = 0; idx < strings.Length; idx++)
            {
                string uniquePrefix = findUniquePrefix(strings[idx], trie);
                prefixes[idx] = uniquePrefix;
            }

            return prefixes;
        }

        static string findUniquePrefix(string str, Trie trie)
        {
            int currentstringIdx = 0;
            TrieNode currentTrieNode = trie.Root;

            while (currentstringIdx < str.Length - 1)
            {
                char currentstringChar = str[currentstringIdx];
                currentTrieNode = currentTrieNode.Children[currentstringChar];
                if (currentTrieNode.Count == 1) break;
                currentstringIdx++;
            }

            return str.Substring(0, currentstringIdx + 1 - 0);
        }
        /* 2416. Sum of Prefix Scores of Strings
        https://leetcode.com/problems/sum-of-prefix-scores-of-strings/description/
         */


        class SumPrefixScoresSol
        {

            // Initialize the root node of the trie.
            TrieNode root = new TrieNode();

            /* 
            Approach: Tries
            Complexity Analysis
            Let N be the size of words array, and M be the average length of the strings in words.
            •	Time complexity: O(N⋅M)
            The insert operation takes O(length) time for a string of size length. The total time taken to perform the insert operations on the strings of the words array is given by O(N⋅M).
            Similarly, the count operation takes O(length) time for a string of size length. The total time taken to perform the count operations on the strings of the words array is given by O(N⋅M).
            Therefore, the total time complexity is given by O(N⋅M).
            •	Space complexity: O(N⋅M)
            The insert operation takes O(length) space for a string of size length. The total space taken to perform the insert operations on the strings of the words array is given by O(N⋅M).
            The count operation does not use any additional space. Therefore, the total time complexity is given by O(N⋅M).
             */
            public int[] UsingTrie(String[] words)
            {
                int N = words.Length;
                // Insert words in trie.
                for (int i = 0; i < N; i++)
                {
                    Insert(words[i]);
                }
                int[] scores = new int[N];
                for (int i = 0; i < N; i++)
                {
                    // Get the count of all prefixes of given string.
                    scores[i] = Count(words[i]);
                }
                return scores;
            }
            // Insert function for the word.
            void Insert(String word)
            {
                TrieNode node = root;
                foreach (char c in word)
                {
                    // If new prefix, create a new trie node.
                    if (node.next[c - 'a'] == null)
                    {
                        node.next[c - 'a'] = new TrieNode();
                    }
                    // Increment the count of the current prefix.
                    node.next[c - 'a'].cnt++;
                    node = node.next[c - 'a'];
                }
            }

            // Calculate the prefix count using this function.
            int Count(String s)
            {
                TrieNode node = root;
                int ans = 0;
                // The ans would store the total sum of counts.
                foreach (char c in s)
                {
                    ans += node.next[c - 'a'].cnt;
                    node = node.next[c - 'a'];
                }
                return ans;
            }


            class TrieNode
            {

                public TrieNode[] next = new TrieNode[26];
                public int cnt = 0;
            }
        }


        /* 745. Prefix and Suffix Search
        https://leetcode.com/problems/prefix-and-suffix-search/description/
         */
        public class WordFilterSol
        {
            class TrieWithSetIntersection
            {
                TrieNode trie;
                /* 
                Approach #1: Trie + Set Intersection [Time Limit Exceeded] 
                Complexity Analysis
•	Time Complexity: O(NK+Q(N+K)) where N is the number of words, K is the maximum length of a word, and Q is the number of queries. If we use memoization in our solution, we could produce tighter bounds for this complexity, as the complex queries are somewhat disjoint.
•	Space Complexity: O(NK), the size of the tries.

                */
                public TrieWithSetIntersection(string[] words)
                {
                    trie = new TrieNode();
                    int weight = 0;
                    foreach (string word in words)
                    {
                        TrieNode currentNode = trie;
                        currentNode.Weight = weight;
                        int length = word.Length;
                        char[] characters = word.ToCharArray();
                        for (int i = 0; i < length; ++i)
                        {
                            TrieNode temporaryNode = currentNode;
                            for (int j = i; j < length; ++j)
                            {
                                int code = (characters[j] - '`') * 27;
                                if (!temporaryNode.Children.ContainsKey(code))
                                {
                                    temporaryNode.Children[code] = new TrieNode();
                                }
                                temporaryNode = temporaryNode.Children[code];
                                temporaryNode.Weight = weight;
                            }

                            temporaryNode = currentNode;
                            for (int k = length - 1 - i; k >= 0; --k)
                            {
                                int code = (characters[k] - '`');
                                if (!temporaryNode.Children.ContainsKey(code))
                                {
                                    temporaryNode.Children[code] = new TrieNode();
                                }
                                temporaryNode = temporaryNode.Children[code];
                                temporaryNode.Weight = weight;
                            }

                            int combinedCode = (characters[i] - '`') * 27 + (characters[length - 1 - i] - '`');
                            if (!currentNode.Children.ContainsKey(combinedCode))
                            {
                                currentNode.Children[combinedCode] = new TrieNode();
                            }
                            currentNode = currentNode.Children[combinedCode];
                            currentNode.Weight = weight;
                        }
                        weight++;
                    }
                }

                public int f(string prefix, string suffix)
                {
                    TrieNode currentNode = trie;
                    int prefixIndex = 0, suffixIndex = suffix.Length - 1;
                    while (prefixIndex < prefix.Length || suffixIndex >= 0)
                    {
                        char char1 = prefixIndex < prefix.Length ? prefix[prefixIndex] : '`';
                        char char2 = suffixIndex >= 0 ? suffix[suffixIndex] : '`';
                        int combinedCode = (char1 - '`') * 27 + (char2 - '`');
                        currentNode = currentNode.Children.ContainsKey(combinedCode) ? currentNode.Children[combinedCode] : null;
                        if (currentNode == null)
                        {
                            return -1;
                        }
                        prefixIndex++;
                        suffixIndex--;
                    }
                    return currentNode.Weight;
                }
            }

            class TrieNode
            {
                public Dictionary<int, TrieNode> Children { get; set; }
                public int Weight { get; set; }

                public TrieNode()
                {
                    Children = new Dictionary<int, TrieNode>();
                    Weight = 0;
                }
            }
            /*   Approach #2: Paired Trie [Accepted]
            Complexity Analysis
  •	Time Complexity: O(NK^2+QK) where N is the number of words, K is the maximum length of a word, and Q is the number of queries.
  •	Space Complexity: O(NK^2), the size of the trie.

   */
            class PairedTrie
            {
                private TrieNode trie;

                public PairedTrie(string[] words)
                {
                    trie = new TrieNode();
                    int weight = 0;
                    foreach (string word in words)
                    {
                        TrieNode currentNode = trie;
                        currentNode.Weight = weight;
                        int length = word.Length;
                        char[] characters = word.ToCharArray();
                        for (int i = 0; i < length; ++i)
                        {
                            TrieNode tempNode = currentNode;
                            for (int j = i; j < length; ++j)
                            {
                                int code = (characters[j] - '`') * 27;
                                if (!tempNode.Children.ContainsKey(code))
                                {
                                    tempNode.Children[code] = new TrieNode();
                                }
                                tempNode = tempNode.Children[code];
                                tempNode.Weight = weight;
                            }

                            tempNode = currentNode;
                            for (int k = length - 1 - i; k >= 0; --k)
                            {
                                int code = (characters[k] - '`');
                                if (!tempNode.Children.ContainsKey(code))
                                {
                                    tempNode.Children[code] = new TrieNode();
                                }
                                tempNode = tempNode.Children[code];
                                tempNode.Weight = weight;
                            }

                            int combinedCode = (characters[i] - '`') * 27 + (characters[length - 1 - i] - '`');
                            if (!currentNode.Children.ContainsKey(combinedCode))
                            {
                                currentNode.Children[combinedCode] = new TrieNode();
                            }
                            currentNode = currentNode.Children[combinedCode];
                            currentNode.Weight = weight;
                        }
                        weight++;
                    }
                }

                public int f(string prefix, string suffix)
                {
                    TrieNode currentNode = trie;
                    int prefixIndex = 0, suffixIndex = suffix.Length - 1;
                    while (prefixIndex < prefix.Length || suffixIndex >= 0)
                    {
                        char charFromPrefix = prefixIndex < prefix.Length ? prefix[prefixIndex] : '`';
                        char charFromSuffix = suffixIndex >= 0 ? suffix[suffixIndex] : '`';
                        int combinedCode = (charFromPrefix - '`') * 27 + (charFromSuffix - '`');
                        if (!currentNode.Children.TryGetValue(combinedCode, out currentNode))
                        {
                            return -1;
                        }
                        prefixIndex++;
                        suffixIndex--;
                    }
                    return currentNode.Weight;
                }
            }
            /* Approach #3: Trie of Suffix Wrapped Words [Accepted]
            Complexity Analysis
•	Time Complexity: O(NK^2+QK) where N is the number of words, K is the maximum length of a word, and Q is the number of queries.
•	Space Complexity: O(NK^2), the size of the trie.

             */
            public class TrieOfSuffixWrappedWords
            {
                private TrieNodeExt trie;

                public TrieOfSuffixWrappedWords(string[] words)
                {
                    trie = new TrieNodeExt();
                    for (int weight = 0; weight < words.Length; ++weight)
                    {
                        string word = words[weight] + "{";
                        for (int i = 0; i < word.Length; ++i)
                        {
                            TrieNodeExt currentNode = trie;
                            currentNode.Weight = weight;
                            for (int j = i; j < 2 * word.Length - 1; ++j)
                            {
                                int index = word[j % word.Length] - 'a';
                                if (currentNode.Children[index] == null)
                                {
                                    currentNode.Children[index] = new TrieNodeExt();
                                }
                                currentNode = currentNode.Children[index];
                                currentNode.Weight = weight;
                            }
                        }
                    }
                }

                public int F(string prefix, string suffix)
                {
                    TrieNodeExt currentNode = trie;
                    foreach (char letter in (suffix + '{' + prefix))
                    {
                        if (currentNode.Children[letter - 'a'] == null)
                        {
                            return -1;
                        }
                        currentNode = currentNode.Children[letter - 'a'];
                    }
                    return currentNode.Weight;
                }
            }

            public class TrieNodeExt
            {
                public TrieNodeExt[] Children;
                public int Weight;

                public TrieNodeExt()
                {
                    Children = new TrieNodeExt[27];
                    Weight = 0;
                }
            }

        }

        /* 1032. Stream of Characters
        https://leetcode.com/problems/stream-of-characters/description/
         */
        class StreamCheckerSol
        {
            class TrieNode
            {
                public Dictionary<char, TrieNode> Children { get; set; } = new Dictionary<char, TrieNode>();
                public bool IsWord { get; set; } = false;
            }
            /*
            Approach 1: Trie
 
            */
            private TrieNode trie = new TrieNode();
            private LinkedList<char> stream = new LinkedList<char>();
            /* 
                      Complexity Analysis
            Let N be the number of input words, and M be the word length.
            •	Time complexity: O(N⋅M).
            We have N words to process. At each step, we either examine or create a node in the trie. That takes only M operations.
            •	Space complexity: O(N⋅M).
            In the worst case, the newly inserted key doesn't share a prefix with the keys already added in the trie. We have to add N⋅M new nodes, which takes O(N⋅M) space.
              */
            public StreamCheckerSol(string[] words)
            {
                foreach (string word in words)
                {
                    TrieNode node = trie;
                    char[] reversedWordArray = word.ToCharArray();
                    Array.Reverse(reversedWordArray);
                    string reversedWord = new string(reversedWordArray);
                    foreach (char ch in reversedWord)
                    {
                        if (!node.Children.ContainsKey(ch))
                        {
                            node.Children[ch] = new TrieNode();
                        }
                        node = node.Children[ch];
                    }
                    node.IsWord = true;
                }
            }

            /* Let M be the maximum length of a word length. i.e. the depth of trie.
            •	Time complexity: O(M)
            •	Space complexity: O(M) to keep a stream of characters.
            One could limit the size of the deque to be equal to the length of the longest input word.
             */
            public bool Query(char letter)
            {
                stream.AddFirst(letter);

                TrieNode node = trie;
                foreach (char ch in stream)
                {
                    if (node.IsWord)
                    {
                        return true;
                    }
                    if (!node.Children.ContainsKey(ch))
                    {
                        return false;
                    }
                    node = node.Children[ch];
                }
                return node.IsWord;
            }
        }


        /* 3043. Find the Length of the Longest Common Prefix
        https://leetcode.com/problems/find-the-length-of-the-longest-common-prefix/description/
         */
        public class LongestCommonPrefixSol
        {

            /* Approach 1: Using Hash Table 
            Complexity Analysis
Let m be the length of arr1, n be the length of arr2, M be the maximum value in arr1, and N be the maximum value in arr2.
•	Time Complexity: O(m⋅log10M+n⋅log10N)
For each number in arr1, we repeatedly divide the number by 10 to generate its prefixes. Since dividing a number by 10 reduces the number of digits logarithmically, this process takes O(log10M) for each number in arr1. Hence, for m numbers, the total time complexity is O(m⋅log10M).
Similarly, for each number in arr2, we reduce it by repeatedly dividing it by 10 to check if it matches any prefix in the set. This also takes O(log10N) for each number in arr2. Hence, for n numbers, the total time complexity is O(n⋅log10N).
Overall, the total time complexity is O(m⋅log10M+n⋅log10N).
•	Space Complexit: O(m⋅log10M)
Each number in arr1 contributes O(log10M) space to the set, as it generates prefixes proportional to the number of digits (logarithmic in the value of the number with base 10). With m numbers in arr1, the total space complexity for the set is O(m⋅log10M).
The algorithm uses constant space for variables like longestPrefix and loop variables, so this doesn’t contribute significantly to the space complexity.
Thus, the total space complexity is O(m⋅log10M).

            */
            public int UsingHashSet(int[] array1, int[] array2)
            {
                HashSet<int> array1Prefixes = new HashSet<int>(); // Set to store all prefixes from array1

                // Step 1: Build all possible prefixes from array1
                foreach (int value in array1)
                {
                    int currentValue = value;
                    while (!array1Prefixes.Contains(currentValue) && currentValue > 0)
                    {
                        // Insert current value as a prefix
                        array1Prefixes.Add(currentValue);
                        // Generate the next shorter prefix by removing the last digit
                        currentValue /= 10;
                    }
                }

                int longestPrefix = 0;

                // Step 2: Check each number in array2 for the longest matching prefix
                foreach (int value in array2)
                {
                    int currentValue = value;
                    while (!array1Prefixes.Contains(currentValue) && currentValue > 0)
                    {
                        // Reduce currentValue by removing the last digit if not found in the prefix set
                        currentValue /= 10;
                    }
                    if (currentValue > 0)
                    {
                        // Length of the matched prefix using Math.Log10 to determine the number of digits
                        longestPrefix = Math.Max(
                            longestPrefix,
                            (int)Math.Log10(currentValue) + 1
                        );
                    }
                }

                return longestPrefix;
            }

            /* Approach 2: Trie
            Complexity Analysis
Let m be the length of arr1, n be the length of arr2.
•	Time Complexity: O(m⋅d+n⋅d)=O(m+n)
For each number in arr1, we insert it into the Trie by processing each digit. Since each number has up to d digits, inserting a single number takes O(d) time. Therefore, inserting all m numbers from arr1 into the Trie takes O(m⋅d) time.
For each number in arr2, we check how long its prefix matches with any prefix in the Trie. This involves traversing up to d digits of the number, which takes O(d) time per number. For all n numbers in arr2, the time complexity for this step is O(n⋅d).
Overall, the total time complexity is O(m⋅d+n⋅d)=O(m+n)
•	Space Complexity: O(m⋅d)=O(m)
Each node in the Trie represents a digit (0-9), and each number from arr1 can contribute up to d nodes. Thus, the total space used by the Trie for storing all prefixes is O(m⋅d).
The algorithm uses constant space for variables like longestPrefix and loop variables, which is negligible compared to the space used by the Trie.
Thus, the total space complexity is O(m⋅d)=O(m).

             */
            public int UsingTrie(int[] arr1, int[] arr2)
            {
                Trie trie = new Trie();

                // Step 1: Insert all numbers from arr1 into the Trie
                foreach (int num in arr1)
                {
                    trie.Insert(num);
                }

                int longestPrefix = 0;

                // Step 2: Find the longest prefix match for each number in arr2
                foreach (int num in arr2)
                {
                    int len = trie.FindLongestPrefix(num);
                    longestPrefix = Math.Max(longestPrefix, len);
                }

                return longestPrefix;
            }
            class TrieNode
            {

                // Each node has up to 10 possible children (digits 0-9)
                public TrieNode[] Children = new TrieNode[10];
            }

            class Trie
            {

                TrieNode root = new TrieNode();

                // Insert a number into the Trie by treating it as a string of digits
                public void Insert(int num)
                {
                    TrieNode node = root;
                    String numStr = num.ToString();
                    foreach (char digit in numStr)
                    {
                        int idx = digit - '0';
                        if (node.Children[idx] == null)
                        {
                            node.Children[idx] = new TrieNode();
                        }
                        node = node.Children[idx];
                    }
                }

                // Find the longest common prefix for a number in arr2 with the Trie
                public int FindLongestPrefix(int num)
                {
                    TrieNode node = root;
                    String numStr = num.ToString();
                    int len = 0;

                    foreach (char digit in numStr)
                    {
                        int idx = digit - '0';
                        if (node.Children[idx] != null)
                        {
                            // Increase length if the current digit matches
                            len++;
                            node = node.Children[idx];
                        }
                        else
                        {
                            // Stop if no match for the current digit
                            break;
                        }
                    }
                    return len;
                }
            }

        }












    }
}