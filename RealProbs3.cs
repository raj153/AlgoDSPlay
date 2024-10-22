using System.Diagnostics;
using System.Numerics;
using System.Text;

namespace AlgoDSPlay;

public partial class RealProbs
{

    /* 2285. Maximum Total Importance of Roads
    https://leetcode.com/problems/maximum-total-importance-of-roads/description/
     */
    class MaxTotalImportanceOfRoadsSol
    {

        /* Approach: Sorting
        Complexity Analysis
        Here, N is the number of nodes in the graph.
        •	Time complexity: O(N^2).
        We iterate over the edges list roads to find the degree of each node. In the worst case, the number of edges in the graph could reach N2, assuming an edge exists between every pair of nodes. Assigning degrees thus requires O(N^2) operations.
        Next, sorting the degrees in ascending order takes O(NlogN). Iterating through the degree array to calculate the total importance is an O(N) operation. Therefore, the overall time complexity remains O(N^2).
        •	Space complexity: O(N)
        We need an array of size N, degree, to keep the edge count of each node.
        Some additional space is required for sorting. The space complexity of the sorting algorithm depends on the programming language.
        o	In Python, the sort method sorts a list using the Tim Sort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space. Additionally, Tim Sort is designed to be a stable algorithm.
        o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting an array.
        o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
        Thus, the inbuilt sort() function might add up to O(log⁡⁡N) or O(N) to the space complexity.

         */
        public long maximumImportance(int n, int[][] roads)
        {
            long[] degree = new long[n];

            foreach (int[] edge in roads)
            {
                degree[edge[0]]++;
                degree[edge[1]]++;
            }

            Array.Sort(degree);

            long value = 1;
            long totalImportance = 0;
            foreach (long d in degree)
            {
                totalImportance += (value * d);
                value++;
            }

            return totalImportance;
        }
    }

    /* 386. Lexicographical Numbers
    https://leetcode.com/problems/lexicographical-numbers/description/
     */
    public class LexicalOrderSol
    {

        /* Approach 1: DFS Approach
        Complexity Analysis
        •	Time Complexity: O(n)
        The algorithm generates all numbers from 1 to n in lexicographical order. Each number is visited exactly once and added to the result list. The total number of operations is proportional to the number of elements generated, which is n.
        •	Space Complexity: O(log10(n))
        We only consider the recursion stack depth. The depth of recursion is proportional to the number of digits d in n. Given that the maximum value for n is 50,000, the maximum number of digits d is 5. Thus, the recursion stack depth and corresponding space complexity is O(d), which simplifies to O(log10(n)), but with a maximum constant value of 5 for practical constraints. It can also be argued as O(1). This is because, when substituting n as 50,000, the result is approximately 5 (specifically 4.698970004336), which is extremely small and does not significantly affect the overall complexity in this range.
        The space complexity analysis does not account for the result list itself, as the problem requires returning a list with n elements. Since we are only storing the elements in the list without performing additional operations on it, the space used by the list is not considered in the complexity analysis.

         */
        public List<int> UsingDFS(int n)
        {
            List<int> lexicographicalNumbers = new();
            // Start generating numbers from 1 to 9
            for (int start = 1; start <= 9; ++start)
            {
                GenerateLexicalNumbers(start, n, lexicographicalNumbers);
            }
            return lexicographicalNumbers;
        }

        private void GenerateLexicalNumbers(
            int currentNumber,
            int limit,
            List<int> result
        )
        {
            // If the current number exceeds the limit, stop recursion
            if (currentNumber > limit) return;

            // Add the current number to the result
            result.Add(currentNumber);

            // Try to append digits from 0 to 9 to the current number
            for (int nextDigit = 0; nextDigit <= 9; ++nextDigit)
            {
                int nextNumber = currentNumber * 10 + nextDigit;
                // If the next number is within the limit, continue recursion
                if (nextNumber <= limit)
                {
                    GenerateLexicalNumbers(nextNumber, limit, result);
                }
                else
                {
                    break; // No need to continue if nextNumber exceeds limit
                }
            }
        }

        /* Approach 2: Iterative Approach
        Complexity Analysis
•	Time Complexity: O(n)
The algorithm generates numbers in lexicographical order and iterates up to n times to populate the lexicographicalNumbers array. Each iteration involves constant-time operations (checking conditions and updating currentNumber). Thus, the time complexity is linear in terms of n.
•	Space Complexity: O(1)
The algorithm uses a constant amount of additional space for variables like currentNumber and loop counters. Therefore, the space complexity is O(1).
The space complexity analysis does not account for the result list itself, as the problem requires returning a list with n elements. Since we are only storing the elements in the list without performing additional operations on it, the space used by the list is not considered in the complexity analysis.

         */
        public List<int> UsingIterative(int n)
        {
            List<int> lexicographicalNumbers = new();
            int currentNumber = 1;

            // Generate numbers from 1 to n
            for (int i = 0; i < n; ++i)
            {
                lexicographicalNumbers.Add(currentNumber);

                // If multiplying the current number by 10 is within the limit, do it
                if (currentNumber * 10 <= n)
                {
                    currentNumber *= 10;
                }
                else
                {
                    // Adjust the current number by moving up one digit
                    while (currentNumber % 10 == 9 || currentNumber >= n)
                    {
                        currentNumber /= 10; // Remove the last digit
                    }
                    currentNumber += 1; // Increment the number
                }
            }

            return lexicographicalNumbers;
        }

    }


    /* 2976. Minimum Cost to Convert String I
    https://leetcode.com/problems/minimum-cost-to-convert-string-i/description/
     */
    class MinimumCostToConvertString1Sol
    {
        /* 
        Approach 1: Dijkstra's Algorithm
Complexity Analysis
Let n be the length of source and m be the length of the original array.
•	Time complexity: O(m+n)
Creating the adjacency list requires O(m) time as the algorithm loops over the contents of the original, changed, and cost array simultaneously.
In our algorithm, the number of vertices is 26 and the number of edges is m, which makes the time complexity of Dijkstra's algorithm O((26+m)log26). We call dijkstra for each of the 26 characters. Thus, the total time complexity is O(26⋅(26+m)log26), which can be simplified to O(m).
To calculate the totalCost, we iterate over the source string, which has a time complexity of O(n).
The total time complexity is the addition of all these elements, i.e., O(m)+O(n)=O(m+n).
•	Space complexity: O(m)
The adjacencyList stores all possible conversions, requiring a space complexity of O(m). minConversionCosts uses O(26×26) space, which simplifies to O(1).
The dijkstra method uses a priority queue that can store at most m elements in the worst case. The array minCosts has a fixed size of 26. Thus, the total space used by the method is O(m).
The total space required by the algorithm is O(m)+O(1)+O(m), which simplifies to O(m).

         */
        public long UsingDijkstraAlgo(
            string source,
            string target,
            char[] original,
            char[] changed,
            int[] cost
        )
        {
            // Create a graph representation of character conversions
            List<int[]>[] adjacencyList = new List<int[]>[26];
            for (int i = 0; i < 26; i++)
            {
                adjacencyList[i] = new List<int[]>();
            }

            // Populate the adjacency list with character conversions
            int conversionCount = original.Length;
            for (int i = 0; i < conversionCount; i++)
            {
                adjacencyList[original[i] - 'a'].Add(
                        new int[] { changed[i] - 'a', cost[i] }
                    );
            }

            // Calculate shortest paths for all possible character conversions
            long[][] minConversionCosts = new long[26][];
            for (int i = 0; i < 26; i++)
            {
                minConversionCosts[i] = Dijkstra(i, adjacencyList);
            }

            // Calculate the total cost of converting source to target
            long totalCost = 0;
            int stringLength = source.Length;
            for (int i = 0; i < stringLength; i++)
            {
                if (source[i] != target[i])
                {
                    long charConversionCost =
                        minConversionCosts[source[i] - 'a'][target[i] - 'a'];
                    if (charConversionCost == -1)
                    {
                        return -1; // Conversion not possible
                    }
                    totalCost += charConversionCost;
                }
            }
            return totalCost;
        }

        // Find minimum conversion costs from a starting character to all other characters
        private long[] Dijkstra(int startChar, List<int[]>[] adjacencyList)
        {
            // Priority queue to store characters with their conversion cost, sorted by cost
            var priorityQueue = new PriorityQueue<(long cost, int character), (long cost, int character)>(
                Comparer<(long cost, int)>.Create((e1, e2) => e1.cost.CompareTo(e2.cost)
            ));

            // Initialize the starting character with cost 0
            priorityQueue.Enqueue((0L, startChar), (0L, startChar));

            // Array to store the minimum conversion cost to each character
            long[] minCosts = new long[26];
            // Initialize all costs to -1 (unreachable)
            Array.Fill(minCosts, -1L);

            while (priorityQueue.Count > 0)
            {
                var currentPair = priorityQueue.Dequeue();
                long currentCost = currentPair.cost;
                int currentChar = currentPair.character;

                if (
                    minCosts[currentChar] != -1L &&
                    minCosts[currentChar] < currentCost
                ) continue;

                // Explore all possible conversions from the current character
                foreach (int[] conversion in adjacencyList[currentChar])
                {
                    int targetChar = conversion[0];
                    long conversionCost = conversion[1];
                    long newTotalCost = currentCost + conversionCost;

                    // If we found a cheaper conversion, update its cost
                    if (
                        minCosts[targetChar] == -1L ||
                        newTotalCost < minCosts[targetChar]
                    )
                    {
                        minCosts[targetChar] = newTotalCost;
                        // Add the updated conversion to the queue for further exploration
                        priorityQueue.Enqueue((newTotalCost, targetChar), (newTotalCost, targetChar));
                    }
                }
            }
            // Return the array of minimum conversion costs from the starting character to all others
            return minCosts;
        }
        /* Approach 2: Floyd-Warshall Algorithm 
Complexity Analysis
Let n be the length of source and m be the length of the original array.
•	Time complexity: O(m+n)
Populating minCosts with the initial conversion costs takes O(m) time.
Each of the three nested loops runs 26 times. Thus, the overall time taken is O(263)=O(1).
To calculate the totalCost, the algorithm loops over the source string, which takes linear time.
Thus, the time complexity of the algorithm is O(m)+O(1)+O(n), which simplifies to O(m+n).
•	Space complexity: O(1)
The minCost array has a fixed size of 26×26. We do not use any other data structures dependent on the length of the input space. Thus, the algorithm has a constant space complexity.

        */
        public long FloydWarshallAlgo(
            String source,
            String target,
            char[] original,
            char[] changed,
            int[] cost
        )
        {
            // Initialize result to store the total minimum cost
            long totalCost = 0;

            // Initialize a 2D array to store the minimum conversion cost
            // between any two characters
            long[][] minCost = new long[26][];
            foreach (long[] row in minCost)
            {
                Array.Fill(row, int.MaxValue);
            }

            // Fill the initial conversion costs from the given original,
            // changed, and cost arrays
            for (int i = 0; i < original.Length; ++i)
            {
                int startChar = original[i] - 'a';
                int endChar = changed[i] - 'a';
                minCost[startChar][endChar] = Math.Min(
                    minCost[startChar][endChar],
                    (long)cost[i]
                );
            }

            // Use Floyd-Warshall algorithm to find the shortest path between any
            // two characters
            for (int k = 0; k < 26; ++k)
            {
                for (int i = 0; i < 26; ++i)
                {
                    for (int j = 0; j < 26; ++j)
                    {
                        minCost[i][j] = Math.Min(
                            minCost[i][j],
                            minCost[i][k] + minCost[k][j]
                        );
                    }
                }
            }

            // Calculate the total minimum cost to transform the source string to
            // the target string
            for (int i = 0; i < source.Length; ++i)
            {
                if (source[i] == target[i])
                {
                    continue;
                }
                int sourceChar = source[i] - 'a';
                int targetChar = target[i] - 'a';

                // If the transformation is not possible, return -1
                if (minCost[sourceChar][targetChar] >= int.MaxValue)
                {
                    return -1;
                }
                totalCost += minCost[sourceChar][targetChar];
            }

            return totalCost;
        }
    }


    /* 1366. Rank Teams by Votes
    https://leetcode.com/problems/rank-teams-by-votes/description/
     */
    class RankTeamsByVotesSol
    {
        /* 

        Time and Space Complexity:
Time Complexity
The time complexity of the provided code is determined by several factors:
1.	Counting the rankings for each team involves iterating over all the votes and updating a list of size n, which is the number of teams. Each vote takes O(n) time to iterate, and this is done for all m votes. So, this part of the algorithm takes O(m * n) time.
2.	Sorting the teams according to their rank counts and in case of a tie, using alphabetic ordering. Python's sort function uses TimSort, which has a worst-case time complexity of O(n * log(n)). Since there are n teams, sorting them takes O(n * log(n)) time.
3.	Generating the final string involves creating a string from the sorted teams, which takes O(n) time.
Combining these factors, the overall time complexity is O(m * n + n * log(n) + n). Since n * log(n) is likely to be the dominant term as n grows in comparison to n, we can approximate the time complexity as O(m * n + n * log(n)).
Space Complexity
The space complexity of the code is determined by:
1.	The space used by the cnt dictionary, which contains a list of counters, of size n, for each distinct team. Since there are n teams, the total size of the cnt dictionary is O(n^2).
2.	The space used by the sorting function could be O(n) in the worst case for the internal working storage during the sort.
Taking these into account, the overall space complexity of the algorithm is O(n^2).
 */
        public String RankTeamsByVotes(string[] votes)
        {
            // The number of teams is determined by the length of a single vote string
            int numTeams = votes[0].Length;

            // Create a 2D array to count the position based votes for each team (A-Z mapped to 0-25)
            int[][] count = new int[26][];
            foreach (String vote in votes)
            {
                for (int i = 0; i < numTeams; ++i)
                {
                    // Increment the vote count for the team at the current position
                    count[vote[i] - 'A'][i]++;
                }
            }

            // Create an array of Characters representing each team in the initial vote order
            char[] teams = new char[numTeams];
            for (int i = 0; i < numTeams; ++i)
            {
                teams[i] = votes[0][i];
            }

            // Sort the array of teams based on the vote counts and then by alphabetical order
            Array.Sort(teams, ((a, b) =>
            {

                int indexA = a - 'A', indexB = b - 'A';
                for (int k = 0; k < numTeams; ++k)
                {
                    // Compare the vote count for the current position
                    int difference = count[indexA][k] - count[indexB][k];
                    if (difference != 0)
                    {
                        // If there's a difference, return the comparison result
                        return difference > 0 ? -1 : 1;
                    }
                }
                // If all vote counts are equal, sort by alphabetical order
                return a - b;

            }));

            // Build the final ranking string based on the sorted array of teams
            StringBuilder result = new StringBuilder();
            foreach (char team in teams)
            {
                result.Append(team);
            }
            return result.ToString();
        }
    }


    /* 648. Replace Words
    https://leetcode.com/problems/replace-words/description/
     */

    class ReplaceWordsSol
    {

        /* Approach 1: Hash Set
Complexity Analysis
Let d be the number of words in the dictionary, s be the number of words in the sentence, and w be the average length of each word.
•	Time complexity: O(d⋅w+s⋅w^2)
Creating a set from the dictionary takes O(d⋅w). Creating the data structure that stores the words in the sentence takes O(s⋅w).
The loop in the helper function runs once for each letter in the word. Building each substring takes the helper function O(w). Hash set lookups take O(1) in the average case. Therefore, the time complexity of the helper function is O(w^2).
The main loop calls the helper function once for each word in the sentence, so it takes O(s⋅w^2).
Converting the result to a string takes O(s⋅w).
Therefore, the overall time complexity is O(d⋅w+s⋅w^2)
•	Space complexity: O(d⋅w+s⋅w)
The set that stores the dictionary requires O(d⋅w) space. The data structure that stores the words in the sentence uses O(s⋅w) space.

         */
        public String UsingHashSet(List<String> dictionary, String sentence)
        {
            String[] wordArray = sentence.Split(" ");
            HashSet<String> dictSet = new(dictionary);

            // Replace each word in sentence with the corresponding shortest root
            for (int i = 0; i < wordArray.Length; i++)
            {
                wordArray[i] = ShortestRoot(wordArray[i], dictSet);
            }

            return String.Join(" ", wordArray);
        }

        private String ShortestRoot(String word, HashSet<String> dictSet)
        {
            // Find the shortest root of the word in the dictionary
            for (int i = 1; i <= word.Length; i++)
            {
                String root = word.Substring(0, i);
                if (dictSet.Contains(root))
                {
                    return root;
                }
            }
            // There is not a corresponding root in the dictionary
            return word;
        }
        /* Approach 2: Prefix Trie
Complexity Analysis
Let d be the number of words in the dictionary, s be the number of words in the sentence, and w be the average length of each word.
•	Time complexity: O(d⋅w+s⋅w)
Creating the Trie takes O(d⋅w). Creating the data structure that stores the words in the sentence takes O(s⋅w).
The loop in the shortestRoot function runs once for each letter in the word. If a corresponding prefix is found, it creates one substring, which takes O(w). Therefore, the time complexity of finding the shortest root is O(w).
The main loop calls the shortestRoot function once for each word in the sentence, so it takes O(s⋅w).
Converting the result to a string takes O(s⋅w).
Therefore, the overall time complexity is O(d⋅w+2⋅s⋅w), which we can simplify to O(d⋅w+s⋅w).
•	Space complexity: O(d⋅w+s⋅w)
The Trie may store up to O(d⋅w) nodes, and each node stores an array with 26 pointers, so the Trie requires O(d⋅w⋅26) space. 26 is a constant factor, so we can simplify this to O(d⋅w). The data structure that stores the words in the sentence uses O(s⋅w) space.
Note: Though the space complexity looks similar to the above approach, this approach will usually require less space because when any words have the same prefix, it stores the prefix only once, while the hash set stores words like "semicircle" and "semitruck" separately.

         */
        public String UsingPrefixTrie(List<String> dictionary, String sentence)
        {
            String[] wordArray = sentence.Split(" ");

            Trie dictTrie = new Trie();
            foreach (String word in dictionary)
            {
                dictTrie.Insert(word);
            }

            // Replace each word in the sentence with the corresponding shortest root
            for (int word = 0; word < wordArray.Length; word++)
            {
                wordArray[word] = dictTrie.ShortestRoot(wordArray[word]);
            }

            return String.Join(" ", wordArray);
        }
        public class TrieNode
        {

            public bool isEnd;
            public TrieNode[] children;

            public TrieNode()
            {
                isEnd = false;
                children = new TrieNode[26];
            }


        }
        class Trie
        {

            private TrieNode root;

            public Trie()
            {
                root = new TrieNode();
            }

            public void Insert(String word)
            {
                TrieNode current = root;
                foreach (char c in word)
                {
                    if (current.children[c - 'a'] == null)
                    {
                        current.children[c - 'a'] = new TrieNode();
                    }
                    current = current.children[c - 'a'];
                }
                current.isEnd = true;
            }

            // Find the shortest root of the word in the trie
            public String ShortestRoot(String word)
            {
                TrieNode current = root;
                for (int i = 0; i < word.Length; i++)
                {
                    char c = word[i];
                    if (current.children[c - 'a'] == null)
                    {
                        // There is not a corresponding root in the trie
                        return word;
                    }
                    current = current.children[c - 'a'];
                    if (current.isEnd)
                    {
                        return word.Substring(0, i + 1);
                    }
                }
                // There is not a corresponding root in the trie
                return word;
            }
        }



    }


    /* 1992. Find All Groups of Farmland 
    https://leetcode.com/problems/find-all-groups-of-farmland/description/
     */
    public class FindFarmlandSol
    {
        // The four directions in which traversal will be done.
        int[][] dirs = { new int[] { -1, 0 }, new int[] { 0, -1 }, new int[] { 0, 1 }, new int[] { 1, 0 } };
        // Global variables with 0 value initially.
        int row2, col2;

        /* 
        Approach 1: Depth-First Search
        Complexity Analysis
        Here, M is the number of rows in the matrix and N is the number of columns in the matrix.
        •	Time complexity: O(M⋅N)
        We will iterate over each cell in the matrix at most once because we used the visited array to prevent re-processing cells. All other helper functions like isWithinFarm are O(1). Hence, the total time complexity is O(M⋅N).
        •	Space complexity: O(M⋅N)
        The array visited is of size M⋅N; also, there will be stack space consumed by DFS that will be equal to the maximum number of active stack calls, which will be equal to M∗N if all cells are 1 in the matrix. Apart from this, there is also array ans, but the space used to store the result isn't considered part of space complexity. Hence, the total space complexity is O(M⋅N).
         */
        public int[][] UsingDFS(int[][] land)
        {
            bool[][] visited = new bool[land.Length][];
            List<int[]> ans = new List<int[]>();

            for (int row1 = 0; row1 < land.Length; row1++)
            {
                for (int col1 = 0; col1 < land[0].Length; col1++)
                {
                    if (land[row1][col1] == 1 && !visited[row1][col1])
                    {
                        row2 = 0; col2 = 0;

                        DFS(land, visited, row1, col1);

                        int[] arr = new int[] { row1, col1, row2, col2 };
                        ans.Add(arr);
                    }
                }
            }

            return ans.ToArray();
        }
        // Returns true if the coordinate is within the boundary of the matrix.
        private bool IsWithinFarm(int x, int y, int N, int M)
        {
            return x >= 0 && x < N && y >= 0 && y < M;
        }

        private void DFS(int[][] land, bool[][] visited, int x, int y)
        {
            visited[x][y] = true;
            // Maximum x and y for the bottom right cell.
            row2 = Math.Max(row2, x); col2 = Math.Max(col2, y);

            foreach (int[] dir in dirs)
            {
                // Neighbor cell coordinates.
                int newX = x + dir[0], newY = y + dir[1];

                // If the neighbor is within the matrix and is a farmland cell and is not visited yet.
                if (IsWithinFarm(newX, newY, land.Length, land[0].Length) && !visited[newX][newY]
                        && land[newX][newY] == 1)
                {
                    DFS(land, visited, newX, newY);
                }
            }
        }

        /* Approach 2: Breadth-First Search
Complexity Analysis
Here, M is the number of rows in the matrix and N is the number of columns in the matrix.
•	Time complexity: O(M⋅N)
We will iterate over each cell in the matrix at most once because of the visited array. All other helper functions like isWithinFarm are O(1). Hence, the total time complexity is O(M⋅N).
•	Space complexity: O(M⋅N)
The array visited is of size M⋅N, also there will be space consumed by the queue that can be equal to M∗N if all cells are 1 in the matrix. Apart from this, there is also array ans, but the space used to store the result isn't considered as part of the space complexity. Hence, the total space complexity is O(M⋅N).

         */
        public int[][] UsingBFS(int[][] land)
        {
            bool[][] visited = new bool[land.Length][];
            for (int i = 0; i < land.Length; i++)
            {
                visited[i] = new bool[land[0].Length];
            }
            List<int[]> result = new List<int[]>();

            for (int row = 0; row < land.Length; row++)
            {
                for (int col = 0; col < land[0].Length; col++)
                {
                    if (land[row][col] == 1 && !visited[row][col])
                    {
                        Queue<ValueTuple<int, int>> queue = new Queue<ValueTuple<int, int>>();

                        queue.Enqueue((row, col));
                        visited[row][col] = true;

                        ValueTuple<int, int> last = BFS(queue, land, visited);

                        int[] array = new int[] { row, col, last.Item1, last.Item2 };
                        result.Add(array);
                    }
                }
            }

            return result.ToArray();
        }
        private ValueTuple<int, int> BFS(Queue<ValueTuple<int, int>> queue, int[][] land, bool[][] visited)
        {
            ValueTuple<int, int> current = (0, 0);

            while (queue.Count > 0)
            {
                current = queue.Dequeue();

                int x = current.Item1;
                int y = current.Item2;

                foreach (int[] direction in dirs)
                {
                    // Neighbor cell coordinates.
                    int newX = x + direction[0], newY = y + direction[1];

                    // If the neighbor is within the matrix and is a farmland cell and not visited yet.
                    if (IsWithinFarm(newX, newY, land.Length, land[0].Length) && !visited[newX][newY] && land[newX][newY] == 1)
                    {
                        visited[newX][newY] = true;
                        queue.Enqueue((newX, newY));
                    }
                }
            }

            return current;
        }
        /* Approach 3: Greedy
Complexity Analysis
Here, M is the number of rows in the matrix and N is the number of columns in the matrix.
•	Time complexity: O(M⋅N)
We will iterate over each cell in the matrix at most once because we mark the visited cells in the land array. Hence, the total time complexity is O(M⋅N).
•	Space complexity: O(1)
The only space required is ans but the space used to store the result isn't considered as part of space complexity. Hence, the total space complexity is constant.

         */
        public int[][] UsingGreedy(int[][] land)
        {
            int N = land.Length, M = land[0].Length;
            List<int[]> ans = new();

            for (int row1 = 0; row1 < N; row1++)
            {
                for (int col1 = 0; col1 < M; col1++)
                {
                    if (land[row1][col1] == 1)
                    {
                        int x = row1, y = col1;

                        for (x = row1; x < N && land[x][col1] == 1; x++)
                        {
                            for (y = col1; y < M && land[x][y] == 1; y++)
                            {
                                land[x][y] = 0;
                            }
                        }

                        int[] arr = new int[] { row1, col1, x - 1, y - 1 };
                        ans.Add(arr);
                    }
                }
            }
            return ans.ToArray();
        }


    }

    /* 846. Hand of Straights
    https://leetcode.com/problems/hand-of-straights/description/
     */
    class IsNStraightHandSol
    {


        /* Approach 1: Using Map
        Complexity Analysis
        Let n be the size of the hand array and k be groupSize.
        •	Time complexity: O(n⋅logn+n⋅k)
        Populating the cardCount map takes O(nlogn) time.
        The outer loop processes the cardCount map until it is empty. In the worst case, it iterates n times.
        Inside the outer loop, getting the smallest card value from the cardCount map takes O(logn) time due to the map implementation.
        Checking for the presence of a consecutive sequence of k cards takes O(k) time. k is limited to the size of the hand array because we can't have groups larger than the hand.
        Each card will be processed exactly once because the more cards we process in each group, the fewer groups we process. Processing each card can take up to O(logn) due to the map or heap insertion and removal.
        Therefore, the overall time complexity is O(nlogn+n⋅k).
        •	Space complexity: O(n)
        The cardCount map stores the count of each card value.
        In the worst case, all cards could have distinct values, resulting in a map size of n.
        Therefore, the space complexity is O(n).	

         */
        public bool IsNStraightHand(int[] hand, int groupSize)
        {
            int handSize = hand.Length;
            if (handSize % groupSize != 0)
            {
                return false;
            }

            // TreeMap to store the count of each card value
            SortedDictionary<int, int> cardCount = new();
            for (int i = 0; i < handSize; i++)
            {
                cardCount[hand[i]] = cardCount.GetValueOrDefault(hand[i], 0) + 1;
            }

            // Process the cards until the map is empty
            while (cardCount.Count > 0)
            {
                // Get the smallest card value
                int currentCard = int.MaxValue;
                foreach (var card in cardCount.Keys)
                {
                    if (card < currentCard)
                    {
                        currentCard = card;
                    }
                }
                // Check each consecutive sequence of groupSize cards
                for (int i = 0; i < groupSize; i++)
                {
                    // If a card is missing or has exhausted its occurrences
                    if (!cardCount.ContainsKey(currentCard + i)) return false;
                    cardCount[currentCard + i] = cardCount[currentCard + i] - 1;

                    // Remove the card value if its occurrences are exhausted
                    if (cardCount[currentCard + i] == 0) cardCount.Remove(
                        currentCard + i
                    );
                }
            }

            return true;
        }
        /* Approach 2: Optimal
        Complexity Analysis
Let n be the size of the hand array and k be groupSize.
•	Time complexity: O(nlogn+n)
The time complexity is O(nlogn+n). This is due to the process of counting and sorting the cards. In C++ and Java, the time complexity is O(n \log n).
•	Space complexity: O(n)
We use a map to count the occurrences of each card and a deque to keep track of the number of open groups. Therefore, the space complexity is O(n).

        */
        public bool UsingSortedDictOptimal(int[] hand, int groupSize)
        {
            // Map to store the count of each card value
            SortedDictionary<int, int> cardCount = new SortedDictionary<int, int>();

            foreach (int card in hand)
            {
                if (cardCount.ContainsKey(card))
                {
                    cardCount[card]++;
                }
                else
                {
                    cardCount[card] = 1;
                }
            }

            // Queue to keep track of the number of new groups
            // starting with each card value
            Queue<int> groupStartQueue = new Queue<int>();
            int lastCard = -1, currentOpenGroups = 0;

            foreach (KeyValuePair<int, int> entry in cardCount)
            {
                int currentCard = entry.Key;

                // Check if there are any discrepancies in the sequence
                // or more groups are required than available cards
                if ((currentOpenGroups > 0 && currentCard > lastCard + 1) ||
                    currentOpenGroups > cardCount[currentCard])
                {
                    return false;
                }

                // Calculate the number of new groups starting with the current card
                groupStartQueue.Enqueue(cardCount[currentCard] - currentOpenGroups);
                lastCard = currentCard;
                currentOpenGroups = cardCount[currentCard];

                // Maintain the queue size to be equal to groupSize
                if (groupStartQueue.Count == groupSize)
                {
                    currentOpenGroups -= groupStartQueue.Dequeue();
                }
            }

            // All groups should be completed by the end
            return currentOpenGroups == 0;
        }

        /* Approach 3: Reverse Decrement (Most Optimal)
Complexity Analysis
Let n be the size of the hand array and k be groupSize.
•	Time complexity: O(n)
Populating the cardCount map takes O(n) time, where n is the length of the hand array.
The outer loop iterates over all cards in the hand array, which takes O(n) time.
For each card card, the algorithm might need to check for the presence of k consecutive cards, which takes O(k) time in the worst case.
Given that the maximum number of cards we need to check consecutively is bounded by the size of the hand, the inner loop does not run k times for each card independently. Instead, it runs k times in total for each sequence of groups.
So, the algorithm forms n/k groups, each of size k. Also, k is limited to the size of the hand array because we can't have groups larger than the hand.
Thus, the inner loop effectively runs n times in total across all iterations of the outer loop, as each of the n cards is processed exactly once within a group.
Therefore, the overall time complexity is O(n):
It's important to note that this O(n) complexity holds because the inner loop, despite appearing nested, does not result in a quadratic increase in iterations but rather spreads the iterations across the total number of cards.
This approach might seem expensive at first glance. If we happen to select a card at the end of a long streak, we'll decrement all the way through the entire streak just to find a single start. However, this is worthwhile because we can then go back up through the streak, deleting it entirely. Overall, we might "visit" each card twice, once on the way down and once on the way up, resulting in O(2n)=O(n) time complexity.
Although it might seem like we go through all the cards and do a lot of work for each card, leading to an O(n2) time complexity, this is not the case. The amount of work we do for each card is proportional to how much we "uncount," and overall, we can't "uncount" more cards than were originally present, which is n. So, the overall time complexity is O(n). For example, perhaps the first number causes us to do O(n) work, "uncounting" every card. But for all other cards, we do essentially nothing (only O(1) work for each).
Thus, the overall complexity is approximately 2n, which simplifies to O(n).
•	Space complexity: O(n)
The cardCount map stores the count of each card value.
In the worst case, all cards could have distinct values, resulting in a map size of n.
Therefore, the space complexity is O(n).

         */
        public bool UsingReverseIncrement(int[] hand, int groupSize)
        {
            if (hand.Length % groupSize != 0)
            {
                return false;
            }

            // HashMap to store the count of each card value
            Dictionary<int, int> cardCount = new();
            foreach (int card in hand)
            {
                int count = cardCount.GetValueOrDefault(card, 0);
                cardCount[card] = count + 1;
            }

            foreach (int card in hand)
            {
                int startCard = card;
                // Find the start of the potential straight sequence
                while (cardCount.GetValueOrDefault(startCard - 1, 0) > 0)
                {
                    startCard--;
                }

                // Process the sequence starting from startCard
                while (startCard <= card)
                {
                    while (cardCount.GetValueOrDefault(startCard, 0) > 0)
                    {
                        // Check if we can form a consecutive sequence
                        // of groupSize cards
                        for (
                            int nextCard = startCard;
                            nextCard < startCard + groupSize;
                            nextCard++
                        )
                        {
                            if (cardCount.GetValueOrDefault(nextCard, 0) == 0)
                            {
                                return false;
                            }
                            cardCount[nextCard] = cardCount[nextCard] - 1;
                        }
                    }
                    startCard++;
                }
            }

            return true;
        }
    }


    /* 3186. Maximum Total Damage With Spell Casting
    https://leetcode.com/problems/maximum-total-damage-with-spell-casting/description/
     */
    class MaximumTotalDamageSol
    {
        public long MaximumTotalDamage(int[] power)
        {
            // Step 1: Count the frequency of each damage value
            Dictionary<int, long> damageFrequency = new();
            foreach (int damage in power)
            {
                damageFrequency[damage] = damageFrequency.GetValueOrDefault(damage, 0L) + 1;
            }

            // Step 2: Extract and sort the unique damage values
            List<int> uniqueDamages = new List<int>(damageFrequency.Keys);
            uniqueDamages.Sort();

            int totalUniqueDamages = uniqueDamages.Count;
            long[] maxDamageDP = new long[totalUniqueDamages];

            // Step 3: Initialize the DP array with the first unique damage
            maxDamageDP[0] = uniqueDamages[0] * damageFrequency[uniqueDamages[0]];

            // Step 4: Fill the DP array with the maximum damage calculations
            for (int i = 1; i < totalUniqueDamages; i++)
            {
                int currentDamageValue = uniqueDamages[i];
                long currentDamageTotal = currentDamageValue * damageFrequency[currentDamageValue];

                // Initially, consider not taking the current damage
                maxDamageDP[i] = maxDamageDP[i - 1];

                // Find the previous damage value that doesn't conflict with the current one
                int previousIndex = i - 1;
                while (previousIndex >= 0 &&
                       (uniqueDamages[previousIndex] == currentDamageValue - 1 ||
                        uniqueDamages[previousIndex] == currentDamageValue - 2 ||
                        uniqueDamages[previousIndex] == currentDamageValue + 1 ||
                        uniqueDamages[previousIndex] == currentDamageValue + 2))
                {
                    previousIndex--;
                }

                // Update the DP value considering the current damage
                if (previousIndex >= 0)
                {
                    maxDamageDP[i] = Math.Max(maxDamageDP[i], maxDamageDP[previousIndex] + currentDamageTotal);
                }
                else
                {
                    maxDamageDP[i] = Math.Max(maxDamageDP[i], currentDamageTotal);
                }
            }

            // Return the maximum damage possible
            return maxDamageDP[totalUniqueDamages - 1];
        }
    }

    /* 1492. The kth Factor of n
    https://leetcode.com/problems/the-kth-factor-of-n/description/
    https://algo.monster/liteproblems/1492
     */
    class KthFactorOfNSol
    {
        /* Time and Space Complexity
The time complexity of the given code can be assessed by examining the two while loops that are run in sequence to find the k-th factor of the integer n.
The first loop runs while i * i < n, which means it will run approximately sqrt(n) times, because it stops when i is just less than the square root of n. Within this loop, the operation performed is a modulo operation to check if i is a factor of n, which is an O(1) operation. Therefore, the time complexity contributed by the first loop is O(sqrt(n)).
The second loop starts with i set to a value slightly less than sqrt(n) (assuming n is not a perfect square) and counts down to 1. For each iteration, it performs a modulo operation, which is O(1). However, not every i will lead to an iteration because the counter is reduced only when (n % (n // i)) == 0, which corresponds to the outer factors of n. Since there are as many factors less than sqrt(n) as there are greater than sqrt(n), we can expect the second loop also to contribute a time complexity of O(sqrt(n)).
Combining both loops, the overall time complexity remains O(sqrt(n)), as they do not compound on each other but are sequential.
The space complexity of the code is O(1) as there are only a finite number of variables used (i, k, n), and no additional space is allocated that would grow with the input size.
 */
        // Method to find the k-th factor of a number n
        public int KthFactor(int n, int k)
        {
            // Starting from 1, trying to find factors in increasing order
            int factor = 1;
            for (; factor <= n / factor; ++factor)
            {
                // If 'factor' is a factor of 'n' and it's the k-th one found
                if (n % factor == 0 && (--k == 0))
                {
                    // Return 'factor' as the k-th factor of 'n'
                    return factor;
                }
            }
            // Adjust 'factor' if we've surpassed the square root of 'n'
            // because we will look for factors in the opposite direction now
            if (factor * factor != n)
            {
                factor--;
            }
            // Starting from the last found factor, searching in decreasing order
            for (; factor > 0; --factor)
            {
                // Calculate the corresponding factor pair
                if (n % (n / factor) == 0)
                {
                    // Decrease k for each factor found
                    k--;
                    // If we found the k-th factor from the largest end
                    if (k == 0)
                    {
                        // Return the factor as it's the k-th factor of 'n'
                        return n / factor;
                    }
                }
            }
            // If no k-th factor is found, return -1
            return -1;
        }
    }


    /* 1884. Egg Drop With 2 Eggs and N Floors
    https://leetcode.com/problems/egg-drop-with-2-eggs-and-n-floors/description/
    https://algo.monster/liteproblems/1884    
     */
    public class TwoEggDropWith2EggsAndNFloorsSol
    {
        public int TwoEggDrop(int n)
        {
            int[][] dp = new int[3][];
            for (int i = 0; i < 3; i++)
            {
                dp[i] = new int[n + 1];
            }

            for (int i = 0; i <= n; i++)
            {
                dp[0][i] = 0;
                dp[1][i] = i;
                dp[2][i] = -1;
            }

            for (int i = 1; i <= n; i++)
            {
                int minMoves = int.MaxValue;
                for (int j = 1; j < i; j++)
                {
                    int moves = Math.Max(dp[0][j - 1], dp[1][i - j]) + 1;
                    minMoves = Math.Min(minMoves, moves);
                }
                dp[2][i] = minMoves;

                if (dp[1][i] == dp[2][i])
                {
                    break;
                }

                dp[1][i] = dp[2][i];
            }

            return dp[2][n];
        }
    }


    /* 781. Rabbits in Forest
    https://leetcode.com/problems/rabbits-in-forest/description/
    https://algo.monster/liteproblems/781
     */

    class NumRabbitsSol
    {
        /* 
                Time Complexity
        The function numRabbits loops once through the answers array to create a counter, which is essentially a histogram of the answers.The time complexity of creating this counter is O(n), where n is the number of elements in answers.
        After that, it iterates over the items in the counter and performs a constant number of arithmetic operations for each distinct answer, in addition to calling the math.ceil function. Since the number of distinct answers is at most n, the time taken for this part is also O(n).
        Therefore, the overall time complexity of the function is O(n).
        Space Complexity
        The main extra space used by this function is the counter, which in the worst case stores a count for each unique answer.In the worst case, every rabbit has a different answer, so the space complexity would also be O(n).
        Hence, the space complexity of the function is also O(n).
         */
        // Function to calculate the minimum probable number of rabbits in the forest
        public int NumRabbits(int[] answers)
        {
            // Create a map to count the frequency of each answer
            Dictionary<int, int> frequencyMap = new();
            // Iterate over the array of answers given by the rabbits
            foreach (int answer in answers)
            {
                // Update the frequency of this particular answer
                frequencyMap[answer] = frequencyMap.GetValueOrDefault(answer, 0) + 1;
            }

            // Initialize the result variable to store the total number of rabbits
            int totalRabbits = 0;
            // Iterate over the entries in the map to calculate the total number of rabbits
            foreach (var entry in frequencyMap)
            {
                // key is the number of other rabbits the current rabbit claims exist
                int otherRabbits = entry.Key;
                // value is the frequency of the above claim from the array of answers
                int frequencyOfClaim = entry.Value;

                // Calculate the number of groups of rabbits with the same claim
                int groupsOfRabbits = (int)Math.Ceiling(frequencyOfClaim / ((otherRabbits + 1) * 1.0));
                // Add the total number of rabbits in these groups to the result
                totalRabbits += groupsOfRabbits * (otherRabbits + 1);
            }

            // Return the total number of rabbits calculated
            return totalRabbits;
        }
    }


    /* 2013. Detect Squares
    https://leetcode.com/problems/detect-squares/description/
    https://algo.monster/liteproblems/2013
     */
    public class DetectSquaresSol
    {
        /* Time and Space Complexity
Time Complexity
•	__init__ method: The initialization of the DetectSquares class sets up a default dictionary with a Counter as its default value. The time complexity for this operation is O(1) because it's a simple assignment operation.
•	add method: Adds a point with coordinates (x, y). This part has a time complexity of O(1) because it's just incrementing the count of the point (x, y) in a hash table, which is an O(1) operation.
•	count method: This calculates the number of squares with one corner at the given point. The method iterates through all unique x coordinates stored in the hash map (the keys of the cnt dictionary). For each x2 other than x1, it calculates two potential square sides (d = x2 - x1 for the side to the right of (x1, y1) and d = x1 - x2 for the side to the left of (x1, y1)). It then checks for points that are d distance away on the y-axis both above and below (x1,y1).
o	For a given point, the number of potential squares is computed by considering each different x2 coordinate as a potential corner. If there are n different x coordinates, there could be up to n - 1 iterations (discounting the current x coordinate).
o	Inside each iteration, the calculation ans += ... involves constant-time dictionary lookups and arithmetic operations. So each iteration can be considered O(1).
o	Therefore, the time complexity for this method will be O(n), where n is the number of unique x coordinates stored so far when count is called.
The time complexity of the add method is insignificant compared to the count method which dominates when analyzing the complexity of the class operations. Hence, the overall time complexity hinges on the count method which is O(n) in the worst case.
Space Complexity
•	The space complexity of the DetectSquares class is dominated by the self.cnt dictionary, which stores the counts of points. This dictionary can potentially grow to include every unique point seen so far.
•	If the total number of points is p, and there are u unique x-coordinates and v unique y-coordinates amongst them, the maximum storage required would be for a complete mapping of these points, which is O(u * v).
•	In the worst-case scenario where every added point has a unique (x, y) combination, the space complexity would be O(p), where p is the number of add operations performed.
Therefore, the space complexity of the DetectSquares class is O(p).
 */

        // Using a dictionary of dictionaries to store the count of points. The outer dictionary's key is the x-coordinate,
        // and the value is another dictionary where the key is the y-coordinate and the value is the count of points.
        private Dictionary<int, Dictionary<int, int>> pointCounts = new Dictionary<int, Dictionary<int, int>>();

        public DetectSquaresSol()
        {
            // Constructor doesn't need to do anything as the pointCounts dictionary is initialized on declaration.
        }

        // Adds a new point to our data structure.
        public void Add(int[] point)
        {
            int x = point[0];
            int y = point[1];
            // If x is not already a key in pointCounts, initialize it with an empty Dictionary.
            // Increase the count of the (x, y) point by 1, summing with the current count if it already exists.
            if (!pointCounts.ContainsKey(x))
            {
                pointCounts[x] = new Dictionary<int, int>();
            }
            if (pointCounts[x].ContainsKey(y))
            {
                pointCounts[x][y]++;
            }
            else
            {
                pointCounts[x][y] = 1;
            }
        }

        // Counts how many ways there are to form a square with one vertex being the given point.
        public int Count(int[] point)
        {
            int x1 = point[0];
            int y1 = point[1];

            // If there are no points with the same x-coordinate as point, return 0 since no square is possible.
            if (!pointCounts.ContainsKey(x1))
            {
                return 0;
            }

            int totalSquares = 0;

            // Loop over all entries in the pointCounts.
            foreach (var entry in pointCounts)
            {
                int x2 = entry.Key;
                // We're not interested in points with the same x-coordinate as the given point (as we need a square).
                if (x2 != x1)
                {
                    int d = x2 - x1; // Calculate the potential side length of the square.

                    // Retrieve the counts dictionaries for x1 and x2.
                    var yCountsForX1 = pointCounts[x1];
                    var yCountsForX2 = entry.Value;

                    // Calculate the number of squares that can be formed with the upper and lower horizontal points.
                    totalSquares += yCountsForX2.GetValueOrDefault(y1, 0) * yCountsForX1.GetValueOrDefault(y1 + d, 0) * yCountsForX2.GetValueOrDefault(y1 + d, 0);
                    totalSquares += yCountsForX2.GetValueOrDefault(y1, 0) * yCountsForX1.GetValueOrDefault(y1 - d, 0) * yCountsForX2.GetValueOrDefault(y1 - d, 0);
                }
            }
            // Return the total number of squares found.
            return totalSquares;
        }
    }

    /* 1258. Synonymous Sentences
    https://leetcode.com/problems/synonymous-sentences/description/
    https://algo.monster/liteproblems/1258
     */
    class SynonymousSentencesSol
    {
        private List<String> answers = new();  // List of all possible sentences
        private List<String> currentSentence = new();  // Holds the current sentence during DFS
        private List<String> uniqueWords;  // List of unique words across all synonyms
        private Dictionary<string, int> wordIdMap;  // Maps word to its index
        private UnionFind unionFind; // Union-Find instance for grouping synonyms
        private List<int>[] synonymGroups;  // Holds groups of synonym indices
        private String[] originalSentence;  // Original sentence split by whitespace

        /* Time and Space Complexity
        Time Complexity
        •	The initialization of the UnionFind class takes O(N) time, where N is the number of unique words in synonyms.
        •	The find function has a time complexity of O(alpha(N)) per call due to path compression, where alpha is the inverse Ackermann function, which is a very slow-growing function and can be considered almost constant for all practical purposes.
        •	The union function is called for each pair of synonyms, thus taking O(M * alpha(N)) time in total, where M is the number of synonym pairs.
        •	Building the dictionary d requires O(N) time.
        •	Constructing g involves iterating over all words and finding their root, requiring O(N * alpha(N)) time.
        •	Sorting the lists in g requires O(K * log(K)) time for each list, where K is the maximum number of synonyms for a word (in the worst case K = N when all words are synonyms of each other), hence in total O(N * log(N)).
        •	The dfs function is the most expensive one. It generates all combinations of synonyms, which could be O(2^L) in the worst-case scenario, where L is the length of the sentence. Each recursive call to dfs may take up to O(L) time to join and append words to form a sentence, thus the dfs has a potential time complexity of O(2^L * L).
        The final time complexity is the sum of all these operations, which is dominated by the dfs function, so the overall time complexity is O(N + M * alpha(N) + N * alpha(N) + N * log(N) + 2^L * L).
        Space Complexity
        •	The space complexity for the UnionFind data structure is O(N) due to storing parents for each node and their respective sizes.
        •	Additional O(N) space for storing words and the dictionary d.
        •	The recursive dfs function could also go up to O(L) calls deep due to recursion stack, where L is the length of the sentence.
        •	The ans list may contain up to O(2^L) sentences, thus requiring O(2^L * L) space to store them when each sentence is O(L) words long.
        Therefore, the space complexity is O(N + 2^L * L) considering the depths of recursion and the potential number of sentences generated.
         */

        // Main method to generate all possible sentences given a list of synonyms and a text string

        public List<String> GenerateSentences(List<List<String>> synonyms, String text)
        {
            HashSet<String> setOfSynonyms = new();  // Set to ensure unique words are collected
            foreach (List<String> pairs in synonyms)
            {
                foreach (var p in pairs)
                {
                    setOfSynonyms.Add(p);
                }
            }
            uniqueWords = new(setOfSynonyms);
            int wordCount = uniqueWords.Count;
            wordIdMap = new(wordCount);

            // Populate the wordIdMap with each unique word's index
            for (int i = 0; i < uniqueWords.Count; ++i)
            {
                wordIdMap[uniqueWords[i]] = i;
            }

            unionFind = new UnionFind(wordCount);
            // Perform union operations for all pairs of synonyms
            foreach (List<String> pairs in synonyms)
            {
                unionFind.Union(wordIdMap[pairs[0]], wordIdMap[pairs[1]]);
            }

            // Initialize synonym groups
            synonymGroups = new List<int>[wordCount];
            for (int i = 0; i < synonymGroups.Length; i++)
            {
                synonymGroups[i] = new List<int>();
            }
            for (int i = 0; i < wordCount; ++i)
            {
                synonymGroups[unionFind.Find(i)].Add(i);
            }
            // Sort groups alphabetically based on the represented words
            for (var idx = 0; idx < synonymGroups.Length; idx++)
            {
                synonymGroups[idx].Sort((i, j) => uniqueWords[i].CompareTo(uniqueWords[j]));

            }

            // Split the original text into words
            originalSentence = text.Split(" ");
            // Start DFS to build possible sentences
            Dfs(0);
            return answers;
        }

        // Helper method to perform DFS and generate all possible sentences
        private void Dfs(int index)
        {
            if (index >= originalSentence.Length)
            {
                answers.Add(String.Join(" ", currentSentence));
                return;
            }
            // If current word has synonyms
            if (wordIdMap.ContainsKey(originalSentence[index]))
            {
                // Iterate through all synonyms
                foreach (int synonymIndex in synonymGroups[unionFind.Find(wordIdMap[originalSentence[index]])])
                {
                    currentSentence.Add(uniqueWords[synonymIndex]);
                    Dfs(index + 1);
                    currentSentence.RemoveAt(currentSentence.Count - 1);
                }
            }
            else
            {
                // No synonym for current word, add it as is
                currentSentence.Add(originalSentence[index]);
                Dfs(index + 1);
                currentSentence.RemoveAt(currentSentence.Count - 1);
            }
        }
        class UnionFind
        {
            private int[] parent;     // parent[i] holds the parent of i in the UF structure
            private int[] size;       // size[i] holds the size of the tree rooted at i

            // Constructor initializes each element's parent to itself and size to 1
            public UnionFind(int n)
            {
                parent = new int[n];
                size = new int[n];
                for (int i = 0; i < n; ++i)
                {
                    parent[i] = i;
                    size[i] = 1;
                }
            }

            // find method with path compression, returns the root of the set x belongs to
            public int Find(int x)
            {
                if (parent[x] != x)
                {
                    parent[x] = Find(parent[x]);
                }
                return parent[x];
            }

            // Union method to merge sets containing a and b
            public void Union(int a, int b)
            {
                int rootA = Find(a);
                int rootB = Find(b);
                if (rootA != rootB)
                {
                    // Merge smaller tree into the larger one
                    if (size[rootA] > size[rootB])
                    {
                        parent[rootB] = rootA;
                        size[rootA] += size[rootB];
                    }
                    else
                    {
                        parent[rootA] = rootB;
                        size[rootB] += size[rootA];
                    }
                }
            }
        }


    }


    /* 2375. Construct Smallest Number From DI String
    https://leetcode.com/problems/construct-smallest-number-from-di-string/description/
    https://algo.monster/liteproblems/2375
     */

    class SmallestNumberSol
    {
        // Array to keep track of visited digits
        private bool[] visited = new bool[10];
        // StringBuilder to construct the sequence incrementally
        private StringBuilder sequence = new StringBuilder();
        // String to store the given pattern
        private String pattern;
        // String to store the final answer sequence
        private String answer;

        /* Time and Space Complexity
        Time Complexity
        The time complexity of the code is determined by the number of recursive calls to the dfs function, which is dependent on the length of the pattern string (n) and the branching factor at each step of the recursion.
        With each recursive call to dfs, the function tries to append each number from 1 to 9 that hasn't already been used to the temporary array t. This means that in the worst case, the first recursive call will have 9 options, the second will have 8 options, and so on, resulting in a factorial time complexity.
        Let's denote the length of the pattern as n. The number of recursive calls can be bounded by 9! (factorial) for small patterns, since we have at most 9 digits to use, and it decreases for each level of the recursion. However, for longer patterns, the maximum branching factor will diminish as the pattern increases beyond 9, so it will be less than 9! for patterns longer than 9.
        Therefore, the time complexity can be approximated as O(9!) for patterns up to length 9. For patterns longer than 9, the time complexity is still bounded by O(9!) due to the early termination of the recursion once all digits are used.
        Space Complexity
        The space complexity is determined by the depth of the recursion (which impacts the call stack size) and the additional data structures used (such as the vis array and the t list).
        Since the maximum depth of the recursion is equal to the length of the pattern plus one (n + 1), the contribution to the space complexity from the call stack is O(n).
        The vis array is always of size 10, representing the digits 1 through 9. The size of t corresponds to the depth of the recursion, which is O(n). Therefore, the space requirements for vis are O(1) whereas for t are O(n).
        Combining the contributions, the total space complexity is O(n) due also to the recursive call stack size being at most n for patterns longer than 9.
        To summarize:
        •	The time complexity is O(9!).
        •	The space complexity is O(n).
         */
        public String smallestNumber(String pattern)
        {
            this.pattern = pattern;
            // Starting the depth-first search (DFS)
            Dfs(0);
            // Return the final answer sequence
            return answer;
        }

        // Helper method for the DFS
        private void Dfs(int position)
        {
            // If an answer is already found, stop the recursion
            if (answer != null)
            {
                return;
            }
            // If the length of sequence equals the length of pattern + 1, we have a complete sequence
            if (position == pattern.Length + 1)
            {
                // Set the current sequence as the answer
                answer = sequence.ToString();
                return; // Stop further recursion
            }
            // Iterate through all possible digits (1 to 9)
            for (int i = 1; i < 10; ++i)
            {
                // If the current digit i has not been used yet
                if (!visited[i])
                {
                    // If the last added digit should be less according to the pattern 'I'
                    if (position > 0 && pattern[position - 1] == 'I' && sequence[position - 1] - '0' >= i)
                    {
                        continue; // Skip this digit since it would break the pattern
                    }
                    // If the last added digit should be more according to the pattern 'D'
                    if (position > 0 && pattern[position - 1] == 'D' && sequence[position - 1] - '0' <= i)
                    {
                        continue; // Skip this digit since it would break the pattern
                    }
                    // Mark the digit as used
                    visited[i] = true;
                    // Add the digit to the sequence
                    sequence.Append(i);
                    // Recurse to the next position with updated sequence and visited digits
                    Dfs(position + 1);
                    // Backtrack: remove the last digit from the sequence
                    sequence.Remove(sequence.Length - 1, 1);
                    // Mark the digit as not used (undo the previous marking)
                    visited[i] = false;
                }
            }
        }
    }


    /* 402. Remove K Digits
    https://leetcode.com/problems/remove-k-digits/description/
     */
    class RemoveKdigitsSol
    {
        /* Approach 1: Brute-force [Time Limit Exceeded] 
           The major caveat is that the algorithm would have an exponential time complexity, since we need to enumerate the combinations of selecting k numbers out of a list of n, i.e. C k to n        
​           Even for a trial example, the algorithm could run out of the time limit.
        */


        /* 
        
Approach 2: Greedy with Stack

        Complexity Analysis
•	Time complexity : O(N). Although there are nested loops, the inner loop is bounded to be run at most k times globally. Together with the outer loop, we have the exact (N+k) number of operations. Since 0<k≤N, the time complexity of the main loop is bounded within 2N.
For the logic outside the main loop, it is clear to see that their time complexity is O(N). As a result, the overall time complexity of the algorithm is O(N).
•	Space complexity : O(N). We have a stack which would hold all the input digits in the worst case.
 */
        public String UsingGreedyWithStack(String num, int k)
        {
            LinkedList<char> stack = new();

            foreach (char digit in num)
            {
                while (stack.Count > 0 && k > 0 && stack.Last() > digit)
                {
                    stack.RemoveLast();
                    k -= 1;
                }
                stack.AddLast(digit);
            }

            /* remove the remaining digits from the tail. */
            for (int i = 0; i < k; ++i)
            {
                stack.RemoveLast();
            }

            // build the final string, while removing the leading zeros.
            StringBuilder ret = new StringBuilder();
            bool leadingZero = true;
            foreach (char digit in stack)
            {
                if (leadingZero && digit == '0') continue;
                leadingZero = false;
                ret.Append(digit);
            }

            /* return the final string  */
            if (ret.Length == 0) return "0";
            return ret.ToString();
        }
    }

    /* 475. Heaters
    https://leetcode.com/problems/heaters/description/
    https://algo.monster/liteproblems/475
     */

    class FindRadiusSol
    {
        /* Time and Space Complexity
        Time Complexity
        The time complexity of the solution is determined by the following factors:
        •	Sorting the houses and heaters arrays, which takes O(mlogm + nlogn) time, where m is the number of houses and n is the number of heaters.
        •	The binary search to find the minimum radius, which takes O(log(max(houses) - min(houses))) iterations. Each iteration performs a check which has a linear pass over the houses and heaters arrays, taking O(m + n) time in the worst case.
        •	Combining these, the overall time complexity is O(mlogm + nlogn + (m + n)log(max(houses) - min(houses))).
        Space Complexity
        The space complexity of the solution is determined by the following factors:
        •	The space used to sort the houses and heaters arrays, which is O(m + n) if we consider the space used by the sorting algorithm.
        •	The space for the variables and pointers used within the findRadius method and the check function, which is O(1) since it's a constant amount of space that doesn't depend on the input size.
        Combining the sorting space and the constant space, the overall space complexity is O(m + n).
         */
        public int FindRadius(int[] houses, int[] heaters)
        {
            // Sort the array of heaters to perform efficient searches later on
            Array.Sort(heaters);

            // Initialize the minimum radius required for heaters to cover all houses
            int minRadius = 0;

            // Iterate through each house to find the minimum radius needed
            foreach (int house in houses)
            {
                // Perform a binary search to find the insertion point or the actual position of the house
                int index = Array.BinarySearch(heaters, house);

                // If the house is not a heater, calculate the potential insert position
                if (index < 0)
                {
                    index = ~index;  // index = -(index + 1)
                }

                // Calculate distance to the previous heater, if any, else set to max value
                int distanceToPreviousHeater = index > 0 ? house - heaters[index - 1] : int.MaxValue;

                // Calculate distance to the next heater, if any, else set to max value
                int distanceToNextHeater = index < heaters.Length ? heaters[index] - house : int.MaxValue;

                // Calculate the minimum distance to the closest heater for this house
                int minDistanceToHeater = Math.Min(distanceToPreviousHeater, distanceToNextHeater);

                // Update the minimum radius to be the maximum of previous radii or the minimum distance for this house
                minRadius = Math.Max(minRadius, minDistanceToHeater);
            }

            // Return the minimum radius required
            return minRadius;
        }
    }

    /* 2747. Count Zero Request Servers
    https://leetcode.com/problems/count-zero-request-servers/description/
    https://algo.monster/liteproblems/2747
     */
    class CountZeroRequestServersSol
    {
        /* Time and Space Complexity
Time Complexity
The overall time complexity of the code can be broken down as follows:
1.	Sorting logs: Sorting the logs list based on timestamp has a time complexity of O(m log m), where m is the length of the logs list.
2.	Iterating over queries: As the queries are sorted together with the count(), the complexity for sorting the zipped list is O(q log q), where q is the number of queries.
3.	Iterating over logs while processing each query: We iterate through the logs twice in a linear fashion, once with k and once with j, but since both k and j only move forward, each log is processed at most twice. This gives us O(m).
Combining these three parts, the dominant terms are the sorting parts, leading to a final time complexity of O(m log m + q log q).
Space Complexity
The space complexity can be broken down as follows:
1.	Storing sorted queries and indexes: This requires O(q) space.
2.	cnt Counter data structure: In the worst case, it will store an entry for every unique server, which is, in the worst case, as large as the length of logs, resulting in O(m) space complexity.
3.	ans list to store the answers: This will take O(q) space.
The combined space complexity, combining the space used by cnt and ans, as well as the temporary space for sorting, is O(m + q).
 */
        // Method to count the number of servers that are not busy during query times
        public int[] CountServers(int totalServers, int[][] logs, int timeFrame, int[] queries)
        {
            // Sort the logs based on the timestamp
            Array.Sort(logs, (a, b) => a[1] - b[1]);

            // Number of queries
            int numQueries = queries.Length;
            // Array to hold query and index as pair
            int[][] sortedQueries = new int[numQueries][];
            // Populate with query and index
            for (int i = 0; i < numQueries; ++i)
            {
                sortedQueries[i] = new int[] { queries[i], i };
            }
            // Sort the queries based on query time
            Array.Sort(sortedQueries, (a, b) => a[0] - b[0]);

            // Use a Map to keep track of the number of busy servers
            Dictionary<int, int> busyServersCount = new();
            // Array to hold the answer for each query
            int[] answers = new int[numQueries];

            // Indexes for processing logs
            int startIndex = 0, endIndex = 0;
            // Loop over each sorted query
            foreach (var query in sortedQueries)
            {
                int queryTime = query[0], originalIndex = query[1];
                int lowerBoundTime = queryTime - timeFrame;
                // Increment the server usage count for logs within the time frame
                while (endIndex < logs.Length && logs[endIndex][1] <= queryTime)
                {
                    if (busyServersCount.ContainsKey(logs[endIndex][0]))
                    {
                        busyServersCount[logs[endIndex][0]]++;
                    }
                    else
                    {
                        busyServersCount[logs[endIndex][0]] = 1;
                    }
                    endIndex++;
                }
                // Decrement the server usage count for logs before the time frame
                while (startIndex < logs.Length && logs[startIndex][1] < lowerBoundTime)
                {
                    int serverId = logs[startIndex][0];
                    if (busyServersCount.ContainsKey(serverId))
                    {
                        busyServersCount[serverId]--;
                        if (busyServersCount[serverId] == 0)
                        {
                            busyServersCount.Remove(serverId);
                        }
                    }
                    startIndex++;
                }
                // Calculate how many servers are not busy
                answers[originalIndex] = totalServers - busyServersCount.Count;
            }
            // Return the array of answers for each query
            return answers;
        }
    }

    /* 2079. Watering Plants
    https://leetcode.com/problems/watering-plants/description/
    https://algo.monster/liteproblems/2079
     */
    class WateringPlantsSol
    {
        /* Time and Space Complexity
        The time complexity of the code is O(n), where n is the number of plants. This is because the code iterates through each plant exactly once.
        The space complexity of the code is O(1) since a fixed amount of extra space is used regardless of the input size. Additional variables ans and cap are used, but their use does not scale with the number of plants.
         */
        public int WateringPlants(int[] plants, int capacity)
        {
            int steps = 0; // This will hold the total number of steps taken
            int currentCapacity = capacity; // This holds the current water capacity in the can

            // Loop through all the plants
            for (int i = 0; i < plants.Length; i++)
            {
                // If there's enough water left to water the current plant
                if (currentCapacity >= plants[i])
                {
                    currentCapacity -= plants[i]; // Water the plant and decrease the can's capacity
                    steps++; // One step to water the plant
                }
                else
                {
                    // If there isn't enough water capacity:
                    // Steps to go back to the river to refill (i steps) 
                    // and return back to this plant (i + 1 steps)
                    steps += 2 * i + 1;
                    currentCapacity = capacity - plants[i]; // Refill the can minus the water needed for current plant
                }
            }
            return steps; // Return the total number of steps taken
        }
    }

    /* 2707. Extra Characters in a String	
    https://leetcode.com/problems/extra-characters-in-a-string/description/
     */
    class MinExtraCharInStringSol
    {
        private int?[] memo;
        private HashSet<string> dictionarySet;

        /* Approach 1: Top Down Dynamic Programming with Substring Method
Complexity Analysis
Let N be the total characters in the string.
Let M be the average length of the strings in dictionary.
Let K be the length of the dictionary.
•	Time complexity: O(N^3). There can be N+1 unique states of the dp method. In each state of dp, we iterate over end, which is O(N) iterations. In each of these iterations, we create a substring, which costs O(N). Hence, the overall cost of the dp method is O(N^3).
•	Space complexity: O(N+M⋅K). The HashSet used to store the strings in the dictionary will incur a cost of O(M⋅K). Additionally, the dp method will consume stack space and traverse to a depth of N in the worst case scenario, resulting in a cost of O(N).

         */
        public int TopDownDPWithSubstring(string s, string[] dictionary)
        {
            int length = s.Length;
            memo = new int?[length];
            dictionarySet = new HashSet<string>(dictionary);
            return Dp(0, length, s);
        }

        private int Dp(int start, int length, string s)
        {
            if (start == length)
            {
                return 0;
            }
            if (memo[start] != null)
            {
                return memo[start].Value;
            }
            // To count this character as a left over character 
            // move to index 'start + 1'
            int answer = Dp(start + 1, length, s) + 1;
            for (int end = start; end < length; end++)
            {
                string currentSubstring = s.Substring(start, end - start + 1);
                if (dictionarySet.Contains(currentSubstring))
                {
                    answer = Math.Min(answer, Dp(end + 1, length, s));
                }
            }

            return (int)(memo[start] = answer);
        }
        /* Approach 2: Bottom Up Dynamic Programming with Substring Method
Complexity Analysis
Let N be the total characters in the string.
Let M be the average length of the strings in dictionary.
Let K be the length of the dictionary.
•	Time complexity: O(N^3). The two nested loops used to perform the dynamic programming operation cost O(N^2). The substring method inside the inner loop costs another O(N). Hence, the overall time complexity is O(N^3).
•	Space complexity: O(N+M⋅K). The HashSet used to store the strings in the dictionary will incur a cost of O(M⋅K). The dp array will incur a cost of O(N).

         */
        public int BottomUpDPWithSubstring(String s, String[] dictionary)
        {
            int n = s.Length;
            var dictionarySet = new List<string>(dictionary);
            var dp = new int[n + 1];

            for (int start = n - 1; start >= 0; start--)
            {
                dp[start] = dp[start + 1] + 1;
                for (int end = start; end < n; end++)
                {
                    var curr = s.Substring(start, end + 1);
                    if (dictionarySet.Contains(curr))
                    {
                        dp[start] = Math.Min(dp[start], dp[end + 1]);
                    }
                }
            }

            return dp[0];
        }
        /* Approach 3: Top Down Dynamic Programming with Trie 
Complexity Analysis
Let N be the total characters in the string.
Let M be the average length of the strings in dictionary.
Let K be the length of the dictionary.
•	Time complexity: O(N^2+M⋅K). There can be N+1 unique states of the dp method. Each state of the dp method costs O(N) to compute. Hence, the overall cost of the dp method is O((N+1)⋅N) or simply O(N^2). Building the trie costs O(M⋅K).
•	Space complexity: O(N+M⋅K). The Trie used to store the strings in the dictionary will incur a cost of O(M⋅K). Additionally, the dp method will consume stack space and traverse to a depth of N, resulting in a cost of O(N).

        */
        private TrieNode root;

        public int MinExtraChar(string s, string[] dictionary)
        {
            int length = s.Length;
            root = BuildTrie(dictionary);
            memo = new int?[length + 1];
            return DpExt(0, length, s);
        }

        private int DpExt(int start, int length, string s)
        {
            if (start == length)
            {
                return 0;
            }
            if (memo[start] != null)
            {
                return memo[start].Value;
            }

            TrieNode node = root;
            // To count this character as a left over character 
            // move to index 'start + 1'
            int answer = DpExt(start + 1, length, s) + 1;
            for (int end = start; end < length; end++)
            {
                char currentChar = s[end];
                if (!node.Children.ContainsKey(currentChar))
                {
                    break;
                }
                node = node.Children[currentChar];
                if (node.IsWord)
                {
                    answer = Math.Min(answer, DpExt(end + 1, length, s));
                }
            }

            return (int)(memo[start] = answer);
        }

        private TrieNode BuildTrie(string[] dictionary)
        {
            TrieNode rootNode = new TrieNode();
            foreach (string word in dictionary)
            {
                TrieNode node = rootNode;
                foreach (char character in word)
                {
                    if (!node.Children.ContainsKey(character))
                    {
                        node.Children[character] = new TrieNode();
                    }
                    node = node.Children[character];
                }
                node.IsWord = true;
            }
            return rootNode;
        }
        class TrieNode
        {
            public Dictionary<char, TrieNode> Children { get; set; } = new Dictionary<char, TrieNode>();
            public bool IsWord { get; set; } = false;
        }
        /* Approach 4: Bottom Up Dynamic Programming with Trie
Complexity Analysis
Let N be the total characters in the string.
Let M be the average length of the strings in dictionary.
Let K be the length of the dictionary.
•	Time complexity: O(N^2+M⋅K). The two nested for loops that are being used for the dynamic programming operation cost O(N2). Building the trie costs O(M⋅K).
•	Space complexity: O(N+M⋅K). The Trie used to store the strings in dictionary will incur a cost of O(M⋅K). The dp array will incur a cost of O(N).

         */
        public int minExtraChar(String s, String[] dictionary)
        {
            int n = s.Length;
            var root = BuildTrie(dictionary);
            var dp = new int[n + 1];

            for (int start = n - 1; start >= 0; start--)
            {
                dp[start] = dp[start + 1] + 1;
                var node = root;
                for (int end = start; end < n; end++)
                {
                    if (!node.Children.ContainsKey(s[end]))
                    {
                        break;
                    }
                    node = node.Children[s[end]];
                    if (node.IsWord)
                    {
                        dp[start] = Math.Min(dp[start], dp[end + 1]);
                    }
                }
            }

            return dp[0];
        }




    }

    /* 319. Bulb Switcher
    https://leetcode.com/problems/bulb-switcher/description/
     */
    class BulbSwitchSol
    {
        /* 
        Approach 1: Math
        Complexity Analysis
Here, n is the number of bulbs and rounds.
•	Time complexity: O(1)
o	In general, the fast inverse square root algorithm is used to compute the square root of a number (which is typically represented using 32 bits) in most programming languages. The algorithm performs a series of bitwise and floating-point operations on the input value to compute an approximation of the inverse square root. The number of operations performed by the algorithm is fixed and does not depend on the input size. Thus, it makes each call to this method an O(1) time operation.

•	Note: If we want to compute the square root of large numbers (e.g: 10^10000), it would be impractical to use the fast inverse square root algorithm. The fast inverse square root algorithm is designed to compute an approximation of the inverse square root of a 32-bit floating-point number, and it may not be accurate enough for very large numbers.
•	Instead, the languages would need to use a different algorithm that is capable of handling very large numbers with high precision. The Newton-Raphson and Babylonina methods are such algorithms that can be used to compute the square root of large numbers with high precision in nearly log-linear time (also called linearithmic time) O(dlogd), where d is the number of digits of the input number.
•	Space complexity: O(1)
o	The implementation of the sqrt method doesn't use any additional space.
 */
        public int UsingMaths(int n)
        {
            return (int)Math.Sqrt(n);
        }
    }

    /* 672. Bulb Switcher II
    https://leetcode.com/problems/bulb-switcher-ii/description/
    https://algo.monster/liteproblems/672
     */
    class BulbSwitchIISol
    {
        /* Approach 1: Reduce Search Space [Accepted]
   Complexity Analysis
   •	Time Complexity: O(1). Our checks are bounded by a constant.
   •	Space Complexity: O(1), the size of the data structures used.

         */
        public int UsingSearchSpaceReduction(int n, int m)
        {
            HashSet<int> seen = new HashSet<int>();
            n = Math.Min(n, 6);
            int shift = Math.Max(0, 6 - n);
            for (int cand = 0; cand < 16; ++cand)
            {
                int bcount = int.PopCount(cand);
                if (bcount % 2 == m % 2 && bcount <= m)
                {
                    int lights = 0;
                    if (((cand >> 0) & 1) > 0) lights ^= 0b111111 >> shift;
                    if (((cand >> 1) & 1) > 0) lights ^= 0b010101 >> shift;
                    if (((cand >> 2) & 1) > 0) lights ^= 0b101010 >> shift;
                    if (((cand >> 3) & 1) > 0) lights ^= 0b100100 >> shift;
                    seen.Add(lights);
                }
            }
            return seen.Count;
        }
        /* Approach 2: Mathematical [Accepted]
        Complexity Analysis
•	Time and Space Complexity: O(1). The entire program uses constants.

         */
        public int UsingMaths(int n, int m)
        {
            n = Math.Min(n, 3);
            if (m == 0) return 1;
            if (m == 1) return n == 1 ? 2 : n == 2 ? 3 : 4;
            if (m == 2) return n == 1 ? 2 : n == 2 ? 4 : 7;
            return n == 1 ? 2 : n == 2 ? 4 : 8;
        }


    }

    /* 1820. Maximum Number of Accepted Invitations
    https://leetcode.com/problems/maximum-number-of-accepted-invitations/description/
    https://algo.monster/liteproblems/1820
     */
    public class MaximumInvitationsSol
    {
        /* Time and Space Complexity
Time Complexity
The given code performs a modification of the Hungarian algorithm for the maximum bipartite matching problem. The time complexity is derived from the nested loops and the find function, which is a depth-first search (DFS):
•	There is an outer loop that iterates m times where m is the number of rows in grid. This loop calls the find function.
•	Within the find function, there is a loop that iterates n times in the worst case, where n is the number of columns, to try to find a matching for each element in the row.
•	The depth-first search (DFS) within the find function can traverse up to n elements in the worst-case scenario.
•	Since match[j] is called recursively within find (specifically, find(match[j])), in the worst case, this can lead to another n iterations if every column is linked to a new row. Hence, the recursive calls can happen n times in the worst case.
Combining these factors, the time complexity of the entire algorithm is O(m * n^2) in the worst case.
Space Complexity
The space complexity can be considered based on the data structures used:
•	The match list uses space proportional to n, which is O(n).
•	The vis set can store up to n unique values in the worst case, as it keeps track of the visited columns for each row during DFS. This also results in space complexity of O(n).
•	The recursive find function can go up to n levels deep due to recursive calls, which could potentially use space O(n) on the call stack.
As these do not depend on each other and only the maximum will determine the overall space complexity, the space complexity of the algorithm is O(n).
 */

        private int[][] grid; // The grid representing invitations
        private bool[] visited; // To track if a column (person in right set) has been visited
        private int[] matched; // To store matched left-side people to right-side people
        private int columns; // Number of columns in the grid

        // Method to compute the maximum number of invitations
        public int MaximumInvitations(int[][] grid)
        {
            int rows = grid.Length; // Number of rows in the grid
            columns = grid[0].Length; // Number of columns in the grid
            this.grid = grid;
            visited = new bool[columns]; // Initialize visited array for tracking
            matched = new int[columns]; // Initialize matches array
            Array.Fill(matched, -1); // Fill the matches array with -1 indicating no match
            int invitations = 0; // Initialize invitation count

            // Iterate over all rows (left side people) to find maximum matchings
            for (int i = 0; i < rows; ++i)
            {
                Array.Fill(visited, false); // Reset the visited array for each iteration
                if (TryFindMatch(i))
                {
                    invitations++; // If a match is found, increment the invitation count
                }
            }
            return invitations; // Return the maximum number of invitations
        }

        // Helper method to find a match for a person in the left set
        private bool TryFindMatch(int personIdx)
        {
            for (int j = 0; j < columns; ++j)
            {
                // If there's an invitation from personIdx to right set person 'j' and 'j' is not visited
                if (grid[personIdx][j] == 1 && !visited[j])
                {
                    visited[j] = true; // Mark 'j' as visited
                                       // If 'j' is not matched, or we can find a match for 'j's current match
                    if (matched[j] == -1 || TryFindMatch(matched[j]))
                    {
                        matched[j] = personIdx; // Match personIdx (left set) with 'j' (right set)
                        return true; // A match was successful
                    }
                }
            }
            return false; // No match was found for personIdx
        }
    }


    /* 274. H-Index
    https://leetcode.com/problems/h-index/description/
     */
    public class HIndexSol
    {

        /* Approach #1 (Sorting) [Accepted]
Complexity Analysis
•	Time complexity : O(nlogn). Comparison sorting dominates the time complexity.
•	Space complexity : O(1). Most libraries using heap sort which costs O(1) extra space in the worst case.

         */
        public int UsingSort(int[] citations)
        {
            // sorting the citations in ascending order
            Array.Sort(citations);
            // finding h-index by linear search
            int i = 0;
            while (i < citations.Length && citations[citations.Length - 1 - i] > i)
            {
                i++;
            }
            return i; // after the while loop, i = i' + 1
        }
        /* 
        Approach #2 (Counting) [Accepted]
        Complexity Analysis
•	Time complexity : O(n). There are two steps. The counting part is O(n) since we traverse the citations array once and only once. The second part of finding the h-index is also O(n) since we traverse the papers array at most once. Thus, the entire algorithm is O(n)
•	Space complexity : O(n). We use O(n) auxiliary space to store the counts.

 */
        public int UsingCounting(int[] citations)
        {
            int n = citations.Length;
            int[] papers = new int[n + 1];
            // counting papers for each citation number
            foreach (int c in citations)
                papers[Math.Min(n, c)]++;

            // finding the h-index
            int k = n;
            for (int s = papers[n]; k > s; s += papers[k])
                k--;
            return k;
        }
    }

    /* 593. Valid Square
    https://leetcode.com/problems/valid-square/description/
     */
    public class ValidSquareSol
    {
        /* Approach #1 Brute Force [Accepted]
Complexity Analysis
•	Time complexity : O(1). Constant number of permutations(4!) are generated.
•	Space complexity : O(1). Constant space is required.

         */
        public bool ValidSquare(int[] p1, int[] p2, int[] p3, int[] p4)
        {
            int[][] p = { p1, p2, p3, p4 };
            return CheckAllPermutations(p, 0);
        }
        private double Dist(int[] p1, int[] p2)
        {
            return (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[0] - p1[0]) * (p2[0] - p1[0]);
        }
        private bool Check(int[] p1, int[] p2, int[] p3, int[] p4)
        {
            return Dist(p1, p2) > 0 && Dist(p1, p2) == Dist(p2, p3) && Dist(p2, p3) == Dist(p3, p4) && Dist(p3, p4) == Dist(p4, p1) && Dist(p1, p3) == Dist(p2, p4);
        }

        private bool CheckAllPermutations(int[][] p, int l)
        {
            if (l == 4)
            {
                return Check(p[0], p[1], p[2], p[3]);
            }
            else
            {
                bool res = false;
                for (int i = l; i < 4; i++)
                {
                    Swap(p, l, i);
                    res |= CheckAllPermutations(p, l + 1);
                    Swap(p, l, i);
                }
                return res;
            }
        }
        public void Swap(int[][] p, int x, int y)
        {
            int[] temp = p[x];
            p[x] = p[y];
            p[y] = temp;
        }
        /*  
        Approach #2 Using Sorting [Accepted]
Complexity Analysis
•	Time complexity : O(1). Sorting 4 points takes constant time.
•	Space complexity : O(1). Constant space is required.

         */
        public bool UsingSort(int[] p1, int[] p2, int[] p3, int[] p4)
        {
            int[][] p = { p1, p2, p3, p4 };
            Array.Sort(p, (l1, l2) => l2[0] == l1[0] ? l1[1] - l2[1] : l1[0] - l2[0]);
            return Dist(p[0], p[1]) != 0 && Dist(p[0], p[1]) == Dist(p[1], p[3]) && Dist(p[1], p[3]) == Dist(p[3], p[2]) && Dist(p[3], p[2]) == Dist(p[2], p[0]) && Dist(p[0], p[3]) == Dist(p[1], p[2]);
        }


        /* 
        Approach #3 Checking every case [Accepted]
        Complexity Analysis
•	Time complexity : O(1). A fixed number of comparisons are done.
•	Space complexity : O(1). No extra space required.
         */
        public bool UsingCheckEveryCase(int[] p1, int[] p2, int[] p3, int[] p4)
        {
            return Check(p1, p2, p3, p4) || Check(p1, p3, p2, p4) || Check(p1, p2, p4, p3);
        }

    }


    /* 2850. Minimum Moves to Spread Stones Over Grid
    https://leetcode.com/problems/minimum-moves-to-spread-stones-over-grid/description/
    https://algo.monster/liteproblems/2850
     */
    class MinimumMovesToSpreadStonesOverGridSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the given code is O(n * 2^n). The reason for this time complexity is due to the combination of iterating over all subsets of "left" and calculating the minimum distances against all elements in "right". Specifically:
•	There are n cities on the left side, resulting in 2^n subsets due to the binary representation used to enumerate these subsets.
•	For each subset (represented by i), we use bit_count() which contributes O(n) as it counts the number of set bits in the binary representation of the integer.
•	The inner loop runs n times for each subset to calculate distances and find the minimum after excluding elements using the XOR operation i ^ (1 << j).
•	Each call to cal() function is constant time, however, it's called n times in the worst case.
Hence, combining these factors, we end up with O(n * 2^n).
Space Complexity
The space complexity of the code is O(2^n). The reasons are as follows:
•	We maintain an array f of size 2^n that keeps track of the minimum distance for every subset of cities on the "left".
•	The left and right lists have a linear space complexity based on the input size, in the worst case it would be O(n), which is eclipsed by the space needed for f.
•	Auxiliary space used by the recursion stack for bit_count() or temporary variables in the loops are constant in comparison to the space used by f.
Overall, the dominant factor here is the f array, hence the space complexity is O(2^n).	
 */
        public int MinimumMoves(int[][] grid)
        {
            List<int[]> emptySpaces = new List<int[]>();
            List<int[]> obstacles = new List<int[]>();

            // Identify empty spaces and obstacles
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    if (grid[row][col] == 0)
                    {
                        emptySpaces.Add(new int[] { row, col });
                    }
                    else
                    {
                        // If the cell is not empty, put obstacles according to the number specified in the cell
                        for (int count = 1; count < grid[row][col]; ++count)
                        {
                            obstacles.Add(new int[] { row, col });
                        }
                    }
                }
            }

            int numEmptySpaces = emptySpaces.Count;
            int[] dp = new int[1 << numEmptySpaces]; // Dynamic programming array to store minimum moves
            Array.Fill(dp, int.MaxValue); // Initialize all moves to a large number
            dp[0] = 0; // Zero moves needed when there's no empty space covered

            // Calculate minimum moves using bit masking to represent covering of each empty spaces.
            for (int mask = 1; mask < (1 << numEmptySpaces); ++mask)
            {
                int moves = CountBits(mask); // Count the number of covered empty spaces
                for (int i = 0; i < numEmptySpaces; ++i)
                {
                    if ((mask >> i & 1) == 1)
                    {
                        // Update the DP table if a space gets covered
                        dp[mask] = Math.Min(dp[mask], dp[mask ^ (1 << i)] + CalculateDistance(emptySpaces[moves - 1], obstacles[i]));
                    }
                }
            }

            // Return the minimum moves to cover all empty spaces
            return dp[(1 << numEmptySpaces) - 1];
        }

        // Helper method to calculate Manhattan distance between two points on the grid
        private int CalculateDistance(int[] pointA, int[] pointB)
        {
            return Math.Abs(pointA[0] - pointB[0]) + Math.Abs(pointA[1] - pointB[1]);
        }

        // Helper method to count the number of bits set to 1
        private int CountBits(int mask)
        {
            int count = 0;
            while (mask > 0)
            {
                count += mask & 1;
                mask >>= 1;
            }
            return count;
        }
    }

    /* 2184. Number of Ways to Build Sturdy Brick Wall
    https://leetcode.com/problems/number-of-ways-to-build-sturdy-brick-wall/description/
    https://algo.monster/liteproblems/2184
     */
    public class NumWaysToBuildSturdyBrickWallSol
    {
        // Store all the unique row configurations
        private List<List<int>> uniqueRows = new List<List<int>>();
        // Temporary list to store current row configuration
        private List<int> currentRow = new List<int>();
        // Modulus value for the result
        private static readonly int MOD = (int)1e9 + 7;
        // The wall's width and the set of brick lengths
        private int wallWidth;
        private int[] brickSizes;
        /* 
        Time Complexity
        The time complexity can be divided into a few parts:
        1.	DFS (dfs function) for generating all possible row combinations: The DFS function is used to generate all possible combinations of bricks that can form a row of the desired width. In the worst case, it generates all permutational combinations. If we denote k as the maximum number of bricks that a row can take, then the time complexity for this step can be approximated as O(b^k), where b is the number of brick sizes in the bricks list.
        2.	Checking compatibility between two rows (check function): The check function is used to determine if two rows are compatible, which means no vertical cracks align. Every row combination is checked against every other combination. This gives us time complexity of O(n^2 * w) for this part, where n is the total number of unique rows and w is the width of the wall since in the worst scenario each element of a row is checked against each element of another row.
        3.	Populating the graph with compatible layers: Based on the outputs of the check function, a graph is populated. This is also O(n^2) as each row is checked against every other row for compatibility.
        4.	Dynamic Programming to calculate the number of ways to build the wall (dp loop): The DP array is filled in height iterations, and each sub-problem depends on the previous layer's sub-problems. We have height * n sub-problems, and each could potentially depend on all n other sub-problems from the previous row. Thus the complexity for this part is O(height * n^2).
        Combining all steps, the total time complexity of the code is:
        T(height, width, b) = O(b^k) + O(n^2 * w) + O(n^2) + O(height * n^2)
        Since the number of unique rows n can be pretty large (up to b^k), the time complexity is dominated by the last term which involves height, leading to a worst-case scenario of:
        T(height, width, b) = O(height * b^(2k))
        Space Complexity
        The space complexity is determined by the storage required for:
        1.	The s list that stores all possible row combinations: In the worst case, this could store up to b^k combinations.
        2.	The g defaultdict that stores the adjacency list of the graph: Since we're dealing with n unique rows and each row can be compatible with up to n-1 other rows, in the worst case, this takes O(n^2) space.
        3.	The dp array: The dp array is a 2D array with height rows and n columns, so it takes O(height * n) space.
        4.	Additional space for the t list used during DFS and function call stack: The t list and the recursion call stack could in the worst case require O(width) space each due to the recursive calls depending on the width of the wall.
        The space complexity of the code is thus the maximum of the space used by these data structures:
        S(height, width, b) = max(O(b^k), O(n^2), O(height * n), O(width))
        Since n can be as large as b^k, the dominant term is:
        S(height, width, b) = O(height * b^(2k)) + O(width)
        Considering O(height * b^(2k)) would generally be larger than O(width), we can approximate the space complexity as:
        S(height, width, b) = O(height * b^(2k))
         */
        public int BuildWall(int height, int width, int[] bricks)
        {
            this.wallWidth = width;
            this.brickSizes = bricks;
            // Generate all possible row configurations
            GenerateConfigurations(0);
            int numConfigs = uniqueRows.Count;
            List<int>[] graph = new List<int>[numConfigs];
            for (int i = 0; i < numConfigs; ++i)
            {
                graph[i] = new List<int>();
            }

            // Build the adjacency list for the configurations graph
            for (int i = 0; i < numConfigs; ++i)
            {
                for (int j = 0; j < numConfigs; ++j)
                {
                    if (i != j && CanPlaceRows(uniqueRows[i], uniqueRows[j]))
                    {
                        graph[i].Add(j);
                    }
                }
            }

            // Dynamic programming array to store counts of ways to build the wall
            int[,] dp = new int[height, numConfigs];
            // Initialize base case for the first row
            for (int j = 0; j < numConfigs; ++j)
            {
                dp[0, j] = 1;
            }

            // Fill the DP array with number of ways to build wall
            for (int i = 1; i < height; ++i)
            {
                for (int j = 0; j < numConfigs; ++j)
                {
                    foreach (int k in graph[j])
                    {
                        dp[i, j] = (dp[i, j] + dp[i - 1, k]) % MOD;
                    }
                }
            }

            // Sum up all the ways to build the wall of the specified height
            int answer = 0;
            for (int j = 0; j < numConfigs; ++j)
            {
                answer = (answer + dp[height - 1, j]) % MOD;
            }
            return answer;
        }

        // Checks if two rows can be placed on top of each other
        private bool CanPlaceRows(List<int> topRow, List<int> bottomRow)
        {
            int sumTop = topRow[0];
            int sumBottom = bottomRow[0];
            int i = 1, j = 1;
            while (i < topRow.Count && j < bottomRow.Count)
            {
                if (sumTop == sumBottom)
                {
                    return false; // Cracks align
                }
                if (sumTop < sumBottom)
                {
                    sumTop += topRow[i++];
                }
                else
                {
                    sumBottom += bottomRow[j++];
                }
            }
            return true; // No alignment of cracks
        }

        // Depth First Search to generate all possible configurations of one row
        private void GenerateConfigurations(int progress)
        {
            if (progress > wallWidth)
            {
                return; // Exceeds width of wall, invalid configuration
            }
            if (progress == wallWidth)
            {
                // Found a valid row configuration
                uniqueRows.Add(new List<int>(currentRow));
                return;
            }
            foreach (int size in brickSizes)
            {
                currentRow.Add(size);
                GenerateConfigurations(progress + size); // Continue to add bricks
                currentRow.RemoveAt(currentRow.Count - 1); // Backtrack to try other bricks
            }
        }
    }

    /* 365. Water and Jug Problem
    https://leetcode.com/problems/water-and-jug-problem/description/
    https://algo.monster/liteproblems/365
     */

    class CanMeasureWaterSol
    {
        /* Time and Space Complexity
The given Python function canMeasureWater determines whether it is possible to measure exactly targetCapacity liters by using two jugs of capacities jug1Capacity and jug2Capacity. It does so using a theorem related to the Diophantine equation which states that a target capacity x can be measured using two jugs with capacities m and n if and only if x is a multiple of the greatest common divisor (GCD) of m and n.
Time Complexity:
The time complexity of the function is predominantly determined by the computation of the GCD of jug1Capacity and jug2Capacity. Here’s how the complexity breaks down:
1.	The function checks if the sum of the capacities of the two jugs is less than the targetCapacity. This comparison is constant time, O(1).
2.	Then, it checks if either jug has a 0 capacity, and in such cases, it also performs constant-time comparisons: O(1).
3.	Finally, it calculates the GCD of the two jug capacities. The GCD is calculated using Euclid's algorithm, which has a worst-case time complexity of O(log(min(a, b))), where a and b are jug1Capacity and jug2Capacity. Since the GCD function is bounded by the smaller of the two numbers, the time complexity for this step is O(log(min(jug1Capacity, jug2Capacity))).
Therefore, the overall time complexity of the function is O(log(min(jug1Capacity, jug2Capacity))).
Space Complexity:
The space complexity of the function is determined by the space used to hold any variables and the stack space used by the recursion (if the implementation of GCD is recursive):
1.	Only a fixed number of integer variables are used, and there’s no use of any data structures that scale with the input size. This contributes a constant space complexity: O(1).
2.	Assuming gcd function from the math library is used, which is typically implemented iteratively, the space complexity remains constant as there are no recursive calls stacking up.
Therefore, the overall space complexity of the function is O(1) constant space.
 */

        /**
         * Determines if it's possible to measure exactly the target capacity using the two jugs.
         *
         * @param jug1Capacity the capacity of jug 1
         * @param jug2Capacity the capacity of jug 2
         * @param targetCapacity the target capacity to measure
         * @return true if it's possible to measure exact target capacity, false otherwise
         */
        public bool CanMeasureWater(int jug1Capacity, int jug2Capacity, int targetCapacity)
        {
            // If the sum of both jug capacities is less than the target, it's not possible to measure.
            if (jug1Capacity + jug2Capacity < targetCapacity)
            {
                return false;
            }
            // If one jug is of 0 capacity, we can only measure the target if 
            // it's 0 or equal to the capacity of the non-zero jug.
            if (jug1Capacity == 0 || jug2Capacity == 0)
            {
                return targetCapacity == 0 || jug1Capacity + jug2Capacity == targetCapacity;
            }
            // The target capacity must be a multiple of the greatest common divisor of the jug capacities.
            return targetCapacity % greatestCommonDivisor(jug1Capacity, jug2Capacity) == 0;
        }

        /**
         * Computes the greatest common divisor (GCD) of two numbers using Euclidean algorithm.
         *
         * @param a the first number
         * @param b the second number
         * @return the greatest common divisor (GCD) of a and b
         */
        private int greatestCommonDivisor(int a, int b)
        {
            // If the second number is 0, return the first number, otherwise, recursively
            // find the GCD of the second number and the remainder of the first number divided by the second number.
            return b == 0 ? a : greatestCommonDivisor(b, a % b);
        }
    }

    /* 1894. Find the Student that Will Replace the Chalk
    https://leetcode.com/problems/find-the-student-that-will-replace-the-chalk/description/
     */

    class ChalkReplacerSol
    {

        /* Approach 1: Prefix Sum
        Complexity Analysis
Let n be the size of the chalk array.
•	Time complexity: O(n)
We iterate through the chalk array exactly twice. Apart from this, all operations are performed in constant time. Therefore, the total time complexity is given by O(n).
•	Space complexity: O(1)
No additional space is used proportional to the array size n. Therefore, the space complexity is given by O(1).

         */
        public int UsingPrefixSum(int[] chalk, int k)
        {
            // Find the sum of all elements.
            long sum = 0;
            for (int i = 0; i < chalk.Length; i++)
            {
                sum += chalk[i];
                if (sum > k)
                {
                    break;
                }
            }
            // Find modulo of k with sum.
            k = k % (int)sum;
            for (int i = 0; i < chalk.Length; i++)
            {
                if (k < chalk[i])
                {
                    return i;
                }
                k = k - chalk[i];
            }
            return 0;
        }
        /* Approach 2: Binary Search
Complexity Analysis
Let n be the size of the chalk array.
•	Time complexity: O(n)
We iterate through the chalk array once. Apart from this, the binary search operation takes O(logn) time. Therefore, the total time complexity is given by O(n).
•	Space complexity: O(n)
We initialize an array prefixSum of size n to store the prefix sums of the chalk array. Apart from this, no additional space is used. Therefore, the space complexity is given by O(n).

         */
        public int UsingBinarySearch(int[] chalk, int k)
        {
            int n = chalk.Length;

            long[] prefixSum = new long[n];
            prefixSum[0] = chalk[0];
            for (int i = 1; i < n; i++)
            {
                prefixSum[i] = prefixSum[i - 1] + chalk[i];
            }

            long sum = prefixSum[n - 1];
            long remainingChalk = (k % sum);

            return BinarySearch(prefixSum, remainingChalk);
        }

        private int BinarySearch(long[] arr, long tar)
        {
            int low = 0, high = arr.Length - 1;

            while (low < high)
            {
                int mid = low + (high - low) / 2;

                if (arr[mid] <= tar)
                {
                    low = mid + 1;
                }
                else
                {
                    high = mid;
                }
            }

            return high;
        }
    }

    /* 1257. Smallest Common Region
    https://leetcode.com/problems/smallest-common-region/description/
     */
    class FindSmallestRegionSol
    {

        /* Approach: Lowest Common Ancestor of a Generic Tree
        Complexity Analysis
Let m be the number of region arrays, and let n be the number of regions in each array.
•	Time Complexity: O(m∗n)
We loop through each item in the regions arrays to map child-parent relationships in a hash map. This takes O(m∗n) time.
To create the path to a region, we traverse the hierarchy, which can have up to n regions. Reversing the path array also takes O(n) time.
To find the lowest common ancestor, we compare paths. In the worst case, each path has n elements, so this takes O(n) time.
Thus, the worst-case time complexity is O(m∗n+n+n)=O(m∗n).
•	Space Complexity: O(m∗n)
We use a hash map to store all child-parent pairs. This uses O(m∗n) space.
We store the paths for two regions in arrays, which each take O(n) space.
So, in the worst case, the space complexity is O(m∗n+n)=O(m∗n).

         */

        public string UsingLowerCommonAncestorOfGenericTree(
            List<List<string>> regions,
            string region1,
            string region2
        )
        {
            // Map to store (child -> parent) relationships for each region.
            Dictionary<string, string> childParentMap = new Dictionary<string, string>();

            // Populate the 'childParentMap' using the provided 'regions' list.
            foreach (List<string> regionArray in regions)
            {
                string parentNode = regionArray[0];
                for (int idx = 1; idx < regionArray.Count; idx++)
                {
                    childParentMap[regionArray[idx]] = parentNode;
                }
            }

            // Store the paths from the root node to 'region1' and 'region2'
            // nodes in their respective lists.
            List<string> path1 = FetchPathForRegion(region1, childParentMap);
            List<string> path2 = FetchPathForRegion(region2, childParentMap);

            int i = 0, j = 0;
            string lowestCommonAncestor = "";
            // Traverse both paths simultaneously until the paths diverge.
            // The last common node is the lowest common ancestor.
            while (
                i < path1.Count &&
                j < path2.Count &&
                path1[i] == path2[j]
            )
            {
                lowestCommonAncestor = path1[i];
                i++;
                j++;
            }

            // Return the lowest common ancestor of 'region1' and 'region2'.
            return lowestCommonAncestor;
        }
        // Function to return a list representing the path from the root node
        // to the current node.
        private List<string> FetchPathForRegion(
            string currentNode,
            Dictionary<string, string> childParentMap
        )
        {
            List<string> path = new List<string>();
            // Start by adding the current node to the path.
            path.Add(currentNode);

            // Traverse upwards through the tree by finding the parent of the
            // current node. Continue until the root node is reached.
            while (childParentMap.ContainsKey(currentNode))
            {
                string parentNode = childParentMap[currentNode];
                path.Add(parentNode);
                currentNode = parentNode;
            }
            // Reverse the path so that it starts from the root and
            // ends at the given current node.
            path.Reverse();
            return path;
        }

    }

    /* 1615. Maximal Network Rank
    https://leetcode.com/problems/maximal-network-rank/description/
     */
    public class MaximalNetworkRankSol
    {
        /*         Approach: Finding the in-degree of nodes
        Complexity Analysis
        Here, E is the number of edges and V is the number of nodes in our graph respectively
        •	Time complexity: O(E+V^2)
        o	We iterate on each edge and store both its nodes in the hashmap which will take O(1) time. Thus, for E edges, it will take us O(E) time.
        o	Then we iterate on all possible pairs of the nodes and calculate the network rank which will take O(1) time. Thus, for V(V−1)/2 pairs, it will take O(V^2) time.
        o	Thus, overall we take O(E+V^2) time.
        •	Space complexity: O(E)
        o	We use a hashmap that stores all the edge's nodes in it which will take O(E) space in a fully connected graph.

         */
        public int MaximalNetworkRank(int numberOfNodes, int[][] roads)
        {
            int maxRank = 0;
            Dictionary<int, HashSet<int>> adjacencyList = new Dictionary<int, HashSet<int>>();
            // Construct adjacency list 'adjacencyList', where adjacencyList[node] stores all nodes connected to 'node'.
            foreach (var road in roads)
            {
                if (!adjacencyList.ContainsKey(road[0]))
                {
                    adjacencyList[road[0]] = new HashSet<int>();
                }
                adjacencyList[road[0]].Add(road[1]);

                if (!adjacencyList.ContainsKey(road[1]))
                {
                    adjacencyList[road[1]] = new HashSet<int>();
                }
                adjacencyList[road[1]].Add(road[0]);
            }

            // Iterate on each possible pair of nodes.
            for (int node1 = 0; node1 < numberOfNodes; ++node1)
            {
                for (int node2 = node1 + 1; node2 < numberOfNodes; ++node2)
                {
                    int currentRank = (adjacencyList.ContainsKey(node1) ? adjacencyList[node1].Count : 0) +
                                      (adjacencyList.ContainsKey(node2) ? adjacencyList[node2].Count : 0);

                    // Find the current pair's respective network rank and store if it's maximum till now.
                    if (adjacencyList.ContainsKey(node1) && adjacencyList[node1].Contains(node2))
                    {
                        --currentRank;
                    }
                    maxRank = Math.Max(maxRank, currentRank);
                }
            }
            // Return the maximum network rank.
            return maxRank;
        }
    }

    /* 299. Bulls and Cows
    https://leetcode.com/problems/bulls-and-cows/description/
     */
    class GetHintSol
    {
        /* Approach 1: HashMap: Two Passes */
        public string UsingDictWithTwoPass(string secret, string guess)
        {
            Dictionary<char, int> characterCount = new Dictionary<char, int>();
            foreach (char character in secret)
            {
                if (characterCount.ContainsKey(character))
                {
                    characterCount[character]++;
                }
                else
                {
                    characterCount[character] = 1;
                }
            }

            int bulls = 0, cows = 0;
            int length = guess.Length;
            for (int index = 0; index < length; ++index)
            {
                char character = guess[index];
                if (characterCount.ContainsKey(character))
                {
                    // corresponding characters match
                    if (character == secret[index])
                    {
                        // update the bulls
                        bulls++;
                        // update the cows 
                        // if all character occurrences from secret 
                        // were used up
                        if (characterCount[character] <= 0)
                            cows--;
                        // corresponding characters don't match
                    }
                    else
                    {
                        // update the cows
                        if (characterCount[character] > 0)
                            cows++;
                    }
                    // character was used
                    characterCount[character]--;
                }
            }

            return bulls.ToString() + "A" + cows.ToString() + "B";
        }
        /*         Approach 2: One Pass
        Complexity Analysis
        •	Time complexity: O(N), we do one pass over the strings.
        •	Space complexity: O(1) to keep hashmap (or array) h which contains at most 10 elements.

         */
        public string UsingDictWithOnePass(string secret, string guess)
        {
            Dictionary<char, int> characterCount = new Dictionary<char, int>();

            int bulls = 0, cows = 0;
            int length = guess.Length;
            for (int index = 0; index < length; ++index)
            {
                char secretChar = secret[index];
                char guessChar = guess[index];
                if (secretChar == guessChar)
                {
                    bulls++;
                }
                else
                {
                    if (characterCount.GetValueOrDefault(secretChar, 0) < 0)
                        cows++;
                    if (characterCount.GetValueOrDefault(guessChar, 0) > 0)
                        cows++;
                    characterCount[secretChar] = characterCount.GetValueOrDefault(secretChar, 0) + 1;
                    characterCount[guessChar] = characterCount.GetValueOrDefault(guessChar, 0) - 1;
                }
            }

            return bulls.ToString() + "A" + cows.ToString() + "B";
        }


    }

    /* 681. Next Closest Time
    https://leetcode.com/problems/next-closest-time/description/
     */

    public class NextClosestTimeSol
    {
        /* Approach #1: Simulation [Accepted] 
        Complexity Analysis
•	Time Complexity: O(1). We try up to 24∗60 possible times until we find the correct time.
•	Space Complexity: O(1).

        */
        public string UsingSimulation(string time)
        {
            int currentMinutes = 60 * int.Parse(time.Substring(0, 2));
            currentMinutes += int.Parse(time.Substring(3));
            HashSet<int> allowedDigits = new HashSet<int>();
            foreach (char character in time.ToCharArray())
            {
                if (character != ':')
                {
                    allowedDigits.Add(character - '0');
                }
            }

            while (true)
            {
                currentMinutes = (currentMinutes + 1) % (24 * 60);
                int[] timeDigits = new int[] { currentMinutes / 60 / 10, currentMinutes / 60 % 10, currentMinutes % 60 / 10, currentMinutes % 60 % 10 };
                bool valid = true;
                foreach (int digit in timeDigits)
                {
                    if (!allowedDigits.Contains(digit))
                    {
                        valid = false;
                        break;
                    }
                }
                if (valid) return string.Format("{0:D2}:{1:D2}", currentMinutes / 60, currentMinutes % 60);

            }
        }
        /* Approach #2: Build From Allowed Digits [Accepted]
Complexity Analysis
•	Time Complexity: O(1). We all 44 possible times and take the best one.
•	Space Complexity: O(1).

         */
        public string USingBuildFromAllowedDigits(string time)
        {
            int start = 60 * int.Parse(time.Substring(0, 2));
            start += int.Parse(time.Substring(3));
            int answer = start;
            int elapsed = 24 * 60;
            HashSet<int> allowedDigits = new HashSet<int>();
            foreach (char c in time.ToCharArray())
            {
                if (c != ':')
                {
                    allowedDigits.Add(c - '0');
                }
            }

            foreach (int hourTens in allowedDigits)
            {
                foreach (int hourUnits in allowedDigits)
                {
                    if (hourTens * 10 + hourUnits < 24)
                    {
                        foreach (int minuteTens in allowedDigits)
                        {
                            foreach (int minuteUnits in allowedDigits)
                            {
                                if (minuteTens * 10 + minuteUnits < 60)
                                {
                                    int current = 60 * (hourTens * 10 + hourUnits) + (minuteTens * 10 + minuteUnits);
                                    int candidateElapsed = Math.Abs(current - start) % (24 * 60);
                                    if (0 < candidateElapsed && candidateElapsed < elapsed)
                                    {
                                        answer = current;
                                        elapsed = candidateElapsed;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return string.Format("{0:D2}:{1:D2}", answer / 60, answer % 60);
        }

    }

    /* 1801. Number of Orders in the Backlog
    https://leetcode.com/problems/number-of-orders-in-the-backlog/description/
    https://algo.monster/liteproblems/1801
     */
    public class GetNumberOfBacklogOrdersSol
    {
        /* Time and Space Complexity
    Time Complexity
    The time complexity of the getNumberOfBacklogOrders function primarily depends on the number of orders being processed and the operations performed on the heap for buy and sell orders.
    For each order, we:
    •	Potentially perform a while loop, which can iterate at most 'a' times, where 'a' is the amount of the current order.
    •	Perform heap operations heappop and heappush, which have a time complexity of O(log n) each, where 'n' is the number of orders in the heap.
    However, since each order is only processed once and we push/pop it from the heap, we can simplify the time complexity analysis to concentrating on the heap operations.
    The worst-case scenario is when each order interacts with the heap, leading to O(n log n) time complexity, where 'n' is the total number of orders. This is due to the fact that potentially each order could result in a heappop and a heappush.
    Space Complexity
    The space complexity is determined by the space required to store the buy and sell heaps. In the worst case, all orders could end up in one of the heaps if they cannot be matched and executed.
    Therefore, the space complexity is O(n), where 'n' is the total number of orders.
     */
        public int GetNumberOfBacklogOrders(int[][] orders)
        {
            // A priority queue to store buy orders with the highest price at the top.
            PriorityQueue<int[], int[]> buyQueue = new PriorityQueue<int[], int[]>(
                Comparer<int[]>.Create((a, b) => b[0].CompareTo(a[0]))); //Maxheap
            // A priority queue to store sell orders with the lowest price at the top.
            PriorityQueue<int[], int[]> sellQueue = new PriorityQueue<int[], int[]>(
                Comparer<int[]>.Create((a, b) => a[0].CompareTo(b[0]))); //Minheap

            foreach (int[] order in orders)
            {
                int price = order[0], amount = order[1], orderType = order[2];

                // Processing for a buy order.
                if (orderType == 0)
                {
                    // Attempt to fulfill the buy order with available sell orders.
                    while (amount > 0 && sellQueue.Count > 0 && sellQueue.Peek()[0] <= price)
                    {
                        int[] sellOrder = sellQueue.Dequeue();
                        int sellPrice = sellOrder[0], sellAmount = sellOrder[1];
                        if (amount >= sellAmount)
                        {
                            amount -= sellAmount;
                        }
                        else
                        {
                            // Partially fulfill the buy order and put the remaining sell order back.
                            sellQueue.Enqueue(new int[] { sellPrice, sellAmount - amount }, new int[] { sellPrice, sellAmount - amount });
                            amount = 0;
                        }
                    }
                    // If there is an outstanding amount, add the buy order to the backlog.
                    if (amount > 0)
                    {
                        buyQueue.Enqueue(new int[] { price, amount }, new int[] { price, amount });
                    }
                }
                else
                {
                    // Processing for a sell order.
                    // Attempt to fulfill the sell order with available buy orders.
                    while (amount > 0 && buyQueue.Count > 0 && buyQueue.Peek()[0] >= price)
                    {
                        int[] buyOrder = buyQueue.Dequeue();
                        int buyPrice = buyOrder[0], buyAmount = buyOrder[1];
                        if (amount >= buyAmount)
                        {
                            amount -= buyAmount;
                        }
                        else
                        {
                            // Partially fulfill the sell order and put the remaining buy order back.
                            buyQueue.Enqueue(new int[] { buyPrice, buyAmount - amount }, new int[] { buyPrice, buyAmount - amount });
                            amount = 0;
                        }
                    }
                    // If there is an outstanding amount, add the sell order to the backlog.
                    if (amount > 0)
                    {
                        sellQueue.Enqueue(new int[] { price, amount }, new int[] { price, amount });
                    }
                }
            }

            // Calculate the total number of backlog orders.
            long backlogCount = 0;
            const int modulo = (int)1e9 + 7;

            // Sum the amounts from all buy backlog orders.
            while (buyQueue.Count > 0)
            {
                backlogCount += buyQueue.Dequeue()[1];
            }
            // Sum the amounts from all sell backlog orders.
            while (sellQueue.Count > 0)
            {
                backlogCount += sellQueue.Dequeue()[1];
            }

            // Return the total backlog count modulo 1e9 + 7.
            return (int)(backlogCount % modulo);
        }
    }

    /* 187. Repeated DNA Sequences
    https://leetcode.com/problems/repeated-dna-sequences/description/
     */
    class FindRepeatedDnaSequencesSol
    {
        /* Approach 1: Linear-time Slice Using Substring + HashSet
        Complexity Analysis
•	Time complexity : O((N−L)L), that is O(N) for the constant L=10. In the loop executed N−L+1, one builds a substring of length L. Overall that results in O((N−L)L) time complexity.
•	Space complexity : O((N−L)L) to keep the hashset, that results in O(N) for the constant L=10.

         */
        public List<String> UsingLineartimeSliceWithSubstringAndHashSet(String s)
        {
            int L = 10, n = s.Length;
            HashSet<String> seen = new(), output = new();

            // iterate over all sequences of length L
            for (int start = 0; start < n - L + 1; ++start)
            {
                String tmp = s.Substring(start, start + L);
                if (seen.Contains(tmp)) output.Add(tmp);
                seen.Add(tmp);
            }
            return new List<String>(output);
        }
        /* Approach 2: Rabin-Karp: Constant-time Slice Using Rolling Hash
        Complexity Analysis
•	Time complexity : O(N−L), that is O(N) for the constant L=10. In the loop executed N−L+1 one builds a hash in a constant time, that results in O(N−L) time complexity.
•	Space complexity : O(N−L) to keep the hashset, that results in O(N) for the constant L=10.

        */
        public List<string> UsingRabinKarpAlgoWithRollingHash(string s)
        {
            int sequenceLength = 10, stringLength = s.Length;
            if (stringLength <= sequenceLength) return new List<string>();

            // rolling hash parameters: base a
            int baseA = 4, baseALength = (int)Math.Pow(baseA, sequenceLength);

            // convert string to array of integers
            Dictionary<char, int> charToInt = new Dictionary<char, int>
        {
            { 'A', 0 },
            { 'C', 1 },
            { 'G', 2 },
            { 'T', 3 }
        };
            int[] nums = new int[stringLength];
            for (int i = 0; i < stringLength; ++i) nums[i] = charToInt[s[i]];

            int hashValue = 0;
            HashSet<int> seenHashes = new HashSet<int>();
            HashSet<string> outputSequences = new HashSet<string>();
            // iterate over all sequences of length L
            for (int start = 0; start < stringLength - sequenceLength + 1; ++start)
            {
                // compute hash of the current sequence in O(1) time
                if (start != 0)
                {
                    hashValue = hashValue * baseA -
                    nums[start - 1] * baseALength +
                    nums[start + sequenceLength - 1];
                }
                // compute hash of the first sequence in O(L) time
                else
                {
                    for (int i = 0; i < sequenceLength; ++i) hashValue = hashValue * baseA + nums[i];
                }

                // update output and hashset of seen sequences
                if (seenHashes.Contains(hashValue)) outputSequences.Add(s.Substring(start, sequenceLength));
                seenHashes.Add(hashValue);
            }
            return new List<string>(outputSequences);
        }


        /* Approach 3: Bit Manipulation: Constant-time Slice Using Bitmask
        Complexity Analysis
•	Time complexity : O(N−L), that is O(N) for the constant L=10. In the loop executed N−L+1 one builds a bitmask in a constant time, that results in O(N−L) time complexity.
•	Space complexity : O(N−L) to keep the hashset, that results in O(N) for the constant L=10.

         */
        public List<string> UsingBitMasking(string s)
        {
            int sequenceLength = 10, stringLength = s.Length;
            if (stringLength <= sequenceLength) return new List<string>();

            // rolling hash parameters: base a
            int baseA = 4, baseALength = (int)Math.Pow(baseA, sequenceLength);

            // convert string to array of integers
            Dictionary<char, int> characterToIntMap = new Dictionary<char, int>
        {
            { 'A', 0 },
            { 'C', 1 },
            { 'G', 2 },
            { 'T', 3 }
        };

            int[] numericArray = new int[stringLength];
            for (int index = 0; index < stringLength; ++index)
            {
                numericArray[index] = characterToIntMap[s[index]];
            }

            int bitmask = 0;
            HashSet<int> seenSequences = new HashSet<int>();
            HashSet<string> outputSequences = new HashSet<string>();

            // iterate over all sequences of length L
            for (int startIndex = 0; startIndex < stringLength - sequenceLength + 1; ++startIndex)
            {
                // compute bitmask of the current sequence in O(1) time
                if (startIndex != 0)
                {
                    // left shift to free the last 2 bits
                    bitmask <<= 2;

                    // add a new 2-bits number in the last two bits
                    bitmask |= numericArray[startIndex + sequenceLength - 1];

                    // unset first two bits: 2L-bit and (2L + 1)-bit
                    bitmask &= ~(3 << (2 * sequenceLength));
                }
                // compute hash of the first sequence in O(L) time
                else
                {
                    for (int index = 0; index < sequenceLength; ++index)
                    {
                        bitmask <<= 2;
                        bitmask |= numericArray[index];
                    }
                }

                // update output and hashset of seen sequences
                if (seenSequences.Contains(bitmask))
                {
                    outputSequences.Add(s.Substring(startIndex, sequenceLength));
                }
                seenSequences.Add(bitmask);
            }
            return new List<string>(outputSequences);
        }

    }

    /* 388. Longest Absolute File Path		
    https://leetcode.com/problems/longest-absolute-file-path/description/
    https://algo.monster/liteproblems/388
     */
    class LengthLongestPathSol
    {

        /* Time and Space Complexity
        Time Complexity
        The time complexity of the code is O(N), where N is the length of the input string. This is because the while loop iterates over each character in the string exactly once. The inner while loops are for counting the indentation (by tabs) and for scanning the current file/directory name, but they do not add to the overall time complexity, as they just break the input into logical segments without re-iterating over previously checked characters. Moreover, the stack operations such as push (append) and pop operations take O(1) time per operation. Since the number of operations is bounded by N (you can’t push or pop more than once per character in the string), the stack operations also do not exceed O(N) time complexity.
        Space Complexity
        The space complexity of the code is O(D), where D is the maximum depth of files/subdirectories. This is because the stack stk only stores information about the current path, and this path cannot be deeper than the maximum depth. In the worst case, where the input string is a single deep path, we have to store each part of the path on the stack (the additional cur does not count towards the space complexity since it is an integer and does not grow with input size).
         */
        public int LengthLongestPath(String input)
        {
            int index = 0; // Pointer to iterate over characters in the input string
            int maxLength = 0; // Maximum length of file path
            Stack<int> stack = new(); // Stack to keep track of the lengths of directories

            while (index < input.Length)
            {
                int level = 0; // Level of the current file or directory (number of '\t' characters)

                // Count the level (number of '\t')
                while (index < input.Length && input[index] == '\t')
                {
                    level++;
                    index++;
                }

                int length = 0; // Current directory or file length
                bool isFile = false; // Flag to check if the current path is a file or directory

                // Calculate the length of the current file or directory name
                while (index < input.Length && input[index] != '\n')
                {
                    length++;
                    if (input[index] == '.')
                    {
                        isFile = true; // It's a file if there is a period ('.')
                    }
                    index++;
                }
                index++; // Move to the next character after '\n'

                // If the current level is less than the stack size,
                // it means we have to go up the directory tree
                while (stack.Count > 0 && stack.Count > level)
                {
                    stack.Pop();
                }

                // If the stack is not empty, add the length of the top directory to 'length',
                // plus one for the '\' character.
                if (stack.Count > 0)
                {
                    length += stack.Peek() + 1;
                }

                // If it's not a file, push the length of the current directory onto the stack
                if (!isFile)
                {
                    stack.Push(length);
                }
                else
                {
                    // If it's a file, update maxLength if necessary
                    maxLength = Math.Max(maxLength, length);
                }
            }
            return maxLength; // Return the maximum length
        }
    }


    /* 2391. Minimum Amount of Time to Collect Garbage
    https://leetcode.com/problems/minimum-amount-of-time-to-collect-garbage/description/	
     */
    public class GarbageCollectionSol
    {

        /* Approach 1: HashMap/Dict + Prefix Sum
        Complexity Analysis
Here, N is the number of houses in the array garbage, and K is the maximum length garbage[i].
•	Time complexity O(N∗K)
We first iterate over the array travel to create the prefixSum, the size of travel is N and hence this will take O(N) time. We then iterate over the garbage array and for each string in the array we iterate over each character to store info in the maps garbageLastPos and garbageCount, this operation will take O(N∗K) time. In the end, we just iterate over the three garbage types and add the corresponding answer to ans. Hence, the total time complexity is equal to O(N∗K)
•	Space complexity O(N)
We have created an array prefixSum of size N. We also have the maps to store the last position and the count, however, the space required by these maps can be considered constant as the only keys we need are three (M, P, G). Therefore, the total space complexity can be written as O(N).

         */
        public int UsingDictWithPrefixSum(string[] garbage, int[] travel)
        {
            // Array to store the prefix sum in travel.
            int[] prefixSum = new int[travel.Length + 1];
            prefixSum[1] = travel[0];
            for (int i = 1; i < travel.Length; i++)
            {
                prefixSum[i + 1] = prefixSum[i] + travel[i];
            }

            // Map to store garbage type to the last house index.
            Dictionary<char, int> garbageLastPosition = new Dictionary<char, int>();

            // Map to store the total count of each type of garbage in all houses.
            Dictionary<char, int> garbageCount = new Dictionary<char, int>();
            for (int i = 0; i < garbage.Length; i++)
            {
                foreach (char c in garbage[i])
                {
                    garbageLastPosition[c] = i;
                    garbageCount[c] = garbageCount.GetValueOrDefault(c, 0) + 1;
                }
            }

            string garbageTypes = "MPG";
            int totalTime = 0;
            foreach (char c in garbageTypes)
            {
                // Add only if there is at least one unit of this garbage.
                if (garbageCount.ContainsKey(c))
                {
                    totalTime += prefixSum[garbageLastPosition[c]] + garbageCount[c];
                }
            }

            return totalTime;
        }
        /* Approach 2: HashMap and In-place Modification
Complexity Analysis
Here, N is the number of houses in the array garbage and K is the maximum length of garbage in the array garbage.
•	Time complexity O(N∗K)
We first iterate over the array travel to create the prefixSum, the size of travel is N and hence this will take O(N) time. We then iterate over the garbage array and for each string in the array we iterate over each character to store info in the maps garbageLastPos, this operation will take O(N∗K) time. In the end, we just iterate over the three garbage types and add the corresponding answer to ans. Hence, the total time complexity is equal to O(N∗K)
•	Space complexity O(1)
The only extra space we used is the map to store the last position, however, the space required by this map can be considered constant as the only keys that we need are three (M, P, G). Therefore, the total space complexity is constant.
         */
        public int UsingDictWihtInPlaceModification(string[] garbage, int[] travel)
        {
            // Store the prefix sum in travel itself.
            for (int i = 1; i < travel.Length; i++)
            {
                travel[i] = travel[i - 1] + travel[i];
            }

            // Dictionary to store garbage type to the last house index.
            Dictionary<char, int> garbageLastPosition = new Dictionary<char, int>();
            int totalTime = 0;
            for (int i = 0; i < garbage.Length; i++)
            {
                foreach (char c in garbage[i])
                {
                    garbageLastPosition[c] = i;
                }
                totalTime += garbage[i].Length;
            }

            string garbageTypes = "MPG";
            foreach (char c in garbageTypes)
            {
                // No travel time is required if the last house is at index 0.
                totalTime += (garbageLastPosition.TryGetValue(c, out int lastPosition) && lastPosition == 0
                        ? 0
                        : travel[garbageLastPosition[c] - 1]);
            }

            return totalTime;
        }

        /* Approach 3: Iterate in Reverse
        Complexity Analysis
Here, N is the number of houses in the array garbage and K is the maximum length of garbage in the array garbage.
•	Time complexity O(N∗K)
We iterate over the array garbage in reverse and for each string in the array, we iterate over each character to and do O(1) work, thus this operation will take O(N∗K) time.
•	Space complexity O(1)
The only extra space we used is the three variables M, P, and G. Therefore, the total space complexity is constant.

         */
        public int UsingIterativeInReverse(String[] garbage, int[] travel)
        {
            bool M = false, P = false, G = false;
            int ans = garbage[0].Length;

            for (int i = garbage.Length - 1; i > 0; i--)
            {
                M |= garbage[i].Contains("M");
                P |= garbage[i].Contains("P");
                G |= garbage[i].Contains("G");
                ans +=
                travel[i - 1] * ((M ? 1 : 0) + (P ? 1 : 0) + (G ? 1 : 0)) +
                garbage[i].Length;
            }

            return ans;
        }
    }


    /* 788. Rotated Digits
    https://leetcode.com/problems/rotated-digits/description/
    https://algo.monster/liteproblems/788
     */
    class RotatedDigitsSol
    {

        /* Time and Space Complexity
    The given code defines a function rotatedDigits which takes an integer n and returns the count of numbers less than or equal to n where the digits 2, 5, 6, or 9 appear at least once (making the number 'good'), and the digits 3, 4, or 7 do not appear at all (since they cannot be rotated to a valid number). It does not include numbers that remain the same after being rotated.
    Time Complexity
    The time complexity of the modified code is O(logN * 4^logN) which simplifies to O(N^2) where N is the number of digits in the input number n. This is because there are logN digits in n, and for each digit position, we iterate over up to 4 possible 'good' digits. There is a factor of logN for each digit position since there are that many recursive calls at each level, considering that limit is set to True only when i == up which means the next function call will honour the limit created by the previous digit.
    Using memoization with @cache, we avoid repetitive calculations for each unique combination of the position of the digit, the flag whether we have encountered a 'good' digit, and whether we are limited by the original number's digit at this position (up) which leads to pruning the search space significantly.
    Space Complexity
    The space complexity of the code is O(logN * 2 * logN) which simplifies to O((logN)^2).
    This includes space for:
    •	The memoization cache which could potentially store all states of the function arguments (pos, ok, limit).
    •	The array a of length 6, indicating the input number is decomposed into up to 6 digits, accounting for integers up to a maximum of 999999. However, this 6 is a constant and does not scale with the input, so it does not affect the time complexity.
    Therefore, the space used by the recursion stack and the memoization dictionary depends on the number of different states the dfs function can be in, which is influenced by the number of digits logN (representing different positions) and the bool flags ok and limit, leading to the complexity of O((logN)^2).
     */
        // Member variable to hold digits of the number
        private int[] digits = new int[6];

        // DP cache to store intermediate results, initialized to -1 indicating uncomputed states
        private int[][] dpCache = new int[6][];

        public int RotatedDigits(int n)
        {
            int length = 0; // Will keep track of the number of digits

            // Initialize the dpCache array with -1, except where it's been computed already
            foreach (int[] row in dpCache)
            {
                Array.Fill(row, -1);
            }

            // Backfill the array 'digits' with individual digits of the number n
            while (n > 0)
            {
                digits[++length] = n % 10; // Store each digit
                n /= 10; // Reduce n by a factor of 10
            }

            // Start the depth-first search from most significant digit with 'ok' as false and limit as true
            return DepthFirstSearch(length, 0, true);
        }
        private int DepthFirstSearch(int position, int countValid, bool limit)
        {
            if (position <= 0)
            {
                // Base case: when all positions are processed, check if we
                // have at least one valid digit (2, 5, 6, or 9)
                return countValid;
            }

            // Check the DP cache to avoid re-computation unless we're limited by the current value
            if (!limit && dpCache[position][countValid] != -1)
            {
                return dpCache[position][countValid];
            }

            // Calculate the upper bound for this digit. If we have a limit, we cannot exceed the given digit
            int upperBound = limit ? digits[position] : 9;
            int ans = 0; // To store the number of valid numbers

            for (int i = 0; i <= upperBound; ++i)
            {
                if (i == 0 || i == 1 || i == 8)
                {
                    // Digits 0, 1, and 8 are valid but not counted towards 'countValid' flag
                    ans += DepthFirstSearch(position - 1, countValid, limit && i == upperBound);
                }
                if (i == 2 || i == 5 || i == 6 || i == 9)
                {
                    // Digits 2, 5, 6, and 9 are valid and counted towards 'countValid' flag
                    ans += DepthFirstSearch(position - 1, 1, limit && i == upperBound);
                }
            }

            // Cache the computed value if there was no limit
            if (!limit)
            {
                dpCache[position][countValid] = ans;
            }
            return ans; // Return the calculated number of valid rotated digits
        }
    }


    /* 1882. Process Tasks Using Servers
    https://leetcode.com/problems/process-tasks-using-servers/description/
    https://algo.monster/liteproblems/1882
     */
    public class AssignTasksSol
    {
        public int[] AssignTasks(int[] servers, int[] tasks)
        {
            int numTasks = tasks.Length, numServers = servers.Length;
            // Create a priority queue to store idle servers, prioritizing by weight then index.
            PriorityQueue<int[], int[]> idleServers = new PriorityQueue<int[], int[]>(
                Comparer<int[]>.Create((server1, server2) =>
                    server1[0] == server2[0] ? server1[1] - server2[1] : server1[0] - server2[0]
            ));

            // Create a priority queue to store busy servers, prioritizing by finish time,
            // then weight, then index.
            PriorityQueue<int[], int[]> busyServers = new PriorityQueue<int[], int[]>(
                Comparer<int[]>.Create((server1, server2) =>
                {
                    if (server1[0] == server2[0])
                    {
                        return server1[1] == server2[1] ? server1[2] - server2[2] : server1[1] - server2[1];
                    }
                    return server1[0] - server2[0];
                }
            ));

            // Initialize the idle servers priority queue with all servers.
            for (int i = 0; i < numServers; ++i)
            {
                idleServers.Enqueue(new int[] { servers[i], i }, new int[] { servers[i], i }); // [weight, index]
            }

            // Initialize an array to hold the server assignment results for each task.
            int[] assignedServers = new int[numTasks];
            int assignedTasksCount = 0;

            // Iterate over each second to assign tasks to servers.
            for (int currentTime = 0; currentTime < numTasks; ++currentTime)
            {
                int taskDuration = tasks[currentTime];

                // Release all servers that have completed their tasks by this second.
                while (busyServers.Count > 0 && busyServers.Peek()[0] <= currentTime)
                {
                    int[] server = busyServers.Dequeue();
                    idleServers.Enqueue(new int[] { server[1], server[2] }, new int[] { server[1], server[2] }); // [weight, index]
                }

                // Assign a server to the current task.
                if (idleServers.Count > 0)
                {
                    // If there are available servers, dequeue the best one.
                    int[] server = idleServers.Dequeue();
                    assignedServers[assignedTasksCount++] = server[1];
                    busyServers.Enqueue(new int[] { currentTime + taskDuration, server[0], server[1] }, new int[] { currentTime + taskDuration, server[0], server[1] });
                }
                else
                {
                    // If no servers are idle, dequeue the server that will become available the soonest.
                    int[] server = busyServers.Dequeue();
                    assignedServers[assignedTasksCount++] = server[2];
                    busyServers.Enqueue(new int[] { server[0] + taskDuration, server[1], server[2] }, new int[] { server[0] + taskDuration, server[1], server[2] });
                }
            }
            return assignedServers;
        }
    }

    /* 722. Remove Comments
    https://leetcode.com/problems/remove-comments/description/
     */

    class RemoveCommentsSol
    {
        /* 
        Approach #1: Parsing [Accepted]
        Complexity Analysis
•	Time Complexity: O(S), where S is the total length of the source code.
•	Space Complexity: O(S), the space used by recording the source code into ans.
 */
        public List<string> RemoveComments(string[] source)
        {
            bool isInBlockComment = false;
            StringBuilder currentLine = new StringBuilder();
            List<string> result = new List<string>();

            foreach (string line in source)
            {
                int index = 0;
                char[] characters = line.ToCharArray();
                if (!isInBlockComment)
                {
                    currentLine = new StringBuilder();
                }
                while (index < line.Length)
                {
                    if (!isInBlockComment && index + 1 < line.Length && characters[index] == '/' && characters[index + 1] == '*')
                    {
                        isInBlockComment = true;
                        index++;
                    }
                    else if (isInBlockComment && index + 1 < line.Length && characters[index] == '*' && characters[index + 1] == '/')
                    {
                        isInBlockComment = false;
                        index++;
                    }
                    else if (!isInBlockComment && index + 1 < line.Length && characters[index] == '/' && characters[index + 1] == '/')
                    {
                        break;
                    }
                    else if (!isInBlockComment)
                    {
                        currentLine.Append(characters[index]);
                    }
                    index++;
                }
                if (!isInBlockComment && currentLine.Length > 0)
                {
                    result.Add(currentLine.ToString());
                }
            }
            return result;
        }
    }


    /* 2895. Minimum Processing Time
    https://leetcode.com/problems/minimum-processing-time/description/
    https://algo.monster/liteproblems/2895
     */
    class MinProcessingTimeSol
    {

        /* Time and Space Complexity
        The time complexity of the code is primarily determined by the sorting operations performed on processorTime and tasks which are O(n log n) where n is the number of tasks. Each list is sorted exactly once, and therefore the time complexity remains O(n log n).
        After the sorting, there is a for loop that iterates through processorTime. The loop itself runs in O(m) time, where m is the number of processors. However, this does not affect the overall time complexity since it is assumed that m is much less than n and because the m loop is not nested within an n loop. Thus, the for loop's complexity does not exceed O(n log n) of the sorting step.
        The space complexity of the sort operation depends on the implementation of the sorting algorithm. Typically, the sort method in Python (Timsort) has a space complexity of O(n). However, in the reference answer, they've noted a space complexity of O(log n). This can be considered correct under the assumption that the sorting algorithm used is an in-place sort like heapsort or in-place mergesort which has an O(log n) space complexity due to the recursion stack during the sort, but Python's default sorting algorithm is not in-place and actually takes O(n) space. If the sizes of the processorTime and tasks lists are immutable, and cannot be changed in place, the space complexity could indeed be O(log n) due to the space used by the sorting algorithm's recursion stack
         */

        /**
         * Calculates the minimum amount of time required to process all tasks given an array of processor times.
         *
         * @param processorTimes A list of integers representing the times each processor requires to be ready for a task.
         * @param tasks A list of integers representing the times required to process each task.
         * @return The minimum processing time to complete all tasks.
         */
        public int MinProcessingTime(List<int> processorTimes, List<int> tasks)
        {
            // Sort the processor times in ascending order
            processorTimes.Sort();

            // Sort the tasks in ascending order
            tasks.Sort();

            int minTime = 0; // Variable to store the minimum processing time required
            int taskIndex = tasks.Count - 1; // Start from the last task (which is the largest due to sorting)

            // Iterate over each processor time
            foreach (int processorTime in processorTimes)
            {
                // If there are no more tasks to allocate, break the loop
                if (taskIndex < 0)
                {
                    break;
                }

                // Calculate the total time for current processor by adding its ready time to the task time
                // and update minTime if this is larger than the current minTime
                minTime = Math.Max(minTime, processorTime + tasks[taskIndex]);

                // Move to the task which is 4 positions earlier in the list since there are 4 processors (0-based index)
                taskIndex -= 4;
            }

            // Return the minimum time needed to complete all tasks
            return minTime;
        }
    }

    /* 2483. Minimum Penalty for a Shop
    https://leetcode.com/problems/minimum-penalty-for-a-shop/description/
     */
    class BestClosingTimeSol
    {

        /*         Approach 1: Two Passes
        Complexity Analysis
        Let n be the length of customers.
        •	Time complexity: O(n)
        o	The first traversal is used to calculate the total count of 'Y' in customers, which takes O(n) time.
        o	In each step of the second traversal, we update cur_penalty, min_penalty, and earliest_hour based on the character customers[i], which can be done in constant time. Therefore, the second traversal also takes O(n) time.
        •	Space complexity: O(1)
        o	We only need to update several parameters, cur_penalty, min_penalty and earliest_hour, which takes O(1) space.

         */
        public int UsingTwoPass(String customers)
        {
            int curPenalty = 0;
            for (int i = 0; i < customers.Length; i++)
            {
                if (customers[i] == 'Y')
                {
                    curPenalty++;
                }
            }

            // Start with closing at hour 0, the penalty equals all 'Y' in closed hours.
            int minPenalty = curPenalty;
            int earliestHour = 0;

            for (int i = 0; i < customers.Length; i++)
            {
                char ch = customers[i];

                // If status in hour i is 'Y', moving it to open hours decrement
                // penalty by 1. Otherwise, moving 'N' to open hours increment
                // penatly by 1.
                if (ch == 'Y')
                {
                    curPenalty--;
                }
                else
                {
                    curPenalty++;
                }

                // Update earliestHour if a smaller penatly is encountered.
                if (curPenalty < minPenalty)
                {
                    earliestHour = i + 1;
                    minPenalty = curPenalty;
                }
            }

            return earliestHour;
        }

        /* Approach 2: One Pass
Complexity Analysis
Let n be the length of customers.
•	Time complexity: O(n)
o	In each step of the traversal, we update cur_penalty, min_penalty, and earliest_hour based on the character customers[i], which can be done in constant time. Therefore, the traversal takes O(n) time.
•	Space complexity: O(1)
o	We only need to update several parameters, cur_penalty, min_penalty and earliest_hour, which takes O(1) space.

         */
        public int UsingOnePass(String customers)
        {
            // Start with closing at hour 0 and assume the current penalty is 0.
            int minPenalty = 0, curPenalty = 0;
            int earliestHour = 0;

            for (int i = 0; i < customers.Length; i++)
            {
                char ch = customers[i];

                // If status in hour i is 'Y', moving it to open hours decrement
                // penalty by 1. Otherwise, moving 'N' to open hours increment
                // penatly by 1.
                if (ch == 'Y')
                {
                    curPenalty--;
                }
                else
                {
                    curPenalty++;
                }

                // Update earliestHour if a smaller penatly is encountered.
                if (curPenalty < minPenalty)
                {
                    earliestHour = i + 1;
                    minPenalty = curPenalty;
                }
            }

            return earliestHour;
        }

    }


    /* 1953. Maximum Number of Weeks for Which You Can Work
    https://leetcode.com/problems/maximum-number-of-weeks-for-which-you-can-work/description/
    https://algo.monster/liteproblems/1953
     */
    class MaxNumWeeksForWorkSol
    {
        /* Time and Space Complexity
        Time Complexity
        The time complexity of the solution is O(N), where N is the number of elements in the milestones list. This is because the algorithm needs to iterate through all elements twice: once to calculate the sum s, and once to find the maximum value mx in the list.
        Space Complexity
        The space complexity of the solution is O(1). It only uses a constant amount of extra space to store variables mx, s, and rest, regardless of the size of the input list milestones.
         */
        public long NumberOfWeeks(int[] milestones)
        {
            int maxMilestone = 0; // Initialize variable to store the maximum value of milestones
            long totalMilestonesSum = 0; // Initialize a variable to store the sum of all milestones

            // Iterate through each milestone and calculate the total sum and max milestone
            foreach (int milestone in milestones)
            {
                totalMilestonesSum += milestone; // Add the current milestone to the total sum
                maxMilestone = Math.Max(maxMilestone, milestone); // Update max milestone if the current one is greater
            }

            // Calculate the sum of all milestones except the maximum one
            long rest = totalMilestonesSum - maxMilestone;

            // If the maximum milestone is more than the sum of the rest plus one,
            // then return twice the sum of the rest plus one as maximum weeks
            // This ensures that we cannot complete the maximum milestone if it's too large compared to the others
            if (maxMilestone > rest + 1)
            {
                return rest * 2 + 1;
            }
            else
            {
                // Otherwise, return the total sum of all milestones meaning all milestones can be completed
                return totalMilestonesSum;
            }
        }
    }

    /* 130. Surrounded Regions
    https://leetcode.com/problems/surrounded-regions/description/
     */
    public class SurroundedRegionsSol
    {
        /*         Approach 1: DFS (Depth-First Search)
        Complexity Analysis
        •	Time Complexity: O(N) where N is the number of cells in the board. In the worst case where it contains only the O cells on the board, we would traverse each cell twice: once during the DFS traversal and the other time during the cell reversion in the last step.
        •	Space Complexity: O(N) where N is the number of cells in the board. There are mainly two places that we consume some additional memory.
        o	We keep a list of border cells as starting points for our traversal. We could consider the number of border cells is proportional to the total number (N) of cells.
        o	During the recursive calls of DFS() function, we would consume some space in the function call stack, i.e. the call stack will pile up along with the depth of recursive calls. And the maximum depth of recursive calls would be N as in the worst scenario mentioned in the time complexity.
        o	As a result, the overall space complexity of the algorithm is O(N).

         */
        public void SolveUsingDFS(char[][] board)
        {
            if (board == null || board.Length == 0)
            {
                return;
            }

            this.ROWS = board.Length;
            this.COLS = board[0].Length;
            List<int[]> borders = new List<int[]>();
            // Step 1). construct the list of border cells to iterate over
            for (int r = 0; r < this.ROWS; ++r)
            {
                borders.Add(new int[] { r, 0 });
                borders.Add(new int[] { r, this.COLS - 1 });
            }

            for (int c = 0; c < this.COLS; ++c)
            {
                borders.Add(new int[] { 0, c });
                borders.Add(new int[] { this.ROWS - 1, c });
            }

            // Step 2). mark the escaped cells
            foreach (var pair in borders)
            {
                this.DFS(board, pair[0], pair[1]);
            }

            // Step 3). flip the cells to their correct final states
            for (int r = 0; r < this.ROWS; ++r)
            {
                for (int c = 0; c < this.COLS; ++c)
                {
                    if (board[r][c] == 'O')
                        board[r][c] = 'X';
                    if (board[r][c] == 'E')
                        board[r][c] = 'O';
                }
            }
        }

        int ROWS, COLS;

        protected void DFS(char[][] board, int row, int col)
        {
            if (board[row][col] != 'O')
                return;
            board[row][col] = 'E';
            if (col < this.COLS - 1)
                DFS(board, row, col + 1);
            if (row < this.ROWS - 1)
                DFS(board, row + 1, col);
            if (col > 0)
                DFS(board, row, col - 1);
            if (row > 0)
                DFS(board, row - 1, col);
        }
        /* Approach 2: BFS (Breadth-First Search)
        Complexity
•	Time Complexity: O(N) where N is the number of cells in the board. In the worst case where it contains only the O cells on the board, we would traverse each cell twice: once during the BFS traversal and the other time during the cell reversion in the last step.
•	Space Complexity: O(N) where N is the number of cells in the board. There are mainly two places that we consume some additional memory.
o	We keep a list of border cells as starting points for our traversal. We could consider the number of border cells is proportional to the total number (N) of cells.
o	Within each invocation of BFS() function, we use a queue data structure to hold the cells to be visited. We then need to estimate the upper bound on the size of the queue. Intuitively we could imagine the unfold of BFS as the structure of an onion.
Each layer of the onion represents the cells that has the same distance to the starting point.
Any given moment the queue would contain no more than two layers of onion, which in the worst case might cover almost all cells in the board.
o	As a result, the overall space complexity of the algorithm is O(N).

         */
        public void SolveUsingBFS(char[][] board)
        {
            if (board == null || board.Length == 0)
            {
                return;
            }

            this.ROWS = board.Length;
            this.COLS = board[0].Length;
            List<Pair<int, int>> borders = new List<Pair<int, int>>();
            for (int r = 0; r < this.ROWS; ++r)
            {
                borders.Add(new Pair<int, int>(r, 0));
                borders.Add(new Pair<int, int>(r, this.COLS - 1));
            }

            for (int c = 0; c < this.COLS; ++c)
            {
                borders.Add(new Pair<int, int>(0, c));
                borders.Add(new Pair<int, int>(this.ROWS - 1, c));
            }

            foreach (Pair<int, int> pair in borders)
            {
                this.BFS(board, pair.first, pair.second);
            }

            for (int r = 0; r < this.ROWS; ++r)
            {
                for (int c = 0; c < this.COLS; ++c)
                {
                    if (board[r][c] == 'O')
                        board[r][c] = 'X';
                    if (board[r][c] == 'E')
                        board[r][c] = 'O';
                }
            }
        }

        protected void BFS(char[][] board, int r, int c)
        {
            Queue<Pair<int, int>> queue = new Queue<Pair<int, int>>();
            queue.Enqueue(new Pair<int, int>(r, c));
            while (queue.Count > 0)
            {
                Pair<int, int> pair = queue.Dequeue();
                int row = pair.first, col = pair.second;
                if (board[row][col] != 'O')
                    continue;
                board[row][col] = 'E';
                if (col < this.COLS - 1)
                    queue.Enqueue(new Pair<int, int>(row, col + 1));
                if (row < this.ROWS - 1)
                    queue.Enqueue(new Pair<int, int>(row + 1, col));
                if (col > 0)
                    queue.Enqueue(new Pair<int, int>(row, col - 1));
                if (row > 0)
                    queue.Enqueue(new Pair<int, int>(row - 1, col));
            }
        }
        public class Pair<T1, T2>
        {
            public T1 first;
            public T2 second;

            public Pair(T1 first, T2 second)
            {
                this.first = first;
                this.second = second;
            }
        }

    }


    /* 1654. Minimum Jumps to Reach Home
    https://leetcode.com/problems/minimum-jumps-to-reach-home/description/
    https://algo.monster/liteproblems/1654
     */

    class MinimumJumpsToReachHomeSol
    {
        /*         Time and Space Complexity
        >>The time complexity of the code is O(N + a + b), where N is the maximum position that can be reached considering the constraints of the problem. Since the algorithm uses a breadth-first search and each position is visited at most once, the time complexity is linear in terms of the number of positions checked. The factor of a + b comes from the maximum range we might need to explore to ensure we can reach position x or determine it's impossible. In the worst case, the algorithm explores up to position x + b (since jumping b back from beyond x is the furthest we would need to go) and the positions backwards until reaching 0. Given the constraint on i (which is that 0 <= i < 6000), N effectively becomes 6000, and thus the time complexity simplifies to O(6000) which is constant, so effectively O(1).
        >>The space complexity of the code is O(N), where N again represents the maximum range of positions considered, which is determined to be 6000 based on the limitations placed within the code (0 <= j < 6000). The space complexity arises from the storage of the vis set containing tuples (position, direction) which tracks the visited positions and the direction from which they were reached. The queue q will also store at most O(N) elements as it holds the positions to be explored. Therefore, space complexity also reduces to O(1) since 6000 is the upper bound regardless of the input size.
         */

        public int MinimumJumps(int[] forbidden, int forwardJump, int backwardJump, int target)
        {
            // Create a hash set to store forbidden positions for fast access
            HashSet<int> forbiddenSet = new HashSet<int>(forbidden);

            // Declare a queue to perform BFS
            Queue<int[]> queue = new Queue<int[]>();
            queue.Enqueue(new int[] { 0, 1 }); // Starting at position 0; 1 indicates next jump can be either forward or backward

            // Define upper bound of positions to avoid infinite searching
            const int MAX_POSITION = 6000;

            // Create a visited matrix to track visited positions; two states for forward [][1] or backward [][0] jump
            bool[,] visited = new bool[MAX_POSITION, 2];
            visited[0, 1] = true; // We start at position 0 and can move either forward or backward

            // BFS to find minimum jumps to reach the target
            for (int steps = 0; queue.Count > 0; ++steps)
            {
                // Process nodes that are at the same level
                for (int size = queue.Count; size > 0; --size)
                {
                    int[] currentPosition = queue.Dequeue();
                    int position = currentPosition[0];
                    int canJumpBackward = currentPosition[1];

                    // If current position is the target, return the number of steps taken
                    if (position == target)
                    {
                        return steps;
                    }

                    // Store next positions from current position
                    List<int[]> nextPositions = new List<int[]>();
                    nextPositions.Add(new int[] { position + forwardJump, 1 }); // Always can jump forward

                    // Check if we can jump backward from the current position
                    if (canJumpBackward == 1)
                    {
                        nextPositions.Add(new int[] { position - backwardJump, 0 }); // After jumping back, can't jump back again right away
                    }

                    // Explore next positions
                    foreach (int[] nextPos in nextPositions)
                    {
                        int newPosition = nextPos[0];
                        int newCanJumpBackward = nextPos[1];

                        // Validate new position (not forbidden, within bounds, and not visited)
                        if (newPosition >= 0 &&
                            newPosition < MAX_POSITION &&
                            !forbiddenSet.Contains(newPosition) &&
                            !visited[newPosition, newCanJumpBackward])
                        {

                            // Add valid position to the queue and mark it as visited
                            queue.Enqueue(new int[] { newPosition, newCanJumpBackward });
                            visited[newPosition, newCanJumpBackward] = true;
                        }
                    }
                }
            }

            // If we exhaust the queue without reaching the target, return -1 indicating it's impossible
            return -1;
        }
    }

    /* 1423. Maximum Points You Can Obtain from Cards
    https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/description/
     */

    class MaxPointsCanObtainFromCardsSol
    {
        /* Approach 1: Dynamic Programming
        Complexity Analysis
Let k be the number of cards we need to select.
•	Time complexity: O(k). Here we are using two for loops of length k to calculate the maximum possible score. This gives us O(2⋅k), which in Big O notation is equal to O(k).
•	Space complexity: O(k). Here we are using two arrays to store the total score obtained by selecting i(0<=i<k) cards from the beginning and i cards from the end. This gives us O(2⋅k), which is equal to O(k).

         */
        public int UsingDP(int[] cardPoints, int k)
        {
            int n = cardPoints.Length;

            int[] frontSetOfCards = new int[k + 1];
            int[] rearSetOfCards = new int[k + 1];

            for (int i = 0; i < k; i++)
            {
                frontSetOfCards[i + 1] = cardPoints[i] + frontSetOfCards[i];
                rearSetOfCards[i + 1] = cardPoints[n - i - 1] + rearSetOfCards[i];
            }

            int maxScore = 0;
            // Each i represents the number of cards we take from the front.
            for (int i = 0; i <= k; i++)
            {
                int currentScore = frontSetOfCards[i] + rearSetOfCards[k - i];
                maxScore = Math.Max(maxScore, currentScore);
            }

            return maxScore;
        }

        /* Approach 2: Dynamic Programming - Space Optimized
Complexity Analysis
Let k be the number of cards we need to select.
•	Time complexity: O(k). We are using two for loops of length k for calculation purposes. This gives us O(2⋅k), which in Big O notation is equal to O(k).
•	Space complexity: O(1). No extra space is used since all the calculations are done impromptu.

         */
        public int UsingDPWithSpaceOptimal(int[] cardPoints, int k)
        {
            int frontScore = 0;
            int rearScore = 0;
            int n = cardPoints.Length;

            for (int i = 0; i < k; i++)
            {
                frontScore += cardPoints[i];
            }

            // take all k cards from the beginning
            int maxScore = frontScore;

            // take i from the beginning and k - i from the end
            for (int i = k - 1; i >= 0; i--)
            {
                rearScore += cardPoints[n - (k - i)];
                frontScore -= cardPoints[i];
                int currentScore = rearScore + frontScore;
                maxScore = Math.Max(maxScore, currentScore);
            }

            return maxScore;
        }

        /* Approach 3: Sliding Window
        Complexity Analysis
Let n be the number of cards we need to select.
•	Time complexity: O(n). In the problem, we are iterating over the array of cards twice. So the time complexity will be O(2⋅n) = O(n).
•	Space complexity: O(1) since no extra space is required.

         */
        public int UsingSlidingWindow(int[] cardPoints, int k)
        {
            int startingIndex = 0;
            int presentSubarrayScore = 0;
            int n = cardPoints.Length;
            int requiredSubarrayLength = n - k;
            int minSubarrayScore;
            int totalScore = 0;

            // Total score obtained on selecting all the cards.
            foreach (int i in cardPoints)
            {
                totalScore += i;
            }

            minSubarrayScore = totalScore;

            if (k == n)
            {
                return totalScore;
            }

            for (int i = 0; i < n; i++)
            {
                presentSubarrayScore += cardPoints[i];
                int presentSubarrayLength = i - startingIndex + 1;
                // If a valid subarray (having size cardsPoints.length - k) is possible.
                if (presentSubarrayLength == requiredSubarrayLength)
                {
                    minSubarrayScore = Math.Min(minSubarrayScore, presentSubarrayScore);
                    presentSubarrayScore -= cardPoints[startingIndex++];
                }
            }
            return totalScore - minSubarrayScore;
        }
    }

    /* 2731. Movement of Robots
    https://leetcode.com/problems/movement-of-robots/description/
    https://algo.monster/liteproblems/2731
     */
    class SumDistanceRobotMovementSol
    {
        /* Time and Space Complexity
The given code's time complexity primarily comes from the sorting operation.
•	Sorting a list of n elements typically takes O(n log n) time. This is the dominating factor in the time complexity of this function.
•	The remaining part of the code iterates over the list once, which is an O(n) operation. However, since O(n log n) + O(n) simplifies to O(n log n), the overall time complexity remains O(n log n).
For space complexity:
•	The given code modifies the input list nums in-place and uses a fixed number of integer variables (mod, ans, s). Thus, apart from the input list, only constant extra space is used.
•	However, since the input list nums itself takes O(n) space, and we consider the space taken by inputs for space complexity analysis, the overall space complexity is O(n).
Therefore, we can conclude that:
•	The time complexity of the code is O(n log n).
•	The space complexity of the code is O(n).
 */
        public int SumDistance(int[] numbers, String direction, int distance)
        {
            // Get the length of the input array
            int n = numbers.Length;

            // Create an array to store adjusted distances
            long[] adjustedDistances = new long[n];

            // Calculate adjusted distances based on direction and store them
            for (int i = 0; i < n; ++i)
            {
                // Subtract or add the distance based on if the direction is 'L' or 'R'
                adjustedDistances[i] = (long)numbers[i] + (direction[i] == 'L' ? -distance : distance);
            }

            // Sort the adjusted distances
            Array.Sort(adjustedDistances);

            // Initialize variables for storing the result and the cumulative sum
            long result = 0, cumulativeSum = 0;

            // Define modulo constant for large number handling
            int modulo = (int)1e9 + 7;

            // Calculate the weighted sum of distances and update result
            for (int i = 0; i < n; ++i)
            {
                // Update the result with the current index times the element minus the cumulative sum so far
                result = (result + i * adjustedDistances[i] - cumulativeSum) % modulo;
                // Update cumulative sum with the current element's value
                cumulativeSum += adjustedDistances[i];
            }

            // Return the result cast back to integer
            return (int)result;
        }
    }

    /* 2594. Minimum Time to Repair Cars
    https://leetcode.com/problems/minimum-time-to-repair-cars/description/
    https://algo.monster/liteproblems/2594
     */
    class MinTimeToRepairCarsSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the code is primarily dictated by two factors: the use of binary search and the check function that is called within it.
The binary search is performed on a range of ranks[0] * cars * cars. Since binary search has a time complexity of O(log N) where N is the size of the element space you are searching over, the log component refers to the powers of two that you divide the search space by. Therefore, this portion contributes a log(ranks[0] * cars * cars) complexity.
However, for each step of the binary search, the check function is called. This function iterates over every rank and performs an operation that takes constant time, int(sqrt(t // r)), and sums the results. Since this iteration is over all mechanics which is n in number, the operation within the check function has a complexity of O(n).
Consequently, the overall time complexity of the algorithm is the product of the two, which corresponds to O(n * log(ranks[0] * cars * cars)). It is simplified to O(n * log n) in the reference answer under the assumption that ranks[0] * cars * cars grows proportionally to n^2, making the log(ranks[0] * cars * cars) a constant multiplier of log n.
Space Complexity
The space complexity is O(1) because the algorithm uses a fixed amount of additional space. The variables used within the check function and the storage for the result of bisect_left do not grow with the size of the input list ranks.
 */
        // Repair cars by using ranks of mechanics to calculate the minimum time.
        public long MinTimeToRepairCars(int[] ranks, int totalCars)
        {
            long low = 0; // Set the lower bound of the search space to 0.
                          // Set the upper bound of the search space:
                          // the product of the maximum rank, total number of cars and total cars squared.
            long high = 1L * ranks[0] * totalCars * totalCars;

            // Implement binary search to find the minimum amount of time needed.
            while (low < high)
            {
                long mid = (low + high) >> 1; // Find the midpoint between low and high.
                long count = 0; // Initialize count of cars that can be repaired in 'mid' time.

                // Calculate the number of cars each mechanic can fix in 'mid' time.
                foreach (int rank in ranks)
                {
                    count += (long)Math.Sqrt(mid / rank);
                }

                // If count is at least equal to the total number of cars,
                // we could potentially reduce the high to mid.
                if (count >= totalCars)
                {
                    high = mid;
                }
                else
                {
                    // Otherwise, we have to increase the low to mid + 1 to find the minimum time.
                    low = mid + 1;
                }
            }

            // When low meets high, we've found the minimum time needed for the repairs.
            return low;
        }
    }


    /* 2021. Brightest Position on Street
    https://leetcode.com/problems/brightest-position-on-street/description/
    https://algo.monster/liteproblems/2021
     */
    class BrightestPositionOnStreetSol
    {
        /* Time and Space Complexity
The time complexity of the code is O(N log N) where N is the number of light ranges in the lights list. This complexity arises because we sort the keys of our dictionary d, which contains at most 2N keys (each light contributes two keys: the start and end of its illumination range). Sorting these keys dominates the runtime complexity.
The space complexity of the code is O(N) since we use a dictionary to store the changes to brightness at each key point. In the worst case, if every light has a unique range, the dictionary could have as many as 2N keys, where N is the number of light ranges in the lights list.
 */
        public int BrightestPosition(int[][] lights)
        {
            // Use a Dictionary to easily manage the range of light contributions on the positions
            SortedDictionary<int, int> deltaBrightness = new SortedDictionary<int, int>();

            // Iterate over each light array to calculate the influence ranges and store them
            foreach (int[] light in lights)
            {
                int leftBoundary = light[0] - light[1]; // Calculate left boundary of the light
                int rightBoundary = light[0] + light[1]; // Calculate right boundary of the light

                // Increase brightness at the start of the range
                if (deltaBrightness.ContainsKey(leftBoundary))
                {
                    deltaBrightness[leftBoundary] += 1;
                }
                else
                {
                    deltaBrightness[leftBoundary] = 1;
                }

                // Decrease brightness right after the end of the range
                if (deltaBrightness.ContainsKey(rightBoundary + 1))
                {
                    deltaBrightness[rightBoundary + 1] -= 1;
                }
                else
                {
                    deltaBrightness[rightBoundary + 1] = -1;
                }
            }

            int brightestPosition = 0; // To hold the result position with the brightest light
            int currentBrightness = 0; // Current accumulated brightness
            int maxBrightness = 0; // Max brightness observed at any point

            // Iterate over the entries in the Dictionary
            foreach (var entry in deltaBrightness)
            {
                int changeInBrightness = entry.Value;
                currentBrightness += changeInBrightness; // Apply the change on the current brightness

                // Check if the current brightness is the maximum observed so far
                if (maxBrightness < currentBrightness)
                {
                    maxBrightness = currentBrightness; // Update the maximum brightness
                    brightestPosition = entry.Key; // Update the position of the brightest light
                }
            }

            return brightestPosition; // Return the position with the maximum brightness
        }
    }

    /* 2237. Count Positions on Street With Required Brightness
    https://leetcode.com/problems/count-positions-on-street-with-required-brightness/description/
    https://algo.monster/liteproblems/2237
     */
    class CountPositionsOnStreetWithRequiredBrightnessSol
    {
        /*Time and Space Complexity
Time Complexity
The main operations within the meetRequirement function are as follows:
1.	Iterating over the lights list to populate the differences in the d array. This has a time complexity of O(m), where m is the length of the lights list.
2.	Using the itertools.accumulate function to compute the prefix sums of the array d. This has a time complexity of O(n) since accumulate will sum across the n elements of the d array.
3.	Zipping the accumulated sums with the requirement array and iterating over it to count the number of positions meeting the requirement. The zipping has a time complexity of O(n) since it operates on two arrays of n elements each. The sum operation also takes O(n).
Therefore, the time complexity of the complete function is O(m + n) because O(m) for the iterations over the lights list and O(n) for the operations involving the d array are independent and do not nest.
Space Complexity
For space complexity, the main data structures that are used in the function include:
1.	The difference array d, which has a fixed maximum size due to its initialization. This maximum size (100010) gives us a space complexity of O(1) since it does not scale with the input size n.
2.	The use of itertools.accumulate which generates an iterator. The space taken by this iterator is proportional to the number of elements in d, leading to a space complexity of O(n).
3.	The intermediate tuples created during the zipping process are not stored; they're generated on-the-fly during iteration, making their additional space impact negligible.
In conclusion, the space complexity of the function is O(n) due to the storage requirements of the difference array d as it scales linearly with the input size n.
  */

        // Method to calculate the number of positions that meet the required brightness
        public int MeetRequirement(int n, int[][] lights, int[] requirement)
        {
            // Array 'brightnessChanges' holds the net change in brightness at each position
            int[] brightnessChanges = new int[100010];

            // Loop through the array of lights to populate the 'brightnessChanges' array
            foreach (int[] light in lights)
            {
                // Calculate the effective range of light for each light bulb
                // Make sure the range does not go below 0 or above n-1
                int start = Math.Max(0, light[0] - light[1]);
                int end = Math.Min(n - 1, light[0] + light[1]);

                // Increment brightness at the start position
                ++brightnessChanges[start];
                // Decrement brightness just after the end position
                --brightnessChanges[end + 1];
            }

            int currentBrightness = 0; // Holds the cumulative brightness at each position
            int positionsMeetingReq = 0; // Number of positions meeting the requirement

            // Iterate over positions from 0 to n-1
            for (int i = 0; i < n; ++i)
            {
                // Calculate the current brightness by adding the net brightness change at position i
                currentBrightness += brightnessChanges[i];

                // If current brightness meets or exceeds the requirement at position i, increase count
                if (currentBrightness >= requirement[i])
                {
                    ++positionsMeetingReq;
                }
            }

            // Return the total number of positions meeting the brightness requirement
            return positionsMeetingReq;
        }
    }



    /* 935. Knight Dialer
    https://leetcode.com/problems/knight-dialer/description/	
     */

    class KnightDialerSol
    {
        int[][] memo;
        int n;
        int MOD = (int)1e9 + 7;
        int[][] jumps = {       new int[]{4, 6},        new int[]{6, 8},        new int[]{7, 9},
                new int[]{4, 8},new int[]        {3, 9, 0},new int[]      {},new int[]    {1, 7, 0},
                new int[]{2, 6},new int[]  {1, 3},new int[] {2, 4}    };


        /* Approach 1: Top-Down Dynamic Programming
Complexity Analysis
•	Time complexity: O(n)
If k is the size of the phone pad, then there are O(n⋅k) states to our DP. Because k=10 in this problem, we can treat k as a constant and thus there are O(n) states to our DP.
Due to memoization, we never calculate a state more than once. Since the number of nextSquare is no more than 3 for each square, calculating each state is done in O(1) as we simply perform a for loop that never iterates more than 3 times.
Overall, we calculate O(n) states with each state costing O(1) to calculate. Thus, our time complexity is O(n).
•	Space complexity: O(n)
The recursion call stack will grow to a size of O(n). With memoization, we also store the results to every DP state. As there are O(n) states, we require O(n) space to store all the results.

         */
        public int TopDownDPRecWithMemo(int n)
        {
            this.n = n;
            memo = new int[n + 1][]; //10
            int ans = 0;
            for (int square = 0; square < 10; square++)
            {
                ans = (ans + Dp(n - 1, square)) % MOD;
            }

            return ans;
        }
        private int Dp(int remain, int square)
        {
            if (remain == 0)
            {
                return 1;
            }

            if (memo[remain][square] != 0)
            {
                return memo[remain][square];
            }

            int ans = 0;
            foreach (int nextSquare in jumps[square])
            {
                ans = (ans + Dp(remain - 1, nextSquare)) % MOD;
            }

            memo[remain][square] = ans;
            return ans;
        }

        /* Approach 2: Bottom-Up Dynamic Programming 
Complexity Analysis
•	Time complexity: O(n)
If k is the size of the phone pad, then there are O(n⋅k) states to our DP. Because k=10 in this problem, we can treat k as a constant and thus there are O(n) states to our DP.
Since the number of nextSquare is no more than 3 for each square, calculating each state is done in O(1) as we simply perform a for loop that never iterates more than 3 times.
Overall, we calculate O(n) states with each state costing O(1) to calculate. Thus, our time complexity is O(n).
•	Space complexity: O(n)
The dp table has a size of O(10n)=O(n).

        */
        public int BottomUpDPIterativeWithMemo(int n)
        {


            int MOD = (int)1e9 + 7;
            int[][] dp = new int[n][];
            for (int square = 0; square < 10; square++)
            {
                dp[0][square] = 1;
            }

            for (int remain = 1; remain < n; remain++)
            {
                for (int square = 0; square < 10; square++)
                {
                    int answer = 0;
                    foreach (int nextSquare in jumps[square])
                    {
                        answer = (answer + dp[remain - 1][nextSquare]) % MOD;
                    }

                    dp[remain][square] = answer;
                }
            }

            int ans = 0;
            for (int square = 0; square < 10; square++)
            {
                ans = (ans + dp[n - 1][square]) % MOD;
            }

            return ans;
        }
        /* Approach 3: Space-Optimized Dynamic Programming 
        Complexity Analysis
•	Time complexity: O(n)
If k is the size of the phone pad, then there are O(n⋅k) states to our DP. Because k=10 in this problem, we can treat k as a constant and thus there are O(n) states to our DP.
Since the number of nextSquare is no more than 3 for each square, calculating each state is done in O(1) as we simply perform a for loop that never iterates more than 3 times.
Overall, we calculate O(n) states with each state costing O(1) to calculate. Thus, our time complexity is O(n).
•	Space complexity: O(1)
We are only using two arrays dp and prevDp. Both have a fixed size of 10, and thus use constant space.

        */
        public int BottomUpDPIterativeWithMemoSpaceOptimal(int n)
        {

            int MOD = (int)1e9 + 7;
            int[] dp = new int[10];
            int[] prevDp = new int[10];
            Array.Fill(prevDp, 1);

            for (int remain = 1; remain < n; remain++)
            {
                dp = new int[10];
                for (int square = 0; square < 10; square++)
                {
                    int answer = 0;
                    foreach (int nextSquare in jumps[square])
                    {
                        answer = (answer + prevDp[nextSquare]) % MOD;
                    }

                    dp[square] = answer;
                }

                prevDp = dp;
            }

            int ans = 0;
            for (int square = 0; square < 10; square++)
            {
                ans = (ans + prevDp[square]) % MOD;
            }

            return ans;
        }
        /* Approach 4: Efficient Iteration On States
Complexity Analysis
•	Time complexity: O(n)
We iterate n−1 times, performing O(1) work at each iteration.
Note that while this algorithm has the same time complexity as the first three approaches, it runs much faster practically as there is far less overhead. From testing, this solution runs 10-20x faster than the first approach, despite having the same time complexity!
•	Space complexity: O(1)
We only use a few integer variables.	

         */
        public int UsingEfficientIternationOnStates(int n)
        {
            if (n == 1)
            {
                return 10;
            }

            int A = 4;
            int B = 2;
            int C = 2;
            int D = 1;
            int MOD = (int)1e9 + 7;

            for (int i = 0; i < n - 1; i++)
            {
                int tempA = A;
                int tempB = B;
                int tempC = C;
                int tempD = D;

                A = ((2 * tempB) % MOD + (2 * tempC) % MOD) % MOD;
                B = tempA;
                C = (tempA + (2 * tempD) % MOD) % MOD;
                D = tempC;
            }

            int ans = (A + B) % MOD;
            ans = (ans + C) % MOD;
            return (ans + D) % MOD;
        }
        /* Approach 5: Linear Algebra
Complexity Analysis
•	Time complexity: O(logn)
Each call to multiply runs in O(1) because the size of our matrices is fixed. We call multiply O(logn) times.
Note that we used three nested loops for matrix multiplication which has a cubic time complexity, but we still treat it as O(1) because of the fixed size of the matrices. There also exist faster algorithms that offer more efficient ways to perform matrix multiplication, reducing time complexity for larger matrix sizes. However, we will not delve into these advanced techniques. Interested readers are recommended to explore efficient matrix computation algorithms further.
•	Space complexity: O(1)
We use extra space for the matrices, but the size of the matrices is fixed, thus we use constant space.

         */
        public int UsingLinearAlgebra(int n)
        {
            if (n == 1)
            {
                return 10;
            }

            int[,] A = {
            {0, 0, 0, 0, 1, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 1, 0, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
            {0, 0, 0, 0, 1, 0, 0, 0, 1, 0},
            {1, 0, 0, 1, 0, 0, 0, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
            {0, 1, 0, 1, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 1, 0, 0, 0, 0, 0}
        };

            int[,] v = { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
            int MOD = 1000000007;

            n--;
            while (n > 0)
            {
                if ((n & 1) == 1)
                {
                    v = Multiply(v, A);
                }

                A = Multiply(A, A);
                n >>= 1;
            }

            return (int)((long)v[0, 0] + v[0, 1] + v[0, 2] + v[0, 3] + v[0, 4] + v[0, 5] + v[0, 6] + v[0, 7] + v[0, 8] + v[0, 9]) % MOD;
        }

        private int[,] Multiply(int[,] A, int[,] B)
        {
            int[,] result = new int[A.GetLength(0), B.GetLength(1)];

            for (int i = 0; i < A.GetLength(0); i++)
            {
                for (int j = 0; j < B.GetLength(1); j++)
                {
                    for (int k = 0; k < A.GetLength(1); k++)
                    {
                        result[i, j] = (result[i, j] + A[i, k] * B[k, j]) % MOD;
                    }
                }
            }

            return result;
        }


    }
    /* 
    1079. Letter Tile Possibilities
    https://leetcode.com/problems/letter-tile-possibilities/description/
    https://algo.monster/liteproblems/1079	
     */
    class NumTilePossibilitiesSol
    {
        /* Time and Space Complexity
The given code uses a depth-first search (DFS) strategy with backtracking to generate all unique permutations of the given tiles string. Let's analyze both the time complexity and space complexity of the code.
Time Complexity
The time complexity of this function is determined by the number of recursive calls made to generate all possible sequences, which grows factorially with the number of unique characters in the input string. In the worst case, all characters are unique, which would mean for a string of length n, there would be n! permutations.
At each level of the recursion, we iterate through all the unique characters left in the counter, decreasing the count for that character and then proceeding with the DFS. The recursion goes as deep as the number of characters in the string, and in each level, it branches based on the number of characters left.
Therefore, the upper bound for the time complexity can be represented as O(n!), where n is the length of the tiles string. However, because the actual running time depends on the number of unique characters, if we let k be the number of unique characters, a more precise representation would be O(k^n), because at each step, we can choose to add any of the remaining k characters to the sequence.
Space Complexity
The space complexity is mainly determined by the call stack used for the recursive DFS calls, and the Counter object that maps characters to their counts. The maximum depth of the recursive call stack is n, which is the number of characters in the tiles string. Therefore, the space complexity for the recursion stack is O(n).
The Counter object will have a space complexity of O(k), where k is the number of unique characters in the tiles string, which is less than or equal to n.
Thus, the total space complexity of the algorithm can be considered as O(n) because the recursion depth dominates the space used by the Counter object.
In conclusion, the time complexity of this code is O(k^n) and the space complexity is O(n), where n is the total length of the input string and k is the number of unique characters.
 */
        // Method to calculate the number of possible permutations of the tiles
        public int NumTilePossibilities(String tiles)
        {
            // Array to hold the count of each uppercase letter from A to Z
            int[] count = new int[26];
            // Increment the respective array position for each character in tiles string
            foreach (char tile in tiles)
            {
                count[tile - 'A']++;
            }
            // Start the recursive Depth-First Search (DFS) to calculate permutations
            return dfs(count);
        }

        // Recursive Depth-First Search method to calculate possible permutations
        private int dfs(int[] count)
        {
            int sum = 0; // Initialize sum to hold number of permutations
                         // Iterate over the count array
            for (int i = 0; i < count.Length; i++)
            {
                // If count of a particular character is positive, process it
                if (count[i] > 0)
                {
                    // Increase the sum as we have found a valid character
                    sum++;
                    // Decrease the count for that character as it is being used
                    count[i]--;
                    // Further deep dive into DFS with reduced count
                    sum += dfs(count);
                    // Once DFS is back from recursion, revert the count used for the character
                    count[i]++;
                }
            }
            // Return the sum of permutations
            return sum;
        }
    }

    /* 1817. Finding the Users Active Minutes
    https://leetcode.com/problems/finding-the-users-active-minutes/description/
    https://algo.monster/liteproblems/1817
     */
    class FindingUsersActiveMinutesSol
    {
        /* Time and Space Complexity
        The time complexity of the code is O(n), where n is the length of the logs array. This is because we iterate over each element of the logs array exactly once, adding the timestamp to a set in the dictionary. Since set operations such as .add() have a time complexity of O(1), iterating over the logs array contributes O(n) to the time complexity.
        Further, we also iterate over the values of the dictionary, with the number of values being at most n. For each of the values, we increment the appropriate count in the ans array, which is another O(1) operation. Hence, the total time complexity remains O(n).
        The space complexity of the code is also O(n) because in the worst-case scenario, each log entry could be for a unique user and a unique timestamp, leading to n unique entries in the dictionary. As such, both the dictionary and the set of timestamps could potentially grow linearly with the number of log entries. Therefore, the space used by the dictionary is proportional to the size of the logs array, n.
         */
        public int[] FindingUsersActiveMinutes(int[][] logs, int k)
        {
            // A hashmap to store unique time stamps for each user
            Dictionary<int, HashSet<int>> userActiveTimes = new();

            // Iterate over each log entry
            foreach (int[] log in logs)
            {
                int userId = log[0]; // Extract the user ID
                int timestamp = log[1]; // Extract the timestamp

                // If the user ID doesn't exist in the map, create a new HashSet for that user.
                // Then, add the timestamp to the user's set of active times.
                if (!userActiveTimes.ContainsKey(userId))
                {
                    userActiveTimes[userId] = new HashSet<int>();
                }
                userActiveTimes[userId].Add(timestamp);

            }

            // Initialize an array to store the UAM count for each possible count 1 through k
            int[] answer = new int[k];

            // Iterate over each user's set of timestamps
            foreach (HashSet<int> timeStamps in userActiveTimes.Values)
            {
                // Count the number of unique timestamps for the user and 
                // increment the respective count in the answer array.
                // Subtract 1 from the size to convert the count into a zero-based index.
                answer[timeStamps.Count - 1]++;
            }

            // Return the filled answer array
            return answer;
        }
    }

    /* 
    2268. Minimum Number of Keypresses
    https://leetcode.com/problems/minimum-number-of-keypresses/description/
    https://algo.monster/liteproblems/2268
     */
    class MinimumKeypressesSol
    {

        /* Time and Space Complexity
        The given Python code aims to compute the minimum number of keypresses required to type a string s, where characters in the string are sorted by frequency, and each successive 9 characters require one additional keypress.
        Time Complexity:
        1.	cnt = Counter(s): Creating a counter for the string s has a time complexity of O(n), where n is the length of string s, as we have to count the frequency of each character in the string.
        2.	sorted(cnt.values(), reverse=True): Sorting the values of the counter has a time complexity of O(k log k), where k is the number of distinct characters in the string s. In the worst case (all characters are distinct), k can be at most 26 for lowercase English letters, resulting in O(26 log 26), which is effectively constant time, but in general, sorting is O(k log k).
        3.	The for loop iterates over the sorted frequencies, which in the worst case is k. The operations inside the loop are constant time, so the loop contributes O(k) to the total time complexity.
        The overall time complexity is therefore O(n + k log k + k). Since k is much smaller than n and has an upper limit, we often consider it a constant, leading to a simplified time complexity of O(n).
        Space Complexity:
        1.	cnt = Counter(s): The space complexity of storing the counter is O(k), where k is the number of distinct characters present in s. As with time complexity, k has an upper bound of 26 for English letters, so this is effectively O(1) constant space.
        2.	The space required for the sorted list of frequencies is also O(k). As before, due to the constant limit on k, we consider this O(1).
        3.	The variables ans, i, and j occupy constant space, contributing O(1) to the space complexity.
        Therefore, the total space complexity of the algorithm is O(k), which simplifies to O(1) due to the constant upper bound on k.
        Overall, the given code has a time complexity of O(n) and a constant space complexity of O(1).
         */
        public int MinimumKeypresses(String s)
        {
            // Initialize a frequency array to store occurrences of each letter
            int[] frequency = new int[26];

            // Fill the frequency array with counts of each character in the input string
            foreach (char character in s)
            {
                frequency[character - 'a']++;
            }

            // Sort the frequency array in ascending order
            Array.Sort(frequency);

            // Initialize variable to store the total number of keypresses
            int totalKeypresses = 0;

            // Initialize a variable to determine the number of keypresses per character
            int keypressesPerChar = 1;

            // Loop through the frequency array from the most frequent to the least frequent character
            for (int i = 1; i <= 26; i++)
            {
                // Add to the total keypress count: keypressesPerChar times the frequency of the character
                totalKeypresses += keypressesPerChar * frequency[26 - i];

                // Every 9th character will require an additional keypress
                if (i % 9 == 0)
                {
                    keypressesPerChar++;
                }
            }

            // Return the total minimum number of keypresses needed
            return totalKeypresses;
        }
    }

    /* 2550. Count Collisions of Monkeys on a Polygon
    https://leetcode.com/problems/count-collisions-of-monkeys-on-a-polygon/description/
    https://algo.monster/liteproblems/2550
     */
    class CountCollisionsOfMonkeysOnAPolygonSol
    {
        /* Time and Space Complexity
The given Python code computes the result of 2^n - 2, modulo 10^9 + 7. It uses the built-in pow function optimized for modular exponentiation.
Time Complexity:
The primary operation of computing 2^n modulo 10^9 + 7 is performed using Python's built-in pow function. This function uses fast exponentiation to compute the result, having a time complexity of O(log n), since it effectively halves the exponent in each step of exponentiation.
Post exponentiation, the subtraction and the modulo operation each take constant time, O(1).
Thus, the time complexity of the entire monkeyMove function is O(log n).
Space Complexity:
The space complexity of the code is O(1) since it uses a constant amount of additional space. There are no data structures being used which grow with the input size n. All operations handle intermediate values which require a constant amount of space.
 */
        // This method calculates the number of ways a monkey can move, given `n` movements.
        public int MonkeyMove(int n)
        {
            // Defining the modulo value as 1e9 + 7 to keep the result within integer limits
            int MOD = (int)1e9 + 7;

            // Use the quick power algorithm to calculate 2 raised to the power of `n`, reduce the result by 2, and ensure it's within the modulo value.
            return (QuickPower(2, n, MOD) - 2 + MOD) % MOD;
        }

        // This helper method efficiently calculates (a^b) mod `mod` using the quick power algorithm.
        private int QuickPower(long baseIdx, int exponent, int mod)
        {
            // Initialize the result to 1 (identity for multiplication).
            long result = 1;
            // Iterate as long as the exponent is greater than 0.
            while (exponent > 0)
            {
                // If the current bit of exponent is '1', multiply the result by the current base and take modulo
                if ((exponent & 1) == 1)
                {
                    result = (result * baseIdx) % mod;
                }
                // Square the base and take modulo for the next bit.
                baseIdx = (baseIdx * baseIdx) % mod;
                // Right shift the exponent to check the next bit.
                exponent >>= 1;
            }
            // Casting the long result back to integer before returning.
            return (int)result;
        }
    }


    /* 855. Exam Room
    https://leetcode.com/problems/exam-room/description/
    https://algo.monster/liteproblems/855
     */

    public class ExamRoom
    {
        /* Time and Space Complexity
Time Complexity
•	__init__: The time complexity of initializing the ExamRoom class is O(log n) due to inserting the initial range (-1, n) into the sorted list, which has a time complexity of O(log n) for insertions.
•	seat:
o	The time complexity for fetching the largest interval (first index) is O(1) because the list is already sorted.
o	Insertion of new intervals after seating a student has a time complexity of O(log n) for each insertion, due to the binary search and insertion into the sorted list. Since two insertions take place, this becomes O(log n) overall.
•	leave:
o	Deletion of intervals in the leave operation has a time complexity of O(log n) for each deletion due to binary search within the sorted list. Two deletions take place, therefore, it is O(log n) overall.
o	Insertion of the new interval after leaving has a time complexity of O(log n) due to the same reasons as in the seat operation.
•	add and delete:
o	The add operation includes inserting into the sorted list self.ts, which has a time complexity of O(log n), and updating two dictionaries, which is O(1).
o	The delete operation includes removing from the sorted list self.ts, which has a time complexity of O(log n), and updating two dictionaries, which is also O(1).
Space Complexity
•	The space complexity is O(n) due to the data structures (self.ts, self.left, and self.right) storing information about the intervals. The sorted list and both dictionaries will potentially hold information for all seats, hence it scales with n.
 */
        private SortedSet<int[]> seatSet = new SortedSet<int[]>(
            Comparer<int[]>.Create((a, b) =>
            {
                //TODO: Fix below compares
                /* int distanceA = CalculateDistance(a);
                int distanceB = CalculateDistance(b);
                // Compare by distance, then by starting index if distances are equal
                return distanceA == distanceB ? a[0] - b[0] : distanceB - distanceA; */
                return 0;
            }
            )
        );
        // Maps to track the nearest occupied seat to the left and right of each seat
        private Dictionary<int, int> leftNeighbour = new Dictionary<int, int>();
        private Dictionary<int, int> rightNeighbour = new Dictionary<int, int>();
        private int seatCount;

        public ExamRoom(int n)
        {
            this.seatCount = n;
            // Initialize with a dummy seat segment representing the whole row
            Add(new int[] { -1, seatCount });
        }

        public int Seat()
        {
            // Get the seat segment representing the largest distance between seated students
            int[] segment = seatSet.First();
            int seatPosition = (segment[0] + segment[1]) / 2;
            // Handle cases where we need to seat at the start or the end
            if (segment[0] == -1)
            {
                seatPosition = 0;
            }
            else if (segment[1] == seatCount)
            {
                seatPosition = seatCount - 1;
            }
            // Remove the current segment and add new segments reflecting the new student being seated
            Remove(segment);
            Add(new int[] { segment[0], seatPosition });
            Add(new int[] { seatPosition, segment[1] });
            return seatPosition;
        }

        public void Leave(int p)
        {
            // Find the immediate neighbours of the leaving student
            int leftIndex = leftNeighbour[p];
            int rightIndex = rightNeighbour[p];
            // Remove the segments created by the leaving student
            Remove(new int[] { leftIndex, p });
            Remove(new int[] { p, rightIndex });
            // Create a new segment reflecting the gap left by the student
            Add(new int[] { leftIndex, rightIndex });
        }

        private int CalculateDistance(int[] segment)
        {
            int l = segment[0], r = segment[1];
            // For seats at the beginning or end, use the whole distance minus one
            if (l == -1 || r == seatCount)
            {
                return r - l - 1;
            }
            else
            {
                // Else, use the half the distance between l and r
                return (r - l) / 2;
            }
        }

        private void Add(int[] segment)
        {
            seatSet.Add(segment);
            leftNeighbour[segment[1]] = segment[0];
            rightNeighbour[segment[0]] = segment[1];
        }

        private void Remove(int[] segment)
        {
            seatSet.Remove(segment);
            leftNeighbour.Remove(segment[1]);
            rightNeighbour.Remove(segment[0]);
        }

    }


    /* 2061. Number of Spaces Cleaning Robot Cleaned
    https://leetcode.com/problems/number-of-spaces-cleaning-robot-cleaned/description/
     */
    class NumberOfCleanRoomsSol
    {


        private readonly int[] DIRECTIONS = { 0, 1, 0, -1, 0 };

        /* Approach 1: Recursive Simulation
        Complexity Analysis
Let n be the number of rows in the room and m be the number of columns. There are a total of m⋅n spaces in the room.
•	Time complexity: O(m⋅n)
We use hash sets for visited and cleaned, which provide constant look-up times in the average case.
The clean function recursively calls itself. We used the visited set, so each cell may be visited four times, once for each direction. This means the clean function can be called up to 4⋅m⋅n times.
Therefore, the overall time complexity is O(4⋅m⋅n), which we can simplify to O(m⋅n).
•	Space complexity: O(m⋅n)
The visited set can store up to four entries for each cell in the room, or 4⋅m⋅n entries. The cleaned set can store up to m⋅n entries.
The clean function can be called up to 4⋅m⋅n times, so the recursive call stack can use up to O(4⋅m⋅n) space.
Therefore, the overall space complexity is O(4mn+4mn+mn), which we can simplify to O(m⋅n).

         */
        public int UsingRecursiveSimulation(int[][] room)
        {
            int rows = room.Length;
            int cols = room[0].Length;
            HashSet<String> visited = new HashSet<String>();
            HashSet<String> cleaned = new HashSet<String>();
            return Clean(room, rows, cols, 0, 0, 0, visited, cleaned);
        }

        private int Clean(
            int[][] room,
            int rows,
            int cols,
            int row,
            int col,
            int direction,
            HashSet<String> visited,
            HashSet<String> cleaned
        )
        {
            // If the robot already visited this space facing this direction
            // Return the number of spaces cleaned
            if (visited.Contains(row + "," + col + "," + direction))
            {
                return cleaned.Count;
            }

            // Mark the space as visited facing this direction and cleaned
            visited.Add(row + "," + col + "," + direction);
            cleaned.Add(row + "," + col);

            // Clean the next space straight ahead if it's empty and in the room
            int nextRow = row + DIRECTIONS[direction];
            int nextCol = col + DIRECTIONS[direction + 1];
            if (
                0 <= nextRow &&
                nextRow < rows &&
                0 <= nextCol &&
                nextCol < cols &&
                room[nextRow][nextCol] == 0
            )
            {
                return Clean(
                    room,
                    rows,
                    cols,
                    nextRow,
                    nextCol,
                    direction,
                    visited,
                    cleaned
                );
            }

            // Otherwise turn right and clean the current space
            return Clean(
                room,
                rows,
                cols,
                row,
                col,
                (direction + 1) % 4,
                visited,
                cleaned
            );
        }
        /* Approach 2: Iterative Simulation
        Complexity Analysis
Let n be the number of rows in the room and m be the number of columns. There are a total of m⋅n spaces in the room.
•	Time complexity: O(m⋅n)
We may visit each space once facing each direction. Hence, the loop may run 4⋅m⋅n times. Therefore, the overall time complexity is O(4⋅m⋅n), which we can simplify to O(m⋅n).
•	Space complexity: O(1)
We use a few variables but no data structures that grow with the input size, so the space complexity is constant, i.e. O(1).	

         */
        public int UsingIterativeSimulation(int[][] room)
        {
            int rows = room.Length;
            int cols = room[0].Length;
            int cleaned = 0;

            int row = 0, col = 0;
            int direction = 0;

            // Clean until we revisit a space facing the same direction
            while (((room[row][col] >> (direction + 1)) & 1) == 0)
            {
                // If the robot hasn't cleaned this space yet, increment cleaned
                if (room[row][col] == 0)
                {
                    cleaned += 1;
                }

                // Mark the space as visited facing this direction
                room[row][col] |= 1 << (direction + 1);

                // Clean the next space straight ahead if it's empty and in the room
                int nextRow = row + DIRECTIONS[direction];
                int nextCol = col + DIRECTIONS[direction + 1];
                if (
                    0 <= nextRow &&
                    nextRow < rows &&
                    0 <= nextCol &&
                    nextCol < cols &&
                    room[nextRow][nextCol] != 1
                )
                {
                    row = nextRow;
                    col = nextCol;
                }
                else
                {
                    // Otherwise turn right and clean the current space
                    direction = (direction + 1) % 4;
                }
            }
            return cleaned;
        }
    }

    /* 2596. Check Knight Tour Configuration
    https://leetcode.com/problems/check-knight-tour-configuration/description/
    https://algo.monster/liteproblems/2596
     */
    class CheckKnightTourConfigurationSol
    {
        /* Time and Space Complexity
The time complexity of the code is O(n^2). This is because the main computations are in the nested for-loops, which iterate over every cell in the given grid. Since the grid is n by n, the iterations run for n^2 times.
The space complexity of the code is O(n^2) as well. This stems from creating a list called pos of size n * n, which is used to store the positions of the integers from the grid in a 1D array. Since the grid stores n^2 elements, and we're converting it to a 1D array, the space taken by pos will also be n^2.
 */
        // Function to check if a given grid represents a valid grid for the given conditions
        public bool CheckValidGrid(int[][] grid)
        {
            // Check if the first element is 0 as required
            if (grid[0][0] != 0)
            {
                return false;
            }

            // Calculate the size of the grid
            int gridSize = grid.Length;

            // Create a position array to store positions of each number in the grid
            int[][] positions = new int[gridSize * gridSize][];
            for (int row = 0; row < gridSize; ++row)
            {
                for (int col = 0; col < gridSize; ++col)
                {
                    // Storing the current number's position
                    positions[grid[row][col]] = new int[] { row, col };
                }
            }

            // Loop to check the validity of the grid based on the position of consecutive numbers
            for (int i = 1; i < gridSize * gridSize; ++i)
            {
                // Get the positions of the current and previous numbers
                int[] previousPosition = positions[i - 1];
                int[] currentPosition = positions[i];

                // Calculate the distance between the current number and the previous number
                int dx = Math.Abs(previousPosition[0] - currentPosition[0]);
                int dy = Math.Abs(previousPosition[1] - currentPosition[1]);

                // Check if the distance satisfies the condition for a knight's move in chess
                bool isValidMove = (dx == 1 && dy == 2) || (dx == 2 && dy == 1);
                if (!isValidMove)
                {
                    // If the move is not valid, the grid is not valid
                    return false;
                }
            }

            // If all moves are valid, the grid is valid
            return true;
        }
    }

    /* 1014. Best Sightseeing Pair
    https://leetcode.com/problems/best-sightseeing-pair/description/
    https://algo.monster/liteproblems/1014
     */
    class MaxScoreSightseeingPairSol
    {
        /* Time and Space Complexity
The given code defines a method to find the maximum score of a sightseeing pair where the score is defined by the sum of the values of the pair reduced by the distance between them (i.e., values[i] + values[j] + i - j for a pair (i, j)).
Time Complexity
The time complexity of the code is O(n), where n is the length of the input list values.
This is because there is a single for-loop that goes through the array values from the second element to the last, doing constant time operations within the loop such as computing the maximum of ans and updating the value of mx. There are no nested loops or other operations that would increase the time complexity beyond linear time.
Space Complexity
The space complexity of the code is O(1).
There are only a few variables used (ans, mx, and j) and their memory consumption does not depend on the input size, which means that there is a constant amount of extra space used regardless of the size of values.
 */
        public int maxScoreSightseeingPair(int[] values)
        {
            // Initialize the answer to 0. This will hold the maximum score.
            int maxScore = 0;

            // Initialize the maximum value seen so far, which is the value at the 0th index 
            // plus its index (because for the first element, index is 0, so it's just the value).
            int maxValueWithIndex = values[0];

            // Iterate over the array starting from the 1st index since we've already considered the 0th index.
            for (int j = 1; j < values.Length; ++j)
            {
                // Update maxScore with the maximum of the current maxScore and 
                // the score of the current sightseeing spot combined with the previous maximum.
                // This score is computed as the value of the current element plus its 'value' score (values[j])
                // subtracted by its distance from the start (j) plus the maxValueWithIndex.
                maxScore = Math.Max(maxScore, values[j] - j + maxValueWithIndex);

                // Update the maxValueWithIndex to be the maximum of the current maxValueWithIndex and
                // the 'value' score of current element added to its index (values[j] + j).
                // This accounts for the fact that as we move right, our index increases,
                // which decreases our score, so we need to keep track of the element
                // which will contribute the most to the score including the index.
                maxValueWithIndex = Math.Max(maxValueWithIndex, values[j] + j);
            }

            // Return the maximum score found.
            return maxScore;
        }
    }

    /* 1733. Minimum Number of People to Teach
    https://leetcode.com/problems/minimum-number-of-people-to-teach/description/
    https://algo.monster/liteproblems/1733
     */
    public class MinimumTeachingsSol
    {
        /* Time and Space Complexity
Time Complexity
The provided code consists of two major parts - the check for whether each pair of friends speak a common language and the counting of languages among the users who need to be taught.
1.	The check function involves two nested loops over the languages spoken by users u and v. If m represents the maximum number of languages any user knows, this function could take up to O(m^2) time in the worst case.
2.	The outer loop for iterating over the friendships list, which calls the check function, runs k times where k is the number of friendships. So, this part of the algorithm will take O(k * m^2) time.
3.	In the worst case, the set s could include all users, which will, therefore, contain at most 2 * k elements, as each friendship involves two users.
4.	The counting of languages again requires an iteration over set s, and for each user in s, iterating over the maximum number of languages they know. In the worst case, this is O(k * m).
5.	The final step is finding the maximum value in the counter, which can take O(n) time, where n is the total number of languages.
Combining these elements, the worst-case time complexity of the code is the sum of these complexities: O(k * m^2 + k * m + n).
Space Complexity
1.	The check function operates in constant space.
2.	The set s can contain up to 2 * k elements, therefore O(k) space complexity.
3.	The cnt Counter objects can contain at most n key-value pairs representing the languages, thus O(n) space is needed.
4.	Temporary space for iterating and counting - This space is constant.
The combined space complexity of the provided code is the maximum of these, which is O(k + n).	
 */
        // Method to compute the minimum number of teachings required
        public int MinimumTeachings(int totalLanguages, int[][] userLanguages, int[][] friendships)
        {
            // Set to maintain unique users that need a language teaching
            HashSet<int> usersToTeach = new HashSet<int>();

            // Check for each pair of friendships whether they have a common language
            foreach (int[] friendship in friendships)
            {
                int user1 = friendship[0];
                int user2 = friendship[1];
                // If two users do not share a common language, add them to the set
                if (!ShareCommonLanguage(user1, user2, userLanguages))
                {
                    usersToTeach.Add(user1);
                    usersToTeach.Add(user2);
                }
            }

            // If no teaching is required, return 0
            if (usersToTeach.Count == 0)
            {
                return 0;
            }

            // Array to count how many users know each language
            int[] languageCount = new int[totalLanguages + 1];

            // For all users that need teaching, count the languages they know
            foreach (int user in usersToTeach)
            {
                foreach (int language in userLanguages[user - 1])
                {
                    languageCount[language]++;
                }
            }

            // Find the language known by the maximum number of users
            int maxLanguageCount = 0;
            foreach (int count in languageCount)
            {
                maxLanguageCount = Math.Max(maxLanguageCount, count);
            }

            // The minimum number of teachings is the number of users to teach
            // minus the maximum common language count
            return usersToTeach.Count - maxLanguageCount;
        }

        // Helper method to check if two users share a common language
        private bool ShareCommonLanguage(int user1, int user2, int[][] userLanguages)
        {
            foreach (int language1 in userLanguages[user1 - 1])
            {
                foreach (int language2 in userLanguages[user2 - 1])
                {
                    if (language1 == language2)
                    {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    /* 1109. Corporate Flight Bookings
    https://leetcode.com/problems/corporate-flight-bookings/description/
    https://algo.monster/liteproblems/1109
     */
    class CorpFlightBookingsSol
    {

        /*Time and Space Complexity
        The time complexity of the provided code is composed of two parts: the iteration through the bookings list and the accumulation of the values in the ans array.
        The iteration through the bookings list happens once for each booking. For each booking, the code performs constant-time operations: an increment at the start position and a decrement at the end position. Therefore, if there are k bookings, this part has a time complexity of O(k).
        The accumulate function is used to compute the prefix sums of the ans array, which has n elements. This operation is linear in the number of flights, so it has a time complexity of O(n).
        Combining these two parts, the overall time complexity of the code is O(k + n), since we have to consider both the number of bookings and the number of flights.
        The space complexity of the code is primarily determined by the storage used for the ans array, which has n elements. The accumulate function returns an iterator in Python, so it does not add additional space complexity beyond what is used for the ans array. Therefore, the space complexity of the provided code is O(n).
          */
        /* 
        This method calculates the number of seats booked on each flight.

        Parameters:
        bookings - an array of bookings, where bookings[i] = [first_i, last_i, seats_i]
        n - the number of flights

        Returns:
        an array containing the total number of seats booked for each flight.
        */
        public int[] CorpFlightBookings(int[][] bookings, int n)
        {
            // Initialize an array to hold the answer, with n representing the number of flights.
            int[] answer = new int[n];

            // Iterate over each booking.
            foreach (int[] booking in bookings)
            {
                // Extract the start and end flight numbers and the number of seats booked.
                int startFlight = booking[0];
                int endFlight = booking[1];
                int seats = booking[2];

                // Increment the seats for the start flight by the number of seats booked.
                answer[startFlight - 1] += seats;

                // If the end flight is less than the number of flights,
                // decrement the seats for the flight immediately after the end flight.
                if (endFlight < n)
                {
                    answer[endFlight] -= seats;
                }
            }

            // Iterate over the answer array, starting from the second element,
            // and update each position with the cumulative sum of seats booked so far.
            for (int i = 1; i < n; ++i)
            {
                answer[i] += answer[i - 1];
            }

            // Return the populated answer array.
            return answer;
        }
    }


    /* 2222. Number of Ways to Select Buildings
    https://leetcode.com/problems/number-of-ways-to-select-buildings/description/
    https://algo.monster/liteproblems/2222
     */
    class NumberOfWaysToSelectBuildingsSol
    {
        /*Time and Space Complexity
The given Python code aims to count the number of ways to select three characters from the string s, such that the selected characters form the pattern "010" or "101".
Time Complexity
The time complexity of the code is O(n), where n is the length of the string s.
•	Calculating cnt0 using s.count("0") requires a single pass over the string, which is O(n).
•	The subsequent loop iterates over each character in the string once, which is O(n).
•	Inside the loop, the operations are constant time, such as updating counters and calculating the values for ans.
Therefore, the overall time complexity is the sum of the two O(n) operations, which is still O(n) since constant factors are ignored.
Space Complexity
The space complexity of the code is O(1).
•	The variables cnt0, cnt1, c0, c1, and ans are all integer counters that use a fixed amount of space.
•	No additional data structures that grow with the input size are used.
Thus, the space required does not scale with the size of the input s, resulting in a constant space complexity.
*/

        // Method to count the number of ways to form a "010" or "101" pattern in the given string.
        public long NumberOfWays(String s)
        {
            // Length of the input string.
            int length = s.Length;
            // Counter for zeros in the input string.
            int countZeros = 0;

            // Count the number of zeros in the input string.
            foreach (char c in s)
            {
                if (c == '0')
                {
                    countZeros++;
                }
            }

            // Counter for ones, which is the total length minus the number of zeros.
            int countOnes = length - countZeros;
            // Variable to store the total number of patterns found.
            long totalWays = 0;
            // Temp counters for zeros and ones as we iterate through the string.
            int tempCountZeros = 0, tempCountOnes = 0;

            // Iterate through the characters of the string to count the patterns.
            foreach (char c in s)
            {
                if (c == '0')
                {
                    // When we find a '0', we increase the total count of valid patterns found
                    // by the number of '1's found before multiplied by the number of '1's that
                    // can potentially come after this '0' to complete the pattern.
                    totalWays += tempCountOnes * (countOnes - tempCountOnes);
                    // Increase the temporary count of zeros since we encountered a '0'.
                    tempCountZeros++;
                }
                else
                {
                    // Similarly, when we find a '1', we increase the count of valid patterns by
                    // the temporary count of '0's multiplied by the number of '0's that can come
                    // after to complete the pattern.
                    totalWays += tempCountZeros * (countZeros - tempCountZeros);
                    // Increase the temporary count of ones since we encountered a '1'.
                    tempCountOnes++;
                }
            }

            // Return the total number of patterns found.
            return totalWays;
        }
    }


    /* 1391. Check if There is a Valid Path in a Grid
    https://leetcode.com/problems/check-if-there-is-a-valid-path-in-a-grid/description/
    https://algo.monster/liteproblems/1391
     */
    public class HasValidPathSol
    {
        private int[] parent;
        private int[][] grid;
        private int numberOfRows;
        private int numberOfColumns;

        /* Time and Space Complexity
        The given Python code uses Union-Find (Disjoint Set Union) to check if there is a valid path in a grid. Union-Find operations (find and union-by-rank) normally have an amortized time complexity of O(α(n)), where α(n) is the inverse Ackermann function and is nearly constant (α(n) <= 4 for any practical value of n).
        •	Time Complexity:
        o	The time complexity is dictated by the number of calls to the find and union operations within the loops. There are m rows and n columns, so we have m * n cells in the grid.
        o	The find operation is called up to four times per cell (left, right, up, down), which means up to 4 * m * n find operations.
        o	However, each find operation takes O(α(m * n)) time due to path compression.
        o	The actual time complexity is therefore O(4 * m * n * α(m * n)), which simplifies to O(m * n * α(m * n)).
        •	Space Complexity:
        o	The parent array p has m * n elements, which is the main space complexity contributor.
        o	Space complexity is therefore O(m * n) because this array scales linearly with the number of cells in the grid.
        In summary, the time complexity is O(m * n * α(m * n)) and the space complexity is O(m * n).
         */

        /// <summary>
        /// Checks if there is a valid path in the grid.
        /// </summary>
        /// <param name="grid">The grid representation where each cell has a street piece.</param>
        /// <returns>True if there's a valid path from top-left to bottom-right, false otherwise.</returns>
        public bool HasValidPath(int[][] grid)
        {
            this.grid = grid;
            numberOfRows = grid.Length;
            numberOfColumns = grid[0].Length;
            parent = new int[numberOfRows * numberOfColumns]; // Array to represent the union-find structure

            // Initialize union-find structure, each cell is its own parent initially
            for (int index = 0; index < parent.Length; ++index)
            {
                parent[index] = index;
            }

            // Union adjacent compatible cells
            for (int row = 0; row < numberOfRows; ++row)
            {
                for (int column = 0; column < numberOfColumns; ++column)
                {
                    int streetPiece = grid[row][column];
                    switch (streetPiece)
                    {
                        case 1:
                            UnionLeft(row, column);
                            UnionRight(row, column);
                            break;
                        case 2:
                            UnionUp(row, column);
                            UnionDown(row, column);
                            break;
                        case 3:
                            UnionLeft(row, column);
                            UnionDown(row, column);
                            break;
                        case 4:
                            UnionRight(row, column);
                            UnionDown(row, column);
                            break;
                        case 5:
                            UnionLeft(row, column);
                            UnionUp(row, column);
                            break;
                        case 6:
                            UnionRight(row, column);
                            UnionUp(row, column);
                            break;
                        default:
                            break;
                    }
                }
            }

            // Check if top left cell and bottom right cell are connected
            return Find(0) == Find(numberOfRows * numberOfColumns - 1);
        }

        /// <summary>
        /// Finds the root of x using path compression.
        /// </summary>
        /// <param name="x">The node to find the root of.</param>
        /// <returns>The root of x.</returns>
        private int Find(int x)
        {
            if (parent[x] != x)
            {
                parent[x] = Find(parent[x]);
            }
            return parent[x];
        }

        /// <summary>
        /// Union the current cell with the cell to the left if compatible.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <param name="column">The column index.</param>
        private void UnionLeft(int row, int column)
        {
            if (column > 0 && (grid[row][column - 1] == 1 || grid[row][column - 1] == 4 || grid[row][column - 1] == 6))
            {
                parent[Find(row * numberOfColumns + column)] = Find(row * numberOfColumns + column - 1);
            }
        }

        /// <summary>
        /// Union the current cell with the cell to the right if compatible.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <param name="column">The column index.</param>
        private void UnionRight(int row, int column)
        {
            if (column < numberOfColumns - 1 && (grid[row][column + 1] == 1 || grid[row][column + 1] == 3 || grid[row][column + 1] == 5))
            {
                parent[Find(row * numberOfColumns + column)] = Find(row * numberOfColumns + column + 1);
            }
        }

        /// <summary>
        /// Union the current cell with the cell above if compatible.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <param name="column">The column index.</param>
        private void UnionUp(int row, int column)
        {
            if (row > 0 && (grid[row - 1][column] == 2 || grid[row - 1][column] == 3 || grid[row - 1][column] == 4))
            {
                parent[Find(row * numberOfColumns + column)] = Find((row - 1) * numberOfColumns + column);
            }
        }

        /// <summary>
        /// Union the current cell with the cell below if compatible.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <param name="column">The column index.</param>
        private void UnionDown(int row, int column)
        {
            if (row < numberOfRows - 1 && (grid[row + 1][column] == 2 || grid[row + 1][column] == 5 || grid[row + 1][column] == 6))
            {
                parent[Find(row * numberOfColumns + column)] = Find((row + 1) * numberOfColumns + column);
            }
        }
    }


    /* 794. Valid Tic-Tac-Toe State
    https://leetcode.com/problems/valid-tic-tac-toe-state/description/
    https://algo.monster/liteproblems/794
     */
    class ValidTicTacToeStateSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the validTicTacToe function is O(1) because the size of the board is fixed at 3x3, and the algorithm iterates over the 9 cells of the board a constant number of times to count occurrences of 'X' and 'O', and to check for wins.
To calculate x and o, we have two double loops that go through the 3x3 board, which would normally result in a time complexity of O(n^2). However, since the board size is constant and does not grow with input, it results in a fixed number of operations that do not depend on any input size variable, so it is O(1).
The win function is called at most two times (once for 'X' and once for 'O'). Within each call, it goes through each row, each column, and both diagonals to check for a win condition, which again, since the board is a fixed 3x3 size, results in a constant number of operations that are O(1).
Space Complexity
The space complexity of the validTicTacToe function is also O(1). No additional space is used that grows with the input size. The function only uses a fixed number of variables (x, o, and the board itself) and the space taken by the recursive stack during the calls to win doesn't depend on the input size since the depth of recursion is not affected by the input but by the fixed size of the 3x3 board.
 */

        private String[] board;

        // Checks if the given Tic-Tac-Toe board state is valid
        public bool ValidTicTacToe(String[] board)
        {
            this.board = board;
            int countX = Count('X'), countO = Count('O');

            // 'X' goes first so there must be either the same amount of 'X' and 'O'
            // or one more 'X' than 'O'
            if (countX != countO && countX - 1 != countO)
            {
                return false;
            }

            // If 'X' has won, there must be one more 'X' than 'O'
            if (HasWon('X') && countX - 1 != countO)
            {
                return false;
            }

            // If 'O' has won, there must be the same number of 'X' and 'O'
            return !(HasWon('O') && countX != countO);
        }

        // Checks if the given player has won
        private bool HasWon(char player)
        {
            // Check all rows and columns
            for (int i = 0; i < 3; ++i)
            {
                if (board[i][0] == player && board[i][1] == player && board[i][2] == player)
                {
                    return true;
                }
                if (board[0][i] == player && board[1][i] == player && board[2][i] == player)
                {
                    return true;
                }
            }
            // Check both diagonals
            if (board[0][0] == player && board[1][1] == player && board[2][2] == player)
            {
                return true;
            }
            return board[0][2] == player && board[1][1] == player && board[2][0] == player;
        }

        // Counts the number of times the given character appears on the board
        private int Count(char character)
        {
            int count = 0;
            foreach (String row in board)
            {
                foreach (char cell in row)
                {
                    if (cell == character)
                    {
                        ++count;
                    }
                }
            }
            return count;
        }
    }


    /* 1267. Count Servers that Communicate
    https://leetcode.com/problems/count-servers-that-communicate/description/
    https://algo.monster/liteproblems/1267
     */

    class CountServersThatCommunicateSol
    {
        /* Time and Space Complexity
        Time Complexity
        The given code consists of two main parts: The first part is a double loop that goes through the entire grid to count the number of servers in each row and column. Since this loop goes through all the elements of the m x n grid exactly once, the time complexity for this part is O(m * n).
        The second part of the code calculates the sum with a generator expression that also iterates through every element in the grid. It checks if there's a server in the given cell (grid[i][j]) and if there is more than one server in the corresponding row or column. Since this is done for each element, it also has a time complexity of O(m * n).
        Therefore, the total time complexity of the entire function is O(m * n) + O(m * n), which simplifies to O(m * n) because we only take the highest order term for big O notation.
        Space Complexity
        For space complexity, the code uses additional arrays row and col to store the counts of servers in each row and column, respectively. The size of row is m and the size of col is n. Hence, the extra space used is O(m + n).
        In conclusion, the time complexity is O(m * n) and the space complexity is O(m + n).
         */
        public int CountServers(int[][] grid)
        {
            int numRows = grid.Length; // Number of rows in the grid
            int numCols = grid[0].Length; // Number of columns in the grid

            // Arrays to store the count of servers in each row and column
            int[] rowCount = new int[numRows];
            int[] colCount = new int[numCols];

            // First iteration to fill in the rowCount and colCount arrays
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    // Count servers in each row and column
                    if (grid[i][j] == 1)
                    {
                        rowCount[i]++;
                        colCount[j]++;
                    }
                }
            }

            // Counter for the total number of connected servers
            int connectedServers = 0;

            // Second iteration to count the servers that can communicate
            // Servers can communicate if they are not the only one in their row or column
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numCols; ++j)
                {
                    if (grid[i][j] == 1 && (rowCount[i] > 1 || colCount[j] > 1))
                    {
                        connectedServers++;
                    }
                }
            }

            // Return the number of connected servers
            return connectedServers;
        }
    }


    /* 1024. Video Stitching
    https://leetcode.com/problems/video-stitching/description/
    https://algo.monster/liteproblems/1024
     */

    class VideoStitchingSol
    {
        /* Time and Space Complexity
Time Complexity
The provided Python code has a for loop that iterates through the clips list, which has a length that we can refer to as n. Inside this loop, we have constant time operations (if comparison and max function), meaning that this part of the algorithm has a time complexity of O(n).
Following that, there is another for loop which iterates through the last list. Since last has a length equal to the time parameter, this loop iterates time times. Once again, we perform constant time operations within the loop (max function, if comparisons, and simple assignments).
Therefore, the second loop has a time complexity of O(time). Since these two loops are not nested but executed in sequence, the overall time complexity of the code combines both complexities, resulting in O(n + time).
Space Complexity
For space complexity, the code allocates an array last of length equal to time, which requires O(time) space. The rest of the variables used in the code (ans, mx, and pre) require constant space, O(1).
Therefore, the overall space complexity of the function is O(time) because this is the largest space requirement that does not change with different inputs of clips.
 */
        public int VideoStitching(int[][] clips, int T)
        {
            int[] maxReach = new int[T];

            // Iterate over each clip and record the furthest end time for each start time
            foreach (int[] clip in clips)
            {
                int start = clip[0], end = clip[1];
                if (start < T)
                {
                    maxReach[start] = Math.Max(maxReach[start], end);
                }
            }

            int count = 0; // the minimum number of clips needed
            int maxEnd = 0; // the farthest end time we can reach so far
            int prevEnd = 0; // the end time of the last clip we have included in the solution

            // Loop through each time unit up to T
            for (int i = 0; i < T; i++)
            {
                maxEnd = Math.Max(maxEnd, maxReach[i]);

                // If the maxEnd we can reach is less or equal to current time 'i',
                // it is impossible to stitch the video up to 'i'
                if (maxEnd <= i)
                {
                    return -1;
                }

                // When we reach the end of the previous clip, increment the count of clips
                // and set the prevEnd to maxEnd to try and reach further in the next iteration
                if (prevEnd == i)
                {
                    count++;
                    prevEnd = maxEnd;
                }
            }

            // Return the minimum number of clips needed
            return count;
        }
    }

    /* 3067. Count Pairs of Connectable Servers in a Weighted Tree Network
    https://leetcode.com/problems/count-pairs-of-connectable-servers-in-a-weighted-tree-network/description/
    https://algo.monster/liteproblems/3067
     */
    public class CountPairsOfConnectableServersSol
    {

        /*Time and Space Complexity
The time complexity of the given code is O(n^2). This is because for each of the n nodes, there is a Depth-First Search (DFS) that is initiated. DFS itself may visit each node one time in the worst case, and since the DFS is occurring in a loop of n nodes, this results in a potential of n * (n-1) operations, hence the O(n^2).
Furthermore, within the DFS function, there is a check for whether ws % signalSpeed is equal to zero that occurs with each recursive call, but this does not add to the complexity in terms of n, as it is a constant-time operation.
The space complexity of the code is O(n). The primary space usage comes from the adjacency list g, which contains a list for each of the n nodes. Each list hold pairs ((b, w)) representing edges and their weights. There is also the ans array of size n that is kept throughout the execution. Temporary variables such as cnt and the stack frames due to recursive calls do not increase the overall space complexity, as they will at most be proportional to the depth of the recursion which is at most n, in the case of a path graph.
Additionally, the recursive nature of the DFS function does entail a call stack, which could have a depth of up to n in a worst-case scenario (such as a linear tree), but since the adjacency list is the dominant space-consuming structure and they are both of O(n) space complexity, the overall space complexity remains O(n).
*/
        private int signalSpeed;                             // The speed of the signal as defined.
        private List<int[]>[] adjacencyList;                // This list represents the graph as an adjacency list.

        // Method to count pairs of connectable servers given the edges and signal speed.
        public int[] CountPairsOfConnectableServers(int[][] edges, int signalSpeed)
        {
            int numServers = edges.Length + 1;                   // Number of servers is one more than the number of edges.
            this.signalSpeed = signalSpeed;                      // Set the global signal speed.
            adjacencyList = new List<int[]>[numServers];       // Initialize the adjacency list for each server.
            for (int i = 0; i < numServers; i++)
            {
                adjacencyList[i] = new List<int[]>();           // Initialize each list in the adjacency list.
            }
            // Convert the edge list to an adjacency list representation of the graph.
            foreach (int[] edge in edges)
            {
                int from = edge[0], to = edge[1], weight = edge[2];
                adjacencyList[from].Add(new int[] { to, weight });
                adjacencyList[to].Add(new int[] { from, weight });
            }
            int[] answer = new int[numServers];                  // Initialize an array to store the counts for each server.
                                                                 // Iterate over each server to find connectable pairs by using depth-first search.
            for (int server = 0; server < numServers; ++server)
            {
                int count = 0;
                // Explore all reachable servers from the current one and calculate counts.
                foreach (int[] edge in adjacencyList[server])
                {
                    int neighbor = edge[0], weight = edge[1];
                    int reachableServers = Dfs(neighbor, server, weight);
                    answer[server] += count * reachableServers;
                    count += reachableServers;
                }
            }
            return answer;                                    // Return the answer array containing counts for each server.
        }

        // Helper method for depth-first search to count reachable servers given a certain accumulated weight.
        private int Dfs(int current, int parent, int accumulatedWeight)
        {
            // If the accumulated weight is a multiple of the signal speed, it means this server is connectable.
            int connectableServers = accumulatedWeight % signalSpeed == 0 ? 1 : 0;
            // Explore all connected servers from the current server.
            foreach (int[] edge in adjacencyList[current])
            {
                int nextServer = edge[0], weight = edge[1];
                // Avoid visiting the server from which the current DFS initiated.
                if (nextServer != parent)
                {
                    connectableServers += Dfs(nextServer, current, accumulatedWeight + weight);
                }
            }
            return connectableServers;                          // Return the count of reachable connectable servers.
        }
    }

    /* 3259. Maximum Energy Boost From Two Drinks
    https://leetcode.com/problems/maximum-energy-boost-from-two-drinks/description/
     */

    public long MaxEnergyBoost(int[] energyDrinkA, int[] energyDrinkB)
    {
        long a0 = 0, a1 = 0, b0 = 0, b1 = 0, n = energyDrinkA.Length;
        for (int i = 0; i < n; i++)
        {
            a1 = Math.Max(a0 + energyDrinkA[i], b0);
            b1 = Math.Max(b0 + energyDrinkB[i], a0);
            a0 = a1; b0 = b1;
        }
        return Math.Max(a1, b1);
    }

    /* 1222. Queens That Can Attack the King
    https://leetcode.com/problems/queens-that-can-attack-the-king/description/
    https://algo.monster/liteproblems/1222
     */
    public class QueensAttackTheKingSol
    {
        /*  Time and Space Complexity
The code performs a search in each of the 8 directions from the king's position until it either hits the end of the board or finds a queen. Let's analyze the time and space complexity:
Time Complexity
The time complexity is O(n), where n is the number of cells in the board (in this case, n is 64, since it's an 8x8 chessboard). We iterate over each direction only once and in the worst case, we traverse the entire length of the board. However, since the board size is fixed, we can consider this time complexity to be O(1) in terms of the input size, because the board doesn't grow with the input.
Space Complexity
The space complexity is O(q), where q is the number of queens, because we store the positions of the queens in a set s. The board size is fixed and does not influence the space complexity beyond the storage of queens. In the worst case where every cell contains a queen, space complexity would be O(n), but since the board's size is constant and doesn't scale with the input, this can also be referred to as O(1) from a practical perspective.
*/

        // Function that returns a list of queens that can attack the king.
        public List<List<int>> QueensAttackTheKing(int[][] queens, int[] king)
        {
            // Define the size of the chessboard
            const int SIZE = 8;

            // Board to track the positions of the queens
            bool[,] board = new bool[SIZE, SIZE];

            // Place queens on the board based on given positions
            foreach (int[] queen in queens)
            {
                board[queen[0], queen[1]] = true;
            }

            // List to store positions of queens that can attack the king
            List<List<int>> attackPositions = new List<List<int>>();

            // Directions - The pair (a, b) represents all 8 possible directions from a cell
            //               (0, -1)  (-1, -1)  (-1, 0)
            //               (0, 1)             (1, 0)
            //              (1, 1)     (1, -1)   
            for (int rowDir = -1; rowDir <= 1; rowDir++)
            {
                for (int colDir = -1; colDir <= 1; colDir++)
                {
                    // Skip standstill direction as the king is not moving
                    if (rowDir != 0 || colDir != 0)
                    {
                        // Starting position for searching in a specific direction
                        int x = king[0] + rowDir, y = king[1] + colDir;

                        // Traverse in the direction until a queen is found or edge of board is reached
                        while (x >= 0 && x < SIZE && y >= 0 && y < SIZE)
                        {
                            // Check if queen is found
                            if (board[x, y])
                            {
                                attackPositions.Add(new List<int> { x, y });
                                break; // Break out of the loop as we're only looking for the closest queen
                            }
                            // Move to next position in the direction
                            x += rowDir;
                            y += colDir;
                        }
                    }
                }
            }

            return attackPositions;
        }
    }

    /* 831. Masking Personal Information
    https://leetcode.com/problems/masking-personal-information/description/
    https://algo.monster/liteproblems/831
     */
    public class MaskPIISol
    {
        /*Time and Space Complexity
Time Complexity:
The time complexity of the maskPII function is determined by several operations that the function performs:
1.	Checking whether the first character is alphabetical: O(1), since it's a constant-time check on a single character.
2.	Converting the string to lower case if it's an email: O(n), where n is the length of the string s.
3.	Finding the position of '@' in the string: O(n) in the worst case, since it requires scanning the string in case of an email address.
4.	Stripping non-digit characters and joining digits into a new string: O(n), because each character of the original string s is checked once.
5.	Constructing the masked string: The actual construction is O(1) because we are appending fixed strings and characters to a masked result.
Combining these, the time complexity is dictated by the string operations, which are linear in the length of the input string s. Therefore, the time complexity is O(n).
Space Complexity:
The space complexity is determined by the additional space used by the function:
1.	A new lower case email id string if s[0].isalpha(): O(n) space for the new string.
2.	The list comprehension that filters digits: O(n) space for the new list of characters before it is joined into a string.
3.	The final masked string has a constant size related to the format of the phone number or email (O(1) space), not dependent on the input size.
Thus, the dominating factor is the creation of new strings, which takes O(n) space. Hence, the space complexity is O(n).
*/
        public string MaskPII(string input)
        {
            // Check if the first character is a letter to determine if it's an email
            if (char.IsLetter(input[0]))
            {
                // Convert the entire input string to lower case for uniformity
                input = input.ToLower();
                // Find the index of the '@' symbol in the email
                int atIndex = input.IndexOf('@');
                // Create a masked email with the first character, five stars, and the domain part
                return input[0] + "*****" + input.Substring(atIndex - 1);
            }

            // StringBuilder to hold only the digits from the input (phone number)
            StringBuilder digitBuilder = new StringBuilder();
            // Loop through the characters in the input
            foreach (char c in input)
            {
                // Append only if the character is a digit
                if (char.IsDigit(c))
                {
                    digitBuilder.Append(c);
                }
            }

            // Convert the StringBuilder to a string containing only digits
            string digits = digitBuilder.ToString();
            // Calculate the number of digits that are in the international code
            int internationalCodeLength = digits.Length - 10;
            // Create a masked string for the last 7 digits of the phone number
            string maskedSuffix = "***-***-" + digits.Substring(digits.Length - 4);

            // Depending on whether there is an international code, mask the phone number appropriately
            if (internationalCodeLength == 0)
            {
                // If there's no international code, return just the masked U.S. number
                return maskedSuffix;
            }
            else
            {
                // If there's an international code, mask it and append the U.S. number
                string starsForInternational = "+";
                for (int i = 0; i < internationalCodeLength; i++)
                {
                    starsForInternational += "*";
                }
                return starsForInternational + "-" + maskedSuffix;
            }
        }
    }

    /* 1705. Maximum Number of Eaten Apples
    https://leetcode.com/problems/maximum-number-of-eaten-apples/description/
    https://algo.monster/liteproblems/1705
     */
    public class EatenApplesSol
    {
        /* 
        Time and Space Complexity
        The time complexity of the code is O(n + k log k), where n is the length of the days array, and k is the total number of different apples received over the days. Each day up to day n, the operation either pushes (heappush) or pops (heappop) an item from the heap q, which takes O(log k) time. After day n, all elements in the heap q will be popped without any further pushes, so the complexity in that phase depends on the number of elements in the heap, which can be at most k. Since each element in the heap can only be pushed and popped once, the number of heap operations is limited to the number of elements k. Therefore, the total time spent on heap operations is O(k log k). Additionally, there is a constant amount of work done for each day i up to day n, contributing to the O(n) factor.
        The space complexity of the code is O(k), as it requires storing each different apple's expiry date and count in the heap q. In the worst case, k could be as large as the number of days if we receive at least one apple on each day with different expiry dates. The heap size will never exceed the total number of apples because once an apple's expiry date has passed, it is removed from the heap.

         */
        public int EatenApples(int[] apples, int[] days)
        {
            // PriorityQueue to store apple batches with their expiry days as the priority.
            // It uses a comparator to ensure that the batch with the earliest expiry (smallest day) is at the top.
            PriorityQueue<int[], int> appleQueue = new PriorityQueue<int[], int>();

            int numberOfDays = apples.Length; // Number of days for which we have apple availability data.
            int totalEatenApples = 0; // Counter to keep track of total apples eaten.
            int currentDay = 0; // Current day, starting from day 0.

            // Loop through each day until we have processed all days or the queue is empty.
            while (currentDay < numberOfDays || appleQueue.Count > 0)
            {
                // If apples are available on the current day, add them to the queue.
                if (currentDay < numberOfDays && apples[currentDay] > 0)
                {
                    appleQueue.Enqueue(new int[] { currentDay + days[currentDay] - 1, apples[currentDay] }, currentDay + days[currentDay] - 1);
                }

                // Remove all batches from the queue that have expired by the current day.
                while (appleQueue.Count > 0 && appleQueue.Peek()[0] < currentDay)
                {
                    appleQueue.Dequeue();
                }

                // If there is at least one batch of apples that hasn't expired, eat one apple.
                if (appleQueue.Count > 0)
                {
                    int[] batch = appleQueue.Dequeue(); // Get the batch with the earliest expiry date.
                    totalEatenApples++; // Increment the count of eaten apples.
                    batch[1]--; // Decrement the count of apples in the batch since one is eaten.

                    // If there are still apples left in the batch and it hasn't expired, put it back in the queue.
                    if (batch[1] > 0 && batch[0] > currentDay)
                    {
                        appleQueue.Enqueue(batch, batch[0]);
                    }
                }
                // Move to the next day.
                currentDay++;
            }

            return totalEatenApples; // Return the total number of apples eaten.
        }
    }

    /* 755. Pour Water
    https://leetcode.com/problems/pour-water/description/
    https://algo.monster/liteproblems/755
     */
    public class PourWaterSol
    {
        // Method to simulate pouring water over a set of columns represented by heights
        public int[] PourWater(int[] heights, int volume, int position)
        {
            /* Time and Space Complexity
The time complexity of the given code is O(V * N), where V is the volume of water to pour, and N is the length of the heights list. This is because for each unit of volume, the code potentially traverses the heights list in both directions (-1 and 1) from the position k until it finds a suitable place to drop the water or returns to the starting index k.
The inner while loop runs for at most N iterations (in the worst case where it goes from one end of the heights list to the other), and since there is a fixed volume V, the outer loop runs V times. Each time we are performing a comparison operation which is an O(1) operation. Therefore, when combining these operations, the overall time complexity becomes O(V * N).
The space complexity of the code is O(1), assuming the input heights list is mutable and does not count towards space complexity (since we're just modifying it in place). This is because no additional significant space is allocated that grows with the size of the input; we only use a few extra variables (i, j, d, k) for indexing and comparison, which is constant extra space.
 */

            // Loop until all units of volume have been poured
            while (volume-- > 0)
            {
                bool hasPoured = false; // Indicator if water has been poured

                // Two directions: left (direction=-1), and right (direction=1)
                for (int direction = -1; direction <= 1 && !hasPoured; direction += 2)
                {
                    int currentIndex = position, lowestIndex = position;

                    // Move from the position to the direction indicated by direction
                    // Check if the current index is within bounds and if the next column is equal or lower
                    while (currentIndex + direction >= 0 && currentIndex + direction < heights.Length &&
                           heights[currentIndex + direction] <= heights[currentIndex])
                    {
                        // Moving to the next column if the condition is met
                        currentIndex += direction;

                        // If the next column is lower, update the lowest index
                        if (heights[currentIndex] < heights[lowestIndex])
                        {
                            lowestIndex = currentIndex;
                        }
                    }

                    // Pouring water into the lowest column if it is different from the starting position
                    if (lowestIndex != position)
                    {
                        hasPoured = true;  // Water has been poured
                        heights[lowestIndex]++;  // Increment the height of the lowest column
                    }
                }

                // If water has not been poured in either direction, pour it at the position
                if (!hasPoured)
                {
                    heights[position]++;
                }
            }
            return heights; // Return the modified ground after pouring all units of volume
        }
    }

    /* 1042. Flower Planting With No Adjacent
    https://leetcode.com/problems/flower-planting-with-no-adjacent/description/
    https://algo.monster/liteproblems/1042
     */
    class FlowerPlantingWithNoAdjacentSolution
    {
        /* Time and Space Complexity
        The code defines a Solution class with the gardenNoAdj method, which assigns a unique type of flower to each garden given the constraints that no adjacent gardens can have the same type of flower. There are n gardens numbered from 1 to n and at most 4 types of flowers.
        Time Complexity:
        The time complexity is determined by the following factors:
        1.	Building the graph: The loop iterates through the paths array, which might contain up to n(n-1)/2 paths in the worst case (a complete graph). This process has a time complexity of O(E) where E is the number of paths (edges).
        2.	Assigning flowers to gardens: There's an outer loop iterating n times for each garden and, inside it, a nested loop that iterates through the adjacent gardens (up to n-1 times in the worst-case scenario for a complete graph). However, since each garden is limited to 3 edges (paths) to prevent excessive adjacency according to the problem statement, the inner loop has a constant factor, and thus, this part has a time complexity of O(n).
        3.	Choosing a flower that hasn't been used: We iterate through the 4 types of flowers. This is a constant operation since the number of flower types does not change with n.
        Thus, the total time complexity is O(E + n), where E is the number of edges or paths.
        Space Complexity:
        The space complexity is determined by the following factors:
        1.	The graph g: In the worst-case scenario (a complete graph), each node connects to n-1 other nodes. Therefore, the space required by this graph is O(E) where E is the number of edges.
        2.	ans array: This is an array of size n, so it consumes O(n) space.
        3.	used set: At most, the set contains 4 elements because there are only 4 different types of flowers. This is a constant space O(1).
        Considering all factors, the total space complexity of the algorithm is O(E + n). The space taken by the used set is negligible in comparison to the graph and ans array.
        Combining both the time and space complexities, we sum them up as O(E + n) for time complexity and O(E + n) for space complexity.
         */
        public int[] GardenNoAdj(int numberOfGardens, int[][] paths)
        {
            // Create an adjacency list to represent gardens and their paths
            List<int>[] gardenGraph = new List<int>[numberOfGardens];
            // Initialize each list within the graph
            for (int index = 0; index < numberOfGardens; index++)
            {
                gardenGraph[index] = new List<int>();
            }
            // Fill the adjacency list with the paths provided
            foreach (int[] path in paths)
            {
                int gardenOne = path[0] - 1; // Subtract 1 to convert to 0-based index
                int gardenTwo = path[1] - 1; // Subtract 1 to convert to 0-based index
                gardenGraph[gardenOne].Add(gardenTwo); // Add a path from gardenOne to gardenTwo
                gardenGraph[gardenTwo].Add(gardenOne); // Add a path from gardenTwo to gardenOne (undirected graph)
            }

            // Answer array to store the type of flowers in each garden
            int[] flowerTypes = new int[numberOfGardens];
            // Array to keep track of used flower types
            bool[] usedFlowerTypes = new bool[5]; // Index 0 is unused, as flower types are 1-4

            // Assign flower types to each garden
            for (int currentGarden = 0; currentGarden < numberOfGardens; ++currentGarden)
            {
                // Reset the usedFlowerTypes array for the current garden
                Array.Fill(usedFlowerTypes, false);
                // Mark the flower types used by adjacent gardens
                foreach (int adjacentGarden in gardenGraph[currentGarden])
                {
                    usedFlowerTypes[flowerTypes[adjacentGarden]] = true;
                }
                // Find the lowest number flower type that hasn't been used by adjacent gardens
                for (int flowerType = 1; flowerType < 5; ++flowerType)
                {
                    if (!usedFlowerTypes[flowerType])
                    {
                        flowerTypes[currentGarden] = flowerType; // Assign this flower type to the current garden
                        break; // Exit loop after assigning a type
                    }
                }
            }
            // Return the array containing the flower types for each garden
            return flowerTypes;
        }
    }

    /* 2860. Happy Students
    https://leetcode.com/problems/happy-students/description/
    https://algo.monster/liteproblems/2860
     */
    class CountWaysSolution

    {

        /* Time Complexity
        The overall time complexity of the countWays function is determined by the sorting operation and the for loop.
        1.	The sort() method applied on nums list is the most costly operation in this snippet. The sort function in Python uses Timsort, which has a worst-case time complexity of O(n log n) where 'n' is the length of the list.
        2.	The for loop runs from 0 to n + 1 where 'n' is the length of the nums list. However, the loops body has continue statements which may terminate the iterations early without performing any additional operations. In the worst case, the loop runs n + 1 times.
        Considering the above points, the dominant part in terms of time complexity is the sorting operation. Therefore, the overall time complexity of the function is O(n log n) due to the sort, irrespective of the for loop, which has a best case of O(1) and a worst case of O(n).
        Space Complexity
        The space complexity of the function is determined by the storage requirements that are not directly dependent on the input size. In the snippet provided:
        1.	The nums list is sorted in place, so no additional space is necessary for sorting beyond the space already used to store nums.
        2.	The variable ans and the loop variable i each take constant space.
        As there are no additional data structures used that grow with the size of the input, the space complexity of the function is O(1), which is constant space.
         */
        /**
         * Counts the ways in which numbers can be assigned to their indices 
         * in the list such that each number is greater than its index.
         *
         * @param nums List of Integer values presumably between 0 and list size
         * @return Count of the possible ways numbers can be arranged fulfilling the condition
         */
        public int countWays(List<int> nums)
        {
            // Sort the list in non-decreasing order
            nums.Sort();

            // Get the size of the list
            int n = nums.Count;

            // Initialize answer count to 0
            int answer = 0;

            // Iterate through all possible positions in the list
            for (int i = 0; i <= n; i++)
            {
                // Check if the current number is greater than it's index (after considering zero-based index adjustment)
                // Also, consider the cases when the index is at the start or the end of the list
                if ((i == 0 || nums[i - 1] < i) && (i == n || nums[i] > i))
                {
                    // Increment the answer if the condition is satisfied
                    answer++;
                }
            }

            // Return the total count of valid ways
            return answer;
        }
    }

    /* 
    2358. Maximum Number of Groups Entering a Competition
    https://leetcode.com/problems/maximum-number-of-groups-entering-a-competition/description/
    https://algo.monster/liteproblems/2358
     */
    class MaximumGroupsEnteringCompetitionSol
    {
        /* Time and Space Complexity
The given Python code is aimed at finding the maximum number of groups with distinct lengths that the list of grades can be divided into. The key part of this code is the use of bisect_right, a binary search algorithm from the Python bisect module, to efficiently find the point where a quadratic equation's result surpasses a certain value, n * 2.
Time Complexity
The binary search performed using bisect_right operates on a range of numbers from 0 to n+1. The time complexity of a binary search is O(log k), where k is the size of the range we're searching within. In our case, because the search is within a numerical range up to n+1, the time complexity can be expressed as O(log n).
The lambda function is applied on each iteration of the binary search to compute the sum of squares and a linear term of the current middle value x. This operation is O(1) as it involves simple arithmetic operations. Since this lambda is called for each step of the binary search, it does not change the overall time complexity, maintaining it as O(log n).
Space Complexity
bisect_right operates within the given range and does not require additional space proportional to the input size. The variables used to store the results of the lambda operation and the lengths of grades (n) are constant with respect to the input size, leading to O(1) space complexity.
Therefore, the space complexity of this code is O(1), signifying constant space usage regardless of the input size.
 */
        public int MaximumGroups(int[] grades)
        {
            // The length of the grades array
            int length = grades.Length;

            // Variables to define the search range, initialized to the entire range of possible group numbers
            int left = 0, right = length;

            // Binary search to find the maximum number of groups
            while (left < right)
            {
                // Calculate the middle point. We calculate it this way to avoid integer overflow.
                int mid = (left + right + 1) >>> 1;

                // Check if the total number of students fits the condition for 'mid' groups
                // The condition is derived from the requirements of forming groups with an increasing number of students.
                // mid * (mid + 1) / 2 is the sum of the first 'mid' integers, which is the minimum number of students needed for 'mid' groups
                // We use long to avoid integer overflow when evaluating the condition.
                if (1L * mid * (mid + 1) > 2L * length)
                {
                    // If total students are insufficient, we decrease the 'right' boundary
                    right = mid - 1;
                }
                else
                {
                    // If total students are sufficient, we increase the 'left' boundary
                    left = mid;
                }
            }

            // When the while loop exits, 'left' will be the maximum number of groups that can be created
            return left;
        }
    }

    /* 2933. High-Access Employees
    https://leetcode.com/problems/high-access-employees/description/
     */

    public class FindHighAccessEmployeesSol
    {
        /* Complexity
•	Time complexity:
O(klogk)
•	Space complexity:
O(n)
 */
        // Function to find high-access employees based on access times.
        public List<string> FindHighAccessEmployees(List<List<string>> accessTimes)
        {
            // Create a dictionary to store access times for each employee.
            Dictionary<string, List<int>> accessTimeMap = new Dictionary<string, List<int>>();

            // Populate the dictionary with access times from the input list.
            foreach (List<string> entry in accessTimes)
            {
                string employee = entry[0];
                int accessTime = int.Parse(entry[1]);
                if (!accessTimeMap.ContainsKey(employee))
                {
                    accessTimeMap[employee] = new List<int>();
                }
                accessTimeMap[employee].Add(accessTime);
            }

            // List to store the names of high-access employees.
            List<string> highAccessEmployees = new List<string>();

            // Iterate through the dictionary to check access patterns for each employee.
            foreach (KeyValuePair<string, List<int>> kvp in accessTimeMap)
            {
                // Sort the access times for each employee.
                kvp.Value.Sort();

                // Get the number of access times for the current employee.
                int accessCount = kvp.Value.Count;

                // Flag to indicate if the employee is a high-access employee.
                bool isHighAccess = false;

                // Check for consecutive accesses within a 100-minute window.
                for (int i = 0; i + 3 <= accessCount; ++i)
                {
                    isHighAccess |= kvp.Value[i + 2] < kvp.Value[i] + 100;
                }

                // If the flag is true, the employee is considered high-access, and their name is added to the result.
                if (isHighAccess)
                {
                    highAccessEmployees.Add(kvp.Key);
                }
            }

            // Return the list containing the names of high-access employees.
            return highAccessEmployees;
        }
    }

    /* 1488. Avoid Flood in The City
    https://leetcode.com/problems/avoid-flood-in-the-city/description/
    https://algo.monster/liteproblems/1488
     */
    class AvoidFloodSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the solution is O(n * log n). This complexity arises due to the following reasons:
1.	We iterate over all n elements in the rains list.
2.	For each rainy day (non-zero element within rains), we use rainy[v] = i to store the last day a lake was filled which is O(1) for each operation.
3.	For each sunny day (zero element within rains), we use sunny.add(i) which is an O(log n) operation since SortedList() maintains a sorted order.
4.	When we need to find a sunny day to dry a lake, we perform sunny.bisect_right(rainy[v]) which is an O(log n) operation since it is a binary search within the sorted list of sunny days.
5.	We then remove the used sunny day slot with sunny.discard(sunny[idx]). This operation is O(log n) as well, since removal from a SortedList requires a search for the element's index followed by the removal, both contributing to the log n complexity.
Since these operations take place within a loop that runs n times, the overall time complexity combines to O(n * log n).
Space Complexity
The space complexity of the solution is O(n). This is because:
1.	We maintain an ans list of size n.
2.	We have a SortedList named sunny which could potentially, in the worst case, hold a separate entry for every day which would also be O(n).
3.	The rainy dictionary in the worst case will store an entry for every lake that gets filled which is bound by the number of days, so it is also O(n) space complexity.
Thus, the maximum of these space complexities dictates the overall space complexity, which is O(n).
 */
        public int[] AvoidFlood(int[] rains)
        {
            int numberOfDays = rains.Length;
            // The result array is initialized with -1's (since 1...n are the lake indices)
            int[] result = new int[numberOfDays];
            Array.Fill(result, -1);

            // A sorted set to keep track of the days without rain (sunny days)
            SortedSet<int> sunnyDays = new SortedSet<int>();
            // A dictionary to keep the latest day when it rained on each lake
            Dictionary<int, int> lastRainDay = new Dictionary<int, int>();

            for (int dayIndex = 0; dayIndex < numberOfDays; ++dayIndex)
            {
                int lake = rains[dayIndex];
                if (lake > 0)
                {
                    // If a lake has been filled before, we need to dry it on a sunny day
                    if (lastRainDay.ContainsKey(lake))
                    {
                        // Find the next available sunny day to dry the lake
                        int? dryingDay = sunnyDays.Where(d => d > lastRainDay[lake]).FirstOrDefault();
                        if (dryingDay == null)
                        {
                            // If there's no sunny day after the last rain day, flooding is inevitable
                            return new int[0];
                        }
                        // Use the found sunny day to dry the lake
                        result[dryingDay.Value] = lake;
                        // And remove that day from the set of available sunny days
                        sunnyDays.Remove(dryingDay.Value);
                    }
                    // Update the dictionary with the last day it rained on the current lake
                    lastRainDay[lake] = dayIndex;
                }
                else
                {
                    // If there's no rain, we add this day to the set of sunny days
                    sunnyDays.Add(dayIndex);
                    // Default drying action is 1, but this would be overridden if it's needed for a lake
                    result[dayIndex] = 1;
                }
            }
            return result;
        }
    }

    /* 2064. Minimized Maximum of Products Distributed to Any Store
    https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/description/
    https://algo.monster/liteproblems/2064
     */
    class MinimizedMaximumOfProductsDistributedToAnyStoreSol
    {
        /* Time and Space Complexity
        The given Python code aims to find a value of x that, when used to distribute the quantities of items amongst n stores, results in a minimized maximum number within a store while ensuring all quantities are distributed. This is achieved by using a binary search via bisect_left on a range of possible values for x.
        Time Complexity:
        The binary search is performed on a range from 1 to a constant value (10**6), resulting in O(log(C)) complexity, where C is the upper limit of the search range. Inside the check function, there is a loop which computes the sum with complexity O(Q) for each check, where Q is the length of the quantities list. Therefore, the time complexity of the entire algorithm is O(Q * log(C)).
        Space Complexity:
        The code uses a constant amount of additional memory outside of the quantities list input. The check function computes the sum using the values in quantities without additional data storage that depends on the size of quantities or the range of values. Hence, the space complexity is O(1), as no significant additional space is consumed in relation to the input size.
         */
        public int MinimizedMaximum(int stores, int[] products)
        {
            // Initial search space: the lowest possible maximum is 1 (each store can have at least one of any product),
            // and the highest possible maximum is assumed to be 100000
            // (based on the problem constraints if given in the problem description).
            int left = 1, right = 100000;

            // Using binary search to find the minimized maximum value
            while (left < right)
            {
                // Midpoint of the current search space
                int mid = (left + right) / 2;

                // Counter for the number of stores needed
                int count = 0;

                // Distribute products among stores
                foreach (int quantity in products)
                {
                    // Each store can take 'mid' amount, calculate how many stores are required
                    // for this particular product, rounding up
                    count += (quantity + mid - 1) / mid;
                }

                // If we can distribute all products to 'stores' or less with 'mid' maximum product per store,
                // we are possibly too high in the product capacity (or just right) so we try a lower capacity
                if (count <= stores)
                {
                    right = mid;
                }
                else
                {
                    // If we are too low and need more than 'stores' to distribute all products,
                    // we need to increase the product capacity per store
                    left = mid + 1;
                }
            }

            // 'left' will be our minimized maximum product per store that fits all products into 'stores' stores.
            return left;
        }
    }

    /* 1169. Invalid Transactions
    https://leetcode.com/problems/invalid-transactions/description/
    https://algo.monster/liteproblems/1169
     */
    public class InvalidTransactionsSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the code can be analyzed as follows:
•	The loop to iterate over all the transactions takes O(n) time, where n is the number of transactions.
•	Inside this loop, splitting each transaction into name, time, amount, and city takes O(m) time, where m is the average length of a transaction string.
•	Appending a tuple (time, city, i) to the list in the dictionary for the respective name is an O(1) operation as appending to a list in Python has an amortized constant time complexity. However, since we do this for each transaction, it contributes to O(n) over the entire loop.
•	The nested loop iterating through each entry in d[name] has a variable runtime that depends on the number of transactions associated with that name. If we assume that each person has at most k transactions, this nested loop will take O(k) time per transaction in the worst case, leading to an overall time complexity of O(nk) for the nested loops across all transactions.
•	The if condition checking for different cities and the time difference <= 60 is O(1).
So, the overall time complexity is O(n * (m + k)).
Space Complexity
The space complexity of the code is given by:
•	The dictionary d, which stores a list of tuples for each unique name. In the worst case, it has an entry for each transaction if all names are unique, which contributes O(n) space complexity.
•	The set idx storing indices of invalid transactions, which in the worst case could include all transactions if they are all invalid, also contributes O(n) space complexity.
Overall, the space complexity is O(n), taking into account the space required for the input and auxiliary data structures.
 */
        public IList<string> InvalidTransactions(string[] transactions)
        {
            // A map to maintain transaction items per user
            var transactionMap = new Dictionary<string, List<TransactionItem>>();
            // A set to keep track of invalid transaction indices
            var invalidIndices = new HashSet<int>();

            // Iterate through all transactions
            for (int i = 0; i < transactions.Length; ++i)
            {
                // Split the transaction string into individual pieces of data
                var transactionDetails = transactions[i].Split(',');
                string name = transactionDetails[0];
                int time = int.Parse(transactionDetails[1]);
                int amount = int.Parse(transactionDetails[2]);
                string city = transactionDetails[3];

                // Add the transaction item under the user's name in the map
                if (!transactionMap.ContainsKey(name))
                {
                    transactionMap[name] = new List<TransactionItem>();
                }
                transactionMap[name].Add(new TransactionItem(time, city, i));

                // If the amount exceeds $1000, mark as invalid
                if (amount > 1000)
                {
                    invalidIndices.Add(i);
                }

                // Check the transaction against other transaction items of the same user
                foreach (var item in transactionMap[name])
                {
                    // If a transaction item with a different city within 60 minutes is found, mark as invalid
                    if (!city.Equals(item.City) && Math.Abs(time - item.Time) <= 60)
                    {
                        invalidIndices.Add(i);
                        invalidIndices.Add(item.Index);
                    }
                }
            }

            // Prepare the list of invalid transaction strings to return
            var answer = new List<string>();
            foreach (int index in invalidIndices)
            {
                answer.Add(transactions[index]);
            }
            return answer;
        }
        // A helper class to represent transaction items
        public class TransactionItem
        {
            public int Time { get; }
            public string City { get; }
            public int Index { get; }

            public TransactionItem(int time, string city, int index)
            {
                Time = time;
                City = city;
                Index = index;
            }
        }
    }


    /* 1226. The Dining Philosophers
    https://leetcode.com/problems/the-dining-philosophers/description/
    https://algo.monster/liteproblems/1226
     */
    public class DiningPhilosophersSol
    {

        /*1. Using Mutexes 
        Time and Space Complexity
Time Complexity
The time complexity of the wantsToEat method in the DiningPhilosophers class isn't determined by a simple algorithmic analysis, because it's primarily dependent on the concurrency and synchronization primitives used (mutexes and locks). Each philosopher (in this case, a thread) attempts to pick up two forks (acquiring two mutexes) before eating. The std::scoped_lock is used to acquire both mutexes atomically, which prevents deadlock. The actual time complexity for each philosopher to eat depends on the order and time at which each thread is scheduled as well as contention for the mutexes.
However, assuming there is no contention and each operation (pickLeftFork, pickRightFork, eat, putLeftFork, and putRightFork) has a constant time complexity O(1), the wantsToEat function would have a time complexity of O(1) for each call in an ideal scenario. Acquiring and releasing a mutex can also be considered to have a time complexity of O(1).
Space Complexity
The space complexity of the DiningPhilosophers class is O(N) where N is the number of philosophers (which is 5 in this case). This is due to the vector<mutex> mutexes_ which contains a mutex for each philosopher's left fork. There are no additional data structures that scale with the number of operations or philosophers, so the space complexity is proportional to the number of philosophers.
In this code, since N is fixed at 5, you could argue that the space complexity can be considered as O(1) since it doesn't scale with input size and is fixed.

        */
        public class DiningPhilosophersUsingMutex
        {
            // Alias for the action functions using Action, since C# has a built-in Action delegate.
            public delegate void Action();

            // Array of Mutex objects representing the forks.
            private readonly Mutex[] forks = new Mutex[5];

            // Constructor initializes each Mutex representing a fork.
            public DiningPhilosophersUsingMutex()
            {
                for (int i = 0; i < forks.Length; i++)
                {
                    forks[i] = new Mutex();
                }
            }

            public void WantsToEat(int philosopher,
                                  Action pickLeftFork,
                                  Action pickRightFork,
                                  Action eat,
                                  Action putLeftFork,
                                  Action putRightFork)
            {
                // The id of the left and right fork, taking into account the special case of the last philosopher
                int leftFork = philosopher;
                int rightFork = (philosopher + 1) % 5;

                // Lock the forks to ensure that no two philosophers can hold the same fork at the same time.
                // Locking is arranged to prevent deadlock.
                forks[leftFork].WaitOne();
                forks[rightFork].WaitOne();

                try
                {
                    // Perform actions with the forks and eating in a critical section.
                    pickLeftFork(); // Pick up left fork
                    pickRightFork(); // Pick up right fork
                    eat(); // Eat
                    putLeftFork(); // Put down left fork
                    putRightFork(); // Put down right fork
                }
                finally
                {
                    // Ensure that forks are always released to avoid deadlock.
                    forks[leftFork].ReleaseMutex();
                    forks[rightFork].ReleaseMutex();
                }
            }
        }

        //https://github.com/seanconnollydev/dining-philosophers

        /* 2. Using Monitor Object Pattern */
        public class DiningPhilosopherUsingMonitors
        {
            private const int TIMES_TO_EAT = 5;
            private int _timesEaten = 0;
            private readonly List<DiningPhilosopherUsingMonitors> _allPhilosophers;
            private readonly int _index;

            public DiningPhilosopherUsingMonitors(List<DiningPhilosopherUsingMonitors> allPhilosophers, int index)
            {
                _allPhilosophers = allPhilosophers;
                _index = index;
                this.Name = string.Format("Philosopher {0}", _index);
                this.State = State.Thinking;
            }

            public string Name { get; private set; }
            public State State { get; private set; }
            public Chopstick LeftChopstick { get; set; }
            public Chopstick RightChopstick { get; set; }

            public DiningPhilosopherUsingMonitors LeftPhilosopher
            {
                get
                {
                    if (_index == 0)
                        return _allPhilosophers[_allPhilosophers.Count - 1];
                    else
                        return _allPhilosophers[_index - 1];
                }
            }

            public DiningPhilosopherUsingMonitors RightPhilosopher
            {
                get
                {
                    if (_index == _allPhilosophers.Count - 1)
                        return _allPhilosophers[0];
                    else
                        return _allPhilosophers[_index + 1];
                }
            }

            public void EatAll()
            {
                // Cycle through thinking and eating until done eating.
                while (_timesEaten < TIMES_TO_EAT)
                {
                    this.Think();
                    if (this.PickUp())
                    {
                        // Chopsticks acquired, eat up
                        this.Eat();

                        // Release chopsticks
                        this.PutDownLeft();
                        this.PutDownRight();
                    }
                }
            }

            private bool PickUp()
            {
                // Try to pick up the left chopstick
                if (Monitor.TryEnter(this.LeftChopstick))
                {
                    Console.WriteLine(this.Name + " picks up left chopstick.");

                    // Now try to pick up the right
                    if (Monitor.TryEnter(this.RightChopstick))
                    {
                        Console.WriteLine(this.Name + " picks up right chopstick.");

                        // Both chopsticks acquired, its now time to eat
                        return true;
                    }
                    else
                    {
                        // Could not get the right chopstick, so put down the left
                        this.PutDownLeft();
                    }
                }

                // Could not acquire chopsticks, try again
                return false;
            }

            private void Eat()
            {
                this.State = State.Eating;
                _timesEaten++;
                Console.WriteLine(this.Name + " eats.");
            }

            private void PutDownLeft()
            {
                Monitor.Exit(this.LeftChopstick);
                Console.WriteLine(this.Name + " puts down left chopstick.");
            }

            private void PutDownRight()
            {
                Monitor.Exit(this.RightChopstick);
                Console.WriteLine(this.Name + " puts down right chopstick.");
            }


            private void Think()
            {
                this.State = State.Thinking;
            }
        }

        public enum State
        {
            Thinking = 0,
            Eating = 1
        }

        [DebuggerDisplay("Name = {Name}")]
        public class Chopstick
        {
            private static int _count = 1;
            public string Name { get; private set; }

            public Chopstick()
            {
                this.Name = "Chopstick " + _count++;
            }
        }
    }

    /* 2100. Find Good Days to Rob the Bank
    https://leetcode.com/problems/find-good-days-to-rob-the-bank/description/
    https://algo.monster/liteproblems/2100
     */
    class GoodDaysToRobBankSol
    {
        /*  Time and Space Complexity
Time Complexity
The time complexity of the provided code can be broken down into the following parts:
1.	Initializing the left and right arrays, each with a length of n: This operation takes O(n) time.
2.	Filling in the left array with the lengths of non-increasing subsequences from the left: This requires iterating over the entire security list of length n, resulting in O(n) time.
3.	Filling in the right array with the lengths of non-decreasing subsequences from the right: Similar to the left array, we iterate once in reverse over the length of the security list, which takes O(n) time.
4.	Constructing the list of good days to rob the bank: We again iterate over the security list and compare the left and right values, which is an O(n) operation.
Adding all these operations together, the time complexity is O(n) + O(n) + O(n) + O(n), which simplifies to O(n).
Space Complexity
The space complexity of the code can also be analyzed based on the data structures being used:
1.	The left and right arrays, both of which are of size n: These contribute 2n space complexity.
2.	The result list that is output has at most n elements. However, this space is accounted for in the result and not typically counted towards auxiliary space being used by the algorithm.
Considering these two factors, the space complexity is O(n) for the auxiliary space used, not including the output data structure.
*/
        public List<int> GoodDaysToRobBank(int[] security, int time)
        {
            // Get the length of the `security` array.
            int length = security.Length;
            // If the length is not sufficient to have days before and after the time period, return an empty list.
            if (length <= time * 2)
            {
                return new List<int>();
            }

            // Arrays to keep track of the non-increasing trend to the left and non-decreasing trend to the right.
            int[] nonIncreasingLeft = new int[length];
            int[] nonDecreasingRight = new int[length];

            // Populate the nonIncreasingLeft array by checking if each day is non-increasing compared to the previous day.
            for (int index = 1; index < length; ++index)
            {
                if (security[index] <= security[index - 1])
                {
                    nonIncreasingLeft[index] = nonIncreasingLeft[index - 1] + 1;
                }
            }

            // Populate the nonDecreasingRight array by checking if each day is non-decreasing compared to the next day.
            for (int index = length - 2; index >= 0; --index)
            {
                if (security[index] <= security[index + 1])
                {
                    nonDecreasingRight[index] = nonDecreasingRight[index + 1] + 1;
                }
            }

            // To store the good days to rob the bank.
            List<int> goodDays = new List<int>();

            // Check each day to see if it can be a good day to rob the bank.
            for (int index = time; index < length - time; ++index)
            {
                // A day is good if there are at least `time` days before and after it forming non-increasing and non-decreasing trends.
                if (time <= Math.Min(nonIncreasingLeft[index], nonDecreasingRight[index]))
                {
                    goodDays.Add(index);
                }
            }
            // Return the list of good days.
            return goodDays;
        }
    }


    /* 2662. Minimum Cost of a Path With Special Roads
    https://leetcode.com/problems/minimum-cost-of-a-path-with-special-roads/description/
    https://algo.monster/liteproblems/2662
     */
    public class MinimumCostOfPathWithSpecialRoadsSol
    {
        /* Time and Space Complexity
The time complexity of the given code is O(n^2 \times \log n), where n is the number of special roads. This complexity arises because, in the worst case, each road is visited once and every time a road is visited, it could potentially lead to all other roads being considered as next steps. For each road, the distance to the next road is computed, which is an O(1) operation, but then a point is added to the priority queue that has a size up to O(n^2) (since every special road could be pushed to the queue with different starting points), and hence, the insertion time will be O(\log n^2) which simplifies to O(2\log n) = O(\log n). Since we could have O(n^2) insertions, the total time complexity is O(n^2 \times \log n).
The space complexity of the code is O(n^2) mainly due to the storage requirements of the priority queue and the visited set. In the worst case, each point in the grid could be added to the visited set and at most, every entry of the special roads could be stored in the priority queue simultaneously, if the points are all unique.
 */
        // Method to calculate the minimum cost to travel from the start point to the target point considering special roads
        public int MinimumCost(int[] start, int[] target, int[][] specialRoads)
        {
            // A large constant for the initial minimum cost
            int minCost = int.MaxValue;
            // Size of the grid
            int gridSize = 1000000;
            // Priority Queue to store states with minimum distance at the top
            PriorityQueue<int[], int[]> priorityQueue = new PriorityQueue<int[], int[]>(
                Comparer<int[]>.Create((a, b) => a[0] - b[0]));
            // Set for visited points to avoid processing a point multiple times
            HashSet<long> visited = new HashSet<long>();
            // Add the starting point to the priority queue with initial cost 0
            priorityQueue.Enqueue(new int[] { 0, start[0], start[1] }, new int[] { 0, start[0], start[1] });
            // Process nodes until the priority queue is empty
            while (priorityQueue.Count > 0)
            {
                int[] point = priorityQueue.Dequeue();
                int currentX = point[1], currentY = point[2];
                // Unique number for the current point (hash)
                long hash = 1L * currentX * gridSize + currentY;
                // Skip if we've already visited this point
                if (visited.Contains(hash))
                {
                    continue;
                }
                visited.Add(hash);
                // Current distance from start to the point
                int distance = point[0];
                // Update minimum cost with the sum of the current distance and direct distance from the current point to target
                minCost = Math.Min(minCost, distance + CalculateManhattanDistance(currentX, currentY, target[0], target[1]));
                // Explore special roads from the current point
                foreach (int[] road in specialRoads)
                {
                    int roadStartX = road[0], roadStartY = road[1], roadEndX = road[2], roadEndY = road[3], roadCost = road[4];
                    // Offer a new state to the queue with the updated cost considering the special road
                    var newState = new int[] {
                    distance + CalculateManhattanDistance(currentX, currentY, roadStartX, roadStartY) + roadCost,
                    roadEndX,
                    roadEndY
                };
                    priorityQueue.Enqueue(newState, newState);
                }
            }
            // Return the minimum cost found
            return minCost;
        }

        // Helper method to calculate Manhattan distance between two points
        private int CalculateManhattanDistance(int x1, int y1, int x2, int y2)
        {
            return Math.Abs(x1 - x2) + Math.Abs(y1 - y2);
        }
    }


    /* 2910. Minimum Number of Groups to Create a Valid Assignment
    https://leetcode.com/problems/minimum-number-of-groups-to-create-a-valid-assignment/description/
    https://algo.monster/liteproblems/2910
     */
    public class MinGroupsForValidAssignmentSol
    {
        /*Time and Space Complexity
    Time Complexity
    The time complexity is actually not O(n) in general, it depends on both n, the length of nums, and also the range of unique values as well as their frequencies. The time complexity can be analyzed as follows:
    •	Counter(nums) has a complexity of O(n) since it goes through each element of nums.
    •	The outer loop runs at most min(cnt.values()) times which depends on the minimum frequency of a number in nums.
    •	The inner loop runs O(u) times where u is the number of unique elements in nums because it iterates through all values in the counter.
    So, the complexity is O(n + min(min(cnt.values()), n/u) * u) which is O(n + min_count * u) if we let min_count be min(cnt.values()).
    Giving a final verdict on time complexity without constraints of input can lead to a misleading statement since it can vary. If min_count is small, it could be close to linear but could also go up to O(n^2) in the worst scenario when all elements are unique.
    Space Complexity
    The space complexity is O(n) for the counter dictionary that stores up to n unique values from nums where n is the length of nums. No other significant space is used.
    */

        /**
         * Calculates the minimum number of groups for a valid assignment based on the input array.
         * The method counts the frequency of each number in the array and determines the smallest
         * number of groups such that the frequency of the numbers is proportionally distributed.
         * @param nums The input array containing numbers.
         * @return The minimum number of groups required.
         */
        public int MinGroupsForValidAssignment(int[] nums)
        {
            // Create a dictionary to store the frequency count of each unique number in nums
            Dictionary<int, int> frequencyCount = new Dictionary<int, int>();
            foreach (int number in nums)
            {
                // Increment the frequency count for each number
                if (frequencyCount.ContainsKey(number))
                {
                    frequencyCount[number]++;
                }
                else
                {
                    frequencyCount[number] = 1;
                }
            }

            // Initialize k as the number of elements in nums, the maximum possible frequency
            int k = nums.Length;

            // Find the smallest value among the frequencies to identify the initial group size
            foreach (int frequency in frequencyCount.Values)
            {
                k = Math.Min(k, frequency);
            }

            // Continuously try smaller values of k to optimize the number of groups
            while (k > 0)
            {
                int groupsNeeded = 0;
                foreach (int frequency in frequencyCount.Values)
                {
                    // If the frequency divided by k leaves a remainder larger than the quotient,
                    // the current value of k isn't a valid group size, break and try a smaller k
                    if (frequency % k > frequency / k)
                    {
                        groupsNeeded = 0;
                        break;
                    }
                    // Calculate the number of groups needed for the current value of k
                    groupsNeeded += (frequency + k - 1) / k;
                }
                // If the number of needed groups is greater than zero, we've found a valid grouping
                if (groupsNeeded > 0)
                {
                    return groupsNeeded;
                }
                // Decrement k and try again for a smaller group size
                k--;
            }

            // The code should never reach this point
            return -1; // This line is just for the sake of completeness; logically, it'll always return from the loop
        }
    }

    /* 1418. Display Table of Food Orders in a Restaurant
    https://leetcode.com/problems/display-table-of-food-orders-in-a-restaurant/description/
    https://algo.monster/liteproblems/1418
     */
    public class DisplayTableOfFoodOrdersInRestaurantSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the code can be broken down into several parts:
1.	Iterating through the list of orders: This takes O(N) time, where N is the total number of orders.
2.	Adding items to and creating the foods and tables sets: Insertions take O(1) on average, so for N orders, the complexity is O(N).
3.	The Counter updates (mp[f'{table}.{food}'] += 1) also occur N times, and they take O(1) time each, thus O(N) in total.
4.	Sorting the foods list takes O(F log F) time, where F is the number of unique foods.
5.	Sorting the tables list takes O(T log T) time, where T is the number of unique tables.
6.	Building the res list involves a double loop which iterates T times outside and F times inside, leading to O(T * F).
Adding these up, the total time complexity is O(N) + O(N) + O(N) + O(F log F) + O(T log T) + O(T * F), which simplifies to O(N + F log F + T log T + T * F).
Space Complexity
The space complexity can also be dissected into:
1.	The tables and foods sets, which take O(T + F) space.
2.	The mp counter, which will store at most N key-value pairs, hence O(N) space.
3.	The res list, which contains a T+1 by F matrix, thus taking O(T * F) space.
Combining these, the overall space complexity is O(T + F + N + T * F). Since N can be at most T * F if every table orders every type of food once, the space complexity simplifies to O(T * F).
 */
        // This method will process a list of orders and display them as a table with food item counts.
        public IList<IList<string>> DisplayTable(IList<IList<string>> orders)
        {
            // Use SortedSet for automatic sorting
            SortedSet<int> tableNumbers = new SortedSet<int>();
            SortedSet<string> menuItems = new SortedSet<string>();

            // This dictionary holds the concatenation of table number and food item as a key, and their count as a value.
            Dictionary<string, int> itemCountMap = new Dictionary<string, int>();

            // Processing each order to populate sets and the itemCountMap
            foreach (IList<string> order in orders)
            {
                int table = int.Parse(order[1]);
                string foodItem = order[2];

                // Add the table number and food item to the respective sets
                tableNumbers.Add(table);
                menuItems.Add(foodItem);

                // Create a unique key for each table-food pair
                string key = table + "." + foodItem;
                if (itemCountMap.ContainsKey(key))
                {
                    itemCountMap[key]++;
                }
                else
                {
                    itemCountMap[key] = 1;
                }
            }

            // Prepare the result list, starting with the title row
            IList<IList<string>> result = new List<IList<string>>();
            IList<string> headers = new List<string>();

            // Adding "Table" as the first column header
            headers.Add("Table");
            // Adding the rest of the food items as headers
            ((List<string>)headers).AddRange(menuItems);
            result.Add(headers);

            // Going through each table number and creating a row for the display table
            foreach (int tableNumber in tableNumbers)
            {
                IList<string> row = new List<string>();
                // First column of the row is the table number
                row.Add(tableNumber.ToString());
                // The rest of the columns are the counts of each food item at this table
                foreach (string menuItem in menuItems)
                {
                    // Forming the key to get the count from the map
                    string key = tableNumber + "." + menuItem;
                    // Adding the count to the row; if not present, add "0"
                    row.Add(itemCountMap.ContainsKey(key) ? itemCountMap[key].ToString() : "0");
                }
                // Add the row to the result list
                result.Add(row);
            }

            // Return the fully formed display table
            return result;
        }
    }
    /* 
    1958. Check if Move is Legal
    https://leetcode.com/problems/check-if-move-is-legal/description/
    https://algo.monster/liteproblems/1958
     */
    public class CheckMoveSol
    {
        /* Time and Space Complexity
        Time Complexity
        The time complexity of the code is determined by how many times we loop over the different directions from the starting move, as well as how far we can go in each direction. We have 8 possible directions to check, and in the worst-case scenario, we could iterate over all n cells in one direction (where n is the size of the board's dimension, which is 8 in this case). Therefore, the worst-case time complexity is O(8n) which simplifies to O(n) because 8 is a constant factor.
        In this specific case, since n is fixed at 8, we can also argue that the time complexity is O(1) since the board size doesn't change and there's a maximum number of steps to be taken.
        Space Complexity
        The space complexity of the code is O(1) since the extra space used does not scale with the size of the input. We have a fixed-size board and the dirs array which consists of 8 directions, and a few variables i, j, and t, which all occupy constant space regardless of input size.
         */
        private static readonly int[][] DIRECTIONS = { // Directions to check for flips
        new int[] {1, 0},   // South
        new int[] {0, 1},   // East
        new int[] {-1, 0},  // North
        new int[] {0, -1},  // West
        new int[] {1, 1},   // Southeast
        new int[] {1, -1},  // Southwest
        new int[] {-1, 1},  // Northeast
        new int[] {-1, -1}  // Northwest
    };
        private const int BOARD_SIZE = 8; // Standard Othello board size

        public bool CheckMove(char[][] board, int rowMove, int columnMove, char color)
        {
            // Loop through all possible directions
            foreach (int[] direction in DIRECTIONS)
            {
                int currentRow = rowMove;
                int currentColumn = columnMove;
                int moveLength = 0; // Length of the potential line of opponent's pieces between our pieces
                int rowDelta = direction[0], columnDelta = direction[1];

                // Keep moving in the direction while the next position is inside the board
                while (0 <= currentRow + rowDelta && currentRow + rowDelta < BOARD_SIZE
                        && 0 <= currentColumn + columnDelta && currentColumn + columnDelta < BOARD_SIZE)
                {
                    moveLength++; // Increase the length of the line
                    currentRow += rowDelta;
                    currentColumn += columnDelta;

                    // If the next position is either empty or contains a piece of the same color, break out of the loop
                    if (board[currentRow][currentColumn] == '.' || board[currentRow][currentColumn] == color)
                    {
                        break;
                    }
                }

                // Check if the last piece in the direction is the same color and the length of opponent's pieces is more than 1
                if (currentRow >= 0 && currentRow < BOARD_SIZE && currentColumn >= 0 && currentColumn < BOARD_SIZE &&
                    board[currentRow][currentColumn] == color && moveLength > 1)
                {
                    return true; // The move is valid as it brackets at least one line of opponent pieces
                }
            }
            return false; // If no direction is valid, the move is invalid
        }
    }


    /* 3092. Most Frequent IDs
    https://leetcode.com/problems/most-frequent-ids/description/
     */
    public class MostFrequentIDsSol
    {
        /*
        Complexity
Time complexity: O(nlogn)
Space complexity: O(n)
  */
        public long[] MostFrequentIDs(int[] nums, int[] freq)
        {
            List<long> ans = new List<long>();
            int n = nums.Length;
            PriorityQueue<long, long> pq = new PriorityQueue<long, long>(Comparer<long>.Create((x, y) => y.CompareTo(x)));
            Dictionary<long, long> dict = new Dictionary<long, long>();

            for (int i = 0; i < n; i++)
            {
                if (dict.ContainsKey(nums[i]))
                    dict[nums[i]] += freq[i];
                else
                    dict[nums[i]] = freq[i];

                pq.Enqueue(nums[i], dict[nums[i]]);

                while (true)
                {
                    pq.TryPeek(out long val, out long priorityval);
                    if (priorityval == dict[val])
                    {
                        ans.Add(priorityval);
                        break;
                    }
                    else
                        pq.Dequeue();
                }
            }

            return ans.ToArray();
        }
    }

    /* 2327. Number of People Aware of a Secret
    https://leetcode.com/problems/number-of-people-aware-of-a-secret/description/
    https://algo.monster/liteproblems/2327
     */
    class PeopleAwareOfSecretSol
    {
        /* Time and Space Complexity
        The given Python function peopleAwareOfSecret computes the number of people aware of a secret on the n-th day, under certain conditions of delay before sharing and forgetting the secret. The function implements a form of dynamic programming using arrays to simulate the process over n days.
        Time Complexity:
        •	The function has a primary loop that iterates over the range 1 to n + 1, which gives us an O(n) component.
        •	Inside this loop, there's a secondary while loop, for sharing the secret from the i + delay-th to the i + forget - 1-th day. In the worst case scenario, this while loop executes (forget - delay) times. Therefore, its contribution is O(forget - delay).
        •	Since both loops are nested and the while loop runs for every value of i, we might initially consider the time complexity to be O(n * (forget - delay)).
        However, the variable nxt is being incremented by 1 on each iteration without being reset for every i, and when reaching i + forget, the loop exits. Therefore, every element in the range 1 to n can only contribute to at most (forget - delay) increments over the entire function execution. Thus, the while-loop does not lead to a full cartesian product across n and (forget - delay).
        The actual time complexity is thus O(n + forget).
        Space Complexity:
        •	Two arrays d and cnt of maximum size m are used, where m = (n << 1) + 10 - sort of a safe upper bound to ensure the array can handle the indices that the algorithm will access. These arrays are the main contributors to space complexity.
        •	Thus, the space complexity of the function is O(m), which simplifies to O(n) since m is just a linear scaling of n.
        To summarize:
        •	Time Complexity: O(n + forget)
        •	Space Complexity: O(n)
         */
        // Constant for modulo operation to ensure the numbers do not get too large.
        private static readonly int MOD = (int)1e9 + 7;

        public int PeopleAwareOfSecret(int n, int delay, int forget)
        {
            // A buffer is added to manage the maximum days needed.
            int bufferLength = (n << 1) + 10;

            // Daily increase tracker.
            long[] dailyIncrease = new long[bufferLength];

            // People count tracker for each day.
            long[] peopleCount = new long[bufferLength];

            // Initially, on day 1, one person knows the secret.
            peopleCount[1] = 1;

            // Loop through each day.
            for (int i = 1; i <= n; ++i)
            {
                // If peopleCount[i] is positive, proceed to share the secret.
                if (peopleCount[i] > 0)
                {
                    // Add to the dailyIncrease.
                    dailyIncrease[i] = (dailyIncrease[i] + peopleCount[i]) % MOD;

                    // Subtract from the dailyIncrease after the forgetting period.
                    dailyIncrease[i + forget] = (dailyIncrease[i + forget] - peopleCount[i] + MOD) % MOD;

                    // Compute the next day when sharing starts, and continue until the forgetting day.
                    int nextShareDay = i + delay;
                    while (nextShareDay < i + forget)
                    {
                        peopleCount[nextShareDay] = (peopleCount[nextShareDay] + peopleCount[i]) % MOD;
                        ++nextShareDay;
                    }
                }
            }

            // Calculate the final answer by summing dailyIncrease for each day.
            long answer = 0;
            for (int i = 1; i <= n; ++i)
            {
                answer = (answer + dailyIncrease[i]) % MOD;
            }

            // Return the final number of people aware of the secret as an integer.
            return (int)answer;
        }
    }

    /* 
    1233. Remove Sub-Folders from the Filesystem
    https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/description/    
     */

    class RemoveSubfoldersFromFileSystemSol
    {
        /*
        Approach 1: Using HashSet
         Complexity Analysis
Let N be the number of folders and L be the maximum length of a folder path.
•	Time Complexity: O(N⋅L+N⋅L^2)=O(N⋅L^2)
Constructing the unordered set folderSet from the input array folder takes O(N). However, each string insertion requires O(L). So, initializing the set takes O(N⋅L).
The primary operation involves iterating over each folder path in the folder array, which is O(N).
o	For each folder, the algorithm checks all possible prefixes (up to L levels deep) in the folderSet. This involves:
o	Finding the position of the last '/' character in the prefix string, which takes O(L) in the worst case.
o	Creating a substring for each prefix level, which is also O(L).
o	Searching for each prefix in the set, which is O(L).
Therefore, checking all prefixes of one folder takes O(L^2), and for N folders, this results in O(N⋅L^2).
The initialization and main loop lead to a time complexity of O(N⋅L+N⋅L^2)≈O(N⋅L^2), as O(N⋅L^2) dominates.
•	Space complexity: O(N⋅L)
The folderSet stores each of the N folder paths. Each path can be as long as L, so the space complexity for the set is O(N⋅L).
The array result stores each non-subfolder path. In the worst case, if none of the folders are subfolders, this array also takes O(N⋅L) space.
Minor additional space is used for variables like isSubFolder and prefix. This additional space is constant, O(1), and does not affect the overall complexity.
The dominant space usage is from the folderSet and result array, leading to a total space complexity of O(N⋅L).
 */
        public IList<string> UsingHashSet(string[] folders)
        {
            // Create a set to store all folder paths for fast lookup
            HashSet<string> folderSet = new HashSet<string>(folders);
            List<string> result = new List<string>();

            // Iterate through each folder to check if it's a sub-folder
            foreach (string folder in folders)
            {
                bool isSubFolder = false;
                string currentPrefix = folder;

                // Check all prefixes of the current folder path
                while (!string.IsNullOrEmpty(currentPrefix))
                {
                    int position = currentPrefix.LastIndexOf('/');
                    if (position == -1) break;

                    // Reduce the prefix to its parent folder
                    currentPrefix = currentPrefix.Substring(0, position);

                    // If the parent folder exists in the set, mark as sub-folder
                    if (folderSet.Contains(currentPrefix))
                    {
                        isSubFolder = true;
                        break;
                    }
                }

                // If not a sub-folder, add it to the result
                if (!isSubFolder)
                {
                    result.Add(folder);
                }
            }

            return result;
        }
        /* Approach 2: Using Sorting
        Complexity Analysis
Let N be the number of folders and L be the maximum length of a folder path.
•	Time complexity: O(N⋅LlogN)
Sorting takes O(N⋅logN) comparisons, but each comparison can involve up to L characters (the maximum length of a folder path). Therefore, this step has a time complexity of O(N⋅LlogN).
The loop runs N−1 times. For each folder, it does the following:
o	Retrieves the last folder from result and appends a '/' to it, which takes O(L) time.
o	Uses compare to check if the current folder starts with the last added folder. This comparison will take O(L) time in the worst case.
Thus, the overall time complexity for this part is: O(N⋅L)
Therefore, combining the sorting and iteration steps, the total time complexity is: O(N⋅LlogN)+O(N⋅L)
Since O(N⋅LlogN) dominates O(N⋅L), we can simplify the time complexity to O(N⋅LlogN).
•	Space complexity: O(N⋅L)
The result array stores each folder that is not a sub-folder. In the worst case, every folder is added to result, which requires O(N⋅L) space.
The space taken by the sorting algorithm depends on the language of implementation:
In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logN).
In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logN).
In Python, the sort() method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has a space complexity of O(N).
Thus, the total space complexity is O(N⋅L)

         */
        public List<string> UsingSorting(string[] folders)
        {
            // Sort the folders alphabetically
            Array.Sort(folders);

            // Initialize the result list and add the first folder
            List<string> result = new List<string>();
            result.Add(folders[0]);

            // Iterate through each folder and check if it's a sub-folder of the last added folder in the result
            for (int index = 1; index < folders.Length; index++)
            {
                string lastAddedFolder = result[result.Count - 1];
                lastAddedFolder += '/';

                // Check if the current folder starts with the last added folder path
                if (!folders[index].StartsWith(lastAddedFolder))
                {
                    result.Add(folders[index]);
                }
            }

            // Return the result containing only non-sub-folders
            return result;
        }
        /* Approach 3: Using Trie
        Complexity Analysis
Let N be the number of folders and L be the maximum length of a folder path.
•	Time complexity: O(N×L)
For each folder path in folderPaths, the algorithm parses the path and inserts it into the Trie. Parsing each path takes O(L) time.
For each segment, checking and inserting into Trie’s map also takes O(L) time on average due to hash table operations (insertions and lookups in the map). Therefore, building the Trie for all N paths results in a total time complexity of O(N×L).
For each folder path, the algorithm traverses the Trie to check if it is a subfolder. Again, parsing the path takes O(L), and each lookup in the map takes O(1) on average. Therefore, checking all N folder paths also requires O(N×L) time.
Overall, both the Trie-building and subfolder-checking phases have a time complexity of O(N×L), so the total time complexity is: O(N×L)
•	Space complexity: O(N×L)
Each folder path can create up to L nodes in the Trie, depending on the path depth. In the worst case, if all folder paths are unique, we would end up storing all N×L segments. Therefore, the space required for the Trie structure is O(N×L).
The result array stores up to N folder paths, so its space requirement is O(N). Intermediate variables like iss and string use O(L) space for each folder path.
Since the Trie is the most space-consuming data structure in this solution, the overall space complexity is: O(N×L)

         */
        public List<String> UsingTrie(String[] folder)
        {
            // Build Trie from folder paths
            foreach (String path in folder)
            {
                TrieNode currentNode = root;
                String[] folderNames = path.Split("/");

                foreach (String folderName in folderNames)
                {
                    // Skip empty folder names
                    if (folderName.Equals("")) continue;
                    // Create new node if it doesn't exist
                    if (!currentNode.Children.ContainsKey(folderName))
                    {
                        currentNode.Children[folderName] = new TrieNode();
                    }
                    currentNode = currentNode.Children[folderName];
                }
                // Mark the end of the folder path
                currentNode.IsEndOfFolder = true;
            }

            // Check each path for subfolders
            List<String> result = new();
            foreach (String path in folder)
            {
                TrieNode currentNode = root;
                String[] folderNames = path.Split("/");
                bool isSubfolder = false;

                for (int i = 0; i < folderNames.Length; i++)
                {
                    // Skip empty folder names
                    if (folderNames[i].Equals("")) continue;

                    TrieNode nextNode = currentNode.Children[folderNames[i]];
                    // Check if the current folder path is a subfolder of an
                    // existing folder
                    if (nextNode.IsEndOfFolder && i != folderNames.Length - 1)
                    {
                        isSubfolder = true;
                        break; // Found a sub-folder
                    }

                    currentNode = nextNode;
                }
                // If not a sub-folder, add to the result
                if (!isSubfolder) result.Add(path);
            }

            return result;
        }
        public class TrieNode
        {

            public bool IsEndOfFolder;
            public Dictionary<String, TrieNode> Children;

            public TrieNode()
            {
                this.IsEndOfFolder = false;
                this.Children = new();
            }
        }

        TrieNode root;

    }

    /* 1976. Number of Ways to Arrive at Destination
    https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/description/
    https://algo.monster/liteproblems/1976
     */
    class CountPathsToArriveAtDestinationSol
    {
        // Define constants for infinite distance and modulo value
        private static readonly long INFINITY = long.MaxValue / 2;
        private static readonly int MOD = (int)1e9 + 7;

        /* Time and Space Complexity
        The provided Python code is an implementation of Dijkstra's algorithm to find the shortest paths from a single source to all other nodes in a graph, here specifically tailored to count all the distinct ways one can travel from node 0 to node n-1 given the list of roads. Each road has two nodes it connects and the time taken to travel that road.
        Time Complexity
        The time complexity of the code depends on two nested loops: the outer loop runs n times (where n is the number of nodes), and the inner loop, which in the worst case, also runs n times to find the node with the minimum distance that hasn't been visited yet.
        Furthermore, the adjacent nodes are visited in another nested loop that also iterates up to n times. Consequently, the worst-case time complexity of this algorithm is O(n^3), since for each node, we potentially inspect all other nodes to update the distances and ways.
        Space Complexity
        The space complexity is primarily dependent on the storage of the graph, distance array, ways array, and the visited array.
        •	Graph g is represented as a 2D matrix and occupies O(n^2) space.
        •	Distance array dist, ways array w, and visited array vis each take O(n) space.
        Adding these up gives us a total space complexity of O(n^2), with the graph matrix being the dominant term.
         */
        public int CountPaths(int n, int[][] roads)
        {
            long[][] graph = new long[n][];
            long[] distances = new long[n];
            long[] ways = new long[n];
            bool[] visited = new bool[n];

            // Initialize the graph with infinite distances and distances array
            for (int i = 0; i < n; ++i)
            {
                Array.Fill(graph[i], INFINITY);
                Array.Fill(distances, INFINITY);
            }

            // Fill the graph with actual road data
            foreach (int[] road in roads)
            {
                int from = road[0], to = road[1], time = road[2];
                graph[from][to] = time;
                graph[to][from] = time;
            }

            // Set the distance from start point to itself as zero
            graph[0][0] = 0;
            distances[0] = 0;
            ways[0] = 1; // There's one way to reach the start point (itself)

            // Dijkstra's Algorithm to find shortest paths
            for (int i = 0; i < n; ++i)
            {
                int current = -1;
                // Find the unvisited vertex with the smallest distance
                for (int j = 0; j < n; ++j)
                {
                    if (!visited[j] && (current == -1 || distances[j] < distances[current]))
                    {
                        current = j;
                    }
                }
                visited[current] = true;

                // Update distances and count of ways for all neighbors
                for (int j = 0; j < n; ++j)
                {
                    if (j == current)
                    {
                        continue; // Skip if it's the current vertex
                    }

                    long newDistance = distances[current] + graph[current][j];

                    // If a shorter path to neighbor is found, update the distance and ways
                    if (distances[j] > newDistance)
                    {
                        distances[j] = newDistance;
                        ways[j] = ways[current];
                    }
                    // If another path with the same length is found, increment the ways
                    else if (distances[j] == newDistance)
                    {
                        ways[j] = (ways[j] + ways[current]) % MOD;
                    }
                }
            }

            // Return the number of ways to reach the last vertex (n-1)
            return (int)ways[n - 1];
        }
    }

    /* 1227. Airplane Seat Assignment Probability
    https://leetcode.com/problems/airplane-seat-assignment-probability/description/
    https://algo.monster/liteproblems/1227
     */
    class AirplaneSeatAssignmentProbabilitySol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the function is O(1) because it simply returns a constant value without any loops or recursive calls. It does not depend on the input size n, except for a single conditional check.
Space Complexity
The space complexity is also O(1) as the function uses a fixed small amount of space, only enough to store the return value, which does not change with the input size n.
 */
        // This method calculates the probability of the nth person getting the nth seat.
        public double NthPersonGetsNthSeat(int n)
        {
            // If there is only one person, the probability of the first person 
            // getting the first seat is 100%.
            if (n == 1)
            {
                return 1.0;
            }
            else
            {
                // For any number of people greater than 1, the probability of the 
                // nth person getting the nth seat is always 50%.
                // This is a result of a known probability puzzle where the first person
                // takes a random seat, and each subsequent person takes their own seat if 
                // available, or a random seat otherwise. Through detailed analysis, the
                // probability converges to 50% for the last person.
                return 0.5;
            }
        }
    }

    /* 2512. Reward Top K Students
    https://leetcode.com/problems/reward-top-k-students/description/
    https://algo.monster/liteproblems/2512
     */
    public class RewardTopKStuendsSol
    {
        /* Time Complexity
        The time complexity of the provided code is O(n * log n + (|ps| + |ns| + n) * |s|). To break it down:
        •	zip(student_id, report) operation is O(n) because it is iterating through the list of student IDs and their corresponding reports.
        •	Splitting the report and checking if each word is in the sets positive_feedback (ps) or negative_feedback (ns) contributes O(n * |s|), where n is the number of reports and |s| is the average length of a report since we consider |s| for the average length of words in a report.
        •	The presence checks in ps and ns is O(1) on average for hash sets, which is multiplied with n * |s| for all words in all reports.
        •	Sorting the arr list of tuples based on a compound key contributes O(n * log n) since Python's sort function uses TimSort, which has this complexity.
        Space Complexity
        The space complexity of the code is O((|ps|+|ns|) * |s| + n). Here's how this is derived:
        •	Space for the sets ps and ns contributes |ps| * |s| and |ns| * |s|, respectively, because each word of average length |s| is stored in these sets.
        •	The arr list, which stores tuples of total score and student IDs for n students, contributes O(n) to space.
         */
        public List<int> TopStudents(string[] positiveFeedback, string[] negativeFeedback,
                                      string[] reports, int[] studentIds, int k)
        {

            // Convert positive and negative feedbacks to sets for quick lookup
            HashSet<string> positiveFeedbackSet = new HashSet<string>(positiveFeedback);
            HashSet<string> negativeFeedbackSet = new HashSet<string>(negativeFeedback);

            // Initialize the number of reports
            int numberOfReports = reports.Length;

            // Create an array to store scores and corresponding student IDs
            int[][] scoresAndStudentIds = new int[numberOfReports][];

            // Iterate over each report
            for (int i = 0; i < numberOfReports; ++i)
            {
                // Get the student ID for the current report
                int studentId = studentIds[i];
                // Initialize score for the current report
                int score = 0;

                // Split the report into words
                foreach (var word in reports[i].Split(' '))
                {
                    // Check if the word is in positive feedback set
                    if (positiveFeedbackSet.Contains(word))
                    {
                        score += 3; // Add 3 to the score for positive feedback
                    }
                    else if (negativeFeedbackSet.Contains(word))
                    {
                        score -= 1; // Subtract 1 from the score for negative feedback
                    }
                }
                // Assign computed score and student ID to the scores array
                scoresAndStudentIds[i] = new int[] { score, studentId };
            }

            // Sort the array first by scores in descending order, and then by student IDs in ascending order
            Array.Sort(scoresAndStudentIds, (a, b) =>
            {
                if (a[0] == b[0]) return a[1].CompareTo(b[1]); // Same score, sort by ID
                return b[0].CompareTo(a[0]); // Different scores, sort by score
            });

            // Initialize the list to store top k student IDs
            List<int> topStudents = new List<int>();

            // Extract the top k student IDs
            for (int i = 0; i < k; ++i)
            {
                topStudents.Add(scoresAndStudentIds[i][1]);
            }

            // Return the list of top k student IDs
            return topStudents;
        }
    }

    /* 1711. Count Good Meals
    https://leetcode.com/problems/count-good-meals/description/
    https://algo.monster/liteproblems/1711
     */
    class CountGoodMealsSol
    {
        // Define the modulus value for large numbers to avoid overflow
        private static readonly int MOD = (int)1e9 + 7;

        // Method to count the total number of pairs with power of two sums
        public int CountPairs(int[] deliciousness)
        {
            // Create a hashmap to store the frequency of each value in the deliciousness array
            Dictionary<int, int> frequencyMap = new();
            foreach (int value in deliciousness)
            {
                frequencyMap[value] = frequencyMap.GetValueOrDefault(value, 0) + 1;
            }

            long pairCount = 0; // Initialize the pair counter to 0

            // Loop through each power of 2 up to 2^21 (because 2^21 is the closest power of 2 to 10^9)
            for (int i = 0; i < 22; ++i)
            {
                int sum = 1 << i; // Calculate the sum which is a power of two
                foreach (var entry in frequencyMap)
                {
                    int firstElement = entry.Key;   // Key in the map is a part of the deliciousness pair
                    int firstCount = entry.Value;   // Value in the map is the count of that element
                    int secondElement = sum - firstElement;   // Find the second element of the pair

                    // Check if the second element exists in the map
                    if (!frequencyMap.ContainsKey(secondElement))
                    {
                        continue; // If it doesn't, continue to the next iteration
                    }

                    // If the second element exists, increment the pair count
                    // If both elements are the same, we must avoid counting the pair twice
                    pairCount += (long)firstCount * (firstElement == secondElement ? firstCount - 1 : frequencyMap[secondElement]);
                }
            }

            // Divide the result by 2 because each pair has been counted twice
            pairCount >>= 1;

            // Return the result modulo MOD to get the answer within the range
            return (int)(pairCount % MOD);
        }
    }

    /* 3295. Report Spam Message
    https://leetcode.com/problems/report-spam-message/description/
     */
    //C# Code
    public class ReportSpamMessageSol
    {
        /* Complexity
Time complexity:
 O(N + M) // N -> MessageSize
Space complexity:
 O(M) // M ->  */
        public bool ReportSpam(string[] message, string[] bannedWords)
        {
            HashSet<string> st = new HashSet<string>();
            foreach (string ban in bannedWords)
                st.Add(ban);
            var cnt = 0;
            foreach (string w in message)
            {
                if (st.Contains(w))
                    cnt++;
            }
            return cnt >= 2;
        }
    }


    /* 544. Output Contest Matches
           https://leetcode.com/problems/output-contest-matches/description/
            */
    class FindContestMatchSol
    {
        /* Approach #1: Simulation [Accepted]
         Complexity Analysis
•	Time Complexity: O(NlogN). Each of O(logN) rounds performs O(N) work.
•	Space Complexity: O(NlogN).
*/
        public String UsingSimulation(int n)
        {
            String[] team = new String[n];
            for (int i = 1; i <= n; ++i)
                team[i - 1] = "" + i;

            for (; n > 1; n /= 2)
                for (int i = 0; i < n / 2; ++i)
                    team[i] = "(" + team[i] + "," + team[n - 1 - i] + ")";

            return team[0];
        }
        /* Approach #2: Linear Write [Accepted]
Complexity Analysis
•	Time Complexity: O(N). We print each of the O(N) characters in order.
•	Space Complexity: O(N).

         */
        private int[] team;
        private int t;
        private StringBuilder ans;

        public string UsingLinearWrite(int n)
        {
            team = new int[n];
            t = 0;
            ans = new StringBuilder();
            Write(n, BitOperations.Log2((uint)n) - BitOperations.TrailingZeroCount((uint)n));
            return ans.ToString();
        }

        public void Write(int n, int round)
        {
            if (round == 0)
            {
                //TODO: fix below code line
                int w = 1; // BitOperations..LowestOneBit(t);
                team[t] = w > 0 ? n / w + 1 - team[t - w] : 1;
                ans.Append(team[t++].ToString());
            }
            else
            {
                ans.Append("(");
                Write(n, round - 1);
                ans.Append(",");
                Write(n, round - 1);
                ans.Append(")");
            }
        }

    }


    /* 1276. Number of Burgers with No Waste of Ingredients
    https://leetcode.com/problems/number-of-burgers-with-no-waste-of-ingredients/description/
    https://algo.monster/liteproblems/1276
     */
    class NumOfBurgersWithNoWasteOfIngredientsSol
    {
        /* Time and Space Complexity
        Time Complexity
        The given code consists of simple arithmetic calculations and conditional checks, which do not depend on the size of the input but are executed a constant number of times. Therefore, the time complexity is O(1).
        Space Complexity
        The space complexity of the code is also O(1) since it uses a fixed amount of space for the variables k, y, x, and the return list regardless of the input size. The solution does not utilize any additional data structures that grow with the size of the input.
         */
        // Method to calculate the number of Jumbo and Small burgers that can be made given
        // the number of tomato and cheese slices.
        public List<int> NumOfBurgers(int tomatoSlices, int cheeseSlices)
        {
            // Calculate the difference between four times the number of cheese slices and tomato slices
            int difference = 4 * cheeseSlices - tomatoSlices;

            // Calculate the number of Small burgers by dividing the difference by 2
            int numSmallBurgers = difference / 2;

            // Calculate the number of Jumbo burgers by subtracting the number of Small burgers 
            // from the total cheese slices
            int numJumboBurgers = cheeseSlices - numSmallBurgers;

            // Check if difference is even and both calculated burger amounts are non-negative
            bool isSolutionValid = difference % 2 == 0 && numSmallBurgers >= 0 && numJumboBurgers >= 0;

            // Return the number of Jumbo and Small burgers, if possible; otherwise, return an empty list
            return isSolutionValid ? new List<int> { numJumboBurgers, numSmallBurgers } : new List<int>();
        }
    }

    /* 1311. Get Watched Videos by Your Friends
    https://leetcode.com/problems/get-watched-videos-by-your-friends/description/
    https://algo.monster/liteproblems/1311
     */

    public class WatchedVideosByFriendsSol
    {
        /* Time and Space Complexity
        Time Complexity
        The time complexity of the code involves several parts:
        1.	BFS Traversal: We are using a Breadth-First Search (BFS) algorithm to find all friends at the given level. BFS has a time complexity of O(V + E), where V is the number of vertices (or friends, in this context) and E is the number of edges (or friendships). In the worst case, the BFS could potentially visit all n vertices and all connections between friends, so this part of the code is O(n + E).
        2.	Counting Frequency: After BFS, the code counts the frequency of each video watched by friends at the given level. Assuming the number of friends at that level is F and each friend has watched at most W videos, the time complexity for counting frequency would be O(FW) because we iterate through each friend and update the counter for every video they have watched.
        3.	Sorting Videos: Next, the algorithm sorts the videos by frequency and by their names if frequencies are equal using a custom sorting function. Assuming there are V distinct videos, the sorting takes O(V * log(V)) time because sorting algorithms generally have an O(n * log(n)) complexity where n is the number of items to sort.
        Therefore, the overall time complexity of the code is O(n + E + FW + V * log(V)). However, considering that E, FW, and V are all limited by n (since E <= n(n-1)/2, FW <= nW, and V <= nW), we can simplify the time complexity to O(n^2 + nW + nW * log(nW)).
        Space Complexity
        The space complexity is determined by the additional space required for the BFS queue, the visitation list, and the counters for the videos:
        1.	Visitation List: The vis list keeps track of whether a friend has been visited during BFS. It contains one entry per friend, so its space complexity is O(n).
        2.	BFS Queue: In the worst case, the BFS queue can have up to n elements (if every friend is at the required level), so its space complexity is O(n).
        3.	Counters for Videos: The Counter for videos and the list of videos after conversion (freq and videos) can store at most FW elements, which is O(FW). In the worst case, the space complexity for counting videos is equal to the total number of videos watched by all friends, which is O(n * W) if each person has watched at most W videos.
        Hence, the overall space complexity is O(n + n + n * W) which can be simplified to O(nW) since W >= 1.
         */
        public List<string> WatchedVideosByFriends(List<List<string>> watchedVideos, int[][] friends, int id, int level)
        {

            int totalFriends = friends.Length; // Total number of friends.
            bool[] visited = new bool[totalFriends]; // Keep track of visited friends.
            Queue<int> queue = new Queue<int>(); // Queue for BFS (Breadth-First Search).
            queue.Enqueue(id); // Starting with the given friend's ID.
            visited[id] = true;

            // Perform BFS to reach the friends at the specified level.
            while (level-- > 0)
            {
                int size = queue.Count;
                for (int i = 0; i < size; ++i)
                {
                    int currentFriend = queue.Dequeue();
                    // Enqueue all unvisited friends of the current friend.
                    foreach (int friendId in friends[currentFriend])
                    {
                        if (!visited[friendId])
                        {
                            queue.Enqueue(friendId);
                            visited[friendId] = true;
                        }
                    }
                }
            }

            // Count the frequency of each video watched by friends at the given level.
            Dictionary<string, int> frequencyMap = new Dictionary<string, int>();
            while (queue.Count > 0)
            {
                int friendAtLevel = queue.Dequeue();
                foreach (string video in watchedVideos[friendAtLevel])
                {
                    if (frequencyMap.ContainsKey(video))
                    {
                        frequencyMap[video]++;
                    }
                    else
                    {
                        frequencyMap[video] = 1;
                    }
                }
            }

            // Convert the frequency map to a list, to sort them.
            List<KeyValuePair<string, int>> frequencyList = new List<KeyValuePair<string, int>>(frequencyMap);
            frequencyList.Sort((entry1, entry2) =>
            {
                if (entry1.Value != entry2.Value)
                {
                    return entry1.Value.CompareTo(entry2.Value);
                }
                // If frequencies are equal, sort alphabetically.
                return entry1.Key.CompareTo(entry2.Key);
            });

            // Extract the sorted video names.
            List<string> result = new List<string>();
            foreach (KeyValuePair<string, int> entry in frequencyList)
            {
                result.Add(entry.Key);
            }

            return result; // Final sorted list of videos.
        }
    }

    /* 1333. Filter Restaurants by Vegan-Friendly, Price and Distance
    https://leetcode.com/problems/filter-restaurants-by-vegan-friendly-price-and-distance/description/
    https://algo.monster/liteproblems/1333
     */
    class FilterRestaurantsSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the provided solution primarily depends on the sorting operation and the subsequent filtering of the restaurants list based on the specified conditions.
•	Sorting: The sort() function is used, which generally has a time complexity of O(n log n), where n is the number of restaurants.
•	Filtering: The for loop iterates over each restaurant, performing constant-time checks (if conditions) for each. This gives us O(n).
The overall time complexity combines the above operations, where sorting dominates, resulting in O(n log n) + O(n). Since the O(n log n) term is the dominant factor, the overall time complexity simplifies to O(n log n).
Space Complexity
The space complexity of the solution involves the storage required for the sorted list and the answer list.
•	Sorted List: The in-place sort() method is used, so it does not require additional space apart from the input list. Hence, the space required remains O(1) as an auxiliary space.
•	Answer List: In the worst case, all restaurants might satisfy the given conditions, so the ans list could contain all restaurant IDs. Therefore, the space complexity for the ans list is O(n).
Taking both into consideration, the overall space complexity of the solution is O(n), where n is the number of restaurants.
 */
        public List<int> FilterRestaurants(int[][] restaurants, int veganFriendly, int maxPrice, int maxDistance)
        {
            // Sort the array of restaurants based on their ratings in descending order
            // If the ratings are the same, sort based on the restaurant IDs in descending order
            Array.Sort(restaurants, (restaurant1, restaurant2) =>
            {
                if (restaurant1[1] == restaurant2[1])
                    return restaurant2[0] - restaurant1[0]; // Sort by ID if ratings are equal
                else
                    return restaurant2[1] - restaurant1[1]; // Sort by Rating
            });

            // Initialize a list to store the IDs of restaurants that satisfy the conditions
            List<int> filteredRestaurants = new List<int>();

            // Iterate through the sorted array of restaurants
            foreach (int[] restaurant in restaurants)
            {
                // Check if the restaurant satisfies the conditions for vegan-friendly, max price and max distance
                if (restaurant[2] >= veganFriendly && restaurant[3] <= maxPrice && restaurant[4] <= maxDistance)
                {
                    // Add the restaurant ID to the list
                    filteredRestaurants.Add(restaurant[0]);
                }
            }

            // Return the list of filtered restaurant IDs
            return filteredRestaurants;
        }
    }

    /* 1348. Tweet Counts Per Frequency
    https://leetcode.com/problems/tweet-counts-per-frequency/description/
    https://algo.monster/liteproblems/1348
     */
    public class TweetCounts
    {
        /* Time and Space Complexity
Time Complexity:
__init__ Method:
•	Time Complexity is O(1) for initializing the dictionary and other variables.
recordTweet Method:
•	Inserting an element into a SortedList costs O(log n) time, where n is the number of elements currently in the list for the respective tweetName. This is due to the binary search used to find the correct position and the insertion operation which on average takes O(log n) time in a balanced binary search tree.
getTweetCountsPerFrequency Method:
•	The time spent in this method is dominated by the while loop and the bisect operations.
•	The while loop runs once for each interval of size f = self.d[freq] in the range from startTime to endTime. There are (endTime - startTime) / f + 1 such intervals.
•	Within the loop, two bisect operations (actually bisect_left) are performed. Each bisect_left takes O(log m) time, where m is the number of recorded times for the tweetName.
•	Hence, the total time complexity for the getTweetCountsPerFrequency method is O((endTime - startTime)/f * log m).
Space Complexity:
__init__ Method:
•	Space Complexity is O(1) for initializing the dictionary and other variables that do not depend on the input size.
recordTweet Method:
•	The space complexity is dominated by the space used to store the tweet times in the SortedList. In the worst case, if all n recordTweet operations are for the same tweetName, the space complexity would be O(n).
getTweetCountsPerFrequency Method:
•	The space complexity is O(k), where k is the number of intervals. This is because we need to store a count for each interval.
Combining the space complexities from the different parts of the class, the overall space complexity of the TweetCounts class is O(n + k), where n is the total number of tweet times recorded, and k is the number of intervals calculated in a single call to getTweetCountsPerFrequency.
 */
        // Dictionary storing tweet names as keys and another SortedDictionary as values,
        // which maps a timestamp to the number of tweets at that time.
        private Dictionary<string, SortedDictionary<int, int>> tweetData = new Dictionary<string, SortedDictionary<int, int>>();

        /**
         * Constructor for TweetCounts.
         */
        public TweetCounts()
        {
            // No initialization needed since tweetData is already initialized.
        }

        /**
         * Record a tweet at a given time.
         *
         * @param tweetName the name of the tweet
         * @param time      the timestamp the tweet occurred
         */
        public void RecordTweet(string tweetName, int time)
        {
            if (!tweetData.ContainsKey(tweetName))
            {
                tweetData[tweetName] = new SortedDictionary<int, int>();
            }
            var timeMap = tweetData[tweetName];
            if (timeMap.ContainsKey(time))
            {
                timeMap[time]++;
            }
            else
            {
                timeMap[time] = 1;
            }
        }

        /**
         * Retrieves the count of tweets per frequency in a time range.
         *
         * @param frequency  the frequency ("minute", "hour", or "day")
         * @param tweetName  the name of the tweet
         * @param startTime  the start of the time range (inclusive)
         * @param endTime    the end of the time range (inclusive)
         * @return a list of tweet counts per frequency
         */
        public List<int> GetTweetCountsPerFrequency(string frequency, string tweetName, int startTime, int endTime)
        {
            // Convert frequency to seconds
            int intervalDurationInSeconds = 60; // Default to minute
            if (frequency == "hour")
            {
                intervalDurationInSeconds = 3600; // 60 minutes * 60 seconds
            }
            else if (frequency == "day")
            {
                intervalDurationInSeconds = 86400; // 24 hours * 60 minutes * 60 seconds
            }

            if (!tweetData.ContainsKey(tweetName))
            {
                return new List<int>(); // Return empty list if tweetName does not exist
            }

            var timeMap = tweetData[tweetName];
            List<int> counts = new List<int>(); // Holds the final counts

            // Iterate through time ranges and count tweets
            for (int i = startTime; i <= endTime; i += intervalDurationInSeconds)
            {
                int sum = 0;
                // Calculate end time for this interval, ensuring we don't exceed endTime
                int intervalEnd = Math.Min(i + intervalDurationInSeconds, endTime + 1);
                // Sum tweet counts within the current interval
                foreach (var count in timeMap)
                {
                    if (count.Key >= i && count.Key < intervalEnd)
                    {
                        sum += count.Value;
                    }
                }
                counts.Add(sum);
            }
            return counts;
        }
        /**
    * The following code showcases how the TweetCounts class may be used.
    * It is not part of the TweetCounts class itself.
    */
        /*
        TweetCounts tweetCounter = new TweetCounts();
        tweetCounter.RecordTweet("tweet1", 10);
        List<int> counts = tweetCounter.GetTweetCountsPerFrequency("minute", "tweet1", 0, 59);
        */
    }


    /* 1487. Making File Names Unique
    https://leetcode.com/problems/making-file-names-unique/description/
    https://algo.monster/liteproblems/1487
     */
    class GetFolderNamesSolution
    {
        /* Time and Space Complexity
The provided code snippet maintains a dictionary to keep track of the number of times a particular folder name has appeared. It then generates unique names by appending a number in parentheses if necessary. Let's analyze the time and space complexity:
Time Complexity
1.	The for loop iterates through each name in the input list, so it gives us an O(n) term, where n is the number of names in the list.
2.	Inside the loop, the code checks if name exists in the dictionary with if name in d, which is an O(1) operation due to hashing.
3.	If there is a conflict for a name, the code enters into a while loop to find the correct suffix k. In the worst case, it may run as many times as there are names with the same base, let's call this number m.
4.	Each iteration of the while loop involves a string concatenation and a dictionary lookup, both of which are O(1) operations.
The while loop could contribute significantly to the time complexity if there are many names with the same base. If m is the maximum number of duplicates for any name, the worst-case time complexity for while loop iterations across the entire input is O(m*n) because each unique name will go through the loop at most m times.
Therefore, the overall worst-case time complexity is O(n + m*n) which simplifies to O(m*n).
Space Complexity
The space complexity of the algorithm can be broken down as:
1.	The dictionary d, which stores each unique folder name. In the worst case (all folder names are unique, and for each original name, there’s a different number of folders), this dictionary could potentially store 2 entries per name (the original and the latest modified version). So, in the worst case, it could hold up to 2n entries, contributing O(n) to the space complexity.
2.	Temporary storage for constructing new folder names during the while loop does not significantly add to the space complexity because it only holds a single name at a time.
Therefore, the overall space complexity is O(n).
To summarize:
•	Time Complexity: O(m*n)
•	Space Complexity: O(n)
 */
        public String[] GetFolderNames(String[] names)
        {
            // Create a map to store the highest integer k used for each original name.
            Dictionary<String, int> nameMap = new();

            // Iterate through each name in the input array.
            for (int i = 0; i < names.Length; ++i)
            {
                // If the current name has already been encountered,
                // append a unique identifier to make it distinct.
                if (nameMap.ContainsKey(names[i]))
                {
                    // Get the current highest value of k used for this name.
                    int k = nameMap[names[i]];
                    // Check if the name with the appended "(k)" already exists.
                    // If it does, increment k until a unique name is found.
                    while (nameMap.ContainsKey(names[i] + "(" + k + ")"))
                    {
                        ++k;
                    }

                    // Update the map with the new highest value of k for this name.
                    nameMap[names[i]] = k;
                    // Modify the current name by appending "(k)" to make it unique.
                    names[i] += "(" + k + ")";
                }

                // Insert or update the current name in the map, setting its value to 1.
                // If the name is new, this records it as seen for the first time.
                // If the name has been modified, this ensures it's treated as new.
                nameMap[names[i]] = 1;
            }

            // Return the modified array of names, with each name now unique.
            return names;
        }
    }

    /* 1583. Count Unhappy Friends
    https://leetcode.com/problems/count-unhappy-friends/description/
    https://algo.monster/liteproblems/1583
     */
    class CountUnhappyFriendsSol
    {
        /* Time and Space Complexity
Time Complexity
The time complexity of the code is determined by several factors:
1.	The comprehension used to create the dictionary d takes O(n^2) time, where n is the number of friends, because the preferences list for each friend is of length n-1, and there are n such lists.
2.	The loop setting up the p dictionary runs for O(n/2) pairs, which simplifies to O(n) because each pair is processed exactly once.
3.	The outer loop to calculate ans traverses n friends, which gives us O(n).
4.	Inside this outer loop, there's an any function call on a generator expression. In the worst case, it scans through n-2 elements (since one element is the friend themselves, and another is the paired friend). Therefore, this gives us O(n-2) for each friend.
When you put these factors together, the worst-case time complexity is O(n^2) due to the initial comprehension for the dictionary d. All the other operations, although they depend on n, don't involve nested loops over n, so they don't contribute a higher order term than n^2.
Thus, the final time complexity is O(n^2).
Space Complexity
Now let's analyze the space complexity:
1.	The d dictionary stores preferences for each of the n friends using a dictionary, so it takes O(n^2) space.
2.	The p dictionary holds the paired friends' information, with n entries (two entries for each pair). Thus, it consumes O(n) space.
3.	The space for ans and loop variables is constant, O(1).
Adding these together, the dominant term is the O(n^2) space required for the d dictionary. Thus, the total space complexity is also O(n^2).
 */
        public int UnhappyFriends(int n, int[][] preferences, int[][] pairs)
        {
            // Distance matrix indicating how strongly each friend prefers other friends
            int[][] preferenceDistances = new int[n][];
            // Fill the preference distance matrix with preference rankings
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n - 1; ++j)
                {
                    preferenceDistances[i][preferences[i][j]] = j;
                }
            }

            // Pairing array where the index is the friend and the value is their pair
            int[] pairings = new int[n];
            // Fill the pairings array based on the given pairs
            foreach (int[] pair in pairs)
            {
                int friend1 = pair[0], friend2 = pair[1];
                pairings[friend1] = friend2;
                pairings[friend2] = friend1;
            }

            // Counter for unhappy friends
            int unhappyCount = 0;
            // Iterate over all friends to determine unhappiness
            for (int friendX = 0; friendX < n; ++friendX)
            {
                int friendY = pairings[friendX];
                bool isUnhappy = false; // Flag to check if the current friend is unhappy

                // Check if there exists a friend that friendX ranks higher than their current paired friendY
                for (int i = 0; i < preferenceDistances[friendX][friendY]; ++i)
                {
                    int otherFriend = preferences[friendX][i];
                    // Check if the other friend (u) prefers friendX over their own pairing
                    if (preferenceDistances[otherFriend][friendX] < preferenceDistances[otherFriend][pairings[otherFriend]])
                    {
                        isUnhappy = true;
                        break;
                    }
                }
                // Increment unhappyCount if friendX is found to be unhappy
                if (isUnhappy)
                {
                    unhappyCount++;
                }
            }
            return unhappyCount; // Return the total number of unhappy friends
        }
    }

    /* 1599. Maximum Profit of Operating a Centennial Wheel
    https://leetcode.com/problems/maximum-profit-of-operating-a-centennial-wheel/description/
    https://algo.monster/liteproblems/1599
     */
    class MinOperationsMaxProfitSol
    {
        /* Time and Space Complexity
        Time Complexity
        The time complexity of the code is O(n), where n is the number of elements in the customers list. This is because the loop runs for each customer in the customers list plus additional iterations for any remaining waiting customers after the end of the list. The operations inside the loop are constant time, which means they do not depend on the size of the input list, so the time complexity is linear with regard to the number of elements in the input.
        Space Complexity
        The space complexity of the code is O(1). The function uses a fixed number of variables (ans, mx, t, wait, i, and up) whose space requirement does not scale with the input size (the number of customers). Hence, the space used by the algorithm is constant.
         */
        public int MinOperationsMaxProfit(int[] customers, int boardingCost, int runningCost)
        {
            // Initialize variables to store:
            // 'maxProfit': the current maximum profit seen (initialized to 0).
            // 'totalOperations': the total number of operations to achieve the 'maxProfit'.
            // 'waitingCustomers': the number of customers waiting to board.
            // 'currentRotation': the current rotation/round of the gondola.
            int maxProfit = 0;
            int totalOperations = -1; // Begins at -1 to handle cases when no profit can be made.
            int waitingCustomers = 0;
            int currentRotation = 0;

            // Loop through all the customers or until there are no more waiting customers.
            while (waitingCustomers > 0 || currentRotation < customers.Length)
            {
                // Add the customers arriving in the current rotation to 'waitingCustomers'.
                if (currentRotation < customers.Length)
                {
                    waitingCustomers += customers[currentRotation];
                }

                // Calculate the number of people to board in this rotation. 
                // It should not exceed 4, which is the gondola's capacity.
                int boardingCustomers = Math.Min(4, waitingCustomers);

                // Decrease the count of 'waitingCustomers' by the number of people who just boarded.
                waitingCustomers -= boardingCustomers;

                // Move to the next rotation.
                currentRotation++;

                // Calculate the total profit after this rotation.
                int profitThisRotation = boardingCustomers * boardingCost - runningCost;

                // Add the profit from this rotation to the total profit.
                maxProfit += profitThisRotation;

                // Check if the total profit we just calculated is greater than the previously recorded maximum profit.
                // If it is, update the 'maxProfit' and 'totalOperations'.
                if (maxProfit > 0 && maxProfit > maxProfit)
                {
                    maxProfit = maxProfit;
                    totalOperations = currentRotation;
                }
            }

            // Return the total number of operations needed to reach maximum profit.
            // If a profit cannot be made, 'totalOperations' would be -1.
            return totalOperations;
        }
    }

    /* 1620. Coordinate With Maximum Network Quality
    https://leetcode.com/problems/coordinate-with-maximum-network-quality/description/
    https://algo.monster/liteproblems/1620
     */

    class BestCoordinateWithMaxNetworkQualitySol
    {
        /* Time and Space Complexity
Time Complexity
The given code runs three nested loops:
•	The first two loops iterate over a fixed range of 51 each, resulting in 51 * 51 iterations.
•	The innermost loop iterates over the list of towers. If n is the total number of towers, this loop will execute n times for each iteration of the first two loops.
Therefore, the time complexity can be determined by multiplying the number of iterations in each loop. This gives us 51 * 51 * n, which simplifies to O(n) since the 51 * 51 is a constant.
Thus, the overall time complexity is O(n).
Space Complexity
The space complexity of the given code is quite straightforward. There are a few integers (mx, i, j, t, x, y, q, d) and a list ans of size 2 being used to store the answer, which do not depend on the input size.
Therefore, since no additional space is used that scales with the input size, the space complexity is O(1), or constant space complexity.
 */
        public int[] BestCoordinate(int[][] towers, int radius)
        {
            int maxSignal = 0; // to keep track of the highest signal quality
            int[] bestCoordinates = new int[] { 0, 0 }; // to hold the best coordinates

            // Loop through each possible coordinate on the grid up to 50x50
            for (int x = 0; x < 51; x++)
            {
                for (int y = 0; y < 51; y++)
                {
                    int signalQuality = 0; // to calculate the total signal quality at the point (x, y)

                    // Check each tower's contribution to the signal quality at the point (x, y)
                    foreach (int[] tower in towers)
                    {
                        // Calculate the distance between tower and point (x, y)
                        double distance = Math.Sqrt(Math.Pow(x - tower[0], 2) + Math.Pow(y - tower[1], 2));

                        // Add the tower's signal contribution if it is within radius
                        if (distance <= radius)
                        {
                            signalQuality += (int)Math.Floor(tower[2] / (1 + distance));
                        }
                    }

                    // Update the maximum signal quality and the best coordinates if the current point is better
                    if (maxSignal < signalQuality)
                    {
                        maxSignal = signalQuality;
                        bestCoordinates = new int[] { x, y };
                    }
                }
            }

            // Return the coordinates with the best signal quality
            return bestCoordinates;
        }
    }

    /* 1765. Map of Highest Peak
    https://leetcode.com/problems/map-of-highest-peak/description/
    https://algo.monster/liteproblems/1765
     */
    public class HighestPeakSol
    {
        /* Time and Space Complexity
The time complexity of the code is O(m * n), where m is the number of rows and n is the number of columns in the matrix isWater. This is because the algorithm must visit every cell at least once to determine its height in the final output matrix, and in the worst case, each cell is enqueued and dequeued exactly once in the BFS process.
The space complexity of the code is also O(m * n) because it requires a queue to store at least one entire level of the grid's cells, and each cell is stored once. Also, the ans matrix of size m * n is maintained to store the heights of each cell in the final output.
 */
        public int[,] HighestPeak(int[,] isWater)
        {
            // Obtain dimensions of the input matrix.
            int rowCount = isWater.GetLength(0);
            int columnCount = isWater.GetLength(1);

            // Initialize the answer matrix with the same dimensions.
            int[,] highestPeaks = new int[rowCount, columnCount];

            // Queue for BFS (Breadth-first search).
            Queue<int[]> queue = new Queue<int[]>();

            // Initialize answer matrix and enqueue all water cells.
            for (int rowIndex = 0; rowIndex < rowCount; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < columnCount; ++columnIndex)
                {
                    // Mark water cells with 0 and land cells with -1
                    highestPeaks[rowIndex, columnIndex] = isWater[rowIndex, columnIndex] - 1;

                    // Add water cell coordinates to the queue.
                    if (highestPeaks[rowIndex, columnIndex] == 0)
                    {
                        queue.Enqueue(new int[] { rowIndex, columnIndex });
                    }
                }
            }

            // Directions for exploring adjacent cells (up, right, down, left).
            int[] directions = { -1, 0, 1, 0, -1 };

            // Perform BFS to find the highest peak values.
            while (queue.Count > 0)
            {
                // Dequeue a cell from the queue.
                int[] position = queue.Dequeue();
                int currentRow = position[0];
                int currentColumn = position[1];

                // Explore all adjacent cells.
                for (int directionIndex = 0; directionIndex < 4; ++directionIndex)
                {
                    // Calculate coordinates of the adjacent cell.
                    int adjacentRow = currentRow + directions[directionIndex];
                    int adjacentColumn = currentColumn + directions[directionIndex + 1];

                    // Check if the adjacent cell is within bounds and if it is land.
                    if (adjacentRow >= 0 && adjacentRow < rowCount && adjacentColumn >= 0 && adjacentColumn < columnCount && highestPeaks[adjacentRow, adjacentColumn] == -1)
                    {
                        // Set the height of the land cell to be one more than the current cell.
                        highestPeaks[adjacentRow, adjacentColumn] = highestPeaks[currentRow, currentColumn] + 1;

                        // Enqueue the position of the land cell.
                        queue.Enqueue(new int[] { adjacentRow, adjacentColumn });
                    }
                }
            }

            // Return the filled highestPeaks matrix.
            return highestPeaks;
        }
    }

    /* 1860. Incremental Memory Leak
    https://leetcode.com/problems/incremental-memory-leak/description/
    https://algo.monster/liteproblems/1860
     */
    class IncrementalMemLeakSol
    {
        /* Time and Space Complexity
The time complexity of the given code depends on how quickly the while loop reaches a point where memory1 and memory2 are both less than i. Since i starts at 1 and increments by 1 on each iteration, and the maximum value of i that doesn't crash (exceeds the remaining memory) is at most max(memory1, memory2), in the worst-case scenario, the loop can execute O(sqrt(max(memory1, memory2))) times. This is because the sum of the first n natural numbers is given by the formula n*(n+1)/2, and we're looking for the point where this sum exceeds memory1 or memory2.
The space complexity of the method is O(1), which is constant, because the amount of extra memory used by the algorithm does not depend on the input size and is limited to a fixed number of integer variables (i, memory1, and memory2).
 */
        public int[] MemLeak(int memory1, int memory2)
        {
            int second = 1; // Initialize a time counter starting at 1

            // The loop runs as long as either memory1 or memory2 is enough for the current time counter.
            // The condition inside the loop checks which memory to reduce based on which one is larger.
            while (second <= Math.Max(memory1, memory2))
            {
                if (memory1 >= memory2)
                {
                    // if memory1 is larger or equal to memory2
                    // memory1 is reduced by the current value of the time counter.
                    memory1 -= second;
                }
                else
                {
                    // if memory2 is larger than memory1
                    // memory2 is reduced by the current value of the time counter.
                    memory2 -= second;
                }
                second++; // Increment the time counter after each iteration
            }

            // Return result as an array containing the value of the time counter
            // when the loop stops, and the remaining memory in both memory slots.
            return new int[] { second, memory1, memory2 };
        }
    }

    /* 1989. Maximum Number of People That Can Be Caught in Tag
    https://leetcode.com/problems/maximum-number-of-people-that-can-be-caught-in-tag/description/
    https://algo.monster/liteproblems/1989
     */
    class CatchMaximumAmountOfPeopleSol
    {
        /* Time and Space Complexity
The given Python code is designed to count the maximum number of people a team can catch within a certain distance dist. The algorithm uses a greedy two-pointer approach.
Time Complexity
The time complexity of the code is O(n), where n is the length of the team list. This is because:
•	The for-loop iterates through the list once, which contributes O(n).
•	Inside the loop, the while-loop moves the second pointer j forward until it finds a valid person to catch or reaches the end of the list. Each element is visited by the j pointer at most once throughout the entire execution of the algorithm. Therefore, the total number of operations contributed by the inner while-loop across all iterations of the for-loop is also O(n).
Hence, the combined time complexity remains O(n) since both the outer for-loop and the cumulative operations of the inner while-loop are linear with respect to the size of the input list.
Space Complexity
The space complexity of the code is O(1).
This is because the algorithm uses a constant amount of extra space for variables ans, j, n, and i. The space used does not depend on the input size and remains constant even as the input list team grows in length.
 */
        public int CatchMaximumAmountOfPeople(int[] team, int dist)
        {
            int maxCatches = 0; // This variable stores the maximum number of people that can be caught
            int teamLength = team.Length; // The length of the team array

            // Two pointers, i for catcher's position and j for nearest catchable runner
            for (int i = 0, j = 0; i < teamLength; ++i)
            {
                // Check if current position has a catcher
                if (team[i] == 1)
                {
                    // Move j to the next catchable runner within the distance
                    while (j < teamLength && (team[j] == 1 || i - j > dist))
                    {
                        ++j;
                    }
                    // If j is a valid runner, increment the catch count and move j to next position
                    if (j < teamLength && Math.Abs(i - j) <= dist)
                    {
                        maxCatches++; // Increase number of people caught
                        j++; // Move j to the next position
                    }
                }
            }
            // Return the computed maximum number of catches
            return maxCatches;
        }
    }






}
