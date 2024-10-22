namespace AlgoDSPlay;

public class SubsetProbs
{

    /*     2597. The Number of Beautiful Subsets
    https://leetcode.com/problems/the-number-of-beautiful-subsets/description/
     */
    class NumberOfBeautifulSubsetsSol
    {

        /* Approach 1: Using Bitset
        Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(n⋅2^n)
Each number in the input array nums can be either included or excluded in a subset, resulting in 2^n possible subsets.
Work done within each recursive call: The function iterates over the previous elements in the current subset to check if any pair satisfies the difference constraint. In the worst case, when all elements are included in the subset, the iteration takes O(n) time.
Combining the number of recursive calls and the work done within each call, the overall time complexity will be O(n⋅2^n).
•	Space complexity: O(n)
The space complexity is dominated by the recursive call stack, which can grow up to the depth of the input array nums. Hence, the space complexity is O(n).

         */

        public int UsingBitset(int[] nums, int k)
        {
            return CountBeautifulSubsets(nums, k, 0, 0);
        }

        private int CountBeautifulSubsets(
            int[] nums,
            int difference,
            int index,
            int mask
        )
        {
            // Base case: Return 1 if mask is greater than 0 (non-empty subset)
            if (index == nums.Length) return mask > 0 ? 1 : 0;

            // Flag to check if the current subset is beautiful
            bool isBeautiful = true;

            // Check if the current number forms a beautiful pair with any previous number
            // in the subset
            for (int j = 0; j < index && isBeautiful; ++j)
            {
                isBeautiful = ((1 << j) & mask) == 0 ||
                Math.Abs(nums[j] - nums[index]) != difference;
            }

            // Recursively calculate beautiful subsets including and excluding the current
            // number
            int skip = CountBeautifulSubsets(nums, difference, index + 1, mask);
            int take;
            if (isBeautiful)
            {
                take = CountBeautifulSubsets(
                    nums,
                    difference,
                    index + 1,
                    mask + (1 << index)
                );
            }
            else
            {
                take = 0;
            }

            return skip + take;
        }
        /* Approach 2: Recursion with Backtracking
Complexity Analysis
Let n be the size of nums array.
•	Time complexity: O(2^n)
The time complexity of the solution is primarily determined by the number of subsets generated. Since the algorithm explores all possible subsets of the input array, the maximum number of subsets that can be generated from an array of size n is 2^n
Additionally, sorting nums takes O(nlogn) time.
Therefore, the overall time complexity is O(2^n), because it is dominated by the subset generation.
•	Space complexity: O(n)
Note that some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Tim Sort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space. Additionally, Tim Sort is designed to be a stable algorithm.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting an array.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
The recursion stack space and the frequency map each use O(n) space. Thus, the total space complexity is O(n).

         */
        public int UsingRecursionWithBacktracking(int[] nums, int k)
        {
            // Frequency map to track elements
            Dictionary<int, int> freqMap = new();
            // Sort nums array
            Array.Sort(nums);
            return CountBeautifulSubsets(nums, k, freqMap, 0) - 1;
        }

        private int CountBeautifulSubsets(
            int[] nums,
            int difference,
            Dictionary<int, int> freqMap,
            int i
        )
        {
            // Base case: Return 1 for a subset of size 1
            if (i == nums.Length)
            {
                return 1;
            }
            // Count subsets where nums[i] is not taken
            int totalCount = CountBeautifulSubsets(
                nums,
                difference,
                freqMap,
                i + 1
            );

            // If nums[i] can be taken without violating the condition
            if (!freqMap.ContainsKey(nums[i] - difference))
            {
                freqMap[nums[i]] = freqMap.GetValueOrDefault(nums[i], 0) + 1; // Mark nums[i] as taken

                // Recursively count subsets where nums[i] is taken
                totalCount +=
                CountBeautifulSubsets(nums, difference, freqMap, i + 1);
                freqMap[nums[i]] = freqMap[nums[i]] - 1; // Backtrack: mark nums[i] as not taken

                // Remove nums[i] from freqMap if its count becomes 0
                if (freqMap[nums[i]] == 0)
                {
                    freqMap.Remove(nums[i]);
                }
            }

            return totalCount;
        }
        /* Approach 3: Optimised Recursion (Deriving Recurrence Relation)
        Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(nlogn+2^n)=O(2^n)
Since the map is sorted and implemented using a Self-Balancing Binary Search Tree (BST), the insert operation is O(logn). Thus, constructing the map takes O(nlogn). With a maximum of k different remainders, there can be up to k subset splits. In the worst-case scenario, where all numbers have the same remainder, and none are repeated (frequency = 1), this approach still results in a time complexity of O(2^n).
In Python3 we use a defaultdict. Inserting a key-value pair into a dictionary takes O(1) on average, resulting in a construction time of O(n). Still, the overall time complexity remains O(2^n).
•	Space complexity: O(n)
The frequency map stores the count of elements based on their remainders when divided by k. In the worst case, this requires O(n) space, as it needs to store counts for each element.
The depth of the recursive call stack can grow up to the number of unique elements in the subset list, which is at most n. Thus, the space used by the call stack is O(n).
For the counts array, which is used for memoization, its size is equal to the number of unique elements in each subset list, which again can be up to n. This results in O(n) space complexity for the counts array.
The subsets list, derived from the frequency map, stores pairs of element values and their counts. In the worst case, there could be n such pairs, resulting in a space complexity of O(n).
So, overall, the space complexity is O(n).

         */
        public int UsingRecursionOptimal(int[] numbers, int difference)
        {
            int totalCount = 1;
            SortedDictionary<int, SortedDictionary<int, int>> frequencyMap = new SortedDictionary<int, SortedDictionary<int, int>>();

            // Calculate frequencies based on remainder
            foreach (int number in numbers)
            {
                SortedDictionary<int, int> frequency = frequencyMap.GetValueOrDefault(
                    number % difference,
                    new SortedDictionary<int, int>()
                );
                frequency[number] = frequency.GetValueOrDefault(number, 0) + 1;
                frequencyMap[number % difference] = frequency;
            }

            // Calculate subsets for each remainder group
            foreach (var entry in frequencyMap)
            {
                List<KeyValuePair<int, int>> subsets = new List<KeyValuePair<int, int>>(entry.Value);
                totalCount *= CountBeautifulSubsets(subsets, subsets.Count, difference, 0);
            }

            return totalCount - 1;
        }

        private int CountBeautifulSubsets(
            List<KeyValuePair<int, int>> subsets,
            int numberOfSubsets,
            int difference,
            int index
        )
        {
            // Base case: Return 1 for a subset of size 1
            if (index == numberOfSubsets)
            {
                return 1;
            }

            // Calculate subsets where the current subset is not taken
            int skip = CountBeautifulSubsets(
                subsets,
                numberOfSubsets,
                difference,
                index + 1
            );
            // Calculate subsets where the current subset is taken
            int take = (1 << subsets[index].Value) - 1;

            // If next number has a 'difference', calculate subsets; otherwise, move
            // to next
            if (
                index + 1 < numberOfSubsets &&
                subsets[index + 1].Key - subsets[index].Key == difference
            )
            {
                take *=
                CountBeautifulSubsets(subsets, numberOfSubsets, difference, index + 2);
            }
            else
            {
                take *=
                CountBeautifulSubsets(subsets, numberOfSubsets, difference, index + 1);
            }

            return skip + take; // Return total count of subsets
        }
        /*Approach 4: Dynamic Programming - Memoization
        Complexity Analysis
    Let n be the size of the nums array.
    •	Time complexity: O(nlogn+2^n)=O(2^n)
    Since the map is sorted and implemented using a Self-Balancing Binary Search Tree (BST), the insert operation is O(logn). Thus, constructing the map takes O(nlogn). With a maximum of k different remainders, there can be up to k subset splits. In the worst-case scenario, where all numbers have the same remainder, and none are repeated (frequency = 1), this approach still results in a time complexity of O(2^n).
    In Python3 we use a defaultdict. Inserting a key-value pair into a dictionary takes O(1) on average, resulting in a construction time of O(n). Still, the overall time complexity remains O(2^n).
    •	Space complexity: O(n)
    The frequency map stores the count of elements based on their remainders when divided by k. In the worst case, this requires O(n) space, as it needs to store counts for each element.
    The depth of the recursive call stack can grow up to the number of unique elements in the subset list, which is at most n. Thus, the space used by the call stack is O(n).
    For the counts array, which is used for memoization, its size is equal to the number of unique elements in each subset list, which again can be up to n. This results in O(n) space complexity for the counts array.
    The subsets list, derived from the frequency map, stores pairs of element values and their counts. In the worst case, there could be n such pairs, resulting in a space complexity of O(n).
    So, overall, the space complexity is O(n).

        */
        public int UsingDPMemoRec(int[] numbers, int k)
        {
            int totalCount = 1;
            SortedDictionary<int, SortedDictionary<int, int>> frequencyMap = new SortedDictionary<int, SortedDictionary<int, int>>();

            // Calculate frequencies based on remainder
            foreach (int number in numbers)
            {
                int remainder = number % k;
                SortedDictionary<int, int> frequency = frequencyMap.ContainsKey(remainder) ? frequencyMap[remainder] : new SortedDictionary<int, int>();
                frequency[number] = frequency.ContainsKey(number) ? frequency[number] + 1 : 1;
                frequencyMap[remainder] = frequency;
            }

            // Calculate subsets for each remainder group
            foreach (var entry in frequencyMap)
            {
                List<KeyValuePair<int, int>> subsets = new List<KeyValuePair<int, int>>();
                foreach (var subset in entry.Value)
                {
                    subsets.Add(new KeyValuePair<int, int>(subset.Key, subset.Value));
                }
                int[] counts = new int[subsets.Count]; // Store counts of subsets for memoization
                Array.Fill(counts, -1);
                totalCount *= CountBeautifulSubsets(subsets, subsets.Count, k, 0, counts);
            }
            return totalCount - 1;
        }

        private int CountBeautifulSubsets(
            List<KeyValuePair<int, int>> subsets,
            int numSubsets,
            int difference,
            int index,
            int[] counts
        )
        {
            // Base case: Return 1 for a subset of size 1
            if (index == numSubsets)
            {
                return 1;
            }

            // If the count is already calculated, return it
            if (counts[index] != -1)
            {
                return counts[index];
            }

            // Calculate subsets where the current subset is not taken
            int skip = CountBeautifulSubsets(
                subsets,
                numSubsets,
                difference,
                index + 1,
                counts
            );

            // Calculate subsets where the current subset is taken
            int take = (1 << subsets[index].Value) - 1; // take the current subset

            // If the next number has a difference of 'difference',
            // calculate subsets accordingly
            if (
                index + 1 < numSubsets &&
                subsets[index + 1].Key - subsets[index].Key == difference
            )
            {
                take *=
                CountBeautifulSubsets(
                    subsets,
                    numSubsets,
                    difference,
                    index + 2,
                    counts
                );
            }
            else
            {
                take *=
                CountBeautifulSubsets(
                    subsets,
                    numSubsets,
                    difference,
                    index + 1,
                    counts
                );
            }

            return counts[index] = skip + take; // Store and return total count of subsets
        }

        /* Approach 5: Dynamic Programming - Iterative 
Complexity Analysis
Let n be the size of the nums arrays.
•	Time complexity: O(nlogn)
Since the map is sorted and implemented using a Self-Balancing Binary Search Tree (BST), the insert operation is O(logn). Thus, constructing the map takes O(nlogn).
Then, iterating through each remainder group and its associated numbers involves nested loops. In the worst-case scenario, each remainder group contains n/k elements. The time complexity of iterating through each remainder group is O(k⋅(n/k)log(n/k)). The number of groups is limited to n, and so is the group size. Therefore, we can we can simplify this to O(nlogn).
•	Space complexity: O(n)
The frequency map stores a remainder group for each unique remainder. Each remainder group stores an entry for each unique element in the group. In the worst case, when each element in nums is unique, n elements will be stored across all of the remainder groups.
For the counts array, which is used for memoization, its size is equal to the number of unique elements in each subset list, which again can be up to n. This results in O(n) space complexity for the counts array.
The subsets list, derived from the frequency map, stores pairs of element values and their counts. In the worst case, there could be n such pairs, resulting in a space complexity of O(n).
Therefore, the total space complexity is O(n).

        */
        public int UsingDPIterative(int[] numbers, int k)
        {
            int totalCount = 1;

            SortedDictionary<int, SortedDictionary<int, int>> frequencyMap = new SortedDictionary<int, SortedDictionary<int, int>>();

            // Calculate frequencies based on remainder
            foreach (int number in numbers)
            {
                int remainder = number % k;
                if (!frequencyMap.ContainsKey(remainder))
                {
                    frequencyMap[remainder] = new SortedDictionary<int, int>();
                }
                if (!frequencyMap[remainder].ContainsKey(number))
                {
                    frequencyMap[remainder][number] = 0;
                }
                frequencyMap[remainder][number]++;
            }

            // Iterate through each remainder group
            foreach (var entry in frequencyMap)
            {
                int n = entry.Value.Count; // Number of elements in the current group

                List<KeyValuePair<int, int>> subsets = new List<KeyValuePair<int, int>>(entry.Value);

                int[] counts = new int[n + 1]; // Array to store counts of subsets

                counts[n] = 1; // Initialize count for the last subset

                // Calculate counts for each subset starting from the second last
                for (int i = n - 1; i >= 0; i--)
                {
                    // Count of subsets skipping the current subset
                    int skip = counts[i + 1];

                    // Count of subsets including the current subset
                    int take = (1 << subsets[i].Value) - 1;

                    // If next number has a 'difference',
                    // calculate subsets; otherwise, move to next
                    if (i + 1 < n && subsets[i + 1].Key - subsets[i].Key == k)
                    {
                        take *= counts[i + 2];
                    }
                    else
                    {
                        take *= counts[i + 1];
                    }

                    // Store the total count for the current subset
                    counts[i] = skip + take;
                }

                totalCount *= counts[0];
            }

            return totalCount - 1;
        }

        /* Approach 6: Dynamic Programming - Optimized Iterative
        Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(nlogn)
The time complexity of this approach primarily arises from the operations on the map data structure. Since up to n values are added to the frequency map, the sorting operation on the frequency map takes O(nlogn) time.
Then, iterating through each remainder group and its associated numbers involves nested loops. In the worst-case scenario, each remainder group contains n/k elements, where n/k is a positive integer. The time complexity of iterating through each remainder group is O(k⋅(n/k)log(n/k)), which we can simplify to O(nlogn).
The (logn) term arises from the usage of the map data structure in the code. map/TreeMap is implemented as a self-balancing binary search tree (such as Red-Black Tree) in C++/Java, which provides logarithmic time complexity for operations such as insertion, deletion, and retrieval.
•	Space complexity: O(n)
The frequency map stores a remainder group for each unique remainder. Each remainder group stores an entry for each unique element in the group. In the worst case, when each element in nums is unique, n elements will be stored across all of the remainder groups. Therefore, the total space complexity is O(n).

         */
        public int UsingDPIterativeOptimal(int[] numbers, int k)
        {
            int totalCount = 1;
            SortedDictionary<int, SortedDictionary<int, int>> frequencyMap = new SortedDictionary<int, SortedDictionary<int, int>>();

            // Calculate frequencies based on remainder
            foreach (int number in numbers)
            {
                SortedDictionary<int, int> frequency = frequencyMap.GetValueOrDefault(number % k, new SortedDictionary<int, int>());
                frequency[number] = frequency.GetValueOrDefault(number, 0) + 1;
                frequencyMap[number % k] = frequency;
            }

            // Iterate through each remainder group
            foreach (KeyValuePair<int, SortedDictionary<int, int>> entry in frequencyMap)
            {
                int previousNumber = -k, previous2 = 0, previous1 = 1, current = 1;

                // Iterate through each number in the current remainder group
                foreach (KeyValuePair<int, int> frequencyEntry in entry.Value)
                {
                    int number = frequencyEntry.Key;
                    int frequency = frequencyEntry.Value;

                    // Count of subsets skipping the current number
                    int skip = previous1;

                    // Count of subsets including the current number
                    // Check if the current number and the previous number
                    // form a beautiful pair
                    int take;
                    if (number - previousNumber == k)
                    {
                        take = ((1 << frequency) - 1) * previous2;
                    }
                    else
                    {
                        take = ((1 << frequency) - 1) * previous1;
                    }

                    current = skip + take; // Store the total count for the current number
                    previous2 = previous1;
                    previous1 = current;
                    previousNumber = number;
                }
                totalCount *= current;
            }
            return totalCount - 1;
        }
    }
    /* 916. Word Subsets
    https://leetcode.com/problems/word-subsets/description/
     */
    class WordSubsetsSol
    {
        /* 
        Approach 1: Reduce to Single Word in B
        Complexity Analysis
        •	Time Complexity: O(A+B), where A and B is the total amount of information in A and B respectively.
        •	Space Complexity: O(A.length+B.length).
         */
        public IList<string> WordSubsets(string[] A, string[] B)
        {
            int[] bmax = Count("");
            foreach (string b in B)
            {
                int[] bCount = Count(b);
                for (int i = 0; i < 26; ++i)
                    bmax[i] = Math.Max(bmax[i], bCount[i]);
            }

            List<string> ans = new List<string>();
            foreach (string a in A)
            {
                int[] aCount = Count(a);
                bool isSubset = true;
                for (int i = 0; i < 26; ++i)
                {
                    if (aCount[i] < bmax[i])
                    {
                        isSubset = false;
                        break;
                    }
                }
                if (isSubset)
                {
                    ans.Add(a);
                }
            }

            return ans;
        }

        public int[] Count(string S)
        {
            int[] ans = new int[26];
            foreach (char c in S)
            {
                ans[c - 'a']++;
            }
            return ans;
        }
    }












}
