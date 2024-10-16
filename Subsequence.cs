using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class Subsequence
    {
        //https://www.algoexpert.io/questions/validate-subsequence
        public static bool IsValidSubsequence(List<int> array, List<int> sequence)
        {
            //T:O(n): S:O(1)
            int arrIdx = 0, seqIdx = 0;
            while (arrIdx < array.Count && seqIdx < sequence.Count)
            {
                if (array[arrIdx] == sequence[seqIdx])
                    seqIdx++;

                arrIdx++;


            }
            return seqIdx == sequence.Count;
        }

        //https://www.algoexpert.io/questions/kadane's-algorithm
        public static int FindSubArryMaxSumKadane(int[] array)
        {
            int maxEndingHere = array[0];
            int maxSoFar = array[0];
            for (int i = 1; i < array.Length; i++)
            {

                maxEndingHere = Math.Max(maxEndingHere + array[i], array[i]);
                maxSoFar = Math.Max(maxSoFar, maxEndingHere);
            }
            return maxSoFar;
        }
        //https://www.algoexpert.io/questions/zero-sum-subarray
        public static bool ZeroSumSubArray(int[] nums)
        {

            //1.Naive - pair of loops to create subarrays to check its sum to zero
            //T:O(n^2) |S:O(1)

            //2. By logic that if current sum repeated a previous one then some subarrys sum is zero
            //[0,x]=>s [o,..,x,.., y,..]=>s then [x+1, y] must zero sum sub array
            //T:O(n^2) |S:O(n)
            HashSet<int> sums = new HashSet<int>();
            sums.Add(0);
            int currentSum = 0;
            foreach (int num in nums)
            {
                currentSum += num;
                if (sums.Contains(currentSum)) return true;

                sums.Add(currentSum);
            }
            return false;
        }
        //https://www.algoexpert.io/questions/longest-subarray-with-sum
        public static int[] LongestSubarrayWithSum(int[] arr, int targetSum)
        {

            //1.Naive - Pair of loops
            //T:O(n^2) | S:O(1)
            int[] indices = LongestSubarrayWithSumNaive(arr, targetSum);

            //2.Optimal - keeping track of previous sum instead looping again n again
            //T:O(n) | S:O(1)
            indices = LongestSubarrayWithSumOptimal(arr, targetSum);
            return indices;
        }

        private static int[] LongestSubarrayWithSumOptimal(int[] arr, int targetSum)
        {
            int[] indices = new int[] { };
            int currentSubArraySum = 0;
            int startIdx = 0, endIdx = 0;

            while (endIdx < arr.Length)
            {

                currentSubArraySum += arr[endIdx];
                while (currentSubArraySum > targetSum && startIdx < endIdx)
                {
                    currentSubArraySum -= arr[startIdx];
                    startIdx++;
                }

                if (currentSubArraySum == targetSum)
                {
                    if (indices.Length == 0 || indices[1] - indices[0] < endIdx - startIdx)
                    {
                        indices = new int[] { startIdx, endIdx };
                    }
                }
                endIdx++;
            }
            return indices;


        }

        private static int[] LongestSubarrayWithSumNaive(int[] arr, int targetSum)
        {
            int[] indices = new int[] { };
            for (int startIdx = 0; startIdx < arr.Length; startIdx++)
            {
                int currentSubArraySum = 0;

                for (int endIdx = startIdx; endIdx < arr.Length; endIdx++)
                {
                    currentSubArraySum += arr[endIdx];

                    if (currentSubArraySum == targetSum)
                    {

                        if (indices.Length == 0 || indices[1] - indices[0] < endIdx - startIdx)
                        {
                            indices = new int[] { startIdx, endIdx };
                        }
                    }
                }

            }
            return indices;
        }        //https://www.algoexpert.io/questions/max-subset-sum-no-adjacent
        public static int MaxSubsetSumNoAdjacent(int[] array)
        {
            // T:O(n) time | S:O(n) space
            int result = MaxSubsetSumNoAdjacentNonOptimal(array);
            // T:O(n) time | S:O(1) space

            result = MaxSubsetSumNoAdjacentOptimal(array);
            return result;
        }
        public static int MaxSubsetSumNoAdjacentOptimal(int[] array)
        {
            if (array.Length == 0)
            {
                return 0;
            }
            else if (array.Length == 1)
            {
                return array[0];
            }
            int second = array[0];
            int first = Math.Max(array[0], array[1]);
            for (int i = 2; i < array.Length; i++)
            {
                int current = Math.Max(first, second + array[i]);
                second = first;
                first = current;
            }
            return first;
        }
        public static int MaxSubsetSumNoAdjacentNonOptimal(int[] array)
        {
            // T:O(n) time | S:O(n) space
            if (array.Length == 0)
            {
                return 0;
            }
            else if (array.Length == 1)
            {
                return array[0];
            }
            int[] maxSums = (int[])array.Clone();
            maxSums[1] = Math.Max(array[0], array[1]);
            for (int i = 2; i < array.Length; i++)
            {
                maxSums[i] = Math.Max(maxSums[i - 1], maxSums[i - 2] + array[i]);
            }
            return maxSums[array.Length - 1];
        }
        //https://www.algoexpert.io/questions/four-number-sum

        public static List<int[]> FourNumberSum(int[] array, int targetSum)
        {
            // Average: O(n^2) time | O(n^2) space
            // Worst: O(n^3) time | O(n^2) space
            Dictionary<int, List<int[]>> allPairSums =
              new Dictionary<int, List<int[]>>();
            List<int[]> quadruplets = new List<int[]>();
            for (int i = 1; i < array.Length - 1; i++)
            {
                for (int j = i + 1; j < array.Length; j++)
                {
                    int currentSum = array[i] + array[j];
                    int difference = targetSum - currentSum;
                    if (allPairSums.ContainsKey(difference))
                    {
                        foreach (int[] pair in allPairSums[difference])
                        {
                            int[] newQuadruplet = { pair[0], pair[1], array[i], array[j] };
                            quadruplets.Add(newQuadruplet);
                        }
                    }
                }
                for (int k = 0; k < i; k++)
                {
                    int currentSum = array[i] + array[k];
                    int[] pair = { array[k], array[i] };
                    if (!allPairSums.ContainsKey(currentSum))
                    {
                        List<int[]> pairGroup = new List<int[]>();
                        pairGroup.Add(pair);
                        allPairSums.Add(currentSum, pairGroup);
                    }
                    else
                    {
                        allPairSums[currentSum].Add(pair);
                    }
                }
            }
            return quadruplets;
        }
        //https://www.algoexpert.io/questions/longest-peak
        public static int LongestPeak(int[] array)
        {
            // O(n) time | O(1) space - where n is the length of the input array
            int longestPeakLength = 0;
            int i = 1;
            while (i < array.Length - 1)
            {
                bool isPeak = array[i - 1] < array[i] && array[i] > array[i + 1];
                if (!isPeak)
                {
                    i += 1;
                    continue;
                }

                int leftIdx = i - 2;
                while (leftIdx >= 0 && array[leftIdx] < array[leftIdx + 1])
                {
                    leftIdx -= 1;
                }

                int rightIdx = i + 2;
                while (rightIdx < array.Length && array[rightIdx] < array[rightIdx - 1])
                {
                    rightIdx += 1;
                }
                int currentPeakLength = rightIdx - leftIdx - 1;
                if (currentPeakLength > longestPeakLength)
                {
                    longestPeakLength = currentPeakLength;
                }
                i = rightIdx;
            }
            return longestPeakLength;
        }
        //https://www.algoexpert.io/questions/longest-increasing-subsequence
        // O(n^2) time | O(n) space
        public static List<int> LongestIncreasingSubsequenceNaive(int[] array)
        {
            int[] sequences = new int[array.Length];
            Array.Fill(sequences, Int32.MinValue);
            int[] lengths = new int[array.Length];
            Array.Fill(lengths, 1);
            int maxLengthIdx = 0;
            for (int i = 0; i < array.Length; i++)
            {
                int currentNum = array[i];
                for (int j = 0; j < i; j++)
                {
                    int otherNum = array[j];
                    if (otherNum < currentNum && lengths[j] + 1 >= lengths[i])
                    {
                        lengths[i] = lengths[j] + 1;
                        sequences[i] = j;
                    }
                }
                if (lengths[i] >= lengths[maxLengthIdx])
                {
                    maxLengthIdx = i;
                }
            }
            return buildSequence(array, sequences, maxLengthIdx);
        }

        public static List<int> buildSequence(
          int[] array, int[] sequences, int currentIdx
        )
        {
            List<int> sequence = new List<int>();
            while (currentIdx != Int32.MinValue)
            {
                sequence.Insert(0, array[currentIdx]);
                currentIdx = sequences[currentIdx];
            }
            return sequence;
        }

        // O(nlogn) time | O(n) space
        public static List<int> LongestIncreasingSubsequenceOptimal(int[] array)
        {
            int[] sequences = new int[array.Length];
            int[] indices = new int[array.Length + 1];
            Array.Fill(indices, Int32.MinValue);
            int length = 0;
            for (int i = 0; i < array.Length; i++)
            {
                int num = array[i];
                int newLength = BinarySearch(1, length, indices, array, num);
                sequences[i] = indices[newLength - 1];
                indices[newLength] = i;
                length = Math.Max(length, newLength);
            }
            return buildSequence(array, sequences, indices[length]);
        }

        public static int BinarySearch(
          int startIdx, int endIdx, int[] indices, int[] array, int num
        )
        {
            if (startIdx > endIdx)
            {
                return startIdx;
            }
            int middleIdx = (startIdx + endIdx) / 2;
            if (array[indices[middleIdx]] < num)
            {
                startIdx = middleIdx + 1;
            }
            else
            {
                endIdx = middleIdx - 1;
            }
            return BinarySearch(startIdx, endIdx, indices, array, num);
        }

        //https://www.algoexpert.io/questions/max-sum-increasing-subsequence
        // O(n^2) time | O(n) space
        public static List<List<int>> MaxSumIncreasingSubsequence(int[] array)
        {
            int[] sequences = new int[array.Length];
            Array.Fill(sequences, Int32.MinValue);
            int[] sums = (int[])array.Clone();
            int maxSumIdx = 0;
            for (int i = 0; i < array.Length; i++)
            {
                int currentNum = array[i];
                for (int j = 0; j < i; j++)
                {
                    int otherNum = array[j];
                    if (otherNum < currentNum && sums[j] + currentNum >= sums[i])
                    {
                        sums[i] = sums[j] + currentNum;
                        sequences[i] = j;
                    }
                }
                if (sums[i] >= sums[maxSumIdx])
                {
                    maxSumIdx = i;
                }
            }
            return buildSequence(array, sequences, maxSumIdx, sums[maxSumIdx]);
        }

        public static List<List<int>> buildSequence(
          int[] array, int[] sequences, int currentIdx, int sums
        )
        {
            List<List<int>> sequence = new List<List<int>>();
            sequence.Add(new List<int>());
            sequence.Add(new List<int>());
            sequence[0].Add(sums);
            while (currentIdx != Int32.MinValue)
            {
                sequence[1].Insert(0, array[currentIdx]);
                currentIdx = sequences[currentIdx];
            }
            return sequence;
        }

        //https://www.algoexpert.io/questions/longest-common-subsequence

        //1. O(nm*min(n, m)) time | O(nm*min(n, m)) space
        public static List<char> LongestCommonSubsequenceNaive(string str1, string str2)
        {
            List<List<List<char>>> lcs = new List<List<List<char>>>();
            for (int i = 0; i < str2.Length + 1; i++)
            {
                lcs.Add(new List<List<char>>());
                for (int j = 0; j < str1.Length + 1; j++)
                {
                    lcs[i].Add(new List<char>());
                }
            }
            for (int i = 1; i < str2.Length + 1; i++)
            {
                for (int j = 1; j < str1.Length + 1; j++)
                {
                    if (str2[i - 1] == str1[j - 1])
                    {
                        List<char> copy = new List<char>(lcs[i - 1][j - 1]);
                        lcs[i][j] = copy;
                        lcs[i][j].Add(str2[i - 1]);
                    }
                    else
                    {
                        if (lcs[i - 1][j].Count > lcs[i][j - 1].Count)
                        {
                            lcs[i][j] = lcs[i - 1][j];
                        }
                        else
                        {
                            lcs[i][j] = lcs[i][j - 1];
                        }
                    }
                }
            }
            return lcs[str2.Length][str1.Length];
        }
        //2. O(nm*min(n, m)) time | O((min(n, m))^2) space
        public static List<char> LongestCommonSubsequenceNaive2(string str1, string str2)
        {
            string small = str1.Length < str2.Length ? str1 : str2;
            string big = str1.Length >= str2.Length ? str1 : str2;
            List<List<char>> evenLcs = new List<List<char>>();
            List<List<char>> oddLcs = new List<List<char>>();
            for (int i = 0; i < small.Length + 1; i++)
            {
                evenLcs.Add(new List<char>());
            }
            for (int i = 0; i < small.Length + 1; i++)
            {
                oddLcs.Add(new List<char>());
            }
            for (int i = 1; i < big.Length + 1; i++)
            {
                List<List<char>> currentLcs;
                List<List<char>> previousLcs;
                if (i % 2 == 1)
                {
                    currentLcs = oddLcs;
                    previousLcs = evenLcs;
                }
                else
                {
                    currentLcs = evenLcs;
                    previousLcs = oddLcs;
                }
                for (int j = 1; j < small.Length + 1; j++)
                {
                    if (big[i - 1] == small[j - 1])
                    {
                        List<char> copy = new List<char>(previousLcs[j - 1]);
                        currentLcs[j] = copy;
                        currentLcs[j].Add(big[i - 1]);
                    }
                    else
                    {
                        if (previousLcs[j].Count > currentLcs[j - 1].Count)
                        {
                            currentLcs[j] = previousLcs[j];
                        }
                        else
                        {
                            currentLcs[j] = currentLcs[j - 1];
                        }
                    }
                }
            }
            return big.Length % 2 == 0 ? evenLcs[small.Length] : oddLcs[small.Length];
        }
        //3. O(nm) time | O(nm) space
        public static List<char> LongestCommonSubsequenceOptimal1(string str1, string str2)
        {
            int[,][] lcs = new int[str2.Length + 1, str1.Length + 1][];
            for (int i = 0; i < str2.Length + 1; i++)
            {
                for (int j = 0; j < str1.Length + 1; j++)
                {
                    lcs[i, j] = new int[] { 0, 0, 0, 0 };
                }
            }
            for (int i = 1; i < str2.Length + 1; i++)
            {
                for (int j = 1; j < str1.Length + 1; j++)
                {
                    if (str2[i - 1] == str1[j - 1])
                    {
                        int[] newEntry = {
            (int)str2[i - 1], lcs[i - 1, j - 1][1] + 1, i - 1, j - 1
          };
                        lcs[i, j] = newEntry;
                    }
                    else
                    {
                        if (lcs[i - 1, j][1] > lcs[i, j - 1][1])
                        {
                            int[] newEntry = { -1, lcs[i - 1, j][1], i - 1, j };
                            lcs[i, j] = newEntry;
                        }
                        else
                        {
                            int[] newEntry = { -1, lcs[i, j - 1][1], i, j - 1 };
                            lcs[i, j] = newEntry;
                        }
                    }
                }
            }
            return buildSequence(lcs);
        }

        public static List<char> buildSequence(int[,][] lcs)
        {
            List<char> sequence = new List<char>();
            int i = lcs.GetLength(0) - 1;
            int j = lcs.GetLength(1) - 1;
            while (i != 0 && j != 0)
            {
                int[] currentEntry = lcs[i, j];
                if (currentEntry[0] != -1)
                {
                    sequence.Insert(0, (char)currentEntry[0]);
                }
                i = currentEntry[2];
                j = currentEntry[3];
            }
            return sequence;
        }

        //4. O(nm) time | O(nm) space
        public static List<char> LongestCommonSubsequenceOptimal2(string str1, string str2)
        {
            int[,] lengths = new int[str2.Length + 1, str1.Length + 1];
            for (int i = 1; i < str2.Length + 1; i++)
            {
                for (int j = 1; j < str1.Length + 1; j++)
                {
                    if (str2[i - 1] == str1[j - 1])
                    {
                        lengths[i, j] = lengths[i - 1, j - 1] + 1;
                    }
                    else
                    {
                        lengths[i, j] = Math.Max(lengths[i - 1, j], lengths[i, j - 1]);
                    }
                }
            }
            return buildSequence1(lengths, str1);
        }

        public static List<char> buildSequence1(int[,] lengths, string str)
        {
            List<char> sequence = new List<char>();
            int i = lengths.GetLength(0) - 1;
            int j = lengths.GetLength(1) - 1;
            while (i != 0 && j != 0)
            {
                if (lengths[i, j] == lengths[i - 1, j])
                {
                    i--;
                }
                else if (lengths[i, j] == lengths[i, j - 1])
                {
                    j--;
                }
                else
                {
                    sequence.Insert(0, str[j - 1]);
                    i--;
                    j--;
                }
            }
            return sequence;
        }

        /*
        2407. Longest Increasing Subsequence II
        https://leetcode.com/problems/longest-increasing-subsequence-ii/description/
        
        Time : O(NlogN)
        Space: O(N)
        */
        public int LengthOfLIS(int[] nums, int k)
        {
            //2. Using Segment Trees as in range boound queries
            SegmentTree segmentTree = new SegmentTree((int)(1e5) + 1);
            foreach (int a in nums)
            {
                int left = Math.Max(0, a - k);
                int right = a - 1;
                int currMax = segmentTree.query(left, right);
                segmentTree.update(a, currMax + 1);
            }
            return segmentTree.max();
        }

        /*
        3. Longest Substring Without Repeating Characters
        https://leetcode.com/problems/longest-substring-without-repeating-characters/description/

        */
        public int LengthOfLongestSubstring(string s)
        {

            /*
            Approach 1: Brute Force
            Complexity Analysis
            •	Time complexity : O(n3).
                    To verify if characters within index range [i,j) are all unique, we need to scan all of them. Thus, it costs O(j−i) time.
                    For a given i, the sum of time costed by each j∈[i+1,n] is
                    ∑i+1nO(j−i)
                    Thus, the sum of all the time consumption is:
                    O(∑i=0n−1(∑j=i+1n(j−i)))=O(∑i=0n−12(1+n−i)(n−i))=O(n3)
            •	Space complexity : O(min(n,m)). We need O(k) space for checking a substring has no duplicate characters, where k is the size of the Set. The size of the Set is upper bounded by the size of the string n and the size of the charset/alphabet m.
        
            */
            int lengthOfLongestSubstring = LengthOfLongestSubstringNaive(s);

            /*
Approach 2: Sliding Window
Complexity Analysis
•	Time complexity : O(2n)=O(n). In the worst case each character will be visited twice by i and j.
•	Space complexity : O(min(m,n)). Same as the previous approach. We need O(k) space for the sliding window, where k is the size of the Set. The size of the Set is upper bounded by the size of the string n and the size of the charset/alphabet m.

*/
            lengthOfLongestSubstring = LengthOfLongestSubstringOptimal1(s);

            /*            
Approach 3: Sliding Window Optimized

Complexity Analysis
•	Time complexity : O(n). Index j will iterate n times.
•	Space complexity : O(min(m,n)). Same as the previous approach.        
*/
            lengthOfLongestSubstring = LengthOfLongestSubstringOptimal2(s);

            return lengthOfLongestSubstring;

        }

        private int LengthOfLongestSubstringOptimal2(string s)
        {
            Dictionary<char, int> map = new Dictionary<char, int>();
            int maxLen = 0;
            int left = 0;
            for (int right = 0; right < s.Length; right++)
            {
                if (map.ContainsKey(s[right]))
                {
                    left = Math.Max(map[s[right]], left);
                }

                maxLen = Math.Max(maxLen, right - left + 1);
                map[s[right]] = right + 1;
            }

            return maxLen;
        }


        private int LengthOfLongestSubstringOptimal1(string s)
        {
            Dictionary<char, int> chars = new Dictionary<char, int>();

            int left = 0;
            int right = 0;

            int res = 0;
            while (right < s.Length)
            {
                char r = s[right];
                if (!chars.ContainsKey(r))
                {
                    chars[r] = 0;
                }

                chars[r]++;

                while (chars[r] > 1)
                {
                    char l = s[left];
                    chars[l]--;
                    left++;
                }

                res = Math.Max(res, right - left + 1);

                right++;
            }

            return res;
        }

        private int LengthOfLongestSubstringNaive(string s)
        {
            int n = s.Length;

            int res = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    if (CheckRepetition(s, i, j))
                    {
                        res = Math.Max(res, j - i + 1);
                    }
                }
            }

            return res;
        }
        private bool CheckRepetition(string s, int start, int end)
        {
            HashSet<char> chars = new HashSet<char>();

            for (int i = start; i <= end; i++)
            {
                char c = s[i];
                if (chars.Contains(c))
                {
                    return false;
                }

                chars.Add(c);
            }

            return true;
        }


        /*
        128. Longest Consecutive Sequence
        https://leetcode.com/problems/longest-consecutive-sequence/description/

        */
        public class LongestConsecutiveSeqOfArraySol
        {
            /*
          Approach 1: Brute Force
  Complexity Analysis
•	Time complexity : O(n^3).
The outer loop runs exactly n times, and because currentNum
increments by 1 during each iteration of the while loop, it runs in
O(n) time. Then, on each iteration of the while loop, an O(n)
lookup in the array is performed. Therefore, this brute force algorithm
is really three nested O(n) loops, which compound multiplicatively to a
cubic runtime.
•	Space complexity : O(1).
The brute force algorithm only allocates a handful of integers, so it uses constant
additional space.

            */
            public int Naive(int[] nums)
            {
                int longestStreak = 0;
                for (int i = 0; i < nums.Length; i++)
                {
                    int currentNum = nums[i];
                    int currentStreak = 1;
                    while (ArrayContains(nums, currentNum + 1))
                    {
                        currentNum += 1;
                        currentStreak += 1;
                    }

                    longestStreak = Math.Max(longestStreak, currentStreak);
                }

                return longestStreak;

                bool ArrayContains(int[] arr, int num)
                {
                    for (int i = 0; i < arr.Length; i++)
                    {
                        if (arr[i] == num)
                        {
                            return true;
                        }
                    }

                    return false;
                }
            }
            /*
            Approach 2: Sorting
Complexity Analysis
•	Time complexity : O(nlogn)
The main for loop does constant work n times, so the algorithm's time
complexity is dominated by the invocation of sort, which will run in
O(nlogn) time for any sensible implementation.
•	Space complexity : O(logn) or O(n)
Note that some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Tim Sort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space. Additionally, Tim Sort is designed to be a stable algorithm.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting an array.

            */

            public int Sort(int[] nums)
            {
                if (nums.Length == 0)
                {
                    return 0;
                }

                Array.Sort(nums);
                int longestStreak = 1;
                int currentStreak = 1;
                for (int i = 1; i < nums.Length; i++)
                {
                    if (nums[i] != nums[i - 1])
                    {
                        if (nums[i] == nums[i - 1] + 1)
                        {
                            currentStreak += 1;
                        }
                        else
                        {
                            longestStreak = Math.Max(longestStreak, currentStreak);
                            currentStreak = 1;
                        }
                    }
                }

                return Math.Max(longestStreak, currentStreak);
            }

            /*
            Approach 3: HashSet and Intelligent Sequence Building
Complexity Analysis
•	Time complexity : O(n).
Although the time complexity appears to be quadratic due to the while
loop nested within the for loop, closer inspection reveals it to be
linear. Because the while loop is reached only when currentNum marks
the beginning of a sequence (i.e. currentNum-1 is not present in
nums), the while loop can only run for n iterations throughout the
entire runtime of the algorithm. This means that despite looking like
O(n⋅n) complexity, the nested loops actually run in O(n+n)=O(n)
time. All other computations occur in constant time, so the overall
runtime is linear.
•	Space complexity : O(n).
In order to set up O(1) containment lookups, we allocate linear space
for a hash table to store the O(n) numbers in nums. Other than that,
the space complexity is identical to that of the brute force solution.

            */
            public int HashSetWithIntelliSeqBuild(int[] nums)
            {
                HashSet<int> num_set = new HashSet<int>(nums);
                int longestStreak = 0;
                foreach (int num in num_set)
                {
                    if (!num_set.Contains(num - 1))
                    {
                        int currentNum = num;
                        int currentStreak = 1;
                        while (num_set.Contains(currentNum + 1))
                        {
                            currentNum += 1;
                            currentStreak += 1;
                        }

                        longestStreak = Math.Max(longestStreak, currentStreak);
                    }
                }

                return longestStreak;
            }
        }



        /* 3165. Maximum Sum of Subsequence With Non-adjacent Elements
        https://leetcode.com/problems/maximum-sum-of-subsequence-with-non-adjacent-elements/description/
         */

        public class MaximumSumSubsequenceWithNonAdjcentElemSol
        {
            private const long inf = 1L << 60;
            private const long mod = 1_000_000_007;

            /* 1. Segment Tree 
            Complexity
            •	Time complexity: O(n+qlogn)
            •	Space complexity: O(n+q)

            */
            public int UsingSegmentTree(List<int> nums, List<List<int>> queries)
            {
                SegmentTree segmentTree = new SegmentTree(nums, 0, (uint)nums.Count - 1);

                long res = 0;
                foreach (var query in queries)
                {
                    segmentTree.Update(query[0], query[1]);
                    res += Math.Max(segmentTree.selected[0, 0], Math.Max(segmentTree.selected[0, 1], Math.Max(segmentTree.selected[1, 0], segmentTree.selected[1, 1])));
                    res %= mod;
                }
                return (int)res;
            }
            private class SegmentTree
            {
                public SegmentTree? l, r;
                public uint lo, hi;
                public long[,] selected = new long[2, 2];

                public SegmentTree(List<int> nums, uint lo, uint hi)
                {
                    this.lo = lo;
                    this.hi = hi;
                    if (lo < hi)
                    {
                        uint mid = (lo + hi) / 2;
                        l = new SegmentTree(nums, lo, mid);
                        r = new SegmentTree(nums, (uint)mid + 1, hi);
                        Combine();
                    }
                    else
                    {
                        selected[0, 0] = 0;
                        selected[0, 1] = -inf;
                        selected[1, 0] = -inf;
                        selected[1, 1] = nums[(int)lo];
                    }
                }

                private void Combine()
                {
                    selected[0, 0] = Math.Max(l!.selected[0, 0] + r!.selected[0, 0], Math.Max(l.selected[0, 1] + r.selected[0, 0], l.selected[0, 0] + r.selected[1, 0]));
                    selected[0, 1] = Math.Max(l.selected[0, 0] + r.selected[0, 1], Math.Max(l.selected[0, 1] + r.selected[0, 1], l.selected[0, 0] + r.selected[1, 1]));
                    selected[1, 0] = Math.Max(l.selected[1, 0] + r.selected[0, 0], Math.Max(l.selected[1, 1] + r.selected[0, 0], l.selected[1, 0] + r.selected[1, 0]));
                    selected[1, 1] = Math.Max(l.selected[1, 0] + r.selected[0, 1], Math.Max(l.selected[1, 1] + r.selected[0, 1], l.selected[1, 0] + r.selected[1, 1]));
                }

                public void Update(int i, long x)
                {
                    if (i < (int)lo || (int)hi < i)
                    {
                        return;
                    }

                    if (lo == hi)
                    {
                        selected[0, 0] = 0;
                        selected[1, 1] = x;
                        return;
                    }

                    l!.Update(i, x);
                    r!.Update(i, x);

                    Combine();
                }
            }



        }


        /* 3177. Find the Maximum Length of a Good Subsequence II
        https://leetcode.com/problems/find-the-maximum-length-of-a-good-subsequence-ii/description/
         */
        public class MaximumLengthSol
        {
            /*      Complexity
Time O(nk)
Space O(nk) */
            public int MaximumLength(int[] nums, int k)
            {

                Dictionary<int, int>[] dynamicProgrammingArray = new Dictionary<int, int>[k + 1];
                for (int index = 0; index <= k; index++)
                {
                    dynamicProgrammingArray[index] = new Dictionary<int, int>();
                }
                int[] resultArray = new int[k + 1];
                foreach (int element in nums)
                {
                    for (int index = k; index >= 0; --index)
                    {
                        int value = dynamicProgrammingArray[index].GetValueOrDefault(element, 0);
                        value = Math.Max(value + 1, (index > 0 ? resultArray[index - 1] + 1 : 0));
                        dynamicProgrammingArray[index][element] = value;
                        resultArray[index] = Math.Max(resultArray[index], value);
                    }
                }
                return resultArray[k];
            }
        }


        /* 940. Distinct Subsequences II
        https://leetcode.com/problems/distinct-subsequences-ii/description/
         */
        class DistinctSubseqIISol
        {

            /* Approach 1: Dynamic Programming
            Complexity Analysis
            •	Time Complexity: O(N), where N is the length of S.
            •	Space Complexity: O(N). It is possible to adapt this solution to take O(1) space.

             */
            public int DP(String S)
            {
                int MOD = 1_000_000_007;
                int N = S.Length;
                int[] dp = new int[N + 1];
                dp[0] = 1;

                int[] last = new int[26];
                Array.Fill(last, -1);

                for (int i = 0; i < N; ++i)
                {
                    int x = S[i] - 'a';
                    dp[i + 1] = dp[i] * 2 % MOD;
                    if (last[x] >= 0)
                        dp[i + 1] -= dp[last[x]];
                    dp[i + 1] %= MOD;
                    last[x] = i;
                }

                dp[N]--;
                if (dp[N] < 0) dp[N] += MOD;
                return dp[N];
            }
        }

        /* 446. Arithmetic Slices II - Subsequence
        https://leetcode.com/problems/arithmetic-slices-ii-subsequence/description/
         */

        class NumberOfArithmeticSlicesSol
        {
            private int length;
            private int result;
            /* Approach #1 Brute Force [Time Limit Exceeded]
            Complexity Analysis
•	Time complexity : O(2^n). For each element in the array, it can be in or outside the subsequence. So the time complexity is O(2^n).
•	Space complexity : O(n). We only need the space to store the array.

             */
            public int Naive(int[] array)
            {
                length = array.Length;
                result = 0;
                List<long> current = new List<long>();
                DepthFirstSearch(0, array, current);
                return (int)result;
            }
            private void DepthFirstSearch(int depth, int[] array, List<long> current)
            {
                if (depth == length)
                {
                    if (current.Count < 3)
                    {
                        return;
                    }
                    long difference = current[1] - current[0];
                    for (int i = 1; i < current.Count; i++)
                    {
                        if (current[i] - current[i - 1] != difference)
                        {
                            return;
                        }
                    }
                    result++;
                    return;
                }
                DepthFirstSearch(depth + 1, array, current);
                current.Add((long)array[depth]);
                DepthFirstSearch(depth + 1, array, current);
                current.Remove((long)array[depth]);
            }
            /* Approach #2 Dynamic Programming [Accepted]
            Complexity Analysis
            •	Time complexity : O(n^2). We can use double loop to enumerate all possible states.
            •	Space complexity : O(n^2). For each i, we need to store at most n distinct common differences, so the total space complexity is O(n^2).

             */
            public int DP(int[] A)
            {
                int n = A.Length;
                long ans = 0;
                Dictionary<int, int>[] cnt = new Dictionary<int, int>[n];
                for (int i = 0; i < n; i++)
                {
                    cnt[i] = new Dictionary<int, int>(i);
                    for (int j = 0; j < i; j++)
                    {
                        long delta = (long)A[i] - (long)A[j];
                        if (delta < int.MinValue || delta > int.MaxValue)
                        {
                            continue;
                        }
                        int diff = (int)delta;
                        int sum = cnt[j].GetValueOrDefault(diff, 0);
                        int origin = cnt[i].GetValueOrDefault(diff, 0);
                        cnt[i][diff] = origin + sum + 1;
                        ans += sum;
                    }
                }
                return (int)ans;
            }

        }


        /* 1425. Constrained Subsequence Sum
        https://leetcode.com/problems/constrained-subsequence-sum/description/
         */
        public class ConstrainedSubsetSumSol
        {
            /* Approach 1: Heap/Priority Queue
Complexity Analysis
Given n as the length of nums,
•	Time complexity: O(n⋅logn)
We iterate over each index of nums once. At each iteration, we have a while loop and some heap operations. The while loop runs in O(1) amortized - because an element can only be popped from the heap once, the while loop cannot run more than O(n) times in total across all iterations.
The heap operations depend on the size of the heap. In an array of only positive integers, we will never pop from the heap. Thus, the size of the heap will grow to O(n) and the heap operations will cost O(logn).
•	Space complexity: O(n)
As mentioned above, heap could grow to a size of n.	

             */
            public int MaxHeapPQ(int[] nums, int k)
            {
                //MaxHeap
                PriorityQueue<int[], int[]> heap = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => b[0] - a[0]));

                heap.Enqueue(new int[] { nums[0], 0 }, new int[] { nums[0], 0 });
                int ans = nums[0];

                for (int i = 1; i < nums.Length; i++)
                {
                    while (i - heap.Peek()[1] > k)
                    {
                        heap.Dequeue();
                    }

                    int curr = Math.Max(0, heap.Peek()[0]) + nums[i];
                    ans = Math.Max(ans, curr);
                    heap.Enqueue(new int[] { curr, i }, new int[] { curr, i });
                }

                return ans;
            }

            /* Approach 2: TreeMap-Like Data Structure /SortedDirectionary in C#
Complexity Analysis
Given n as the length of nums,
•	Time complexity: O(n⋅logk)
We iterate over each index of nums once. At each iteration, we have some operations with window. The cost of these operations is a function of the size of window. As window will never exceed a size of k, these operations cost O(logk).
•	Space complexity: O(n)
window will not exceed a size of k, but dp requires O(n) space.

             */
            public int UsingSortedDict(int[] numbers, int k)
            {
                SortedDictionary<int, int> slidingWindow = new SortedDictionary<int, int>();
                slidingWindow[0] = 0;

                int[] dynamicProgrammingArray = new int[numbers.Length];

                for (int index = 0; index < numbers.Length; index++)
                {
                    dynamicProgrammingArray[index] = numbers[index] + slidingWindow.Last().Key;
                    slidingWindow[dynamicProgrammingArray[index]] = slidingWindow.GetValueOrDefault(dynamicProgrammingArray[index], 0) + 1;

                    if (index >= k)
                    {
                        slidingWindow[dynamicProgrammingArray[index - k]]--;
                        if (slidingWindow[dynamicProgrammingArray[index - k]] == 0)
                        {
                            slidingWindow.Remove(dynamicProgrammingArray[index - k]);
                        }
                    }
                }

                int result = int.MinValue;
                foreach (int value in dynamicProgrammingArray)
                {
                    result = Math.Max(result, value);
                }

                return result;
            }
            /* Approach 3: Monotonic Deque
Complexity Analysis
Given n as the length of nums,
•	Time complexity: O(n)
We iterate over each index once. At each iteration, we have a while loop. This while loop runs in O(1) amortized. Each element in nums can only be pushed and popped from queue at most once. Thus, this while loop will not run more than n times across all n iterations. Everything else in each iteration runs in O(1). Thus, each iteration costs O(1) amortized.
•	Space complexity: O(n)
dp requires O(n) space.
Since we always remove out-of-range elements from queue, so it contains at most k elements and requires O(k) space.

             */
            public int UsingMonotonicDeque(int[] nums, int k)
            {
                LinkedList<int> queue = new LinkedList<int>();
                int[] dp = new int[nums.Length];

                for (int index = 0; index < nums.Length; index++)
                {
                    if (queue.Count > 0 && index - queue.First() > k)
                    {
                        queue.RemoveFirst();
                    }

                    dp[index] = (queue.Count > 0 ? dp[queue.First()] : 0) + nums[index];
                    while (queue.Count > 0 && dp[queue.Last.Value] < dp[index])
                    {
                        queue.RemoveLast();
                    }

                    if (dp[index] > 0)
                    {
                        queue.AddLast(index);
                    }
                }

                int answer = int.MinValue;
                foreach (var num in dp)
                {
                    answer = Math.Max(answer, num);
                }

                return answer;
            }
        }



















    }
}