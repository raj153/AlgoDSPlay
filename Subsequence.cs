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




    }
}