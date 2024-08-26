using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class Subsequence
    {
        //https://www.algoexpert.io/questions/validate-subsequence
        public static bool IsValidSubsequence(List<int> array, List<int> sequence){
            //T:O(n): S:O(1)
            int arrIdx=0, seqIdx=0;
            while(arrIdx< array.Count && seqIdx < sequence.Count){
                if(array[arrIdx] == sequence[seqIdx])
                    seqIdx++;
                
                arrIdx++;


            }
            return seqIdx == sequence.Count;
        }

        //https://www.algoexpert.io/questions/kadane's-algorithm
        public static int FindSubArryMaxSumKadane(int[] array){
            int maxEndingHere = array[0];
            int maxSoFar = array[0];
            for(int i=1; i< array.Length; i++){

                maxEndingHere = Math.Max(maxEndingHere+array[i], array[i]);
                maxSoFar = Math.Max(maxSoFar, maxEndingHere);
            }
            return maxSoFar;
        }
        //https://www.algoexpert.io/questions/max-subset-sum-no-adjacent
        public static int MaxSubsetSumNoAdjacent(int[] array) {
            // T:O(n) time | S:O(n) space
            int result = MaxSubsetSumNoAdjacentNonOptimal(array);
            // T:O(n) time | S:O(1) space

            result = MaxSubsetSumNoAdjacentOptimal(array);
            return result;
        }
        public static int MaxSubsetSumNoAdjacentOptimal(int[] array) {
            if (array.Length == 0) {
            return 0;
            } else if (array.Length == 1) {
            return array[0];
            }
            int second = array[0];
            int first = Math.Max(array[0], array[1]);
            for (int i = 2; i < array.Length; i++) {
            int current = Math.Max(first, second + array[i]);
            second = first;
            first = current;
            }
            return first;
        }
         public static int MaxSubsetSumNoAdjacentNonOptimal(int[] array) {
            // T:O(n) time | S:O(n) space
            if (array.Length == 0) {
            return 0;
            } else if (array.Length == 1) {
            return array[0];
            }
            int[] maxSums = (int[])array.Clone();
            maxSums[1] = Math.Max(array[0], array[1]);
            for (int i = 2; i < array.Length; i++) {
            maxSums[i] = Math.Max(maxSums[i - 1], maxSums[i - 2] + array[i]);
            }
            return maxSums[array.Length - 1];
        }
        //https://www.algoexpert.io/questions/four-number-sum

        public static List<int[]> FourNumberSum(int[] array, int targetSum) {
         // Average: O(n^2) time | O(n^2) space
        // Worst: O(n^3) time | O(n^2) space
    Dictionary<int, List<int[]> > allPairSums =
      new Dictionary<int, List<int[]> >();
    List<int[]> quadruplets = new List<int[]>();
    for (int i = 1; i < array.Length - 1; i++) {
      for (int j = i + 1; j < array.Length; j++) {
        int currentSum = array[i] + array[j];
        int difference = targetSum - currentSum;
        if (allPairSums.ContainsKey(difference)) {
          foreach (int[] pair in allPairSums[difference]) {
            int[] newQuadruplet = { pair[0], pair[1], array[i], array[j] };
            quadruplets.Add(newQuadruplet);
          }
        }
      }
      for (int k = 0; k < i; k++) {
        int currentSum = array[i] + array[k];
        int[] pair = { array[k], array[i] };
        if (!allPairSums.ContainsKey(currentSum)) {
          List<int[]> pairGroup = new List<int[]>();
          pairGroup.Add(pair);
          allPairSums.Add(currentSum, pairGroup);
        } else {
          allPairSums[currentSum].Add(pair);
        }
      }
    }
    return quadruplets;
  }
    //https://www.algoexpert.io/questions/longest-peak
    public static int LongestPeak(int[] array) {
          // O(n) time | O(1) space - where n is the length of the input array
    int longestPeakLength = 0;
    int i = 1;
    while (i < array.Length - 1) {
      bool isPeak = array[i - 1] < array[i] && array[i] > array[i + 1];
      if (!isPeak) {
        i += 1;
        continue;
      }

      int leftIdx = i - 2;
      while (leftIdx >= 0 && array[leftIdx] < array[leftIdx + 1]) {
        leftIdx -= 1;
      }

      int rightIdx = i + 2;
      while (rightIdx < array.Length && array[rightIdx] < array[rightIdx - 1]) {
        rightIdx += 1;
      }
      int currentPeakLength = rightIdx - leftIdx - 1;
      if (currentPeakLength > longestPeakLength) {
        longestPeakLength = currentPeakLength;
      }
      i = rightIdx;
    }
    return longestPeakLength;
  }
        


    }
}