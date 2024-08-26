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
        //https://www.algoexpert.io/questions/zero-sum-subarray
        public static bool ZeroSumSubArray(int[] nums){

            //1.Naive - pair of loops to create subarrays to check its sum to zero
            //T:O(n^2) |S:O(1)

            //2. By logic that if current sum repeated a previous one then some subarrys sum is zero
            //[0,x]=>s [o,..,x,.., y,..]=>s then [x+1, y] must zero sum sub array
            //T:O(n^2) |S:O(n)
            HashSet<int> sums = new HashSet<int>();
            sums.Add(0);
            int currentSum =0;
            foreach(int num in nums){
                currentSum +=num;
                if(sums.Contains(currentSum)) return true;

                sums.Add(currentSum);
            }
            return false;
        }
        //https://www.algoexpert.io/questions/longest-subarray-with-sum
        public static int[] LongestSubarrayWithSum(int[] arr, int targetSum){

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
            int[] indices = new int[]{};
            int currentSubArraySum =0;
            int startIdx=0, endIdx =0;

            while(endIdx < arr.Length){

                currentSubArraySum += arr[endIdx];
                while( currentSubArraySum > targetSum && startIdx < endIdx ){
                    currentSubArraySum -= arr[startIdx];
                    startIdx++;
                }

                if(currentSubArraySum == targetSum){
                    if(indices.Length == 0 || indices[1]-indices[0] < endIdx-startIdx){
                        indices = new int[]{startIdx, endIdx};
                    }
                }
                endIdx++;
            }
            return indices;


        }

        private static int[] LongestSubarrayWithSumNaive(int[] arr, int targetSum)
        {
            int[] indices = new int[]{};
            for(int startIdx=0; startIdx< arr.Length; startIdx++){
                int currentSubArraySum =0;
                
                for(int endIdx=startIdx; endIdx < arr.Length; endIdx++ ){
                    currentSubArraySum +=  arr[endIdx];

                    if(currentSubArraySum == targetSum){

                        if(indices.Length == 0 || indices[1]-indices[0] < endIdx-startIdx){
                            indices = new int[]{startIdx, endIdx};
                        }
                    }
                }

            }
            return indices;
        }
    }
}