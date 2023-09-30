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

    }
}