using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class ArrayOps
    {
        public static int[] TwoNumberSum(int[] array, int targetSum){
            
            //1.Naive - T:O(n^2) | O(1) - Pair of loops

            //2.Optimal - T: O(n) | O(n)
            var result= TwoNumberSumOptimal(array, targetSum);

            //3.Optimal - T: O(nlog(n)) | O(1)
            result = TwoNumberSumOptimal2(array, targetSum);
            return result;

        }

        private static int[] TwoNumberSumOptimal2(int[] array, int targetSum)
        {
            //IntroSort: insertion sort, heapsort and quick sort based on partitions and call stack
            Array.Sort(array);
            int left=0, right= array.Length-1;

            while(left <right ){

                int currentSum = array[left]+array[right];

                if(currentSum == targetSum){
                    return new int[]{array[left], array[right]};
                }
                else if(currentSum < targetSum){
                    left++;
                }
                else if(currentSum > targetSum)
                    right--;
            }
            return new int[0];

        }

        private static int[] TwoNumberSumOptimal(int[] array, int targetSum)
        {
            HashSet<int> set = new HashSet<int>();

            foreach(int val in array){
                int potentialMatch = targetSum-val;
                if(set.Contains(potentialMatch))
                    return new int[]{val, potentialMatch};
                else
                    set.Add(val);
            }
            return new int[0];
            
            
        }
    }
}