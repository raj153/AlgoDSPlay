using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class SubArrayProbs
    {

        /*         2444. Count Subarrays With Fixed Bounds
        https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description/
         */
        public class CountSubarraysWithFixedBoundsSol
        {
            /*
            Approach: Two Pointers
Complexity Analysis
Let n be the length of the input array nums.
•	Time complexity: O(n)
o	We need one iteration over nums, for each step during the iteration, we need to update some variables which take constant time.
o	The overall time complexity is O(n).
•	Space complexity: O(1)
o	We only need to maintain four integer variables, minPosition, maxPosition, leftBound and answer.

            */
            public long CountSubarrays(int[] numbers, int minimumValue, int maximumValue)
            {
                // minPosition, maxPosition: the MOST RECENT positions of minK and maxK.
                // leftBound: the MOST RECENT value outside the range [minK, maxK].
                long totalSubarrays = 0;
                int minPosition = -1, maxPosition = -1, leftBound = -1;

                // Iterate over nums, for each number at index i:
                for (int index = 0; index < numbers.Length; ++index)
                {
                    // If the number is outside the range [minK, maxK], update the most recent leftBound.
                    if (numbers[index] < minimumValue || numbers[index] > maximumValue)
                        leftBound = index;

                    // If the number is minK or maxK, update the most recent position.
                    if (numbers[index] == minimumValue)
                        minPosition = index;
                    if (numbers[index] == maximumValue)
                        maxPosition = index;

                    // The number of valid subarrays equals the number of elements between leftBound and 
                    // the smaller of the two most recent positions (minPosition and maxPosition).
                    totalSubarrays += Math.Max(0, Math.Min(maxPosition, minPosition) - leftBound);
                }
                return totalSubarrays;
            }
        }
    
    
    
    
    }
}