using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class GreedyProbs
    {
        /*
        2136. Earliest Possible Day of Full Bloom
https://leetcode.com/problems/earliest-possible-day-of-full-bloom/description/

Greedy approach
Complexity Analysis
Let n denote the number of seeds.
•	Time complexity: O(nlogn).
We sort the seeds with O(nlogn) time and iterate it with O(n) time.
•	Space complexity: O(n).
We use O(n) memory for indices and sorting.
 
        */
        public int EarliestFullBloom(int[] plantTime, int[] growTime)
        {
            int numberOfPlants = growTime.Length;
            List<int> indices = new List<int>();
            for (int i = 0; i < numberOfPlants; ++i)
            {
                indices.Add(i);
            }
            indices.Sort((i, j) => growTime[j].CompareTo(growTime[i]));
            int result = 0;
            for (int i = 0, currentPlantTime = 0; i < numberOfPlants; ++i)
            {
                int index = indices[i];
                int time = currentPlantTime + plantTime[index] + growTime[index];
                result = Math.Max(result, time);
                currentPlantTime += plantTime[index];
            }
            return result;
        }

/*
2548. Maximum Price to Fill a Bag
https://leetcode.com/problems/maximum-price-to-fill-a-bag/description/ 
https://algo.monster/liteproblems/2548

**/
        public class MaxPriceToFillABagSol
        {
            /*
            Approach: Greedy
            Time and Space Complexity
Time Complexity
The time complexity of the provided code consists of two main operations: sorting the list and iterating through the list.
•	Sorting the list of items has a time complexity of O(NlogN), where N is the length of the items. This is because the built-in sorted() function in Python uses TimSort (a combination of merge sort and insertion sort) which has this time complexity for sorting an array.
•	The iteration through the sorted list has a time complexity of O(N) since each item is being accessed once to calculate the proportional value and update the remaining capacity.
Combining these two operations, the overall time complexity is O(NlogN) + O(N), which simplifies to O(NlogN) as the sorting operation is the dominant term.
Space Complexity
•	The space complexity of sorting in Python is O(N) because the sorted() function generates a new list.
•	The additional space used in the code for variables like ans and v are constant O(1).
Therefore, the overall space complexity is O(N) due to the sorted list that is created and used for iteration.

            */

            // Function to calculate the maximum price achievable within the given capacity
            public double Greedy(int[][] items, int capacity)
            {
                // Sort the items array based on value-to-weight ratio in descending order
                Array.Sort(items, (item1, item2)=>item2[0] * item1[1] - item1[0] * item2[1]);

                // Variable to store the cumulative value of chosen items
                double totalValue = 0;

                // Iterate through each item
                foreach (int[] item in items)
                {
                    int price = item[0];
                    int weight = item[1];

                    // Determine the weight to take, up to the remaining capacity
                    int weightToTake = Math.Min(weight, capacity);

                    // Compute value contribution of this item based on the weight taken
                    double valueContribution = (double)weightToTake / weight * price;

                    // Add the value contribution to the total value
                    totalValue += valueContribution;

                    // Subtract the weight taken from the remaining capacity
                    capacity -= weightToTake;

                    // If no capacity is left, break the loop as no more items can be taken
                    if (capacity == 0)
                    {
                        break;
                    }
                }

                // If there is unused capacity, the requirement to fill the exact capacity is not met
                // In this context, return -1 to indicate the requirement is not fulfilled
                return capacity > 0 ? -1 : totalValue;
            }
        }







































    }
}