using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class MinMaxProbs
    {
        /*
1883. Minimum Skips to Arrive at Meeting On Time	
https://leetcode.com/problems/minimum-skips-to-arrive-at-meeting-on-time/description/	

Complexity
Time O(n^2)
Space O(n)

        */
        public int MinSkips(int[] dist, int s, int target)
        {
            int n = dist.Length;
            int[] dp = new int[n + 1];
            for (int i = 0; i < n; ++i)
            {
                for (int j = n; j >= 0; --j)
                {
                    dp[j] += dist[i];
                    if (i < n - 1)
                        dp[j] = (dp[j] + s - 1) / s * s; // take a rest
                    if (j > 0)
                        dp[j] = Math.Min(dp[j], dp[j - 1] + dist[i]);
                }
            }
            for (int i = 0; i < n; ++i)
            {
                if (dp[i] <= (long)s * target)
                    return i;
            }
            return -1;
        }

        /*
        2188. Minimum Time to Finish the Race	
        https://leetcode.com/problems/minimum-time-to-finish-the-race/description/

        */
        public int MinimumFinishTime(int[][] tires, int changeTime, int numLaps)
        {
            int n = tires.Length;
            int[] dp = new int[numLaps + 1];
            Array.Fill(dp, int.MaxValue);
            dp[0] = 0;
            for (int i = 1; i <= numLaps; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    int time = tires[j][0], sum = 0;
                    for (int k = 1; k <= numLaps - i + 1; ++k)
                    {
                        sum += time;
                        dp[i] = Math.Min(dp[i], dp[i - k] + sum);
                        time *= tires[j][1];
                    }
                }
                dp[i] += changeTime;
            }
            return dp[numLaps];
        }

        /*
        3075. Maximize Happiness of Selected Children
        https://leetcode.com/problems/maximize-happiness-of-selected-children/description/

        */
        public class MaximumHappinessSumSolution
        {
            /*
            Approach 1: Sort + Greedy (SG)
Complexity Analysis
Given n as the length of happiness,
•	Time complexity: O(n⋅logn)
Sorting the happiness array requires O(n⋅logn) time.
Iterating through the first k elements of the sorted array takes O(k) time.
Inside the loop, the max() function and addition operations take constant time.
Overall, the time complexity of the solution is dominated by the sorting step, making the time complexity O(n⋅logn).
•	Space complexity: O(n)
In Python, the sort method sorts a list using the Timesort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space.
In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting two arrays. We also convert the array into an Integer array which has an additional space complexity of O(n).
As the dominating term is O(n), the overall space complexity is O(n).

            */
            public long MaximumHappinessSumSG(int[] happiness, int k)
            {
                int happinessSize = happiness.Length;

                // Convert the array to an Integer array for sorting in descending order
                int[] happinessArray = new int[happinessSize];
                for (int i = 0; i < happinessSize; i++)
                {
                    happinessArray[i] = happiness[i];
                }

                Array.Sort(happinessArray);
                Array.Reverse(happinessArray);

                long totalHappinessSum = 0;
                int turns = 0;

                // Calculate the maximum happiness sum
                for (int i = 0; i < k; i++)
                {
                    // Adjust happiness and ensure it's not negative
                    totalHappinessSum += Math.Max(happinessArray[i] - turns, 0);

                    // Increment turns for the next iteration
                    turns++;
                }

                return totalHappinessSum;
            }

            /*
            Approach 2: Max Heap / Priority Queue + Greedy (MHG)
            Complexity Analysis
            Given n as the length of happiness, and noting that insertion and deletion for the priority_queue data structure takes O(logn) time,
            •	Time complexity: O(n⋅logn+k⋅logn) (C++ and Java) or O(n+k⋅logn) (Python3)
            C++ and Java: Building the priority queue pq involves pushing all elements from the happiness array, which takes O(n⋅logn) time.
            Python3: Building the priority queue pq using heapify() takes O(n) time.
            Iterating through the first k elements of pq takes O(k⋅logn) time. In each iteration, a pop() operation (deletion) is performed, which takes O(logn) time.
            Therefore, the overall time complexity of the solution is O(n⋅logn+k⋅logn) for C++ and Java, and O(n+k⋅logn) for Python3. Since both terms depend on the number of elements in happiness and the value of k, no term can be neglected.
            •	Space complexity: O(n)
            The space complexity is primarily determined by pq, which stores all elements of happiness, making its space complexity O(n).
            Additionally, there are constant space variables used such as totalHappinessSum, turns, i, and a temporary variable for iterating over happiness.
            Therefore, the overall space complexity of the solution is O(n), with pq dominating the space usage.


            */
            public long MaximumHappinessSumMHG(int[] happiness, int k)
            {
                // Create a max heap using PriorityQueue with a custom comparator
                //TODO: check below MaxhHeap condition
                PriorityQueue<int, int> maxHeap = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b.CompareTo(a)));

                // Add all elements to the priority queue
                foreach (int h in happiness)
                {
                    maxHeap.Enqueue(h, h);
                }

                long totalHappinessSum = 0;
                int turns = 0;

                for (int i = 0; i < k; i++)
                {
                    // Add the current highest value to the total happiness sum and remove it from the max heap 
                    totalHappinessSum += Math.Max(maxHeap.Dequeue() - turns, 0);

                    // Increment turns for the next iteration
                    turns++;
                }

                return totalHappinessSum;
            }

        }

        /*
        2244. Minimum Rounds to Complete All Tasks
https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/description/	
Complexity Analysis
Here, N is the number integers in the given array.
•	Time complexity: O(N).
We iterate over the integer array to store the frequencies in the map, this will take O(N) time, then we iterate over the map to find the minimum group needed for each integer, which again will cost O(N). Therefore, the total time complexity is equal to O(N).
•	Space complexity: O(N).
We need the map to store the frequencies of the integers, hence the total space complexity is equal to O(N).

        */
        public int MinimunRounds(int[] tasks)
        {
            Dictionary<int, int> freq = new Dictionary<int, int>();

            // Store the frequencies in the map.
            foreach (int task in tasks)
            {
                if (!freq.ContainsKey(task))
                    freq[task] = 0;
                freq[task]++;
            }

            int minimumRounds = 0;
            foreach (int count in freq.Values)
            {
                // If the frequency is 1, it's not possible to complete tasks.
                if (count == 1)
                    return -1;

                if (count % 3 == 0)
                {
                    // Group all the task in triplets.
                    minimumRounds += count / 3;
                }
                else
                {
                    // If count % 3 = 1; (count / 3 - 1) groups of triplets and 2 pairs.
                    // If count % 3 = 2; (count / 3) groups of triplets and 1 pair.
                    minimumRounds += count / 3 + 1;
                }
            }

            return minimumRounds;
        }


















    }
}