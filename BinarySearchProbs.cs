using System;
using System.Collections.Generic;
using System.Formats.Asn1;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class BinarySearchProbs
    {
        //https://www.algoexpert.io/questions/shifted-binary-search
        public static int ShiftedBinarySearch(int[] array, int target)
        {

            //1.Recursive
            //T:O(log(n)) | S:(n)
            int result = ShiftedBinarySearchRec(array, target, 0, array.Length - 1);

            //2.Iterative
            //T:O(log(n)) | S:(1)
            result = ShiftedBinarySearchIterative(array, target, 0, array.Length - 1);

            return result;
        }

        private static int ShiftedBinarySearchIterative(int[] array, int target, int left, int right)
        {
            while (left <= right)
            {
                int middle = (left + right) / 2;
                int potentialMatch = array[middle];
                int leftNum = array[left];
                int rightNum = array[right];

                if (target == potentialMatch) return middle;
                else if (leftNum <= potentialMatch)
                {
                    if (target < potentialMatch && target >= leftNum)
                    {
                        right = middle - 1;
                    }
                    else left = middle + 1;
                }
                else
                {
                    if (target > potentialMatch && target <= rightNum)
                    {
                        left = middle + 1;
                    }
                    else right = middle - 1;
                }
            }
            return -1;
        }

        private static int ShiftedBinarySearchRec(int[] array, int target, int left, int right)
        {
            if (left > right) return -1;

            int middle = (left + right) / 2;
            int leftNum = array[left];
            int rightNum = array[right];
            int potentialMatch = array[middle];
            if (target == potentialMatch)
            {
                return middle;
            }
            else if (left <= potentialMatch)
            {
                if (target < potentialMatch && target >= leftNum)
                {
                    return ShiftedBinarySearchRec(array, target, left, middle - 1);
                }
                else
                {
                    return ShiftedBinarySearchRec(array, target, middle + 1, right);
                }
            }
            else
            {
                if (target > potentialMatch && target <= rightNum)
                {
                    return ShiftedBinarySearchRec(array, target, middle + 1, rightNum);
                }
                else
                {
                    return ShiftedBinarySearchRec(array, target, left, middle - 1);
                }
            }


        }
        public static int[] SearchForRange(int[] array, int target)
        {
            int[] finalRange = new int[] { -1, -1 };

            //1. Recursion
            //T:O(log(n)) | S:O(log(n))
            AlteredBinarySearchRec(array, target, 0, array.Length - 1, finalRange, true);
            AlteredBinarySearchRec(array, target, 0, array.Length - 1, finalRange, false);

            //2. Iterative with constant space
            //T:O(log(n)) | S:O(1)
            AlteredBinarySearchIterative(array, target, 0, array.Length - 1, finalRange, true);
            AlteredBinarySearchIterative(array, target, 0, array.Length - 1, finalRange, false);
            return finalRange;
        }

        private static void AlteredBinarySearchIterative(int[] array, int target, int left, int right, int[] finalRange, bool goLeft)
        {
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (array[mid] < target)
                {
                    left = mid + 1;
                }
                else if (array[mid] > target)
                {
                    right = mid - 1;
                }
                else
                {
                    if (goLeft)
                    {
                        if (mid == 0 || array[mid - 1] != target)
                        {
                            finalRange[0] = mid;
                            return;
                        }
                        else
                        {
                            right = mid - 1;
                        }
                    }
                    else
                    {
                        if (mid == array.Length - 1 || array[mid + 1] != target)
                        {
                            finalRange[1] = mid;
                        }
                        else
                        {
                            left = mid + 1;
                        }
                    }
                }
            }

        }

        private static void AlteredBinarySearchRec(int[] array, int target, int left, int right, int[] finalRange, bool goLeft)
        {
            if (left > right) return;

            int mid = (left + right) / 2;

            if (array[mid] < target)
            {
                AlteredBinarySearchRec(array, target, mid + 1, right, finalRange, goLeft);
            }
            else if (array[mid] > target)
            {
                AlteredBinarySearchRec(array, target, left, mid - 1, finalRange, goLeft);
            }
            else
            {
                if (goLeft)
                {
                    if (mid == 0 || array[mid - 1] != target)
                    {
                        finalRange[0] = mid;
                    }
                    else
                    {
                        AlteredBinarySearchRec(array, target, left, mid - 1, finalRange, goLeft);
                    }
                }
                else
                {
                    if (mid == array.Length - 1 || array[mid + 1] != target)
                    {
                        finalRange[1] = mid;
                    }
                    else
                    {
                        AlteredBinarySearchRec(array, target, mid + 1, right, finalRange, goLeft);
                    }
                }
            }
        }
        //https://www.algoexpert.io/questions/binary-search
        // O(log(n)) time | O(log(n)) space
        public static int BinarySearchRec(int[] array, int target)
        {
            return BinarySearchRec(array, target, 0, array.Length - 1);
        }

        public static int BinarySearchRec(int[] array, int target, int left, int right)
        {
            if (left > right)
            {
                return -1;
            }
            int middle = (left + right) / 2;
            int potentialMatch = array[middle];
            if (target == potentialMatch)
            {
                return middle;
            }
            else if (target < potentialMatch)
            {
                return BinarySearchRec(array, target, left, middle - 1);
            }
            else
            {
                return BinarySearchRec(array, target, middle + 1, right);
            }
        }
        // O(log(n)) time | O(1) space
        public static int BinarySearchIterative(int[] array, int target)
        {
            return BinarySearchIterative(array, target, 0, array.Length - 1);
        }

        public static int BinarySearchIterative(int[] array, int target, int left, int right)
        {
            while (left <= right)
            {
                int middle = (left + right) / 2;
                int potentialMatch = array[middle];
                if (target == potentialMatch)
                {
                    return middle;
                }
                else if (target < potentialMatch)
                {
                    right = middle - 1;
                }
                else
                {
                    left = middle + 1;
                }
            }
            return -1;
        }

        /*875. Koko Eating Bananas
    https://leetcode.com/problems/koko-eating-bananas/description/
    */
        public class KokoEatingBananaSolution
        {
            /*            
Approach 1: Brute Force
Complexity Analysis
Let n be the length of input array piles and m be the upper bound of elements in piles.
•	Time complexity: O(nm)
o	For each eating speed speed, we iterate over piles and calculate the overall time, which takes O(n) time.
o	Before finding the first workable eating speed, we must try every smaller eating speed. Suppose in the worst-case scenario (when the answer is m), we have to try every eating speed from 1 to m, that is a total of m iterations over the array.
o	To sum up, the overall time complexity is O(nm)
•	Space complexity: O(1)
o	For each eating speed speed, we iterate over the array and calculate the total hours Koko spends, which costs constant space.
o	Therefore, the overall space complexity is O(1).
            */
            public int MinEatingSpeed(int[] piles, int h)
            {
                int maxPile = piles.Max();
                for (int speed = 1; speed <= maxPile; speed++)
                {
                    // hourSpent stands for the total hour Koko spends with 
                    // the given eating speed.                    
                    int hourSpent = 0;
                    // Iterate over the piles and calculate hourSpent.
                    // We increase the hourSpent by (pile / speed)
                    foreach (int pile in piles)
                    {
                        hourSpent += (pile + speed - 1) / speed;
                    }
                    // Check if Koko can finish all the piles within h hours,
                    // If so, return speed. 
                    if (hourSpent <= h)
                    {
                        return speed;
                    }
                }
                return -1;
            }

            /*
Approach 2: Binary Search
Complexity Analysis
Let n be the length of input array piles and m be the upper bound of elements in piles.
•	Time complexity: O(nlog(m))
o	For each eating speed speed, we iterate over the array and calculate the total hours Koko spends, which takes O(n) time.
o	Before finding the first workable eating speed, we must try every smaller eating speed. Suppose in the worst-case scenario (when the answer is m), we have to try every eating speed from 1 to m, that is a total of m iterations over the array.
o	To sum up, the overall time complexity is O(nlog(m))
•	Space complexity: O(1)
o	For each eating speed speed, we iterate over the array and calculate the total hours Koko spends, which costs constant space.
o	Therefore, the overall space complexity is O(1).
            */
            public int MinEatingSpeedBinarySearch(int[] piles, int h)
            {
                int left = 1;
                int right = piles.Max();
                while (left <= right)
                {
                    // Get the middle index between left and right boundary indexes.
                    // hourSpent stands for the total hour Koko spends.
                    int mid = left + (right - left) / 2;
                    int hourSpent = 0;
                    // Iterate over the piles and calculate hourSpent.
                    // We increase the hourSpent by ceil(pile / middle)                    
                    foreach (int pile in piles)
                    {
                        hourSpent += (pile + mid - 1) / mid;
                    }
                    // Check if middle is a workable speed, and cut the search space by half.
                    if (hourSpent <= h)
                    {
                        right = mid - 1;
                    }
                    else
                    {
                        left = mid + 1;
                    }
                }
                // Once the left and right boundaries coincide, we find the target value,
                // that is, the minimum workable eating speed.
                return left;
            }


        }
        /*
        774. Minimize Max Distance to Gas Station	
        https://leetcode.com/problems/minimize-max-distance-to-gas-station/description/
        */
        public class MinMaxGasDistSolution
        {
            /*
            Approach #1: Dynamic Programming [Memory Limit Exceeded]
            Complexity Analysis
            •	Time Complexity: O(NK^2), where N is the length of stations.
            •	Space Complexity: O(NK), the size of dp.

            */
            public double MinMaxGasDistDP(int[] stations, int K)
            {
                int N = stations.Length;
                double[] deltas = new double[N - 1];
                for (int i = 0; i < N - 1; ++i)
                    deltas[i] = stations[i + 1] - stations[i];

                double[][] dp = new double[N - 1][];
                //dp[i][j] = answer for deltas[:i+1] when adding j gas stations
                for (int i = 0; i <= K; ++i)
                    dp[0][i] = deltas[0] / (i + 1);

                for (int p = 1; p < N - 1; ++p)
                    for (int k = 0; k <= K; ++k)
                    {
                        double bns = 999999999;
                        for (int x = 0; x <= k; ++x)
                            bns = Math.Min(bns, Math.Max(deltas[p] / (x + 1), dp[p - 1][k - x]));
                        dp[p][k] = bns;
                    }

                return dp[N - 2][K];
            }

            /*
            
Approach #2: Brute Force [Time Limit Exceeded]
Complexity Analysis
•	Time Complexity: O(NK), where N is the length of stations.
•	Space Complexity: O(N), the size of deltas and count.

            */
            public double MinMaxGasDistNaive(int[] stations, int K)
            {
                int N = stations.Length;
                double[] deltas = new double[N - 1];
                for (int i = 0; i < N - 1; ++i)
                    deltas[i] = stations[i + 1] - stations[i];

                int[] count = new int[N - 1];
                Array.Fill(count, 1);

                for (int k = 0; k < K; ++k)
                {
                    // Find interval with largest part
                    int best = 0;
                    for (int i = 0; i < N - 1; ++i)
                        if (deltas[i] / count[i] > deltas[best] / count[best])
                            best = i;

                    // Add gas station to best interval
                    count[best]++;
                }

                double ans = 0;
                for (int i = 0; i < N - 1; ++i)
                    ans = Math.Max(ans, deltas[i] / count[i]);

                return ans;
            }
            /*
            Approach #3: Heap [Time Limit Exceeded]
            Complexity Analysis
            Let N be the length of stations, and K be the number of gas stations to add.
•	Time Complexity: O(N+KlogN)
o	First of all, we scan the stations to obtain a list of intervals between each adjacent stations.
o	Then it takes another O(N) to build a heap out of the list of intervals.
o	Finally, we repeatedly pop out an element and push in a new element into the heap, which takes O(logN) respectively. In total, we repeat this step for K times (i.e. to add K gas stations).
o	To sum up, the overall time complexity of the algorithm is O(N)+O(N)+O(K⋅logN)=O(N+K⋅logN).
•	Space Complexity: O(N), the size of deltas and count.

*/
            public double MinMaxGasDistHeap(int[] stations, int K)
            {
                int N = stations.Length;
                PriorityQueue<int[], int[]> pq = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) =>
                    (double)b[0] / b[1] < (double)a[0] / a[1] ? -1 : 1));
                for (int i = 0; i < N - 1; ++i)
                {
                    var tmp = new int[] { stations[i + 1] - stations[i], 1 };
                    pq.Enqueue(tmp, tmp);
                }

                for (int k = 0; k < K; ++k)
                {
                    int[] nodeTmp = pq.Dequeue();
                    nodeTmp[1]++;
                    pq.Enqueue(nodeTmp, nodeTmp);
                }

                int[] node = pq.Dequeue();
                return (double)node[0] / node[1];
            }

            /*
            
            Approach #4: Binary Search [Accepted] (BS)
            Complexity Analysis
•	Time Complexity: O(NlogW), where N is the length of stations, and W=10^14 is the range of possible answers (108), divided by the acceptable level of precision (10^−6).
•	Space Complexity: O(1) in additional space complexity.

            */
            public double MinMaxGasDistBS(int[] stations, int K)
            {
                double lo = 0, hi = 1e8;
                while (hi - lo > 1e-6)
                {
                    double mi = (lo + hi) / 2.0;
                    if (Possible(mi, stations, K))
                        hi = mi;
                    else
                        lo = mi;
                }
                return lo;

            }
            private bool Possible(double D, int[] stations, int K)
            {
                int used = 0;
                for (int i = 0; i < stations.Length - 1; ++i)
                    used += (int)((stations[i + 1] - stations[i]) / D);
                return used <= K;
            }

        }
        /*
        https://leetcode.com/problems/maximum-candies-allocated-to-k-children/

        Approach: Binary Search
        Complexity
        Time O(nlog10000000)
        Space O(1)

        */
        public int MaximumCandies(int[] candies, long k)
        {
            int left = 0, right = 10_000_000;
            while (left < right)
            {
                long sum = 0;
                int mid = (left + right + 1) / 2;
                foreach (int candiNum in candies)
                {
                    sum += candiNum / mid;
                }
                if (k > sum)
                    right = mid - 1;
                else
                    left = mid;
            }
            return left;


        }
        /*
        1870. Minimum Speed to Arrive on Time
https://leetcode.com/problems/minimum-speed-to-arrive-on-time/description/

Approach: Binary Search
Complexity Analysis
Here, N is the number of rides, and K is the size of the search space. For this problem, K is equal to 107.
•	Time complexity: O(NlogK)
After each iteration, the search space gets reduced to half; hence the while loop will take logK operations. For each such operation, we need to call the timeRequired function, which takes O(N) time. Therefore, the total time complexity equals O(NlogK).
•	Space complexity: O(1)
No extra space is required other than the three variables. Hence the space complexity is constant.

        */

        public int MinSpeedOnTime(int[] dist, double hour)
        {
            int left = 1;
            int right = 10000000; ;
            int minSpeed = -1;

            while (left <= right)
            {
                int mid = (left + right) / 2;

                // Move to the left half.
                if (TimeRequired(dist, mid) <= hour)
                {
                    minSpeed = mid;
                    right = mid - 1;
                }
                else
                {
                    // Move to the right half.
                    left = mid + 1;
                }
            }
            return minSpeed;

        }
        double TimeRequired(int[] dist, int speed)
        {
            double time = 0.0;
            for (int i = 0; i < dist.Length; i++)
            {
                double t = (double)dist[i] / (double)speed;
                // Round off to the next integer, if not the last ride.
                time += (i == dist.Length - 1 ? t : Math.Ceiling(t));
            }
            return time;
        }

        /*
        2187. Minimum Time to Complete Trips
https://leetcode.com/problems/minimum-time-to-complete-trips/description/
Approach: Binary Search
Complexity Analysis
Let n be the length of time, m be the upper limit of totalTrips and k be the maximum time taken by one trip.
•	Time complexity: O(n⋅log(m⋅k))
o	We set the right boundary of the searching space as m⋅k. The searching space is cut by half each time, thus it takes O(log(m⋅k)) steps to finish the binary search.
o	In each step, we iterate the entire array time to calculate the number of trips made in the given time, it takes O(n) time.
o	To sum up, the time complexity is O(n⋅log(m⋅k)).
•	Space complexity: O(1)
o	During the binary search, we only need to record the two boundaries left and right, and the number of trips made in each given time mid. Therefore the space complexity is O(1).

        */
        // Can these buses finish 'totalTrips' of trips in 'givenTime'? 
        public bool TimeEnough(int[] time, long givenTime, int totalTrips)
        {
            long actualTrips = 0;
            foreach (int t in time)
                actualTrips += givenTime / t;
            return actualTrips >= totalTrips;
        }

        public long MinimumTime(int[] time, int totalTrips)
        {
            // Initialize the left and right boundaries.
            int max_time = 0;
            foreach (int t in time)
            {
                max_time = Math.Max(max_time, t);
            }
            long left = 1, right = (long)max_time * totalTrips;

            // Binary search to find the minimum time to finish the task.
            while (left < right)
            {
                long mid = (left + right) / 2;
                if (TimeEnough(time, mid, totalTrips))
                {
                    right = mid;
                }
                else
                {
                    left = mid + 1;
                }
            }
            return left;
        }

        /*
        1482. Minimum Number of Days to Make m Bouquets	
    https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/description/

    Approach: Binary Search
    Complexity Analysis
    Here, N is the number of flowers and D is the highest value in the array bloomDay.
    •	Time complexity: O(NlogD).
    The search space is from 1 to D and for each of the chosen values of mid in the binary search we will iterate over the N flowers. Therefore the time complexity is equal to O(NlogD).
    •	Space complexity: O(1)

        */

        public int MinDaysToMakeBoq(int[] bloomDay, int m, int k)
        {
            int start = 0;
            int end = 0;
            foreach (int day in bloomDay)
            {
                end = Math.Max(end, day);
            }

            int minDays = -1;
            while (start <= end)
            {
                int mid = (start + end) / 2;

                if (GetNumOfBouquets(bloomDay, mid, k) >= m)
                {
                    minDays = mid;
                    end = mid - 1;
                }
                else
                {
                    start = mid + 1;
                }
            }

            return minDays;
        }
        // Return the number of maximum bouquets that can be made on day mid.
        private int GetNumOfBouquets(int[] bloomDay, int mid, int k)
        {
            int numOfBouquets = 0;
            int count = 0;

            for (int i = 0; i < bloomDay.Length; i++)
            {
                // If the flower is bloomed, add to the set. Else reset the count.
                if (bloomDay[i] <= mid)
                {
                    count++;
                }
                else
                {
                    count = 0;
                }

                if (count == k)
                {
                    numOfBouquets++;
                    count = 0;
                }
            }

            return numOfBouquets;
        }

        /*
        74. Search a 2D Matrix
https://leetcode.com/problems/search-a-2d-matrix/description/
Approach: Binary Search
Complexity Analysis
•	Time complexity : O(log(mn)) since it's a standard binary search.
•	Space complexity : O(1).

        */

        public bool SearchMatrix(int[][] matrix, int target)
        {
            int m = matrix.Length;
            if (m == 0)
                return false;
            int n = matrix[0].Length;
            int left = 0, right = m * n - 1;
            int pivotIdx, pivotElement;
            while (left <= right)
            {
                pivotIdx = (left + right) / 2;
                pivotElement = matrix[pivotIdx / n][pivotIdx % n];
                if (target == pivotElement)
                    return true;
                else
                {
                    if (target < pivotElement)
                        right = pivotIdx - 1;
                    else
                        left = pivotIdx + 1;
                }
            }

            return false;
        }

        /*
        2604. Minimum Time to Eat All Grains
https://leetcode.com/problems/minimum-time-to-eat-all-grains/description/
https://algo.monster/liteproblems/2604
        */
        class MinimumTimeToEatAllGrainsSol
        {
            private int[] hensPositions;
            private int[] grainPositions;
            private int totalGrains;

        /*
        Approach: Binary Search
     Time and Space Complexity
Time Complexity
The time complexity of the given code can be analyzed as follows:
1.	Sorting both the hens and grains lists: hens.sort() and grains.sort() both have a time complexity of O(n \log n) and O(m \log m) respectively, where n is the number of hens and m is the number of grains.
2.	Binary search: The bisect_left function performs binary search on a range of size r. The size of this range can be considered as U because it is determined by the maximum gap between positions of grains. Since it performs the check function at each step of the binary search, the time it takes is O(\log U) for the binary search times the complexity of the check function itself.
3.	check function: Inside the binary search, the check function is called, which runs in O(m + n) because it goes through all grains and potentially all hens in the worst case.
Combining these factors, we get a total time complexity of O(n \log n + m \log m + (m + n) \log U).
Space Complexity
The space complexity of the given code can be analyzed as follows:
1.	Sorting requires O(1) additional space if the sort is in-place (which Python's sort method is, for example).
2.	The binary search uses O(1) space.
3.	The check function uses O(1) space since it only uses a few variables and all operations are done in place.
Adding these up, we get a space complexity of O(\log m + \log n) for the recursion stack of the sorting algorithms if the sort implementation used is not in-place.
   
        */
            public int BinarySearchWithSort(int[] hens, int[] grains)
            {
                totalGrains = grains.Length;
                hensPositions = hens;
                grainPositions = grains;
                Array.Sort(hensPositions);
                Array.Sort(grainPositions);

                // Set the initial range of time to search for the minimum time needed.
                // It's between 0 and the maximum possible distance a hen needs to travel minus the first grain position.
                int left = 0;
                int right = Math.Abs(hensPositions[0] - grainPositions[0]) + grainPositions[totalGrains - 1] - grainPositions[0];

                // Use binary search to find the minimum time needed.
                while (left < right)
                {
                    int mid = (left + right) / 2;
                    if (IsTimeSufficient(mid))
                    {
                        right = mid;
                    }
                    else
                    {
                        left = mid + 1;
                    }
                }

                // 'left' will hold the minimum time needed when the while-loop finishes.
                return left;
            }

            private bool IsTimeSufficient(int allowedTime)
            {
                int grainIndex = 0;

                // Check if the time allowed is sufficient for each hen.
                foreach (int henPosition in hensPositions)
                {
                    // If all grains have been checked, return true.
                    if (grainIndex == totalGrains)
                    {
                        return true;
                    }

                    int grainPosition = grainPositions[grainIndex];
                    if (grainPosition <= henPosition)
                    {
                        int distance = henPosition - grainPosition;
                        if (distance > allowedTime)
                        {
                            // If the current grain is too far for the time allowed, return false.
                            return false;
                        }

                        // Find the next grain that is farther than the hen but within the allowed time.
                        while (grainIndex < totalGrains && grainPositions[grainIndex] <= henPosition)
                        {
                            grainIndex++;
                        }

                        // Attempt to get as close as possible to the current hen position without exceeding the allowed time.
                        while (grainIndex < totalGrains && Math.Min(distance, grainPositions[grainIndex] - henPosition) + grainPositions[grainIndex] - grainPosition <= allowedTime)
                        {
                            grainIndex++;
                        }
                    }
                    else
                    {
                        // Find the grain that is within the allowed time for the current hen.
                        while (grainIndex < totalGrains && grainPositions[grainIndex] - henPosition <= allowedTime)
                        {
                            grainIndex++;
                        }
                    }
                }

                // Return true if all grains have been assigned, false otherwise.
                return grainIndex == totalGrains;
            }
        }


















    }
}