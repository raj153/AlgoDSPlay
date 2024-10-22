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


        /* 992. Subarrays with K Different Integers
        https://leetcode.com/problems/subarrays-with-k-different-integers/description/
         */
        class SubarraysWithKDistinctSol
        {
            /*
             Approach 1: Sliding Window
             Complexity Analysis
Let n be the length of the nums array.
•	Time complexity: O(n)
The time complexity is O(n) because the slidingWindowAtMost function iterates through the array once using the sliding window technique, and each element is processed at most twice (once when it enters the window and once when it exits the window). Inside the loop, the operations of updating the frequency map and shrinking the window take O(1) time on average, assuming the underlying hash table implementation has constant-time operations. Therefore, the overall time complexity is linear with respect to the size of the input array.
•	Space complexity: O(n)
The space complexity is O(n) due to the use of the freqMap to store the frequency of elements in the current window. In the worst case, when all elements in the array are distinct, the freqMap will store all the elements, resulting in a space complexity of O(n).
It's important to note that the space complexity is also affected by the underlying implementation of the hash table used for the freqMap. Some implementations may have additional overhead, leading to a slightly higher space complexity.

             */
            public int UsingSlidingWindow(int[] nums, int k)
            {
                return SlidingWindowAtMost(nums, k) - SlidingWindowAtMost(nums, k - 1);
            }

            // Helper function to count the number of subarrays with at most k distinct elements.
            private int SlidingWindowAtMost(int[] nums, int distinctK)
            {
                // To store the occurrences of each element.
                Dictionary<int, int> frequencyMap = new Dictionary<int, int>();
                int leftPointer = 0, totalCount = 0;

                // Right pointer of the sliding window iterates through the array.
                for (int rightPointer = 0; rightPointer < nums.Length; rightPointer++)
                {
                    if (frequencyMap.ContainsKey(nums[rightPointer]))
                    {
                        frequencyMap[nums[rightPointer]]++;
                    }
                    else
                    {
                        frequencyMap[nums[rightPointer]] = 1;
                    }

                    // If the number of distinct elements in the window exceeds k,
                    // we shrink the window from the left until we have at most k distinct elements.
                    while (frequencyMap.Count > distinctK)
                    {
                        frequencyMap[nums[leftPointer]]--;
                        if (frequencyMap[nums[leftPointer]] == 0)
                        {
                            frequencyMap.Remove(nums[leftPointer]);
                        }
                        leftPointer++;
                    }

                    // Update the total count by adding the length of the current subarray.
                    totalCount += (rightPointer - leftPointer + 1);
                }
                return totalCount;
            }
            /* 
            Approach 2: Sliding Window in One Pass
Complexity Analysis
Let n be the length of the nums array.
•	Time complexity: O(n)
The time complexity is O(n) because the algorithm iterates through the array once using the sliding window technique, and each element is processed at most twice (once when it enters the window and once when it exits the window), resulting in linear time complexity.
•	Space complexity: O(n)
The space complexity is also O(n) because the algorithm uses a mapping array to store the count of distinct elements encountered in the current window. In the worst case, this array can grow to the size of the input array; hence, the space complexity is linear with respect to the size of the input.

             */
            public int UsingSlidingWindowInOnePass(int[] nums, int k)
            {
                // Array to store the count of distinct values encountered
                int[] distinctCount = new int[nums.Length + 1];

                int totalCount = 0;
                int left = 0;
                int right = 0;
                int currCount = 0;

                while (right < nums.Length)
                {
                    // Increment the count of the current element in the window
                    if (distinctCount[nums[right++]]++ == 0)
                    {
                        // If encountering a new distinct element, decrement K
                        k--;
                    }

                    // If K becomes negative, adjust the window from the left
                    if (k < 0)
                    {
                        // Move the left pointer until the count of distinct elements becomes valid again
                        --distinctCount[nums[left++]];
                        k++;
                        currCount = 0;
                    }

                    // If K becomes zero, calculate subarrays
                    if (k == 0)
                    {
                        // While the count of left remains greater than 1, keep shrinking the window from the left
                        while (distinctCount[nums[left]] > 1)
                        {
                            --distinctCount[nums[left++]];
                            currCount++;
                        }
                        // Add the count of subarrays with K distinct elements to the total count
                        totalCount += (currCount + 1);
                    }
                }
                return totalCount;
            }
        }

        /* 1793. Maximum Score of a Good Subarray
        https://leetcode.com/problems/maximum-score-of-a-good-subarray/description/
         */
        public class MaximumScoreOfAGoodSubarraySol
        {

            /* Approach 1: Binary Search
Complexity Analysis
Given n as the length of nums,
•	Time complexity: O(n⋅logn)
We require O(n) time to create left and right. Then, we iterate over the indices of right, which is not more than O(n) iterations. At each iteration, we perform a binary search over left, which does not cost more than O(logn). Thus, solve costs O(n⋅logn), and we call it twice.
•	Space complexity: O(n)
left and right have a combined length of n.

             */
            public int UsingBinarySearch(int[] nums, int k)
            {
                int answer = Solve(nums, k);
                for (int i = 0; i < nums.Length / 2; i++)
                {
                    int temporary = nums[i];
                    nums[i] = nums[nums.Length - i - 1];
                    nums[nums.Length - i - 1] = temporary;
                }

                return Math.Max(answer, Solve(nums, nums.Length - k - 1));
            }

            private int Solve(int[] nums, int k)
            {
                int n = nums.Length;
                int[] left = new int[k];
                int currentMin = int.MaxValue;
                for (int i = k - 1; i >= 0; i--)
                {
                    currentMin = Math.Min(currentMin, nums[i]);
                    left[i] = currentMin;
                }

                List<int> right = new List<int>();
                currentMin = int.MaxValue;
                for (int i = k; i < n; i++)
                {
                    currentMin = Math.Min(currentMin, nums[i]);
                    right.Add(currentMin);
                }

                int answer = 0;
                for (int j = 0; j < right.Count; j++)
                {
                    currentMin = right[j];
                    int i = BinarySearch(left, currentMin);
                    int size = (k + j) - i + 1;
                    answer = Math.Max(answer, currentMin * size);
                }

                return answer;
            }

            private int BinarySearch(int[] nums, int num)
            {
                int left = 0;
                int right = nums.Length;

                while (left < right)
                {
                    int mid = (left + right) / 2;
                    if (nums[mid] < num)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }

                return left;
            }

            /* Approach 2: Monotonic Stack
            Complexity Analysis
Given n as the length of nums,
•	Time complexity: O(n)
It costs O(n) to calculate left and right. We iterate over each index once and perform amortized O(1) work at each iteration. The reason it amortizes to O(1), despite the while loop, is because the while loop can run a maximum of n times across all iterations, and each index can only be pushed onto and popped from the stack once.
To calculate ans, we iterate over the indices once and perform O(1) work at each iteration.
•	Space complexity: O(n)
left, right, and stack all require O(n) space.

             */
            public int UsingMononoticStack(int[] numbers, int k)
            {
                int length = numbers.Length;
                int[] leftIndices = new int[length];
                Array.Fill(leftIndices, -1);
                Stack<int> indexStack = new Stack<int>();

                for (int index = length - 1; index >= 0; index--)
                {
                    while (indexStack.Count > 0 && numbers[indexStack.Peek()] > numbers[index])
                    {
                        leftIndices[indexStack.Pop()] = index;
                    }

                    indexStack.Push(index);
                }

                int[] rightIndices = new int[length];
                Array.Fill(rightIndices, length);
                indexStack = new Stack<int>();

                for (int index = 0; index < length; index++)
                {
                    while (indexStack.Count > 0 && numbers[indexStack.Peek()] > numbers[index])
                    {
                        rightIndices[indexStack.Pop()] = index;
                    }

                    indexStack.Push(index);
                }

                int maximumResult = 0;
                for (int index = 0; index < length; index++)
                {
                    if (leftIndices[index] < k && rightIndices[index] > k)
                    {
                        maximumResult = Math.Max(maximumResult, numbers[index] * (rightIndices[index] - leftIndices[index] - 1));
                    }
                }

                return maximumResult;
            }

            /* Approach 3: Greedy
            Complexity Analysis
  Given n as the length of nums,
  •	Time complexity: O(n)
  At each iteration, our left or right pointers move closer to the edges of the array by 1. Thus, we perform O(n) iterations. Each iteration costs O(1).
  •	Space complexity: O(1)
  We aren't using any extra space other than a few integers.

             */
            public int UsingGreedy(int[] nums, int k)
            {
                int n = nums.Length;
                int left = k;
                int right = k;
                int ans = nums[k];
                int currMin = nums[k];

                while (left > 0 || right < n - 1)
                {
                    if ((left > 0 ? nums[left - 1] : 0) < (right < n - 1 ? nums[right + 1] : 0))
                    {
                        right++;
                        currMin = Math.Min(currMin, nums[right]);
                    }
                    else
                    {
                        left--;
                        currMin = Math.Min(currMin, nums[left]);
                    }

                    ans = Math.Max(ans, currMin * (right - left + 1));
                }

                return ans;
            }
        }

        /* 862. Shortest Subarray with Sum at Least K
        https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/
         */
        class ShortestSubarrayWithSumAtLeastKSol
        {

            /* Approach 1: Sliding Window
            Complexity Analysis
            •	Time Complexity: O(N), where N is the length of A.
            •	Space Complexity: O(N).

             */
            public int UsingSlidingWindow(int[] nums, int K)
            {
                int N = nums.Length;
                long[] P = new long[N + 1];
                for (int i = 0; i < N; ++i)
                    P[i + 1] = P[i] + (long)nums[i];

                // Want smallest y-x with P[y] - P[x] >= K
                int ans = N + 1; // N+1 is impossible
                LinkedList<int> monoq = new LinkedList<int>(); //opt(y) candidates, as indices of P

                for (int y = 0; y < P.Length; ++y)
                {
                    // Want opt(y) = largest x with P[x] <= P[y] - K;
                    while (monoq.Count > 0 && P[y] <= P[monoq.Last()])
                        monoq.RemoveLast();
                    while (monoq.Count > 0 && P[y] >= P[monoq.First()] + K)
                    {
                        ans = Math.Min(ans, y - monoq.First());
                        monoq.RemoveFirst();
                    }

                    monoq.AddLast(y);
                }

                return ans < N + 1 ? ans : -1;
            }
        }

        /* 1063. Number of Valid Subarrays
        https://leetcode.com/problems/number-of-valid-subarrays/description/
         */
        class NumberOfValidSubarraysSolution
        {
            /*             Approach: Monotonic Stack
            Complexity Analysis
            Here, N is the size of the array nums.
            •	Time complexity: O(N)
            We iterate over the elements in the array nums, and each element will be added to the stack only once and then popped from it. Hence the total time complexity would be O(N).
            •	Space complexity: O(N)
            The only space required is the stack which can have N elements in the worst-case scenario when the input is increasing, and hence the total space complexity will be equal to O(N).

             */

            public int UsingMononoticStack(int[] nums)
            {
                int ans = 0;

                Stack<int> st = new Stack<int>();
                for (int i = 0; i < nums.Length; i++)
                {
                    // Keep popping elements from the stack
                    // until the current element becomes greater than the top element.
                    while (st.Count > 0 && nums[i] < nums[st.Peek()])
                    {
                        // The diff between the current index and the stack top would be the subarray size.
                        // Which is equal to the number of subarrays.
                        ans += (i - st.Peek());
                        st.Pop();
                    }
                    st.Push(i);
                }

                // For all remaining elements, the last element will be considered as the right endpoint.
                while (st.Count > 0)
                {
                    ans += (nums.Length - st.Peek());
                    st.Pop();
                }

                return ans;
            }
        }

        /* 689. Maximum Sum of 3 Non-Overlapping Subarrays
        https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/description/
         */
        class MaxSumOfThreeSubarraysSol
        {

            /* Approach #1: Ad-Hoc [Accepted]
Complexity Analysis
•	Time Complexity: O(N), where N is the length of the array.
Every loop is bounded in the number of steps by N, and does O(1) work.
•	Space complexity: O(N). W, left, and right all take O(N) memory.

             */
            public int[] UsingAdhocApproach(int[] nums, int k)
            {
                // W is an array of sums of windows
                int[] W = new int[nums.Length - k + 1];
                int currSum = 0;
                for (int i = 0; i < nums.Length; i++)
                {
                    currSum += nums[i];
                    if (i >= k)
                    {
                        currSum -= nums[i - k];
                    }
                    if (i >= k - 1)
                    {
                        W[i - k + 1] = currSum;
                    }
                }

                int[] left = new int[W.Length];
                int best = 0;
                for (int i = 0; i < W.Length; i++)
                {
                    if (W[i] > W[best]) best = i;
                    left[i] = best;
                }

                int[] right = new int[W.Length];
                best = W.Length - 1;
                for (int i = W.Length - 1; i >= 0; i--)
                {
                    if (W[i] >= W[best])
                    {
                        best = i;
                    }
                    right[i] = best;
                }

                int[] ans = new int[] { -1, -1, -1 };
                for (int j = k; j < W.Length - k; j++)
                {
                    int i = left[j - k], l = right[j + k];
                    if (ans[0] == -1 || W[i] + W[j] + W[l] > W[ans[0]] + W[ans[1]] + W[ans[2]])
                    {
                        ans[0] = i;
                        ans[1] = j;
                        ans[2] = l;
                    }
                }
                return ans;
            }
        }


        /* 644. Maximum Average Subarray II
        https://leetcode.com/problems/maximum-average-subarray-ii/description/
         */

        public class FindMaxAverageSubarraySol
        {
            /*             Approach #1 Iterative method [Time Limit Exceeded]
            Complexity Analysis
            •	Time complexity : O(n^2). Two for loops iterating over the whole length of nums with n elements.
            •	Space complexity : O(1). Constant extra space is used.

             */
            public double NaiveIterative(int[] nums, int k)
            {
                double res = int.MaxValue;
                for (int s = 0; s < nums.Length - k + 1; s++)
                {
                    long sum = 0;
                    for (int i = s; i < nums.Length; i++)
                    {
                        sum += nums[i];
                        if (i - s + 1 >= k)
                            res = Math.Max(res, sum * 1.0 / (i - s + 1));
                    }
                }
                return res;
            }
            /*             Approach #2 Using Binary Search [Accepted]
Complexity Analysis
Let N be the number of element in the array, and range be the difference between the maximal and minimal values in the array, i.e. range = max_val - min_val, and finally the error be the precision required in the problem.
•	Time complexity : O(N⋅log2  ((max_val−min_val)/ 0.00001)).
o	The algorithm consists of a binary search loop in the function of findMaxAverage().
o	At each iteration of the loop, the check() function dominates the time complexity, which is of O(N) for each invocation.
o	It now boils down to how many iterations the loop would run eventually. To calculate the number of iterations, let us break it down in the following steps.
o	After the first iteration, the error would be range/2, as one can see. Further on, at each iteration, the error would be reduced into half. For example, after the second iteration, we would have the error as range/2⋅1/2.
o	As a result, after K iterations, the error would become error=range⋅2^(−K). Given the condition of the loop, i.e. error<0.00001, we can deduct that K>log2 (range/0.00001)=log2 ( (max_val−min_val)/ 0.00001)
o	To sum up, the time complexity of the algorithm would be O(N⋅K)=O(N⋅log2 ((max_val−min_val)/ 0.00001)).
•	Space complexity : O(1). Constant Space is used.

             */
            public double UsingBinarySearch(int[] nums, int k)
            {
                double max_val = int.MinValue;
                double min_val = int.MaxValue;
                foreach (int n in nums)
                {
                    max_val = Math.Max(max_val, n);
                    min_val = Math.Min(min_val, n);
                }
                double prev_mid = max_val, error = int.MaxValue;
                while (error > 0.00001)
                {
                    double mid = (max_val + min_val) * 0.5;
                    if (Check(nums, mid, k))
                        min_val = mid;
                    else
                        max_val = mid;
                    error = Math.Abs(prev_mid - mid);
                    prev_mid = mid;
                }
                return min_val;
            }
            public bool Check(int[] nums, double mid, int k)
            {
                double sum = 0, prev = 0, min_sum = 0;
                for (int i = 0; i < k; i++)
                    sum += nums[i] - mid;
                if (sum >= 0)
                    return true;
                for (int i = k; i < nums.Length; i++)
                {
                    sum += nums[i] - mid;
                    prev += nums[i - k] - mid;
                    min_sum = Math.Min(prev, min_sum);
                    if (sum >= min_sum)
                        return true;
                }
                return false;
            }
        }

        /* 560. Subarray Sum Equals K
        https://leetcode.com/problems/subarray-sum-equals-k/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class SubarraySumEqualsKSol
        {
            /*           Approach 1: Brute Force
            Complexity Analysis
            •	Time complexity : O(n^3). Considering every possible subarray takes O(n^2) time. For each of the subarray we calculate the sum taking O(n) time in the worst case, taking a total of O(n^3) time.
            •	Space complexity : O(1). Constant space is used.

             */
            public int Naive(int[] nums, int k)
            {
                int count = 0;
                for (int start = 0; start < nums.Length; start++)
                {
                    for (int end = start + 1; end <= nums.Length; end++)
                    {
                        int sum = 0;
                        for (int i = start; i < end; i++)
                            sum += nums[i];
                        if (sum == k)
                            count++;
                    }
                }
                return count;
            }
            /*             Approach 2: Using Cumulative Sum
Complexity Analysis
•	Time complexity : O(n^2). Considering every possible subarray takes O(n^2) time. Finding out the sum of any subarray takes O(1) time after the initial processing of O(n) for creating the cumulative sum array.
•	Space complexity : O(n). Cumulative sum array sum of size n+1 is used.

             */
            public int UsingCumulativeSum(int[] nums, int k)
            {
                int count = 0;
                int[] sum = new int[nums.Length + 1];
                sum[0] = 0;
                for (int i = 1; i <= nums.Length; i++)
                    sum[i] = sum[i - 1] + nums[i - 1];
                for (int start = 0; start < nums.Length; start++)
                {
                    for (int end = start + 1; end <= nums.Length; end++)
                    {
                        if (sum[end] - sum[start] == k)
                            count++;
                    }
                }
                return count;
            }
            /* Approach 3: Without Auxiliary Space 
            Complexity Analysis
•	Time complexity : O(n^2). We need to consider every subarray possible.
•	Space complexity : O(1). Constant space is used.

            */
            public int SpaceOptimal(int[] nums, int k)
            {
                int count = 0;
                for (int start = 0; start < nums.Length; start++)
                {
                    int sum = 0;
                    for (int end = start; end < nums.Length; end++)
                    {
                        sum += nums[end];
                        if (sum == k)
                            count++;
                    }
                }
                return count;
            }
            /*             Approach 4: Using Hashmap
            Complexity Analysis
            •	Time complexity : O(n). The entire nums array is traversed only once.
            •	Space complexity : O(n). Hashmap map can contain up to n distinct entries in the worst case.

             */
            public int UsingDict(int[] nums, int k)
            {
                int count = 0, sum = 0;
                Dictionary<int, int> sumOccurrencesMap = new();
                sumOccurrencesMap.Add(0, 1);
                for (int i = 0; i < nums.Length; i++)
                {
                    sum += nums[i];
                    if (sumOccurrencesMap.ContainsKey(sum - k))
                        count += sumOccurrencesMap[sum - k];

                    sumOccurrencesMap[sum] = sumOccurrencesMap.GetValueOrDefault(sum, 0) + 1;
                }
                return count;
            }

        }
        /* 
        523. Continuous Subarray Sum
        https://leetcode.com/problems/continuous-subarray-sum/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class CheckContinuousSubarraySumSolution
        {/* 
            Approach 1: Prefix Sum and Hashing
            Complexity Analysis
Let n be the number of elements in nums.
•	Time complexity: O(n)
We iterate through the array exactly once. In each iteration, we perform a search operation in the hashmap that takes O(1) time. Therefore, the time complexity can be stated as O(n).
•	Space complexity: O(n)
In each iteration, we insert a key-value pair in the hashmap. The space complexity is O(n) because the size of the hashmap is proportional to the size of the list after n iterations.

             */
            public bool CheckSubarraySum(int[] numbers, int divisor)
            {
                int prefixModulus = 0;
                Dictionary<int, int> modulusSeen = new Dictionary<int, int>();
                modulusSeen.Add(0, -1);

                for (int index = 0; index < numbers.Length; index++)
                {
                    prefixModulus = (prefixModulus + numbers[index]) % divisor;

                    if (modulusSeen.ContainsKey(prefixModulus))
                    {
                        // ensures that the size of subarray is at least 2
                        if (index - modulusSeen[prefixModulus] > 1)
                        {
                            return true;
                        }
                    }
                    else
                    {
                        // mark the value of prefixModulus with the current index.
                        modulusSeen.Add(prefixModulus, index);
                    }
                }

                return false;
            }
        }

        /* 209. Minimum Size Subarray Sum
        https://leetcode.com/problems/minimum-size-subarray-sum/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class MinSubArrayLenSol
        {
            public int MinSubArrayLen(int target, int[] nums)
            {
                int left = 0, right = 0, sumOfCurrentWindow = 0;
                int res = int.MaxValue;

                for (right = 0; right < nums.Length; right++)
                {
                    sumOfCurrentWindow += nums[right];

                    while (sumOfCurrentWindow >= target)
                    {
                        res = Math.Min(res, right - left + 1);
                        sumOfCurrentWindow -= nums[left++];
                    }
                }

                return res == int.MaxValue ? 0 : res;
            }
        }


        /* 325. Maximum Size Subarray Sum Equals k
        https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class MaxSubArraySumEqualsKLenSolution
        {
            /* Approach: Prefix Sum + Hash Map
            Complexity Analysis
Given N as the length of nums,
•	Time complexity: O(N)
We only make one pass through nums, each time doing a constant amount of work. All hash map operations are O(1).
•	Space complexity: O(N)
Our hash map can potentially hold as many key-value pairs as there are numbers in nums. An example of this is when there are no negative numbers in the array.

             */
            public int UsingPrefixSumWithDict(int[] nums, int k)
            {
                int prefixSum = 0;
                int longestSubarray = 0;
                Dictionary<int, int> indices = new();
                for (int i = 0; i < nums.Length; i++)
                {
                    prefixSum += nums[i];

                    // Check if all of the numbers seen so far sum to k.
                    if (prefixSum == k)
                    {
                        longestSubarray = i + 1;
                    }

                    // If any subarray seen so far sums to k, then
                    // update the length of the longest_subarray. 
                    if (indices.ContainsKey(prefixSum - k))
                    {
                        longestSubarray = Math.Max(longestSubarray, i - indices[prefixSum - k]);
                    }

                    // Only add the current prefix_sum index pair to the 
                    // map if the prefix_sum is not already in the map.
                    if (!indices.ContainsKey(prefixSum))
                    {
                        indices[prefixSum] = i;
                    }
                }

                return longestSubarray;
            }
        }

        /* 1508. Range Sum of Sorted Subarray Sums
        https://leetcode.com/problems/range-sum-of-sorted-subarray-sums/description/
         */
        public class RangeSumOfSortedSubarraySumsSol
        {
            /* Approach 1: Brute Force
            Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(n^2⋅logn)
We iterate through nums twice to store all the subarray sums. This operation takes O(n^2) time. Then, we sort this array storing all the subarray sums. The time complexity for this operation is O(n^2⋅logn). Iterating all indices between left and right also takes O(n^2) time in the worst case.
Therefore, the total time complexity is given by O(n^2⋅logn).
•	Space complexity: O(n^2)
We create a storeSubarray array with size proportional to O(n^2). Apart from this, no additional memory is used.
Therefore, the total space complexity is given by O(n^2).

             */
            public int Naive(int[] nums, int n, int left, int right)
            {
                List<int> storeSubarray = new List<int>();
                for (int i = 0; i < nums.Length; i++)
                {
                    int sum = 0;
                    for (int j = i; j < nums.Length; j++)
                    {
                        sum += nums[j];
                        storeSubarray.Add(sum);
                    }
                }
                storeSubarray.Sort();

                int rangeSum = 0, mod = (int)1e9 + 7;
                for (int i = left - 1; i <= right - 1; i++)
                {
                    rangeSum = (rangeSum + storeSubarray[i]) % mod;
                }
                return rangeSum;
            }
            /*             Approach 2: Priority Queue
Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(n^2⋅logn)
We iterate through nums once to store all the one-sized subarray sums. This operation takes O(n) time. Then, we iterate all indices between left and right, performing pop operation in each iteration, which takes O(n^2⋅logn) time total in the worst case.
Therefore, the total time complexity is given by O(n^2⋅logn).
•	Space complexity: O(n)
The size of pq never exceeds n. Apart from this, no additional memory is used.
Therefore, the total space complexity is given by O(n).

             */
            public int UsingPQ(int[] nums, int n, int left, int right)
            {
                PriorityQueue<int[], int[]> pq = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => a[0] - b[0]));
                for (int i = 0; i < n; i++)
                {
                    pq.Enqueue(new int[] { nums[i], i }, new int[] { nums[i], i });
                }

                int ans = 0, mod = 1000000007;
                for (int i = 1; i <= right; i++)
                {
                    int[] p = pq.Dequeue();
                    if (i >= left) ans = (ans + p[0]) % mod;
                    if (p[1] < n - 1)
                    {
                        p[0] += nums[++p[1]];
                        pq.Enqueue(p, p);
                    }
                }
                return ans;
            }

            /* Approach 3: Binary Search and Sliding Window
Complexity Analysis
Let n be the size and sum be the total sum of the nums array.
•	Time complexity: O(n log sum)
The total size of the search space is O(sum). Therefore, time complexity for binary search is O(logsum). Inside each binary search operation, the countAndSum function takes O(n) time.
Therefore, the total time complexity is given by O(n⋅logsum).
•	Space complexity: O(1)
Apart from some constant sized variables, no additional memory is used. Therefore, the total space complexity is given by O(n).	

             */
            private const int MOD = 1000000007;

            public int UsingBinarySearchAndSlidingWindow(int[] nums, int n, int left, int right)
            {
                long result = (SumOfFirstK(nums, n, right) - SumOfFirstK(nums, n, left - 1)) % MOD;
                return (int)((result + MOD) % MOD);
            }

            private KeyValuePair<int, long> CountAndSum(int[] nums, int n, int target)
            {
                int count = 0;
                long currentSum = 0, totalSum = 0, windowSum = 0;
                for (int j = 0, i = 0; j < n; ++j)
                {
                    currentSum += nums[j];
                    windowSum += nums[j] * (j - i + 1);
                    while (currentSum > target)
                    {
                        windowSum -= currentSum;
                        currentSum -= nums[i++];
                    }
                    count += j - i + 1;
                    totalSum += windowSum;
                }
                return new KeyValuePair<int, long>(count, totalSum);
            }

            private long SumOfFirstK(int[] nums, int n, int k)
            {
                int minSum = nums.Min();
                int maxSum = nums.Sum();
                int left = minSum, right = maxSum;

                while (left <= right)
                {
                    int mid = left + (right - left) / 2;
                    if (CountAndSum(nums, n, mid).Key >= k) right = mid - 1;
                    else left = mid + 1;
                }
                KeyValuePair<int, long> result = CountAndSum(nums, n, left);
                return result.Value - left * (result.Key - k);
            }


        }

        /* 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
        https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/description/	
         */
        public class LongestContinuousSubarrayWithAbsoluteDiffLessThanOrEqualToLimitSol
        {
            /* Approach 1: Two Heaps 
Complexity Analysis
Let n be the length of the array nums.
•	Time Complexity: O(n⋅logn)
Initializing the two heaps takes O(1) time.
Iterating through the array nums from left to right involves a single loop that runs n times.
Adding each element to the heaps takes O(logn) time per operation due to the properties of heaps. Over the entire array, this results in O(n⋅logn) time for both heaps combined.
Checking the condition and potentially shrinking the window involves comparing the top elements of the heaps and moving the left pointer. Removing elements from the heaps that are outside the current window also takes O(logn) time per operation. Over the entire array, this results in O(n⋅logn) time.
Updating the maxLength variable involves a simple comparison and assignment, each taking O(1) time per iteration. Over the entire array, this takes O(n) time.
Therefore, the total time complexity is O(n⋅logn).
•	Space Complexity: O(n)
The two heaps, maxHeap and minHeap, store elements of the array along with their indices. In the worst case, each heap could store all n elements of the array.
The additional variables left, right, and maxLength use constant space.
Therefore, the space complexity is O(n) due to the heaps storing up to n elements in the worst case.	

            */

            public int UsingTwoHeaps(int[] nums, int limit)
            {
                PriorityQueue<int[], int[]> maxHeap = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => b[0] - a[0]));
                PriorityQueue<int[], int[]> minHeap = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => a[0] - b[0]));

                int left = 0, maxLength = 0;

                for (int right = 0; right < nums.Length; ++right)
                {
                    maxHeap.Enqueue(new int[] { nums[right], right }, new int[] { nums[right], right });
                    minHeap.Enqueue(new int[] { nums[right], right }, new int[] { nums[right], right });

                    // Check if the absolute difference between the maximum and minimum values in the current window exceeds the limit
                    while (maxHeap.Peek()[0] - minHeap.Peek()[0] > limit)
                    {
                        // Move the left pointer to the right until the condition is satisfied.
                        // This ensures we remove the element causing the violation
                        left = Math.Min(maxHeap.Peek()[1], minHeap.Peek()[1]) + 1;

                        // Remove elements from the heaps that are outside the current window
                        while (maxHeap.Peek()[1] < left)
                        {
                            maxHeap.Dequeue();
                        }
                        while (minHeap.Peek()[1] < left)
                        {
                            minHeap.Dequeue();
                        }
                    }
                    // Update maxLength with the length of the current valid window
                    maxLength = Math.Max(maxLength, right - left + 1);
                }

                return maxLength;
            }
            /* Approach 2: Multiset/SortedDict
            Complexity Analysis
Let n be the length of the array nums.
•	Time Complexity: O(n⋅logn)
Initializing the multiset takes O(1) time.
Iterating through the array nums from left to right involves a single loop that runs n times.
Adding each element to the multiset takes O(logn) time per operation due to the properties of the balanced tree. Over the entire array, this results in O(n⋅logn) time.
Checking the condition and potentially shrinking the window involves comparing the maximum and minimum values in the multiset and moving the left pointer. Removing elements from the multiset that are outside the current window also takes O(logn) time per operation. Over the entire array, this results in O(n⋅logn) time.
Updating the maxLength variable involves a simple comparison and assignment, each taking O(1) time per iteration. Over the entire array, this takes O(n) time.
Therefore, the total time complexity is O(n⋅logn).
•	Space Complexity: O(n)
The multiset stores elements of the array. In the worst case, the multiset could store all n elements of the array.
The additional variables left, right, and maxLength use constant space.
Therefore, the space complexity is O(n) due to the multiset storing up to n elements in the worst case.

             */
            public int UsingSortedDict(int[] numbers, int limit)
            {
                // SortedDictionary to maintain the elements within the current window
                SortedDictionary<int, int> window = new SortedDictionary<int, int>();
                int leftPointer = 0, maximumLength = 0;

                for (int rightPointer = 0; rightPointer < numbers.Length; ++rightPointer)
                {
                    if (window.ContainsKey(numbers[rightPointer]))
                    {
                        window[numbers[rightPointer]]++;
                    }
                    else
                    {
                        window[numbers[rightPointer]] = 1;
                    }

                    // Check if the absolute difference between the maximum and minimum values in the current window exceeds the limit
                    while (window.Keys.Max() - window.Keys.Min() > limit)
                    {
                        // Remove the element at the left pointer from the SortedDictionary
                        window[numbers[leftPointer]]--;
                        if (window[numbers[leftPointer]] == 0)
                        {
                            window.Remove(numbers[leftPointer]);
                        }
                        // Move the left pointer to the right to exclude the element causing the violation
                        ++leftPointer;
                    }

                    // Update maximumLength with the length of the current valid window
                    maximumLength = Math.Max(maximumLength, rightPointer - leftPointer + 1);
                }

                return maximumLength;
            }
            /* Approach 3: Two Deques
Complexity Analysis
Let n be the length of the array nums.
•	Time Complexity: O(n)
Initializing the two deques, maxDeque and minDeque, takes O(1) time.
Iterating through the array nums from left to right involves a single loop that runs n times.
Maintaining maxDeque and minDeque involves adding and removing elements. Each element can be added and removed from the deques at most once, resulting in O(1) time per operation. Over the entire array, this results in O(n) time for both deques combined.
Checking the condition and potentially shrinking the window involves deque operations, which each take O(1) time. Over the entire array, this takes O(n) time.
Updating the maxLength variable involves a simple comparison and assignment, each taking O(1) time per iteration. Over the entire array, this takes O(n) time.
Therefore, the total time complexity is O(n).
•	Space Complexity: O(n)
The two deques, maxDeque and minDeque, store elements of the array. In the worst case, each deque could store all n elements of the array.
The additional variables left, right, and maxLength use constant space.
Therefore, the space complexity is O(n) due to the deques storing up to n elements in the worst case.

             */
            public int UsingTwoDeques(int[] nums, int limit)
            {
                LinkedList<int> maxDeque = new LinkedList<int>();
                LinkedList<int> minDeque = new LinkedList<int>();
                int left = 0;
                int maxLength = 0;

                for (int right = 0; right < nums.Length; ++right)
                {
                    // Maintain the maxDeque in decreasing order
                    while (maxDeque.Count != 0 && maxDeque.Last() < nums[right])
                    {
                        maxDeque.RemoveLast();
                    }
                    maxDeque.AddLast(nums[right]);

                    // Maintain the minDeque in increasing order
                    while (minDeque.Count != 0 && minDeque.Last() > nums[right])
                    {
                        minDeque.RemoveLast();
                    }
                    minDeque.AddLast(nums[right]);

                    // Check if the current window exceeds the limit
                    while (maxDeque.First() - minDeque.First() > limit)
                    {
                        // Remove the elements that are out of the current window
                        if (maxDeque.First() == nums[left])
                        {
                            maxDeque.RemoveFirst();
                        }
                        if (minDeque.First() == nums[left])
                        {
                            minDeque.RemoveFirst();
                        }
                        ++left;
                    }

                    maxLength = Math.Max(maxLength, right - left + 1);
                }

                return maxLength;
            }

        }


        /* 974. Subarray Sums Divisible by K
        https://leetcode.com/problems/subarray-sums-divisible-by-k/description/
         */
        class SubarraySumDivByKSolution
        {

            /*             Approach: Prefix Sums and Counting
            Complexity Analysis
            Here, n is the length of nums and k is the given integer.
            •	Time complexity: O(n+k)
            o	We require O(k) time to initialize the modGroups array.
            o	We also require O(n) time to iterate over all the elements of the nums array. The computation of the prefixSum and the calculation of the subarrays divisible by k take O(1) time for each index of the array.
            •	Space complexity: O(k)
            o	We require O(k) space for the modGroups array.

             */
            public int UsingPrefixSumAndCounting(int[] nums, int k)
            {
                int n = nums.Length;
                int prefixMod = 0, result = 0;

                // There are k mod groups 0...k-1.
                int[] modGroups = new int[k];
                modGroups[0] = 1;

                foreach (int num in nums)
                {
                    // Take modulo twice to avoid negative remainders.
                    prefixMod = (prefixMod + num % k + k) % k;
                    // Add the count of subarrays that have the same remainder as the current
                    // one to cancel out the remainders.
                    result += modGroups[prefixMod];
                    modGroups[prefixMod]++;
                }

                return result;
            }
        }

        /* 152. Maximum Product Subarray
        https://leetcode.com/problems/maximum-product-subarray/description/
         */

        class MaxProductSubarraySol
        {

            /* Approach 1: Brute Force (Python TLE)
Complexity Analysis
•	Time complexity : O(N^2) where N is the size of nums. Since we are checking every possible contiguous subarray following every element in nums we have quadratic runtime.
•	Space complexity : O(1) since we are not consuming additional space other than two variables: result to hold the final result and accu to accumulate product of preceding contiguous subarrays.

             */
            public int Naive(int[] nums)
            {
                if (nums.Length == 0) return 0;

                int result = nums[0];

                for (int i = 0; i < nums.Length; i++)
                {
                    int accu = 1;
                    for (int j = i; j < nums.Length; j++)
                    {
                        accu *= nums[j];
                        result = Math.Max(result, accu);
                    }
                }

                return result;
            }
            /* Approach 2: Dynamic Programming
Complexity Analysis
•	Time complexity : O(N) where N is the size of nums. The algorithm achieves linear runtime since we are going through nums only once.
•	Space complexity : O(1) since no additional space is consumed rather than variables which keep track of the maximum product so far, the minimum product so far, current variable, temp variable, and placeholder variable for the result.

             */
            public int UsingDP(int[] nums)
            {
                if (nums.Length == 0) return 0;

                int max_so_far = nums[0];
                int min_so_far = nums[0];
                int result = max_so_far;

                for (int i = 1; i < nums.Length; i++)
                {
                    int curr = nums[i];
                    int temp_max = Math.Max(
                        curr,
                        Math.Max(max_so_far * curr, min_so_far * curr)
                    );
                    min_so_far = Math.Min(
                        curr,
                        Math.Min(max_so_far * curr, min_so_far * curr)
                    );

                    // Update max_so_far after min_so_far to avoid overwriting it
                    max_so_far = temp_max;
                    // Update the result with the maximum product found so far
                    result = Math.Max(max_so_far, result);
                }

                return result;
            }

        }

        /* 1248. Count Number of Nice Subarrays
        https://leetcode.com/problems/count-number-of-nice-subarrays/description/
         */

        class CountNumberOfNiceSubarraysSol
        {
            /* Approach 1: Hashing
Complexity Analysis
Let n be the number of elements in nums.
•	Time complexity: O(n)
We iterate through the array exactly once. In each iteration, we perform insertion and search operations in the hashmap that take O(1) time. Therefore, the time complexity can be stated as O(n).
•	Space complexity: O(n)
In each iteration, we insert a key-value pair in the hashmap. The space complexity is O(n) because the size of the hashmap is proportional to the size of the list after n iterations.

             */
            public int UsingDict(int[] nums, int k)
            {
                int currSum = 0, subarrays = 0;
                Dictionary<int, int> prefixSum = new();
                prefixSum.Add(currSum, 1);

                for (int i = 0; i < nums.Length; i++)
                {
                    currSum += nums[i] % 2;
                    // Find subarrays with sum k ending at i
                    if (prefixSum.ContainsKey(currSum - k))
                    {
                        subarrays = subarrays + prefixSum[currSum - k];
                    }
                    // Increment the current prefix sum in hashmap
                    prefixSum[currSum] = prefixSum.GetValueOrDefault(currSum, 0) + 1;
                }

                return subarrays;
            }
            /* Approach 2: Sliding Window using Queue
            Complexity Analysis
Let n be the number of elements in the array nums.
•	Time complexity: O(n)
We iterate through the array exactly once. In each iteration of the array, we perform queue operations such as push, pop, and accessing the front element that takes O(1) time. Therefore, the time complexity can be stated as O(n).
•	Space complexity: O(n)
In each iteration, we perform one push operation in the queue. The space complexity is O(n) because the queue size is proportional to the size of the list after n iterations.
             */
            public int UsingSlidingWindowWithQueue(int[] nums, int k)
            {
                Queue<int> oddIndices = new();
                int subarrays = 0;
                int lastPopped = -1;
                int initialGap = 0;

                for (int i = 0; i < nums.Length; i++)
                {
                    // If element is odd, append its index to the list.
                    if (nums[i] % 2 == 1)
                    {
                        oddIndices.Enqueue(i);
                    }
                    // If the number of odd numbers exceeds k, remove the first odd index.
                    if (oddIndices.Count > k)
                    {
                        lastPopped = oddIndices.Dequeue();
                    }
                    // If there are exactly k odd numbers, add the number of even numbers
                    // in the beginning of the subarray to the result.
                    if (oddIndices.Count == k)
                    {
                        initialGap = oddIndices.Peek() - lastPopped;
                        subarrays += initialGap;
                    }
                }

                return subarrays;
            }
            /* Approach 3: Sliding Window (Space Optimisation of queue-based approach) 
Complexity Analysis
Let n be the number of elements in the array nums.
•	Time complexity: O(n)
We iterate through the array exactly once. The start pointer can move at most n steps through all iterations. Therefore, the time complexity can be stated as O(n).
•	Space complexity: O(1)
We do not allocate any additional auxiliary memory in our algorithm. Therefore, overall space complexity is given by O(1).	

            */
            public int UsingSlidingWindowOptimal(int[] nums, int k)
            {
                int subarrays = 0, initialGap = 0, qsize = 0, start = 0;
                for (int end = 0; end < nums.Length; end++)
                {
                    // If current element is odd, increment qsize by 1.
                    if (nums[end] % 2 == 1)
                    {
                        qsize++;
                    }
                    // If qsize is k, calculate initialGap and add it in the answer.
                    if (qsize == k)
                    {
                        initialGap = 0;
                        // Calculate the number of even elements in the beginning of
                        // subarray.
                        while (qsize == k)
                        {
                            qsize -= nums[start] % 2;
                            initialGap++;
                            start++;
                        }
                    }
                    subarrays += initialGap;
                }
                return subarrays;
            }
            /*             
            Approach 4: Sliding Window (subarray sum at most k)
Complexity Analysis
Let n be the number of elements in the array nums.
•	Time complexity: O(n)
We call the atMost function 2 times. We iterate through the array exactly once in the function. The start pointer can move atmost n steps through all iterations. Therefore, the time complexity can be stated as O(n).
•	Space complexity: O(1)

We do not allocate any additional auxiliary memory in our algorithm. Therefore, overall space complexity is given by O(1).


            */
            public int UsingSlidingWindowAtMostK(int[] nums, int k)
            {
                return AtMost(nums, k) - AtMost(nums, k - 1);
            }

            private int AtMost(int[] nums, int k)
            {
                int windowSize = 0, subarrays = 0, start = 0;

                for (int end = 0; end < nums.Length; end++)
                {
                    windowSize += nums[end] % 2;
                    // Find the first index start where the window has exactly k odd elements.
                    while (windowSize > k)
                    {
                        windowSize -= nums[start] % 2;
                        start++;
                    }
                    // Increment number of subarrays with end - start + 1.
                    subarrays += end - start + 1;
                }

                return subarrays;
            }

        }

        /* 2419. Longest Subarray With Maximum Bitwise AND
        https://leetcode.com/problems/longest-subarray-with-maximum-bitwise-and/description/	
         */
        class LongestSubarrayWithMaxBitWiseANDSol
        {
            /* Approach: Longest consecutive sequence of the maximum value
            Implementation
Let N be the length of nums.
•	Time Complexity: O(N)
The time complexity is O(N) because the function processes each element of the nums list exactly once. This is done through a single loop that iterates over the array. Each operation inside the loop—whether it's comparisons, assignments, or finding the maximum—takes constant time. As a result, the total time required scales linearly with the size of the input array.
•	Space Complexity: O(1)
The function uses a fixed amount of extra space regardless of the size of the input array nums. Specifically, it only requires a few variables (max_val, ans, current_streak, and num) to keep track of intermediate values. This fixed space usage means the space complexity remains constant.

             */

            public int LongestSubarray(int[] nums)
            {
                int maxVal = 0, ans = 0, currentStreak = 0;

                foreach (int num in nums)
                {
                    if (maxVal < num)
                    {
                        maxVal = num;
                        ans = currentStreak = 0;
                    }

                    if (maxVal == num)
                    {
                        currentStreak++;
                    }
                    else
                    {
                        currentStreak = 0;
                    }

                    ans = Math.Max(ans, currentStreak);
                }
                return ans;
            }
        }


        /* 907. Sum of Subarray Minimums
        https://leetcode.com/problems/sum-of-subarray-minimums/description/
         */
        class SumSubarrayMinsSol
        {
            /* Approach 1: Monotonic Stack - Contribution of Each Element
            Complexity Analysis
With n elements in the given array arr -
•	Time complexity: O(n). While building a monotonic stack, each element is pushed in once and popped out once. Every time an item is popped, we calculate the contribution of that item. All of these are constant time operations which are done n times. So the time complexity is O(n).
•	Space complexity: O(n). In the worst-case scenario, where the elements are in increasing order. The stack would contain all the elements. So the space requirement grows linearly with the size of the input. It is O(n).

             */
            public int UsingMononoticStack(int[] arr)
            {
                int MOD = 1000000007;

                Stack<int> stack = new Stack<int>();
                long sumOfMinimums = 0;

                // building monotonically increasing stack
                for (int i = 0; i <= arr.Length; i++)
                {

                    // when i reaches the array length, it is an indication that
                    // all the elements have been processed, and the remaining
                    // elements in the stack should now be popped out.

                    while (stack.Count > 0 && (i == arr.Length || arr[stack.Peek()] >= arr[i]))
                    {

                        // Notice the sign ">=", This ensures that no contribution
                        // is counted twice. rightBoundary takes equal or smaller 
                        // elements into account while leftBoundary takes only the
                        // strictly smaller elements into account

                        int mid = stack.Pop();
                        int leftBoundary = stack.Count == 0 ? -1 : stack.Peek();
                        int rightBoundary = i;

                        // count of subarrays where mid is the minimum element
                        long count = (mid - leftBoundary) * (rightBoundary - mid) % MOD;

                        sumOfMinimums += (count * arr[mid]) % MOD;
                        sumOfMinimums %= MOD;
                    }
                    stack.Push(i);
                }

                return (int)(sumOfMinimums);
            }
            /* Approach 2: Monotonic Stack + Dynamic Programming
            Complexity Analysis
With n elements in the given array arr -
•	Time complexity: O(n). Creating a monotonic stack takes O(n) time. As we build the monotonic stack, we fill the dp array at the same time. Filling the dp array for each element takes constant time, so for the all the items, it'd be O(n). In the end, we take the sum of all elements of the dp array, which also takes O(n). So the upper bound always remains under O(n).
•	Space complexity: O(n). We use two external data structures - dp array occupies O(n) space, stack can take O(n) space in the worst case scenario. So, the program requires O(2n) space. Upon removing the constant factor 2, we get O(n) as the final space complexity.

             */
            public int UsingMononoticStackAndDP(int[] arr)
            {
                int MOD = 1000000007;

                Stack<int> stack = new Stack<int>();

                // make a dp array of the same size as the input array `arr`
                int[] dp = new int[arr.Length];

                // making a monotonic increasing stack
                for (int i = 0; i < arr.Length; i++)
                {
                    // pop the stack until it is empty or
                    // the top of the stack is greater than or equal to
                    // the current element
                    while (stack.Count > 0 && arr[stack.Peek()] >= arr[i])
                    {
                        stack.Pop();
                    }

                    // either the previousSmaller element exists
                    if (stack.Count > 0)
                    {
                        int previousSmaller = stack.Peek();
                        dp[i] = dp[previousSmaller] + (i - previousSmaller) * arr[i];
                    }
                    else
                    {
                        // or it doesn't exist, in this case the current element
                        // contributes with all subarrays ending at i
                        dp[i] = (i + 1) * arr[i];
                    }
                    // push the current index
                    stack.Push(i);
                }

                // Add all elements of the dp to get the answer
                long sumOfMinimums = 0;
                foreach (int count in dp)
                {
                    sumOfMinimums += count;
                    sumOfMinimums %= MOD;
                }

                return (int)sumOfMinimums;
            }

        }


        /* 2461. Maximum Sum of Distinct Subarrays With Length K
        https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/description/
        https://algo.monster/liteproblems/2461
         */
        class MaximumSumOfDistinctSubarraysWithLengthEqualKSol
        {
            /* Time and Space Complexity
The given Python code defines a method maximumSubarraySum within a class Solution to calculate the maximum sum of a subarray of size k with unique elements. The code uses a sliding window approach by keeping a counter for the number of occurrences of each element within the current window of size k and computes the sum of elements in the window.
Time Complexity:
The time complexity of the algorithm is O(n), where n is the total number of elements in the input list nums. This is because the code iterates through all the elements of nums once. For each element in the iteration, the time to update the cnt counter (Counter from the collections module) for the current window is constant on average due to the hash map data structure used internally. The operations inside the loop including incrementing, decrementing, deleting from the counter, and computing the sum are done in constant time. Since these operations are repeated for every element just once, it amounts to O(n).
Space Complexity:
The space complexity of the algorithm is O(k). The cnt counter maintains the count of elements within a sliding window of size k. In the worst-case scenario, if all elements within the window are unique, the counter will hold k key-value pairs. Hence, the amount of extra space used is proportional to the size of the window k.
 */
            public long MaximumSubarraySum(int[] nums, int k)
            {
                int arrayLength = nums.Length;   // Store the length of input array nums
                                                 // Create a Dictionary to count the occurrences of each number in a subarray of size k
                Dictionary<int, int> countDictionary = new Dictionary<int, int>();
                long currentSum = 0;  // Initialize sum of elements in the current subarray

                // Traverse the first subarray of size k and initialize the countDictionary and sum
                for (int i = 0; i < k; ++i)
                {
                    if (countDictionary.ContainsKey(nums[i]))
                    {
                        countDictionary[nums[i]]++;
                    }
                    else
                    {
                        countDictionary[nums[i]] = 1;
                    }
                    currentSum += nums[i];
                }

                // Initialize the answer with the sum of the first subarray if all elements are unique
                long maxSum = countDictionary.Count == k ? currentSum : 0;

                // Loop over the rest of the array, updating subarrays and checking for maximum sum
                for (int i = k; i < arrayLength; ++i)
                {
                    // Add current element to the countDictionary and update the sum
                    if (countDictionary.ContainsKey(nums[i]))
                    {
                        countDictionary[nums[i]]++;
                    }
                    else
                    {
                        countDictionary[nums[i]] = 1;
                    }
                    currentSum += nums[i];

                    // Remove the element that's k positions behind the current one from countDictionary and update the sum
                    if (countDictionary.ContainsKey(nums[i - k]))
                    {
                        countDictionary[nums[i - k]]--;
                        if (countDictionary[nums[i - k]] == 0)
                        {
                            countDictionary.Remove(nums[i - k]);
                        }
                    }
                    currentSum -= nums[i - k];

                    // Update maxSum if the countDictionary indicates that we have a subarray with k unique elements
                    if (countDictionary.Count == k)
                    {
                        maxSum = Math.Max(maxSum, currentSum);
                    }
                }

                // Return the maximum sum found
                return maxSum;
            }
        }

        /* 930. Binary Subarrays With Sum
           https://leetcode.com/problems/binary-subarrays-with-sum/description/
            */
        class CountBinarySubarraysWithSumSol
        {
            /* Approach 1: Prefix Sum 
            Complexity Analysis
Let n be the length of the input array nums.
•	Time complexity: O(n)
We iterate through the array once to calculate the prefix sums and update the frequency map.
•	Space complexity: O(n)
We use an unordered map (freq) to store the frequency of prefix sums. In the worst case, all prefix sums can be distinct, resulting in n unique entries in the map. Therefore, the space required is proportional to the size of the input array.

            */
            public int UsingPrefixSum(int[] numbers, int targetSum)
            {
                int totalCount = 0;
                int currentSum = 0;
                // {prefix: number of occurrence}
                Dictionary<int, int> prefixSumFrequency = new Dictionary<int, int>(); // To store the frequency of prefix sums

                foreach (int number in numbers)
                {
                    currentSum += number;
                    if (currentSum == targetSum)
                    {
                        totalCount++;
                    }

                    // Check if there is any prefix sum that can be subtracted from the current sum to get the desired target sum
                    if (prefixSumFrequency.ContainsKey(currentSum - targetSum))
                    {
                        totalCount += prefixSumFrequency[currentSum - targetSum];
                    }

                    if (prefixSumFrequency.ContainsKey(currentSum))
                    {
                        prefixSumFrequency[currentSum]++;
                    }
                    else
                    {
                        prefixSumFrequency[currentSum] = 1;
                    }
                }

                return totalCount;
            }

            /* Approach 2: Sliding Window
            Complexity Analysis
Let n be the length of the nums array.
•	Time complexity: O(n)
The function slidingWindowAtMost uses two pointers, start and end to process the elements in the array. Although there is a nested loop, each pointer starts at 0 and gets incremented at most n times, so each pointer makes just 1 pass through the array. This means the time complexity of the function slidingWindowAtMost is O(n). We call slidingWindowAtMost twice, resulting in an overall time complexity of O(n).
•	Space complexity: O(1)
The space complexity is O(1) because the algorithm uses a constant amount of space for variables such as start, currentSum, and totalCount. The space required does not depend on the size of the input array.

             */
            public int UsingSlidingWindow(int[] nums, int goal)
            {
                return SlidingWindowAtMost(nums, goal) - SlidingWindowAtMost(nums, goal - 1);
            }
            // Helper function to count the number of subarrays with sum at most the given goal
            private int SlidingWindowAtMost(int[] nums, int goal)
            {
                int start = 0, currentSum = 0, totalCount = 0;

                // Iterate through the array using a sliding window approach
                for (int end = 0; end < nums.Length; end++)
                {
                    currentSum += nums[end];

                    // Adjust the window by moving the start pointer to the right
                    // until the sum becomes less than or equal to the goal
                    while (start <= end && currentSum > goal)
                    {
                        currentSum -= nums[start++];
                    }

                    // Update the total count by adding the length of the current subarray
                    totalCount += end - start + 1;
                }
                return totalCount;
            }
            /* Approach 3: Sliding Window in One Pass

Complexity Analysis
Let n be the length of the nums array.
•	Time complexity: O(n)
The function iterates through the nums array once using a single for loop (end loop).
Inside the loop, the while loop might contract the window, but the total number of iterations within this loop is still bounded by the number of elements in the array (n).
Therefore, the overall time complexity is dominated by the single iteration through the array, resulting in O(n).
•	Space complexity: O(1)
The space complexity is O(1) because the algorithm uses a constant amount of space for variables such as start, currentSum, and totalCount. The space required does not depend on the size of the input array.

             */
            public int UsingSlidingWindowInOnePass(int[] nums, int goal)
            {
                int start = 0;
                int prefixZeros = 0;
                int currentSum = 0;
                int totalCount = 0;

                // Loop through the array using end pointer
                for (int end = 0; end < nums.Length; end++)
                {
                    // Add current element to the sum
                    currentSum += nums[end];

                    // Slide the window while condition is met
                    while (start < end && (nums[start] == 0 || currentSum > goal))
                    {
                        if (nums[start] == 1)
                        {
                            prefixZeros = 0;
                        }
                        else
                        {
                            prefixZeros++;
                        }

                        currentSum -= nums[start];
                        start++;
                    }

                    // Count subarrays when window sum matches the goal
                    if (currentSum == goal)
                    {
                        totalCount += 1 + prefixZeros;
                    }
                }

                return totalCount;
            }

        }


        /* 1493. Longest Subarray of 1's After Deleting One Element
        https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/
         */
        class LongestSubarrayOf1sAfterDeletingOneElementSol
        {

            /* Approach: Sliding Window
Complexity Analysis
Here, N is the size of the array nums.
•	Time complexity: O(N)
Each element in the array will be iterated over twice at most. Each element will be iterated over for the first time in the for loop; then, it might be possible to re-iterate while shrinking the window in the while loop. No element can be iterated more than twice. Therefore, the total time complexity would be O(N).
•	Space complexity: O(1)
Apart from the three variables, we don't need any extra space; hence the total space complexity is constant.	

             */
            public int UsingSlidingWindow(int[] nums)
            {
                // Number of zero's in the window.
                int zeroCount = 0;
                int longestWindow = 0;
                // Left end of the window.
                int start = 0;

                for (int i = 0; i < nums.Length; i++)
                {
                    zeroCount += (nums[i] == 0 ? 1 : 0);

                    // Shrink the window until the count of zero's
                    // is less than or equal to 1.
                    while (zeroCount > 1)
                    {
                        zeroCount -= (nums[start] == 0 ? 1 : 0);
                        start++;
                    }

                    longestWindow = Math.Max(longestWindow, i - start);
                }

                return longestWindow;
            }
        }


        /* 1590. Make Sum Divisible by P
        https://leetcode.com/problems/make-sum-divisible-by-p/description/
         */
        public class MinSubarraySol
        {
            /* Approach 1: Brute Force (Time Limit Exceeded)
            Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(n^2)
The outer loop runs n times, iterating over the starting index of the subarray. The inner loop also runs up to n times for each iteration of the outer loop, as it sums the elements from the starting index to the end. Therefore, the overall time complexity of this nested loop structure results in O(n^2).
•	Space complexity: O(1)
The algorithm uses a constant amount of extra space, as it only stores a few variables (like totalSum, target, subSum, and minLen). The space used does not depend on the size of the input array, making the space complexity constant.

             */

            public int MinSubarray(int[] nums, int p)
            {
                int n = nums.Length;
                long totalSum = 0;

                // Calculate the total sum
                foreach (int num in nums)
                {
                    totalSum += num;
                }

                // If the total sum is already divisible by p, no subarray needs to be removed
                if (totalSum % p == 0) return 0;

                int minLen = n; // Initialize minLen to the size of the array

                // Try removing every possible subarray
                for (int start = 0; start < n; ++start)
                {
                    long subSum = 0; // Initialize subarray sum
                    for (int end = start; end < n; ++end)
                    {
                        subSum += nums[end]; // Calculate the subarray sum

                        // Check if removing this subarray makes the remaining sum divisible by p
                        long remainingSum = (totalSum - subSum) % p;

                        if (remainingSum == 0)
                        {
                            minLen = Math.Min(minLen, end - start + 1); // Update the smallest subarray length
                        }
                    }
                }

                // If no valid subarray is found, return -1
                return minLen == n ? -1 : minLen;
            }
            /* Approach 2: Prefix Sum Modulo
            Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(n)
The algorithm iterates through the nums array twice: once to calculate the total sum and again to find the minimum length of the subarray that needs to be removed. Both of these operations take linear time, resulting in an overall time complexity of O(n).
•	Space complexity: O(n)
The algorithm uses a hash map (modMap) to store the remainders and their corresponding indices. In the worst case, this hash map could store up to n different remainders (one for each element in the array), leading to a space complexity of O(n).

             */
            public int UsingPrefixSumModulo(int[] nums, int p)
            {
                int n = nums.Length;
                int totalSum = 0;

                // Step 1: Calculate total sum and target remainder
                foreach (int num in nums)
                {
                    totalSum = (totalSum + num) % p;
                }

                int target = totalSum % p;
                if (target == 0)
                {
                    return 0; // The array is already divisible by p
                }

                // Step 2: Use a hash map to track prefix sum mod p
                Dictionary<int, int> modMap = new();
                modMap.Add(0, -1); // To handle the case where the whole prefix is the answer
                int currentSum = 0;
                int minLen = n;

                // Step 3: Iterate over the array
                for (int i = 0; i < n; ++i)
                {
                    currentSum = (currentSum + nums[i]) % p;

                    // Calculate what we need to remove
                    int needed = (currentSum - target + p) % p;

                    // If we have seen the needed remainder, we can consider this subarray
                    if (modMap.ContainsKey(needed))
                    {
                        minLen = Math.Min(minLen, i - modMap[needed]);
                    }

                    // Store the current remainder and index
                    modMap[currentSum] = i;
                }

                // Step 4: Return result
                return minLen == n ? -1 : minLen;
            }
        }

        /* 2537. Count the Number of Good Subarrays
        https://leetcode.com/problems/count-the-number-of-good-subarrays/description/
        https://algo.monster/liteproblems/2537
         */
        class CountGoodSubarraysSol
        {
            /* Time and Space Complexity
            Time Complexity
            The given code has two nested loops. However, the inner loop (while-loop) only decreases cur down to a point where it is less than k, and since elements can only be added to cur when the outer loop (for-loop) runs, the inner loop can run at most as many times as the outer loop throughout the whole execution.
            Because the inner loop pointer i is only incremented and never reset, each element is processed once by both the outer and inner loops together. This leads to a linear relationship with the number of elements in nums. Therefore, the overall time complexity is O(n), where n is the length of nums.
            Space Complexity
            The space complexity is driven by the use of the Counter object that stores counts for elements in nums. In the worst case, if all elements are unique, the counter would require space proportional to n. Hence, the space complexity is also O(n).
             */
            public long CountGoodSubarrays(int[] numbers, int threshold)
            {
                // Dictionary to store the frequency of each number in the current subarray
                Dictionary<int, int> frequencyCounter = new Dictionary<int, int>();
                long totalCount = 0; // Total count of good subarrays
                long currentSize = 0; // Number of times a number has been repeated in the current subarray
                int startIndex = 0; // Start index for the sliding window

                // Iterate over the array using 'number' as the current element
                foreach (int number in numbers)
                {
                    // Update currentSize for the current value
                    currentSize += frequencyCounter.ContainsKey(number) ? frequencyCounter[number] : 0;
                    // Increase the frequency counter for number
                    frequencyCounter[number] = frequencyCounter.ContainsKey(number) ? frequencyCounter[number] + 1 : 1;

                    // Shrink the window from the left until the number of repeated elements is less than threshold
                    while (currentSize - (frequencyCounter.ContainsKey(numbers[startIndex]) ? frequencyCounter[numbers[startIndex]] : 0) + 1 >= threshold)
                    {
                        // Decrease the currentSize by the number of times the number at startIndex is in the window
                        currentSize -= frequencyCounter[numbers[startIndex]] > 0 ? frequencyCounter[numbers[startIndex]] : 0;
                        frequencyCounter[numbers[startIndex]]--;
                        // Move the start index of the subarray window to the right
                        startIndex++;
                    }

                    // If the number of repeated elements is at least threshold, we count this as a 'good' subarray
                    if (currentSize >= threshold)
                    {
                        // Add to the total number (startIndex + 1 indicates that we have a 'good' subarray up to the current index i)
                        totalCount += startIndex + 1;
                    }
                }

                // Return the total count of good subarrays
                return totalCount;
            }
        }

        /* 2598. Smallest Missing Non-negative Integer After Operations
        https://leetcode.com/problems/smallest-missing-non-negative-integer-after-operations/description/
        https://algo.monster/liteproblems/2598
         */

        class FindSmallestNonNegativeIntegerSol
        {
            /* Time and Space Complexity
            The given Python code defines a method findSmallestInteger meant to find the smallest integer that is not present in the input list nums when considering each number modulo value. It manages this by creating a frequency counter for the modulo results and then iteratively checking for the smallest index that is not present in the frequency counter.
            Time Complexity
            The time complexity of the function is O(n), where n is the length of the input list nums. This arises from the following operations:
            1.	Initializing the counter with (x % value for x in nums) has a linear runtime proportional to the size of nums.
            2.	The subsequent for loop runs for n + 1 iterations in the worst case, as it stops as soon as it finds an integer that is not in the counter. Each operation within the for loop (accessing cnt and decrementing the count) is O(1) thanks to Python's Counter which is a hash map under the hood.
            Therefore, as the initialization of the counter and the for loop are both linear in terms of n and are not nested, the overall time complexity remains linear or O(n).
            Space Complexity
            The space complexity of the algorithm is O(value), where value is the input parameter to the method. This complexity is attributed to the following points:
            1.	The counter cnt stores at most value different keys, since each number in nums is taken modulo value, which results in a possible range of [0, value).
            2.	No other data structures are used that depend on the size of the input or value.
            Thus, the space required by the counter is directly proportional to the value, leading to a space complexity of O(value).	
             */
            // Method to find the smallest integer that is not present in the array when modulus with value
            public int FindSmallestInteger(int[] nums, int value)
            {
                // Create an array to count occurrences of each modulus result
                int[] countModulo = new int[value];

                // Iterate over each number in nums and increment the count for the corresponding modulus
                foreach (int num in nums)
                {
                    countModulo[(num % value + value) % value]++;
                }

                // Start looking for the smallest integer that is not in the array, by checking modulus occurrences
                for (int i = 0; ; ++i)
                { // no termination condition here since we are guaranteed to find a number eventually
                  // Use the i % value to wrap around the countModulo array
                  // Check if the current number has a count of zero, which means it's not present in the nums array when modulus with value
                    if (countModulo[i % value] == 0)
                    {
                        // If it's not present, this is the smallest number we are looking for
                        return i;
                    }
                    // Otherwise, decrease the count and keep looking
                    countModulo[i % value]--;
                }
            }
        }

        










    }
}