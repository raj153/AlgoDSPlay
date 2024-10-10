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


















    }
}