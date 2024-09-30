using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class SlidingWindowProbs
    {

        /*
        2024. Maximize the Confusion of an Exam
        https://leetcode.com/problems/maximize-the-confusion-of-an-exam/description/

        */
        public class MaximizeConfusionOfExamSolution
        {

            /*
           Approach 1: Binary Search + Fixed Size Sliding Window (BSFZSW)
Complexity Analysis
Let n be the length of the input string answerKey.
•	Time complexity: O(n⋅logn)
o	We set the search space to [1, n], it takes at most O(logn) search steps.
o	At each step, we iterate over answerKey which takes O(n) time.
•	Space complexity: O(1)
o	We only need to update some parameters left, right. During the iteration, we need to count the number of T and F, which also takes O(1) space.

            */
            public int MaximizeConfusionOfExamWithConsecutiveAnswersBSFZSW(string answerKey, int k)
            {
                int n = answerKey.Length;
                int left = k, right = n;

                while (left < right)
                {
                    int mid = (left + right + 1) / 2;

                    if (IsValid(answerKey, mid, k))
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                return left;

            }
            private bool IsValid(String answerKey, int size, int k)
            {
                int n = answerKey.Length;
                Dictionary<char, int> counter = new Dictionary<char, int>();

                for (int i = 0; i < size; i++)
                {
                    char c = answerKey[i];
                    counter.Add(c, counter.GetValueOrDefault(c, 0) + 1);
                }

                if (Math.Min(counter.GetValueOrDefault('T', 0), counter.GetValueOrDefault('F', 0)) <= k)
                {
                    return true;
                }

                for (int i = size; i < n; i++)
                {
                    char c1 = answerKey[i];
                    counter.Add(c1, counter.GetValueOrDefault(c1, 0) + 1);
                    char c2 = answerKey[i - size];
                    counter.Add(c2, counter.GetValueOrDefault(c2, 0) - 1);

                    if (Math.Min(counter.GetValueOrDefault('T', 0), counter.GetValueOrDefault('F', 0)) <= k)
                    {
                        return true;
                    }
                }

                return false;
            }


            /*
            Approach 2: Sliding Window (SW) 
            Complexity Analysis
            Let n be the length of the input string answerKey.
            •	Time complexity: O(n)
            o	In the iteration of the right boundary right, we shift it from 0 to n - 1. Although we may move the left boundary left in each step, left always stays to the left of right, which means left moves at most n - 1 times.
            o	At each step, we update the value of an element in the hash map count, which takes constant time.
            o	To sum up, the overall time complexity is O(n).
            •	Space complexity: O(1)
            o	We only need to update two indices left and right. During the iteration, we need to count the number of T and F, which also takes O(1) space.

            */
            public int MaximizeConfusionOfExamWithConsecutiveAnswersSW(string answerKey, int k)
            {
                int maxSize = k;
                Dictionary<char, int> count = new Dictionary<char, int>();
                for (int i = 0; i < k; i++)
                {
                    count.Add(answerKey[i], count.GetValueOrDefault(answerKey[i], 0) + 1);
                }

                int left = 0;
                for (int right = k; right < answerKey.Length; right++)
                {
                    count.Add(answerKey[right], count.GetValueOrDefault(answerKey[right], 0) + 1);

                    while (Math.Min(count.GetValueOrDefault('T', 0), count.GetValueOrDefault('F', 0)) > k)
                    {
                        count.Add(answerKey[left], count[answerKey[left]] - 1);
                        left++;
                    }

                    maxSize = Math.Max(maxSize, right - left + 1);
                }

                return maxSize;
            }
            /*
            Approach 3: Advanced Sliding Window (ASW)

            Complexity Analysis
    Let n be the length of the input string answerKey.
    •	Time complexity: O(n)
    o	In the iteration of the right boundary right, we shift it from 0 to n - 1.
    o	At each step, we update the number of answerKey[right] and/or the number of answerKey[right - max_size] in the hash map count, which takes constant time.
    o	To sum up, the overall time complexity is O(n).
    •	Space complexity: O(1)
    o	We only need to update two parameters max_size and right. During the iteration, we need to count the number of T and F, which also takes O(1) space.

            */

            public int MaxConsecutiveAnswersASW(String answerKey, int k)
            {
                int maxSize = 0;
                Dictionary<char, int> count = new Dictionary<char, int>();

                for (int right = 0; right < answerKey.Length; right++)
                {
                    count.Add(answerKey[right], count.GetValueOrDefault(answerKey[right], 0) + 1);
                    int minor = Math.Min(count.GetValueOrDefault('T', 0), count.GetValueOrDefault('F', 0));

                    if (minor <= k)
                    {
                        maxSize++;
                    }
                    else
                    {
                        count.Add(answerKey[right - maxSize], count[answerKey[right - maxSize]] - 1);
                    }
                }

                return maxSize;
            }
        }


        /* 239. Sliding Window Maximum
        https://leetcode.com/problems/sliding-window-maximum/description/
         */
        public class MaxSlidingWindowSol
        {
            /*
            
Approach: Monotonic Deque
Complexity Analysis
Here n is the size of nums.
•	Time complexity: O(n).
o	At first glance, it may look like the time complexity of this algorithm should be O(n2), because there is a nested while loop inside the for loop. However, each element can only be added to the deque once, which means the deque is limited to n pushes. Every iteration of the while loop uses 1 pop, which means the while loop will not iterate more than n times in total, across all iterations of the for loop.
o	An easier way to think about this is that in the worst case, every element will be pushed and popped once. This gives a time complexity of O(2⋅n)=O(n).
•	Space complexity: O(k).
o	The size of the deque can grow a maximum up to a size of k.

            */
            public int[] UsingMonotonicDeque(int[] nums, int k)
            {
                //TODO: Replace below LinkedList with Deque, as LinkedList != to Deque(DoubleEndedQueue)
                LinkedList<int> deque = new LinkedList<int>();
                List<int> result = new List<int>();

                for (int i = 0; i < k; i++)
                {
                    while (deque.Count > 0 && nums[i] >= nums[deque.Last.Value])
                    {
                        deque.RemoveLast();
                    }
                    deque.AddLast(i);
                }
                result.Add(nums[deque.First.Value]);

                for (int i = k; i < nums.Length; i++)
                {
                    if (deque.First.Value == i - k)
                    {
                        deque.RemoveFirst();
                    }
                    while (deque.Count > 0 && nums[i] >= nums[deque.Last.Value])
                    {
                        deque.RemoveLast();
                    }

                    deque.AddLast(i);
                    result.Add(nums[deque.First.Value]);
                }
                // Return the result as an array.
                return result.ToArray();
            }
        }


















    }
}