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



        /* 480. Sliding Window Median
        https://leetcode.com/problems/sliding-window-median/description/
         */

        public class MedianSlidingWindowSol
        {
            /* 
            Approach 1: Simple Sorting
                        Complexity Analysis
            •	Time complexity: O(n⋅klogk) to O(n⋅k).
            o	Copying elements into the container takes about O(k) time each. This happens about (n−k) times.
            o	Sorting for each of the (n−k) sliding window instances takes about O(klogk) time each.
            o	Bisected insertion or deletion takes about O(logk) for searching and O(k) for actual shifting of elements. This takes place about n times.
            •	Space complexity: O(k) extra linear space for the window container.
             */
            public IList<double> UsingSorting(int[] nums, int k)
            {
                List<double> medians = new List<double>();

                for (int i = 0; i + k <= nums.Length; i++)
                {
                    List<int> window = nums.Skip(i).Take(k).ToList();
                    window.Sort();

                    if (k % 2 == 1)
                        medians.Add(window[k / 2]);
                    else
                        medians.Add((window[k / 2 - 1] + (double)window[k / 2]) / 2.0);
                }

                return medians;
            }
            /*
                         Approach 2: Two Heaps (Lazy Removal)
            Complexity Analysis
            •	Time complexity: O(2⋅nlogk)+O(n−k)≈O(nlogk).
            o	Either (or sometimes both) of the heaps gets every element inserted into it at least once. Collectively each of those takes about O(logk) time. That is n such insertions.
            o	About (n−k) removals from the top of the heaps take place (the number of sliding window instances). Each of those takes about O(logk) time.
            o	Hash table operations are assumed to take O(1) time each. This happens roughly the same number of times as removals from heaps take place.
            •	Space complexity: O(k)+O(n)≈O(n) extra linear space.
            o	The heaps collectively require O(k) space.
            o	The hash table needs about O(n−k) space.

             */
            public IList<double> UsingTwoHeapsPQ(int[] nums, int k)
            {
                List<double> medians = new List<double>();
                Dictionary<int, int> hashTable = new Dictionary<int, int>();
                PriorityQueue<int, int> lo = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b.CompareTo(a))); // max heap
                PriorityQueue<int, int> hi = new PriorityQueue<int, int>(); // min heap

                int i = 0; // index of current incoming element being processed

                // initialize the heaps
                while (i < k)
                    lo.Enqueue(nums[i++], nums[i - 1]);
                for (int j = 0; j < k / 2; j++)
                {
                    hi.Enqueue(lo.Dequeue(), lo.Peek());
                }

                while (true)
                {
                    // get median of current window
                    medians.Add(k % 2 == 1 ? lo.Peek() : ((double)lo.Peek() + (double)hi.Peek()) * 0.5);

                    if (i >= nums.Length)
                        break; // break if all elements processed

                    int outNum = nums[i - k], // outgoing element
                        inNum = nums[i++], // incoming element
                        balance = 0; // balance factor

                    // number `outNum` exits window
                    balance += (outNum <= lo.Peek() ? -1 : 1);
                    if (hashTable.ContainsKey(outNum))
                        hashTable[outNum]++;
                    else
                        hashTable[outNum] = 1;

                    // number `inNum` enters window
                    if (lo.Count > 0 && inNum <= lo.Peek())
                    {
                        balance++;
                        lo.Enqueue(inNum, inNum);
                    }
                    else
                    {
                        balance--;
                        hi.Enqueue(inNum, inNum);
                    }

                    // re-balance heaps
                    if (balance < 0)
                    { // `lo` needs more valid elements
                        lo.Enqueue(hi.Dequeue(), hi.Peek());
                        balance++;
                    }
                    if (balance > 0)
                    { // `hi` needs more valid elements
                        hi.Enqueue(lo.Dequeue(), lo.Peek());
                        balance--;
                    }

                    // remove invalid numbers that should be discarded from heap tops
                    while (lo.Count > 0 && hashTable[lo.Peek()] > 0)
                    {
                        hashTable[lo.Peek()]--;
                        lo.Dequeue();
                    }
                    while (hi.Count > 0 && hashTable[hi.Peek()] > 0)
                    {
                        hashTable[hi.Peek()]--;
                        hi.Dequeue();
                    }
                }

                return medians;
            }
            /* 
            Approach 3: Two Multisets/SortedSets
Complexity Analysis
•	Time complexity: O((n−k)⋅6⋅logk)≈O(nlogk).
o	At worst, there are three set insertions and three set deletions from the start or end. Each of these takes about O(logk) time.
o	Finding the mean takes constant O(1) time since the start or ends of sets are directly accessible.
o	Each of these steps takes place about (n−k) times (the number of sliding window instances).
•	Space complexity: O(k) extra linear space to hold contents of the window.

             */
            public IList<double> UsingTwoSortedSets(int[] nums, int k)
            {
                List<double> medians = new List<double>();
                //TODO: Do we really need SortedSet?
                SortedSet<int> lo = new SortedSet<int>();
                SortedSet<int> hi = new SortedSet<int>();

                for (int i = 0; i < nums.Length; i++)
                {
                    // remove outgoing element
                    if (i >= k)
                    {
                        if (nums[i - k] <= lo.Last())
                            lo.Remove(nums[i - k]);
                        else
                            hi.Remove(nums[i - k]);
                    }

                    // insert incoming element
                    lo.Add(nums[i]);

                    // balance the sets
                    hi.Add(lo.Last());
                    lo.Remove(lo.Last());

                    if (lo.Count < hi.Count)
                    {
                        lo.Add(hi.First());
                        hi.Remove(hi.First());
                    }

                    // get median
                    if (i >= k - 1)
                    {
                        if (k % 2 == 1)
                            medians.Add(lo.Last());
                        else
                            medians.Add((lo.Last() + hi.First()) / 2.0);
                    }
                }

                return medians;
            }
            /* 
            Approach 4: Multiset and Two Pointers
            Complexity Analysis
            •	Time complexity: O((n−k)logk)+O(k)≈O(nlogk).
            o	Initializing mid takes about O(k) time.
            o	Inserting or deleting a number takes O(logk) time for a standard multiset scheme. 4
            o	Finding the mean takes constant O(1) time since the median elements are directly accessible from mid iterator.
            o	The last two steps take place about (n−k) times (the number of sliding window instances).
            •	Space complexity: O(k) extra linear space to hold contents of the window.

             */
            public IList<double> SortedSetWithTwoPointers(int[] nums, int k)
            {
                List<double> medians = new List<double>();
                //TODO: Test below logic on creating slice of nums to SortedSet
                SortedSet<int> window = new SortedSet<int>(nums.ToList().GetRange(0, k));

                var mid = window.ElementAt(k / 2);

                for (int i = k; i < nums.Length; i++)
                {
                    medians.Add(((double)mid + window.ElementAt((k - 1) % 2)) / 2);

                    window.Add(nums[i]);
                    if (nums[i] < mid)
                        mid = window.ElementAt(mid - 1);

                    window.Remove(nums[i - k]);
                    if (nums[i - k] <= mid)
                        mid = window.ElementAt(mid + 1);
                }

                medians.Add(((double)mid + window.ElementAt((k - 1) % 2)) / 2);

                return medians;
            }


        }

        /* 1004. Max Consecutive Ones III
        https://leetcode.com/problems/max-consecutive-ones-iii/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class MaxConsecutiveOnesIIISol
        {
/*             Approach: Sliding Window
Complexity Analysis
•	Time Complexity: O(N), where N is the number of elements in the array. In worst case we might end up visiting every element of array twice, once by left pointer and once by right pointer.
•	Space Complexity: O(1). We do not use any extra space.

 */
            public int UsingSlidingWindow(int[] nums, int k)
            {
                int left = 0, right;
                for (right = 0; right < nums.Length; right++)
                {
                    // If we included a zero in the window we reduce the value of k.
                    // Since k is the maximum zeros allowed in a window.
                    if (nums[right] == 0)
                    {
                        k--;
                    }
                    // A negative k denotes we have consumed all allowed flips and window has
                    // more than allowed zeros, thus increment left pointer by 1 to keep the window size same.
                    if (k < 0)
                    {
                        // If the left element to be thrown out is zero we increase k.
                        k += 1 - nums[left];
                        left++;
                    }
                }
                return right - left;
            }
        }










    }
}