using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    295. Find Median from Data Stream
    https://leetcode.com/problems/find-median-from-data-stream/description/

    */
    public class MedianFinder
    {
        /*
        Approach 1: Simple Sorting (SS)
        Time complexity: O(nlogn)+O(1)≃O(nlogn).
            Adding a number takes amortized O(1) time for a container with an efficient resizing scheme.
            Finding the median is primarily dependent on the sorting that takes place. This takes O(nlogn) time for a standard comparative sort.
        Space complexity: O(n) linear space to hold input in a container. No extra space other than that needed (since sorting can usually be done in-place).
        */
        class MedianFinderSS
        {
            List<int> numberStore;

            public MedianFinderSS()
            {
                numberStore = new List<int>();
            }

            // Adds a number into the data structure.
            public void AddNumber(int number)
            {
                numberStore.Add(number);
            }

            // Returns the median of current data stream
            public double FindMedian()
            {
                numberStore.Sort();

                int count = numberStore.Count;
                return (count % 2 == 1) ? numberStore[count / 2] : ((double)numberStore[count / 2 - 1] + numberStore[count / 2]) * 0.5;
            }
        }
        /*
        Approach 2: Insertion Sort(IS)
        Time complexity: O(n)+O(logn)≈O(n).
            Binary Search takes O(logn) time to find correct insertion position.
            Insertion can take up to O(n) time since elements have to be shifted inside the container to make room for the new element.
            Pop quiz: Can we use a linear search instead of a binary search to find insertion position, without incurring any significant runtime penalty?
        Space complexity: O(n) linear space to hold input in a container.
        */
        class MedianFinderIS
        {
            private List<int> numbers; // resize-able container

            public MedianFinderIS()
            {
                numbers = new List<int>();
            }

            // Adds a number into the data structure.
            public void AddNumber(int number)
            {
                if (numbers.Count == 0)
                    numbers.Add(number);
                else
                    numbers.Insert(BinarySearchInsertIndex(number), number); // binary search and insertion combined
            }

            // Returns the median of current data stream
            public double FindMedian()
            {
                int count = numbers.Count;
                return count % 2 == 1 ? numbers[count / 2] : ((double)numbers[count / 2 - 1] + numbers[count / 2]) * 0.5;
            }

            private int BinarySearchInsertIndex(int number)
            {
                int left = 0, right = numbers.Count;
                while (left < right)
                {
                    int mid = left + (right - left) / 2;
                    if (numbers[mid] < number)
                        left = mid + 1;
                    else
                        right = mid;
                }
                return left;
            }
        }

        /*
        Approach 3: Two Heaps 
        Time complexity: O(5⋅logn)+O(1)≈O(logn).
                At worst, there are three heap insertions and two heap deletions from the top. Each of these takes about O(logn) time.
                Finding the median takes constant O(1) time since the tops of heaps are directly accessible.
        Space complexity: O(n) linear space to hold input in containers.
        */
        class MedianFinderHeap
        {
            private PriorityQueue<int, int> maxHeap;                              // max heap
            private PriorityQueue<int, int> minHeap;                              // min heap

            public MedianFinderHeap()
            {
                maxHeap = new PriorityQueue<int, int>();
                minHeap = new PriorityQueue<int, int>(Comparer<int>.Create((x, y) => x.CompareTo(y)));
            }

            // Adds a number into the data structure.
            public void AddNum(int num)
            {
                maxHeap.Enqueue(num,num);                                    // Add to max heap
                int val =maxHeap.Dequeue();
                minHeap.Enqueue(val, val);                      // balancing step

                if (maxHeap.Count < minHeap.Count)
                {                     // maintain size property
                    val =maxHeap.Dequeue();
                    maxHeap.Enqueue(val, val);
                }

            }
            // Returns the median of current data stream
            public double FindMedian()
            {
                return maxHeap.Count > minHeap.Count ? maxHeap.Peek() : ((double)maxHeap.Peek() + minHeap.Peek()) * 0.5;
            }

        }
    }

    /**
    * Your MedianFinder object will be instantiated and called as such:
    * MedianFinder obj = new MedianFinder();
    * obj.AddNum(num);
    * double param_2 = obj.FindMedian();
     */
}