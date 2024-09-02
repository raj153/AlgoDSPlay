using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    362. Design Hit Counter‘
    https://leetcode.com/problems/design-hit-counter/

    Approach #1: Using Queue
•	Time Complexity
        o	hit - Since inserting a value in the queue takes place in O(1) time, hence hit method works in O(1).
        o	getHits - Assuming a total of n values present in the queue at a time and the total number of timestamps encountered throughout is N. 
            In the worst case scenario, we might end up removing all the entries from the queue in getHits method if the difference in timestamp is greater than or equal to 300. 
            Hence in the worst case, a "single" call to the getHits method can take O(n) time. 
            However, we must notice that each timestamp is processed only twice (first while adding the timestamp in the queue in hit method and second while removing the timestamp from the queue in the getHits method). 
            Hence if the total number of timestamps encountered throughout is N, the overall time taken by getHits method is O(N). This results in an amortized time complexity of O(1) for a single call to getHits method.
•	Space Complexity: Considering the total timestamps encountered throughout to be N, the queue can have upto N elements, hence overall space complexity of this approach is O(N).

    */
    public class HitCounter
    {
        private Queue<int> hits;

        public HitCounter()
        {
            this.hits = new Queue<int>();
        }

        /** Record a hit.
            @param timestamp - The current timestamp (in seconds granularity). */
        public void Hit(int timestamp)
        {
            this.hits.Enqueue(timestamp);
        }

        /** Return the number of hits in the past 5 minutes.
        @param timestamp - The current timestamp (in seconds granularity). */
        public int GetHits(int timestamp)
        {
            while (this.hits.Count > 0)
            {
                int diff = timestamp - this.hits.Peek();
                if (diff >= 3000) this.hits.Dequeue();
                else break;
            }
            return this.hits.Count;
        }

    }
    /*
    Approach #2: Using Deque with Pairs

    In the worst case, when there are not many repetitions, the time complexity and space complexity of Approach 2 is the same as Approach 1. 
    However in case we have repetitions (say k repetitions of a particular ith timestamp), the time complexity and space complexities are as follows.
    
    Time Complexity:hit - O(1).
        getHits - If there are a total of n pairs present in the deque, worst case time complexity can be O(n).However, by clubbing all the timestamps with same value together, 
                    for the ith timestamp with k repetitions, the time complexity is O(1) as here, instead of removing all those k repetitions, we only remove a single entry from the deque.
    Space complexity: If there are a total of N elements that we encountered throughout, the space complexity is O(N) (similar to Approach 1). However, in the case of repetitions, the space required for storing those k values O(1).
    */
    class HitCounterOptimal
    {
        private int total;
        private LinkedList<(int, int)> hits;

        /** Initialize your data structure here. */
        public HitCounterOptimal()
        {
            // Initialize total to 0
            this.total = 0;
            this.hits = new LinkedList<(int, int)>();
        }

        /** Record a hit.
            @param timestamp - The current timestamp (in seconds granularity). */
        public void Hit(int timestamp)
        {
            if (this.hits.Count == 0 || this.hits.Last.Value.Item1 != timestamp)
            {
                // Insert the new timestamp with count = 1
                this.hits.AddLast((timestamp, 1));
            }
            else
            {
                // Update the count of latest timestamp by incrementing the count by 1

                // Obtain the current count of the latest timestamp 
                int prevCount = this.hits.Last.Value.Item2;
                // Remove the last pair of (timestamp, count) from the deque
                this.hits.RemoveLast();
                // Insert a new pair of (timestamp, updated count) in the deque
                this.hits.AddLast((timestamp, prevCount + 1));
            }
            // Increment total
            this.total++;
        }

        /** Return the number of hits in the past 5 minutes.
            @param timestamp - The current timestamp (in seconds granularity). */
        public int GetHits(int timestamp)
        {
            while (this.hits.Count > 0)
            {
                int diff = timestamp - this.hits.First.Value.Item1;
                if (diff >= 300)
                {
                    // Decrement total by the count of the oldest timestamp
                    this.total -= this.hits.First.Value.Item2;
                    this.hits.RemoveFirst();
                }
                else break;
            }
            return this.total;
        }
    }

}