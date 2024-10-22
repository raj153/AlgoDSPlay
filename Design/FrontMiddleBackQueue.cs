using System;

namespace AlgoDSPlay.Design
{
    /* 1670. Design Front Middle Back Queue
https://leetcode.com/problems/design-front-middle-back-queue/description/
https://algo.monster/liteproblems/1670
 */

    /* Time and Space Complexity
   Time Complexity
   •	__init__: O(1) - Constant time complexity as it initializes two deques.
   •	pushFront: Amortized O(1) - Appending to the front of a deque is typically O(1), rebalance may cause an O(1) operation.
   •	pushMiddle: O(1) - Appending to the end of q1 deque is O(1), rebalance may cause an O(1) operation.
   •	pushBack: Amortized O(1) - Appending to the end of a deque is typically O(1), rebalance may cause an O(1) operation.
   •	popFront: O(1) - Popping from the front of the deque is O(1), rebalance may cause an O(1) operation.
   •	popMiddle: O(1) - Popping from q1 or q2 is O(1), rebalance may cause an O(1) operation.
   •	popBack: O(1) - Popping from the back of q2 is O(1), rebalance may cause an O(1) operation.
   •	rebalance: O(1) - The rebalance function does at most one operation of moving an element from one deque to another, which is O(1).
   For the rebalance operations, when we say amortized O(1), it is because deque operations are generally O(1) unless a resize of the underlying array is necessary. Given that the average case does not trigger a resize, we consider these operations to have an amortized constant time complexity. Note that amortized analysis is a way to represent the complexity of an algorithm over a sequence of operations, reflecting the average cost per operation in the worst case.
   Space Complexity
   •	Overall: O(n) - The space complexity is based on the number of elements in the queue which could be up to n.
   •	__init__: O(1) - Only two deques are initialized, no elements are added initially.
   •	pushFront, pushMiddle, pushBack, popFront, popMiddle, popBack: These operations don't change the overall space complexity, which remains O(n) depending on the number of elements at any point in time.
    */
    public class FrontMiddleBackQueue
    {
        private LinkedList<int> frontQueue = new LinkedList<int>();
        private LinkedList<int> backQueue = new LinkedList<int>();

        // Constructor for the queue.
        public FrontMiddleBackQueue()
        {
            // No initialization is required as the Deques are initialized already.
        }

        // Adds an element to the front of the queue.
        public void PushFront(int value)
        {
            frontQueue.AddFirst(value);
            RebalanceQueues();
        }

        // Adds an element to the middle of the queue.
        public void PushMiddle(int value)
        {
            if (frontQueue.Count <= backQueue.Count)
            {
                frontQueue.AddLast(value);
            }
            else
            {
                backQueue.AddFirst(value);
            }
            RebalanceQueues();
        }

        // Adds an element to the back of the queue.
        public void PushBack(int value)
        {
            backQueue.AddLast(value);
            RebalanceQueues();
        }

        // Removes and returns the front element of the queue.
        public int PopFront()
        {
            if (IsEmpty())
            {
                return -1; // Return -1 if the queue is empty.
            }
            int value = frontQueue.Count == 0 ? backQueue.First.Value : frontQueue.First.Value;
            if (frontQueue.Count > 0)
            {
                frontQueue.RemoveFirst();
            }
            else
            {
                backQueue.RemoveFirst();
            }
            RebalanceQueues();
            return value;
        }

        // Removes and returns the middle element of the queue.
        public int PopMiddle()
        {
            if (IsEmpty())
            {
                return -1; // Return -1 if the queue is empty.
            }
            int value = frontQueue.Count >= backQueue.Count ? frontQueue.Last.Value : backQueue.First.Value;
            if (frontQueue.Count >= backQueue.Count)
            {
                frontQueue.RemoveLast();
            }
            else
            {
                backQueue.RemoveFirst();
            }
            RebalanceQueues();
            return value;
        }

        // Removes and returns the back element of the queue.
        public int PopBack()
        {
            if (backQueue.Count == 0)
            {
                return -1; // Return -1 if the queue is empty.
            }
            int value = backQueue.Last.Value;
            backQueue.RemoveLast();
            RebalanceQueues();
            return value;
        }

        // Helper method to check if the queue is empty.
        private bool IsEmpty()
        {
            return frontQueue.Count == 0 && backQueue.Count == 0;
        }

        // Maintains the balance between the two deques so that they represent the proper ordering of elements.
        private void RebalanceQueues()
        {
            // Rebalance if frontQueue has more elements than backQueue.
            if (frontQueue.Count > backQueue.Count + 1)
            {
                backQueue.AddFirst(frontQueue.Last.Value);
                frontQueue.RemoveLast();
            }
            // Rebalance if backQueue has more elements than frontQueue.
            if (backQueue.Count > frontQueue.Count)
            {
                frontQueue.AddLast(backQueue.First.Value);
                backQueue.RemoveFirst();
            }
        }
    }
}
