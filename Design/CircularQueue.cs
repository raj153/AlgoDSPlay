using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    622. Design Circular Queue	
    https://leetcode.com/problems/design-circular-queue/description/

    */
    public class CircularQueue
    {
        /*
        Approach 1: Array

        Complexity
        •	Time complexity: O(1). All of the methods in our circular data structure is of constant time complexity.
        •	Space Complexity: O(N). The overall space complexity of the data structure is linear, where N is the pre-assigned capacity of the queue. However, it is worth mentioning that the memory consumption of the data structure remains as its pre-assigned capacity during its entire life cycle.

        */

        public class CircularQueueArray
        {
            private Node head, tail;
            private int count;
            private int capacity;
            // Additional variable to secure the access of our queue
            private readonly ReaderWriterLockSlim queueLock = new ReaderWriterLockSlim();

            /** Initialize your data structure here. Set the size of the queue to be k. */
            public CircularQueueArray(int k)
            {
                this.capacity = k;
            }

            /** Insert an element into the circular queue. Return true if the operation is successful. */
            public bool EnQueue(int value)
            {
                // ensure the exclusive access for the following block.
                queueLock.EnterWriteLock();
                try
                {
                    if (this.count == this.capacity)
                        return false;

                    Node newNode = new Node(value);
                    if (this.count == 0)
                    {
                        head = tail = newNode;
                    }
                    else
                    {
                        tail.NextNode = newNode;
                        tail = newNode;
                    }
                    this.count += 1;
                }
                finally
                {
                    queueLock.ExitWriteLock();
                }
                return true;
            }
        }

        /*
        Approach 2: Singly-Linked List (SLL)

        Complexity
        •	Time complexity: O(1) for each method in our circular queue.
        •	Space Complexity: The upper bound of the memory consumption for our circular queue would be O(N), same as the array approach. However, it should be more memory efficient as we discussed in the intuition section.

        */
        public class CircularQueueSLL
        {

            private Node head, tail;
            private int count;
            private int capacity;

            /** Initialize your data structure here. Set the size of the queue to be k. */
            public CircularQueueSLL(int k)
            {
                this.capacity = k;
            }

            /** Insert an element into the circular queue. Return true if the operation is successful. */
            public bool EnQueue(int value)
            {
                if (this.count == this.capacity)
                    return false;

                Node newNode = new Node(value);
                if (this.count == 0)
                {
                    head = tail = newNode;
                }
                else
                {
                    tail.NextNode = newNode;
                    tail = newNode;
                }
                this.count += 1;
                return true;
            }

            /** Delete an element from the circular queue. Return true if the operation is successful. */
            public bool DeQueue()
            {
                if (this.count == 0)
                    return false;
                this.head = this.head.NextNode;
                this.count -= 1;
                return true;
            }

            /** Get the front item from the queue. */
            public int Front()
            {
                if (this.count == 0)
                    return -1;
                else
                    return this.head.Value;
            }

            /** Get the last item from the queue. */
            public int Rear()
            {
                if (this.count == 0)
                    return -1;
                else
                    return this.tail.Value;
            }

            /** Checks whether the circular queue is empty or not. */
            public bool IsEmpty()
            {
                return (this.count == 0);
            }

            /** Checks whether the circular queue is full or not. */
            public bool IsFull()
            {
                return (this.count == this.capacity);
            }
        }

        public class Node
        {
            public int Value { get; set; }
            public Node NextNode { get; set; }

            public Node(int value)
            {
                Value = value;
                NextNode = null;
            }
        }
        /**
 * Your MyCircularQueue object will be instantiated and called as such:
 * MyCircularQueue obj = new MyCircularQueue(k);
 * bool param_1 = obj.EnQueue(value);
 * bool param_2 = obj.DeQueue();
 * int param_3 = obj.Front();
 * int param_4 = obj.Rear();
 * bool param_5 = obj.IsEmpty();
 * bool param_6 = obj.IsFull();
 */

    }
}