using System;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay.Design
{
    /*     641. Design Circular Deque
    https://leetcode.com/problems/design-circular-deque/description/
     */
    public class CircularDequeSol
    {
        /*Approach 1: Linked List 
        Complexity Analysis
•	Time Complexity: O(1)
Because we maintain access to the front and rear elements at all times, all operations simply involve pointer manipulations that take O(1) time.
•	Space Complexity: O(k)
In the worst case, there will be maximum k nodes in our doubly linked list, which will involve instantiating k node objects and thus take O(k) space.
 */
        class CircularDequeUsingLinkedList
        {

            ListNode head;
            ListNode rear;
            int size;
            int capacity;

            public CircularDequeUsingLinkedList(int k)
            {
                size = 0;
                capacity = k;
            }

            public bool InsertFront(int value)
            {
                if (IsFull()) return false;
                if (head == null)
                {
                    // first element in list
                    head = new ListNode(value, null, null);
                    rear = head;
                }
                else
                {
                    // add new head
                    ListNode newHead = new ListNode(value, head, null);
                    head.Prev = newHead;
                    head = newHead;
                }
                size++;
                return true;
            }

            public bool InsertLast(int value)
            {
                if (IsFull()) return false;
                if (head == null)
                {
                    // first element in list
                    head = new ListNode(value, null, null);
                    rear = head;
                }
                else
                {
                    // add new element to end
                    rear.Next = new ListNode(value, null, rear);
                    rear = rear.Next;
                }
                size++;
                return true;
            }

            public bool DeleteFront()
            {
                if (IsEmpty()) return false;
                if (size == 1)
                {
                    head = null;
                    rear = null;
                }
                else
                {
                    head = head.Next;
                }
                size--;
                return true;
            }

            public bool DeleteLast()
            {
                if (IsEmpty()) return false;
                if (size == 1)
                {
                    head = null;
                    rear = null;
                }
                else
                {
                    // update rear to the previous node
                    rear = rear.Prev;
                }
                size--;
                return true;
            }

            public int GetFront()
            {
                if (IsEmpty()) return -1;
                return head.Val;
            }

            public int GetRear()
            {
                if (IsEmpty()) return -1;
                return rear.Val;
            }

            public bool IsEmpty()
            {
                return size == 0;
            }

            public bool IsFull()
            {
                return size == capacity;
            }
        }

        /* 
        Approach 2: Fixed Array with Circular Ordering 

    Complexity Analysis
    •	Time Complexity: O(1)
    Similar to Approach 1, we maintain the references for the front and rear elements at all times, where all operations are simply arithmetic operations that take O(1) time.
    •	Space Complexity: O(k)
    Our fixed-sized array will always have k elements and thus will take O(k) space.	

        */
        class CircularDequeUsingFixedArrayWithCirularOrdering
        {

            int[] array;
            int front;
            int rear;
            int size;
            int capacity;

            public CircularDequeUsingFixedArrayWithCirularOrdering(int k)
            {
                array = new int[k];
                size = 0;
                capacity = k;
                front = 0;
                rear = k - 1;
            }

            public bool InsertFront(int value)
            {
                if (IsFull()) return false;
                front = (front - 1 + capacity) % capacity;
                array[front] = value;
                size++;
                return true;
            }

            public bool InsertLast(int value)
            {
                if (IsFull()) return false;
                rear = (rear + 1) % capacity;
                array[rear] = value;
                size++;
                return true;
            }

            public bool DeleteFront()
            {
                if (IsEmpty()) return false;
                front = (front + 1) % capacity;
                size--;
                return true;
            }

            public bool DeleteLast()
            {
                if (IsEmpty()) return false;
                rear = (rear - 1 + capacity) % capacity;
                size--;
                return true;
            }

            public int GetFront()
            {
                if (IsEmpty()) return -1;
                return array[front];
            }

            public int GetRear()
            {
                if (IsEmpty()) return -1;
                return array[rear];
            }

            public bool IsEmpty()
            {
                return size == 0;
            }

            public bool IsFull()
            {
                return size == capacity;
            }
        }


    }


}
