using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Runtime.ExceptionServices;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;
namespace AlgoDSPlay
{
    public class LinkedListOps
    {
        //https://www.algoexpert.io/questions/remove-kth-node-from-end
        public static void RemoveKthNodeFromEnd(LinkedList head, int k)
        {
            //T:O(n)| S:O(1)
            int counter = 1;
            LinkedList first = head;
            LinkedList second = head;
            while (counter <= k)
            {
                second = second.Next;
                counter++;
            }

            if (second == null)
            {
                //0->1->2->Null
                head.Value = head.Next.Value;
                head.Next = head.Next.Next;
                return;
            }
            while (second.Next != null)
            {
                second = second.Next;
                first = first.Next;
            }
            first.Next = first.Next.Next;

        }

        //https://www.algoexpert.io/questions/merge-linked-lists
        public static LinkedList MergeLinkedLists(LinkedList headOne, LinkedList headTwo)
        {
            //1.Iterative - T:O(n+m)| S: O(1)
            LinkedList mergedList = MergeLinkedListsIterative(headOne, headTwo);
        //2.Recursion - T:O(n+m) S:(n+m)
        TODO:
            return mergedList;
        }

        private static LinkedList MergeLinkedListsIterative(LinkedList headOne, LinkedList headTwo)
        {
            LinkedList p1 = headOne;
            LinkedList p2 = headTwo;
            LinkedList p1Prev = null;

            while (p1 != null && p2 != null)
            {
                if (p1.Value < p2.Value)
                {
                    p1Prev = p1;
                    p1 = p1.Next;
                }
                else
                {

                    if (p1Prev != null)
                        p1Prev.Next = p2;
                    p1Prev = p2;
                    p2 = p2.Next;
                    p1Prev.Next = p1;
                }
            }
            if (p1 == null) p1Prev.Next = p2;

            return headOne.Value < headTwo.Value ? headOne : headTwo;
        }
        //https://www.algoexpert.io/questions/middle-node
        public static LinkedList MiddleNode(LinkedList linkedList)
        {
            LinkedList node = default;

            //1.Naive - T:O(n) | S:O(1)
            node = MiddleNodeNaive(linkedList);

            //2.Two Pointer - T:O(n) | S:O(1) 
            node = MiddleNodeTwoPointer(linkedList);
            return node;


        }

        private static LinkedList? MiddleNodeTwoPointer(LinkedList linkedList)
        {
            LinkedList slowNode = linkedList, fastNode = linkedList;


            while (fastNode != null && fastNode.Next != null)
            {
                slowNode = slowNode.Next;
                fastNode = fastNode.Next.Next;

            }
            return slowNode;
        }

        private static LinkedList? MiddleNodeNaive(LinkedList linkedList)
        {
            int count = 0;

            LinkedList currentNode = linkedList;

            while (currentNode != null)
            {
                count++;
                currentNode = currentNode.Next;
            }
            LinkedList middleNode = linkedList;

            for (int i = 0; i < count / 2; i++)
            {
                middleNode = middleNode.Next;
            }

            return middleNode;

        }
        //https://www.algoexpert.io/questions/remove-duplicates-from-linked-list
        public static LinkedList RemoveDuplicatesFromLinkedList(LinkedList linkedList)
        {

            //T:O(n) | S:O(1)
            LinkedList currentNode = linkedList;
            while (currentNode != null)
            {
                LinkedList nextDistinctNode = currentNode.Next;
                while (nextDistinctNode != null &&
                        currentNode.Value == nextDistinctNode.Value)
                {

                    nextDistinctNode = nextDistinctNode.Next;
                }
                currentNode.Next = nextDistinctNode;
                currentNode = nextDistinctNode;
            }
            return linkedList;
        }

        public static LinkedList RemoveDuplicatesFromLinkedList1(LinkedList linkedList)
        {
            LinkedList currentNode = linkedList;
            LinkedList prevNode = null;
            int prevNodeVal = Int32.MinValue;

            while (currentNode != null)
            {
                if (prevNodeVal != Int32.MinValue && prevNodeVal == currentNode.Value)
                {
                    prevNode.Next = currentNode.Next;
                    currentNode = currentNode.Next;

                }
                else
                {
                    prevNode = currentNode;
                    prevNodeVal = currentNode.Value;
                    currentNode = currentNode.Next;
                }


            }
            // Write your code here.
            return linkedList;

        }
        //https://www.algoexpert.io/questions/merging-linked-lists
        public static LinkedList MergingLinkedLists(LinkedList linkedList1, LinkedList linkedList2)
        {

            //1.Naive - T:O(n+m) | S:O(n) - Using HashSet
            LinkedList intersectNode = MergingLinkedListsNaive(linkedList1, linkedList2);

            //2.Optimla w/ou extra space - T:O(n+m) | S:O(1) 
            intersectNode = MergingLinkedListsOptimal(linkedList1, linkedList2);

            return intersectNode;


        }

        private static LinkedList MergingLinkedListsOptimal(LinkedList linkedList1, LinkedList linkedList2)
        {
            LinkedList currentNodeOne = linkedList1;
            int countOne = 0;
            while (currentNodeOne != null)
            {
                countOne++;
                currentNodeOne = currentNodeOne.Next;
            }
            LinkedList currentNodeTwo = linkedList2;
            int countTwo = 0;
            while (currentNodeTwo != null)
            {
                countTwo++;
                currentNodeTwo = currentNodeTwo.Next;
            }

            int difference = Math.Abs(countTwo - countOne);

            LinkedList biggerCurrentNode = countOne > countTwo ? linkedList1 : linkedList2;
            LinkedList smallerCurrentNode = countOne > countTwo ? linkedList2 : linkedList1;

            for (int i = 0; i < difference; i++)
            {
                biggerCurrentNode = biggerCurrentNode.Next;
            }
            while (biggerCurrentNode != smallerCurrentNode)
            {
                biggerCurrentNode = biggerCurrentNode.Next;
                smallerCurrentNode = smallerCurrentNode.Next;
            }
            return biggerCurrentNode;
        }

        private static LinkedList MergingLinkedListsNaive(LinkedList linkedList1, LinkedList linkedList2)
        {
            HashSet<LinkedList> listOneNodeSet = new HashSet<LinkedList>();

            LinkedList currentNodeOne = linkedList1;
            while (currentNodeOne != null)
            {
                listOneNodeSet.Add(currentNodeOne);
                currentNodeOne = currentNodeOne.Next;
            }

            LinkedList currentNodeTwo = linkedList2;

            while (currentNodeTwo != null)
            {

                if (listOneNodeSet.Contains(currentNodeTwo))
                    return currentNodeTwo;
                currentNodeTwo = currentNodeTwo.Next;
            }

            return null;

        }
        //https://www.algoexpert.io/questions/find-loop
        public static LinkedList FindLoop(LinkedList head)
        {

            //1.Naive - Using Set to store nodes and loop exists if a node exists in Set
            //T:O(n) : S:O(1)

            //2.Optimal - Two pointers

            LinkedList slow = head;
            LinkedList fast = head.Next.Next;

            while (slow != fast)
            {
                slow = slow.Next;
                fast = fast.Next.Next;
            }
            slow = head;
            while (slow != fast)
            {

                slow = slow.Next;
                fast = fast.Next;
            }
            return slow;

        }
        /*
        206. Reverse Linked List
https://leetcode.com/problems/reverse-linked-list/description/ 
https://www.youtube.com/watch?v=N6dOwBde7-M
//https://www.algoexpert.io/questions/reverse-linked-list

        */
        public class ReverseLinkedListSol
        {
            /*
            Approach 1: Iterative
Time complexity : O(n).
Assume that n is the list's length, the time complexity is O(n).

Space complexity : O(1).


            */
            public static LinkedList ReverseLinkedListIterative(LinkedList head)
            {
                //T:O(n) | S:O(1)
                LinkedList prevNode = null;
                LinkedList currNode = head;

                while (currNode != null)
                {
                    LinkedList nxtNode = currNode.Next;
                    currNode.Next = prevNode;
                    prevNode = currNode;
                    currNode = nxtNode;

                }
                return prevNode;
            }

            /*
            Approach 2: Recursive

Complexity Analysis

Time complexity : O(n).
Assume that n is the list's length, the time complexity is O(n).

Space complexity : O(n).
The extra space comes from implicit stack space due to recursion. The recursion could go up to n levels deep.


            */
            public ListNode ReverseLinkedListRec(ListNode head)
            {
                if (head == null || head.Next == null)
                {
                    return head;
                }
                ListNode p = ReverseLinkedListRec(head.Next);
                head.Next.Next = head;
                head.Next = null;
                return p;
            }
        }
        //https://www.algoexpert.io/questions/node-swap
        public static LinkedList NodeSwap(LinkedList head)
        {

            //1.Recurvie with Call stack space
            //T:O(n) | S:O(n)
            LinkedList swappedNodeHead = NodeSwapOptimal1(head);

            //2.Iterative with no extra space
            //T:O(n) | S:O(1)
            swappedNodeHead = NodeSwapOptimal2(head);

            return swappedNodeHead;
        }

        private static LinkedList NodeSwapOptimal2(LinkedList head)
        {
            LinkedList tempNode = new LinkedList(0);
            tempNode.Next = head;

            LinkedList prevNode = tempNode;
            while (prevNode.Next != null && prevNode.Next.Next != null)
            {
                LinkedList firstNode = prevNode.Next;
                LinkedList secondNode = prevNode.Next.Next;

                firstNode.Next = secondNode.Next;
                secondNode.Next = firstNode;
                prevNode.Next = secondNode;

                prevNode = firstNode;
            }
            return tempNode.Next;
        }

        private static LinkedList NodeSwapOptimal1(LinkedList head)
        {
            if (head == null && head.Next == null) return head;

            LinkedList nextNode = head.Next;
            head.Next = NodeSwapOptimal1(head.Next);
            nextNode.Next = head;
            return nextNode;
        }

        //https://www.algoexpert.io/questions/rearrange-linked-list
        // O(n) time | O(1) space - where n is the number of nodes in the Linked List
        public static LinkedList RearrangeLinkedList(LinkedList head, int k)
        {
            LinkedList smallerListHead = null;
            LinkedList smallerListTail = null;
            LinkedList equalListHead = null;
            LinkedList equalListTail = null;
            LinkedList greaterListHead = null;
            LinkedList greaterListTail = null;

            LinkedList node = head;
            while (node != null)
            {
                if (node.Value < k)
                {
                    LinkedListPair smallerList =
                      growLinkedList(smallerListHead, smallerListTail, node);
                    smallerListHead = smallerList.head;
                    smallerListTail = smallerList.tail;
                }
                else if (node.Value > k)
                {
                    LinkedListPair greaterList =
                      growLinkedList(greaterListHead, greaterListTail, node);
                    greaterListHead = greaterList.head;
                    greaterListTail = greaterList.tail;
                }
                else
                {
                    LinkedListPair equalList =
                      growLinkedList(equalListHead, equalListTail, node);
                    equalListHead = equalList.head;
                    equalListTail = equalList.tail;
                }

                LinkedList prevNode = node;
                node = node.Next;
                prevNode.Next = null;
            }

            LinkedListPair firstPair = connectLinkedLists(
              smallerListHead, smallerListTail, equalListHead, equalListTail
            );
            LinkedListPair finalPair = connectLinkedLists(
              firstPair.head, firstPair.tail, greaterListHead, greaterListTail
            );
            return finalPair.head;
        }

        public static LinkedListPair growLinkedList(
          LinkedList head, LinkedList tail, LinkedList node
        )
        {
            LinkedList newHead = head;
            LinkedList newTail = node;

            if (newHead == null) newHead = node;
            if (tail != null) tail.Next = node;

            return new LinkedListPair(newHead, newTail);
        }

        public static LinkedListPair connectLinkedLists(
          LinkedList headOne,
          LinkedList tailOne,
          LinkedList headTwo,
          LinkedList tailTwo
        )
        {
            LinkedList newHead = headOne == null ? headTwo : headOne;
            LinkedList newTail = tailTwo == null ? tailOne : tailTwo;

            if (tailOne != null) tailOne.Next = headTwo;

            return new LinkedListPair(newHead, newTail);

        }
        public class LinkedListPair
        {
            public LinkedList head;
            public LinkedList tail;

            public LinkedListPair(LinkedList head, LinkedList tail)
            {
                this.head = head;
                this.tail = tail;
            }
        }


        public class LinkedList
        {
            public int Value;
            public LinkedList Next = null;

            public LinkedList(int value)
            {
                this.Value = value;
            }
        }
        //https://www.algoexpert.io/questions/sum-of-linked-lists
        // O(max(n, m)) time | O(max(n, m)) space - where n is the length of the
        // first Linked List and m is the length of the second Linked List
        public LinkedList SumOfLinkedLists(
          LinkedList linkedListOne, LinkedList linkedListTwo
        )
        {
            // This variable will store a dummy node whose .Next
            // attribute will point to the head of our new LL.
            LinkedList newLinkedListHeadPointer = new LinkedList(0);
            LinkedList currentNode = newLinkedListHeadPointer;
            int carry = 0;

            LinkedList nodeOne = linkedListOne;
            LinkedList nodeTwo = linkedListTwo;

            while (nodeOne != null || nodeTwo != null || carry != 0)
            {
                int valueOne = (nodeOne != null) ? nodeOne.Value : 0;
                int valueTwo = (nodeTwo != null) ? nodeTwo.Value : 0;
                int sumOfValues = valueOne + valueTwo + carry;

                int newValue = sumOfValues % 10;
                LinkedList newNode = new LinkedList(newValue);
                currentNode.Next = newNode;
                currentNode = newNode;

                carry = sumOfValues / 10;
                nodeOne = (nodeOne != null) ? nodeOne.Next : null;
                nodeTwo = (nodeTwo != null) ? nodeTwo.Next : null;
            }

            return newLinkedListHeadPointer.Next;
        }
        //https://www.algoexpert.io/questions/shift-linked-list
        // O(n) time | O(1) space - where n is the number of nodes in the Linked List
        public static LinkedList ShiftLinkedList(LinkedList head, int k)
        {
            int listLength = 1;
            LinkedList listTail = head;
            while (listTail.Next != null)
            {
                listTail = listTail.Next;
                listLength++;
            }

            int offset = Math.Abs(k) % listLength;
            if (offset == 0) return head;
            int newTailPosition = k > 0 ? listLength - offset : offset;
            LinkedList newTail = head;
            for (int i = 1; i < newTailPosition; i++)
            {
                newTail = newTail.Next;
            }

            LinkedList newHead = newTail.Next;
            newTail.Next = null;
            listTail.Next = head;
            return newHead;
        }


        /*
        19. Remove Nth Node From End of List
        https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/

        */
        public ListNode RemoveNthFromEnd(ListNode head, int n)
        {
            /*
Approach 1: Two pass algorithm (TP)
Complexity Analysis
•	Time complexity : O(L).
The algorithm makes two traversal of the list, first to calculate list length L and second to find the (L−n) th node. There are 2L−n operations and time complexity is O(L).	
•	Space complexity : O(1).
We only used constant extra space.
            
            */
            ListNode headNodeAfterRemovingNthNode = RemoveNthFromEndTP(head, n);
            /*
    Approach 2: One pass algorithm (OP)
    Complexity Analysis
    •	Time complexity : O(L).
    The algorithm makes one traversal of the list of L nodes. Therefore time complexity is O(L).
    •	Space complexity : O(1).
    We only used constant extra space.

            */
            headNodeAfterRemovingNthNode = RemoveNthFromEndOP(head, n);

            return headNodeAfterRemovingNthNode;

        }

        public ListNode RemoveNthFromEndTP(ListNode head, int n)
        {
            ListNode dummy = new ListNode(0);
            dummy.Next = head;
            int length = 0;
            ListNode first = head;
            while (first != null)
            {
                length++;
                first = first.Next;
            }

            length -= n;
            first = dummy;
            while (length > 0)
            {
                length--;
                first = first.Next;
            }

            first.Next = first.Next.Next;
            return dummy.Next;
        }
        public ListNode RemoveNthFromEndOP(ListNode head, int n)
        {
            ListNode dummy = new ListNode(0);
            dummy.Next = head;
            ListNode first = dummy;
            ListNode second = dummy;
            // Advances first pointer so that the gap between first and second is n
            // nodes apart
            for (int i = 1; i <= n + 1; i++)
            {
                first = first.Next;
            }

            // Move first to the end, maintaining the gap
            while (first != null)
            {
                first = first.Next;
                second = second.Next;
            }

            second.Next = second.Next.Next;
            return dummy.Next;
        }

        /*
        21. Merge Two Sorted Lists
https://leetcode.com/problems/merge-two-sorted-lists/description/	
        */
        public ListNode MergeTwoLists(ListNode list1, ListNode list2)
        {
            /*
  Approach 1: Recursion
Complexity Analysis
•	Time complexity : O(n+m)
Because each recursive call increments the pointer to l1 or l2 by one (approaching the dangling null at the end of each list), there will be exactly one call to mergeTwoLists per element in each list. Therefore, the time complexity is linear in the combined size of the lists.
•	Space complexity : O(n+m)
The first call to mergeTwoLists does not return until the ends of both l1 and l2 have been reached, so n+m stack frames consume O(n+m) space.
       
            
            */
            ListNode mergedList = MergeTwoListsRec(list1, list2);

            /*
  Approach 2: Iteration          
   Complexity Analysis
•	Time complexity : O(n+m)
Because exactly one of l1 and l2 is incremented on each loop
iteration, the while loop runs for a number of iterations equal to the
sum of the lengths of the two lists. All other work is constant, so the
overall complexity is linear.
•	Space complexity : O(1)
The iterative approach only allocates a few pointers, so it has a
constant overall memory footprint.
         
            */

            mergedList = MergeTwoListsIterative(list1, list2);

            return mergedList;

        }
        public ListNode MergeTwoListsRec(ListNode l1, ListNode l2)
        {
            if (l1 == null)
            {
                return l2;
            }
            else if (l2 == null)
            {
                return l1;
            }
            else if (l1.Val < l2.Val)
            {
                l1.Next = MergeTwoListsRec(l1.Next, l2);
                return l1;
            }
            else
            {
                l2.Next = MergeTwoListsRec(l1, l2.Next);
                return l2;
            }
        }
        public ListNode MergeTwoListsIterative(ListNode l1, ListNode l2)
        {
            // maintain an unchanging reference to node ahead of the return node.
            ListNode prehead = new ListNode(-1);
            ListNode prev = prehead;
            while (l1 != null && l2 != null)
            {
                if (l1.Val <= l2.Val)
                {
                    prev.Next = l1;
                    l1 = l1.Next;
                }
                else
                {
                    prev.Next = l2;
                    l2 = l2.Next;
                }

                prev = prev.Next;
            }

            // At least one of l1 and l2 can still have nodes at this point, so
            // connect the non-null list to the end of the merged list.
            prev.Next = l1 == null ? l2 : l1;
            return prehead.Next;
        }

        /*
  23. Merge k Sorted Lists
https://leetcode.com/problems/merge-k-sorted-lists/description/
      

        */
        public ListNode MergeKLists(ListNode[] lists)
        {

            /*
     Approach 1: Brute Force       
     Complexity Analysis
    •	Time complexity : O(NlogN) where N is the total number of nodes.
    o	Collecting all the values costs O(N) time.
    o	A stable sorting algorithm costs O(NlogN) time.
    o	Iterating for creating the linked list costs O(N) time.
    •	Space complexity : O(N).
    o	Sorting cost O(N) space (depends on the algorithm you choose).
    o	Creating a new linked list costs O(N) space.

            */
            ListNode mergedNode = MergeKListsNaive(lists);

            /*
     Approach 2: Compare one by one       
    Complexity Analysis
    •	Time complexity : O(kN) where k is the number of linked lists.
    o	Almost every selection of node in final linked costs O(k) (k-1 times comparison).
    o	There are N nodes in the final linked list.
    •	Space complexity :
    o	O(n) Creating a new linked list costs O(n) space.
    o	O(1) It's not hard to apply in-place method - connect selected nodes instead of creating new nodes to fill the new linked list

            */

            /*
    Approach 3: Optimize Approach 2 by Priority Queue
    Complexity Analysis
    •	Time complexity : O(Nlogk) where k is the number of linked lists.
    o	The comparison cost will be reduced to O(logk) for every pop and insertion to priority queue. But finding the node with the smallest value just costs O(1) time.
    o	There are N nodes in the final linked list.
    •	Space complexity :
    o	O(n) Creating a new linked list costs O(n) space.
    o	O(k) The code above present applies in-place method which cost O(1) space. And the priority queue (often implemented with heaps) costs O(k) space (it's far less than N in most situations).

            */
            mergedNode = MergeKListsPQ(lists);
            /*
      Approach 4: Merge lists one by one      
    Complexity Analysis
    •	Time complexity : O(kN) where k is the number of linked lists.
    o	We can merge two sorted linked list in O(n) time where n is the total number of nodes in two lists.
    o	Sum up the merge process and we can get: O(∑i=1k−1(i∗(kN)+kN))=O(kN).
    •	Space complexity : O(1)
    o	We can merge two sorted linked list in O(1) space.

            */

            /*
    Approach 5: Merge with Divide And Conquer (DAC)
    Complexity Analysis
    •	Time complexity : O(Nlogk) where k is the number of linked lists.
    o	We can merge two sorted linked list in O(n) time where n is the total number of nodes in two lists.
    o	Sum up the merge process and we can get: O(∑i=1 to log2k(N))=O(Nlogk)
    •	Space complexity : O(1)
    o	We can merge two sorted linked lists in O(1) space

            */
            mergedNode = MergeKListsDAC(lists);

            return mergedNode;

        }
        public ListNode MergeKListsNaive(ListNode[] lists)
        {
            List<int> nodes = new List<int>();
            ListNode head = new ListNode(0);
            ListNode point = head;
            foreach (ListNode listNode in lists)
            {
                ListNode list = listNode;
                while (list != null)
                {
                    nodes.Add(list.Val);
                    list = list.Next;
                }
            }

            nodes.Sort();
            foreach (int val in nodes)
            {
                point.Next = new ListNode(val);
                point = point.Next;
            }

            return head.Next;
        }

        public ListNode MergeKListsPQ(ListNode[] lists)
        {
            ListNode head = new ListNode(0);
            ListNode point = head;
            var q = new PriorityQueue<ListNode, int>();

            foreach (var l in lists)
            {
                if (l != null)
                {
                    q.Enqueue(l, l.Val);
                }
            }

            while (q.Count > 0)
            {
                point.Next = q.Dequeue();
                point = point.Next;
                if (point.Next != null)
                {
                    q.Enqueue(point.Next, point.Next.Val);
                }
            }

            return head.Next;
        }
        public ListNode MergeKListsDAC(ListNode[] lists)
        {
            int amount = lists.Length;
            int interval = 1;
            while (interval < amount)
            {
                for (int i = 0; i < amount - interval; i += interval * 2)
                {
                    lists[i] = Merge2Lists(lists[i], lists[i + interval]);
                }

                interval *= 2;
            }

            return amount > 0 ? lists[0] : null;
        }

        public ListNode Merge2Lists(ListNode l1, ListNode l2)
        {
            ListNode head = new ListNode(0);
            ListNode point = head;
            while (l1 != null && l2 != null)
            {
                if (l1.Val <= l2.Val)
                {
                    point.Next = l1;
                    l1 = l1.Next;
                }
                else
                {
                    point.Next = l2;
                    l2 = l1;
                    l1 = point.Next.Next;
                }

                point = point.Next;
            }

            if (l1 == null)
                point.Next = l2;
            else
                point.Next = l1;
            return head.Next;
        }

        /*
        24. Swap Nodes in Pairs
        https://leetcode.com/problems/swap-nodes-in-pairs/description/

        */
        public ListNode SwapPairs(ListNode head)
        {
            /*
Approach 1: Recursive Approach
 Complexity Analysis
•	Time Complexity: O(N) where N is the size of the linked list.
•	Space Complexity: O(N) stack space utilized for recursion
           
            */
            ListNode swappedNodes = SwapPairsRec(head);
            /*
Approach 2: Iterative Approach            
Complexity Analysis
•	Time Complexity : O(N) where N is the size of the linked list.
•	Space Complexity : O(1).

            */
            swappedNodes = SwapPairsIterative(head);

            return swappedNodes;

        }
        public ListNode SwapPairsRec(ListNode head)
        {
            // If the list has no node or has only one node left.
            if ((head == null) || (head.Next == null))
            {
                return head;
            }

            // Nodes to be swapped
            ListNode firstNode = head;
            ListNode secondNode = head.Next;
            // Swapping
            firstNode.Next = SwapPairs(secondNode.Next);
            secondNode.Next = firstNode;
            // Now the head is the second node
            return secondNode;
        }
        public ListNode SwapPairsIterative(ListNode head)
        {
            // Dummy node acts as the prevNode for the head node
            // of the list and hence stores pointer to the head node.
            ListNode dummy = new ListNode(-1);
            dummy.Next = head;
            ListNode prevNode = dummy;
            while ((head != null) && (head.Next != null))
            {
                // Nodes to be swapped
                ListNode firstNode = head;
                ListNode secondNode = head.Next;
                // Swapping
                prevNode.Next = secondNode;
                firstNode.Next = secondNode.Next;
                secondNode.Next = firstNode;
                // Reinitializing the head and prevNode for next swap
                prevNode = firstNode;
                head = firstNode.Next;  // jump
            }

            // Return the new head node.
            return dummy.Next;
        }
        /*
        25. Reverse Nodes in k-Group
https://leetcode.com/problems/reverse-nodes-in-k-group/description/

        */
        public ListNode ReverseKGroup(ListNode head, int k)
        {
            /*
Approach 1: Recursion
Complexity Analysis
•	Time Complexity: O(N) since we process each node exactly twice. Once when we are counting the number of nodes in each recursive call, and then once when we are actually reversing the sub-list. A slightly optimized implementation here could be that we don't count the number of nodes at all and simply reverse k nodes. If at any point we find that we didn't have enough nodes, we can re-reverse the last set of nodes so as to keep the original structure as required by the problem statement. That ways, we can get rid of the extra counting.
•	Space Complexity: O(N/k) used up by the recursion stack. The number of recursion calls is determined by both k and N. In every recursive call, we process k nodes and then make a recursive call to process the rest.

            */
            ListNode headAfterReverse = ReverseKGroupRec(head, k);
            /*
Approach 2: Iterative O(1) space

•	Time Complexity: O(N) since we process each node exactly twice. Once when we are counting the number of nodes in each recursive call, and then once when we are actually reversing the sub-list.
•	Space Complexity: O(1).
            */
            headAfterReverse = ReverseKGroupIterative(head, k);

            return headAfterReverse;

        }
        public ListNode ReverseLinkedList(ListNode head, int k)
        {
            ListNode new_head = null;
            ListNode ptr = head;
            while (k > 0)
            {
                ListNode next_node = ptr.Next;
                ptr.Next = new_head;
                new_head = ptr;
                ptr = next_node;
                k--;
            }

            return new_head;
        }

        public ListNode ReverseKGroupRec(ListNode head, int k)
        {
            int count = 0;
            ListNode ptr = head;
            while (count < k && ptr != null)
            {
                ptr = ptr.Next;
                count++;
            }

            if (count == k)
            {
                ListNode reversedHead = this.ReverseLinkedList(head, k);
                head.Next = this.ReverseKGroupRec(ptr, k);
                return reversedHead;
            }

            return head;
        }


        public ListNode ReverseKGroupIterative(ListNode head, int k)
        {
            ListNode ptr = head;
            ListNode ktail = null;
            ListNode newHead = null;
            while (ptr != null)
            {
                int count = 0;
                ptr = head;
                while (count < k && ptr != null)
                {
                    ptr = ptr.Next;
                    count += 1;
                }

                if (count == k)
                {
                    ListNode revHead = this.ReverseLinkedList(head, k);
                    if (newHead == null)
                    {
                        newHead = revHead;
                    }

                    if (ktail != null)
                    {
                        ktail.Next = revHead;
                    }

                    ktail = head;
                    head = ptr;
                }
            }

            if (ktail != null)
            {
                ktail.Next = head;
            }

            return newHead == null ? head : newHead;
        }
        /*
        61. Rotate List
        https://leetcode.com/problems/rotate-list/description/

        Complexity Analysis
        •	Time complexity : O(N) where N is a number of elements in the list.
        •	Space complexity : O(1) since it's a constant space solution.


        */
        public ListNode RotateList(ListNode head, int k)
        {
            // base cases
            if (head == null)
                return null;
            if (head.Next == null)
                return head;
            // close the linked list into the ring
            ListNode old_tail = head;
            int n;
            for (n = 1; old_tail.Next != null; n++) old_tail = old_tail.Next;
            old_tail.Next = head;
            // find new tail : (n - k % n - 1)th node
            // and new head : (n - k % n)th node
            ListNode new_tail = head;
            for (int i = 0; i < n - k % n - 1; i++) new_tail = new_tail.Next;
            ListNode new_head = new_tail.Next;
            // break the ring
            new_tail.Next = null;
            return new_head;
        }

        /*
        83. Remove Duplicates from Sorted List
https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/

Approach 1: Straight-Forward Approach
Complexity Analysis
•	Time complexity : O(n). Because each node in the list is checked exactly once to determine if it is a duplicate or not, the total run time is O(n), where n is the number of nodes in the list.
•	Space complexity : O(1). No additional space is used.

        */
        public ListNode DeleteDuplicatesSortedList(ListNode head)
        {
            ListNode current = head;
            while (current != null && current.Next != null)
            {
                if (current.Next.Val == current.Val)
                {
                    current.Next = current.Next.Next;
                }
                else
                {
                    current = current.Next;
                }
            }

            return head;
        }

        /*
        82. Remove Duplicates from Sorted List II
https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/

Approach 1: Sentinel Head + Predecessor
Complexity Analysis
•	Time complexity: O(N) since it's one pass along the input list.
•	Space complexity: O(1) because we don't allocate any additional data structure.

        */
        public ListNode DeleteDuplicatesSortedListII(ListNode head)
        {
            // sentinel
            ListNode sentinel = new ListNode(0, head);
            // predecessor = the last node
            // before the sublist of duplicates
            ListNode pred = sentinel;
            while (head != null)
            {
                // If it's a beginning of duplicates sublist
                // skip all duplicates
                if (head.Next != null && head.Val == head.Next.Val)
                {
                    // move till the end of duplicates sublist
                    while (head.Next != null && head.Val == head.Next.Val)
                    {
                        head = head.Next;
                    }

                    // Skip all duplicates
                    pred.Next = head.Next;
                    // otherwise, move predecessor
                }
                else
                {
                    pred = pred.Next;
                }

                // move forward
                head = head.Next;
            }

            return sentinel.Next;
        }

        /*
        86. Partition List
https://leetcode.com/problems/partition-list/description/

Approach 1: Two Pointer Approach

Complexity Analysis
•	Time Complexity: O(N), where N is the number of nodes in the original
linked list and we iterate the original list.
•	Space Complexity: O(1), we have not utilized any extra space, the point to
note is that we are reforming the original list, by moving the original nodes, we
have not used any extra space as such.

        */
        public ListNode PartitionList(ListNode head, int x)
        {
            ListNode before_head = new ListNode(0);
            ListNode before = before_head;
            ListNode after_head = new ListNode(0);
            ListNode after = after_head;
            while (head != null)
            {
                if (head.Val < x)
                {
                    before.Next = head;
                    before = before.Next;
                }
                else
                {
                    after.Next = head;
                    after = after.Next;
                }

                head = head.Next;
            }

            after.Next = null;
            before.Next = after_head.Next;
            return before_head.Next;
        }


        /*
        92. Reverse Linked List II
        https://leetcode.com/problems/reverse-linked-list-ii/description/
        */
        public class ReverseListBetweenSol
        {
            /*
            Approach 1: Recursion
            Complexity Analysis
•	Time Complexity: O(N) since we process all the nodes at-most twice. Once during the normal recursion process and once during the backtracking process. During the backtracking process we only just swap half of the list if you think about it, but the overall complexity is O(N).
•	Space Complexity: O(N) in the worst case when we have to reverse the entire list. This is the space occupied by the recursion stack.


            */
            public ListNode ReverseListBetweenRec(ListNode head, int m, int n)
            {
                stop = false;
                left = head;
                RecurseAndReverse(head, m, n);
                return head;

            }
            private ListNode left = null;
            private bool stop = false;

            private void RecurseAndReverse(ListNode right, int m, int n)
            {
                if (n == 1)
                    return;
                right = right.Next;
                if (m > 1)
                    this.left = this.left.Next;
                this.RecurseAndReverse(right, m - 1, n - 1);
                if (this.left == right || right.Next == this.left)
                    this.stop = true;
                if (!this.stop)
                {
                    int tmp = this.left.Val;
                    this.left.Val = right.Val;
                    right.Val = tmp;
                    this.left = this.left.Next;
                }
            }

            /*
            Approach 2: Iterative Link Reversal. (ILR)

            Complexity Analysis
•	Time Complexity: O(N) considering the list consists of N nodes. We process each of the nodes at most once (we don't process the nodes after the nth node from the beginning.
•	Space Complexity: O(1) since we simply adjust some pointers in the original linked list and only use O(1) additional memory for achieving the final result.

            */
            public ListNode ReverseListBetweenILR(ListNode head, int m, int n)
            {
                // Empty list
                if (head == null)
                {
                    return null;
                }

                // Move the two pointers until they reach the proper starting point
                // in the list.
                ListNode cur = head, prev = null;
                while (m > 1)
                {
                    prev = cur;
                    cur = cur.Next;
                    m--;
                    n--;
                }

                // The two pointers that will fix the final connections.
                ListNode con = prev, tail = cur;
                // Iteratively reverse the nodes until n becomes 0.
                ListNode third = null;
                while (n > 0)
                {
                    third = cur.Next;
                    cur.Next = prev;
                    prev = cur;
                    cur = third;
                    n--;
                }

                // Adjust the final connections as explained in the algorithm
                if (con != null)
                {
                    con.Next = prev;
                }
                else
                {
                    head = prev;
                }

                tail.Next = cur;
                return head;
            }

        }

        /* 138. Copy List with Random Pointer
        https://leetcode.com/problems/copy-list-with-random-pointer/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class CopyListWithRandomPointerSol
        {
            // Dictionary which holds old nodes as keys and new nodes as its values.
            private Dictionary<Node, Node> visited = new Dictionary<Node, Node>();

            /* Approach 1: Recursive
            Complexity Analysis
            •	Time Complexity: O(N) where N is the number of nodes in the linked list.
            •	Space Complexity: O(N). If we look closely, we have the recursion stack and we also have the space complexity to keep track of nodes already cloned i.e. using the visited dictionary. But asymptotically, the complexity is O(N).

             */
            public Node CopyRandomListRec(Node head)
            {
                if (head == null)
                {
                    return null;
                }

                // If we have already processed the current node, then we simply return
                // the cloned version of it.
                if (this.visited.ContainsKey(head))
                {
                    return this.visited[head];
                }

                // Create a new node with the value same as old node. (i.e., copy the
                // node)
                Node node = new Node(head.val, null, null);
                // Save this value in the hash map. This is needed since there might be
                // loops during traversal due to randomness of random pointers and this
                // would help us avoid them.
                this.visited[head] = node;
                // Recursively copy the remaining linked list starting once from the
                // next pointer and then from the random pointer. Thus we have two
                // independent recursive calls. Finally we update the next and random
                // pointers for the new node created.
                node.next = this.CopyRandomListRec(head.next);
                node.random = this.CopyRandomListRec(head.random);
                return node;
            }
            public class Node
            {
                public int val;
                public Node next;
                public Node random;
                public Node(int _val, Node _next, Node _random)
                {
                    val = _val;
                    next = _next;
                    random = _random;
                }
                public Node(int _val)
                {
                    val = _val;
                    next = null;
                    random = null;
                }
            }
            /* Approach 2: Iterative with O(N) Space
            Complexity Analysis
•	Time Complexity : O(N) because we make one pass over the original linked list.
•	Space Complexity : O(N) as we have a dictionary containing mapping from old list nodes to new list nodes. Since there are N nodes, we have O(N) space complexity.

             */
            public Node CopyRandomListIterative(Node head)
            {
                if (head == null)
                {
                    return null;
                }

                Node oldNode = head;
                Node newNode = new Node(oldNode.val, null, null);
                visited[oldNode] = newNode;
                while (oldNode != null)
                {
                    newNode.random = this.GetClonedNode(oldNode.random);
                    newNode.next = this.GetClonedNode(oldNode.next);
                    oldNode = oldNode.next;
                    newNode = newNode.next;
                }

                return visited[head];
            }
            public Node GetClonedNode(Node node)
            {
                if (node != null)
                {
                    if (visited.ContainsKey(node))
                    {
                        return visited[node];
                    }
                    else
                    {
                        visited[node] = new Node(node.val, null, null);
                        return visited[node];
                    }
                }

                return null;
            }
            /* Approach 3: Iterative with O(1) Space
            Complexity Analysis
•	Time Complexity : O(N)
•	Space Complexity : O(1)

             */
            public Node CopyRandomListIterativeWithConstanceSpace(Node head)
            {
                if (head == null)
                {
                    return null;
                }

                // Creating a new weaved list of original and copied nodes.
                Node ptr = head;
                while (ptr != null)
                {
                    // Cloned node
                    Node newNode = new Node(ptr.val);

                    // Inserting the cloned node just next to the original node.
                    // If A->B->C is the original linked list,
                    // Linked list after weaving cloned nodes would be
                    // A->A'->B->B'->C->C'
                    newNode.next = ptr.next;
                    ptr.next = newNode;
                    ptr = newNode.next;
                }

                ptr = head;

                // Now link the random pointers of the new nodes created.
                // Iterate the newly created list and use the original nodes' random
                // pointers, to assign references to random pointers for cloned nodes.
                while (ptr != null)
                {
                    ptr.next.random = (ptr.random != null) ? ptr.random.next : null;
                    ptr = ptr.next.next;
                }

                // Unweave the linked list to get back the original linked list and the
                // cloned list. i.e. A->A'->B->B'->C->C' would be broken to A->B->C and
                // A'->B'->C'
                Node ptr_old_list = head;       // A->B->C
                Node ptr_new_list = head.next;  // A'->B'->C'
                Node head_old = head.next;
                while (ptr_old_list != null)
                {
                    ptr_old_list.next = ptr_old_list.next.next;
                    ptr_new_list.next =
                        (ptr_new_list.next != null) ? ptr_new_list.next.next : null;
                    ptr_old_list = ptr_old_list.next;
                    ptr_new_list = ptr_new_list.next;
                }

                return head_old;
            }

        }


        /* 708. Insert into a Sorted Circular Linked List
        https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class InsertIntoSortedCircularLinkedListSol
        {
            /*             Approach 1: Two-Pointers Iteration
            Complexity Analysis
            •	Time Complexity: O(N) where N is the size of list. In the worst case, we would iterate through the entire list.
            •	Space Complexity: O(1). It is a constant space solution.

             */
            public DataStructures.ListNode Insert(DataStructures.ListNode head, int insertVal)
            {
                if (head == null)
                {
                    DataStructures.ListNode newNode = new DataStructures.ListNode(insertVal);
                    newNode.Next = newNode;
                    return newNode;
                }

                DataStructures.ListNode prev = head;
                DataStructures.ListNode curr = head.Next;
                bool toInsert = false;

                do
                {
                    if (prev.Val <= insertVal && insertVal <= curr.Val)
                    {
                        // Case 1).
                        toInsert = true;
                    }
                    else if (prev.Val > curr.Val)
                    {
                        // Case 2).
                        if (insertVal >= prev.Val || insertVal <= curr.Val)
                            toInsert = true;
                    }

                    if (toInsert)
                    {
                        prev.Next = new DataStructures.ListNode(insertVal, curr);
                        return head;
                    }

                    prev = curr;
                    curr = curr.Next;
                } while (prev != head);

                // Case 3).
                prev.Next = new DataStructures.ListNode(insertVal, curr);
                return head;
            }
        }


        /* 2. Add Two Numbers
        https://leetcode.com/problems/add-two-numbers/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class AddTwoNumbersSol
        {
            /*             Approach 1: Elementary Math
            Complexity Analysis
            •	Time complexity : O(max(m,n)). Assume that m and n represents the length of l1 and l2 respectively, the algorithm above iterates at most max(m,n) times.
            •	Space complexity : O(1). The length of the new list is at most max(m,n)+1 However, we don't count the answer as part of the space complexity.

             */
            public ListNode AddTwoNumbers(ListNode l1, ListNode l2)
            {
                ListNode dummyHead = new ListNode(0);
                ListNode curr = dummyHead;
                int carry = 0;
                while (l1 != null || l2 != null || carry != 0)
                {
                    int x = (l1 != null) ? l1.Val : 0;
                    int y = (l2 != null) ? l2.Val : 0;
                    int sum = carry + x + y;
                    carry = sum / 10;
                    curr.Next = new ListNode(sum % 10);
                    curr = curr.Next;
                    if (l1 != null)
                        l1 = l1.Next;
                    if (l2 != null)
                        l2 = l2.Next;
                }

                return dummyHead.Next;
            }
        }


        /* 2058. Find the Minimum and Maximum Number of Nodes Between Critical Points
        https://leetcode.com/problems/find-the-minimum-and-maximum-number-of-nodes-between-critical-points/description/
         */
        class NodesBetweenCriticalPointsSol
        {
            /* 
            Approach: One Pass 
            Complexity Analysis
Let n be the the length of the linked list.
•	Time complexity: O(n)
The algorithm traverses the list only once, making the time complexity O(n).
•	Space complexity: O(1)
The algorithm has a constant space complexity since it does not utilize any additional data structures.

            */
            public int[] UsingOnePass(ListNode head)
            {
                int[] result = { -1, -1 };

                // Initialize minimum distance to the maximum possible value
                int minDistance = int.MaxValue;

                // Pointers to track the previous node, current node, and indices
                ListNode previousNode = head;
                ListNode currentNode = head.Next;
                int currentIndex = 1;
                int previousCriticalIndex = 0;
                int firstCriticalIndex = 0;

                while (currentNode.Next != null)
                {
                    // Check if the current node is a local maxima or minima
                    if (
                        (currentNode.Val < previousNode.Val &&
                            currentNode.Val < currentNode.Next.Val) ||
                        (currentNode.Val > previousNode.Val &&
                            currentNode.Val > currentNode.Next.Val)
                    )
                    {
                        // If this is the first critical point found
                        if (previousCriticalIndex == 0)
                        {
                            previousCriticalIndex = currentIndex;
                            firstCriticalIndex = currentIndex;
                        }
                        else
                        {
                            // Calculate the minimum distance between critical points
                            minDistance = Math.Min(
                                minDistance,
                                currentIndex - previousCriticalIndex
                            );
                            previousCriticalIndex = currentIndex;
                        }
                    }

                    // Move to the next node and update indices
                    currentIndex++;
                    previousNode = currentNode;
                    currentNode = currentNode.Next;
                }

                // If at least two critical points were found
                if (minDistance != int.MaxValue)
                {
                    int maxDistance = previousCriticalIndex - firstCriticalIndex;
                    result = new int[] { minDistance, maxDistance };
                }

                return result;
            }
        }


        /* 2487. Remove Nodes From Linked List
        https://leetcode.com/problems/remove-nodes-from-linked-list/description/	
         */

        class RemoveNodesSol
        {
            /* Approach 1: Stack
Complexity Analysis
Let n be the length of the original linked list.
•	Time complexity: O(n)
Adding the nodes from the original linked list to the stack takes O(n).
Removing nodes from the stack and adding them to the result takes O(n), as each node is popped from the stack exactly once.
Therefore, the time complexity is O(2n), which simplifies to O(n).
•	Space complexity: O(n)
We add each of the nodes from the original linked list to the stack, making its size n.
We only use resultList to store the result, so it does not contribute to the space complexity.
Therefore, the space complexity is O(n).

             */
            public ListNode UsingStack(ListNode head)
            {
                Stack<ListNode> stack = new();
                ListNode current = head;

                // Add nodes to the stack
                while (current != null)
                {
                    stack.Push(current);
                    current = current.Next;
                }

                current = stack.Pop();
                int maximum = current.Val;
                ListNode resultList = new ListNode(maximum);

                // Remove nodes from the stack and add to result
                while (stack.Count > 0)
                {
                    current = stack.Pop();
                    // Current should not be added to the result
                    if (current.Val < maximum)
                    {
                        continue;
                    }
                    // Add new node with current's value to front of the result
                    else
                    {
                        ListNode newNode = new ListNode(current.Val);
                        newNode.Next = resultList;
                        resultList = newNode;
                        maximum = current.Val;
                    }
                }

                return resultList;
            }

            /* Approach 2: Recursion
Complexity Analysis
Let n be the length of the original linked list.
•	Time complexity: O(n)
We call removeNodes() once for each node in the original linked list. The other operations inside the function all take constant time, so the time complexity is dominated by the recursive calls. Thus, the time complexity is O(n).
•	Space complexity: O(n)
Since we make n recursive calls to removeNodes(), the call stack can grow up to size n. Therefore, the space complexity is O(n).

            */
            public ListNode UsingRecursion(ListNode head)
            {
                // Base case, reached end of the list
                if (head == null || head.Next == null)
                {
                    return head;
                }

                // Recursive call
                ListNode nextNode = UsingRecursion(head.Next);

                // If the next node has greater value than head, delete the head
                // Return next node, which removes the current head and makes next the new head
                if (head.Val < nextNode.Val)
                {
                    return nextNode;
                }

                // Keep the head
                head.Next = nextNode;
                return head;
            }
            /* Approach 3: Reverse Twice
Complexity Analysis
Let n be the length of the original linked list.
•	Time complexity: O(n)
Reversing the original linked list takes O(n).
Traversing the reversed original linked list and removing nodes takes O(n).
Reversing the modified linked list takes an additional O(n) time.
Therefore, the total time complexity is O(3n), which simplifies to O(n).
•	Space complexity: O(1)
We use a few variables and pointers that use constant extra space. Since we don't use any data structures that grow with input size, the space complexity remains O(1).	

             */
            public ListNode UsingReverseTwice(ListNode head)
            {
                // Reverse the original linked list
                head = ReverseList(head);

                int maximum = 0;
                ListNode prev = null;
                ListNode current = head;

                // Traverse the list deleting nodes
                while (current != null)
                {
                    maximum = Math.Max(maximum, current.Val);

                    // Delete nodes that are smaller than maximum
                    if (current.Val < maximum)
                    {
                        // Delete current by skipping
                        prev.Next = current.Next;
                        ListNode deleted = current;
                        current = current.Next;
                        deleted.Next = null;
                    }

                    // Current does not need to be deleted
                    else
                    {
                        prev = current;
                        current = current.Next;
                    }
                }

                // Reverse and return the modified linked list
                return ReverseList(head);
            }
            private ListNode ReverseList(ListNode head)
            {
                ListNode prev = null;
                ListNode current = head;
                ListNode nextTemp = null;

                // Set each node's next pointer to the previous node
                while (current != null)
                {
                    nextTemp = current.Next;
                    current.Next = prev;
                    prev = current;
                    current = nextTemp;
                }

                return prev;
            }


        }

        /* 725. Split Linked List in Parts
        https://leetcode.com/problems/split-linked-list-in-parts/description/
         */

        class SplitListToPartsSol
        {
            /* 
            Approach 1: Create New Parts
            Complexity Analysis
            Let N be the size of the linked list head.
            •	Time Complexity: O(N)
            We traverse the entire linked list head twice, where each time takes O(N) time. Thus, the total time complexity is O(N).
            •	Space Complexity: O(N)
            There are N new nodes created. This results in a space complexity of O(N). We ignore the O(K) space needed for ans since the array is required for the question.

             */
            public ListNode[] UsingCreateNewParts(ListNode head, int k)
            {
                ListNode[] ans = new ListNode[k];

                // get total size of linked list
                int size = 0;
                ListNode current = head;
                while (current != null)
                {
                    size++;
                    current = current.Next;
                }

                // minimum size for the k parts
                int splitSize = size / k;

                // Remaining nodes after splitting the k parts evenly.
                // These will be distributed to the first (size % k) nodes
                int numRemainingParts = size % k;

                current = head;
                for (int i = 0; i < k; i++)
                {
                    // create the i-th part
                    ListNode newPart = new ListNode(0);
                    ListNode tail = newPart;

                    int currentSize = splitSize;
                    if (numRemainingParts > 0)
                    {
                        numRemainingParts--;
                        currentSize++;
                    }
                    int j = 0;
                    while (j < currentSize)
                    {
                        tail.Next = new ListNode(current.Val);
                        tail = tail.Next;
                        j++;
                        current = current.Next;
                    }
                    ans[i] = newPart.Next;
                }

                return ans;
            }
            /* Approach 2: Modify Linked List
            Complexity Analysis
Let N be the size of the linked list head.
•	Time Complexity: O(N)
head is traversed twice, which takes O(N) time.
•	Space Complexity: O(1)
In contrast to Approach 1, no new nodes are created and the input is modified to create k parts. Thus, the space complexity is a constant O(1).

             */
            public ListNode[] UsingModifyLinkedList(ListNode head, int k)
            {
                ListNode[] ans = new ListNode[k];

                // get total size of linked list
                int size = 0;
                ListNode current = head;
                while (current != null)
                {
                    size++;
                    current = current.Next;
                }

                // minimum size for the k parts
                int splitSize = size / k;

                // Remaining nodes after splitting the k parts evenly.
                // These will be distributed to the first (size % k) nodes
                int numRemainingParts = size % k;

                current = head;
                ListNode prev = current;
                for (int i = 0; i < k; i++)
                {
                    // create the i-th part
                    ListNode newPart = current;
                    // calculate size of i-th part
                    int currentSize = splitSize;
                    if (numRemainingParts > 0)
                    {
                        numRemainingParts--;
                        currentSize++;
                    }

                    // traverse to end of new part
                    int j = 0;
                    while (j < currentSize)
                    {
                        prev = current;
                        current = current.Next;
                        j++;
                    }
                    // cut off the rest of linked list
                    if (prev != null)
                    {
                        prev.Next = null;
                    }

                    ans[i] = newPart;
                }

                return ans;
            }

        }


        /* 2181. Merge Nodes in Between Zeros
        https://leetcode.com/problems/merge-nodes-in-between-zeros/description/
         */

        public class MergeNodesInBetweenZerosSol
        {

            /* Approach 1: Two-Pointer (One-Pass)	
Complexity Analysis
Let n be the size of the linked list.
•	Time complexity: O(n)
All the nodes of the linked list are visited exactly once. Therefore, the total time complexity is given by O(n).
•	Space complexity: O(1)
Apart from the original list, we don't use any additional space. Therefore, the total space complexity is given by O(1).	

             */
            public ListNode UsingTwoPointersWithOnePass(ListNode head)
            {
                // Initialize a sentinel/dummy node with the first non-zero value.
                ListNode modify = head.Next;
                ListNode nextSum = modify;

                while (nextSum != null)
                {
                    int sum = 0;
                    // Find the sum of all nodes until you encounter a 0.
                    while (nextSum.Val != 0)
                    {
                        sum += nextSum.Val;
                        nextSum = nextSum.Next;
                    }

                    // Assign the sum to the current node's value.
                    modify.Val = sum;
                    // Move nextSum to the first non-zero value of the next block.
                    nextSum = nextSum.Next;
                    // Move modify also to this node.
                    modify.Next = nextSum;
                    modify = modify.Next;
                }
                return head.Next;
            }
            /* Approach 2: Recursion
Complexity Analysis
Let n be the size of the linked list.
•	Time complexity: O(n)
All the nodes of the linked list are visited exactly once. Therefore, the total time complexity is given by O(n).
•	Space complexity: O(n)
The extra space comes from implicit stack space due to recursion. The recursion could go up to n levels deep. Therefore, the total space complexity is given by O(n).

             */
            public ListNode UsingRecursion(ListNode head)
            {
                // Start with the first non-zero value.
                head = head.Next;
                if (head == null)
                {
                    return head;
                }

                // Initialize a dummy head node.
                ListNode temp = head;
                int sum = 0;
                while (temp.Val != 0)
                {
                    sum += temp.Val;
                    temp = temp.Next;
                }

                // Store the sum in head's value.
                head.Val = sum;
                // Store head's next node as the recursive result for temp node.
                head.Next = UsingRecursion(temp);
                return head;
            }
        }

        /* 237. Delete Node in a Linked List
        https://leetcode.com/problems/delete-node-in-a-linked-list/description/
         */
        class DeleteNodeSol
        {

            /* 
            Approach: Data Overwriting

            Complexity Analysis
            •	Time Complexity: O(1)
            o	The method involves a constant number of operations: updating the data of the current node and altering its next pointer. Each of these operations requires a fixed amount of time, irrespective of the size of the linked list.
            •	Space Complexity: O(1)
            o	This deletion technique does not necessitate any extra memory allocation, as it operates directly on the existing nodes without creating additional data structures.
             */
            public void UsingDataOverwriting(ListNode node)
            {
                // Overwrite data of next node on current node.
                node.Val = node.Next.Val;
                // Make current node point to next of next node.
                node.Next = node.Next.Next;
            }
        }
        /* 
        1367. Linked List in Binary Tree
        https://leetcode.com/problems/linked-list-in-binary-tree/description/
         */
        class IsSubPathSol
        {
            /* Approach 1: DFS
            Complexity Analysis
Let n be the number of nodes in the tree and m be the length of the linked list.
•	Time complexity: O(n×m)
In the worst case, we might need to check every node in the tree as a potential starting point for the linked list. For each node, we might need to traverse up to m nodes in the linked list.
•	Space complexity: O(n+m)
The space complexity remains the same as Approach 1 due to the recursive nature of the solution.

             */
            public bool UsingDFS(ListNode head, TreeNode root)
            {
                if (root == null) return false;
                return CheckPath(root, head);
            }

            private bool CheckPath(TreeNode node, ListNode head)
            {
                if (node == null) return false;
                if (Dfs(node, head)) return true; // If a matching path is found
                                                  // Recursively check left and right subtrees
                return CheckPath(node.Left, head) || CheckPath(node.Right, head);
            }

            private bool Dfs(TreeNode node, ListNode head)
            {
                if (head == null) return true; // All nodes in the list matched
                if (node == null) return false; // Reached end of tree without matching all nodes
                if (node.Val != head.Val) return false; // Value mismatch
                return Dfs(node.Left, head.Next) || Dfs(node.Right, head.Next);
            }

            /* Approach 2: Iterative Approach
            	Complexity Analysis
Let n be the number of nodes in the tree and m be the length of the linked list.
•	Time complexity: O(n×m)
We potentially visit each node in the tree once. For each node, we might need to check up to m nodes in the linked list.
•	Space complexity: O(n)
The space is used by the stack, which in the worst case might contain all nodes of the tree. We don't need extra space for the linked list traversal as it's done iteratively.

             */
            public bool UsingIterative(ListNode head, TreeNode root)
            {
                if (root == null) return false;

                Stack<TreeNode> nodes = new Stack<TreeNode>();
                nodes.Push(root);

                while (nodes.Count != 0)
                {
                    TreeNode node = nodes.Pop();

                    if (IsMatch(node, head))
                    {
                        return true;
                    }

                    if (node.Left != null)
                    {
                        nodes.Push(node.Left);
                    }
                    if (node.Right != null)
                    {
                        nodes.Push(node.Right);
                    }
                }

                return false;
            }

            private bool IsMatch(TreeNode node, ListNode lst)
            {
                Stack<KeyValuePair<TreeNode, ListNode>> s = new Stack<KeyValuePair<TreeNode, ListNode>>();
                s.Push(new KeyValuePair<TreeNode, ListNode>(node, lst));

                while (s.Count != 0)
                {
                    KeyValuePair<TreeNode, ListNode> entry = s.Pop();
                    TreeNode currentNode = entry.Key;
                    ListNode currentList = entry.Value;

                    while (currentNode != null && currentList != null)
                    {
                        if (currentNode.Val != currentList.Val)
                        {
                            break;
                        }
                        currentList = currentList.Next;

                        if (currentList != null)
                        {
                            if (currentNode.Left != null)
                            {
                                s.Push(
                                    new KeyValuePair<TreeNode, ListNode>(
                                        currentNode.Left,
                                        currentList
                                    )
                                );
                            }
                            if (currentNode.Right != null)
                            {
                                s.Push(
                                    new KeyValuePair<TreeNode, ListNode>(
                                        currentNode.Right,
                                        currentList
                                    )
                                );
                            }
                            break;
                        }
                    }

                    if (currentList == null)
                    {
                        return true;
                    }
                }

                return false;
            }

            /* Approach 3: Knuth-Morris-Pratt (KMP) Algorithm
Complexity Analysis
Let n be the number of nodes in the tree and m be the length of the linked list.
•	Time complexity: O(n+m)
Building the pattern and prefix table takes O(m), and searching the tree is O(n).
•	Space complexity: O(n+m)
We need O(m) space for the pattern and prefix table. The recursive call stack in the worst case (skewed tree) can take up to O(n) space.

             */
            public bool UsingKMPAlgo(ListNode head, TreeNode root)
            {
                // Build the pattern and prefix table from the linked list
                List<int> pattern = new List<int>();
                List<int> prefixTable = new List<int>();
                pattern.Add(head.Val);
                prefixTable.Add(0);
                int patternIndex = 0;
                head = head.Next;

                while (head != null)
                {
                    while (patternIndex > 0 && head.Val != pattern[patternIndex])
                    {
                        patternIndex = prefixTable[patternIndex - 1];
                    }
                    patternIndex += head.Val == pattern[patternIndex] ? 1 : 0;
                    pattern.Add(head.Val);
                    prefixTable.Add(patternIndex);
                    head = head.Next;
                }

                // Perform DFS to search for the pattern in the tree
                return SearchInTree(root, 0, pattern, prefixTable);
            }

            private bool SearchInTree(
                TreeNode node,
                int patternIndex,
                List<int> pattern,
                List<int> prefixTable
            )
            {
                if (node == null) return false;

                while (patternIndex > 0 && node.Val != pattern[patternIndex])
                {
                    patternIndex = prefixTable[patternIndex - 1];
                }
                patternIndex += node.Val == pattern[patternIndex] ? 1 : 0;

                // Check if the pattern is fully matched
                if (patternIndex == pattern.Count) return true;

                // Recursively search left and right subtrees
                return (
                    SearchInTree(node.Left, patternIndex, pattern, prefixTable) ||
                    SearchInTree(node.Right, patternIndex, pattern, prefixTable)
                );
            }



        }


        /* 2816. Double a Number Represented as a Linked List
        https://leetcode.com/problems/double-a-number-represented-as-a-linked-list/description/
         */

        public class DoubleItSol
        {
            /* Approach 1: Reversing the List
Complexity Analysis
Let n be the number of nodes in the linked list.
•	Time complexity: O(n)
The algorithm involves traversing the linked list once to double the values and handle carry, performing constant-time operations for each node. So, it takes O(n) time.
Reversing the list also takes O(n) time.
Thus, the overall time complexity of the algorithm is O(n).
•	Space complexity: O(1)
In-place reversal is performed, so it doesn't incur significant extra space usage. Thus, the space complexity remains O(1).

             */
            public ListNode UsingReverseList(ListNode head)
            {
                // Reverse the linked list
                ListNode reversedList = ReverseList(head);
                // Initialize variables to track carry and previous node
                int carry = 0;
                ListNode current = reversedList, previous = null;

                // Traverse the reversed linked list
                while (current != null)
                {
                    // Calculate the new value for the current node
                    int newValue = current.Val * 2 + carry;
                    // Update the current node's value
                    current.Val = newValue % 10;
                    // Update carry for the next iteration
                    if (newValue > 9)
                    {
                        carry = 1;
                    }
                    else
                    {
                        carry = 0;
                    }
                    // Move to the next node
                    previous = current;
                    current = current.Next;
                }

                // If there's a carry after the loop, add an extra node
                if (carry != 0)
                {
                    ListNode extraNode = new ListNode(carry);
                    previous.Next = extraNode;
                }

                // Reverse the list again to get the original order
                ListNode result = ReverseList(reversedList);

                return result;
            }

            // Method to reverse the linked list
            private ListNode ReverseList(ListNode node)
            {
                ListNode previous = null, current = node, nextNode;

                // Traverse the original linked list
                while (current != null)
                {
                    // Store the next node
                    nextNode = current.Next;
                    // Reverse the link
                    current.Next = previous;
                    // Move to the next nodes
                    previous = current;
                    current = nextNode;
                }
                // Previous becomes the new head of the reversed list
                return previous;
            }

            /* Approach 2: Using Stack
    Complexity Analysis
    Let n be the number of nodes in the linked list.
    •	Time complexity: O(n)
    The algorithm traverses the linked list once to push its values onto the stack, which takes O(n) time. Then, it iterates over the stack and performs operations to create the new linked list, which also takes O(n) time, as the stack contains n elements.
    Therefore, the overall time complexity of the algorithm is O(n).
    •	Space complexity: O(n)
    The space complexity mainly depends on the additional space used by the stack to store the values of the linked list, which takes O(n) space.
    Additionally, the space used for the new linked list is also O(n) since we are creating a new node for each element in the original linked list.
    Therefore, the overall space complexity of the algorithm is O(n).

             */
            public ListNode UsingStack(ListNode head)
            {
                // Initialize a stack to store the values of the linked list
                Stack<int> values = new();
                int val = 0;

                // Traverse the linked list and push its values onto the stack
                while (head != null)
                {
                    values.Push(head.Val);
                    head = head.Next;
                }

                ListNode newTail = null;

                // Iterate over the stack of values and the carryover
                while (values.Count > 0 || val != 0)
                {
                    // Create a new ListNode with value 0 and the previous tail as its next node
                    newTail = new ListNode(0, newTail);

                    // Calculate the new value for the current node
                    // by doubling the last digit, adding carry, and getting the remainder
                    if (values.Count > 0)
                    {
                        val += values.Pop() * 2;
                    }
                    newTail.Val = val % 10;
                    val /= 10;
                }

                // Return the tail of the new linked list
                return newTail;
            }
            /* Approach 3: Recursion
            Complexity Analysis
Let n be the number of nodes in the linked list.
•	Time complexity: O(n)
The twiceOfVal function recursively traverses the entire linked list once, performing constant-time operations at each node. Therefore, the time complexity of the twiceOfVal function is O(n).
The doubleIt function calls the twiceOfVal function once, which has a time complexity of O(n). Additionally, inserting a new node at the beginning of the linked list takes constant time. Hence, the overall time complexity of the doubleIt function is O(n).
Therefore, the overall time complexity of the algorithm is O(n).
•	Space complexity: O(n)
The twiceOfVal function is tail-recursive, meaning it should typically use O(1) space on the call stack due to the recursive calls in C++ and Java. However, in languages like Python, which don't optimize tail recursion, each recursive call consumes additional space on the call stack. Therefore, the space complexity of twiceOfVal is O(n) due to the recursive call stack.
The doubleIt function uses no additional space apart from the space required for the input linked list. Hence, its space complexity is O(1).
Therefore, the overall space complexity of the algorithm is dominated by the recursive call stack, making it O(n).
s
             */
            public ListNode UsingRecursion(ListNode head)
            {
                int carry = TwiceOfVal(head);

                // If there's a carry, insert a new node at the beginning with the carry value
                if (carry != 0)
                {
                    head = new ListNode(carry, head);
                }

                return head;
            }
            // To compute twice the value of each node's value and propagate carry
            private int TwiceOfVal(ListNode head)
            {
                // Base case: if head is null, return 0
                if (head == null) return 0;

                // Double the value of current node and add the result of next nodes
                int doubledValue = head.Val * 2 + TwiceOfVal(head.Next);

                // Update current node's value with the units digit of the result
                head.Val = doubledValue % 10;

                // Return the carry (tens digit of the result)
                return doubledValue / 10;
            }

            /* Approach 4: Two Pointers
Complexity Analysis
Let n be the number of nodes in the linked list.
•	Time complexity: O(n)
The algorithm traverses the entire linked list once. Within the loop, each operation (including arithmetic operations and pointer manipulations) takes constant time.
Therefore, the time complexity of the algorithm is O(n).
•	Space complexity: O(1)
The algorithm uses only a constant amount of additional space for storing pointers and temporary variables, regardless of the size of the input linked list.
Therefore, the space complexity is O(1).

             */
            public ListNode UsingTwoPointers(ListNode head)
            {
                ListNode curr = head;
                ListNode prev = null;

                // Traverse the linked list
                while (curr != null)
                {
                    int twiceOfVal = curr.Val * 2;

                    // If the doubled value is less than 10
                    if (twiceOfVal < 10)
                    {
                        curr.Val = twiceOfVal;
                    }
                    // If doubled value is 10 or greater
                    else if (prev != null)
                    { // other than first node
                      // Update current node's value with units digit of the doubled value
                        curr.Val = twiceOfVal % 10;
                        // Add the carry to the previous node's value
                        prev.Val = prev.Val + 1;
                    }
                    // If it's the first node and doubled value is 10 or greater
                    else
                    { // first node
                      // Create a new node with carry as value and link it to the current node
                        head = new ListNode(1, curr);
                        // Update current node's value with units digit of the doubled value
                        curr.Val = twiceOfVal % 10;
                    }

                    // Update prev and curr pointers
                    prev = curr;
                    curr = curr.Next;
                }
                return head;
            }

            /* Approach 5: Single Pointer
Complexity Analysis
Let n be the number of nodes in the linked list.
•	Time complexity: O(n)
The algorithm traverses the entire linked list once, visiting each node. Within the loop, each operation (including arithmetic operations and pointer manipulations) takes constant time.
Therefore, the time complexity of the algorithm is O(n).
•	Space complexity: O(1)
The algorithm uses only a constant amount of additional space for storing pointers and temporary variables, regardless of the size of the input linked list.
Therefore, the space complexity is O(1).	

             */
            public ListNode UsingSinglePointer(ListNode head)
            {
                // If the value of the head node is greater than 4, 
                // insert a new node at the beginning
                if (head.Val > 4)
                {
                    head = new ListNode(0, head);
                }

                // Traverse the linked list
                for (ListNode node = head; node != null; node = node.Next)
                {
                    // Double the value of the current node 
                    // and update it with the units digit
                    node.Val = (node.Val * 2) % 10;

                    // If the current node has a next node 
                    // and the next node's value is greater than 4,
                    // increment the current node's value to handle carry
                    if (node.Next != null && node.Next.Val > 4)
                    {
                        node.Val++;
                    }
                }

                return head;
            }

        }

        /* 143. Reorder List
        https://leetcode.com/problems/reorder-list/description/
         */
        public class ReorderListSol
        {

            /*             Approach 1: Reverse the Second Part of the List and Merge Two Sorted Lists
            Complexity Analysis
            •	Time complexity: O(N). There are three steps here. To identify the middle node takes O(N) time. To reverse the second part of the list, one needs N/2 operations. The final step, to merge two lists, requires N/2 operations as well. In total, that results in O(N) time complexity.
            •	Space complexity: O(1), since we do not allocate any additional data structures

             */
            public void ReverseSecondPartOfListAndMergeTwoSortedLists(ListNode head)
            {
                if (head == null)
                    return;
                // find the middle of linked list [Problem 876]
                // in 1->2->3->4->5->6 find 4
                ListNode slow = head, fast = head;
                while (fast != null && fast.Next != null)
                {
                    slow = slow.Next;
                    fast = fast.Next.Next;
                }

                // reverse the second part of the list [Problem 206]
                // convert 1->2->3->4->5->6 into 1->2->3->4 and 6->5->4
                // reverse the second half in-place
                ListNode prev = null, curr = slow, tmp;
                while (curr != null)
                {
                    tmp = curr.Next;
                    curr.Next = prev;
                    prev = curr;
                    curr = tmp;
                }

                // merge two sorted linked lists [Problem 21]
                // merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
                ListNode first = head, second = prev;
                while (second.Next != null)
                {
                    tmp = first.Next;
                    first.Next = second;
                    first = tmp;
                    tmp = second.Next;
                    second.Next = first;
                    second = tmp;
                }
            }
        }


        /* 3217. Delete Nodes From Linked List Present in Array
        https://leetcode.com/problems/delete-nodes-from-linked-list-present-in-array/description/
         */
        public class DeleteNodesFromLinkedListPresentInArraySolution
        {
            /* 
            Approach: Hash Set
            Complexity Analysis
Let m and n be the lengths of the nums array and the linked list, respectively.
•	Time complexity: O(m+n)
Iterating through the nums array and inserting each element into the hash set takes O(m) time, as each insertion into the set is O(1) on average.
The algorithm traverses the entire linked list exactly once, checking if each node's value is in the hash set. This operation takes O(n) time.
Thus, the overall time complexity of the algorithm is O(m)+O(n)=O(m+n).
•	Space complexity: O(m)
The hash set can store up to m elements, one for each unique value in the nums array, leading to a space complexity of O(m). All additional variables used take constant space.
 */
            public ListNode UsingHashSet(int[] nums, ListNode head)
            {
                // Create a HashSet for efficient lookup of values in nums
                HashSet<int> valuesToRemove = new HashSet<int>();
                foreach (int num in nums)
                {
                    valuesToRemove.Add(num);
                }

                // Handle the case where the head node needs to be removed
                while (head != null && valuesToRemove.Contains(head.Val))
                {
                    head = head.Next;
                }

                // If the list is empty after removing head nodes, return null
                if (head == null)
                {
                    return null;
                }

                // Iterate through the list, removing nodes with values in the set
                ListNode current = head;
                while (current.Next != null)
                {
                    if (valuesToRemove.Contains(current.Next.Val))
                    {
                        // Skip the next node by updating the pointer
                        current.Next = current.Next.Next;
                    }
                    else
                    {
                        // Move to the next node
                        current = current.Next;
                    }
                }

                return head;
            }
        }

        /* 1721. Swapping Nodes in a Linked List
        https://leetcode.com/problems/swapping-nodes-in-a-linked-list/description/
         */
        class SwapNodesSol
        {
            /* 
            Approach 1: Three Pass Approach
            Complexity Analysis
•	Time Complexity : O(n), where n is the length of the Linked List. We are iterating over the Linked List thrice.
In the first pass, we are finding the length of the Linked List which would take O(n) time.
In second pass and third pass, we are iterating k and n - k times respectively. This would take O(k+n−k) i.e O(n) time.
Thus, the total time complexity would be O(n)+O(n)=O(n).
•	Space Complexity: O(1), as we are using constant extra space to maintain list node pointers frontNode, endNode and currentNode.

             */
            public ListNode UsingThreePass(ListNode head, int k)
            {
                int listLength = 0;
                ListNode currentNode = head;
                // find the length of linked list
                while (currentNode != null)
                {
                    listLength++;
                    currentNode = currentNode.Next;
                }
                // set the front node at kth node
                ListNode frontNode = head;
                for (int i = 1; i < k; i++)
                {
                    frontNode = frontNode.Next;
                }
                //set the end node at (listLength - k)th node
                ListNode endNode = head;
                for (int i = 1; i <= listLength - k; i++)
                {
                    endNode = endNode.Next;
                }
                // swap the values of front node and end node
                int temp = frontNode.Val;
                frontNode.Val = endNode.Val;
                endNode.Val = temp;
                return head;
            }
            /* Approach 2: Two Pass Approach
Complexity Analysis
•	Time Complexity : O(n), where n is the length of the Linked List. We are iterating over the Linked List twice.
In the first pass, we are finding the length of the Linked List and setting the frontNode which would take O(n) time.
In the second pass, we are setting the endNode by iterating n - k times.
Thus, the total time complexity would be O(n)+O(n−k) which is equivalent to O(n).
•	Space Complexity: O(1), as we are using constant extra space to maintain list node pointers frontNode, endNode and currentNode.

             */
            public ListNode UsingTwoPass(ListNode head, int k)
            {
                int listLength = 0;
                ListNode frontNode = null;
                ListNode endNode = null;
                ListNode currentNode = head;
                // find the length of list and set the front node
                while (currentNode != null)
                {
                    listLength++;
                    if (listLength == k)
                    {
                        frontNode = currentNode;
                    }
                    currentNode = currentNode.Next;
                }
                // set the end node at (listLength - k)th node
                endNode = head;
                for (int i = 1; i <= listLength - k; i++)
                {
                    endNode = endNode.Next;
                }
                // swap front node and end node values
                int temp = frontNode.Val;
                frontNode.Val = endNode.Val;
                endNode.Val = temp;
                return head;
            }
            /* Approach 3: Single Pass Approach
            Complexity Analysis
•	Time Complexity : O(n), where n is the size of Linked List. We are iterating over the entire Linked List once.
•	Space Complexity: O(1), as we are using constant extra space to maintain list node pointers frontNode, endNode and currentNode.

             */
            public ListNode UsingSinlePass(ListNode head, int k)
            {
                int listLength = 0;
                ListNode frontNode = null;
                ListNode endNode = null;
                ListNode currentNode = head;
                // set the front node and end node in single pass
                while (currentNode != null)
                {
                    listLength++;
                    if (endNode != null)
                        endNode = endNode.Next;
                    // check if we have reached kth node
                    if (listLength == k)
                    {
                        frontNode = currentNode;
                        endNode = head;
                    }
                    currentNode = currentNode.Next;
                }
                // swap the values of front node and end node
                int temp = frontNode.Val;
                frontNode.Val = endNode.Val;
                endNode.Val = temp;
                return head;
            }

        }

        /* 148. Sort List
        https://leetcode.com/problems/sort-list/description/
         */

        public class SortListSol
        {
            /* Approach 1: Top Down Merge Sort 
            Complexity Analysis
•	Time Complexity: O(nlogn), where n is the number of nodes in linked list.
The algorithm can be split into 2 phases, Split and Merge.
Let's assume that n is power of 2. For n = 16, the split and merge operation in Top Down fashion can be visualized as follows
 
Split
The recursion tree expands in form of a complete binary tree, splitting the list into two halves recursively. The number of levels in a complete binary tree is given by log2n. For n=16, number of splits = log216=4
Merge
At each level, we merge n nodes which takes O(n) time.
For n=16, we perform merge operation on 16 nodes in each of the 4 levels.
So the time complexity for split and merge operation is O(nlogn)
•	Space Complexity: O(logn) , where n is the number of nodes in linked list. Since the problem is recursive, we need additional space to store the recursive call stack. The maximum depth of the recursion tree is logn
s
            */

            public ListNode UsingTopDownMergeSort(ListNode head)
            {
                if (head == null || head.Next == null)
                    return head;
                ListNode mid = GetMid(head);
                ListNode left = UsingTopDownMergeSort(head);
                ListNode right = UsingTopDownMergeSort(mid);
                return Merge(left, right);
            }

            private ListNode Merge(ListNode list1, ListNode list2)
            {
                ListNode dummyHead = new ListNode(0);
                ListNode tail = dummyHead;
                while (list1 != null && list2 != null)
                {
                    if (list1.Val < list2.Val)
                    {
                        tail.Next = list1;
                        list1 = list1.Next;
                    }
                    else
                    {
                        tail.Next = list2;
                        list2 = list2.Next;
                    }

                    tail = tail.Next;
                }

                tail.Next = list1 != null ? list1 : list2;
                return dummyHead.Next;
            }

            private ListNode GetMid(ListNode head)
            {
                ListNode midPrev = null;
                while (head != null && head.Next != null)
                {
                    midPrev = midPrev == null ? head : midPrev.Next;
                    head = head.Next.Next;
                }

                ListNode mid = midPrev.Next;
                midPrev.Next = null;
                return mid;
            }
            /* Approach 2: Bottom Up Merge Sort
            Complexity Analysis
•	Time Complexity: O(nlogn), where n is the number of nodes in linked list.
Let's analyze the time complexity of each step:
1.	Count Nodes - Get the count of number nodes in the linked list requires O(n) time.
2.	Split and Merge - This operation is similar to Approach 1 and takes O(nlogn) time.
For n = 16, the split and merge operation in Bottom Up fashion can be visualized as follows
 
This gives us total time complexity as
O(n)+O(nlogn)=O(nlogn)
•	Space Complexity: O(1) We use only constant space for storing the reference pointers tail , nextSubList etc.

             */
            private ListNode tail;
            private ListNode nextSubList;

            // Sorts the linked list using merge sort
            public ListNode SortList(ListNode head)
            {
                if (head == null || head.Next == null)
                    return head;

                int n = GetCount(head);
                ListNode start = head;
                ListNode dummyHead = new ListNode();
                for (int size = 1; size < n; size *= 2)
                {
                    tail = dummyHead;
                    while (start != null)
                    {
                        if (start.Next == null)
                        {
                            tail.Next = start;
                            break;
                        }

                        ListNode mid = Split(start, size);
                        Merge(start, mid);
                        start = nextSubList;
                    }

                    start = dummyHead.Next;
                }

                return dummyHead.Next;
                void Merge(ListNode list1, ListNode list2)
                {
                    ListNode dummyHead = new ListNode();
                    ListNode newTail = dummyHead;
                    while (list1 != null && list2 != null)
                    {
                        if (list1.Val < list2.Val)
                        {
                            newTail.Next = list1;
                            list1 = list1.Next;
                            newTail = newTail.Next;
                        }
                        else
                        {
                            newTail.Next = list2;
                            list2 = list2.Next;
                            newTail = newTail.Next;
                        }
                    }

                    newTail.Next = (list1 != null) ? list1 : list2;
                    // Traverse till the end of merged list to get the newTail
                    while (newTail.Next != null)
                    {
                        newTail = newTail.Next;
                    }

                    // Link the old tail with the head of merged list
                    tail.Next = dummyHead.Next;
                    // Update the old tail to the new tail of merged list
                    tail = newTail;
                }
            }

            // Splits the list into two and returns the middle node
            private ListNode Split(ListNode start, int size)
            {
                ListNode midPrev = start;
                ListNode end = start.Next;
                // Use fast and slow approach to find middle and end of second linked
                // list
                for (int index = 1; index < size && (midPrev.Next != null ||
                                                     (end != null && end.Next != null));
                     index++)
                {
                    if (end != null && end.Next != null)
                    {
                        end = (end.Next.Next != null) ? end.Next.Next : end.Next;
                    }

                    if (midPrev.Next != null)
                    {
                        midPrev = midPrev.Next;
                    }
                }

                ListNode mid = midPrev.Next;
                midPrev.Next = null;
                nextSubList = end != null ? end.Next : null;
                if (end != null)
                    end.Next = null;
                // Return the start of second linked list
                return mid;
            }

            // Merges two sorted lists


            // Counts the number of nodes in the list
            private int GetCount(ListNode head)
            {
                int cnt = 0;
                ListNode ptr = head;
                while (ptr != null)
                {
                    ptr = ptr.Next;
                    cnt++;
                }

                return cnt;
            }


        }

        /* 328. Odd Even Linked List
        https://leetcode.com/problems/odd-even-linked-list/description/
         */
        public class OddEvenListSol
        {
            /* 
            Complexity Analysis
            •	Time complexity : O(n). There are total n nodes and we visit each node once.
            •	Space complexity : O(1). All we need is the four pointers.
             */
            public ListNode OddEvenList(ListNode head)
            {
                if (head == null) return null;
                ListNode odd = head, even = head.Next, evenHead = even;
                while (even != null && even.Next != null)
                {
                    odd.Next = even.Next;
                    odd = odd.Next;
                    even.Next = odd.Next;
                    even = even.Next;
                }
                odd.Next = evenHead;
                return head;
            }
        }


        /* 142. Linked List Cycle II
        https://leetcode.com/problems/linked-list-cycle-ii/description/
         */
        public class DetectCycleIISol
        {
            /* Approach 1: Hash Set
            Implementation
Complexity Analysis
Let n be the total number of nodes in the linked list.
•	Time complexity: O(n).
We have to visit all nodes once.
•	Space complexity: O(n).
We have to store all nodes in the hash set.

             */
            public ListNode UsingHashSet(ListNode head)
            {
                // Initialize an empty hash set
                HashSet<ListNode> nodesSeen = new HashSet<ListNode>();
                // Start from the head of the linked list
                ListNode node = head;
                while (node != null)
                {
                    // If the current node is in nodesSeen, we have a cycle
                    if (nodesSeen.Contains(node))
                    {
                        return node;
                    }
                    else
                    {
                        // Add this node to nodesSeen and move to the next node
                        nodesSeen.Add(node);
                        node = node.Next;
                    }
                }

                // If we reach a null node, there is no cycle
                return null;
            }


            /* Approach 2: Floyd's Tortoise and Hare Algorithm
            Complexity Analysis
Let n be the total number of nodes in the linked list.
•	Time complexity: O(n).
The algorithm consists of two phases. In the first phase, we use two pointers (the "hare" and the "tortoise") to traverse the list. The slow pointer (tortoise) will go through the list only once until it meets the hare. Therefore, this phase runs in O(n) time.
In the second phase, we again have two pointers traversing the list at the same speed until they meet. The maximum distance to be covered in this phase will not be greater than the length of the list (recall that the hare just needs to get back to the entrance of the cycle). So, this phase also runs in O(n) time.
As a result, the total time complexity of the algorithm is O(n)+O(n), which simplifies to O(n).
•	Space complexity: O(1).
The space complexity is constant, O(1), because we are only using a fixed amount of space to store the slow and fast pointers. No additional space is used that scales with the input size. So the space complexity of the algorithm is O(1), which means it uses constant space.

             */
            public ListNode UsingFloydsTortoiseAndHareAlgo(ListNode head)
            {
                // Initialize tortoise and hare pointers
                ListNode tortoise = head;
                ListNode hare = head;
                // Move tortoise one step and hare two steps
                while (hare != null && hare.Next != null)
                {
                    tortoise = tortoise.Next;
                    hare = hare.Next.Next;
                    // Check if the hare meets the tortoise
                    if (tortoise == hare)
                    {
                        break;
                    }
                }

                // Check if there is no cycle
                if (hare == null || hare.Next == null)
                {
                    return null;
                }

                // Reset either tortoise or hare pointer to the head
                hare = head;
                // Move both pointers one step until they meet again
                while (tortoise != hare)
                {
                    tortoise = tortoise.Next;
                    hare = hare.Next;
                }

                // Return the node where the cycle begins
                return tortoise;
            }

        }


        /* 1669. Merge In Between Linked Lists
        https://leetcode.com/problems/merge-in-between-linked-lists/description/
         */
        class MergeInBetweenSol
        {
            /* Approach 1: Merge Values in Array 
            Complexity Analysis
Let n be the length of list1 and m be the length of list2.
•	Time complexity: O(n+m)
The algorithm traverses list1 and list2 to add the nodes to the array, taking n+m computational steps.
Then, the array is traversed once to create the resulting linked list. The size of the array will be at most n+m.
Therefore, the time complexity is O(n+m).
•	Space complexity: O(n+m)
We use mergeArray, which can contain the values of list1 and list2. It can have at most n+m elements. Therefore, the space complexity is O(n+m).

            */
            public ListNode UsingMergeValuesInArray(ListNode list1, int a, int b, ListNode list2)
            {
                List<int> mergeArray = new();

                // Add list1 node values before `a` to the array.
                int index = 0;
                ListNode current1 = list1;
                while (index < a)
                {
                    mergeArray.Add(current1.Val);
                    current1 = current1.Next;
                    index++;
                }

                // Add list2 node values to the array
                ListNode current2 = list2;
                while (current2 != null)
                {
                    mergeArray.Add(current2.Val);
                    current2 = current2.Next;
                }

                // Find node b + 1
                while (index < b + 1)
                {
                    current1 = current1.Next;
                    index++;
                }

                // Add list1 node values after `b` to the array.
                while (current1 != null)
                {
                    mergeArray.Add(current1.Val);
                    current1 = current1.Next;
                }

                // Build a linked list with the result by iterating over the array
                // in reverse order and inserting new nodes to the front of the list
                ListNode resultList = null;
                for (int i = mergeArray.Count - 1; i >= 0; i--)
                {
                    ListNode newNode = new ListNode(mergeArray[i], resultList);
                    resultList = newNode;
                }

                return resultList;
            }

            /* Approach 2: Two Pointer
Complexity Analysis
Let n be the length of list1 and m be the length of list2.
•	Time complexity: O(n+m)
The algorithm traverses list1 once to find the nodes start and end. Note that list1 is not fully traversed for every input, but in the worst case, we may need to traverse at most n nodes. list2 is traversed once to find its tail. The other operations all take constant time.
Therefore, the time complexity is O(n+m).
The recursive implementation has the same time complexity as the iterative implementation.
•	Space complexity: O(1)
We use a few variables and pointers, including index, start, and end, which use constant extra space. We don't use any data structures that grow with input size, so the space complexity of the iterative implementation is O(1).
The recursive implementation may use up to O(n+m) space for the recursive call stack, though this space may be reduced through the use of tail recursion, depending on the implementation language.

             */
            public ListNode UsingTwoPointer(ListNode list1, int a, int b, ListNode list2)
            {
                ListNode start = null;
                ListNode end = list1;

                // Set start to node a - 1 and end to node b
                for (int index = 0; index < b; index++)
                {
                    if (index == a - 1)
                    {
                        start = end;
                    }
                    end = end.Next;
                }
                // Connect the start node to list2
                start.Next = list2;

                // Find the tail of list2
                while (list2.Next != null)
                {
                    list2 = list2.Next;
                }
                // Set the tail of list2 to end.Next
                list2.Next = end.Next;
                end.Next = null;

                return list1;
            }

        }





    }
}
