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
        //https://www.algoexpert.io/questions/linked-list-

        //1.
        // O(n) time | O(n) space - where n is the number of nodes in the Linked List
        public bool LinkedListPalindromeNaive(LinkedList head)
        {
            LinkedListInfo isPalindromeResults = isPalindrome(head, head);
            return isPalindromeResults.outerNodesAreEqual;
        }

        public LinkedListInfo isPalindrome(
          LinkedList leftNode, LinkedList rightNode
        )
        {
            if (rightNode == null) return new LinkedListInfo(true, leftNode);

            LinkedListInfo recursiveCallResults =
              isPalindrome(leftNode, rightNode.Next);
            LinkedList leftNodeToCompare = recursiveCallResults.leftNodeToCompare;
            bool outerNodesAreEqual = recursiveCallResults.outerNodesAreEqual;

            bool recursiveIsEqual =
              outerNodesAreEqual && (leftNodeToCompare.Value == rightNode.Value);
            LinkedList nextLeftNodeToCompare = leftNodeToCompare.Next;

            return new LinkedListInfo(recursiveIsEqual, nextLeftNodeToCompare);
        }

        //2.
        // O(n) time | O(1) space - where n is the number of nodes in the Linked List
        public bool LinkedListPalindrome(LinkedList head)
        {
            LinkedList slowNode = head;
            LinkedList fastNode = head;

            while (fastNode != null && fastNode.Next != null)
            {
                slowNode = slowNode.Next;
                fastNode = fastNode.Next.Next;
            }

            LinkedList reversedSecondHalfNode = reverseLinkedList(slowNode);
            LinkedList firstHalfNode = head;

            while (reversedSecondHalfNode != null)
            {
                if (reversedSecondHalfNode.Value != firstHalfNode.Value) return false;
                reversedSecondHalfNode = reversedSecondHalfNode.Next;
                firstHalfNode = firstHalfNode.Next;
            }

            return true;
        }

        public static LinkedList reverseLinkedList(LinkedList head)
        {
            LinkedList previousNode = null;
            LinkedList currentNode = head;
            while (currentNode != null)
            {
                LinkedList nextNode = currentNode.Next;
                currentNode.Next = previousNode;
                previousNode = currentNode;
                currentNode = nextNode;
            }
            return previousNode;
        }

        public class LinkedListInfo
        {
            public bool outerNodesAreEqual;
            public LinkedList leftNodeToCompare;
            public LinkedListInfo(
              bool outerNodesAreEqual, LinkedList leftNodeToCompare
            )
            {
                this.outerNodesAreEqual = outerNodesAreEqual;
                this.leftNodeToCompare = leftNodeToCompare;
            }
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
        public class ListNode
        {
            public int Val;
            public ListNode Next;

            public ListNode(int val = 0, ListNode next = null)
            {
                this.Val = val;
                this.Next = next;
            }

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













    }
}
