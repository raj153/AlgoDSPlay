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
        public static void RemoveKthNodeFromEnd(LinkedList head, int k){
            //T:O(n)| S:O(1)
            int counter=1;
            LinkedList first = head;
            LinkedList second = head;
            while(counter <=k){
                second = second.Next;
                counter++;
            }

            if(second == null){
                //0->1->2->Null
                head.Value = head.Next.Value;
                head.Next = head.Next.Next;
                return;
            }
            while(second.Next != null){
                second = second.Next;
                first = first.Next;
            }
            first.Next = first.Next.Next;

        }

        //https://www.algoexpert.io/questions/merge-linked-lists
        public static LinkedList MergeLinkedLists(LinkedList headOne, LinkedList headTwo){
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

            while(p1 != null && p2 != null){
                if(p1.Value < p2.Value){
                    p1Prev = p1;
                    p1=p1.Next;
                }else{

                    if(p1Prev != null)
                        p1Prev.Next=p2;
                    p1Prev = p2;
                    p2 = p2.Next;
                    p1Prev.Next = p1;
                }
            }
            if(p1 == null) p1Prev.Next = p2;

            return headOne.Value< headTwo.Value ? headOne: headTwo;
        }
        //https://www.algoexpert.io/questions/middle-node
        public static LinkedList MiddleNode(LinkedList linkedList){
            LinkedList node=default;
            
            //1.Naive - T:O(n) | S:O(1)
            node = MiddleNodeNaive(linkedList);

            //2.Two Pointer - T:O(n) | S:O(1) 
            node = MiddleNodeTwoPointer(linkedList);
            return node;


        }

        private static LinkedList? MiddleNodeTwoPointer(LinkedList linkedList)
        {
            LinkedList slowNode=linkedList, fastNode=linkedList;
            
            
            while(fastNode !=null && fastNode.Next != null)
            {
                slowNode = slowNode.Next;
                fastNode = fastNode.Next.Next;

            }
            return slowNode;
        }

        private static LinkedList? MiddleNodeNaive(LinkedList linkedList)
        {
            int count =0;

            LinkedList currentNode = linkedList;

            while(currentNode != null)
            {
                count++;
                currentNode= currentNode.Next;
            }
            LinkedList middleNode = linkedList;

            for(int i=0; i<count/2; i++){
                middleNode = middleNode.Next;
            }

            return middleNode;

        }
        //https://www.algoexpert.io/questions/remove-duplicates-from-linked-list
        public static LinkedList RemoveDuplicatesFromLinkedList(LinkedList linkedList){

            //T:O(n) | S:O(1)
            LinkedList currentNode = linkedList;
            while(currentNode != null){
                LinkedList nextDistinctNode = currentNode.Next;                                
                while(nextDistinctNode !=null &&
                        currentNode.Value == nextDistinctNode.Value)
                {

                    nextDistinctNode = nextDistinctNode.Next;
                }
                currentNode.Next = nextDistinctNode;
                currentNode = nextDistinctNode;
            }
            return linkedList;
        }

        public static LinkedList RemoveDuplicatesFromLinkedList1(LinkedList linkedList){
            LinkedList currentNode = linkedList;
            LinkedList prevNode = null;
            int prevNodeVal = Int32.MinValue;
            
            while(currentNode != null){          
                if(prevNodeVal != Int32.MinValue && prevNodeVal == currentNode.Value){
                        prevNode.Next = currentNode.Next;
                        currentNode = currentNode.Next;
                            
                }else{
                    prevNode = currentNode;
                    prevNodeVal = currentNode.Value;
                    currentNode = currentNode.Next;
                }
          
          
            }
            // Write your code here.
            return linkedList;

        }
        //https://www.algoexpert.io/questions/merging-linked-lists
        public static LinkedList MergingLinkedLists(LinkedList linkedList1, LinkedList linkedList2){

            //1.Naive - T:O(n+m) | S:O(n) - Using HashSet
            LinkedList intersectNode =MergingLinkedListsNaive(linkedList1, linkedList2);

            //2.Optimla w/ou extra space - T:O(n+m) | S:O(1) 
            intersectNode =MergingLinkedListsOptimal(linkedList1, linkedList2);

            return intersectNode;

            
        }

        private static LinkedList MergingLinkedListsOptimal(LinkedList linkedList1, LinkedList linkedList2)
        {
            LinkedList currentNodeOne = linkedList1;
            int countOne =0;
            while(currentNodeOne != null){
                countOne++;
                currentNodeOne = currentNodeOne.Next;
            }
            LinkedList currentNodeTwo = linkedList2;
            int countTwo=0;
            while(currentNodeTwo != null){
                countTwo++;
                currentNodeTwo = currentNodeTwo.Next;
            }

            int difference = Math.Abs(countTwo-countOne);

            LinkedList biggerCurrentNode = countOne > countTwo ? linkedList1 : linkedList2;
            LinkedList smallerCurrentNode = countOne > countTwo ? linkedList2 : linkedList1;

            for(int i=0; i< difference; i++){
                biggerCurrentNode = biggerCurrentNode.Next;
            }
            while(biggerCurrentNode != smallerCurrentNode){
                biggerCurrentNode = biggerCurrentNode.Next;
                smallerCurrentNode = smallerCurrentNode.Next;
            }
            return biggerCurrentNode;
        }

        private static LinkedList MergingLinkedListsNaive(LinkedList linkedList1, LinkedList linkedList2)
        {
            HashSet<LinkedList> listOneNodeSet = new HashSet<LinkedList>();

            LinkedList currentNodeOne = linkedList1;
            while(currentNodeOne != null){
                listOneNodeSet.Add(currentNodeOne);
                currentNodeOne = currentNodeOne.Next;
            }

            LinkedList currentNodeTwo =  linkedList2;

            while(currentNodeTwo != null){

                if(listOneNodeSet.Contains(currentNodeTwo))
                    return currentNodeTwo;
                currentNodeTwo = currentNodeTwo.Next;
            }

            return null;

        }
        //https://www.algoexpert.io/questions/find-loop
        public static LinkedList FindLoop(LinkedList head){

            //1.Naive - Using Set to store nodes and loop exists if a node exists in Set
            //T:O(n) : S:O(1)

            //2.Optimal - Two pointers

            LinkedList slow = head;
            LinkedList fast = head.Next.Next;

            while(slow != fast){
                slow = slow.Next;
                fast = fast.Next.Next;
            }
            slow = head;
            while(slow != fast){

                slow = slow.Next;
                fast  = fast.Next;
            }
            return slow;

        }
        //https://www.algoexpert.io/questions/reverse-linked-list
        public static LinkedList ReverseLinkedList(LinkedList head) {
            //T:O(n) | S:O(1)
            LinkedList prevNode = null;
            LinkedList currNode = head;

            while(currNode != null){
                LinkedList nxtNode = currNode.Next;
                currNode.Next= prevNode;
                prevNode = currNode;
                currNode = nxtNode;
                
            }
           return prevNode;
        }

        //https://www.algoexpert.io/questions/node-swap
        public static LinkedList NodeSwap(LinkedList head){

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
            while(prevNode.Next != null && prevNode.Next.Next != null){
                LinkedList  firstNode = prevNode.Next;
                LinkedList secondNode =  prevNode.Next.Next;

                firstNode.Next = secondNode.Next;
                secondNode.Next =firstNode;
                prevNode.Next = secondNode;

                prevNode = firstNode;
            }
            return tempNode.Next;
        }

        private static LinkedList NodeSwapOptimal1(LinkedList head)
        {
            if(head == null && head.Next ==null) return head;

            LinkedList nextNode = head.Next;
            head.Next =NodeSwapOptimal1(head.Next);
            nextNode.Next = head;
            return nextNode;
        }
    }

}