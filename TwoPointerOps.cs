using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;
namespace AlgoDSPlay
{
    public class TwoPointerOps
    {
        //https://www.algoexpert.io/questions/zip-linked-list
        public static LinkedList ZipLinkedList(LinkedList linkedList){
            //T:O(n) : S:O(1)
            if(linkedList.Next == null || linkedList.Next.Next == null)
                return linkedList;

            LinkedList firstHalfHead = linkedList;
            LinkedList secondHalfHead = SplitLinkedList(linkedList);
            LinkedList reverseSecondHalfHead = ReverseSecondHalfHead(secondHalfHead);

            return InterweaveLinkedList(firstHalfHead, reverseSecondHalfHead);            
        }
        private static LinkedList InterweaveLinkedList(LinkedList linkedList1, LinkedList linkedList2){

            LinkedList linkedList1Iterator = linkedList1;
            LinkedList linkedList2Iterator = linkedList2;

            while(linkedList1Iterator != null && linkedList2Iterator!=null){
                //1->2->3->4
                //5->6
                LinkedList firstHalfIteratorNext = linkedList1Iterator.Next;
                LinkedList secondHalfIteratorNext = linkedList2Iterator.Next;

                linkedList2Iterator.Next= firstHalfIteratorNext;
                linkedList1Iterator.Next = linkedList2Iterator;

                linkedList1Iterator = firstHalfIteratorNext; 
                linkedList2Iterator= secondHalfIteratorNext;


            }
            return linkedList1;
        }
        private static LinkedList ReverseSecondHalfHead(LinkedList linkedList)
        {
            LinkedList previousNode= null;
            LinkedList currentNode = linkedList;
            //null->4->5->6->null
            //null->6->5->4->null
            while(currentNode != null){
                
                LinkedList nextNode = currentNode.Next;
                currentNode.Next = previousNode;
                previousNode = currentNode;
                
                currentNode =nextNode;
            }
            return previousNode;

        }

        private static LinkedList SplitLinkedList(LinkedList linkedList)
        {
            LinkedList slowIterator = linkedList;
            LinkedList fastIterator = linkedList;

            while(fastIterator != null && fastIterator.Next != null){
                slowIterator = slowIterator.Next;
                fastIterator = fastIterator.Next.Next;
            }
            LinkedList secondHalfHead = slowIterator.Next;
            slowIterator.Next = null;
            return secondHalfHead;
        }
    }
}