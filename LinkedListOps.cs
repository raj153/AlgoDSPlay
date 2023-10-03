using System;
using System.Collections.Generic;
using System.Linq;
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
    }
}