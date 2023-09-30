using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Data.Common;
using AlgoDSPlay;
using AlgoDSPlay.DP;
using AlgoDSPlay.DataStructures;
// See https://aka.ms/new-console-template for more information
using AlgoDSPlay.Sorting;

//2,4,3,6,7,10,9
var sortedArray =BubbleSort.Sort(new List<int>(){10,2,4,1});
sortedArray =BubbleSort.OptimalSort(new List<int>(){10,2,4,1});
//foreach(int value in sortedArray)
    //Console.WriteLine(value);
Console.WriteLine(MaxMin.FindMaxValueOfExpNaive(new int[]{3,6,1,-3,2,7}));
Console.WriteLine(MaxMin.FindMaxValueOfExpDP(new int[]{3,6,1,-3,2,7}));
Console.WriteLine();
//Console.WriteLine(MinJump.FindMinimumNumberOfJumpsOptimal(new int[]{3,4,2,1,2,3,7,1,1,1,3}));
LinkedList list = new LinkedList(1);
list.Next = new LinkedList(2);
list.Next.Next=new LinkedList(3);
list.Next.Next.Next=new LinkedList(4);
list.Next.Next.Next.Next=new LinkedList(5);
list.Next.Next.Next.Next.Next= new LinkedList(6);
var lst = TwoPointerOps.ZipLinkedList(list);
var currentNode=lst;
while (currentNode != null){
    Console.Write(currentNode.Value);
    currentNode=lst.Next;
}
//var result = RealProbs.CalendarMatching()
//big - bigger
//
//0  1  2 3 4
//6, 4, 3,2,5 = a-b+c-d
//6, 6, 6,6,6
//   2, 3,4,4 => a -b
//    , 5,5,9 => a-b+c
//    ,  ,3,3  => a-b+c-d