using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Data.Common;
using AlgoDSPlay;
using AlgoDSPlay.DP;
using AlgoDSPlay.DataStructures;
// See https://aka.ms/new-console-template for more information
using AlgoDSPlay.Sorting;
using AlgoDSPlay.Design;
using static AlgoDSPlay.Design.SparseVector;

int res1= (4-3)/2;
double res2 = (double)(4-3)/2 ;
var output = ArrayOps.CommonCharacters(new string[]{"abc","bcd", "cbaccd"});
var res3 = RealProbs.GroupAnagrams(new List<string>{"yo", "act", "flop", "tac", "foo", "cat", "oy", "olfp"});
var resg2= RealProbs.ShortenPath("/foo/../test/../test/../foo//bar/./baz");
var resG = RealProbs.KrusaklMST(new int[][][]{
                                new int[][]{
                                    new int[]{1,3},
                                    new int[]{2,5}
                                },
                                new int[][]{
                                    new int[]{0,3},
                                    new int[]{2,10},
                                    new int[]{3,12}
                                },
                                new int[][]{
                                    new int[]{0,5},
                                    new int[]{1,10}
                                },
                                new int[][]{
                                    new int[]{1,12},                                  
                                },
                                }
);

LinkedList lse = new LinkedList(1);
lse.Next=new LinkedList(3);
lse.Next.Next=new LinkedList(4);
lse.Next.Next.Next=new LinkedList(5);
lse.Next.Next.Next.Next=new LinkedList(6);
//LinkedList lst2 = LinkedListOps.RemoveDuplicatesFromLinkedList1(lse);
string str2= StringOps.ReverseWordsInString("I AM ");
var resu1 = ArrayOps.IndexEqualsValue(new int[]{-5, -4, -3, -2, -1, 0, 1, 3, 5, 6, 7, 11, 12, 14, 19, 20});
var resss = MatrixOps.KnapsackProblem(new int[,]{
    {465, 100},
    {400, 85},
    {255, 55},
    {350, 45},
    {650, 130},
    {1000, 190},
    {455, 100},
    {100, 25},
    {1200, 190},
    {320, 65},
    {750, 100},
    {50, 45},
    {550, 65},
    {100, 50},
    {600, 70},
    {240, 40}
},200);
var resu = RealProbs.GenerateDivTags(3);
var ress = StringOps.IsBigStringContainsSmallStringUsingKMP("abxabcabcaby","abcaby");
//var res = StackProbs.CollidingAsteroids(new int[]{-3,5,-8,6,7,-4,-7});
var str=SubstringOps.LongestSubstringWithoutDuplication("clementisacap");
MatrixOps.MinimumPassesOfMatrix(new int[][]{
                        new int[]{0,-1,-3,2,0},
                        new int[]{1,-2,-5,-1,-3},
                        new int[]{3,0,0,-4,-1},
                        }
                        );
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