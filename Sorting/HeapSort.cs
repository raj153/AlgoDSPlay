using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Sorting
{
    //https://www.algoexpert.io/questions/heap-sort
    public class HeapSort
    {
        public static int[] Sort(int[] array){
            // Best: O(nlog(n)) time | O(1) space
            // Average: O(nlog(n)) time | O(1) space
            // Worst: O(nlog(n)) time | O(1) space
            BuildMaxHeap(array);
            for(int endIdx=array.Length-1; endIdx>0; endIdx--){
                Swap(0, endIdx, array);
                SiftDown(0, endIdx-1, array);
            }
            return array;
        }

        private static void BuildMaxHeap(int[] array)
        {
            int firstParentIdx = (array.Length-2)/2;
            for(int curIdx = firstParentIdx; curIdx >=0; curIdx--){
                SiftDown(curIdx, array.Length-1, array);
            }
        }

        private static void SiftDown(int curIdx, int endIdx, int[] heap)
        {
            int childOneIdx = curIdx*2+1;
            while(childOneIdx <= endIdx){
                int childTwoIdx = curIdx*2+2 <= endIdx ? curIdx*2+2 :-1;
                int idxToSwap;
                if(childTwoIdx != -1 && heap[childTwoIdx] > heap[childOneIdx]){
                    idxToSwap = childTwoIdx;
                }else{
                    idxToSwap = childOneIdx;
                }
                if(heap[idxToSwap] > heap[curIdx]){
                    Swap(curIdx, idxToSwap, heap);
                }else {
                    return;
                }
            }
        }

        private static void Swap(int curIdx, int idxToSwap, int[] heap)
        {
            int temp = heap[idxToSwap];
            heap[idxToSwap] = heap[curIdx];
            heap[curIdx]= temp;
        }
    }
}