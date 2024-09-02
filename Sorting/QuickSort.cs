using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Sorting
{
    //https://www.algoexpert.io/questions/quick-sort
    public class QuickSort
    {
        // Best: O(nlog(n)) time | O(log(n)) space
        // Average: O(nlog(n)) time | O(log(n)) space
        // Worst: O(n^2) time | O(log(n)) space
        public static int[] Sort(int[] array)
        {
            Sort(array, 0, array.Length - 1);
            return array;
        }

        public static void Sort(int[] array, int startIdx, int endIdx)
        {
            if (startIdx >= endIdx)
            {
                return;
            }
            int pivotIdx = startIdx;
            int leftIdx = startIdx + 1;
            int rightIdx = endIdx;
            while (rightIdx >= leftIdx)
            {
                if (array[leftIdx] > array[pivotIdx] && array[rightIdx] < array[pivotIdx])
                {
                    swap(leftIdx, rightIdx, array);
                }
                if (array[leftIdx] <= array[pivotIdx])
                {
                    leftIdx += 1;
                }
                if (array[rightIdx] >= array[pivotIdx])
                {
                    rightIdx -= 1;
                }
            }
            swap(pivotIdx, rightIdx, array);
            bool leftSubarrayIsSmaller =
            rightIdx - 1 - startIdx < endIdx - (rightIdx + 1);
            if (leftSubarrayIsSmaller)
            {
                Sort(array, startIdx, rightIdx - 1);
                Sort(array, rightIdx + 1, endIdx);
            }
            else
            {
                Sort(array, rightIdx + 1, endIdx);
                Sort(array, startIdx, rightIdx - 1);
            }
        }

        public static void swap(int i, int j, int[] array)
        {
            int temp = array[j];
            array[j] = array[i];
            array[i] = temp;
        }

    }
}