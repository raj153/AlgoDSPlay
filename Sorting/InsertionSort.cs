using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Sorting
{
    public class InsertionSort
    {   
        // Best: O(n) time | O(1) space
        // Average: O(n^2) time | O(1) space
        // Worst: O(n^2) time | O(1) space
        public static int[] Sort(int[] array){

            if(array.Length ==0) return new int[]{};


            for(int i=1; i< array.Length; i++)
            {   
                int j=i;
                while(j>0 && array[j] < array[j-1]){
                    Swap(array, array[j], array[j-1]);
                    j--;
                }
            }
            return array;

        }

        private static void Swap(int[] array, int j, int i)
        {
            int temp = array[i];
            array[j]=array[i];
            array[i]=temp;
        }
    }
}