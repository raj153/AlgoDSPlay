using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Sorting
{
    //https://www.algoexpert.io/questions/radix-sort
    public class RadixSort
    {
        //O(d * (n + b)) time | O(n + b) space - where n is the length of the input array, 
        //d is the max number of digits, and b is the base of the numbering system used

        public List<int> Sort(List<int> array){
            if(array.Count ==0 ) return array;

            int maxNumber = array.Max();

            int digit =0;
            while((maxNumber/Math.Pow(10, digit)) >0){
                CountingSort(array , digit);
                digit +=1;
            }
            return array;
        }

        private void CountingSort(List<int> array, int digit)
        {
            int[] sortedArray = new int[array.Count];
            int[] countArray = new int[10]; //0-9

            int digitColumn =(int)(Math.Pow(10, digit));            
            foreach(var num in array){
                int countIndex = (num/digitColumn)%10;
                countArray[countIndex]+=1;
            }

            for(int idx=1; idx< 10; idx++){
                countArray[idx]+=countArray[idx-1];
            }
            //Right to left traverse to keep algo stable asin to keep ordering intact for already sorted numbers ex: 67, 67 or 67,68
            for(int idx=array.Count-1; idx>-1; idx--){ //idx>-1 == idx >=0
                int countIndex = (array[idx]/digitColumn)%10;
                countArray[countIndex] -=1;
                int sortedIndex = countArray[countIndex];
                sortedArray[sortedIndex] = array[idx];
            }
            for(int idx=0; idx < array.Count; idx++){ //Copy elements to original arry to have sort numbers by respective digit 
                array[idx]=  sortedArray[idx];
            }
        }
    }
}