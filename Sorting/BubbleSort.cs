using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Sorting
{
    //https://www.algoexpert.io/questions/bubble-sort
    public class BubbleSort
    {
        public static List<int> Sort(List<int> numArray){

            for(int current=0; current < numArray.Count; ++current){

                for(int next=current+1; next < numArray.Count; ++next ){
                    if(numArray[current]>numArray[next]){
         
                        Swap(current, next, numArray);
                    }
                }
                
            }

            return numArray;
        }

        private static void Swap(int current, int next, List<int> numArray)
        {
            int tmp = numArray[current];
            numArray[current]= numArray[next];
            numArray[next]= tmp;
                        
        }

        public static List<int> OptimalSort(List<int> numArray){

            int counter =0;
            bool isSorted = false;

            while(!isSorted){
                isSorted= true;
                for(int step=0; step< numArray.Count-1-counter; ++step){
                    if(numArray[step]>numArray[step+1]){
                        Swap(step, step+1, numArray);
                        isSorted= false;
                    }
         
                }
                counter++;
            }

            return numArray;
        }
    }
}