using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DP
{
    public class MinJump
    {
         //https://www.algoexpert.io/questions/min-number-of-jumps
        public static int FindMinimumNumberOfJumpsNaive(int[] steps){
            //T:O(n^2)| S:O(n)
            int[] jumps = new int[steps.Length];
            Array.Fill(jumps, Int32.MaxValue);
            jumps[0]=0;
            //3,4,2,1,2,3,7,1,1,1,3
            for(int i=1; i<steps.Length; i++){

                for(int j=0; j<i; j--){
                    if(steps[j]+j >=i){
                        jumps[i]= Math.Min(jumps[j]+1,jumps[i]);
                    }

                }
            }   
            return jumps[jumps.Length-1];

        }
        public static int FindMinimumNumberOfJumpsOptimal(int[] steps){
            if(steps.Length ==1) return 0;
            //T:O(n)| S:O(1)
            int jumps=0;
            int maxReach=steps[0];
            int stepCounter = steps[0];
            //0 1 2 3 4 5 6 7 8 9 10
            //3,4,2,1,2,3,7,1,1,1,3         
            for(int i=1; i<steps.Length-1;i++){
                maxReach =  Math.Max(maxReach, i+steps[i]);
                stepCounter--;
                if(stepCounter ==0){
                    jumps++;
                    stepCounter = maxReach-i;
                }
            }
            return jumps+1;

        }

    }
}