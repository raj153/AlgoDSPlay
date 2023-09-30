using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DP
{
    public class MaxMin
    {

        //https://www.algoexpert.io/questions/maximize-expression
        public static int FindMaxValueOfExpDP(int[] numArray){
            
            //T:O(n)| S:O(n)
            if (numArray.Length < 4)
                return 0;

            List<int> maxOfA = new List<int>{numArray[0]};
            List<int> maxOfAMinuB = new List<int>{Int32.MinValue};
            List<int> maxOfAMinueBPlusC = new List<int>{Int32.MaxValue, Int32.MinValue};
            List<int> maxOfAMinueBPlusCMinusD = new List<int>{Int32.MaxValue, Int32.MinValue,Int32.MinValue};
            
            for(int idx=1; idx<numArray.Length; ++idx){
                int currentMax = Math.Max(maxOfA[idx-1],numArray[idx]);
                maxOfA.Add(currentMax);
            }

            for(int idx=1; idx< numArray.Length;++idx){
                int currentMax = Math.Max(maxOfAMinuB[idx-1], maxOfA[idx]-numArray[idx]);
                maxOfAMinuB.Add(currentMax);
            }

            for(int idx=2; idx< numArray.Length; ++idx){
                int currentMax = Math.Max(maxOfAMinueBPlusC[idx-1], maxOfAMinuB[idx-1]+numArray[idx]);
                maxOfAMinueBPlusC.Add(currentMax);
            }

            for(int idx=3; idx<numArray.Length; ++idx){
                int currentMax = Math.Max(maxOfAMinueBPlusCMinusD[idx-1], maxOfAMinueBPlusC[idx-1]-numArray[idx]);
                maxOfAMinueBPlusCMinusD.Add(currentMax);
            }

            return maxOfAMinueBPlusCMinusD[numArray.Length-1];
        }
        public static int FindMaxValueOfExpNaive(int[] numArray){
            //T:O(n^4)| S:O(1)
            if (numArray.Length < 4)
                return 0;

            int maxValueFound = int.MinValue;

            for(int a=0; a< numArray.Length; ++a){
                int aValue = numArray[a];

                for(int b=a+1; b< numArray.Length;++b){
                    int bValue = numArray[b];

                    for(int c=b+1; c<numArray.Length; ++c){
                        int cValue = numArray[c];

                        for(int d=c+1; d<numArray.Length; ++d){
                            int exprValue = EvaluateExpression(aValue, bValue, cValue, numArray[d]);
                            maxValueFound = Math.Max(exprValue, maxValueFound);
                        }


                    }



                }

            }
            return maxValueFound;
        }

        private static int EvaluateExpression(int aValue, int bValue, int cValue, int dValue)
        {
            return aValue-bValue+cValue-dValue;
        }
    }
}