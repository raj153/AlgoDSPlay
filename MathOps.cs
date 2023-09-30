using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class MathOps
    {
        //https://www.algoexpert.io/questions/nth-fibonacci
        public static int GetNthFib(int n){

            //1.Naive Recursion- T:O(2^n) | S:O(n)
            int result = GetNthFibNaiveRec(n);

            //2.Recursion with Memorization- T:O(n) | S:O(n)
            Dictionary<int, int> memoize = new Dictionary<int, int>();
            memoize.Add(1,0);
            memoize.Add(2,1);
            result = GetNthFibMemorizeRec(n,memoize);

            //3.Iterative - T:O(n) | S:O(1)
            result = GetNthFibIterative(n);
            result = GetNthFibIterative2(n);
            return result;
        }

        private static int GetNthFibIterative2(int n)
        {
            int[] lastTwo = new int[]{0,1};

            int counter =3;

            while(counter<=n){
                int nextFib = lastTwo[0]+lastTwo[1];
                lastTwo[0]=lastTwo[1];
                lastTwo[1] = nextFib;
                counter++;
            }
            return n > 1 ? lastTwo[1] : lastTwo[0];
        }

        private static int GetNthFibIterative(int n)
        {
            int first=0, second=1;
            //0 1 1 2 3 5
            int result=0;
            for(int i=3; i<=n;i++){                
                int nextFib = first+second;
                first = second;
                second = nextFib;
                //result += nextFib;
                result = nextFib;
            }
            return result;
        }

        private static int GetNthFibMemorizeRec(int n, Dictionary<int, int> memoize)
        {
            if(memoize.ContainsKey(n))
                return memoize[n];
            
            memoize.Add(n, GetNthFibMemorizeRec(n-1, memoize)+GetNthFibMemorizeRec(n-2, memoize));
            return memoize[n];

        }

        private static int GetNthFibNaiveRec(int n)
        {
            if (n == 2) return 1;
            if( n==1) return 0;

            return GetNthFibNaiveRec(n-1)+GetNthFibNaiveRec(n-2);

        }
    }
}