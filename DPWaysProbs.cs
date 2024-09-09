using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class DPWaysProbs
    {

        /*
     70. Climbing Stairs	
https://leetcode.com/problems/climbing-stairs/description/
   
        */
        public class WaysToClimbStairsSolution
        {
            /*
            Approach 1: Brute Force
•	Time complexity : O(2^n). Size of recursion tree will be 2^n. 
•	Space complexity : O(n). The depth of the recursion tree can go upto n.
            */
            public int WaysClimbStairsNaive(int n)
            {
                return WaysToClimbStairsRec(0, n);
            }
            private static int WaysToClimbStairsRec(int i, int n)
            {
                if (i > n)
                {
                    return 0;
                }

                if (i == n)
                {
                    return 1;
                }

                return WaysToClimbStairsRec(i + 1, n) + WaysToClimbStairsRec(i + 2, n);

            }
            /*
            Approach 2: Recursion with Memoization
          Complexity Analysis
        •	Time complexity : O(n). Size of recursion tree can go up to n.
        •	Space complexity : O(n). The depth of recursion tree can go up to n.

            */
            public int WaysToClimbStairsMemo(int n)
            {
                int[] memo = new int[n + 1];
                return WaysToClimbStairsMemoRec(0, n, memo);
            }
            private static int WaysToClimbStairsMemoRec(int i, int n, int[] memo)
            {
                if (i > n)
                {
                    return 0;
                }

                if (i == n)
                {
                    return 1;
                }

                if (memo[i] > 0)
                {
                    return memo[i];
                }

                memo[i] = WaysToClimbStairsMemoRec(i + 1, n, memo) + WaysToClimbStairsMemoRec(i + 2, n, memo);
                return memo[i];
            }

            /*
            Approach 3: Dynamic Programming
            Complexity Analysis
            •	Time complexity : O(n). Single loop up to n.
            •	Space complexity : O(n). dp array of size n is used.

            */
            public static int WaysToClimbStairsDP(int n)
            {
                if (n == 1)
                {
                    return 1;
                }
                int[] dp = new int[n + 1];
                dp[0] = 1;
                dp[1] = 1;
                dp[2] = 2;
                for (int i = 3; i <= n; i++)
                    dp[i] = dp[i - 1] + dp[i - 2];
                return dp[n];
            }
            /*
            Approach 4: Fibonacci Number
            Complexity Analysis
            •	Time complexity : O(n). Single loop up to n.
            •	Space complexity : O(1). Constant space is used.
            */
            public static int WaysToClimbStairsFib(int n)
            {
                if (n == 1)
                {
                    return 1;
                }
                int first = 1;
                int second = 2;
                for (int i = 3; i <= n; i++)
                {
                    int third = first + second;
                    first = second;
                    second = third;
                }
                return second;
            }
            /*
            Approach 5: Binets Method
            Complexity Analysis
            •	Time complexity : O(logn). Traversing on logn bits.
            •	Space complexity : O(1). Constant space is used.
            */
            public static int WaysToClimbStairsBinets(int n)
            {
                int[][] q = new int[2][] { new int[2] { 1, 1 }, new int[2] { 1, 0 } };
                int[][] res = Pow(q, n);
                return res[0][0];
            }
            public static int[][] Pow(int[][] a, int n)
            {
                int[][] ret =
                    new int[2][] { new int[2] { 1, 0 }, new int[2] { 0, 1 } };
                while (n > 0)
                {
                    if ((n & 1) == 1)
                    {
                        ret = Multiply(ret, a);
                    }

                    n >>= 1;
                    a = Multiply(a, a);
                }

                return ret;
            }

            public static int[][] Multiply(int[][] a, int[][] b)
            {
                int[][] c = new int[2][] { new int[2] { 0, 0 }, new int[2] { 0, 0 } };
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
                    }
                }

                return c;
            }
            /*
            Approach 6: Fibonacci Formula
            Complexity Analysis
            •	Time complexity : O(logn). pow method with O(logn) complexity.
            •	Space complexity : O(1). Constant space is used.
            */
            public static int WaysToClimbStairsFibFormula(int n)
            {
                double goldenRatio = (1 + Math.Sqrt(5)) / 2;
                return (int)Math.Round(Math.Pow(goldenRatio, n + 1) / Math.Sqrt(5));
                /*        double sqrt5 = Math.Sqrt(5);
                        double phi = (1 + sqrt5) / 2;
                        double psi = (1 - sqrt5) / 2;
                        return (int)((Math.Pow(phi, n + 1) - Math.Pow(psi, n + 1)) / sqrt5);*/
            }

        }






















    }
}