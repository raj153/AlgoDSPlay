using System;

namespace AlgoDSPlay
{
    public class IntervalProbs
    {
        /*         986. Interval List Intersections.
        https://leetcode.com/problems/interval-list-intersections/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class IntervalIntersectionSol
        {
/*             Approach 1: Merge Intervals
Complexity Analysis
•	Time Complexity: O(M+N), where M,N are the lengths of A and B respectively.
•	Space Complexity: O(M+N), the maximum size of the answer.

 */
            public int[][] IntervalIntersection(int[][] firstList, int[][] secondList)
            {
                List<int[]> ans = new();
                int i = 0, j = 0;

                while (i < firstList.Length && j < secondList.Length)
                {
                    // Let's check if A[i] intersects B[j].
                    // lo - the startpoint of the intersection
                    // hi - the endpoint of the intersection
                    int lo = Math.Max(firstList[i][0], secondList[j][0]);
                    int hi = Math.Min(firstList[i][1], secondList[j][1]);
                    if (lo <= hi)
                        ans.Add(new int[] { lo, hi });

                    // Remove the interval with the smallest endpoint
                    if (firstList[i][1] < secondList[j][1])
                        i++;
                    else
                        j++;
                }
                List<int>[] a =new List<int>[10];
               
                return ans.ToArray();
            }
        }








    }
}
