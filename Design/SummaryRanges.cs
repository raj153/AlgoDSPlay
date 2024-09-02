using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    352. Data Stream as Disjoint Intervals
    https://leetcode.com/problems/data-stream-as-disjoint-intervals/description/

    Approach 1: Save all values in an ordered set

    Here, N is the total number of calls of addNum.
•	Time complexity: O(log(N)) for addNum, O(N) for getIntervals.
    For addNum, in the worst case, we remove 2 entries from the TreeMap and add 1 entry, the time complexity for each operation is O(log(N)).
    For getIntervals, we iterate all the entries in the TreeMap which is the same as traversing the whole tree, so the time complexity is O(N).
•	Space complexity: O(N).
    This is just the space to save all the intervals in the TreeMap.

    */
    public class SummaryRanges
    {
        private SortedSet<int> values;

        public SummaryRanges()
        {
            values = new SortedSet<int>();
        }

        public void addNum(int value)
        {
            values.Add(value);
        }

        public int[][] getIntervals()
        {
            if (values.Count == 0)
            {
                return new int[0][];
            }
            List<int[]> intervals = new List<int[]>();
            int left = -1, right = -1;
            foreach (int value in values)
            {
                if (left < 0)
                {
                    left = right = value;
                }
                else if (value == right + 1)
                {
                    right = value;
                }
                else
                {
                    intervals.Add(new int[] { left, right });
                    left = right = value;
                }
            }
            intervals.Add(new int[] { left, right });
            return intervals.ToArray();
        }


    }
}