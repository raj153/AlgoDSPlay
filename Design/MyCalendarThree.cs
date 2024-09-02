using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    732. My Calendar III
    https://leetcode.com/problems/my-calendar-iii/

    */
    public class MyCalendarThree
    {
        //Approach 1: Sweep-line Algorithm(SLA)
        /*
        Let N be the number of events booked.
        Time Complexity: O(N^2). For each new event, we update the changes at two points in O(logN) because we keep the HashMap in sorted order. Then we traverse diff in O(N) time.
        Space Complexity: O(N), the size of diff.
        */
        public class MyCalendarThreeSLA
        {
            private SortedDictionary<int, int> diff;

            public MyCalendarThreeSLA()
            {
                diff = new SortedDictionary<int, int>();
            }

            public int Book(int startTime, int endTime)
            {
                diff[startTime] = diff.GetValueOrDefault(startTime, 0) + 1;
                diff[endTime] = diff.GetValueOrDefault(endTime, 0) - 1;

                int res = 0, cur = 0;
                foreach (int delta in diff.Values)
                {
                    cur += delta;
                    res = Math.Max(res, cur);
                }
                return res;
            }
        }
        //Approach 2: Segment Tree(ST)
        /*
        Let N be the number of events booked and C be the largest time (i.e., 10^9  in this problem)
        Time Complexity: O(NlogC). The max possible depth of the segment tree is logC. At most O(logC) nodes will be visited in each update operation. Thus, the time complexity of booking N new events is O(NlogC).
        Space Complexity: O(NlogC). Instead of creating a segment tree of 4C at first, we create tree nodes dynamically when needed. Every time update is called, we create at most O(logC) nodes because the max depth of the segment tree is logC.
        */
        public class MyCalendarThreeST
        {
            private Dictionary<int, int> vals;
            private Dictionary<int, int> lazy;

            public MyCalendarThreeST()
            {
                vals = new Dictionary<int, int>();
                lazy = new Dictionary<int, int>();

            }

            public int Book(int startTime, int endTime)
            {
                Update(startTime, endTime - 1, 0, 1000000000, 1);
                return vals.GetValueOrDefault(1, 0);

            }
            public void Update(int startTime, int endTime, int left, int right, int idx)
            {
                if (startTime > right || endTime < left)
                    return;
                if (startTime <= left && right <= endTime)
                {
                    vals[idx] = vals.GetValueOrDefault(idx, 0) + 1;
                    lazy[idx] = lazy.GetValueOrDefault(idx, 0) + 1;
                }
                else
                {
                    int mid = (left + right) / 2;
                    Update(startTime, endTime, left, mid, idx * 2);
                    Update(startTime, endTime, mid + 1, right, idx * 2 + 1);
                    vals[idx] = lazy.GetValueOrDefault(idx, 0)
                            + Math.Max(vals.GetValueOrDefault(idx * 2, 0), vals.GetValueOrDefault(idx * 2 + 1, 0));
                }
            }

        }
        //Approach 3: Balanced Tree(BT)
        /*
        Let N be the number of events booked.
        Time Complexity: O(N^2) in the worst case. For each new [start, end), we find the intervals that contains point start and end in O(logN) time, split and add new intervals in O(logN) time. We increase at most 2 new intervals each time, so the size of intervals(or starts) is at most 2N+1. 
                        Finally, we enumerate all intervals contained in [start, end) to get the max number of events, which takes O(N) time. Therefore, the overall time complexity of booking N events is O(N^2)
                        Though the time complexity looks not ideal in the worst case, if the given [start, end) is distributed uniformly, the time complexity is O(NloglogN) (See also: Crate chtholly_tree). 
                        The proof is not easy so we ignore it here.
        Space Complexity: O(N), the size of intervals(or starts) is at most 2N+1 as we analyzed before.
        */
        public class MyCalendarThreeBT
        {
            private SortedDictionary<int, int> starts;
            private int res;

            public MyCalendarThreeBT()
            {
                starts = new SortedDictionary<int, int>();
                starts[0] = 0;
                res = 0;
            }

            private void Split(int x)
            {
                var prev = starts.Keys.Where(k => k <= x).DefaultIfEmpty().Max();
                var next = starts.Keys.Where(k => k >= x).DefaultIfEmpty().Min();
                if (next == x)
                    return;
                starts[x] = starts[prev];
            }

            public int Book(int start, int end)
            {
                Split(start);
                Split(end);
                foreach (var interval in starts.Where(kvp => kvp.Key >= start && kvp.Key < end))
                {
                    res = Math.Max(res, (starts[interval.Key] += 1) + 1);
                }
                return res;
            }
        }


    }
}