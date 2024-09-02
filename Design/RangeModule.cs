using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    715. Range Module
    https://leetcode.com/problems/range-module/
    */
    public class RangeModule
    {
        /*
        Approach #1: Maintain Sorted Disjoint Intervals(DI)

        Time Complexity: Let K be the number of elements in ranges. addRange and removeRange operations have O(K) complexity. queryRange has O(logK) complexity. 
                         Because addRange, removeRange adds at most 1 interval at a time, you can bound these further. For example, if there are A addRange, R removeRange, and Q queryRange number of operations respectively, 
                         we can express our complexity as O((A+R)^2*Qlog(A+R)).
        Space Complexity: O(A+R), the space used by ranges. 
        */
        public class RangeModuleDI
        {
            private class Interval
            {
                public int Start { get; }
                public int End { get; }

                public Interval(int start, int end)
                {
                    Start = start;
                    End = end;
                }
            }

            private List<Interval> intervals;

            public RangeModuleDI()
            {
                intervals = new List<Interval>();
            }

            public void AddRange(int left, int right)
            {
                List<Interval> result = new List<Interval>();
                int n = intervals.Count, cur = 0;
                for (; cur < n; cur++)
                {
                    Interval interval = intervals[cur];
                    if (interval.End < left)
                    {
                        result.Add(interval);
                    }
                    else if (interval.Start > right)
                    {
                        result.Add(new Interval(left, right));
                        break;
                    }
                    else
                    {
                        left = Math.Min(left, interval.Start);
                        right = Math.Max(right, interval.End);
                    }
                }
                if (cur == n)
                {
                    result.Add(new Interval(left, right));
                }
                while (cur < n) result.Add(intervals[cur++]);
                intervals = result;
            }

            public bool QueryRange(int left, int right)
            {
                int l = 0, r = intervals.Count - 1;
                while (l <= r)
                {
                    int mid = l + (r - l) / 2;
                    Interval interval = intervals[mid];
                    if (interval.Start >= right)
                    {
                        r = mid - 1;
                    }
                    else if (interval.End <= left)
                    {
                        l = mid + 1;
                    }
                    else
                    {
                        return interval.Start <= left && interval.End >= right;
                    }
                }
                return false;
            }

            public void RemoveRange(int left, int right)
            {
                List<Interval> result = new List<Interval>();
                int n = intervals.Count, cur = 0;
                for (; cur < n; cur++)
                {
                    Interval interval = intervals[cur];
                    if (interval.End <= left)
                    {
                        result.Add(interval);
                    }
                    else if (interval.Start >= right)
                    {
                        result.Add(interval);
                    }
                    else
                    {
                        if (interval.Start < left) result.Add(new Interval(interval.Start, left));
                        if (interval.End > right) result.Add(new Interval(right, interval.End));
                    }
                }
                intervals = result;
            }

        }
        //2, SortedSet
        public class RangeModuleSS
        {
            private SortedSet<Interval> ranges;

            public RangeModuleSS()
            {
                ranges = new SortedSet<Interval>();
            }

            public void AddRange(int left, int right)
            {
                IEnumerator<Interval> iterator = ranges.GetViewBetween(new Interval(0, left), new Interval(int.MaxValue, int.MaxValue)).GetEnumerator();
                while (iterator.MoveNext())
                {
                    Interval interval = iterator.Current;
                    if (right < interval.Left)
                    {
                        break;
                    }
                    left = Math.Min(left, interval.Left);
                    right = Math.Max(right, interval.Right);
                    iterator.Dispose();
                    ranges.Remove(interval);
                }
                ranges.Add(new Interval(left, right));
            }

            public bool QueryRange(int left, int right)
            {
                Interval interval = ranges.TryGetValue(new Interval(0, left), out interval) ? interval : null;
                return (interval != null && interval.Left <= left && right <= interval.Right);
            }

            public void RemoveRange(int left, int right)
            {
                IEnumerator<Interval> iterator = ranges.GetViewBetween(new Interval(0, left - 1), new Interval(int.MaxValue, int.MaxValue)).GetEnumerator();
                List<Interval> toAdd = new List<Interval>();
                while (iterator.MoveNext())
                {
                    Interval interval = iterator.Current;
                    if (right < interval.Left)
                    {
                        break;
                    }
                    if (interval.Left < left)
                    {
                        toAdd.Add(new Interval(interval.Left, left));
                    }
                    if (right < interval.Right)
                    {
                        toAdd.Add(new Interval(right, interval.Right));
                    }
                    iterator.Dispose();
                    ranges.Remove(interval);
                }
                foreach (Interval interval in toAdd)
                {
                    ranges.Add(interval);
                }
            }
        }

        public class Interval : IComparable<Interval>
        {
            public int Left { get; }
            public int Right { get; }

            public Interval(int left, int right)
            {
                this.Left = left;
                this.Right = right;
            }

            public int CompareTo(Interval other)
            {
                if (this.Right == other.Right)
                {
                    return this.Left - other.Left;
                }
                return this.Right - other.Right;
            }
        }
    }
    /**
    * Your RangeModule object will be instantiated and called as such:
    * RangeModule obj = new RangeModule();
    * obj.AddRange(left,right);
    * bool param_2 = obj.QueryRange(left,right);
    * obj.RemoveRange(left,right);
     */
}