using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    729. My Calendar I
    https://leetcode.com/problems/my-calendar-i/

    */
    public class MyCalendar
    {

        /*      
        Approach #1: Brute Force
        Complexity Analysis
        Let N be the number of events booked.
        •	Time Complexity: O(N2). For each new event, we process every previous event to decide whether the new event can be booked. This leads to ∑kNO(k)=O(N2) complexity.
        •	Space Complexity: O(N), the size of the calendar.
        */
        public class MyCalendarNaive
        {
            private List<int[]> calendar;

            public MyCalendarNaive()
            {
                calendar = new List<int[]>();
            }

            public bool Book(int start, int end)
            {
                foreach (int[] interval in calendar)
                {
                    if (interval[0] < end && start < interval[1])
                    {
                        return false;
                    }
                }
                calendar.Add(new int[] { start, end });
                return true;
            }
        }


        /*      
         Approach #2: Sorted List + Binary Search
         Complexity Analysis
         Like Approach 1, let N be the number of events booked.
         •	Time Complexity: O(NlogN). For each new event, we search that the event is legal in O(logN) time, then insert it in O(logN) time.
         •	Space Complexity: O(N), the size of the data structures used.

         */
        class MyCalendarSortedDict
        {
            private SortedDictionary<int, int> calendar;

            public MyCalendarSortedDict()
            {
                calendar = new SortedDictionary<int, int>();
            }

            public bool Book(int start, int end)
            {
                int? previousKey = null;
                int? nextKey = null;

                foreach (var key in calendar.Keys)
                {
                    if (key <= start)
                    {
                        previousKey = key;
                    }
                    if (key >= start && nextKey == null)
                    {
                        nextKey = key;
                    }
                }

                if ((previousKey == null || calendar[previousKey.Value] <= start) &&
                    (nextKey == null || end <= nextKey))
                {
                    calendar[start] = end;
                    return true;
                }
                return false;
            }
        }

    }
}