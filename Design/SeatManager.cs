using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1845. Seat Reservation Manager
    https://leetcode.com/problems/seat-reservation-manager/description/

    */
    public class SeatManager
    {
        /*        
        Approach 1: Min Heap

        Complexity Analysis
        Let m be the maximum number of calls made.
        •	Time complexity: O((m+n)⋅logn)
            o	While initializing the SeatManager object, we iterate over all n seats and push it into our heap, each push operation takes O(logn) time, thus, overall it will take O(nlogn) time.
            o	In the reserve() method, we pop the minimum-valued element from the availableSeats heap, which takes O(logn) time.
            o	In the unreserve(seatNumber) method, we push the seatNumber into the availableSeats heap which will also take O(logn) time.
            o	There are a maximum of m calls to reserve() or unreserve() methods, thus the overall time complexity is O(m⋅logn).
        •	Space complexity: O(n)
            o	The availableSeats heap contains all n elements, taking O(n) space.       

        */

        class SeatManagerMinHeap
        {
            // Min heap to store all unreserved seats.
            private PriorityQueue<int, int> availableSeats;

            public SeatManagerMinHeap(int n)
            {
                // Initially all seats are unreserved.
                availableSeats = new PriorityQueue<int, int>();
                for (int seatNumber = 1; seatNumber <= n; ++seatNumber)
                {
                    availableSeats.Enqueue(seatNumber, seatNumber);
                }
            }

            public int Reserve()
            {
                // Get the smallest-numbered unreserved seat from the min heap.
                int seatNumber = availableSeats.Dequeue();
                return seatNumber;
            }

            public void Unreserve(int seatNumber)
            {
                // Push the unreserved seat back into the min heap.
                availableSeats.Enqueue(seatNumber, seatNumber);
            }
        }

        /*        
        Approach 2: Min Heap (without pre-initialization)

        Complexity Analysis
        Let m be the maximum number of calls made.
        •	Time complexity: O(m⋅logn)
            o	While initializing the SeatManager object, we perform constant time operations.
            o	In the reserve() method, in the worst-case, we will pop the minimum-valued element from the availableSeats heap which will take O(logn).
            o	In the unreserve(seatNumber) method, we push the seatNumber into the availableSeats heap which will also take O(logn) time.
            o	There are a maximum of m calls to reserve() or unreserve() methods, thus the overall time complexity is O(m⋅logn).
        •	Space complexity: O(n)
            o	The availableSeats heap can contain n elements in it. So in the worst case, it will take O(n) space.

        */
        class SeatManagerMinHeapExt
        {
            // Marker to point to unreserved seats.
            int marker;

            // Min heap to store all unreserved seats.
            private PriorityQueue<int, int> availableSeats;

            public SeatManagerMinHeapExt(int n)
            {
                // Set marker to the first unreserved seat.
                marker = 1;
                // Initialize the min heap.
                availableSeats = new PriorityQueue<int, int>();
            }

            public int Reserve()
            {
                // If min-heap has any element in it, then,
                // get the smallest-numbered unreserved seat from the min heap.
                if (availableSeats.Count > 0)
                {
                    return availableSeats.Dequeue();

                }

                // Otherwise, the marker points to the smallest-numbered seat.
                int seatNumber = marker;
                marker++;
                return seatNumber;
            }

            public void Unreserve(int seatNumber)
            {
                // Push unreserved seat in the min heap.
                availableSeats.Enqueue(seatNumber, seatNumber);
            }
        }

        /*        
        Approach 3: Sorted/Ordered Set

        Complexity Analysis
        Let m be the maximum number of calls made.
        •	Time complexity: O(m⋅logn)
        o	While initializing the SeatManager object, we perform constant time operations.
        o	In the reserve() method, we pop the minimum-valued element from the availableSeats set which takes O(logn) time.
        o	In the unreserve(seatNumber) method, we push the seatNumber into the availableSeats set which will also take O(logn) time.
        o	There are a maximum of m calls to reserve() or unreserve() methods, thus the overall time complexity is O(m⋅logn).
        •	Space complexity: O(n)
        o	The availableSeats set can contain n elements in it. So in the worst case, it will take O(n) space.

        */
        class SeatManagerSortedSet
        {
            // Marker to point to unreserved seats.
            int marker;

            // Sorted set to store all unreserved seats.
            SortedSet<int> availableSeats;

            public SeatManagerSortedSet(int n)
            {
                // Set marker to the first unreserved seat.
                marker = 1;
                // Initialize the sorted set.
                availableSeats = new SortedSet<int>();
            }

            public int Reserve()
            {
                // If the sorted set has any element in it, then,
                // get the smallest-numbered unreserved seat from it.
                if (availableSeats.Count >0)
                {
                    int seatNum = availableSeats.First();
                    availableSeats.Remove(seatNum);
                    return seatNum;
                }

                // Otherwise, the marker points to the smallest-numbered seat.
                int seatNumber = marker;
                marker++;
                return seatNumber;
            }

            public void Unreserve(int seatNumber)
            {
                // Push the unreserved seat in the sorted set.
                availableSeats.Add(seatNumber);
            }
        }

    }
    /**
 * Your SeatManager object will be instantiated and called as such:
 * SeatManager obj = new SeatManager(n);
 * int param_1 = obj.Reserve();
 * obj.Unreserve(seatNumber);
 */
}