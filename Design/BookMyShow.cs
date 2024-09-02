using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2286. Booking Concert Tickets in Groups
    https://leetcode.com/problems/booking-concert-tickets-in-groups/description/	

    Using segment tree 
    
    Time complexity: Building the segment tree takes O(nlog⁡n) Each query and update operation takes O(log⁡n)
    Space complexity: The segment tree takes O(n) space

    */
    public class BookMyShow
    {
        private int rowCount;
        private int seatCount;
        private Tuple<long, long>[] treeData;

        public BookMyShow(int n, int m)
        {
            rowCount = n;
            seatCount = m;

            int treeSize = 1;
            while (treeSize < 2 * n)
                treeSize *= 2;

            treeData = new Tuple<long, long>[treeSize];
            BuildTree(0, 0, n - 1);
        }

        private void BuildTree(int node, int start, int end)
        {
            if (start == end)
            {
                treeData[node] = Tuple.Create((long)seatCount, (long)seatCount);
                return;
            }
            int mid = (start + end) / 2;
            BuildTree(2 * node + 1, start, mid);
            BuildTree(2 * node + 2, mid + 1, end);
            treeData[node] = Tuple.Create((long)seatCount, (long)(end - start + 1) * seatCount);
        }

        public int[] Gather(int k, int maxRow)
        {
            var result = QueryMax(0, 0, rowCount - 1, k, maxRow);
            if (result != null)
            {
                UpdateMax(0, 0, rowCount - 1, result[0], k);
                return result;
            }
            return new int[0];
        }

        private int[] QueryMax(int node, int start, int end, int k, int maxRow)
        {
            if (start > maxRow || treeData[node].Item1 < k)
                return null;
            if (start == end)
                return new int[] { start, seatCount - (int)treeData[node].Item1 };

            int mid = (start + end) / 2;
            var leftResult = QueryMax(2 * node + 1, start, mid, k, maxRow);
            if (leftResult != null)
                return leftResult;

            return QueryMax(2 * node + 2, mid + 1, end, k, maxRow);
        }

        private void UpdateMax(int node, int start, int end, int row, int k)
        {
            if (start > row || end < row)
                return;
            if (start == end)
            {
                treeData[node] = Tuple.Create(treeData[node].Item1 - k, treeData[node].Item2 - k);
                return;
            }

            int mid = (start + end) / 2;
            UpdateMax(2 * node + 1, start, mid, row, k);
            UpdateMax(2 * node + 2, mid + 1, end, row, k);
            treeData[node] = Tuple.Create(Math.Max(treeData[2 * node + 1].Item1, treeData[2 * node + 2].Item1), treeData[node].Item2 - k);
        }

        public bool Scatter(int k, int maxRow)
        {
            long totalSeats = QuerySum(0, 0, rowCount - 1, maxRow);
            if (totalSeats < k)
                return false;
            UpdateSum(0, 0, rowCount - 1, k, maxRow);
            return true;
        }

        private long QuerySum(int node, int start, int end, int maxRow)
        {
            if (start > maxRow)
                return 0;
            if (end <= maxRow)
                return treeData[node].Item2;

            int mid = (start + end) / 2;
            return QuerySum(2 * node + 1, start, mid, maxRow) + QuerySum(2 * node + 2, mid + 1, end, maxRow);
        }

        private void UpdateSum(int node, int start, int end, int k, int maxRow)
        {
            if (start > maxRow)
                return;

            if (start == end)
            {
                treeData[node] = Tuple.Create(treeData[node].Item1 - k, treeData[node].Item2 - k);
                return;
            }

            int mid = (start + end) / 2;
            treeData[node] = Tuple.Create(treeData[node].Item1, treeData[node].Item2 - k);

            if (treeData[2 * node + 1].Item2 >= k)
            {
                UpdateSum(2 * node + 1, start, mid, k, maxRow);
            }
            else
            {
                int remaining = k - (int)treeData[2 * node + 1].Item2;
                UpdateSum(2 * node + 1, start, mid, (int)treeData[2 * node + 1].Item2, maxRow);
                UpdateSum(2 * node + 2, mid + 1, end, remaining, maxRow);
            }

            treeData[node] = Tuple.Create(Math.Max(treeData[2 * node + 1].Item1, treeData[2 * node + 2].Item1), treeData[node].Item2);
        }
    }
    /**
 * Your BookMyShow object will be instantiated and called as such:
 * BookMyShow obj = new BookMyShow(n, m);
 * int[] param_1 = obj.Gather(k,maxRow);
 * bool param_2 = obj.Scatter(k,maxRow);
 */

}