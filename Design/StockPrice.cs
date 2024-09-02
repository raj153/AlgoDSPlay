using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2034. Stock Price Fluctuation 
    https://leetcode.com/problems/stock-price-fluctuation/

    */
    public class StockPrice
    {
        /*
        Approach 1: Hashed and Sorted Map
        If N is the number of records in the input stream.
        Time complexity: O(NlogN)
            1. In the update function, we add and remove a record in both hashmap and sorted map. In hashmap, both operations take constant time, but in the sorted map they take O(logN) time.            
            2. Each call to the maximum, minimum, or current function will take only constant time to return the result.
                But in Java, getting first or last element of tree map takes log(N) time. Thus, here maximum, minimum functions will take O(log(N)) time for each function call.
            3. In the worst-case scenario, all N calls will be to the update function, which will require a total of O(NlogN) time.
        Space complexity: O(N)
            1. In the update function, we add and remove a record in both the hashmap and sorted map. Thus each function call takes O(1) space. So for N update calls, it will take O(N) space.
            2. The maximum, minimum, and current functions do not use any additional space.
            3. Thus, in the worst-case, we will add all N records in both the hashmap and sorted map, which takes O(N) space.
        */
        class StockPriceSortedDict
        {
            private int latestTime;
            // Store price of each stock at each timestamp.
            private Dictionary<int, int> timestampPriceMap;
            // Store stock prices in increasing order to get min and max price.
            private SortedDictionary<int, int> priceFrequency;

            public StockPriceSortedDict()
            {
                latestTime = 0;
                timestampPriceMap = new Dictionary<int, int>();
                priceFrequency = new SortedDictionary<int, int>();
            }

            public void Update(int timestamp, int price)
            {
                // Update latestTime to latest timestamp.
                latestTime = Math.Max(latestTime, timestamp);

                // If same timestamp occurs again, previous price was wrong. 
                if (timestampPriceMap.ContainsKey(timestamp))
                {
                    // Remove previous price.
                    int oldPrice = timestampPriceMap[timestamp];
                    priceFrequency[oldPrice]--;

                    // Remove the entry from the map.
                    if (priceFrequency[oldPrice] == 0)
                    {
                        priceFrequency.Remove(oldPrice);
                    }
                }

                // Add latest price for timestamp.
                timestampPriceMap[timestamp] = price;
                if (priceFrequency.ContainsKey(price))
                {
                    priceFrequency[price]++;
                }
                else
                {
                    priceFrequency[price] = 1;
                }
            }

            public int Current()
            {
                // Return latest price of the stock.
                return timestampPriceMap[latestTime];
            }

            public int Maximum()
            {
                // Return the maximum price stored at the end of sorted-map.
                return priceFrequency.Keys.Last();
            }

            public int Minimum()
            {
                // Return the maximum price stored at the front of sorted-map.
                return priceFrequency.Keys.First();
            }
        }

        /*
        Approach 2: Hashmap and Heaps
        If N is the number of records in the input stream.
        Time complexity: O(NlogN)
            1. In the update function, we add one record to the hashmap and to each heap. Adding the record to the hashmap takes constant time. However, for a heap, each push operation takes O(logN) time. So for N update calls, it will take O(NlogN) worst-case time.
            2. Each current function call takes only constant time to return the result.
            3. In the maximum and minimum functions, we pop any outdated records that are at the top of the heap. In the worst-case scenario, we might pop (N−1) elements and each pop takes O(logN) time, 
                so it might seem for one function call the time complexity is NlogN, so for N functions calls it could be N^2logN. However, when we pop a record from the heap, it's gone and won't be popped again. 
                So overall, if we push N elements into a heap, we cannot pop more than N elements, taking into account all function calls. Thus, calls to maximum and minimum will at most require O(NlogN) time.
        Space complexity: O(N)
            1. In the update function, we add and remove a record in both the hashmap and each heap. Thus each function call takes O(1) space. So for N update calls, it will take O(N) space.
            2. The current function does not use any additional space.
            3. The maximum, minimum, we only remove elements from the heap thus these do not use any additional space.
            4. Thus, in the worst-case, we will add all N records to hashmap and to both heaps, which takes O(N) space.
        */
        public class StockPriceDictHeaps
        {

            Dictionary<int, int> cache;
            PriorityQueue<(int, int), int> maxQueue;
            PriorityQueue<(int, int), int> minQueue;
            int latestTimestamp;

            public StockPriceDictHeaps()
            {
                this.cache = new Dictionary<int, int>();
                this.latestTimestamp = 0;
                this.maxQueue = new PriorityQueue<(int, int), int>();
                this.minQueue = new PriorityQueue<(int, int), int>();
            }

            public void Update(int timestamp, int price)
            {
                this.cache[timestamp] = price;
                this.latestTimestamp = Math.Max(this.latestTimestamp, timestamp);
                this.minQueue.Enqueue((timestamp, price), price);
                this.maxQueue.Enqueue((timestamp, price), -price);

                while (maxQueue.Count > 0)
                {
                    var temp = this.maxQueue.Peek();
                    var tempTime = temp.Item1;
                    var tempCost = temp.Item2;

                    if (this.cache[tempTime] == tempCost)
                    {
                        break;
                    }

                    _ = maxQueue.Dequeue();
                }

                while (minQueue.Count > 0)
                {
                    var temp = this.minQueue.Peek();
                    var tempTime = temp.Item1;
                    var tempCost = temp.Item2;

                    if (this.cache[tempTime] == tempCost)
                    {
                        break;
                    }

                    _ = minQueue.Dequeue();
                }
            }

            public int Current()
            {
                return this.cache[this.latestTimestamp];
            }

            public int Maximum()
            {
                var temp = this.maxQueue.Peek();
                return temp.Item2;
            }

            public int Minimum()
            {
                var temp = this.minQueue.Peek();
                return temp.Item2;
            }
        }

        public class StockPriceDictSortedSet
        {
            private int latestTime;
            // Store price of each stock at each timestamp.
            private Dictionary<int, int> timestampPriceMap;

            // Store stock prices in sorted order to get min and max price.
            private SortedSet<int[]> minHeap, maxHeap;

            public StockPriceDictSortedSet()
            {
                latestTime = 0;
                timestampPriceMap = new Dictionary<int, int>();
                //TODO: Replace below SortedSets with PriorityQueue
                minHeap = new SortedSet<int[]>(Comparer<int[]>.Create((a, b) => a[0] == b[0] ? a[1].CompareTo(b[1]) : a[0].CompareTo(b[0])));
                maxHeap = new SortedSet<int[]>(Comparer<int[]>.Create((a, b) => a[0] == b[0] ? b[1].CompareTo(a[1]) : b[0].CompareTo(a[0])));

                //PriorityQueue<int[], int[]> pqTest = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[0] == b[0] ? a[1].CompareTo(b[1]) : a[0].CompareTo(b[0])));
            }

            public void Update(int timestamp, int price)
            {
                // Update latestTime to latest timestamp.
                latestTime = Math.Max(latestTime, timestamp);

                // Add latest price for timestamp.
                timestampPriceMap[timestamp] = price;

                minHeap.Add(new int[] { price, timestamp });
                maxHeap.Add(new int[] { price, timestamp });
            }

            public int Current()
            {
                // Return latest price of the stock.
                return timestampPriceMap[latestTime];
            }

            public int Maximum()
            {
                int[] top = maxHeap.Max;
                // Pop pairs from heap with the price doesn't match with dictionary.
                while (timestampPriceMap[top[1]] != top[0])
                {
                    maxHeap.Remove(top);
                    top = maxHeap.Max;
                }

                return top[0];
            }

            public int Minimum()
            {
                int[] top = minHeap.Min;
                // Pop pairs from heap with the price doesn't match with dictionary.
                while (timestampPriceMap[top[1]] != top[0])
                {
                    minHeap.Remove(top);
                    top = minHeap.Min;
                }

                return top[0];
            }
        }

    }
    /**
 * Your StockPrice object will be instantiated and called as such:
 * StockPrice obj = new StockPrice();
 * obj.Update(timestamp,price);
 * int param_2 = obj.Current();
 * int param_3 = obj.Maximum();
 * int param_4 = obj.Minimum();
 */
}