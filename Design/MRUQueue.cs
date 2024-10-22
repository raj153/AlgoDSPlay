using System;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay.Design
{/* 
    1756. Design Most Recently Used Queue
https://leetcode.com/problems/design-most-recently-used-queue/description/ 
https://algo.monster/liteproblems/1756 moves the 

Time and Space Complexity
Time Complexity
The Binary Indexed Tree (BIT) provides efficient methods for updating an element and querying the prefix sum. Both operations (update and query) have a time complexity of O(log n).
•	The update(x, v) function is executed at most O(log n) times for each update, since every time it updates the BIT, it jumps to the next value by adding the least significant bit (LSB).
•	The query(x) function also has a time complexity of O(log n) since it sums up the value of the elements by subtracting the LSB until it reaches zero.
The MRUQueue uses the BIT to manage the fetching:
•	The fetch(k) operation has a binary search which has a time complexity of O(log n) (where n is the current length of the queue), combined with querying the BIT O(log n), yielding a total of O((log n)^2). This is because every iteration of the binary search invokes a single tree.query(mid) call, and there are O(log n) iterations overall.
•	Every fetch(k) operation also appends an element to the queue and updates the BIT, both of which have a time complexity of O(log n).
Consequently, a single fetch(k) operation has an overall time complexity of O((log n)^2 + log n), which simplifies to O((log n)^2).
Space Complexity
•	The space complexity of the BinaryIndexedTree class is O(n), due to the array self.c which has a size n + 1.
•	The MRUQueue class maintains a queue self.q and an instance of BinaryIndexedTree. The queue would have a space complexity of O(n), where n is the number of fetch operations since every fetch would add an element to the queue.
•	The BinaryIndexedTree within MRUQueue initially has a size of n + 2010 for the array self.c, so the space complexity is O(n), assuming that the 2010 constant does not majorly impact the space complexity for large n.
The MRUQueue space complexity is dominated by the size of the queue and the BIT, so the overall space complexity is O(n), considering that n refers to the number of elements in the queue plus the modifications during the fetch operations.

 */
    public class MRUQueue
    {
        private int currentSize; // Current size of the MRUQueue
        private int[] queue; // Array to hold the values of the MRUQueue
        private BinaryIndexedTree binaryIndexedTree; // Instance of BIT to support operations

        // Constructor
        public MRUQueue(int size)
        {
            this.currentSize = size;
            this.queue = new int[size + 2010]; // Initialize with extra space for modifications
            for (int i = 1; i <= size; ++i)
            {
                queue[i] = i;
            }
            // Create a BIT with extra space for modifications
            binaryIndexedTree = new BinaryIndexedTree(size + 2010);
        }

        // Fetches the k-th element and moves it to the end
        public int Fetch(int k)
        {
            int left = 1, right = currentSize;
            while (left < right)
            {
                int mid = (left + right) >> 1; // Find the midpoint
                                               // Modify the condition to find the k-th unaffected position
                if (mid - binaryIndexedTree.Query(mid) >= k)
                {
                    right = mid;
                }
                else
                {
                    left = mid + 1;
                }
            }
            // Retrieve and update the queue with the fetched element
            int value = queue[left];
            queue[++currentSize] = value; // Add the fetched value to the end
            binaryIndexedTree.Update(left, 1); // Mark the original position as affected
            return value; // Return the fetched value
        }
    }

    /**
     * The usage of MRUQueue would be as follows:
     * MRUQueue mruQueue = new MRUQueue(n);
     * int element = mruQueue.Fetch(k);
     */




}
