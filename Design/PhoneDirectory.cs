using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    379. Design Phone Directory
    https://leetcode.com/problems/design-phone-directory/description/

    */
    public class PhoneDirectory
    {
        /*        
        Approach 1: Design using Arrays
    
        Complexity Analysis
        Let n be the maximum number of slots in the phone directory, i.e. n = maxNumbers.
        •	Time complexity:
            o	In each get method call, we iterate over the isSlotAvailable array until we reach the end or find the first available slot, one call will take O(n) time on average.
            o	In each check method call, we only check if the value stored at the respective index in the isSlotAvailable array is true or not, thus each call will take O(1) time.
            o	In each release method call, we mark the value at the respective index in the isSlotAvailable array as true, thus each call will take O(1) time.
        •	Space complexity: O(n)
            o	We use an auxiliary array isSlotAvailable of size n, to mark the availability status of n slots.

        */
        class PhoneDirectoryArray
        {
            // Array to mark if a slot is available.
            private bool[] isSlotAvailable;

            public PhoneDirectoryArray(int maxNumbers)
            {
                isSlotAvailable = new bool[maxNumbers];
                Array.Fill(isSlotAvailable, true);
            }

            public int Get()
            {
                // Traverse the 'isSlotAvailable' array to find an empty slot.
                // If found then return the respective index.
                for (int i = 0; i < isSlotAvailable.Length; ++i)
                {
                    if (isSlotAvailable[i])
                    {
                        isSlotAvailable[i] = false;
                        return i;
                    }
                }

                // Otherwise, return -1 when all slots are occupied.
                return -1;
            }

            public bool Check(int number)
            {
                // Check if the slot at index 'number' is available.
                return isSlotAvailable[number];
            }

            public void Release(int number)
            {
                // Mark the slot at index 'number' as available.
                isSlotAvailable[number] = true;
            }
        }

        /*                
        Approach 2: Design using Queue / LinkedList

    
        Complexity Analysis
        Let n be the maximum number of slots in the phone directory, i.e. n = maxNumbers.
        •	Time complexity:
            o	In each get method call, we pop the first element from slotsAvailableQueue and mark it as not available in isSlotAvailable, both of which are constant time operations, thus each call will only take O(1) time.
            o	In each check method call, we only check if the value stored at the respective index in the isSlotAvailable array is true or not, thus each call will take O(1) time.
            o	In each release method call, we mark the value at the respective index in the isSlotAvailable array as true and push it in slotsAvailableQueue both of which are constant time operations, thus each call will take O(1) time.
        •	Space complexity: O(n)
            o	We use an additional queue slotsAvailableQueue and an array isSlotAvailable, both of which have a maximum size of n.

        */

        class PhoneDirectoryQueue
        {
            // Queue to store all available slots.
            private Queue<int> slotsAvailableQueue;

            // Array to mark if a slot is available.
            private bool[] isSlotAvailable;

            public PhoneDirectoryQueue(int maxNumbers)
            {
                // Initially, all slots are available.
                isSlotAvailable = new bool[maxNumbers];
                for (int i = 0; i < maxNumbers; ++i)
                {
                    isSlotAvailable[i] = true;
                }
                slotsAvailableQueue = new Queue<int>(maxNumbers);
                for (int i = 0; i < maxNumbers; ++i)
                {
                    slotsAvailableQueue.Enqueue(i);
                }
            }

            public int Get()
            {
                // If the queue is empty it means no slot is available.
                if (slotsAvailableQueue.Count == 0)
                {
                    return -1;
                }

                // Otherwise, poll the first element from the queue,
                // mark that slot as not available and return the slot.
                int slot = slotsAvailableQueue.Dequeue();
                isSlotAvailable[slot] = false;
                return slot;
            }

            public bool Check(int number)
            {
                // Check if the slot at index 'number' is available or not.
                return isSlotAvailable[number];
            }

            public void Release(int number)
            {
                // If the slot is already present in the queue, we don't do anything.
                if (isSlotAvailable[number])
                {
                    return;
                }

                // Otherwise, mark the slot 'number' as available.
                slotsAvailableQueue.Enqueue(number);
                isSlotAvailable[number] = true;
            }
        }
        /*                
        Approach 3: Design using Hash Table /HashSet

    
        Complexity Analysis
        Let n be the maximum number of slots in the phone directory, i.e. n = maxNumbers.
        •	Time complexity:
            o	In each get method call, we return the first element from the slotsAvailable hash set, thus each call will only take O(1) time.
            o	In each check method call, we check if the value is present in the slotsAvailable hash set or not, thus each call will take O(1) time.
            o	In each release method call, we insert the value in the slotsAvailable hash set, thus each call will take O(1) time.
        •	Space complexity: O(n)
            o	We use an additional hash set slotsAvailable of maximum size n.

        */
        class PhoneDirectoryHashSet
        {
            // Hash set to store all available slots.
            private HashSet<int> slotsAvailable;

            public PhoneDirectoryHashSet(int maxNumbers)
            {
                // Initially, all slots are available.
                slotsAvailable = new HashSet<int>();
                for (int i = 0; i < maxNumbers; ++i)
                {
                    slotsAvailable.Add(i);
                }
            }

            public int Get()
            {
                // If the hash set is empty it means no slot is available.
                if (slotsAvailable.Count == 0)
                {
                    return -1;
                }

                // Otherwise, remove and return the first element from the hash set.
                int slot = slotsAvailable.First();
                slotsAvailable.Remove(slot);
                return slot;
            }

            public bool Check(int number)
            {
                // Check if the slot at index 'number' is available or not.
                return slotsAvailable.Contains(number);
            }

            public void Release(int number)
            {
                // Mark the slot 'number' as available.
                slotsAvailable.Add(number);
            }
        }



    }
    /**
 * Your PhoneDirectory object will be instantiated and called as such:
 * PhoneDirectory obj = new PhoneDirectory(maxNumbers);
 * int param_1 = obj.Get();
 * bool param_2 = obj.Check(number);
 * obj.Release(number);
 */
}