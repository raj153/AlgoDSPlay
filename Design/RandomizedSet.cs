using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    380. Insert Delete GetRandom O(1)
    https://leetcode.com/problems/insert-delete-getrandom-o1/description/

    Time complexity. GetRandom is always O(1). Insert and Delete both have O(1) average time complexity, 
                     and O(N) in the worst-case scenario when the operation exceeds the capacity of currently allocated array/hashmap and invokes space reallocation.
    Space complexity: O(N), to store N elements.


    */
    public class RandomizedSet
    {
        private Dictionary<int, int> elementIndexMap;
        private List<int> elementsList;
        private Random randomGenerator;

        /** Initialize your data structure here. */
        public RandomizedSet()
        {
            elementIndexMap = new Dictionary<int, int>();
            elementsList = new List<int>();
            randomGenerator = new Random();
        }

        /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
        public bool Insert(int value)
        {
            if (elementIndexMap.ContainsKey(value)) return false;

            elementIndexMap[value] = elementsList.Count;
            elementsList.Add(value);
            return true;
        }

        /** Removes a value from the set. Returns true if the set contained the specified element. */
        public bool Remove(int value)
        {
            if (!elementIndexMap.ContainsKey(value)) return false;

            // move the last element to the place idx of the element to delete
            int lastElement = elementsList[elementsList.Count - 1];
            int index = elementIndexMap[value];
            elementsList[index] = lastElement;
            elementIndexMap[lastElement] = index;
            // delete the last element
            elementsList.RemoveAt(elementsList.Count - 1);
            elementIndexMap.Remove(value);
            return true;
        }

        /** Get a random element from the set. */
        public int GetRandom()
        {
            return elementsList[randomGenerator.Next(elementsList.Count)];
        }
    }
}