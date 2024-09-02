using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    706. Design HashMap
    https://leetcode.com/problems/design-hashmap/description/
    
    */

    public class HashMap
    {
        /*
        Approach 1: Using Array

        Complexity
        Time complexity:    O(1) - as we perform all operations in constant time

        Space complexity:   O(1) - as we use constant amount of memory (1000001) for all inputs

        This will be bad design for smaller inputs
        */
        public class HashMapArr
        {
            int[] map;

            public HashMapArr()
            {
                //0 is treated as no key present
                //hence while adding/storing the value, we'll add 1 to it
                //and remove 1 from the value before returning it
                map = new int[1000001];
            }

            public void Put(int key, int value)
            {
                map[key] = value + 1;
            }

            public int Get(int key)
            {
                return map[key] - 1;
            }

            public void Remove(int key)
            {
                map[key] = 0;
            }

        }

        /*
        Approach 2: Modulo + Array

        Complexity
        Time complexity:    O(1) - as we perform all operations in constant time

        Space complexity:   O(1) - as we use constant amount of memory (1000001) for all inputs

        This will be bad design for smaller inputs
        */
        class HashMapList
        {
            private int keySpace;
            private List<Bucket> hashTable;

            public HashMapList()
            {
                keySpace = 2069;
                hashTable = new List<Bucket>();
                for (int i = 0; i < keySpace; ++i)
                {
                    hashTable.Add(new Bucket());
                }
            }

            public void Put(int key, int value)
            {
                int hashKey = key % keySpace;
                hashTable[hashKey].Update(key, value);
            }

            public int Get(int key)
            {
                int hashKey = key % keySpace;
                return hashTable[hashKey].Get(key);
            }

            public void Remove(int key)
            {
                int hashKey = key % keySpace;
                hashTable[hashKey].Remove(key);
            }
        }

        class Pair<U, V>
        {
            public U First { get; set; }
            public V Second { get; set; }

            public Pair(U first, V second)
            {
                First = first;
                Second = second;
            }
        }

        class Bucket
        {
            private List<Pair<int, int>> bucket;

            public Bucket()
            {
                bucket = new List<Pair<int, int>>();
            }

            public int Get(int key)
            {
                foreach (var pair in bucket)
                {
                    if (pair.First.Equals(key))
                        return pair.Second;
                }
                return -1;
            }

            public void Update(int key, int value)
            {
                bool found = false;
                foreach (var pair in bucket)
                {
                    if (pair.First.Equals(key))
                    {
                        pair.Second = value;
                        found = true;
                    }
                }
                if (!found)
                    bucket.Add(new Pair<int, int>(key, value));
            }

            public void Remove(int key)
            {
                for (int i = 0; i < bucket.Count; i++)
                {
                    if (bucket[i].First.Equals(key))
                    {
                        bucket.RemoveAt(i);
                        break;
                    }
                }
            }
        }



        /**
     * Your MyHashMap object will be instantiated and called as such:
     * MyHashMap obj = new MyHashMap();
     * obj.Put(key,value);
     * int param_2 = obj.Get(key);
     * obj.Remove(key);
     */
    }
}