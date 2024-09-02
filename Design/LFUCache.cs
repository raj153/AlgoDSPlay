using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    460. LFU Cache	
    https://leetcode.com/problems/lfu-cache/description/

    Here, N is the total number of operations.
    •	Time complexity: O(1), as required by the question.Since we only have basic HashMap/(Linked)HashSet operations. For details,
                Our utility function insert puts the key- value pair into the cache, queries and possibly puts an empty LinedHashSet in the frequencies, then queries frequencies again and adds a key into the associated value which is a LinkedHashSet. All the operations are based on the hash calculating for simple type (int or Integer) and the time complexity is constant.
                For each get operation, in the worst case, we query the frequencies and remove a key from the associated value which is a LinkedHashSet and call insert function once. All the operations have the constant time complexity based on the hash calculating for simple type.
                For each put operation, in the simple case we just insert the new key-value pair into the cache and call get function once. In the worst case, we query the frequencies to get the associated value, namely all the keys with the same frequencies which is a LinkedHashSet. And then we get the first key from the LinkedHashSet, remove it from both cache and frequencies. All the operations have the constant time complexity based on the hash calculating for simple type.
    •	Space complexity: O(N). We save all the key-value pairs as well as all the keys with frequencies in the 2 HashMaps (plus a LinkedHashSet), so there are at most $min(N, capacity) keys and values at any given time.
    */
    public class LFUCache
    {
        // key: original key, value: frequency and original value.
        private Dictionary<int, (int frequency, int value)> cache;
        // key: frequency, value: All keys that have the same frequency.
        private Dictionary<int, HashSet<int>> frequencies;
        private int minf;
        private int capacity;

        private void Insert(int key, int frequency, int value)
        {
            cache[key] = (frequency, value);
            if (!frequencies.ContainsKey(frequency))
            {
                frequencies[frequency] = new HashSet<int>();
            }
            frequencies[frequency].Add(key);
        }

        public LFUCache(int capacity)
        {
            cache = new Dictionary<int, (int frequency, int value)>();
            frequencies = new Dictionary<int, HashSet<int>>();
            minf = 0;
            this.capacity = capacity;
        }

        public int Get(int key)
        {
            if (!cache.TryGetValue(key, out var frequencyAndValue))
            {
                return -1;
            }
            int frequency = frequencyAndValue.frequency;
            var keys = frequencies[frequency];
            keys.Remove(key);
            if (keys.Count == 0)
            {
                frequencies.Remove(frequency);
                if (minf == frequency)
                {
                    ++minf;
                }
            }
            int value = frequencyAndValue.value;
            Insert(key, frequency + 1, value);
            return value;
        }

        public void Put(int key, int value)
        {
            if (capacity <= 0)
            {
                return;
            }
            if (cache.TryGetValue(key, out var frequencyAndValue))
            {
                cache[key] = (frequencyAndValue.frequency, value);
                Get(key);
                return;
            }
            if (capacity == cache.Count)
            {
                var keys = frequencies[minf];
                int keyToDelete = keys.First();
                cache.Remove(keyToDelete);
                keys.Remove(keyToDelete);
                if (keys.Count == 0)
                {
                    frequencies.Remove(minf);
                }
            }
            minf = 1;
            Insert(key, 1, value);
        }

    }
}