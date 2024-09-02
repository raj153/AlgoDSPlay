using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    981. Time Based Key-Value Store
    https://leetcode.com/problems/time-based-key-value-store/description/

    */
    public class TimeMap
    {
        /*
        Approach 1: Hashmap + Linear Search

        If M is the number of set function calls, N is the number of get function calls, and L is average length of key and value strings.
        Time complexity:
            In the set() function, in each call, we store a value at (key, timestamp) location, which takes O(L) time to hash the string.
                                    Thus, for M calls overall it will take, O(M⋅L) time.
            In the get() function, in each call, we iterate linearly from timestamp to 1 which takes O(timestamp) time and again to hash the string it takes O(L) time.
                                    Thus, for N calls overall it will take, O(N⋅timestamp⋅L) time.
                                    Note: This approach can be TLE, since the time complexity is not optimal given the current data range in the problem description.
        Space complexity:

            In the set() function, in each call we store one value string of length L, which takes O(L) space.
                                    Thus, for M calls we may store M unique values, so overall it may take O(M⋅L) space.

            In the get() function, we are not using any additional space. Thus, for all N calls it is a constant space operation.
        */
        public class TimeMapDictLS
        {
            Dictionary<string, Dictionary<int, string>> keyTimeMap;

            public TimeMapDictLS()
            {
                keyTimeMap = new Dictionary<string, Dictionary<int, string>>();
            }

            public void Set(string key, string value, int timestamp)
            {
                if (!keyTimeMap.ContainsKey(key))
                {
                    keyTimeMap[key] = new Dictionary<int, string>();
                }
                // Store '(timestamp, value)' pair in 'key' bucket.

                keyTimeMap[key][timestamp] = value;
            }

            public string Get(string key, int timestamp)
            {
                // If the 'key' does not exist in map we will return empty string.

                if (!keyTimeMap.ContainsKey(key))
                {
                    return "";
                }
                // Iterate on time from 'timestamp' to '1'.                
                for (int currTime = timestamp; currTime >= 1; --currTime)
                {
                    // If a value for current time is stored in key's bucket we return the value.
                    if (keyTimeMap[key].ContainsKey(currTime))
                    {
                        return keyTimeMap[key][currTime];
                    }
                }

                return "";
            }
        }

        /*
        Approach 2: Sorted Map + Binary Search

        If M is the number of set function calls, N is the number of get function calls, and L is average length of key and value strings.
        Time complexity:
            In the set() function, in each call we store a value at (key, timestamp) location, which takes O(L⋅logM) time as the internal implementation of sorted maps is some kind of balanced binary tree and in worst case we might have to compare logM nodes (height of tree) of length L each with our key.
                            Thus, for M calls overall it will take, O(L⋅M⋅logM) time.

            In the get() function, we will find correct key in our map, which can take O(L⋅logM) time and then use binary search on that bucket which can have at most M elements, which takes O(logM) time.
                            peekitem in python will also take O(logN) time to get the value, but the upper bound remains the same.
                            Thus, for N calls overall it will take, O(N⋅(L⋅logM+logM)) time.

        Space complexity:

            In the set() function, in each call we store one value string of length L, which takes O(L) space.
                            Thus, for M calls we may store M unique values, so overall it may take O(M⋅L) space.

            In the get() function, we are not using any additional space.
                            Thus, for all N calls it is a constant space operation.

        */
        public class TimeMapSortedDictBS
        {
            private Dictionary<string, SortedDictionary<int, string>> keyTimeMap;

            public TimeMapSortedDictBS()
            {
                keyTimeMap = new Dictionary<string, SortedDictionary<int, string>>();
            }

            public void Set(string key, string value, int timestamp)
            {
                if (!keyTimeMap.ContainsKey(key))
                {
                    keyTimeMap[key] = new SortedDictionary<int, string>();
                }

                // Store '(timestamp, value)' pair in 'key' bucket.
                keyTimeMap[key][timestamp] = value;
            }

            public string Get(string key, int timestamp)
            {
                // If the 'key' does not exist in map we will return empty string.
                if (!keyTimeMap.ContainsKey(key))
                {
                    return "";
                }

                int? floorKey = null;
                //TODO: Test below
                foreach (var k in keyTimeMap[key].Keys)
                {
                    if (k <= timestamp)
                    {
                        floorKey = k;
                    }
                    else
                    {
                        break;
                    }
                }

                // Return searched time's value, if exists.
                if (floorKey.HasValue)
                {
                    return keyTimeMap[key][floorKey.Value];
                }

                return "";
            }
        }

        /*
          Approach 3: Array + Binary Search

          If M is the number of set function calls, N is the number of get function calls, and L is average length of key and value strings.
          Time complexity:
              In the set() function, in each call, we push a (timestamp, value) pair in the key bucket, which takes O(L) time to hash the string.
                              Thus, for M calls overall it will take, O(M⋅L) time.

              In the get() function, we use binary search on the key's bucket which can have at most M elements and to hash the string it takes O(L) time, thus overall it will take O(L⋅logM) time for binary search.
                               And, for N calls overall it will take, O(N⋅L⋅logM) time.


          Space complexity:

              In the set() function, in each call we store one value string of length L, which takes O(L) space.
                              Thus, for M calls we may store M unique values, so overall it may take O(M⋅L) space.

              In the get() function, we are not using any additional space.
                              Thus, for all N calls it is a constant space operation.

          */
        public class TimeMapArrayBS
        {
            private Dictionary<string, List<KeyValuePair<int, string>>> keyTimeMap;

            public TimeMapArrayBS()
            {
                keyTimeMap = new Dictionary<string, List<KeyValuePair<int, string>>>();
            }

            public void Set(string key, string value, int timestamp)
            {
                if (!keyTimeMap.ContainsKey(key))
                {
                    keyTimeMap[key] = new List<KeyValuePair<int, string>>();
                }

                // Store '(timestamp, value)' pair in 'key' bucket.
                keyTimeMap[key].Add(new KeyValuePair<int, string>(timestamp, value));
            }

            public string Get(string key, int timestamp)
            {
                // If the 'key' does not exist in map we will return empty string.
                if (!keyTimeMap.ContainsKey(key))
                {
                    return string.Empty;
                }

                if (timestamp < keyTimeMap[key][0].Key)
                {
                    return string.Empty;
                }

                // Using binary search on the list of pairs.
                int left = 0;
                int right = keyTimeMap[key].Count;

                while (left < right)
                {
                    int mid = (left + right) / 2;
                    if (keyTimeMap[key][mid].Key <= timestamp)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }

                // If iterator points to first element it means, no time <= timestamp exists.
                if (right == 0)
                {
                    return string.Empty;
                }

                return keyTimeMap[key][right - 1].Value;
            }
        }
    }
}