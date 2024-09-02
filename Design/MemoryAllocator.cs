using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2502. Design Memory Allocator
    https://leetcode.com/problems/design-memory-allocator/description/

    */
    public class MemoryAllocator
    {
        public class Allocator
        {
            private Dictionary<int, SortedDictionary<int, int>> memory = new Dictionary<int, SortedDictionary<int, int>>();

            public Allocator(int n)
            {
                var ranges = new SortedDictionary<int, int>(); // <startAdd, endAdd>
                ranges.Add(0, n - 1); // address start from 0
                memory.Add(0, ranges); // 0 means free block
            }

            public int Allocate(int size, int mID)
            {
                int addr = -1;
                int[] availableRange = new int[] { -1, -1 };
                foreach (var range in memory[0])
                {
                    int startAdd = range.Key;
                    int endAdd = range.Value;
                    if (endAdd - startAdd + 1 >= size)
                    {
                        addr = startAdd;
                        availableRange[0] = startAdd;
                        availableRange[1] = endAdd;
                        break;
                    }
                }
                if (addr != -1)
                {
                    if (!memory.ContainsKey(mID))
                    {
                        memory[mID] = new SortedDictionary<int, int>();
                    }
                    memory[mID].Add(availableRange[0], availableRange[0] + size - 1);
                    memory[0].Remove(availableRange[0]);
                    if (availableRange[1] - availableRange[0] + 1 > size)
                    {
                        memory[0].Add(availableRange[0] + size, availableRange[1]);
                    }
                }

                MergeRanges(mID);

                return addr;
            }

            public int Free(int mID)
            {
                int cnt = 0;
                if (memory.TryGetValue(mID, out var freeRanges))
                {
                    foreach (var range in freeRanges)
                    {
                        int startAdd = range.Key;
                        int endAdd = range.Value;
                        cnt += endAdd - startAdd + 1;
                        memory[0].Add(startAdd, endAdd);
                    }
                }

                memory.Remove(mID);
                MergeRanges(0);
                return cnt;
            }

            private void MergeRanges(int mID)
            {
                if (!memory.TryGetValue(mID, out var curRanges)) return;

                var mergedRanges = new SortedDictionary<int, int>();
                int[] lastRange = new int[] { int.MinValue, int.MinValue };

                foreach (var range in curRanges)
                {
                    int startAdd = range.Key;
                    int endAdd = range.Value;
                    if (startAdd - 1 == lastRange[1])
                    {
                        lastRange[1] = endAdd;
                    }
                    else
                    {
                        if (lastRange[0] != int.MinValue)
                        {
                            mergedRanges.Add(lastRange[0], lastRange[1]);
                        }
                        lastRange[0] = startAdd;
                        lastRange[1] = endAdd;
                    }
                }
                if (lastRange[0] != int.MinValue)
                {
                    mergedRanges.Add(lastRange[0], lastRange[1]);
                }

                memory[mID] = mergedRanges;
            }
        }

    }
    /**
 * Your Allocator object will be instantiated and called as such:
 * Allocator obj = new Allocator(n);
 * int param_1 = obj.Allocate(size,mID);
 * int param_2 = obj.Free(mID);
 */
}