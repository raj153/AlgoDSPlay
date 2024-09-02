using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    895. Maximum Frequency Stack
    https://leetcode.com/problems/maximum-frequency-stack/description/	

    Complexity Analysis
    •	Time Complexity: O(1) for both push and pop operations.
    •	Space Complexity: O(N), where N is the number of elements in the FreqStack.


    */
    public class FreqStack
    {
        Dictionary<int, int> frequencyMap;
        Dictionary<int, Stack<int>> groupMap;
        int maxfreq;

        public FreqStack()
        {
            frequencyMap = new Dictionary<int, int>();
            groupMap = new();
            maxfreq = 0;
        }

        public void Push(int val)
        {
            int frequency = frequencyMap.GetValueOrDefault(val) + 1;
            frequencyMap[val] = frequency;
            if (frequency > maxfreq)
                maxfreq = frequency;

            if (!groupMap.ContainsKey(frequency))
            {
                groupMap[frequency] = new Stack<int>();
            }
            groupMap[frequency].Push(val);
        }

        public int Pop()
        {
            int value = groupMap[maxfreq].Pop();
            frequencyMap[value]--;

            if (groupMap[maxfreq].Count == 0)
                maxfreq--;

            return value;
        }

    }
}