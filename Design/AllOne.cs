using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    432. All O`one Data Structure
    https://leetcode.com/problems/all-oone-data-structure/description/

    •	Time complexity:
            O(1) - all functionalities
            Adding and removing nodes from the linked list here runs at O(1) as we are using the functions AddAfter, AddBefore and Remove which takes node reference as parameter and hence performs the respective operations in O(1)
    •	Space complexity:
            O(N) - as we store all the strings, mapping each string to one frequency value.
    */
    public class AllOne
    {

        //string will point to the linked list node where it is mapped to a frequency
        Dictionary<string, LinkedListNode<FreqStrings>> strMap;

        //this linked list will contains nodes in order of increasing frequency of strings
        LinkedList<FreqStrings> dll;

        public AllOne()
        {
            strMap = new Dictionary<string, LinkedListNode<FreqStrings>>();
            dll = new LinkedList<FreqStrings>();
        }

        public void Inc(string key)
        {
            if (!strMap.ContainsKey(key))
            {
                //add new node in front of linked list if it is empty or has not node with frequency of '1'
                if (dll.Count == 0 || dll.First.Value.freq != 1)
                {
                    FreqStrings node = new FreqStrings(1);
                    dll.AddFirst(node);
                }
                dll.First.Value.strs.Add(key);
                strMap.Add(key, dll.First);
            }
            else
            {
                var node = strMap[key];
                //remove the string mapping from previous frequency
                node.Value.strs.Remove(key);
                int nxtFreq = node.Value.freq + 1;
                if (node.Next == null || node.Next.Value.freq != nxtFreq)
                {
                    FreqStrings newNode = new FreqStrings(nxtFreq);
                    dll.AddAfter(node, newNode);
                }
                //map the string to new frequency
                node.Next.Value.strs.Add(key);
                strMap[key] = node.Next;

                if (node.Value.strs.Count == 0)
                    dll.Remove(node);
            }
        }

        public void Dec(string key)
        {
            var node = strMap[key];
            //remove the string mapping from previous frequency
            node.Value.strs.Remove(key);
            int prevFreq = node.Value.freq - 1;
            if (prevFreq > 0)
            {
                if (node.Previous == null || node.Previous.Value.freq != prevFreq)
                {
                    FreqStrings newNode = new FreqStrings(prevFreq);
                    dll.AddBefore(node, newNode);
                }
                //only if reduced frequency is not '0'
                //map the string to new frequency
                node.Previous.Value.strs.Add(key);
                strMap[key] = node.Previous;
            }

            if (node.Value.strs.Count == 0)
                dll.Remove(node);

            if (prevFreq == 0)
            {
                strMap.Remove(key);
            }

        }

        public string GetMaxKey()
        {
            if (dll.Count == 0)
                return "";
            //take any string from the last node (largest available frequency)
            return dll.Last.Value.strs.First();
        }

        public string GetMinKey()
        {
            if (dll.Count == 0)
                return "";

            //take any string from the first node (smallest available frequency)
            return dll.First.Value.strs.First();
        }
    }

    public class FreqStrings
    {
        public FreqStrings(int f)
        {
            freq = f;
            strs = new HashSet<string>();
        }

        public int freq;
        public HashSet<string> strs;
    }

}