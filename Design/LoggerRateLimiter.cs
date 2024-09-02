using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    359. Logger Rate Limiter
    https://leetcode.com/problems/logger-rate-limiter/description/

    Approach 1: Queue + Set
    Time Complexity: O(N) where N is the size of the queue. In the worst case, all the messages in the queue become obsolete. As a result, we need clean them up.
    Space Complexity: O(N) where N is the size of the queue. We keep the incoming messages in both the queue and set. The upper bound of the required space would be 2N, if we have no duplicate at all.

    */
    public class LoggerRateLimiter
    {
        private LinkedList<(string, int)> msgQueue;
        private HashSet<string> msgSet;

        /** Initialize your data structure here. */
        public LoggerRateLimiter()
        {
            msgQueue = new LinkedList<(string, int)>();
            msgSet = new HashSet<string>();
        }

        /**
         * Returns true if the message should be printed in the given timestamp, otherwise returns false.
         */
        public bool ShouldPrintMessage(int timestamp, string message)
        {
            // clean up.
            while (msgQueue.Count > 0)
            {
                var head = msgQueue.First.Value;
                if (timestamp - head.Item2 >= 10)
                {
                    msgQueue.RemoveFirst();
                    msgSet.Remove(head.Item1);
                }
                else
                {
                    break;
                }
            }

            if (!msgSet.Contains(message))
            {
                var newEntry = (message, timestamp);
                msgQueue.AddLast(newEntry);
                msgSet.Add(message);
                return true;
            }
            else
            {
                return false;
            }
        }

    }
    /*
    Approach 2: Hashtable / Dictionary
    Time Complexity: O(1). The lookup and update of the hashtable takes a constant time.
    Space Complexity: O(M) where M is the size of all incoming messages. Over the time, the hashtable would have an entry for each unique message that has appeared.

    */
    public class LoggerRateLimiterOptimal
    {
        private Dictionary<string, int> msgDict;
        
        /** Initialize your data structure here. */
        public LoggerRateLimiterOptimal()
        {
            msgDict = new Dictionary<string, int>();
        }

        /**
         * Returns true if the message should be printed in the given timestamp, otherwise returns false.
         */
        public bool ShouldPrintMessage(int timestamp, string message)
        {
            
            if (!this.msgDict.ContainsKey(message))
            {
                this.msgDict[message] = timestamp;
                return true;
            }
            int oldTimeStamp = this.msgDict[message];
            if(timestamp - oldTimeStamp >=10){
                this.msgDict[message] = timestamp;
                return true;
            }else return false;
            
        }

    }
}