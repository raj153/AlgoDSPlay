using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    635. Design Log Storage System
    https://leetcode.com/problems/design-log-storage-system/description

    Approach #1 Converting timestamp into a number
    */
    public class LogSystem
    {
        private List<long[]> logEntries;

        public LogSystem()
        {
            logEntries = new List<long[]>();
        }

        //The putmethod takes O(1) time to insert a new entry into the given set of logs.

        public void Put(int id, string timestamp)
        {
            int[] timeComponents = Array.ConvertAll(timestamp.Split(':'), int.Parse);
            logEntries.Add(new long[] { Convert(timeComponents), id });
        }

        public long Convert(int[] timeComponents)
        {
            timeComponents[1] = timeComponents[1] - (timeComponents[1] == 0 ? 0 : 1);
            timeComponents[2] = timeComponents[2] - (timeComponents[2] == 0 ? 0 : 1);
            return (timeComponents[0] - 1999) * (31 * 12) * 24 * 60 * 60 + timeComponents[1] * 31 * 24 * 60 * 60 + timeComponents[2] * 24 * 60 * 60 + timeComponents[3] * 60 * 60 + timeComponents[4] * 60 + timeComponents[5];
        }

        /*
        The retrieve method takes O(n) time to retrieve the logs in the required range. 
        Determining the granularity takes O(1) time. But, to find the logs lying in the required range, 
        we need to iterate over the set of logs atleast once. Here, n refers to the number of entries in the current set of logs.
        */
        public List<int> Retrieve(string startTime, string endTime, string granularity)
        {
            List<int> result = new List<int>();
            long start = Granularity(startTime, granularity, false);
            long end = Granularity(endTime, granularity, true);
            for (int i = 0; i < logEntries.Count; i++)
            {
                if (logEntries[i][0] >= start && logEntries[i][0] < end)
                    result.Add((int)logEntries[i][1]);
            }
            return result;
        }

        public long Granularity(string time, string granularity, bool isEnd)
        {
            Dictionary<string, int> granularityMap = new Dictionary<string, int>
        {
            { "Year", 0 },
            { "Month", 1 },
            { "Day", 2 },
            { "Hour", 3 },
            { "Minute", 4 },
            { "Second", 5 }
        };

            string[] timeComponents = new string[] { "1999", "00", "00", "00", "00", "00" };
            string[] inputComponents = time.Split(':');
            for (int i = 0; i <= granularityMap[granularity]; i++)
            {
                timeComponents[i] = inputComponents[i];
            }
            int[] parsedTimeComponents = Array.ConvertAll(timeComponents, int.Parse);
            if (isEnd)
                parsedTimeComponents[granularityMap[granularity]]++;
            return Convert(parsedTimeComponents);
        }

    }
    /* 
    Approach #2 Better Retrieval
    This method remains almost the same as the last approach, except that, in order to store the timestamp data, we make use
    of a TreeMap instead of a list, with the key values being the timestamps converted in seconds form and the values being the
    ids of the corresponding logs.
    */
    public class LogSystemOptimal
    {
        private SortedDictionary<long, int> map;

        public LogSystemOptimal()
        {
            map = new SortedDictionary<long, int>();

        }

        /*
        In Java, The TreeMap is maintained internally as a Red-Black(balanced) tree. Thus, the putmethod takes O(log(n)) time to insert a new entry into the given set of logs. 
        Here, n refers to the number of entries currently present in the given set of logs.
        In C# herein, TreeMap replae wth SortedDirctionary.
        */

        public void Put(int id, string timestamp)
        {
            int[] st = timestamp.Split(':').Select(int.Parse).ToArray();
            map[Convert(st)] = id;
        }

        public long Convert(int[] timeComponents)
        {
            timeComponents[1] = timeComponents[1] - (timeComponents[1] == 0 ? 0 : 1);
            timeComponents[2] = timeComponents[2] - (timeComponents[2] == 0 ? 0 : 1);
            return (timeComponents[0] - 1999) * (31 * 12) * 24 * 60 * 60 + timeComponents[1] * 31 * 24 * 60 * 60 + timeComponents[2] * 24 * 60 * 60 + timeComponents[3] * 60 * 60 + timeComponents[4] * 60 + timeComponents[5];
        }

        /*
        The retrieve method takes O(m start ) time to retrieve the logs in the required range. Determining the granularity takes O(1) time. To find the logs in the required range, we only need to iterate over those elements which already lie in the required range. Here, m 
        start  refers to the number of entries in the current set of logs which have a timestamp greater than the current start value.
        */
        public List<int> Retrieve(string startTime, string endTime, string gra)
        {
            List<int> res = new List<int>();
            long start = Granularity(startTime, gra, false);
            long end = Granularity(endTime, gra, true);
            foreach (var key in map.Keys.Where(key => key >= start && key < end))
            {
                res.Add(map[key]);
            }
            return res;

        }

        public long Granularity(string time, string granularity, bool isEnd)
        {
            Dictionary<string, int> granularityMap = new Dictionary<string, int>
        {
            { "Year", 0 },
            { "Month", 1 },
            { "Day", 2 },
            { "Hour", 3 },
            { "Minute", 4 },
            { "Second", 5 }
        };

            string[] timeComponents = new string[] { "1999", "00", "00", "00", "00", "00" };
            string[] inputComponents = time.Split(':');
            for (int i = 0; i <= granularityMap[granularity]; i++)
            {
                timeComponents[i] = inputComponents[i];
            }
            int[] parsedTimeComponents = Array.ConvertAll(timeComponents, int.Parse);
            if (isEnd)
                parsedTimeComponents[granularityMap[granularity]]++;
            return Convert(parsedTimeComponents);
        }

    }
}
