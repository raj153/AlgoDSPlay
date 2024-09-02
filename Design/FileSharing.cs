using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1500. Design a File Sharing System
    https://leetcode.com/problems/design-a-file-sharing-system/description/

    */
    public class FileSharing
    {
        private Dictionary<int, HashSet<int>> userChunksMap = new Dictionary<int, HashSet<int>>();
        private PriorityQueue<int,int> availableUserIds = new PriorityQueue<int,int>();

        public FileSharing(int m)
        {
            availableUserIds.Enqueue(1,1);
        }

        public int Join(List<int> ownedChunks)
        {
            int userId = availableUserIds.Dequeue();
            if (availableUserIds.Count == 0)
                availableUserIds.Enqueue(userId + 1, userId + 1);
            userChunksMap[userId] = new HashSet<int>(ownedChunks);
            return userId;
        }

        public void Leave(int userId)
        {
            availableUserIds.Enqueue(userId, userId);
            userChunksMap.Remove(userId);
        }

        public List<int> Request(int userId, int chunkId)
        {
            List<int> result = new List<int>();
            foreach (var id in userChunksMap.Keys)
            {
                if (userChunksMap[id].Contains(chunkId))
                    result.Add(id);
            }
            if (result.Count > 0) userChunksMap[userId].Add(chunkId);
            result.Sort();
            return result;
        }
    }

}
