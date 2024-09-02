using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    355. Design Twitter
    https://leetcode.com/problems/design-twitter/description/

    */
    public class Twitter
    {
        private static int timeStamp = 0;

        // Easy to find if user exists
        private Dictionary<int, User> userMap;

        // Tweet linked to the next Tweet so that we can save a lot of time
        // when we execute getNewsFeed(userId)
        public class Tweet
        {
            public int Id;
            public int Time;
            public Tweet Next;

            public Tweet(int id)
            {
                Id = id;
                Time = timeStamp++;
                Next = null;
            }
        }

        // OO design so User can follow, unfollow and post itself
        public class User
        {
            public int Id;
            public HashSet<int> Followed;
            public Tweet TweetHead;

            public User(int id)
            {
                Id = id;
                Followed = new HashSet<int>();
                Follow(id); // first follow itself
                TweetHead = null;
            }

            public void Follow(int id)
            {
                Followed.Add(id);
            }

            public void Unfollow(int id)
            {
                Followed.Remove(id);
            }

            // Every time user posts a new tweet, add it to the head of the tweet list.
            public void Post(int id)
            {
                Tweet tweet = new Tweet(id);
                tweet.Next = TweetHead;
                TweetHead = tweet;
            }
        }

        /** Initialize your data structure here. */
        public Twitter()
        {
            userMap = new Dictionary<int, User>();
        }

        /** Compose a new tweet. */
        public void PostTweet(int userId, int tweetId)
        {
            if (!userMap.ContainsKey(userId))
            {
                User user = new User(userId);
                userMap[userId] = user;
            }
            userMap[userId].Post(tweetId);
        }

        // Best part of this.
        // First get all tweets lists from one user including itself and all people it followed.
        // Second add all heads into a max heap. Every time we poll a tweet with 
        // largest time stamp from the heap, then we add its next tweet into the heap.
        // So after adding all heads we only need to add 9 tweets at most into this 
        // heap before we get the 10 most recent tweets.
        public List<int> GetNewsFeed(int userId)
        {
            List<int> result = new List<int>();

            if (!userMap.ContainsKey(userId)) return result;

            HashSet<int> users = userMap[userId].Followed;
            SortedSet<Tweet> tweetHeap = new SortedSet<Tweet>(Comparer<Tweet>.Create((a, b) => b.Time.CompareTo(a.Time)));
            //Replace above with below PQ as a heap
            PriorityQueue<Tweet, Tweet> heap = new PriorityQueue<Tweet, Tweet>(Comparer<Tweet>.Create((t1, t2)=>t1.Time.CompareTo(t2.Time)));
            foreach (int user in users)
            {
                Tweet tweet = userMap[user].TweetHead;
                // Very important! If we add null to the head we are screwed.
                if (tweet != null)
                {
                    tweetHeap.Add(tweet);
                }
            }
            int n = 0;
            while (tweetHeap.Count > 0 && n < 10)
            {
                Tweet tweet = tweetHeap.Min;
                tweetHeap.Remove(tweet);
                result.Add(tweet.Id);
                n++;
                if (tweet.Next != null)
                    tweetHeap.Add(tweet.Next);
            }

            return result;
        }

        /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
        public void Follow(int followerId, int followeeId)
        {
            if (!userMap.ContainsKey(followerId))
            {
                User user = new User(followerId);
                userMap[followerId] = user;
            }
            if (!userMap.ContainsKey(followeeId))
            {
                User user = new User(followeeId);
                userMap[followeeId] = user;
            }
            userMap[followerId].Follow(followeeId);
        }

        /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
        public void Unfollow(int followerId, int followeeId)
        {
            if (!userMap.ContainsKey(followerId) || followerId == followeeId)
                return;
            userMap[followerId].Unfollow(followeeId);
        }
    }

    /**
     * Your Twitter object will be instantiated and called as such:
     * Twitter obj = new Twitter();
     * obj.PostTweet(userId, tweetId);
     * List<int> param_2 = obj.GetNewsFeed(userId);
     * obj.Follow(followerId, followeeId);
     * obj.Unfollow(followerId, followeeId);
     */
}