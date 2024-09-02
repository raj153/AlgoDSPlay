using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1244. Design A Leaderboard
    https://leetcode.com/problems/design-a-leaderboard/description/

    */
    public class Leaderboard
    {
        /*
        Approach 1: Brute Force.

        Complexity Analysis
        •	Time Complexity:
            o	O(1) for addScore.
            o	O(1) for reset.
            o	O(NlogN) for top where N represents the total number of players since we sort all of the player scores and then take the top K from the sorted list.
        •	Space Complexity:
            o	O(N) used by the scores dictionary and also by the new list formed using the dictionary values in the top function.

        */
        class LeaderboardNaive
        {
            private Dictionary<int, int> playerScores;

            public LeaderboardNaive()
            {
                // Since this is a single threaded application and we don't need synchronized access, a 
                // Dictionary is a good choice of data structure as compared to a Hashtable.
                this.playerScores = new Dictionary<int, int>();
            }

            public void AddScore(int playerId, int score)
            {
                if (!this.playerScores.ContainsKey(playerId))
                {
                    this.playerScores[playerId] = 0;
                }

                this.playerScores[playerId] += score;
            }

            public int Top(int k)
            {
                List<int> scoreValues = new List<int>(this.playerScores.Values);
                scoreValues.Sort((a, b) => b.CompareTo(a)); // Sorting in descending order

                int totalScore = 0;
                for (int i = 0; i < k; i++)
                {
                    totalScore += scoreValues[i];
                }

                return totalScore;
            }

            public void Reset(int playerId)
            {
                this.playerScores[playerId] = 0;
            }
        }

        /*
       Approach 2: Heap for top-K.

       Complexity Analysis
       •	Time Complexity:
           o	O(1) for addScore.
           o	O(1) for reset.
           o	O(K)+O(NlogK) = O(NlogK). It takes O(K) to construct the initial heap and then for the rest of the N−K elements, we perform the extractMin and add operations on the heap each of which take (logK) time.
       •	Space Complexity:
           o	O(N+K) where O(N) is used by the scores dictionary and O(K) is used by the heap.


       */
        public class LeaderboardHeaps
        {
            private Dictionary<int, int> playerScores;

            public LeaderboardHeaps()
            {
                this.playerScores = new Dictionary<int, int>();
            }

            public void AddScore(int playerId, int score)
            {
                if (!this.playerScores.ContainsKey(playerId))
                {
                    this.playerScores[playerId] = 0;
                }

                this.playerScores[playerId] += score;
            }

            public int Top(int K)
            {
                // A min-heap in C# containing entries of a dictionary. Note that we have to provide
                // a comparison function to ensure we get the ordering right of these objects.
                var minHeap = new SortedSet<KeyValuePair<int, int>>(Comparer<KeyValuePair<int, int>>.Create((a, b) => a.Value == b.Value ? a.Key.CompareTo(b.Key) : a.Value.CompareTo(b.Value)));
                //TODO: Replace above SortedSet with PriorityQueue

                foreach (var entry in this.playerScores)
                {
                    minHeap.Add(entry);
                    if (minHeap.Count > K)
                    {
                        minHeap.Remove(minHeap.Min);
                    }
                }

                int totalScore = 0;
                foreach (var entry in minHeap)
                {
                    totalScore += entry.Value;
                }

                return totalScore;
            }

            public void Reset(int playerId)
            {
                this.playerScores[playerId] = 0;
            }
        }

        /*
       Approach 3: Using a TreeMap / SortedMap/ SortedDictionary

       Complexity Analysis
       •	Time Complexity:
           o	O(logN) for addScore. This is because each addition to the BST takes a logarithmic time for search. The addition itself once the location of the parent is known, takes constant time.
           o	O(logN) for reset since we need to search for the score in the BST and then update/remove it. Note that this complexity is in the case when every player always maintains a unique score.
           o	It takes O(K) for our top function since we simply iterate over the keys of the TreeMap and stop once we're done considering K scores. Note that if the data structure doesn't provide a natural iterator, then we can simply get a list of all the key-value pairs and they will naturally be sorted due to the nature of this data structure. In that case, the complexity would be O(N) since we would be forming a new list.
       •	Space Complexity:
           o	O(N) used by the scores dictionary. Also, if you obtain all the key-value pairs in a new list in the top function, then an additional O(N) would be used.


       */
        public class LeaderboardSortedDict
        {
            private Dictionary<int, int> playerScores;
            private SortedDictionary<int, int> sortedPlayerScores;

            public LeaderboardSortedDict()
            {
                this.playerScores = new Dictionary<int, int>();
                this.sortedPlayerScores = new SortedDictionary<int, int>(Comparer<int>.Create((x, y) => y.CompareTo(x)));
            }

            public void AddScore(int playerId, int score)
            {
                // The playerScores dictionary simply contains the mapping from the
                // playerId to their score. The sortedPlayerScores contain a BST with 
                // key as the score and value as the number of players that have
                // that score.        
                if (!this.playerScores.ContainsKey(playerId))
                {
                    this.playerScores[playerId] = score;
                    this.sortedPlayerScores[score] = this.sortedPlayerScores.GetValueOrDefault(score, 0) + 1;
                }
                else
                {
                    // Since the current player's score is changing, we need to
                    // update the sortedPlayerScores map to reduce count for the old
                    // score.
                    int previousScore = this.playerScores[playerId];
                    int playerCount = this.sortedPlayerScores[previousScore];

                    // If no player has this score, remove it from the tree.
                    if (playerCount == 1)
                    {
                        this.sortedPlayerScores.Remove(previousScore);
                    }
                    else
                    {
                        this.sortedPlayerScores[previousScore] = playerCount - 1;
                    }

                    // Updated score
                    int newScore = previousScore + score;
                    this.playerScores[playerId] = newScore;
                    this.sortedPlayerScores[newScore] = this.sortedPlayerScores.GetValueOrDefault(newScore, 0) + 1;
                }
            }

            public int Top(int K)
            {
                int count = 0;
                int sum = 0;

                // In-order traversal over the scores in the SortedDictionary
                foreach (var entry in this.sortedPlayerScores)
                {
                    // Number of players that have this score.
                    int times = entry.Value;
                    int key = entry.Key;

                    for (int i = 0; i < times; i++)
                    {
                        sum += key;
                        count++;

                        // Found top-K scores, break.
                        if (count == K)
                        {
                            break;
                        }
                    }

                    // Found top-K scores, break.
                    if (count == K)
                    {
                        break;
                    }
                }

                return sum;
            }

            public void Reset(int playerId)
            {
                int previousScore = this.playerScores[playerId];
                this.sortedPlayerScores[previousScore] = this.sortedPlayerScores[previousScore] - 1;
                if (this.sortedPlayerScores[previousScore] == 0)
                {
                    this.sortedPlayerScores.Remove(previousScore);
                }

                this.playerScores.Remove(playerId);
            }
        }

    }
}