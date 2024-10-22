using System;

namespace AlgoDSPlay.Design
{
    /*     244. Shortest Word Distance II
    https://leetcode.com/problems/shortest-word-distance-ii/description/
     */
    class WordDistance
    {
        private Dictionary<string, List<int>> wordLocations;

        public WordDistance(string[] words)
        {
            this.wordLocations = new Dictionary<string, List<int>>();

            // Prepare a mapping from a word to all its locations (indices).
            for (int index = 0; index < words.Length; index++)
            {
                List<int> locations = this.wordLocations.GetValueOrDefault(words[index], new List<int>());
                locations.Add(index);
                this.wordLocations[words[index]] = locations;
            }
        }

        public int Shortest(string word1, string word2)
        {
            List<int> locations1, locations2;

            // Location lists for both the words
            // the indices will be in SORTED order by default
            locations1 = this.wordLocations[word1];
            locations2 = this.wordLocations[word2];

            int index1 = 0, index2 = 0, minimumDifference = int.MaxValue;
            while (index1 < locations1.Count && index2 < locations2.Count)
            {
                minimumDifference = Math.Min(minimumDifference, Math.Abs(locations1[index1] - locations2[index2]));
                if (locations1[index1] < locations2[index2])
                {
                    index1++;
                }
                else
                {
                    index2++;
                }
            }

            return minimumDifference;
        }
    }
}
