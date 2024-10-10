using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class SubstringOps
    {
        //https://www.algoexpert.io/questions/smallest-substring-containing
        public static string SmallestSubstringContaining(string bigStr, string smallStr)
        {
            //T:O(b+s) | S:O(b+s) where b is length of big string and s is length of small string

            Dictionary<char, int> targetCharCounts = GetCharCounts(smallStr);
            List<int> subStringBounds = GetSubstringBounds(bigStr, targetCharCounts);
            return GetStringFromBounds(bigStr, targetCharCounts);
        }

        private static string GetStringFromBounds(string bigStr, Dictionary<char, int> targetCharCounts)
        {
            throw new NotImplementedException();
        }

        private static List<int> GetSubstringBounds(string str, Dictionary<char, int> targetCharCounts)
        {
            List<int> substringBounds = new List<int> { 0, Int32.MaxValue };

            Dictionary<char, int> substringCharCounts = new Dictionary<char, int>();
            int numUniqueChars = targetCharCounts.Count;
            int numUniqueCharsDone = 0;
            int leftIdx = 0, rightIdx = 0;

            while (rightIdx < str.Length)
            {
                char rightChar = str[rightIdx];

                if (!targetCharCounts.ContainsKey(rightChar))
                {
                    rightIdx++;
                    continue;
                }
                if (!substringCharCounts.ContainsKey(rightChar))
                    substringCharCounts[rightChar] = 0;

                substringCharCounts[rightChar]++;

                if (substringCharCounts[rightChar] == targetCharCounts[rightChar])
                {
                    numUniqueCharsDone++;
                }

                while (numUniqueCharsDone == numUniqueChars && leftIdx <= rightIdx)
                {

                    //substringBounds = GetCloserBound 
                    // TODO:

                }

            }
            return substringBounds;

        }

        private static Dictionary<char, int> GetCharCounts(string smallStr)
        {
            Dictionary<char, int> charCounts = new Dictionary<char, int>();

            foreach (char c in smallStr)
            {
                if (!charCounts.ContainsKey(c))
                    charCounts[c] = 0;
                charCounts[c]++;
            }

            return charCounts;
        }

        //https://www.algoexpert.io/questions/longest-substring-without-duplication
        public static string LongestSubstringWithoutDuplication(string str)
        {
            //T:O(n) | S:O(min(n,a)) where n is length of input string and a is length of unique letters  represented in the input string
            Dictionary<char, int> lastSeen = new Dictionary<char, int>();
            int startIndex = 0;
            int subStrLen = 1;
            lastSeen[str[0]] = 0;
            int[] longest = { 0, 1 };
            for (int curIndex = 1; curIndex < str.Length; curIndex++)
            {

                if (lastSeen.ContainsKey(str[curIndex]))
                {

                    startIndex = Math.Max(startIndex, lastSeen[str[curIndex]] + 1);
                    lastSeen[str[curIndex]] = curIndex;
                }

                int curSubstrLen = curIndex - startIndex + 1;
                if (curSubstrLen > subStrLen)
                {
                    subStrLen = curSubstrLen;
                    longest[0] = startIndex;
                    longest[1] = curIndex;
                }
                lastSeen[str[curIndex]] = curIndex;


            }
            return str.Substring(longest[0], longest[1] - longest[0] + 1);

        }

        /*
        30. Substring with Concatenation of All Words	
https://leetcode.com/problems/substring-with-concatenation-of-all-words/description/	

        */
        public IList<int> FindSubstring(string s, string[] words)
        {
            /*

Approach 1: Check All Indices Using a Hash Table (HT)
Complexity Analysis
Given n as the length of s, a as the length of words, and b as the length of each word:
•	Time complexity: O(n⋅a⋅b−(a⋅b)^2)
First, let's analyze the time complexity of check. We start by creating a copy of our hash table, which in the worst case will take O(a) time, when words only has unique elements. Then, we iterate a times (from i to i + substringSize, wordLength at a time): substringSize / wordLength = words.length = a. At each iteration, we create a substring, which takes wordLength = b time. Then we do a hash table check.
That means each call to check uses O(a+a⋅(b+1)) time, simplified to O(a⋅b). How many times do we call check? Only n - substringSize times. Recall that substringSize is equal to the length of words times the length of words[0], which we have defined as a and b respectively here. That means we call check n−a⋅b times.
This gives us a time complexity of O((n−a⋅b)⋅a⋅b), which can be expanded to O(n⋅a⋅b−(a⋅b)2).
•	Space complexity: O(a+b)
Most of the time, the majority of extra memory we use is the hash table to store word counts. In the worst-case scenario where words only has unique elements, we will store up to a keys.
We also store substrings in sub which requires O(b) space. So the total space complexity of this approach is O(a+b). However, because for this particular problem the upper bound for b is very small (30), we can consider the space complexity to be O(a).

            
            */
            IList<int> foundWords = FindSubstringHT(s, words);

            /*
 Approach 2: Sliding Window     (SW)      
  Complexity Analysis
Given n as the length of s, a as the length of words, and b as the length of each word:
•	Time complexity: O(a+n⋅b)
First, let's analyze the time complexity of slidingWindow(). The for loop in this function iterates from the starting index left up to n, at increments of wordLength. This results in n / b total iterations. At each iteration, we create a substring of length wordLength, which costs O(b).
Although there is a nested while loop, the left pointer can only move over each word once, so this inner loop will only ever perform a total of n / wordLength iterations summed across all iterations of the outer for loop. Inside that while loop, we also take a substring which costs O(b), which means each iteration will cost at most O(2⋅b) on average.
This means that each call to slidingWindow costs O(bn⋅2⋅b), or O(n). How many times do we call slidingWindow? wordLength, or b times. This means that all calls to slidingWindow costs O(n⋅b).
On top of the calls to slidingWindow, at the start of the algorithm we create a dictionary wordCount by iterating through words, which costs O(a). This gives us our final time complexity of O(a+n⋅b).
Notice that the length of words a is not multiplied by anything, which makes this approach much more efficient than the first approach due to the bounds of the problem, as n>a≫b.
•	Space complexity: O(a+b)
Most of the times, the majority of extra memory we use is due to the hash tables used to store word counts. In the worst-case scenario where words only has unique elements, we will store up to a keys in the tables.
We also store substrings in sub which requires O(b) space. So the total space complexity of this approach is O(a+b). However, because for this particular problem the upper bound for b is very small (30), we can consider the space complexity to be O(a).
          
            */
            foundWords = FindSubstringSW(s, words);

            return foundWords;

        }

        private Dictionary<string, int> wordCount = new Dictionary<string, int>();
        private int wordLength;
        private int substringSize;
        private int k;

        private bool Check(int i, string s)
        {
            // Copy the original dictionary to use for this index
            var remaining = new Dictionary<string, int>(wordCount);
            int wordsUsed = 0;
            // Each iteration will check for a match in words
            for (int j = i; j < i + substringSize; j += wordLength)
            {
                string sub = s.Substring(j, wordLength);
                if (remaining.ContainsKey(sub) && remaining[sub] != 0)
                {
                    remaining[sub] = remaining[sub] - 1;
                    wordsUsed++;
                }
                else
                {
                    break;
                }
            }

            return wordsUsed == k;
        }

        public IList<int> FindSubstringHT(string s, string[] words)
        {
            int n = s.Length;
            k = words.Length;
            wordLength = words[0].Length;
            substringSize = wordLength * k;
            foreach (var word in words)
                wordCount[word] =
                    wordCount.ContainsKey(word) ? wordCount[word] + 1 : 1;
            IList<int> answer = new List<int>();
            for (int i = 0; i < n - substringSize + 1; i++)
            {
                if (Check(i, s))
                {
                    answer.Add(i);
                }
            }

            return answer;
        }
        int n;
        void SlidingWindow(int left, string s, List<int> answer)
        {
            Dictionary<string, int> wordsFound = new Dictionary<string, int>();
            int wordsUsed = 0;
            bool excessWord = false;
            for (int right = left; right <= n - wordLength; right += wordLength)
            {
                string sub = s.Substring(right, wordLength);
                if (!wordCount.ContainsKey(sub))
                {
                    // Mismatched word - reset the window
                    wordsFound.Clear();
                    wordsUsed = 0;
                    excessWord = false;
                    left = right + wordLength;
                }
                else
                {
                    // If we reached max window size or have an excess word
                    while (right - left == substringSize || excessWord)
                    {
                        string leftmostWord = s.Substring(left, wordLength);
                        left += wordLength;
                        wordsFound[leftmostWord]--;
                        if (wordsFound[leftmostWord] >= wordCount[leftmostWord])
                        {
                            // This word was an excess word
                            excessWord = false;
                        }
                        else
                        {
                            // Otherwise we actually needed it
                            wordsUsed--;
                        }
                    }

                    // Keep track of how many times this word occurs in the window
                    if (!wordsFound.ContainsKey(sub))
                    {
                        wordsFound[sub] = 0;
                    }

                    wordsFound[sub]++;
                    if (wordsFound[sub] <= wordCount[sub])
                    {
                        wordsUsed++;
                    }
                    else
                    {
                        // Found too many instances already
                        excessWord = true;
                    }

                    if (wordsUsed == k && !excessWord)
                    {
                        // Found a valid substring
                        answer.Add(left);
                    }
                }
            }
        }

        public IList<int> FindSubstringSW(string s, string[] words)
        {
            n = s.Length;
            k = words.Length;
            wordLength = words[0].Length;
            substringSize = wordLength * k;
            foreach (string word in words)
            {
                if (!wordCount.ContainsKey(word))
                    wordCount[word] = 0;
                wordCount[word]++;
            }

            List<int> answer = new List<int>();
            for (int i = 0; i < wordLength; i++)
            {
                SlidingWindow(i, s, answer);
            }

            return answer;
        }


        /*
76. Minimum Window Substring
https://leetcode.com/problems/minimum-window-substring/description/

        */

        public string MinWindow(string s, string t)
        {
            /*
   Approach 1: Sliding Window (SW)         
   Complexity Analysis
•	Time Complexity: O(∣S∣+∣T∣) where |S| and |T| represent the lengths of strings S and T.
In the worst case we might end up visiting every element of string S twice, once by left pointer and once by right pointer. ∣T∣ represents the length of string T.
•	Space Complexity: O(∣S∣+∣T∣). ∣S∣ when the window size is equal to the entire string S. ∣T∣ when T has all unique characters
         
            */
            string minWindowSubstr = MinWindowSW(s, t);

            /*
 Approach 2: Optimized Sliding Window  (OSW)
  Complexity Analysis
•	Time Complexity : O(∣S∣+∣T∣) where |S| and |T| represent the lengths of strings S and T. The complexity is same as the previous approach. But in certain cases where ∣filtered_S∣ <<< ∣S∣, the complexity would reduce because the number of iterations would be 2∗∣filtered_S∣+∣S∣+∣T∣.
•	Space Complexity : O(∣S∣+∣T∣).
          
            */
            minWindowSubstr = MinWindowOSW(s, t);

            return minWindowSubstr;

        }
        public string MinWindowSW(string s, string t)
        {
            if (s.Length == 0 || t.Length == 0)
            {
                return "";
            }

            // Dictionary which keeps a count of all the unique characters in t.
            Dictionary<char, int> dictT = new Dictionary<char, int>();
            for (int i = 0; i < t.Length; i++)
            {
                if (dictT.ContainsKey(t[i]))
                {
                    dictT[t[i]]++;
                }
                else
                {
                    dictT[t[i]] = 1;
                }
            }

            // Number of unique characters in t, which need to be present in the
            // desired window.
            int required = dictT.Count;
            // Left and Right pointer
            int l = 0, r = 0;
            // formed is used to keep track of how many unique characters in t
            // are present in the current window in its desired frequency.
            // e.g. if t is "AABC" then the window must have two A's, one B and one
            // C. Thus formed would be = 3 when all these conditions are met.
            int formed = 0;
            // Dictionary which keeps a count of all the unique characters in the
            // current window.
            Dictionary<char, int> windowCounts = new Dictionary<char, int>();
            // ans list of the form (window length, left, right)
            int[] ans = { -1, 0, 0 };
            while (r < s.Length)
            {
                // Add one character from the right to the window
                char c = s[r];
                if (windowCounts.ContainsKey(c))
                {
                    windowCounts[c]++;
                }
                else
                {
                    windowCounts[c] = 1;
                }

                // If the frequency of the current character added equals to the
                // desired count in t then increment the formed count by 1.
                if (dictT.ContainsKey(c) && windowCounts[c] == dictT[c])
                {
                    formed++;
                }

                // Try and contract the window till the point where it ceases to be
                // 'desirable'.
                while (l <= r && formed == required)
                {
                    c = s[l];
                    // Save the smallest window until now.
                    if (ans[0] == -1 || r - l + 1 < ans[0])
                    {
                        ans[0] = r - l + 1;
                        ans[1] = l;
                        ans[2] = r;
                    }

                    // The character at the position pointed by the
                    // `Left` pointer is no longer a part of the window.
                    windowCounts[c]--;
                    if (dictT.ContainsKey(c) && windowCounts[c] < dictT[c])
                    {
                        formed--;
                    }

                    // Move the left pointer ahead, this would help to look for a
                    // new window.
                    l++;
                }

                // Keep expanding the window once we are done contracting.
                r++;
            }

            return ans[0] == -1 ? "" : s.Substring(ans[1], ans[2] - ans[1] + 1);
        }
        public string MinWindowOSW(string s, string t)
        {
            if (s.Length == 0 || t.Length == 0)
            {
                return "";
            }

            Dictionary<char, int> dictT = new Dictionary<char, int>();
            for (int i = 0; i < t.Length; i++)
            {
                int count = dictT.ContainsKey(t[i]) ? dictT[t[i]] : 0;
                dictT[t[i]] = count + 1;
            }

            int required = dictT.Count;
            List<KeyValuePair<int, char>> filteredS =
                new List<KeyValuePair<int, char>>();
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (dictT.ContainsKey(c))
                {
                    filteredS.Add(new KeyValuePair<int, char>(i, c));
                }
            }

            int l = 0, r = 0, formed = 0;
            Dictionary<char, int> windowCounts = new Dictionary<char, int>();
            int[] ans = { -1, 0, 0 };
            while (r < filteredS.Count)
            {
                char c = filteredS[r].Value;
                int count = windowCounts.ContainsKey(c) ? windowCounts[c] : 0;
                windowCounts[c] = count + 1;
                if (dictT.ContainsKey(c) && windowCounts[c] == dictT[c])
                {
                    formed++;
                }

                while (l <= r && formed == required)
                {
                    c = filteredS[l].Value;
                    int end = filteredS[r].Key;
                    int start = filteredS[l].Key;
                    if (ans[0] == -1 || end - start + 1 < ans[0])
                    {
                        ans[0] = end - start + 1;
                        ans[1] = start;
                        ans[2] = end;
                    }

                    windowCounts[c] = windowCounts[c] - 1;
                    if (dictT.ContainsKey(c) && windowCounts[c] < dictT[c])
                    {
                        formed--;
                    }

                    l++;
                }

                r++;
            }

            return ans[0] == -1 ? "" : s.Substring(ans[1], ans[2] - ans[1] + 1);
        }


        /* 1044. Longest Duplicate Substring
      https://leetcode.com/problems/longest-duplicate-substring/description/
       */

        class LongestDupSubstringSol
        {
            /*
              Rabin-Karp with polynomial rolling hash.
              Search a substring of given length that occurs at least 2 times.
              Return start position if the substring exists and -1 otherwise.
            */
            private string inputString;

            /*
             Approach 1: Binary Search + Rabin-Karp
             Complexity Analysis
            Let N be the length of input S.
            •	Time complexity: O(NlogN).
            Performing a binary search requires O(logN) iterations. At each iteration, we spend on average O(N) time for the Rabin-Karp algorithm. Note that the worst-case scenario for the Rabin-Karp algorithm is when every substring of length L has the same hash value and there are no duplicate substrings of length L. This would require O(L⋅(N−L)/2) time to compare each of the O(N−L) substrings to all previous substrings resulting in O(L⋅(N−L)^2).
            However, because of the problem constraints, there can be at most 30,000 substrings and because we have 109+7 bins, the probability of a collision occurring between two different substrings is small. It is quite possible that there will be some collisions, but the probability of there being many collisions (on the order of N−L collisions) is extraordinarily small. So the average time complexity of the Rabin-Karp algorithm will be O(N−L) which simplifies to O(N).
            •	Space complexity: O(N)
            We use a hashmap seen to store the starting index and hash value for each substring. This will contain at most N key-value pairs.

             */
            public string BinarySearchAndRabinKarpAlgo(string s)
            {
                inputString = s;
                int stringLength = inputString.Length;

                // Convert string to array of integers to implement constant time slice
                int[] nums = new int[stringLength];
                for (int i = 0; i < stringLength; ++i)
                {
                    nums[i] = (int)inputString[i] - (int)'a';
                }

                // Base value for the rolling hash function
                int baseValue = 26;

                // modulus value for the rolling hash function to avoid overflow
                int modulus = 1_000_000_007;

                // Binary search, length = repeating string length
                int left = 1, right = stringLength;
                while (left <= right)
                {
                    int mid = left + (right - left) / 2;
                    if (Search(mid, baseValue, modulus, stringLength, nums) != -1)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                int start = Search(left - 1, baseValue, modulus, stringLength, nums);
                return inputString.Substring(start, left - 1);
            }
            private int Search(int length, int baseValue, long modulus, int stringLength, int[] nums)
            {
                // Compute the hash of string inputString[:length]
                long hashValue = 0;
                for (int i = 0; i < length; ++i)
                {
                    hashValue = (hashValue * baseValue + nums[i]) % modulus;
                }

                // Store the already seen hash values for substrings of length length.
                Dictionary<long, List<int>> seenHashes = new Dictionary<long, List<int>>();

                // Initialize the dictionary with the substring starting at index 0.
                if (!seenHashes.ContainsKey(hashValue))
                {
                    seenHashes[hashValue] = new List<int>();
                }
                seenHashes[hashValue].Add(0);

                // Const value to be used often : baseValue**length % modulus
                long baseValueToLength = 1;
                for (int i = 1; i <= length; ++i)
                {
                    baseValueToLength = (baseValueToLength * baseValue) % modulus;
                }

                for (int start = 1; start < stringLength - length + 1; ++start)
                {
                    // Compute rolling hash in O(1) time
                    hashValue = (hashValue * baseValue - nums[start - 1] * baseValueToLength % modulus + modulus) % modulus;
                    hashValue = (hashValue + nums[start + length - 1]) % modulus;
                    if (seenHashes.TryGetValue(hashValue, out List<int> hits))
                    {
                        // Check if the current substring matches any of 
                        // the previous substrings with hash hashValue.
                        string currentSubstring = inputString.Substring(start, length);
                        foreach (int index in hits)
                        {
                            string candidateSubstring = inputString.Substring(index, length);
                            if (candidateSubstring.Equals(currentSubstring))
                            {
                                return index;
                            }
                        }
                    }
                    // Add the current substring's hash value and starting index to seen.
                    if (!seenHashes.ContainsKey(hashValue))
                    {
                        seenHashes[hashValue] = new List<int>();
                    }
                    seenHashes[hashValue].Add(start);
                }
                return -1;
            }


        }


        /* 1520. Maximum Number of Non-Overlapping Substrings
        https://leetcode.com/problems/maximum-number-of-non-overlapping-substrings/description/
         */
        public class SubstringChecker
        {
/* Approach: Greedy O(n)	
Complexity Analysis
•	Time: O(n). In the worst case, we search for substring 26 times, and each search is O(n)
•	Memory: O(1). We store left and right positions for 26 characters.
o	For the complexity analysis purposes, we ignore memory required by inputs and outputs.

 */
            public List<string> MaxNumberOfSubstrings(string inputString)
            {
                int[] leftIndices = new int[26];
                int[] rightIndices = new int[26];
                Array.Fill(leftIndices, inputString.Length);
                var result = new List<string>();

                // Record the leftmost and rightmost index for each character.
                for (int i = 0; i < inputString.Length; ++i)
                {
                    var characterIndex = inputString[i] - 'a';
                    leftIndices[characterIndex] = Math.Min(leftIndices[characterIndex], i);
                    rightIndices[characterIndex] = i;
                }

                int rightBoundary = -1;
                
                // check if it forms a valid solution.
                for (int i = 0; i < inputString.Length; ++i)
                {
                    if (i == leftIndices[inputString[i] - 'a'])
                    {
                        int newRightBoundary = CheckSubstring(inputString, i, leftIndices, rightIndices);
                        if (newRightBoundary != -1)
                        {
                            if (i > rightBoundary)
                                result.Add("");
                            rightBoundary = newRightBoundary;
                            result[result.Count - 1] = inputString.Substring(i, rightBoundary - i + 1);
                        }
                    }
                }
                return result;
            }
            private int CheckSubstring(string inputString, int index, int[] leftIndices, int[] rightIndices)
            {
                int rightBoundary = rightIndices[inputString[index] - 'a'];
                for (int j = index; j <= rightBoundary; ++j)
                {
                    if (leftIndices[inputString[j] - 'a'] < index)
                        return -1;
                    rightBoundary = Math.Max(rightBoundary, rightIndices[inputString[j] - 'a']);
                }
                return rightBoundary;
            }

        }




































































    }
}