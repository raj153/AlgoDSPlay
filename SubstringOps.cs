using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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

        /* 1190. Reverse Substrings Between Each Pair of Parentheses
        https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/description/
         */
        public class ReverseSubstringsBetweenEachPairOfParenthesesSol
        {

            /* Approach 1: Straightforward Way
            Complexity Analysis
            Let n be the length of the string.
            •	Time complexity: O(n^2)
            The algorithm iterates through each character of the input string once. For each character, we have three cases:
            o	If it's (, we push its index or the starting position where the reversal takes place to the stack. This is O(1).
            o	If it's ), we pop from the stack and reverse a portion of the result string. Popping is O(1).
            o	The reverse operation can take up to O(n) time in the worst case (when we reverse the entire string).
            o	For other characters, we append to the result string, which is typically O(1) (amortized).
            The worst-case scenario occurs when we have to reverse large portions of the string multiple times. In the worst case, we might end up reversing the entire string for each closing parenthesis. Therefore, the overall time complexity is O(n^2) in the worst case.
            •	Space complexity: O(n)
            The algorithm uses a stack to store the indices of opening parentheses. In the worst case (when all characters are opening parentheses), this could take O(n) space. The reverse function typically doesn't use extra space proportional to the input size. Therefore, the overall space complexity is O(n).

             */
            public string ReverseParentheses(string s)
            {
                Stack<int> openParenthesesIndices = new Stack<int>();
                StringBuilder result = new StringBuilder();

                foreach (char currentChar in s)
                {
                    if (currentChar == '(')
                    {
                        // Store the current length as the start index for future reversal
                        openParenthesesIndices.Push(result.Length);
                    }
                    else if (currentChar == ')')
                    {
                        int start = openParenthesesIndices.Pop();
                        // Reverse the substring between the matching parentheses
                        Reverse(result, start, result.Length - 1);
                    }
                    else
                    {
                        // Append non-parenthesis characters to the processed string
                        result.Append(currentChar);
                    }
                }

                return result.ToString();
            }

            private void Reverse(StringBuilder sb, int start, int end)
            {
                while (start < end)
                {
                    char temp = sb[start];
                    sb[start++] = sb[end];
                    sb[end--] = temp;
                }
            }
            /* Approach 2: Wormhole Teleportation technique
            Complexity Analysis
    Let n be the length of the string.
    •	Time complexity: O(n)
    We iterate through the string once to pair up parentheses using a stack. Each character is processed once, resulting in O(n) time complexity.
    After pairing, we iterate through the string again to construct the final result string. During this pass, each character is processed once, and we navigate through pairs in constant time. This results in another O(n) time complexity.
    Converting a StringBuilder to a String in Java using toString() takes O(n) time, where n is the length of the StringBuilder. Joining elements of a list into a string in Python using ''.join() also takes O(n) time. Combined, the total time complexity is O(n).
    •	Space complexity: O(n)
    We use a stack to track indices of opening parentheses. In the worst case, the stack may hold up to O(n/2) elements (when all are opening parentheses), resulting in O(n) space complexity. An array pair of size n is used to store indices of matching parentheses. This contributes O(n) space complexity.
    Converting a StringBuilder to a String in Java generally does not increase space complexity beyond the size of the resulting string itself. However, StringBuilder internally manages a character array whose size might be slightly larger than the resulting string due to its capacity management strategy. The additional space complexity for ''.join() in Python is O(n), accounting for the space needed to store the new string object.
    Therefore, the total space complexity is O(n).

             */
            public String UsingWormholeTeleportationTechnique(String s)
            {
                int n = s.Length;
                Stack<int> openParenthesesIndices = new();
                int[] pair = new int[n];

                // First pass: Pair up parentheses
                for (int i = 0; i < n; ++i)
                {
                    if (s[i] == '(')
                    {
                        openParenthesesIndices.Push(i);
                    }
                    if (s[i] == ')')
                    {
                        int j = openParenthesesIndices.Pop();
                        pair[i] = j;
                        pair[j] = i;
                    }
                }

                // Second pass: Build the result string
                StringBuilder result = new StringBuilder();
                for (
                    int currIndex = 0, direction = 1;
                    currIndex < n;
                    currIndex += direction
                )
                {
                    if (s[currIndex] == '(' || s[currIndex]

                     == ')')
                    {
                        currIndex = pair[currIndex];
                        direction = -direction;
                    }
                    else
                    {
                        result.Append(s[currIndex]);
                    }
                }

                return result.ToString();
            }
        }

        /* 1915. Number of Wonderful Substrings
        https://leetcode.com/problems/number-of-wonderful-substrings/description/
         */

        class NumberOfWonderfulSubstringsSol
        {
            /*
            
Approach: Count Parity Prefixes

            Complexity Analysis
•	Time complexity: O(NA).
The number of distinct characters that can appear in word is defined as A. For each of the N characters in word, we iterate through all possible characters that can be the odd character. Therefore, the time complexity of O(NA), where A≤10, because only letters "a" through "j" will appear.
•	Space complexity: O(N).
The frequency map can store up to N key/entry pairs, hence the linear space complexity.
  */
            public long UsingCountParityPrefixes(String word)
            {
                int N = word.Length;

                // Create the frequency map
                // Key = bitmask, Value = frequency of bitmask key
                Dictionary<int, int> freq = new();

                // The empty prefix can be the smaller prefix, which is handled like this
                freq.Add(0, 1);

                int mask = 0;
                long res = 0L;
                for (int i = 0; i < N; i++)
                {
                    char c = word[i];
                    int bit = c - 'a';

                    // Flip the parity of the c-th bit in the running prefix mask
                    mask ^= (1 << bit);

                    // Count smaller prefixes that create substrings with no odd occurring letters
                    res += freq.GetValueOrDefault(mask, 0);

                    // Increment value associated with mask by 1
                    freq[mask] = freq.GetValueOrDefault(mask, 0) + 1;

                    // Loop through every possible letter that can appear an odd number of times in a substring
                    for (int odd_c = 0; odd_c < 10; odd_c++)
                    {
                        res += freq.GetValueOrDefault(mask ^ (1 << odd_c), 0);
                    }
                }
                return res;
            }
        }

        /* 1717. Maximum Score From Removing Substrings
        https://leetcode.com/problems/maximum-score-from-removing-substrings/description/
         */
        class MaximumScoreFromRemovingSubstringsSol
        {

            /* Approach 1: Greedy Way (Stack)
            Complexity Analysis
            Let n be the length of the string s.
            •	Time complexity: O(n)
            The removeSubstring method is called twice in the algorithm. In it, the algorithm iterates over each character in the input string, which has a time complexity of O(n). Reconstructing the string from the stack also takes O(n). Thus, the total time complexity of the algorithm is 2⋅(O(n)+O(n)), which simplifies to O(n).
            •	Space complexity: O(n)
            The stringAfterFirstPass and stringAfterSecondPass variables can use an additional space of O(n) in the worst case. In the removeSubstring method, the stack can store at most n characters, and the reconstructed string can also store at most n characters, resulting in a space complexity of O(n) for each. When considering all these individual complexities together, the space complexity of the algorithm amounts to O(n).

             */
            public int UsingGreedyWithStack(String s, int x, int y)
            {
                int totalScore = 0;
                String highPriorityPair = x > y ? "ab" : "ba";
                String lowPriorityPair = highPriorityPair.Equals("ab") ? "ba" : "ab";

                // First pass: remove high priority pair
                String stringAfterFirstPass = RemoveSubstring(s, highPriorityPair);
                int removedPairsCount =
                    (s.Length - stringAfterFirstPass.Length) / 2;

                // Calculate score from first pass
                totalScore += removedPairsCount * Math.Max(x, y);

                // Second pass: remove low priority pair
                String stringAfterSecondPass = RemoveSubstring(
                    stringAfterFirstPass,
                    lowPriorityPair
                );
                removedPairsCount = (stringAfterFirstPass.Length -
                    stringAfterSecondPass.Length) /
                2;

                // Calculate score from second pass
                totalScore += removedPairsCount * Math.Min(x, y);

                return totalScore;
            }

            private String RemoveSubstring(String input, String targetPair)
            {
                Stack<char> charStack = new();

                // Iterate through each character in the input string
                for (int i = 0; i < input.Length; i++)
                {
                    char currentChar = input[i];

                    // Check if current character forms the target pair with the top of the stack
                    if (
                        currentChar == targetPair[1] &&
                        charStack.Count > 0 &&
                        charStack.Peek() == targetPair[0]
                    )
                    {
                        charStack.Pop(); // Remove the matching character from the stack
                    }
                    else
                    {
                        charStack.Push(currentChar);
                    }
                }

                // Reconstruct the remaining string after removing target pairs
                StringBuilder remainingChars = new StringBuilder();
                while (charStack.Count > 0)
                {
                    remainingChars.Append(charStack.Pop());
                }
                return remainingChars.ToString().Reverse().ToString();
            }

            /* Approach 2: Greedy Way (Without Stack)
            Complexity Analysis
            Let n be the length of the string s
            •	Time complexity: O(n)
            The algorithm calls removeSubstring twice, each iterating through the entire string once. All operations within the loop—such as character comparisons and index manipulations—are constant time. Thus, the time complexity is 2⋅O(n), which can be simplified to O(n).
            •	Space complexity: O(1) or O(n)
            In the C++ implementation of the algorithm, where strings are mutable, we do not use any additional data structures which scale with input size. Thus, the space complexity remains O(1).
            In the Java and Python3 implementations, we use an additional data structure to bypass the caveat of immutable strings. This takes O(n) space, which is the space complexity of the algorithm.

             */


            public int UsingGreedyWithoutStack(String s, int x, int y)
            {
                StringBuilder text = new StringBuilder(s);
                int totalPoints = 0;

                if (x > y)
                {
                    // Remove "ab" first (higher points), then "ba"
                    totalPoints += RemoveSubstring(text, "ab", x);
                    totalPoints += RemoveSubstring(text, "ba", y);
                }
                else
                {
                    // Remove "ba" first (higher or equal points), then "ab"
                    totalPoints += RemoveSubstring(text, "ba", y);
                    totalPoints += RemoveSubstring(text, "ab", x);
                }

                return totalPoints;
                int RemoveSubstring(
               StringBuilder inputString,
               String targetSubstring,
               int pointsPerRemoval
           )
                {
                    int totalPoints = 0;
                    int writeIndex = 0;

                    // Iterate through the string
                    for (int readIndex = 0; readIndex < inputString.Length; readIndex++)
                    {
                        // Add the current character
                        inputString[writeIndex++] = inputString[readIndex];

                        // Check if we've written at least two characters and
                        // they match the target substring
                        if (
                            writeIndex > 1 &&
                            inputString[writeIndex - 2] ==
                            targetSubstring[0] &&
                            inputString[writeIndex - 1] == targetSubstring[1]
                        )
                        {
                            writeIndex -= 2; // Move write index back to remove the match
                            totalPoints += pointsPerRemoval;
                        }
                    }

                    // Trim the StringBuilder to remove any leftover characters
                    inputString.Capacity = writeIndex;

                    return totalPoints;

                }
            }
            /* Approach 3: Greedy Way (Counting)
             Complexity Analysis
Let n be the length of the given string s.
•	Time complexity: O(n)
The algorithm reverses the string in the worst case and iterates over each character of the string exactly once, with each operation taking O(n) time. Therefore, the time complexity of the algorithm is O(n).
•	Space complexity: O(1) or O(n)
In the C++ implementation of the algorithm, the string reversal takes constant space since reverse() flips the string in-place.
For the Java and Python3 implementations, the string reversal requires O(n) space.
We do not use any other data structures that scale with the input size. Therefore, the space complexity of the algorithm is O(1) for C++, and O(n) for Java and Python3.

            */
            public int UsingGreedyCounting(String s, int x, int y)
            {
                // Ensure "ab" always has higher points than "ba"
                if (x < y)
                {
                    // Swap points
                    int temp = x;
                    x = y;
                    y = temp;
                    // Reverse the string to maintain logic
                    s = new StringBuilder(s).ToString().Reverse().ToString();
                }

                int aCount = 0, bCount = 0, totalPoints = 0;

                for (int i = 0; i < s.Length; i++)
                {
                    char currentChar = s[i];

                    if (currentChar == 'a')
                    {
                        aCount++;
                    }
                    else if (currentChar == 'b')
                    {
                        if (aCount > 0)
                        {
                            // Can form "ab", remove it and add points
                            aCount--;
                            totalPoints += x;
                        }
                        else
                        {
                            // Can't form "ab", keep 'b' for potential future "ba"
                            bCount++;
                        }
                    }
                    else
                    {
                        // Non 'a' or 'b' character encountered
                        // Calculate points for any remaining "ba" pairs
                        totalPoints += Math.Min(bCount, aCount) * y;
                        // Reset counters for next segment
                        aCount = bCount = 0;
                    }
                }

                // Calculate points for any remaining "ba" pairs at the end
                totalPoints += Math.Min(bCount, aCount) * y;

                return totalPoints;
            }



        }

        /* 1371. Find the Longest Substring Containing Vowels in Even Counts
        https://leetcode.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/description/
         */
        class FindTheLongestSubstringContainingVowelsinEvenCountsSol
        {
            /* Approach: Bitmasking
            Complexity Analysis
            Let m be the size of the given s string.
            •	Time complexity: O(n)
            We iterate through the string s exactly once. Apart from this, all operations are constant time. Therefore, the total time complexity is given by O(max(m,n)).
            •	Space complexity: O(1)
            Apart from the characterMap and mp array, no additional space is used to solve the problem. Therefore, the space complexity is given by O(26)+O(32)≈O(1).

             */
            public int UsingBitMasking(String s)
            {
                int prefixXOR = 0;
                int[] characterMap = new int[26];
                characterMap['a' - 'a'] = 1;
                characterMap['e' - 'a'] = 2;
                characterMap['i' - 'a'] = 4;
                characterMap['o' - 'a'] = 8;
                characterMap['u' - 'a'] = 16;
                int[] mp = new int[32];
                for (int i = 0; i < 32; i++) mp[i] = -1;
                int longestSubstring = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    prefixXOR ^= characterMap[s[i] - 'a'];
                    if (mp[prefixXOR] == -1 && prefixXOR != 0) mp[prefixXOR] = i;
                    longestSubstring = Math.Max(longestSubstring, i - mp[prefixXOR]);
                }
                return longestSubstring;
            }
        }

        /* 395. Longest Substring with At Least K Repeating Characters
        https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/description/
         */

        class LongestSubstringWithAtleastKRepeatingCharSol
        {
            /*
                         Approach 1: Brute Force
Complexity Analysis
•	Time Complexity : O(n^2), where n is equal to length of string s. The nested for loop that generates all substrings from string s takes O(n^2) time, and for each substring, we iterate over countMap array of size 26.
This gives us time complexity as O(26⋅n^2) = O(n^2).
This approach is exhaustive and results in Time Limit Exceeded (TLE).
•	Space Complexity: O(1) We use constant extra space of size 26 for countMap array.

             */
            public int LongestSubstring(String s, int k)
            {
                if (s == null || s.Length == 0 || k > s.Length)
                {
                    return 0;
                }
                int[] countMap = new int[26];
                int n = s.Length;
                int result = 0;
                for (int start = 0; start < n; start++)
                {
                    // reset the count map
                    Array.Fill(countMap, 0);
                    for (int end = start; end < n; end++)
                    {
                        countMap[s[end] - 'a']++;
                        if (IsValid(s, start, end, k, countMap))
                        {
                            result = Math.Max(result, end - start + 1);
                        }
                    }
                }
                return result;
            }

            private bool IsValid(String s, int start, int end, int k, int[] countMap)
            {
                int countLetters = 0, countAtLeastK = 0;
                foreach (int freq in countMap)
                {
                    if (freq > 0) countLetters++;
                    if (freq >= k) countAtLeastK++;
                }
                return countAtLeastK == countLetters;
            }
            /* Approach 2: Divide And Conquer
Complexity Analysis
•	Time Complexity : O(N^2), where N is the length of string s. Though the algorithm performs better in most cases, the worst case time complexity is still O(N^2).
In cases where we perform split at every index, the maximum depth of recursive call could be O(N). For each recursive call it takes O(N) time to build the countMap resulting in O(n^2) time complexity.
•	Space Complexity: O(N) This is the space used to store the recursive call stack. The maximum depth of recursive call stack would be O(N).
             */
            public int UsingDivideAndConquer(String s, int k)
            {
                return LongestSubstringUtil(s, 0, s.Length, k);
            }
            private int LongestSubstringUtil(String s, int start, int end, int k)
            {
                if (end < k) return 0;
                int[] countMap = new int[26];
                // update the countMap with the count of each character
                for (int i = start; i < end; i++)
                    countMap[s[i] - 'a']++;
                for (int mid = start; mid < end; mid++)
                {
                    if (countMap[s[mid] - 'a'] >= k) continue;
                    int midNext = mid + 1;
                    while (midNext < end && countMap[s[midNext] - 'a'] < k) midNext++;
                    return Math.Max(LongestSubstringUtil(s, start, mid, k),
                            LongestSubstringUtil(s, midNext, end, k));
                }
                return (end - start);
            }
            /* Approach 3: Sliding Window 
Complexity Analysis
•	Time Complexity : O(maxUnique⋅N). We iterate over the string of length N, maxUnqiue times. Ideally, the number of unique characters in the string would not be more than 26 (a to z). Hence, the time complexity is approximately O(26⋅N) = O(N)
•	Space Complexity: O(1) We use constant extra space of size 26 to store the countMap.

            */
            public int UsingSlidingWindow(String s, int k)
            {
                char[] str = s.ToCharArray();
                int[] countMap = new int[26];
                int maxUnique = GetMaxUniqueLetters(s);
                int result = 0;
                for (int currUnique = 1; currUnique <= maxUnique; currUnique++)
                {
                    // reset countMap
                    Array.Fill(countMap, 0);
                    int windowStart = 0, windowEnd = 0, idx = 0, unique = 0, countAtLeastK = 0;
                    while (windowEnd < str.Length)
                    {
                        // expand the sliding window
                        if (unique <= currUnique)
                        {
                            idx = str[windowEnd] - 'a';
                            if (countMap[idx] == 0) unique++;
                            countMap[idx]++;
                            if (countMap[idx] == k) countAtLeastK++;
                            windowEnd++;
                        }
                        // shrink the sliding window
                        else
                        {
                            idx = str[windowStart] - 'a';
                            if (countMap[idx] == k) countAtLeastK--;
                            countMap[idx]--;
                            if (countMap[idx] == 0) unique--;
                            windowStart++;
                        }
                        if (unique == currUnique && unique == countAtLeastK)
                            result = Math.Max(windowEnd - windowStart, result);
                    }
                }

                return result;
            }
            // get the maximum number of unique letters in the string s
            private int GetMaxUniqueLetters(String s)
            {
                bool[] map = new bool[26];
                int maxUnique = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    if (!map[s[i] - 'a'])
                    {
                        maxUnique++;
                        map[s[i] - 'a'] = true;
                    }
                }
                return maxUnique;
            }

        }

        /* 2743. Count Substrings Without Repeating Character
        https://leetcode.com/problems/count-substrings-without-repeating-character/description/
         */
        class NumberOfSpecialSubstringsSol
        {

            /* Approach: Sliding Window
            Complexity Analysis
            Here, N is the number of characters in the string s.
            •	Time complexity: O(N).
            We can iterate over each character at most twice. This is because we will iterate over the character for the first time while extending the sliding window from the right side and then we can again iterate over while shrinking the window from the left end. Hence, the total number of operations could be 2∗N and therefore, the total time complexity is equal to O(N).
            •	Space complexity: O(1).
            We need an array freq to keep the frequencies of characters in the current window. Since there can only be lowercase English letters the size of freq is only 26 and hence is independent of s length. Therefore, the total space complexity is constant.

             */
            public int UsingSlidingWindow(String s)
            {
                int substringCount = 0;

                int start = 0;
                int[] freq = new int[26];
                for (int end = 0; end < s.Length; end++)
                {
                    freq[s[end] - 'a']++;

                    while (freq[s[end] - 'a'] > 1)
                    {
                        freq[s[start] - 'a']--;
                        start++;
                    }

                    substringCount += (end - start + 1);
                }

                return substringCount;
            }
        }


        /* 2734. Lexicographically Smallest String After Substring Operation
        https://leetcode.com/problems/lexicographically-smallest-string-after-substring-operation/description/
        https://algo.monster/liteproblems/2734	
         */

        class LexicographicallySmallestStringAfterSubstringOperationSol
        {
            /*  Time and Space Complexity
Time Complexity:
The time complexity of this code is O(n), where n is the length of the input string s.
•	The first while loop runs in O(n) in the worst case when all characters are "a". At best, it exits immediately if the first character is not "a".
•	The second while loop also runs in O(n) in the worst case, if there are no "a" characters following the first non-"a" character. At best, it exits immediately if the next character is "a".
•	The line with join and chr(ord(c) - 1) inside list comprehension again runs in O(n) because it iterates through the substring s[i:j]. This substring can potentially be the entire string s in the worst case.
These loops are sequential and not nested, so the time complexity remains O(n).
Space Complexity:
The space complexity of the code is also O(n).
•	This is because the code creates a new string with the join operation, which can potentially contain as many characters as the original string s in the worst case.
•	The space used to store indexes i and j is constant and does not scale with the size of the input string, therefore their contribution to space complexity is O(1).
To sum up, the space complexity of the code is dominated by the space required for the new string generated in the join operation, which is O(n).
*/
            public String SmallestString(String s)
            {
                int stringLength = s.Length;
                int firstNonAIndex = 0;

                // Find the first instance of a character that is not 'a'
                while (firstNonAIndex < stringLength && s[firstNonAIndex] == 'a')
                {
                    firstNonAIndex++;
                }

                // If there's no character other than 'a', replace the last 'a' with 'z'
                if (firstNonAIndex == stringLength)
                {
                    return s.Substring(0, stringLength - 1) + "z";
                }

                // Convert the string to a character array for manipulation
                char[] chars = s.ToCharArray();

                // Start decreasing the value of characters until an 'a' is reached
                int reduceIndex = firstNonAIndex;
                while (reduceIndex < stringLength && chars[reduceIndex] != 'a')
                {
                    chars[reduceIndex] = (char)(chars[reduceIndex] - 1);
                    reduceIndex++;
                }

                // Return the new string constructed from the character array
                return chars.ToString();
            }
        }


        /* 340. Longest Substring with At Most K Distinct Characters
        https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/description/
         */
        class LengthOfLongestSubstringWithAtMostKKDistinctCharsSol
        {

            /* Approach 1: Binary Search + Fixed Size Sliding Window
            Complexity Analysis
Let n be the length of the input string s.
•	Time complexity: O(n⋅logn)
o	We set the search space as [k, n], it takes at most O(logn) binary search steps.
o	At each step, we iterate over s which takes O(n) time.
•	Space complexity: O(n)
o	We need to update the boundary indices left and right.
o	During the iteration, we use a hash map counter which could contain at most O(n) distinct characters.

             */
            public int UsingBinarySearchWithFixedSizeSlidingWindow(String s, int k)
            {
                int n = s.Length;
                if (k >= n)
                {
                    return n;
                }

                int left = k, right = n;
                while (left < right)
                {
                    int mid = (left + right + 1) / 2;

                    if (IsValid(s, mid, k))
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                return left;
            }

            private bool IsValid(String s, int size, int k)
            {
                int n = s.Length;
                Dictionary<char, int> counter = new();

                for (int i = 0; i < size; i++)
                {
                    char c = s[i];
                    counter[c] = counter.GetValueOrDefault(c, 0) + 1;
                }

                if (counter.Count <= k)
                {
                    return true;
                }

                for (int i = size; i < n; i++)
                {
                    char c1 = s[i];
                    counter[c1] = counter.GetValueOrDefault(c1, 0) + 1;

                    char c2 = s[i - size];
                    counter[c2] = counter.GetValueOrDefault(c2, 0) + 1;

                    if (counter[c2] == 0)
                    {
                        counter.Remove(c2);
                    }
                    if (counter.Count <= k)
                    {
                        return true;
                    }
                }

                return false;
            }

            /* Approach 2: Sliding Window
            Complexity Analysis
Let n be the length of the input string s and k be the maximum number of distinct characters.
•	Time complexity: O(n)
o	In the iteration of the right boundary right, we shift it from 0 to n - 1. Although we may move the left boundary left in each step, left always stays to the left of right, which means left moves at most n - 1 times.
o	At each step, we update the value of an element in the hash map counter, which takes constant time.
o	To sum up, the overall time complexity is O(n).
•	Space complexity: O(k)
o	We need to record the occurrence of each distinct character in the valid window. During the iteration, there might be at most O(k+1) unique characters in the window, which takes O(k) space.

             */
            public int UsingSlidingWindow(String s, int k)
            {
                int n = s.Length;
                int maxSize = 0;
                Dictionary<char, int> counter = new();

                int left = 0;
                for (int right = 0; right < n; right++)
                {
                    counter[s[right]] = counter.GetValueOrDefault(s[right], 0) + 1;

                    while (counter.Count > k)
                    {
                        counter[s[left]] = counter[s[left]] - 1;
                        if (counter[s[left]] == 0)
                        {
                            counter.Remove(s[left]);
                        }
                        left++;
                    }

                    maxSize = Math.Max(maxSize, right - left + 1);
                }

                return maxSize;
            }
            /* Approach 3: Sliding Window II
            Complexity Analysis
Let n be the length of the input string s and k be the maximum number of distinct characters.
•	Time complexity: O(n)
o	In the iteration of the right boundary right, we shift it from 0 to n - 1.
o	At each step, we update the number of s[right] and/or the number of s[right - max_size] in the hash map counter, which takes constant time.
o	To sum up, the overall time complexity is O(n).
•	Space complexity: O(k)
o	We need to record the occurrence of each distinct character in the valid window. During the iteration, there might be at most O(k+1) unique characters in the window, which takes O(k) space.

             */
            public int UsingSlidingWindowII(string inputString, int k)
            {
                int stringLength = inputString.Length;
                int maximumSize = 0;
                Dictionary<char, int> characterCount = new Dictionary<char, int>();

                for (int right = 0; right < stringLength; right++)
                {
                    if (characterCount.ContainsKey(inputString[right]))
                    {
                        characterCount[inputString[right]]++;
                    }
                    else
                    {
                        characterCount[inputString[right]] = 1;
                    }

                    if (characterCount.Count <= k)
                    {
                        maximumSize++;
                    }
                    else
                    {
                        char leftChar = inputString[right - maximumSize];
                        characterCount[leftChar]--;

                        if (characterCount[leftChar] == 0)
                        {
                            characterCount.Remove(leftChar);
                        }
                    }
                }

                return maximumSize;
            }

        }

        /* 1062. Longest Repeating Substring
        https://leetcode.com/problems/longest-repeating-substring/description/
         */
        class LongestRepeatingSubstringSol
        {

            /* Approach 1: Brute Force with Set
            Complexity Analysis
Let n be the length of the string.
•	Time complexity: O(n^3)
The primary time-consuming operations are the nested loops and the substring extraction for every combination of start and end positions, which involves up to n^2 iterations, and each substring extraction and set operation takes O(n) time.
•	Space complexity: O(n^2)
O(n^2), as we may store up to O(n^2) substrings of various lengths in the set.

             */
            public int NaiveWithHashSet(String s)
            {
                HashSet<String> seenSubstrings = new HashSet<string>();
                int maxLength = s.Length - 1;

                for (int start = 0; start <= s.Length; start++)
                {
                    int end = start;
                    // If the remaining substring is shorter than maxLength,
                    // reset the loop
                    if (end + maxLength > s.Length)
                    {
                        if (--maxLength == 0) break;
                        start = -1;
                        seenSubstrings.Clear();
                        continue;
                    }
                    // Extract substring of length maxLength
                    String currentSubstring = s.Substring(end, end + maxLength);
                    // If the substring is already in the set,
                    // it means we've found a repeating substring
                    if (!seenSubstrings.Add(currentSubstring))
                    {
                        return maxLength;
                    }
                }
                return maxLength;
            }
            /* Approach 2: Brute Force with Incremental Search
            Complexity Analysis
Let n be the length of the string.
•	Time complexity: O(n^3)
For each possible starting index start, the algorithm generates substrings of length maxLength + 1. As maxLength increases, substring generation involves examining substrings of lengths up to n. The number of substrings generated can be up to O(n^2).
Each substring extraction takes O(n) time in the worst case because it involves copying a portion of the original string.
Given that each substring extraction is O(n) and there are up to O(n^2) substrings, the overall time complexity is O(n^3) due to the nested loops and substring operations.
•	Space complexity: O(n^2)
The set is used to store substrings that have been seen. In the worst case, the number of unique substrings stored can be up to O(n^2), and each substring can be up to length n. Thus, the space complexity for the set is O(n^2).

             */
            public int NaiveWithIncrementalSearch(String s)
            {
                int length = s.Length, maxLength = 0;
                HashSet<String> seenSubstrings = new HashSet<string>();

                for (int start = 0; start < length; start++)
                {
                    int end = start;
                    // Stop if it's not possible to find a longer repeating substring
                    if (end + maxLength >= length)
                    {
                        return maxLength;
                    }
                    // Generate substrings of length maxLength + 1
                    String currentSubstring = s.Substring(end, end + maxLength + 1);
                    // If a repeating substring is found, increase maxLength and restart
                    if (!seenSubstrings.Add(currentSubstring))
                    {
                        start = -1; // Restart search for new length
                        seenSubstrings.Clear();
                        maxLength++;
                    }
                }
                return maxLength;
            }

            /* Approach 3: Suffix Array with Sorting
Complexity Analysis
Let n be the length of the string.
•	Time complexity: O(n^2logn)
The time complexity for generating all suffixes is O(n^2) because we have to create n suffixes and each suffix, in the worst case, can be up to length n.
Sorting the suffixes involves comparing pairs of suffixes, each comparison taking up to O(n) time. Sorting n suffixes takes O(nlogn) time, resulting in an overall time complexity of O(n^2logn).
Comparing adjacent suffixes to find the longest common prefix takes up to O(n) time per comparison. With n suffixes, this step takes O(n^2) time.
Combining these, the overall time complexity is dominated by the sorting step, resulting in O(n^2logn).
•	Space complexity: O(n^2)
We store all n suffixes, each of which can be up to length n. This results in O(n^2) space for storing the suffixes.
Some extra space is used when we sort an array of size n in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has a space complexity of O(n)
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logn)
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn)
Apart from storing suffixes and sorting space, other variables and operations use O(1) space.
Thus, the overall space complexity remains O(n^2) due to the dominant factor being the storage of suffixes. However, the space used by the sorting algorithm (whether O(n), O(logn), or similar) adds to the total space usage, though it is less significant in comparison.

             */
            public int UsingSuffixArrayWithSorting(String s)
            {
                int length = s.Length;
                String[] suffixes = new String[length];

                // Create suffix array
                for (int i = 0; i < length; i++)
                {
                    suffixes[i] = s.Substring(i);
                }
                // Sort the suffixes
                Array.Sort(suffixes);

                int maxLength = 0;
                // Find the longest common prefix between consecutive sorted suffixes
                for (int i = 1; i < length; i++)
                {
                    int j = 0;
                    while (
                        j < Math.Min(suffixes[i].Length, suffixes[i - 1].Length) &&
                        suffixes[i][j] == suffixes[i - 1][j]
                    )
                    {
                        j++;
                    }
                    maxLength = Math.Max(maxLength, j);
                }
                return maxLength;
            }
            /* 
            Approach 4: Binary Search with Set
            Complexity Analysis
Let n be the length of string.
•	Time complexity: O(n^2logn)
O(n^2logn), where logn comes from the binary search and O(n^2) from the set operations for each substring length check.
•	Space complexity: O(n^2)
O(n^2), for storing substrings in the set.

 */
            public int UsingBinarySearchWithHashSet(String s)
            {
                char[] characters = s.ToCharArray();
                int start = 1, end = characters.Length - 1;

                while (start <= end)
                {
                    int mid = (start + end) / 2;
                    // Check if there's a repeating substring of length mid
                    if (HasRepeatingSubstring(characters, mid))
                    {
                        start = mid + 1;
                    }
                    else
                    {
                        end = mid - 1;
                    }
                }
                return start - 1;
            }

            private bool HasRepeatingSubstring(char[] characters, int length)
            {
                HashSet<String> seenSubstrings = new HashSet<string>();
                // Check for repeating substrings of given length
                for (int i = 0; i <= characters.Length - length; i++)
                {
                    String substring = new String(characters, i, length);
                    if (!seenSubstrings.Add(substring))
                    {
                        return true;
                    }
                }
                return false;
            }

            /* Approach 5: Dynamic Programming
            Complexity Analysis
Let n be the length of the string.
•	Time complexity: O(n^2)
The nested loops each run up to n times, filling in the DP table.
•	Space complexity: O(n^2)
O(n^2), for the DP table used to store the lengths of common substrings.

             */
            public int UsingDP(String s)
            {
                int length = s.Length;
                int[][] dp = new int[length + 1][];
                int maxLength = 0;

                // Use DP to find the longest common substring
                for (int i = 1; i <= length; i++)
                {
                    for (int j = i + 1; j <= length; j++)
                    {
                        // If characters match, extend the length of
                        // the common substring
                        if (s[i - 1] == s[j - 1])
                        {
                            dp[i][j] = dp[i - 1][j - 1] + 1;
                            maxLength = Math.Max(maxLength, dp[i][j]);
                        }
                    }
                }
                return maxLength;
            }

            /* Approach 6: MSD Radix  (Most Significant Digit Radix Sort)
Complexity Analysis
Let n be the length of the string.
•	Time complexity: O(n^2)
The main operations are creating the suffix array, sorting it using MSD Radix Sort, and then comparing consecutive suffixes, each taking O(n^2) time in the worst case.
•	Space complexity: O(n^2)
O(n^2), for the storage of the suffixes and auxiliary arrays during the sorting process.

             */
            public int UsingMSDRadix(String s)
            {
                int length = s.Length;
                String[] suffixes = new String[length];

                // Create suffix array
                for (int i = 0; i < length; i++)
                {
                    suffixes[i] = s.Substring(i);
                }
                // Sort the suffix array using MSD Radix Sort
                MsdRadixSort(suffixes);

                int maxLength = 0;
                // Find the longest common prefix between consecutive sorted suffixes
                for (int i = 1; i < length; i++)
                {
                    int j = 0;
                    while (
                        j < Math.Min(suffixes[i].Length, suffixes[i - 1].Length) &&
                        suffixes[i][j] == suffixes[i - 1][j])
                    {
                        j++;
                    }
                    maxLength = Math.Max(maxLength, j);
                }
                return maxLength;
            }

            // Main method to perform MSD Radix Sort
            private void MsdRadixSort(String[] input)
            {
                Sort(input, 0, input.Length - 1, 0, new String[input.Length]);
            }

            // Helper method for sorting
            private void Sort(String[] input, int lo, int hi, int depth, String[] aux)
            {
                if (lo >= hi) return;

                int[] count = new int[28];
                for (int i = lo; i <= hi; i++)
                {
                    count[CharAt(input[i], depth) + 1]++;
                }
                for (int i = 1; i < 28; i++)
                {
                    count[i] += count[i - 1];
                }
                for (int i = lo; i <= hi; i++)
                {
                    aux[count[CharAt(input[i], depth)]++] = input[i];
                }
                for (int i = lo; i <= hi; i++)
                {
                    input[i] = aux[i - lo];
                }
                for (int i = 0; i < 27; i++)
                {
                    Sort(input, lo + count[i], lo + count[i + 1] - 1, depth + 1, aux);
                }
            }

            // Returns the character value or 0 if index exceeds string length
            private int CharAt(String s, int index)
            {
                if (index >= s.Length) return 0;
                return s[index] - 'a' + 1;
            }

        }


        /* 1208. Get Equal Substrings Within Budget
        https://leetcode.com/problems/get-equal-substrings-within-budget/description/

         */
        class GetEqualSubstringsWithinBudgetSol
        {

            /*
Approach: Sliding Window

            Complexity Analysis
            Here, N is the length of the strings s and t.
            •	Time complexity: O(N)
            We will process each index of s and t at most twice. This is because we iterate over the character while extending the window from the right side, and again while contracting the window from the left end. Therefore, the total time complexity is equal to O(N).
            •	Space complexity: O(1)
            We do not need any extra space apart from some variables, and hence, the space complexity is constant.
             */
            public int UsingSlidingWindow(String s, String t, int maxCost)
            {
                int N = s.Length;

                int maxLen = 0;
                // Starting index of the current substring
                int start = 0;
                // Cost of converting the current substring in s to t
                int currCost = 0;

                for (int i = 0; i < N; i++)
                {
                    // Add the current index to the substring
                    currCost += Math.Abs(s[i] - t[i]);

                    // Remove the indices from the left end till the cost becomes less than or equal to maxCost
                    while (currCost > maxCost)
                    {
                        currCost -= Math.Abs(s[start] - t[start]);
                        start++;
                    }

                    maxLen = Math.Max(maxLen, i - start + 1);
                }

                return maxLen;
            }
        }


        /* 1839. Longest Substring Of All Vowels in Order
        https://leetcode.com/problems/longest-substring-of-all-vowels-in-order/description/
        https://algo.monster/liteproblems/1839
         */
        class LongestBeautifulSubstringSol
        {
            /* Time and Space Complexity
            Time Complexity
            The time complexity of the given code can be analyzed in the following steps:
            1.	Constructing the arr list: This involves a single pass through the input string word with a pair of pointers i and j. For each unique character in the word, the loop checks for consecutive occurrences and adds a tuple (character, count) to arr. This operation has a time complexity of O(n) where n is the length of the input string since each character is considered exactly once.
            2.	Looping through arr for finding the longest beautiful substring: The second loop runs with an upper limit of len(arr) - 4, and for each iteration, it checks a fixed sequence of 5 elements (not considering nested loops). The check and max call are O(1) operations. The number of iterations depends on the number of unique characters in word, but since it's strictly less than n, the loop has a time complexity of O(n).
            Combining both parts, the overall time complexity is O(n) + O(n) = O(n).
            Space Complexity
            The space complexity is determined by additional space used apart from the input:
            1.	The arr list: In the worst case, if every character in word is unique, arr would have n tuples. Therefore, the space complexity due to arr is O(n).
            2.	Constant space for variables i, j, and ans, which doesn't depend on the size of the input.
            Hence, the overall space complexity of the code is O(n).
             */
            // Method to find the length of the longest beautiful substring in the input string
            public int LongestBeautifulSubstring(String word)
            {
                int wordLength = word.Length; // Store the length of the word
                List<CharGroup> charGroups = new(); // List to store groups of consecutive identical characters

                // Loop through the string and group consecutive identical characters
                for (int i = 0; i < wordLength;)
                {
                    int j = i;
                    // Find the end index of the group of identical characters
                    while (j < wordLength && word[j] == word[i])
                    {
                        ++j;
                    }
                    // Add the group to the list
                    charGroups.Add(new CharGroup(word[i], j - i));
                    i = j; // Move to the next group
                }

                int maxBeautyLength = 0; // Variable to track the maximum length of a beautiful substring

                // Iterate through the list of char groups to find the longest beautiful substring
                for (int i = 0; i < charGroups.Count - 4; ++i)
                {
                    // Get five consecutive char groups
                    CharGroup a = charGroups[i],
                              b = charGroups[i + 1],
                              c = charGroups[i + 2],
                              d = charGroups[i + 3],
                              e = charGroups[i + 4];

                    // Check if the groups form a sequence 'a', 'e', 'i', 'o', 'u'
                    if (a.Character == 'a' && b.Character == 'e' && c.Character == 'i'
                        && d.Character == 'o' && e.Character == 'u')
                    {
                        // Calculate the total length of the beautiful substring and update the max length
                        maxBeautyLength = Math.Max(maxBeautyLength, a.Count + b.Count + c.Count + d.Count + e.Count);
                    }
                }

                return maxBeautyLength; // Return the maximum length found
            }
            // Helper class to represent a group of consecutive identical characters
            public class CharGroup
            {
                public char Character; // The character in the group
                public int Count;      // The count of how many times the character is repeated

                // Constructor for the helper class
                public CharGroup(char character, int count)
                {
                    this.Character = character;
                    this.Count = count;
                }
            }
        }

        /* 2950. Number of Divisible Substrings
        https://leetcode.com/problems/number-of-divisible-substrings/description/
        https://algo.monster/liteproblems/2950	
         */
        class CountDivisibleSubstringsSol
        {
            /*             Time and Space Complexity
            Time Complexity
            The time complexity of the given code is O(n^2). This is because there are two nested loops. The outer loop runs for n iterations (n being the length of the word), and for each iteration of the outer loop, the inner loop runs for at most n iterations - starting from the current index of the outer loop to the end of the word. During each iteration of the inner loop, a constant number of operations are executed. So, for each element, we potentially loop through every other element to the right of it, leading to the n * (n-1) / 2 term, which simplifies to O(n^2).
            Space Complexity
            The space complexity of the code is O(C) where C is the size of the character set. In this case, C=26 as there are 26 lowercase English letters. The space is used to store the mapping of each character to its associated integer, which in this instance does not change with the size of the input string and is hence constant.
             */
            public int CountDivisibleSubstrings(String word)
            {
                // Array of strings representing groups of characters
                String[] groups = { "ab", "cde", "fgh", "ijk", "lmn", "opq", "rst", "uvw", "xyz" };
                // Mapping for characters to their respective group values
                int[] mapping = new int[26];

                // Initialize the mapping for each character to its group value
                for (int i = 0; i < groups.Length; ++i)
                {
                    foreach (char c in groups[i])
                    {
                        mapping[c - 'a'] = i + 1;
                    }
                }

                // Initialize count of divisible substrings
                int count = 0;
                int length = word.Length;

                // Iterate over all possible starting points of substrings
                for (int i = 0; i < length; ++i)
                {
                    // 'sum' will hold the sum of the group values for the current substring
                    int sum = 0;
                    // Iterate over all possible ending points of substrings
                    for (int j = i; j < length; ++j)
                    {
                        // Add group value of the current character to 'sum'
                        sum += mapping[word[j] - 'a'];
                        // Increment the count if sum is divisible by the length of the substring
                        count += sum % (j - i + 1) == 0 ? 1 : 0;
                    }
                }

                // Return the total count of divisible substrings
                return count;
            }
        }






























    }
}