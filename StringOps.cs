using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class StringOps
    {


        //https://www.algoexpert.io/questions/knuth-morris-pratt-algorithm KMP
        public static bool IsBigStringContainsSmallStringUsingKMP(string str, string substring)
        {
        //T:O(n+m) | S:O(m)
        TODO:
            int[] pattern = BuildPattern(substring);
            return IsBigStringContainsSmallStringUsingKMP(str, substring, pattern);
        }

        private static bool IsBigStringContainsSmallStringUsingKMP(string str, string substring, int[] pattern)
        {
            int i = 0, j = 0;
            while (i + substring.Length - j <= str.Length)
            {

                if (str[i] == substring[j])
                {
                    if (j == substring.Length - 1) return true;
                    i++; j++;
                }
                else if (j > 0)
                {
                    j = pattern[j - 1] + 1;
                }
                else i++;
            }
            return false;
        }

        private static int[] BuildPattern(string substring)
        {
            int[] pattern = new int[substring.Length];
            Array.Fill(pattern, -1);
            int j = 0, i = 1;

            while (i < substring.Length)
            {
                if (substring[i] == substring[j])
                {
                    pattern[i] = j;
                    i++;
                    j++;
                }
                else if (j > 0)
                {
                    j = pattern[j - 1] + 1;
                }
                else i++;

            }
            return pattern;
        }
        //https://www.algoexpert.io/questions/reverse-words-in-string
        public static string ReverseWordsInString(string str)
        {

            //1. T:O(n) | S:O(n)
            string reversedStr = ReverseWordsInString1(str);

            //2. T:O(n) | S:O(n) -- Reverse entire string and re-reverse each word

            //3. T:O(n) | S:O(n) using Stack
            reversedStr = ReverseWordsInStringWithStack(str);

            return reversedStr;
        }

        private static string ReverseWordsInStringWithStack(string str)
        {
            Stack<string> reverseStack = new Stack<string>();
            while (str.Length > 0)
            {

                int idxSpace = str.IndexOf(" ");
                if (idxSpace >= 0)
                {
                    string str1 = str.Substring(0, idxSpace);
                    if (str1.Length != 0)
                        reverseStack.Push(str1);
                    reverseStack.Push(" ");
                    str = str.Substring(idxSpace + 1);
                    //len=str.Length;                                                     
                }
                else
                {
                    reverseStack.Push(str);
                    str = "";
                }
            }
            StringBuilder reverStr = new StringBuilder();
            while (reverseStack.Count > 0)
            {
                string s1 = reverseStack.Pop();
                reverStr.Append(s1);

            }
            return reverStr.ToString();

        }

        private static string ReverseWordsInString1(string str)
        {
            List<string> words = new List<string>();
            int startOfWord = 0;

            for (int idx = 0; idx < str.Length; idx++)
            {
                char c = str[idx];
                if (c == ' ')
                {
                    words.Add(str.Substring(startOfWord, idx - startOfWord));
                    startOfWord = idx;
                }
                else if (str[startOfWord] == ' ')
                {
                    words.Add(" ");
                    startOfWord = idx;
                }

            }
            words.Add(str.Substring(startOfWord));
            words.Reverse();
            return string.Join("", words);
        }
        //https://www.algoexpert.io/questions/semordnilap
        public static List<List<string>> Semordnilap(string[] words)
        {

            //T:O(n*m) | S:O(n*m) where n represents total words and m is length of longest word
            HashSet<string> wordsSet = new HashSet<string>(words);
            List<List<string>> semoPairs = new List<List<string>>();

            foreach (var word in words)
            { // O(n)
                char[] chars = word.ToCharArray();
                Array.Reverse(chars); // O(m)
                string reverse = new string(chars);
                if (wordsSet.Contains(reverse) && (!word.Equals(reverse)))
                {
                    List<string> semoPair = new List<string> { word, reverse };
                    semoPairs.Add(semoPair);
                    wordsSet.Remove(word);
                    wordsSet.Remove(reverse);

                }
            }
            return semoPairs;
        }
        //https://www.algoexpert.io/questions/levenshtein-distance
        //Minimum Edit operations(insert/delete/replace) required
        public static int LevenshteinDistiance(string str1, string str2)
        {

            //1.Using full-blown DP(dyn pro) table 
            //T:O(nm) | S:O(nm)
            int numMinEditOps = LevenshteinDistiance1(str1, str2);

            //2.Using full-blown DP(dyn pro) table 
            //T:O(nm) | S:O(min(n,m))
            numMinEditOps = LevenshteinDistianceOptimal(str1, str2);

            return numMinEditOps;
        }

        private static int LevenshteinDistianceOptimal(string str1, string str2)
        {
            string small = str1.Length < str2.Length ? str1 : str2;
            string big = str1.Length >= str2.Length ? str2 : str1;

            int[] evenEdits = new int[small.Length + 1];
            int[] oddEdits = new int[small.Length + 1];
            for (int j = 0; j < small.Length; j++)
            {
                evenEdits[j] = j;
            }
            int[] currentEdit, prevEdit;
            for (int i = 1; i < big.Length + 1; i++)
            {
                if (i % 2 == 1)
                {
                    currentEdit = oddEdits;
                    prevEdit = evenEdits;
                }
                else
                {
                    currentEdit = evenEdits;
                    prevEdit = oddEdits;
                }
                currentEdit[0] = i;
                for (int j = 1; j < small.Length + 1; j++)
                {
                    if (big[j - 1] == small[j - 1])
                    {
                        currentEdit[j] = prevEdit[j - 1];
                    }
                    else
                    {
                        currentEdit[j] = 1 + Math.Min(prevEdit[j - 1],
                        Math.Min(prevEdit[j], currentEdit[j - 1]));
                    }
                }
            }
            return big.Length % 2 == 0 ? evenEdits[small.Length] : oddEdits[small.Length];
        }

        private static int LevenshteinDistiance1(string str1, string str2)
        {
            int[,] edits = new int[str2.Length + 1, str1.Length + 1];
            for (int row = 0; row < str2.Length + 1; row++)
            {
                for (int col = 0; col < str1.Length + 1; col++)
                {
                    edits[row, col] = col;
                }
                edits[row, 0] = row;
            }
            for (int row = 1; row < str2.Length + 1; row++)
            {
                for (int col = 1; col < str1.Length + 1; col++)
                {
                    if (str2[row - 1] == str1[col - 1])
                    {
                        edits[row, col] = edits[row - 1, col - 1];
                    }
                    else
                    {
                        edits[row, col] = 1 + Math.Min(edits[row - 1, col - 1],
                                            Math.Min(edits[row - 1, col], edits[row, col - 1]));
                    }

                }
            }
            return edits[str2.Length, str1.Length];

        }
        //https://www.algoexpert.io/questions/underscorify-substring
        //Merge Intervals/ Overlap intervals
        public static string UnderscorifySubstring(string str, string substring)
        {
            //Average case - T:O(n+m) | S:O(n) - n is length of main string and m is length of substring
            List<int[]> locations = MergeIntervals(GetLocations(str, substring));

            return Underscorify(str, locations);
        }

        private static string Underscorify(string str, List<int[]> locations)
        {
            int locationIdx = 0, strIdx = 0;
            bool inBetweenUnderscores = false;
            List<string> finalChars = new List<string>();
            int i = 0;
            while (strIdx < str.Length && locationIdx < locations.Count)
            {
                if (strIdx == locations[locationIdx][i])
                {
                    finalChars.Add("_");

                    inBetweenUnderscores = !inBetweenUnderscores;

                    if (!inBetweenUnderscores) locationIdx++;

                    i = i == 1 ? 0 : 1;
                }
                finalChars.Add(str[strIdx].ToString());
                strIdx += 1;
            }
            if (locationIdx < locations.Count)
            { // substring found at the end of main string
                finalChars.Add("_");
            }
            else if (strIdx < str.Length) //Adding remaining main string
                finalChars.Add(str.Substring(strIdx));

            return String.Join("", finalChars);

        }

        private static List<int[]> MergeIntervals(List<int[]> locations)
        {
            if (locations.Count == 0) return locations;

            List<int[]> newLocations = new List<int[]>();
            newLocations.Add(locations[0]);
            int[] previous = newLocations[0];
            for (int i = 1; i < locations.Count; i++)
            {
                int[] current = locations[i];
                if (current[0] <= previous[1]) //Overlap check
                    previous[1] = current[1];
                else
                {
                    newLocations.Add(current);
                    previous = current;
                }
            }
            return newLocations;
        }

        private static List<int[]> GetLocations(string str, string substring)
        {
            List<int[]> locations = new List<int[]>();
            int startIdx = 0;
            while (startIdx < str.Length)
            {
                int nextIdx = str.IndexOf(substring, startIdx); //O(n+m)
                if (nextIdx != -1)
                {
                    locations.Add(new int[] { nextIdx, nextIdx + substring.Length });
                    startIdx = nextIdx + 1;
                }
                else
                {
                    break;
                }
            }
            return locations;
        }
        //https://www.algoexpert.io/questions/longest-string-chain
        public class StringChain
        {
            public string nextstring;
            public int maxChainLength;

            public StringChain(string nextstring, int maxChainLength)
            {
                this.nextstring = nextstring;
                this.maxChainLength = maxChainLength;
            }
        }

        // O(n * m^2 + nlog(n)) time | O(nm) space - where n is the number of strings
        public static void FindLongeststringChain(
            string str, Dictionary<string, StringChain> stringChains
        )
        {
            // Try removing every letter of the current string to see if the
            // remaining strings form a string chain.
            for (int i = 0; i < str.Length; i++)
            {
                string smallerstring = GetSmallerstring(str, i);
                if (!stringChains.ContainsKey(smallerstring)) continue;
                TryUpdateLongeststringChain(str, smallerstring, stringChains);
            }
        }

        public static string GetSmallerstring(string str, int index)
        {
            return str.Substring(0, index) + str.Substring(index + 1);
        }

        public static void TryUpdateLongeststringChain(
            string currentstring,
            string smallerstring,
            Dictionary<string, StringChain> stringChains
        )
        {
            int smallerstringChainLength = stringChains[smallerstring].maxChainLength;
            int currentstringChainLength = stringChains[currentstring].maxChainLength;
            // Update the string chain of the current string only if the smaller string
            // leads to a longer string chain.
            if (smallerstringChainLength + 1 > currentstringChainLength)
            {
                stringChains[currentstring].maxChainLength = smallerstringChainLength + 1;
                stringChains[currentstring].nextstring = smallerstring;
            }
        }

        public static List<string> buildLongeststringChain(
            List<string> strings, Dictionary<string, StringChain> stringChains
        )
        {
            // Find the string that starts the longest string chain.
            int maxChainLength = 0;
            string chainStartingstring = "";
            foreach (string str in strings)
            {
                if (stringChains[str].maxChainLength > maxChainLength)
                {
                    maxChainLength = stringChains[str].maxChainLength;
                    chainStartingstring = str;
                }
            }

            // Starting at the string found above, build the longest string chain.
            List<string> ourLongeststringChain = new List<string>();
            string currentstring = chainStartingstring;
            while (currentstring != "")
            {
                ourLongeststringChain.Add(currentstring);
                currentstring = stringChains[currentstring].nextstring;
            }

            return ourLongeststringChain.Count == 1 ? new List<string>()
                                                    : ourLongeststringChain;
        }

        /*
        97. Interleaving String
        https://leetcode.com/problems/interleaving-string/description/

        https://www.algoexpert.io/questions/interweaving-strings


        */
        public class InterleavingStringsSol
        {
            /*
            
            Approach 1: Brute Force
Complexity Analysis
•	Time complexity : O(2^(m+n)). m is the length of s1 and n is the length of s2.
•	Space complexity : O(m+n). The size of stack for recursive calls can go upto m+n.

            */

            public static bool Naive(string one, string two, string three)
            {
                if (three.Length != one.Length + two.Length)
                {
                    return false;
                }

                return AreInterwoven(one, two, three, 0, 0);
            }

            public static bool AreInterwoven(
              string s1, string s2, string s3, int i = 0, int j = 0, string res = ""
            )
            {
                // If result matches with third string and we have reached the end of
                // the all strings, return true.
                if (res == s3 && i == s1.Length && j == s2.Length)
                    return true;
                bool ans = false;
                // Recurse for s1 & s2 if "ans" is false
                if (i < s1.Length)
                    ans |= AreInterwoven(s1, s2, s3, i + 1, j, res + s1[i]);
                if (j < s2.Length)
                    ans |= AreInterwoven(s1, s2, s3, i, j + 1, res + s2[j]);
                return ans;
            }
            /*
            Approach 2: Recursion with memoization (RecMem)
            
            Complexity Analysis
•	Time complexity: O(m⋅n), where
m is the length of s1 and n is the length of s2.
That's a consequence of the fact that each (i, j) combination is computed only once.
•	Space complexity: O(m⋅n) to keep double array memo.

            */

            public static bool RecMemo(string one, string two, string three)
            {
                if (three.Length != one.Length + two.Length)
                {
                    return false;
                }

                bool?[,] cache = new bool?[one.Length + 1, two.Length + 1];
                return AreInterwoven(one, two, three, 0, 0, cache);
            }

            private static bool AreInterwoven(
              string one, string two, string three, int i, int j, bool?[,] cache
            )
            {
                if (cache[i, j].HasValue)
                {
                    return cache[i, j].Value;
                }

                int k = i + j;
                if (k == three.Length)
                {
                    return true;
                }

                if (i < one.Length && one[i] == three[k])
                {
                    cache[i, j] = AreInterwoven(one, two, three, i + 1, j, cache);
                    if (cache[i, j].HasValue && cache[i, j].Value)
                    {
                        return true;
                    }
                }

                if (j < two.Length && two[j] == three[k])
                {
                    cache[i, j] = AreInterwoven(one, two, three, i, j + 1, cache);
                    return cache[i, j].Value;
                }

                cache[i, j] = false;
                return false;
            }

            /*            
Approach 3: Using 2D Dynamic Programming
Complexity Analysis
•	Time complexity : O(m⋅n). dp array of size m∗n is filled.
•	Space complexity : O(m⋅n). 2D dp of size (m+1)∗(n+1) is required. m and n are the lengths of strings s1 and s2 respectively.
            
            */
            public bool DP2D(string s1, string s2, string s3)
            {
                if (s3.Length != s1.Length + s2.Length)
                {
                    return false;
                }

                bool[,] dp = new bool[s1.Length + 1, s2.Length + 1];
                for (int i = 0; i <= s1.Length; i++)
                {
                    for (int j = 0; j <= s2.Length; j++)
                    {
                        if (i == 0 && j == 0)
                        {
                            dp[i, j] = true;
                        }
                        else if (i == 0)
                        {
                            dp[i, j] = dp[i, j - 1] && s2[j - 1] == s3[i + j - 1];
                        }
                        else if (j == 0)
                        {
                            dp[i, j] = dp[i - 1, j] && s1[i - 1] == s3[i + j - 1];
                        }
                        else
                        {
                            dp[i, j] = (dp[i - 1, j] && s1[i - 1] == s3[i + j - 1]) ||
                                       (dp[i, j - 1] && s2[j - 1] == s3[i + j - 1]);
                        }
                    }
                }

                return dp[s1.Length, s2.Length];
            }

            /*
            
Approach 4: Using 1D Dynamic Programming
Complexity Analysis
•	Time complexity : O(m⋅n). dp array of size n is filled m times.
•	Space complexity : O(n). n is the length of the string s1.
            */
            public bool DP1D(string s1, string s2, string s3)
            {
                if (s3.Length != s1.Length + s2.Length)
                {
                    return false;
                }

                bool[] dp = new bool[s2.Length + 1];
                for (int i = 0; i <= s1.Length; i++)
                {
                    for (int j = 0; j <= s2.Length; j++)
                    {
                        if (i == 0 && j == 0)
                        {
                            dp[j] = true;
                        }
                        else if (i == 0)
                        {
                            dp[j] = dp[j - 1] && s2[j - 1] == s3[i + j - 1];
                        }
                        else if (j == 0)
                        {
                            dp[j] = dp[j] && s1[i - 1] == s3[i + j - 1];
                        }
                        else
                        {
                            dp[j] = (dp[j] && s1[i - 1] == s3[i + j - 1]) ||
                                    (dp[j - 1] && s2[j - 1] == s3[i + j - 1]);
                        }
                    }
                }

                return dp[s2.Length];
            }

        }


        //https://www.algoexpert.io/questions/first-non-repeating-character
        // O(n^2) time | O(1) space - where n is the length of the input string
        public int FirstNonRepeatingCharacterNaive(string str)
        {
            for (int idx = 0; idx < str.Length; idx++)
            {
                bool foundDuplicate = false;
                for (int idx2 = 0; idx2 < str.Length; idx2++)
                {
                    if (str[idx] == str[idx2] && idx != idx2)
                    {
                        foundDuplicate = true;
                    }
                }

                if (!foundDuplicate) return idx;
            }

            return -1;
        }
        // O(n) time | O(1) space - where n is the length of the input string
        // The constant space is because the input string only has lowercase
        // English-alphabet letters; thus, our hash table will never have more
        // than 26 character frequencies.
        public int FirstNonRepeatingCharacterOptimal(string str)
        {
            Dictionary<char, int> characterFrequencies = new Dictionary<char, int>();

            for (int idx = 0; idx < str.Length; idx++)
            {
                char character = str[idx];
                characterFrequencies[character] =
                  characterFrequencies.GetValueOrDefault(character, 0) + 1;
            }

            for (int idx = 0; idx < str.Length; idx++)
            {
                char character = str[idx];
                if (characterFrequencies[character] == 1)
                {
                    return idx;
                }
            }

            return -1;
        }





        //https://www.algoexpert.io/questions/one-edit
        // O(n + m) time | O(n + m) space - where n is the length of stringOne, and
        // m is the length of stringTwo
        public bool OneEditNaive(string stringOne, string stringTwo)
        {
            int lengthOne = stringOne.Length;
            int lengthTwo = stringTwo.Length;
            if (Math.Abs(lengthOne - lengthTwo) > 1)
            {
                return false;
            }

            for (int i = 0; i < Math.Min(lengthOne, lengthTwo); i++)
            {
                if (stringOne[i] != stringTwo[i])
                {
                    if (lengthOne > lengthTwo)
                    {
                        return stringOne.Substring(i + 1).Equals(stringTwo.Substring(i));
                    }
                    else if (lengthTwo > lengthOne)
                    {
                        return stringOne.Substring(i).Equals(stringTwo.Substring(i + 1));
                    }
                    else
                    {
                        return stringOne.Substring(i + 1).Equals(stringTwo.Substring(i + 1));
                    }
                }
            }
            return true;
        }
        // O(n) time | O(1) space - where n is the length of the shorter string
        public bool OneEditOptimal(string stringOne, string stringTwo)
        {
            int lengthOne = stringOne.Length;
            int lengthTwo = stringTwo.Length;
            if (Math.Abs(lengthOne - lengthTwo) > 1)
            {
                return false;
            }

            bool madeEdit = false;
            int indexOne = 0;
            int indexTwo = 0;

            while (indexOne < lengthOne && indexTwo < lengthTwo)
            {
                if (stringOne[indexOne] != stringTwo[indexTwo])
                {
                    if (madeEdit)
                    {
                        return false;
                    }
                    madeEdit = true;

                    if (lengthOne > lengthTwo)
                    {
                        indexOne++;
                    }
                    else if (lengthTwo > lengthOne)
                    {
                        indexTwo++;
                    }
                    else
                    {
                        indexOne++;
                        indexTwo++;
                    }
                }
                else
                {
                    indexOne++;
                    indexTwo++;
                }
            }

            return true;
        }





        /*
        6. Zigzag Conversion
        https://leetcode.com/problems/zigzag-conversion/description/	

        */
        public string ZigZagConvert(string s, int numRows)
        {

            /*

Approach 1: Simulate Zig-Zag Movement
Complexity Analysis
Here, n is the length of the input string, and numRows is the number of rows of the zig-zag pattern.
•	Time complexity: O(numRows⋅n).
o	We initialize an empty 2-dimensional matrix of size numRows×numCols, where O(numCols)=O(n). So it takes O(numRows⋅n) time.
o	Then we iterate on the input string, which takes O(n) time, and again traverse on the matrix to generate the final string, which takes O(numRows⋅n) time.
o	Thus, overall we take O(2(numRows⋅n)+n)=O(numRows⋅n) time.
•	Space complexity: O(numRows⋅n).
o	We use an additional 2-dimensional array of size numRows×numCols, where O(numCols)=O(n)

            
            */
            string result = ZigZagConvertNaive(s, numRows);
            /*
 Approach 2: String Traversal

 Complexity Analysis
Here, n is the length of the input string.
•	Time complexity: O(n).
o	We iterate over each index of the input only once and perform constant work at each index.
•	Space complexity: O(1).
o	We have not used any additional space other than for building the output, which is not counted.
           
            
            */
            result = ZigZagConvertNaive(s, numRows);

            return result;


        }
        public string ZigZagConvertOptimal(string s, int numRows)
        {
            if (numRows == 1)
            {
                return s;
            }

            StringBuilder answer = new StringBuilder();
            int n = s.Length;
            int charsInSection = 2 * (numRows - 1);

            for (int currRow = 0; currRow < numRows; ++currRow)
            {
                int index = currRow;

                while (index < n)
                {
                    answer.Append(s[index]);

                    // If currRow is not the first or last row
                    // then we have to add one more character of current section.
                    if (currRow != 0 && currRow != numRows - 1)
                    {
                        int charsInBetween = charsInSection - 2 * currRow;
                        int secondIndex = index + charsInBetween;

                        if (secondIndex < n)
                        {
                            answer.Append(s[secondIndex]);
                        }
                    }

                    // Jump to same row's first character of next section.
                    index += charsInSection;
                }
            }

            return answer.ToString();
        }
        private string ZigZagConvertNaive(string s, int numRows)
        {
            if (numRows == 1)
            {
                return s;
            }

            int n = s.Length;
            int sections = (int)Math.Ceiling(n / (2 * numRows - 2.0));
            int numCols = sections * (numRows - 1);

            char[][] matrix = new char[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                matrix[i] = new char[numCols];
                matrix[i] = Enumerable.Repeat(' ', numCols).ToArray();
            }

            int currRow = 0, currCol = 0;
            int currStringIndex = 0;

            // Iterate in zig-zag pattern on matrix and fill it with string
            // characters.
            while (currStringIndex < n)
            {
                // Move down.
                while (currRow < numRows && currStringIndex < n)
                {
                    matrix[currRow][currCol] = s[currStringIndex];
                    currRow++;
                    currStringIndex++;
                }

                currRow -= 2;
                currCol++;

                // Move up (with moving right also).
                while (currRow > 0 && currCol < numCols && currStringIndex < n)
                {
                    matrix[currRow][currCol] = s[currStringIndex];
                    currRow--;
                    currCol++;
                    currStringIndex++;
                }
            }

            string answer = "";
            foreach (char[] row in matrix)
            {
                string rowStr = new string(row).Replace(" ", "");
                answer += rowStr;
            }

            return answer;
        }

        /*
        7. Reverse Integer
        https://leetcode.com/problems/reverse-integer/description/
        
        Approach 1: Pop and Push Digits & Check before Overflow

Complexity Analysis
•	Time Complexity: O(log(x)). There are roughly log10(x) digits in x.
•	Space Complexity: O(1).

        */
        public int ReverseInteger(int x)
        {
            int rev = 0;
            while (x != 0)
            {
                int pop = x % 10;
                x /= 10;
                if (rev > int.MaxValue / 10 ||
                    (rev == int.MaxValue / 10 && pop > 7))
                    return 0;
                if (rev < int.MinValue / 10 ||
                    (rev == int.MinValue / 10 && pop < -8))
                    return 0;
                rev = rev * 10 + pop;
            }

            return rev;
        }

        /*
        8. String to Integer (atoi)
https://leetcode.com/problems/string-to-integer-atoi/description/	

        */
        public int ConvertStringToInt(string s)
        {

            /*
            Approach 1: Follow the Rules
    Complexity Analysis
    If N is the number of characters in the input string.
    •	Time complexity: O(N)
    We visit each character in the input at most once and for each character we spend a constant amount of time.
    •	Space complexity: O(1)
    We have used only constant space to store the sign and the result.
            */
            int result = ConvertStringToIntUsingRules(s);
            /*
            Approach 2: Deterministic Finite Automaton (DFA)
    Complexity Analysis
    If N is the number of characters in the input string.
    •	Time complexity: O(N)
    We iterate over the input string exactly once, and each state transition only requires constant time.
    •	Space complexity: O(1)
    We have used only constant space to store the state, sign, and result.

            */
            result = ConvertStringToIntDFA(s);

            return result;

        }
        public int ConvertStringToIntDFA(string s)
        {
            StateMachine Q = new StateMachine();
            for (int i = 0; i < s.Length && Q.getState() != State.qd; ++i)
            {
                Q.transition(s[i]);
            }

            return Q.getInteger();

        }
        public enum State { q0, q1, q2, qd }

        public class StateMachine
        {
            private State currentState;
            private long result;
            private int sign;

            public StateMachine()
            {
                currentState = State.q0;
                result = 0;
                sign = 1;
            }

            private void toStateQ1(char ch)
            {
                sign = (ch == '-') ? -1 : 1;
                currentState = State.q1;
            }

            private void toStateQ2(int digit)
            {
                currentState = State.q2;
                appendDigit(digit);
            }

            private void toStateQd()
            {
                currentState = State.qd;
            }

            public void appendDigit(int digit)
            {
                if ((result > int.MaxValue / 10) ||
                    (result == int.MaxValue / 10 && digit > int.MaxValue % 10))
                {
                    if (sign == 1)
                    {
                        result = int.MaxValue;
                    }
                    else
                    {
                        result = int.MinValue;
                        sign = 1;
                    }

                    toStateQd();
                }
                else
                {
                    result = result * 10 + digit;
                }
            }

            public void transition(char ch)
            {
                if (currentState == State.q0)
                {
                    if (ch == ' ')
                    {
                        return;
                    }
                    else if (ch == '-' || ch == '+')
                    {
                        toStateQ1(ch);
                    }
                    else if (char.IsDigit(ch))
                    {
                        toStateQ2(ch - '0');
                    }
                    else
                    {
                        toStateQd();
                    }
                }
                else if (currentState == State.q1 || currentState == State.q2)
                {
                    if (char.IsDigit(ch))
                    {
                        toStateQ2(ch - '0');
                    }
                    else
                    {
                        toStateQd();
                    }
                }
            }

            public int getInteger()
            {
                return (int)(sign * result);
            }

            public State getState()
            {
                return currentState;
            }
        }
        private int ConvertStringToIntUsingRules(string s)
        {
            int sign = 1;
            int result = 0;
            int i = 0;

            while (i < s.Length && s[i] == ' ')
            {
                i++;
            }

            if (i < s.Length && s[i] == '+')
            {
                sign = 1;
                i++;
            }
            else if (i < s.Length && s[i] == '-')
            {
                sign = -1;
                i++;
            }

            while (i < s.Length && char.IsDigit(s[i]))
            {
                if (result > int.MaxValue / 10 ||
                    (result == int.MaxValue / 10 &&
                     s[i] - '0' > int.MaxValue % 10))
                {
                    return sign == 1 ? int.MaxValue : int.MinValue;
                }

                result = result * 10 + (s[i++] - '0');
            }

            return sign * result;
        }
        /*
890. Find and Replace Pattern
https://leetcode.com/problems/find-and-replace-pattern/	

        */
        public List<string> FindAndReplacePattern(string[] words, string pattern)
        {
            /*
Approach 1: Two Maps/Dictioneries
Complexity Analysis
•	Time Complexity: O(N∗K), where N is the number of words, and K is the length of each word.
•	Space Complexity: O(N∗K), the space used by the answer.

            */
            List<string> result = FindAndReplacePatternTwoDict(words, pattern);
            /*
            Approach 2: One Map/Dict
Complexity Analysis
•	Time Complexity: O(N∗K), where N is the number of words, and K is the length of each word.
•	Space Complexity: O(N∗K), the space used by the answer.
            
            */
            result = FindAndReplacePatternSingleDict(words, pattern);

            return result;

        }
        private List<string> FindAndReplacePatternTwoDict(string[] words, string pattern)
        {
            List<string> result = new List<string>();
            foreach (string word in words)
            {
                if (Match(word, pattern))
                    result.Add(word);
            }
            return result;

            bool Match(string word, string pattern)
            {
                Dictionary<char, char> map1 = new Dictionary<char, char>();
                Dictionary<char, char> map2 = new Dictionary<char, char>();

                for (int i = 0; i < word.Length; ++i)
                {
                    char w = word[i];
                    char p = pattern[i];
                    if (!map1.ContainsKey(w)) map1[w] = p;
                    if (!map2.ContainsKey(p)) map2[p] = w;
                    if (map1[w] != p || map2[p] != w)
                        return false;
                }

                return true;
            }

        }
        public List<string> FindAndReplacePatternSingleDict(string[] words, string pattern)
        {
            List<string> result = new List<string>();
            foreach (string word in words)
            {
                if (Match(word, pattern))
                {
                    result.Add(word);
                }
            }
            return result;


            bool Match(string word, string pattern)
            {
                Dictionary<char, char> characterMap = new Dictionary<char, char>();
                for (int i = 0; i < word.Length; ++i)
                {
                    char currentWordChar = word[i];
                    char currentPatternChar = pattern[i];
                    if (!characterMap.ContainsKey(currentWordChar))
                    {
                        characterMap[currentWordChar] = currentPatternChar;
                    }
                    if (characterMap[currentWordChar] != currentPatternChar)
                    {
                        return false;
                    }
                }

                bool[] seenCharacters = new bool[26];
                foreach (char mappedChar in characterMap.Values)
                {
                    if (seenCharacters[mappedChar - 'a'])
                    {
                        return false;
                    }
                    seenCharacters[mappedChar - 'a'] = true;
                }
                return true;
            }
        }



        /*
        10. Regular Expression Matching
        https://leetcode.com/problems/regular-expression-matching/

        */
        public bool IsMatch(string text, string pattern)
        {
            /*
            Approach 1: Recursion
Complexity Analysis
•	Time Complexity: Let T,P be the lengths of the text and the pattern respectively. In the worst case, a call to match(text[i:], pattern[2j:]) will be made (ii+j) times, and strings of the order O(T−i) and O(P−2∗j) will be made. Thus, the complexity has the order ∑i=0T∑j=0P/2(ii+j)O(T+P−i−2j). With some effort outside the scope of this article, we can show this is bounded by O((T+P)2T+2P).
•	Space Complexity: For every call to match, we will create those strings as described above, possibly creating duplicates. If memory is not freed, this will also take a total of O((T+P)2T+2P) space, even though there are only order O(T2+P2) unique suffixes of P and T that are actually required.
            */
            bool isMatch = IsMatchRec(text, pattern);
            /*
            Approach 2: Dynamic Programming Top Down(Recursion) 
            Approach 3: Dynamic Programming Bottom Up(Iterative)

Complexity Analysis
•	Time Complexity: Let T,P be the lengths of the text and the pattern respectively. The work for every call to dp(i, j) for i=0,...,T; j=0,...,P is done once, and it is O(1) work. Hence, the time complexity is O(TP).
•	Space Complexity: The only memory we use is the O(TP) boolean entries in our cache. Hence, the space complexity is O(TP).
            */
            isMatch = IsMatchDPTopDown(text, pattern);

            isMatch = IsMatchDPBottomUp(text, pattern);

            return isMatch;

        }
        public bool IsMatchDPBottomUp(string text, string pattern)
        {
            bool[,] dp = new bool[text.Length + 1, pattern.Length + 1];
            dp[text.Length, pattern.Length] = true;

            for (int i = text.Length; i >= 0; i--)
            {
                for (int j = pattern.Length - 1; j >= 0; j--)
                {
                    bool firstMatch =
                        (i < text.Length &&
                            (pattern[j] == text[i] ||
                                pattern[j] == '.'));
                    if (j + 1 < pattern.Length && pattern[j + 1] == '*')
                    {
                        dp[i, j] = dp[i, j + 2] || (firstMatch && dp[i + 1, j]);
                    }
                    else
                    {
                        dp[i, j] = firstMatch && dp[i + 1, j + 1];
                    }
                }
            }
            return dp[0, 0];
        }
        Result[][] memo;

        public bool IsMatchDPTopDown(string text, string pattern)
        {
            memo = new Result[text.Length + 1][];
            for (int k = 0; k < memo.Length; k++)
            {
                memo[k] = new Result[pattern.Length + 1];
            }
            return IsMatchDPTopDown(0, 0, text, pattern);
        }

        public bool IsMatchDPTopDown(int i, int j, string text, string pattern)
        {
            if (memo[i][j] != 0) // Assuming default value of Result is FALSE
            {
                return memo[i][j] == Result.TRUE;
            }
            bool ans;
            if (j == pattern.Length)
            {
                ans = i == text.Length;
            }

            {
                bool firstMatch =
                    (i < text.Length &&
                        (pattern[j] == text[i] ||
                            pattern[j] == '.'));

                if (j + 1 < pattern.Length && pattern[j + 1] == '*')
                {
                    ans = (IsMatchDPTopDown(i, j + 2, text, pattern) ||
                        (firstMatch && IsMatchDPTopDown(i + 1, j, text, pattern)));
                }
                else
                {
                    ans = firstMatch && IsMatchDPTopDown(i + 1, j + 1, text, pattern);
                }
            }
            memo[i][j] = ans ? Result.TRUE : Result.FALSE;
            return ans;
        }

        enum Result
        {
            TRUE,
            FALSE,
        }

        private bool IsMatchRec(string inputText, string inputPattern)
        {
            if (string.IsNullOrEmpty(inputPattern)) return string.IsNullOrEmpty(inputText);
            bool isFirstMatch =
                (!string.IsNullOrEmpty(inputText) &&
                    (inputPattern[0] == inputText[0] ||
                        inputPattern[0] == '.'));

            if (inputPattern.Length >= 2 && inputPattern[1] == '*')
            {
                return (
                    IsMatchRec(inputText, inputPattern.Substring(2)) ||
                    (isFirstMatch && IsMatchRec(inputText.Substring(1), inputPattern))
                );
            }
            else
            {
                return (
                    isFirstMatch && IsMatchRec(inputText.Substring(1), inputPattern.Substring(1))
                );
            }
        }
        /*
        44. Wildcard Matching
        https://leetcode.com/problems/wildcard-matching/description/
        */
        public bool IsWildCardMatch(string text, string pattern)
        {
            /*
Approach 1: Recursion with Memoization
Complexity Analysis
•	Time complexity: O(S⋅P⋅(S+P))
o	Removing duplicate stars requires us to traverse the string p once, this requires O(P) time.
o	Regarding the helper function, every non-memoized recursive call we will:
1.	Check if helper(s, p) has already been calculated. This takes O(S+P) time to create a hash of the tuple (s, p) the first time and O(1) time to check if the result has already been cached.
2.	Go through our if statements. If (s, p) is one of the base cases, this will take O(min(S,P)) time for the string equality check or just O(1) time for other checks, otherwise, it will take O(S+P) time to create a substring s[1:] and a substring p[1:]. Here, let's assume the worst-case scenario where most of the non-memoized recursive calls require O(S+P) time.
3.	Then we will cache our result, which takes O(1) time since the hash for tuple (s, p) was already created when we checked if the result for (s, p) is already cached.
So in total, we spend O(2⋅(S+P))=O(S+P) time on every non-memoized call (S+P for creating a hash and S+P for creating substrings). We can only have as many non-memoized calls as there are combinations of s and p. Therefore, in the worst case, we can have S⋅P non-memoized calls. This gives us a total time spent on non-memoized calls of O(S⋅P⋅(S+P)).
o	As for the memoized calls, for each non-memoized call, we can make at most 2 additional calls to helper. This means that there will be at most S⋅P memoized calls. Each memoized call takes O(S+P) time to create the hash for (s, p) and O(1) time to get the cached result. So the total time spent on memoized calls is O(S⋅P⋅(S+P)) which is a loose upper bound.
o	Adding all 3 time complexities together we get: O(P+2⋅S⋅P⋅(S+P))=O(S⋅P⋅(S+P)).
Note: This approach can be optimized by using two pointers to track the current position on s and p instead of passing substrings of s and p as arguments. To improve readability, this was not implemented here, however, doing so will reduce the time complexity to O(S⋅P) since hashing two integers takes O(1) time and each recursive call to helper would no longer require creating new substrings which takes linear time. Thus the total time complexity is O(1) per call for a maximum of S⋅P non-memoized calls and S⋅P memoized calls.
•	Space complexity: O(S⋅P). Creating a new string p requires O(P) space. The recursion call stack may exceed max(S, P) in cases such as (s, p) = (aaab, *a*b), however, it is bounded by O(S+P). Lastly, the hashmap requires O(S⋅P) space to memoize the result of each call to helper.

            */
            bool isWildCardMatch = IsWildCardMatchRecMemo(text, pattern);

            /*
Approach 2: Dynamic Programming
Complexity Analysis

Time complexity: O(S⋅P) where S and P are lengths of the input string and the pattern respectively.
Space complexity: O(S⋅P) to store the matrix.
            
            */
            isWildCardMatch = IsWildCardMatchDP(text, pattern);

            /*
Approach 3: Backtracking
Complexity Analysis

Time complexity: O(min(S,P)) for the best case and better than O(SlogP) for the average case, where S and P are lengths of the input string and the pattern correspondingly. Please refer to this article for detailed proof. However, in the worst-case scenario, this algorithm requires O(S⋅P) time.
Space complexity: O(1) since it's a constant space solution.
            */
            isWildCardMatch = IsWildCardMatchBacktrack(text, pattern);

            return isWildCardMatch;

        }
        public bool IsWildCardMatchBacktrack(string s, string p)
        {
            int sLen = s.Length, pLen = p.Length;
            int sIdx = 0, pIdx = 0;
            int starIdx = -1, sTmpIdx = -1;
            while (sIdx < sLen)
            {
                // If the pattern character = string character
                // or pattern character = '?'
                if (pIdx < pLen && (p[pIdx] == '?' || p[pIdx] == s[sIdx]))
                {
                    ++sIdx;
                    ++pIdx;
                }
                // If pattern character = '*'
                else if (pIdx < pLen && p[pIdx] == '*')
                {
                    // Check the situation
                    // when '*' matches no characters
                    starIdx = pIdx;
                    sTmpIdx = sIdx;
                    ++pIdx;
                }
                // If pattern character != string character
                // or pattern is used up
                // and there was no '*' character in pattern
                else if (starIdx == -1)
                {
                    return false;
                }
                // If pattern character != string character
                // or pattern is used up
                // and there was '*' character in pattern before
                else
                {
                    // Backtrack: check the situation
                    // when '*' matches one more character
                    pIdx = starIdx + 1;
                    sIdx = sTmpIdx + 1;
                    sTmpIdx = sIdx;
                }
            }

            // The remaining characters in the pattern should all be '*' characters
            for (int i = pIdx; i < pLen; i++)
            {
                if (p[i] != '*')
                {
                    return false;
                }
            }

            return true;
        }
        public bool IsWildCardMatchDP(string s, string p)
        {
            int sLen = s.Length, pLen = p.Length;
            // base cases
            if (p == s)
            {
                return true;
            }

            if (pLen > 0 && p.All(c => c == '*'))
            {
                return true;
            }

            if (p.Length == 0 || s.Length == 0)
            {
                return false;
            }

            // init all matrix except [0][0] element as False
            bool[,] d = new bool[pLen + 1, sLen + 1];
            d[0, 0] = true;
            // DP compute
            for (int pIdx = 1; pIdx < pLen + 1; pIdx++)
            {
                // the current character in the pattern is '*'
                if (p[pIdx - 1] == '*')
                {
                    int sIdx = 1;
                    // d[p_idx - 1][s_idx - 1] is a string-pattern match
                    // on the previous step, i.e. one character before.
                    // Find the first idx in string with the previous math.
                    while ((!d[pIdx - 1, sIdx - 1]) && (sIdx < sLen + 1))
                    {
                        sIdx++;
                    }

                    // If (string) matches (pattern),
                    // when (string) matches (pattern)* as well
                    d[pIdx, sIdx - 1] = d[pIdx - 1, sIdx - 1];
                    // If (string) matches (pattern),
                    // when (string)(whatever_characters) matches (pattern)* as well
                    while (sIdx < sLen + 1)
                    {
                        d[pIdx, sIdx++] = true;
                    }
                    // the current character in the pattern is '?'
                }
                else if (p[pIdx - 1] == '?')
                {
                    for (int sIdx = 1; sIdx < sLen + 1; sIdx++)
                    {
                        d[pIdx, sIdx] = d[pIdx - 1, sIdx - 1];
                    }
                    // the current character in the pattern is not '*' or '?'
                }
                else
                {
                    for (int sIdx = 1; sIdx < sLen + 1; sIdx++)
                    {
                        // Match is possible if there is a previous match
                        // and current characters are the same
                        d[pIdx, sIdx] =
                            d[pIdx - 1, sIdx - 1] && (p[pIdx - 1] == s[sIdx - 1]);
                    }
                }
            }

            return d[pLen, sLen];
        }
        Dictionary<string, bool> dp = new Dictionary<string, bool>();
        string text;
        string pattern;

        string RemoveDuplicateStars(string p)
        {
            var new_string = new StringBuilder();
            foreach (var c in p)
            {
                if (new_string.Length == 0 || c != '*')
                    new_string.Append(c);
                else if (new_string[new_string.Length - 1] != '*')
                    new_string.Append(c);
            }

            return new_string.ToString();
        }

        bool Helper(int si, int pi)
        {
            var key = si + "," + pi;
            if (dp.ContainsKey(key))
                return dp[key];
            if (pi == pattern.Length)
                dp[key] = si == text.Length;
            else if (si == text.Length)
                dp[key] = pi + 1 == pattern.Length && pattern[pi] == '*';
            else if (pattern[pi] == text[si] || pattern[pi] == '?')
                dp[key] = Helper(si + 1, pi + 1);
            else if (pattern[pi] == '*')
                dp[key] = Helper(si, pi + 1) || Helper(si + 1, pi);
            else
                dp[key] = false;
            return dp[key];
        }

        public bool IsWildCardMatchRecMemo(string text, string pattern)
        {
            dp.Clear();
            this.text = text;
            this.pattern = RemoveDuplicateStars(this.pattern);
            return Helper(0, 0);
        }



        /*
        28. Find the Index of the First Occurrence in a String
https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/	

        */
        public int FindIndexOfFirstOccur(string haystack, string needle)
        {
            /*
 Approach 1: Sliding Window (SW)          
 Complexity Analysis
•	Time complexity: O(nm). For every window_start, we may have to iterate at most m times. There are n−m+1 such window_start's. Thus, it is O((n−m+1)⋅m), which is O(nm).
One example where the worst case occurs is when needle is "aaaaab", while haystack is all a's (Let's say, "aaaaaaaaaa"). In this, we always have to take the last character of needle into comparison to conclude that the current m-substring is not equal to needle. Thus, we will have to iterate m times for every window_start.
•	Space complexity: O(1).
There are a handful of variables in the code (m, n, i, window_start), and all of them use constant space, hence, the space complexity is constant.
           
            */
            int indexOfFirstOccur = FindIndexOfFirstOccurSW(haystack, needle);
            /*
 Approach 2: Rabin-Karp Algorithm (Single Hash)(RKSH)           
Complexity Analysis
•	Time complexity: O(nm).
In the worst case, hashNeedle might match with the hash value of all haystack substrings. Hence, we then have to iterate character by character in each window. There are n−m+1 such windows of length m. Hence, the time complexity is O(nm).
But in the best case, if no hash value of the haystack substring matches with hashNeedle, then we don't have to iterate character by character in each window. In that case, it will be O(n+m). Computing the hash value of haystack and needle will be O(m) and for traversing all windows, we will have O(n−m) time complexity. And during traversal, we are doing constant number of operations, hence, in that case, it will be O(n−m+2m)=O(n+m).
•	Space complexity: O(1). We are not using any extra space.
There are a handful of variables in the code, namely, hashNeedle, hashHay, windowStart, windowEnd, m, n, MAX_WEIGHT, RADIX, MOD. All of them use constant space, and hence, the space complexity is O(1).
            
            */
            indexOfFirstOccur = FindIndexOfFirstOccurRKSH(haystack, needle);
            /*
Approach 3: Rabin-Karp algorithm (Double Hash) (RKDH)
Complexity Analysis
•	Time complexity: O(n).
o	For computing hash pairs of needle, we have to do O(m) work.
o	For checking for a match, we have to iterate over n−m+1 times. Out of these n−m+1 operations, we have to do O(1) work for n−m times, and O(m) work for 1 time.
o	Hence, total time complexity is O(m+(n−m)⋅1+(1)⋅m), which is O(n+m)
o	Moreover, we are proceeding only when n≥m, thus final time complexity is O(n) only. In this case, O(m+n) has an upper bound of O(2⋅n), that's why we can ignore the m term. When n<m we are simply returning -1. Thus, only n is dominating in Time Complexity, and not m.
•	Space complexity: O(1). We are not using any extra space.
There are a handful of variables in the code, and all of them use constant space, hence, the space complexity is O(1).

            */
            indexOfFirstOccur = FindIndexOfFirstOccurRKDH(haystack, needle);
            /*
Approach 4: Knuth–Morris–Pratt Algorithm (KMP)
Implementation
Complexity Analysis
•	Time complexity: O(n).
o	If n<m, then we immediately return -1. Hence, it is O(1) in this case.
o	Otherwise,
o	Pre-processing takes O(m) time.
o	In the case of "Matching", or "Mismatch (Empty Previous Border)", we simply increment i, which is O(1).
o	In the case of "Mismatch (Non-Empty Previous Border)", we reduce prev to longest_border[prev-1]. In other words, we try to reduce at most as many times as the while loop has executed. There will be at most m−1 such reductions.
o	Thus, it will be O(m).
o	Searching takes O(n) time.
o	We never backtrack/reset the haystack_pointer. We increment it by 1 in Matching or Zero-Matching.
o	In Partial-Matching, we don't immediately increment and try to reduce to a condition of Matching, or Zero-Matching. For this, we set needle_pointer to longest_border[needle_pointer-1], which always reduces to 0 or matches. The maximum number of rollbacks of needle_pointer is bounded by needle_pointer. For any mismatch, we can only roll back as much as we have advanced up to the mismatch..
o	Thus, for searching it is O(2n), which is O(n).
o	Hence, it is O(m+n) and since n≥m, we can ignore m term. The final upper bound is O(2⋅n), which is O(n).
o	Therefore, overall it is O(n)+O(1), which is O(n).
o	No worst-case or accidental inputs exist here.
•	Space complexity: O(m). To store the longest_border array, we need O(m) extra space.
Note: Although KMP is fast, still built-in functions of many programming languages use Brute Force. KMP is based on assumption that there would be many duplicate similar substrings. In real-world strings, this is not the case. So, KMP is not used in real-world applications. Moreover, it requires linear space.
            
            */
            indexOfFirstOccur = FindIndexOfFirstOccurKMP(haystack, needle);

            return indexOfFirstOccur;

        }
        public int FindIndexOfFirstOccurSW(string haystack, string needle)
        {
            int m = needle.Length;
            int n = haystack.Length;

            for (int windowStart = 0; windowStart <= n - m; windowStart++)
            {
                for (int i = 0; i < m; i++)
                {
                    if (needle[i] != haystack[windowStart + i])
                    {
                        break;
                    }

                    if (i == m - 1)
                    {
                        return windowStart;
                    }
                }
            }

            return -1;
        }
        private long hashValue(string str, int RADIX, int MOD, int m)
        {
            long ans = 0;
            long factor = 1;
            for (int i = m - 1; i >= 0; --i)
            {
                ans = (ans + (str[i] - 'a') * factor % MOD + MOD) % MOD;
                factor = (factor * RADIX) % MOD;
            }

            return ans;
        }

        public int FindIndexOfFirstOccurRKSH(string haystack, string needle)
        {
            int m = needle.Length, n = haystack.Length;
            if (n < m)
                return -1;
            const int RADIX = 26, MOD = 1000000033;
            long MAX_WEIGHT = 1;
            for (int i = 0; i < m; ++i) MAX_WEIGHT = (MAX_WEIGHT * RADIX) % MOD;
            long hashNeedle = hashValue(needle, RADIX, MOD, m),
                 hashHay = hashValue(haystack, RADIX, MOD, m);
            for (int windowStart = 0; windowStart <= n - m; ++windowStart)
            {
                if (windowStart != 0)
                {
                    hashHay =
                        ((hashHay * RADIX) % MOD -
                         (haystack[windowStart - 1] - 'a') * MAX_WEIGHT % MOD +
                         (haystack[windowStart + m - 1] - 'a') + MOD) %
                        MOD;
                }

                if (hashNeedle == hashHay)
                {
                    for (int i = 0; i < m; ++i)
                    {
                        if (needle[i] != haystack[i + windowStart])
                            break;
                        if (i == m - 1)
                            return windowStart;
                    }
                }
            }

            return -1;
        }
        // CONSTANTS
        const int RADIX_1 = 26;
        const int MOD_1 = 1000000033;
        const int RADIX_2 = 27;
        const int MOD_2 = 2147483647;

        public long[] hashPair(string str, int m)
        {
            long hash1 = 0, hash2 = 0;
            long factor1 = 1, factor2 = 1;
            for (int i = m - 1; i >= 0; i--)
            {
                hash1 += ((int)(str[i] - 'a') * (factor1)) % MOD_1;
                factor1 = (factor1 * RADIX_1) % MOD_1;
                hash2 += ((int)(str[i] - 'a') * (factor2)) % MOD_2;
                factor2 = (factor2 * RADIX_2) % MOD_2;
            }

            return new long[] { hash1 % MOD_1, hash2 % MOD_2 };
        }

        public int FindIndexOfFirstOccurRKDH(string haystack, string needle)
        {
            int m = needle.Length;
            int n = haystack.Length;
            if (n < m)
                return -1;
            long MAX_WEIGHT_1 = 1;
            long MAX_WEIGHT_2 = 1;
            for (int i = 0; i < m; i++)
            {
                MAX_WEIGHT_1 = (MAX_WEIGHT_1 * RADIX_1) % MOD_1;
                MAX_WEIGHT_2 = (MAX_WEIGHT_2 * RADIX_2) % MOD_2;
            }

            long[] hashNeedle = hashPair(needle, m);
            long[] hashHay = new long[2];
            for (int windowStart = 0; windowStart <= n - m; windowStart++)
            {
                if (windowStart == 0)
                {
                    hashHay = hashPair(haystack, m);
                }
                else
                {
                    hashHay[0] =
                        ((hashHay[0] * RADIX_1) % MOD_1 -
                         ((int)(haystack[windowStart - 1] - 'a') * MAX_WEIGHT_1) %
                             MOD_1 +
                         (int)(haystack[windowStart + m - 1] - 'a') + MOD_1) %
                        MOD_1;
                    hashHay[1] =
                        ((hashHay[1] * RADIX_2) % MOD_2 -
                         ((int)(haystack[windowStart - 1] - 'a') * MAX_WEIGHT_2) %
                             MOD_2 +
                         (int)(haystack[windowStart + m - 1] - 'a') + MOD_2) %
                        MOD_2;
                }

                if (hashNeedle[0] == hashHay[0] && hashNeedle[1] == hashHay[1])
                {
                    return windowStart;
                }
            }

            return -1;
        }
        public int FindIndexOfFirstOccurKMP(string haystack, string needle)
        {
            int m = needle.Length;
            int n = haystack.Length;

            if (n < m)
                return -1;

            // PREPROCESSING
            // longest_border array
            int[] longest_border = new int[m];
            // Length of Longest Border for prefix before it.
            int prev = 0;
            // Iterating from index-1. longest_border[0] will always be 0
            int i = 1;

            while (i < m)
            {
                if (needle[i] == needle[prev])
                {
                    // Length of Longest Border Increased
                    prev += 1;
                    longest_border[i] = prev;
                    i += 1;
                }
                else
                {
                    // Only empty border exist
                    if (prev == 0)
                    {
                        longest_border[i] = 0;
                        i += 1;
                    }
                    // Try finding longest border for this i with reduced prev
                    else
                    {
                        prev = longest_border[prev - 1];
                    }
                }
            }

            // SEARCHING
            // Pointer for haystack
            int haystackPointer = 0;
            // Pointer for needle.
            // Also indicates number of characters matched in current window.
            int needlePointer = 0;

            while (haystackPointer < n)
            {
                if (haystack[haystackPointer] == needle[needlePointer])
                {
                    // Matched Increment Both
                    needlePointer += 1;
                    haystackPointer += 1;
                    // All characters matched
                    if (needlePointer == m)
                    {
                        // m characters behind last matching will be start of window
                        return haystackPointer - m;
                    }
                }
                else
                {
                    if (needlePointer == 0)
                    {
                        // Zero Matched
                        haystackPointer += 1;
                    }
                    else
                    {
                        // Optimally shift left needlePointer. Don't change
                        // haystackPointer
                        needlePointer = longest_border[needlePointer - 1];
                    }
                }
            }

            return -1;
        }

        /*
        58. Length of Last Word
        https://leetcode.com/problems/length-of-last-word/description/

        */
        public int LengthOfLastWord(string s)
        {
            /*
            Approach 1: String Index Manipulation with Two Loops (SIMTL)
Complexity
•	Time Complexity: O(N), where N is the length of the input string.
In the worst case, the input string might contain only a single word, which implies that we would need to iterate through the entire string to obtain the result.
•	Space Complexity: O(1), only constant memory is consumed, regardless the input.

            */
            int len = LengthOfLastWordSIMTL(s);

            /*
Approach 2: One-Loop Iteration (OLI)
  Complexity
•	Time Complexity: O(N), where N is the length of the input string.
This approach has the same time complexity as the previous approach. The only difference is that we combined two loops into one.
•	Space Complexity: O(1), again a constant memory is consumed, regardless the input.
          
            */
            len = LengthOfLastWordOLI(s);

            /*
   Approach 3: Built-in String Functions (BSF)        
Complexity Analysis
•	Time Complexity: O(N), where N is the length of the input string.
Since we use some built-in function from the String data type, we should look into the complexity of each built-in function that we used, in order to obtain the overall time complexity of our algorithm.
It would be safe to assume the time complexity of the methods such as str.split() and String.lastIndexOf() to be O(N), since in the worst case we would need to scan the entire string for both methods.
•	Space Complexity: O(N). Again, we should look into the built-in functions that we used in the algorithm.
In the Java implementation, we used the function String.trim() which returns a copy of the input string without leading and trailing whitespace. Therefore, we would need O(N) space for our algorithm to hold this copy.

            */
            len = LengthOfLastWordBSF(s);

            return len;


        }
        public int LengthOfLastWordSIMTL(string s)
        {
            // trim the trailing spaces
            int p = s.Length - 1;
            while (p >= 0 && s[p] == ' ')
            {
                p--;
            }

            // compute the length of last word
            int length = 0;
            while (p >= 0 && s[p] != ' ')
            {
                p--;
                length++;
            }

            return length;
        }

        public int LengthOfLastWordOLI(string s)
        {
            int p = s.Length, length = 0;
            while (p > 0)
            {
                p--;
                // we're in the middle of the last word
                if (s[p] != ' ')
                {
                    length++;
                }
                // here is the end of last word
                else if (length > 0)
                {
                    return length;
                }
            }

            return length;
        }
        public int LengthOfLastWordBSF(string s)
        {
            s = s.Trim();  // trim the trailing spaces in the string
            return s.Length - s.LastIndexOf(" ") - 1;
        }

        /*
        65. Valid Number		
        https://leetcode.com/problems/valid-number/description/

        */
        public bool IsValidNumber(string s)
        {
            /*
   Approach 1: Follow The Rules! (FTR)         
   Complexity Analysis
•	Time complexity: O(N), where N is the length of s.
We simply iterate over the input once. The number of operations we perform for each character in the input is independent of the length of the string, and therefore only requires constant time. This results in N⋅O(1)=O(N).
•	Space complexity: O(1).
Regardless of the input size, we only store 3 variables, seenDigit, seenExponent, and seenDot.
         
            */
            bool isValidNumber = IsValidNumberFTR(s);
            /*
   Approach 2: Deterministic Finite Automaton (DFA)         
    Complexity Analysis
•	Time complexity: O(N), where N is the length of s.
We simply iterate through the input once. The number of operations we perform for each character in the input is independent of the length of the string, and therefore each operation requires constant time. So we get N⋅O(1)=O(N).
•	Space complexity: O(1).
We will construct the same DFA regardless of the input size.
        
            */
            isValidNumber = IsValidNumberDFA(s);

            return isValidNumber;

        }
        public bool IsValidNumberFTR(string s)
        {
            bool seenDigit = false;
            bool seenExponent = false;
            bool seenDot = false;
            for (int i = 0; i < s.Length; i++)
            {
                char curr = s[i];
                if (Char.IsDigit(curr))
                {
                    seenDigit = true;
                }
                else if (curr == '+' || curr == '-')
                {
                    if (i > 0 && s[i - 1] != 'e' && s[i - 1] != 'E')
                    {
                        return false;
                    }
                }
                else if (curr == 'e' || curr == 'E')
                {
                    if (seenExponent || !seenDigit)
                    {
                        return false;
                    }

                    seenExponent = true;
                    seenDigit = false;
                }
                else if (curr == '.')
                {
                    if (seenDot || seenExponent)
                    {
                        return false;
                    }

                    seenDot = true;
                }
                else
                {
                    return false;
                }
            }

            return seenDigit;
        }

        public bool IsValidNumberDFA(string s)
        {
            // This is the DFA we have designed above
            var dfa = new Dictionary<string, int>[] {
            new Dictionary<string, int> {
                { "digit", 1 }, { "sign", 2 }, { "dot", 3 }
            },
            new Dictionary<string, int> {
                { "digit", 1 }, { "dot", 4 }, { "exponent", 5 }
            },
            new Dictionary<string, int> { { "digit", 1 }, { "dot", 3 } },
            new Dictionary<string, int> { { "digit", 4 } },
            new Dictionary<string, int> { { "digit", 4 }, { "exponent", 5 } },
            new Dictionary<string, int> { { "sign", 6 }, { "digit", 7 } },
            new Dictionary<string, int> { { "digit", 7 } },
            new Dictionary<string, int> { { "digit", 7 } }
        };
            int currentState = 0;
            string group;
            foreach (char curr in s)
            {
                if (Char.IsDigit(curr))
                {
                    group = "digit";
                }
                else if (curr == '+' || curr == '-')
                {
                    group = "sign";
                }
                else if (curr == 'e' || curr == 'E')
                {
                    group = "exponent";
                }
                else if (curr == '.')
                {
                    group = "dot";
                }
                else
                {
                    return false;
                }

                if (!dfa[currentState].ContainsKey(group))
                {
                    return false;
                }

                currentState = dfa[currentState][group];
            }

            return currentState == 1 || currentState == 4 || currentState == 7;
        }


        /*
        68. Text Justification
        https://leetcode.com/problems/text-justification/description/
        Complexity Analysis
        Let n be words.Length, k be the average length of a word, and m be maxWidth.
        Here, we are assuming that you are using immutable strings. A language like C++ has mutable strings and thus the complexity analysis will be slightly different.
        •	Time complexity: O(n⋅k)
        getWords
        The work done in each while loop iteration is O(1). Thus the cost of each call is equal to the number of times the while loop runs in each call. This is amortized throughout the runtime of the algorithm - each index of words can only be iterated over once throughout all calls, so the time complexity of all calls to getWords is O(n).
        createLine
        First, we iterate over the words in line to calculate baseLength. Again, this is amortized over the runtime of the algorithm as each word in the input can only be iterated over once here. Therefore, this loop contributes O(n) over all calls to createLine.
        If we are dealing with the special case (one word line or last lane), we create a string of length maxWidth. This costs O(m).
        Otherwise, we iterate over the words in line and perform string operations on each. The first for loop which adds the mandatory space costs O(k) per iteration. In the worst-case scenario, we won't have any lines with only one word and the final line has only one word. In this scenario, over the runtime of the algorithm, this for loop will iterate over every word except for the final one, which would cost O(n⋅k).
        The second for loop which adds the extra spaces is harder to analyze. At a minimum, each operation will cost O(k). The amount of spaces we add is a function of maxWidth and the number of words in line, as well as the sum of their lengths. One thing is for certain though: on a given call, the strings we create in this for loop cannot exceed maxWidth in length combined. Therefore, we can say that this for loop costs O(m) per call to createLine.
        Finally, we join the line with a delimiter, which costs O(m).
        Overall, this function contributes O(n⋅k) to the overall runtime, and O(m) per call.
        Main section
        We already determined that all calls to getWords contribute O(n) in total, so we don't have to worry about that.
        Each call to createLine costs O(m). We call it in each while loop iteration. The number of while loop iterations is a function of n, k, and m. On average, we can fit km words per line. Because we have n words, that implies O(kmn)=O(mn⋅k) iterations. Each iteration costs O(m), so this gives us O(n⋅k).
        Summing it all up and canceling constants, we have a time complexity of O(n⋅k) - the sum of the characters in all the words.
        •	Space complexity: O(m)
        We don't count the answer as part of the space complexity.
        We handle one line at a time and each line has a length of m. The intermediate arrays we use like currentLine hold strings, but the sum of the lengths of these strings cannot exceed m either.


        */
        public class FullTextJustifySolution
        {
            public IList<string> FullTextJustify(string[] words, int maxWidth)
            {
                var ans = new List<string>();
                int i = 0;
                while (i < words.Length)
                {
                    var currentLine = GetWords(i, words, maxWidth);
                    i += currentLine.Count;
                    ans.Add(CreateLine(currentLine, i, words, maxWidth));
                }

                return ans;
            }

            private List<string> GetWords(int i, string[] words, int maxWidth)
            {
                var currentLine = new List<string>();
                int currLength = 0;
                while (i < words.Length && currLength + words[i].Length <= maxWidth)
                {
                    currentLine.Add(words[i]);
                    currLength += words[i].Length + 1;
                    i++;
                }

                return currentLine;
            }

            private string CreateLine(List<string> line, int i, string[] words,
                                      int maxWidth)
            {
                int baseLength = -1;
                foreach (var word in line)
                {
                    baseLength += word.Length + 1;
                }

                int extraSpaces = maxWidth - baseLength;
                if (line.Count == 1 || i == words.Length)
                {
                    return string.Join(" ", line) + new string(' ', extraSpaces);
                }

                int wordCount = line.Count - 1;
                int spacesPerWord = extraSpaces / wordCount;
                int needsExtraSpace = extraSpaces % wordCount;
                for (int j = 0; j < needsExtraSpace; j++)
                {
                    line[j] += " ";
                }

                for (int j = 0; j < wordCount; j++)
                {
                    line[j] += new string(' ', spacesPerWord);
                }

                return string.Join(" ", line);
            }
        }

        /*
        87. Scramble String
https://leetcode.com/problems/scramble-string/description/

Approach: Dynamic Programming

Complexity Analysis
•	Time complexity: O(n4).
We have four nested for loops (for length, i, j, newLength), each doing O(n) iterations.
•	Space complexity: O(n3).
We store the matrix dp[n+1][n][n] for dynamic programming.

        */
        public bool IsStringScrambled(string s1, string s2)
        {
            int n = s1.Length;
            bool[][][] dp = new bool[n + 1][][];
            for (int i = 0; i < dp.Length; i++)
            {
                dp[i] = new bool[n][];
                for (int j = 0; j < dp[i].Length; j++)
                {
                    dp[i][j] = new bool[n];
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    dp[1][i][j] = s1[i] == s2[j];
                }
            }

            for (int length = 2; length <= n; length++)
            {
                for (int i = 0; i < n + 1 - length; i++)
                {
                    for (int j = 0; j < n + 1 - length; j++)
                    {
                        for (int newLength = 1; newLength < length; newLength++)
                        {
                            bool[] dp1 = dp[newLength][i];
                            bool[] dp2 = dp[length - newLength][i + newLength];
                            dp[length][i][j] |= dp1[j] && dp2[j + newLength];
                            dp[length][i][j] |=
                                dp1[j + length - newLength] && dp2[j];
                        }
                    }
                }
            }

            return dp[n][0][0];
        }
        /*
        91. Decode Ways
        https://leetcode.com/problems/decode-ways/description/

        */

        public class NumWaysDecodeStringSol
        {
            /*
            Approach 1: Recursive Approach with Memoization
Complexity Analysis
•	Time Complexity: O(N), where N is length of the string. Memoization helps in pruning the recursion tree and hence decoding for an index only once. Thus this solution is linear time complexity.
•	Space Complexity: O(N). The dictionary used for memoization would take the space equal to the length of the string. There would be an entry for each index value. The recursion stack would also be equal to the length of the string.

            */
            public int NumWaysDecodeStringRecMemo(string s)
            {
                return RecursiveWithMemo(0, s);
            }
            private Dictionary<int, int> memo = new Dictionary<int, int>();

            private int RecursiveWithMemo(int index, string s)
            {
                if (memo.ContainsKey(index))
                {
                    return memo[index];
                }

                if (index == s.Length)
                {
                    return 1;
                }

                if (s[index] == '0')
                {
                    return 0;
                }

                if (index == s.Length - 1)
                {
                    return 1;
                }

                int ans = RecursiveWithMemo(index + 1, s);
                if (int.Parse(s.Substring(index, 2)) <= 26)
                {
                    ans += RecursiveWithMemo(index + 2, s);
                }

                memo[index] = ans;
                return ans;
            }

            /*
            Approach 2: Iterative Approach
    Complexity Analysis
    •	Time Complexity: O(N), where N is length of the string. We iterate the length of dp array which is N+1.
    •	Space Complexity: O(N). The length of the DP array.

            */
            public int NumWaysDecodeStringIterative(string s)
            {
                // DP array to store the subproblem results
                var dp = new int[s.Length + 1];
                dp[0] = 1;
                // Ways to decode a string of size 1 is 1. Unless the string is '0'.
                // '0' doesn't have a single digit decode.
                dp[1] = s[0] == '0' ? 0 : 1;
                for (var i = 2; i < dp.Length; i++)
                {
                    // Check if successful single digit decode is possible.
                    if (s[i - 1] != '0')
                    {
                        dp[i] = dp[i - 1];
                    }

                    // Check if successful two digit decode is possible.
                    int twoDigit = Int32.Parse(s.Substring(i - 2, 2));
                    if (twoDigit >= 10 && twoDigit <= 26)
                    {
                        dp[i] += dp[i - 2];
                    }
                }

                return dp[s.Length];
            }

            /*
            Approach 3: Iterative, Constant Space

    Complexity Analysis
    •	Time Complexity: O(N), where N is length of the string. We're essentially doing the same work as what we were in Approach 2, except this time we're throwing away calculation results when we no longer need them.
    •	Space Complexity: O(1). Instead of a dp array, we're simply using two variables.

            */


            public int NumWaysDecodeStringIterativeSpaceOptimal(string s)
            {
                if (s[0] == '0')
                {
                    return 0;
                }

                int n = s.Length;
                int twoBack = 1;
                int oneBack = 1;
                for (int i = 1; i < n; i++)
                {
                    int current = 0;
                    if (s[i] != '0')
                    {
                        current = oneBack;
                    }

                    int twoDigit = int.Parse(s.Substring(i - 1, 2));
                    if (twoDigit >= 10 && twoDigit <= 26)
                    {
                        current += twoBack;
                    }

                    twoBack = oneBack;
                    oneBack = current;
                }

                return oneBack;
            }

        }

        /*
        639. Decode Ways II
https://leetcode.com/problems/decode-ways-ii/description/

        */
        public class NumWaysDecodeStringIISol
        {
            /*
            Approach 1: Recursion with Memoization

            Complexity Analysis
•	Time complexity : O(n). Size of recursion tree can go up to n, since memo array is filled exactly once. Here, n refers to the length of the input
string.
•	Space complexity : O(n). The depth of recursion tree can go up to n.

            */

            private const int Modulus = 1000000007;

            public int NumWaysDecodeStringIIRecMemo(string inputString)
            {
                long?[] memoizationArray = new long?[inputString.Length];
                return (int)CalculateWays(inputString, inputString.Length - 1, memoizationArray);
            }

            private long CalculateWays(string inputString, int index, long?[] memoizationArray)
            {
                if (index < 0)
                    return 1;

                if (memoizationArray[index] != null)
                    return memoizationArray[index].Value;

                if (inputString[index] == '*')
                {
                    long result = 9 * CalculateWays(inputString, index - 1, memoizationArray) % Modulus;
                    if (index > 0 && inputString[index - 1] == '1')
                        result = (result + 9 * CalculateWays(inputString, index - 2, memoizationArray)) % Modulus;
                    else if (index > 0 && inputString[index - 1] == '2')
                        result = (result + 6 * CalculateWays(inputString, index - 2, memoizationArray)) % Modulus;
                    else if (index > 0 && inputString[index - 1] == '*')
                        result = (result + 15 * CalculateWays(inputString, index - 2, memoizationArray)) % Modulus;

                    memoizationArray[index] = result;
                    return memoizationArray[index].Value;
                }

                long resultForNonStar = inputString[index] != '0' ? CalculateWays(inputString, index - 1, memoizationArray) : 0;

                if (index > 0 && inputString[index - 1] == '1')
                    resultForNonStar = (resultForNonStar + CalculateWays(inputString, index - 2, memoizationArray)) % Modulus;
                else if (index > 0 && inputString[index - 1] == '2' && inputString[index] <= '6')
                    resultForNonStar = (resultForNonStar + CalculateWays(inputString, index - 2, memoizationArray)) % Modulus;
                else if (index > 0 && inputString[index - 1] == '*')
                    resultForNonStar = (resultForNonStar + (inputString[index] <= '6' ? 2 : 1) * CalculateWays(inputString, index - 2, memoizationArray)) % Modulus;

                memoizationArray[index] = resultForNonStar;
                return memoizationArray[index].Value;
            }


            /*
            Approach 2: Dynamic Programming

            Complexity Analysis
    •	Time complexity : O(n). dp array of size n+1 is filled once only. Here, n refers to the length of the input string.
    •	Space complexity : O(n). dp array of size n+1 is used.

            */
            public int NumWaysDecodeStringIIDP(String s)
            {
                long[] dp = new long[s.Length + 1];
                dp[0] = 1;
                dp[1] = s[0] == '*' ? 9 : s[0] == '0' ? 0 : 1;
                for (int i = 1; i < s.Length; i++)
                {
                    if (s[i] == '*')
                    {
                        dp[i + 1] = 9 * dp[i] % Modulus;
                        if (s[i - 1] == '1')
                            dp[i + 1] = (dp[i + 1] + 9 * dp[i - 1]) % Modulus;
                        else if (s[i - 1] == '2')
                            dp[i + 1] = (dp[i + 1] + 6 * dp[i - 1]) % Modulus;
                        else if (s[i - 1] == '*')
                            dp[i + 1] = (dp[i + 1] + 15 * dp[i - 1]) % Modulus;
                    }
                    else
                    {
                        dp[i + 1] = s[i] != '0' ? dp[i] : 0;
                        if (s[i - 1] == '1')
                            dp[i + 1] = (dp[i + 1] + dp[i - 1]) % Modulus;
                        else if (s[i - 1] == '2' && s[i] <= '6')
                            dp[i + 1] = (dp[i + 1] + dp[i - 1]) % Modulus;
                        else if (s[i - 1] == '*')
                            dp[i + 1] = (dp[i + 1] + (s[i] <= '6' ? 2 : 1) * dp[i - 1]) % Modulus;
                    }
                }
                return (int)dp[s.Length];
            }
            /*
            Approach 3: Constant Space Dynamic Programming (DPCS)
    Complexity Analysis
    •	Time complexity : O(n). Single loop up to n is required to find the required result. Here, n refers to the length of the input string s.
    •	Space complexity : O(1). Constant space is used.


            */
            public int NumWaysDecodeStringIIDPCS(string inputString)
            {
                long previousDecodings = 1;
                long currentDecodings = inputString[0] == '*' ? 9 : inputString[0] == '0' ? 0 : 1;

                for (int index = 1; index < inputString.Length; index++)
                {
                    long tempDecodings = currentDecodings;

                    if (inputString[index] == '*')
                    {
                        currentDecodings = 9 * currentDecodings % Modulus;
                    }
                    else
                    {
                        currentDecodings = (currentDecodings + previousDecodings) % Modulus;
                        if (inputString[index - 1] == '*')
                        {
                            currentDecodings = (currentDecodings + previousDecodings) % Modulus;
                        }
                    }

                    previousDecodings = tempDecodings;
                }

                return (int)currentDecodings;
            }

        }

        /*
                115. Distinct Subsequences
        https://leetcode.com/problems/distinct-subsequences/description/
        */

        public class NumDistinctSubseqSol
        {
            /*
            Approach 1: Recursion + Memoization

Complexity Analysis
•	Time Complexity: The time complexity for a recursive solution is defined by two things: the number of recursive calls that we make and the time it takes to process a single call.
o	If you notice the solution closely, all we are doing in the function is to check the dictionary for a key, and then we make a couple of function calls. So the time it takes to process a single call is actually O(1).
o	The number of unique recursive calls is defined by the two state variables that we have. Potentially, we can make O(M×N) calls where M and N represent the lengths of the two strings. Thus, the time complexity for this solution would be O(M×N).
•	Space Complexity: The maximum space is utilized by the dictionary that we are using and the size of that dictionary would also be controlled by the total possible combinations of i and j which turns out to be O(M×N) as well. We also have the space utilized by the recursion stack which is O(M) where M is the length of string S. This is because in one of our recursion calls, we don't progress at all in the string T. Hence, we would have a branch in the tree where only the index i progresses one step until it reaches the end of string S. The number of nodes in this branch would be equal to the length of string S.
            */

            public static int RecMemo(string s, string t)
            {
                Dictionary<string, int> memo;
                if (s.Length < t.Length)
                    return 0;
                if (s == t || t == "")
                    return 1;
                memo = new Dictionary<string, int>();
                return DistinctHelper(s.Substring(0, s.Length - 1), t) +
                       ((s[s.Length - 1] == t[t.Length - 1])
                            ? DistinctHelper(s.Substring(0, s.Length - 1),
                                             t.Substring(0, t.Length - 1))
                            : 0);

                int DistinctHelper(string s, string t)
                {
                    if (memo.ContainsKey(s + "," + t))
                        return memo[s + "," + t];
                    if (s.Length < t.Length)
                        return 0;
                    if (s == t || t == "")
                        return 1;
                    memo[s + "," + t] = DistinctHelper(s.Substring(0, s.Length - 1), t) +
                                        ((s[s.Length - 1] == t[t.Length - 1])
                                             ? DistinctHelper(s.Substring(0, s.Length - 1),
                                                              t.Substring(0, t.Length - 1))
                                             : 0);
                    return memo[s + "," + t];
                }

            }

            /*
            Approach 2: Iterative Dynamic Programming

Complexity Analysis
Time Complexity: The time complexity is much more clear in this approach since we have two for loops with clearly defined executions. The outer loop runs for M+1 iterations while the inner loop runs for N+1 iterations. So, combined together we have a time complexity of O(M×N).
Space Complexity: O(M×N) which is occupied by the 2D dp array that we create.

            */
            public int IterateDP(string s, string t)
            {
                int M = s.Length;
                int N = t.Length;
                int[,] dp = new int[M + 1, N + 1];
                // Base case initialization
                for (int j = 0; j <= N; j++) dp[M, j] = 0;
                // Base case initialization
                for (int i = 0; i <= M; i++) dp[i, N] = 1;
                // Iterate over the strings in reverse so as to
                // satisfy the way we've modeled our recursive solution
                for (int i = M - 1; i >= 0; i--)
                {
                    for (int j = N - 1; j >= 0; j--)
                    {
                        // Remember, we always need this result
                        dp[i, j] = dp[i + 1, j];
                        // If the characters match, we add the
                        // result of the next recursion call (in this
                        // case, the value of a cell in the dp table)
                        if (s[i] == t[j])
                            dp[i, j] += dp[i + 1, j + 1];
                    }
                }

                return dp[0, 0];
            }

            /*
            Approach 3: Space optimized Dynamic Programming
Complexity Analysis
Time Complexity: O(M×N)
Space Complexity: O(N) since we are using a single array which is the size of the string T. This is a major size reduction over the previous solution and this is a much more elegant solution than the initial recursive solution we saw earlier on.

            */
            public int IterateDPSpaceOptimal(string s, string t)
            {
                int M = s.Length;
                int N = t.Length;
                int[] dp = new int[N];
                int prev = 1;
                for (int i = M - 1; i >= 0; i--)
                {
                    prev = 1;
                    for (int j = N - 1; j >= 0; j--)
                    {
                        int old_dpj = dp[j];
                        if (s[i] == t[j])
                        {
                            dp[j] += prev;
                        }

                        prev = old_dpj;
                    }
                }

                return dp[0];
            }

        }


        /*
        127. Word Ladder
        https://leetcode.com/problems/word-ladder/description/

        */
        public class LadderLengthSol
        {
            /*
            Approach 1: Breadth First Search
            Complexity Analysis
•	Time Complexity: O(M2×N), where M is the length of each word and N is the total number of words in the input word list.
o	For each word in the word list, we iterate over its length to find all the intermediate words corresponding to it. Since the length of each word is M and we have N words, the total number of iterations the algorithm takes to create all_combo_dict is M×N. Additionally, forming each of the intermediate word takes O(M) time because of the substring operation used to create the new string. This adds up to a complexity of O(M2×N).
o	Breadth first search in the worst case might go to each of the N words. For each word, we need to examine M possible intermediate words/combinations. Notice, we have used the substring operation to find each of the combination. Thus, M combinations take O(M2) time. As a result, the time complexity of BFS traversal would also be O(M2×N).
Combining the above steps, the overall time complexity of this approach is O(M2×N).
•	Space Complexity: O(M2×N).
o	Each word in the word list would have M intermediate combinations. To create the all_combo_dict dictionary we save an intermediate word as the key and its corresponding original words as the value. Note, for each of M intermediate words we save the original word of length M. This simply means, for every word we would need a space of M2 to save all the transformations corresponding to it. Thus, all_combo_dict would need a total space of O(M2×N).
o	Visited dictionary would need a space of O(M×N) as each word is of length M.
o	Queue for BFS in worst case would need a space for all O(N) words and this would also result in a space complexity of O(M×N).
Combining the above steps, the overall space complexity is O(M2×N) + O(M∗N) + O(M∗N) = O(M2×N) space.
Optimization:
We can definitely reduce the space complexity of this algorithm by storing the indices corresponding to each word instead of storing the word itself.


            */
            public static int BFS(string beginWord, string endWord,
                         IList<string> wordList)
            {
                int L = beginWord.Length;
                Dictionary<string, List<string>> allComboDict =
                    new Dictionary<string, List<string>>();
                foreach (string word in wordList)
                {
                    for (int i = 0; i < L; i++)
                    {
                        string newWord = word.Substring(0, i) + '*' +
                                         word.Substring(i + 1, L - i - 1);
                        if (!allComboDict.ContainsKey(newWord))
                            allComboDict[newWord] = new List<string>();
                        allComboDict[newWord].Add(word);
                    }
                }

                Queue<Tuple<string, int>> Q = new Queue<Tuple<string, int>>();
                Q.Enqueue(new Tuple<string, int>(beginWord, 1));
                Dictionary<string, bool> visited = new Dictionary<string, bool>();
                visited[beginWord] = true;
                while (Q.Any())
                {
                    var node = Q.Dequeue();
                    string word = node.Item1;
                    int level = node.Item2;
                    for (int i = 0; i < L; i++)
                    {
                        string newWord = word.Substring(0, i) + '*' +
                                         word.Substring(i + 1, L - i - 1);
                        foreach (string adjacentWord in allComboDict.GetValueOrDefault(
                                     newWord, new List<string>()))
                        {
                            if (adjacentWord.Equals(endWord))
                                return level + 1;
                            if (!visited.ContainsKey(adjacentWord))
                            {
                                visited[adjacentWord] = true;
                                Q.Enqueue(
                                    new Tuple<string, int>(adjacentWord, level + 1));
                            }
                        }
                    }
                }

                return 0;
            }

            /*
            Approach 2: Bidirectional Breadth First Search
            Complexity Analysis
    •	Time Complexity: O(M2×N), where M is the length of words and N is the total number of words in the input word list. Similar to one directional, bidirectional also takes O(M2×N) time for finding out all the transformations. But the search time reduces to half, since the two parallel searches meet somewhere in the middle.
    •	Space Complexity: O(M2×N), to store all M transformations for each of the N words in the all_combo_dict dictionary, same as one directional. But bidirectional reduces the search space. It narrows down because of meeting in the middle.

            */




            public int LadderLength(string beginWord, string endWord,
                                    IList<string> wordList)
            {
                int L;
                Dictionary<string, List<string>> allComboDict;
                Queue<Tuple<string, int>> Q_begin;
                Queue<Tuple<string, int>> Q_end;
                Dictionary<string, int> visitedBegin;
                Dictionary<string, int> visitedEnd;
                if (!wordList.Contains(endWord))
                {
                    return 0;
                }

                L = beginWord.Length;
                allComboDict = new Dictionary<string, List<string>>();
                foreach (string word in wordList)
                {
                    for (int i = 0; i < L; i++)
                    {
                        string newWord = word.Substring(0, i) + '*' +
                                         word.Substring(i + 1, L - i - 1);
                        if (allComboDict.ContainsKey(newWord))
                        {
                            allComboDict[newWord].Add(word);
                        }
                        else
                        {
                            List<string> tempList = new List<string>();
                            tempList.Add(word);
                            allComboDict.Add(newWord, tempList);
                        }
                    }
                }

                Q_begin = new Queue<Tuple<string, int>>();
                Q_begin.Enqueue(new Tuple<string, int>(beginWord, 1));
                Q_end = new Queue<Tuple<string, int>>();
                Q_end.Enqueue(new Tuple<string, int>(endWord, 1));
                visitedBegin = new Dictionary<string, int> { { beginWord, 1 } };
                visitedEnd = new Dictionary<string, int> { { endWord, 1 } };
                while (Q_begin.Count != 0 && Q_end.Count != 0)
                {
                    int ans = -1;
                    if (Q_begin.Count <= Q_end.Count)
                    {
                        ans = VisitWordNode(Q_begin, visitedBegin,
                                                 visitedEnd);
                    }
                    else
                    {
                        ans = VisitWordNode(Q_end, visitedEnd,
                                                 visitedBegin);
                    }

                    if (ans > -1)
                    {
                        return ans;
                    }
                }

                return 0;

                int VisitWordNode(Queue<Tuple<string, int>> Q,
                                      Dictionary<string, int> visited,
                                      Dictionary<string, int> othersVisited)
                {
                    int x = Q.Count;
                    while (x > 0)
                    {
                        var node = Q.Dequeue();
                        string word = node.Item1;
                        int level = node.Item2;
                        for (int i = 0; i < L; i++)
                        {
                            string newWord = word.Substring(0, i) + '*' +
                                             word.Substring(i + 1, L - i - 1);
                            if (allComboDict.ContainsKey(newWord))
                            {
                                foreach (string adjacentWord in allComboDict[newWord])
                                {
                                    if (othersVisited.ContainsKey(adjacentWord))
                                    {
                                        return level + othersVisited[adjacentWord];
                                    }

                                    if (!visited.ContainsKey(adjacentWord))
                                    {
                                        visited.Add(adjacentWord, level + 1);
                                        Q.Enqueue(new Tuple<string, int>(adjacentWord,
                                                                         level + 1));
                                    }
                                }
                            }
                        }

                        x--;
                    }

                    return -1;
                }
            }

        }


        /*
        126. Word Ladder II
        https://leetcode.com/problems/word-ladder-ii/description/

        */
        public class FindLaddersSol
        {
            /*
            
Approach 1: Breadth-First Search (BFS) + Backtracking
Complexity Analysis
•	Time complexity: O(NK2+α).
Here N is the number of words in wordList, K is the maximum length of a word, α is the number of possible paths from beginWord to endWord in the directed graph we have.
Copying the wordList into the set will take O(N).
In BFS, every word will be traversed and for each word, we will find the neighbors using the function findNeighbors which has a time complexity of O(K2). Therefore the total complexity for all the N words will be O(NK2). Also, each word will be enqueued and will be removed from the set hence it will take O(N). The total time complexity of BFS will therefore be equal to O(NK2).
While backtracking, we will essentially be finding all the paths from beginWord to
endWord. Thus the time complexity will be equal to O(α).
We can estimate the upper bound for α by assuming that every layer except the first and the last layer in the DAG has x number of words and is fully connected to the next layer. Let h represent the height of the DAG, so the total number of paths will be xh (because we can choose any one word out of x words in each layer and each choice will be part of a valid shortest path that leads to the endWord). Here, h equals (N−2)/x. This would result in x(N−2)/x total paths, which is maximized when x=2.718, which we will round to 3 because x must be an integer. Thus the upper bound for α is 3(N/3), however, this is a very loose bound because the nature of this problem precludes the possibility of a DAG where every layer is fully connected to the next layer.
The total time complexity is therefore equal to O(NK2+α).
•	Space complexity: O(NK).
Here N is the Number of words in wordList, K is the Maximum length of a word.
Storing the words in a set will take O(NK) space.
To build the adjacency list O(N) space is required as the BFS will produce a directed
graph and hence there will be at max (N−1) edges.
In backtracking, stack space will be consumed which will be equal to the maximum number of active functions in the stack which is equal to the N as the path can have all the words in the wordList. Hence space required is O(N).
The total space complexity is therefore equal to O(NK).


            */


            public IList<IList<string>> BSFWithBacktrack(string beginWord, string endWord,
                                                    IList<string> wordList)
            {
                Dictionary<string, List<string>> adjList =
   new Dictionary<string, List<string>>();

                List<string> currPath = new List<string>();
                List<IList<string>> shortestPaths = new List<IList<string>>();
                // copying the words into the set for efficient deletion in BFS
                HashSet<string> copiedWordList = new HashSet<string>(wordList);
                BFS(beginWord, endWord, copiedWordList);

                // every path will start from the endWord
                currPath.Add(endWord);
                // traverse the DAG to find all the paths between endWord and beginWord
                Backtrack(endWord, beginWord);

                return shortestPaths;



                List<string> FindNeighbors(string word, HashSet<string> wordList)
                {
                    List<string> neighbors = new List<string>();
                    char[] charList = word.ToCharArray();
                    for (int i = 0; i < word.Length; i++)
                    {
                        char oldChar = charList[i];

                        // replace the i-th character with all letters from a to z except
                        // the original character
                        for (char c = 'a'; c <= 'z'; c++)
                        {
                            charList[i] = c;

                            // skip if the character is same as original or if the word is
                            // not present in the wordList
                            if (c == oldChar ||
                                !wordList.Contains(string.Join("", charList)))
                            {
                                continue;
                            }

                            neighbors.Add(string.Join("", charList));
                        }

                        charList[i] = oldChar;
                    }

                    return neighbors;
                }

                void Backtrack(string source, string destination)
                {
                    // store the path if we reached the endWord
                    if (source.Equals(destination))
                    {
                        List<string> tempPath = new List<string>(currPath);
                        tempPath.Reverse();
                        shortestPaths.Add(tempPath);
                    }

                    if (!adjList.ContainsKey(source))
                    {
                        return;
                    }

                    for (int i = 0; i < adjList[source].Count; i++)
                    {
                        currPath.Add(adjList[source][i]);
                        Backtrack(adjList[source][i], destination);
                        currPath.RemoveAt(currPath.Count - 1);
                    }
                }

                void BFS(string beginWord, string endWord,
                                HashSet<string> wordList)
                {
                    Queue<string> q = new Queue<string>();
                    q.Enqueue(beginWord);

                    // remove the root word which is the first layer in the BFS
                    if (wordList.Contains(beginWord))
                    {
                        wordList.Remove(beginWord);
                    }

                    Dictionary<string, int> isEnqueued = new Dictionary<string, int>();
                    isEnqueued[beginWord] = 1;

                    while (q.Count > 0)
                    {
                        List<string> visited = new List<string>();
                        for (int i = q.Count - 1; i >= 0; i--)
                        {
                            string currWord = q.Peek();
                            q.Dequeue();

                            // findNeighbors will have the adjacent words of the currWord
                            List<string> neighbors = FindNeighbors(currWord, wordList);
                            foreach (string word in neighbors)
                            {
                                visited.Add(word);
                                if (!adjList.ContainsKey(word))
                                {
                                    adjList[word] = new List<string>();
                                }

                                // add the edge from word to currWord in the list
                                adjList[word].Add(currWord);
                                if (!isEnqueued.ContainsKey(word))
                                {
                                    q.Enqueue(word);
                                    isEnqueued[word] = 1;
                                }
                            }
                        }

                        // removing the words of the previous layer
                        for (int i = 0; i < visited.Count; i++)
                        {
                            if (wordList.Contains(visited[i]))
                            {
                                wordList.Remove(visited[i]);
                            }
                        }
                    }
                }
            }

            /*
            Approach 2: Bidirectional Breadth-First Search (BFS) + Backtracking
Complexity Analysis
•	Time complexity: O(NK2+α).
Here N is the Number of words in wordList, K is the maximum length of a word, α is the Number of possible paths from beginWord to endWord in the directed graph we have.
Copying the wordList into the set will take O(N).
In the worst-case scenario, the number of operations in the bidirectional BFS will be equal to the BFS approach discussed before. However, in some cases, this approach will perform better because the search space is reduced by selecting the shorter queue at each iteration. In bidirectional BFS, at most, every word will be traversed once, and for each word, we will find the neighbors using the function findNeighbors which has a time complexity of O(K2). Therefore the total complexity for all the N words will be O(NK2). Also, each word will be enqueued and will be removed from the set which will take O(N). Thus, the total time complexity of bidirectional BFS will be O(NK22).
In the backtracking process, we will essentially find all of the paths from beginWord to endWord. Thus, the time complexity is equal to O(α).
We can estimate the upper bound for α by assuming that every layer except the first and the last layer in the DAG has x number of words and is fully connected to the next layer. Let h represent the height of the DAG, so the total number of paths will be xh (because we can choose any one word out of x words in each layer and each choice will be part of a valid shortest path that leads to the endWord). Here, h equals (N−2)/x. This would result in x(N−2)/x total paths, which is maximized when x=2.718, which we will round to 3 because x must be an integer. Thus the upper bound for α is 3(N/3), however, this is a very loose bound because the nature of this problem precludes the possibility of a DAG where every layer is fully connected to the next layer.
The total time complexity is therefore equal to O(NK2+α).
•	Space complexity: O(NK).
Here N is the Number of words in wordList, K is the Maximum length of a word.
Storing the words in a set will take O(NK) space.
To build the adjacency list O(N) space is required as the BFS will produce a directed graph and hence there will be at most (N−1) edges. Also, in the worst-case scenario, the combined size of both queues will be equal to N.
In backtracking, stack space will be consumed which will be equal to the maximum number of active functions in the stack, which is equal to the N as the path can have all the words in the wordList. Hence the space required is O(N).
The total space complexity is therefore equal to O(NK).

            */


            public IList<IList<string>> BidirectionBFSWithBacktrack(string beginWord, string endWord,
                                                    IList<string> wordList)
            {
                Dictionary<string, List<string>> adjList =
   new Dictionary<string, List<string>>();

                List<string> currPath = new List<string>();
                List<List<string>> shortestPaths = new List<List<string>>();
                HashSet<string> copiedWordList = new HashSet<string>(wordList);
                bool sequence_found = BFS(beginWord, endWord, copiedWordList);
                if (sequence_found == false)
                {
                    return shortestPaths.ToArray();
                }

                currPath.Add(beginWord);
                Backtrack(beginWord, endWord);
                return shortestPaths.ToArray();



                List<string> findNeighbors(string word, HashSet<string> wordList)
                {
                    List<string> neighbors = new List<string>();
                    char[] charList = word.ToCharArray();
                    for (int i = 0; i < word.Length; i++)
                    {
                        char oldChar = charList[i];
                        for (char c = 'a'; c <= 'z'; c++)
                        {
                            charList[i] = c;
                            if (c == oldChar || !wordList.Contains(new String(charList)))
                            {
                                continue;
                            }

                            neighbors.Add(new String(charList));
                        }

                        charList[i] = oldChar;
                    }

                    return neighbors;
                }

                void Backtrack(string source, string destination)
                {
                    if (source.Equals(destination))
                    {
                        List<string> tempPath = new List<string>(currPath);
                        shortestPaths.Add(tempPath);
                    }

                    if (!adjList.ContainsKey(source))
                    {
                        return;
                    }

                    for (int i = 0; i < adjList[source].Count; i++)
                    {
                        currPath.Add(adjList[source][i]);
                        Backtrack(adjList[source][i], destination);
                        currPath.RemoveAt(currPath.Count - 1);
                    }
                }

                void AddEdge(string word1, string word2, int direction)
                {
                    if (direction == 1)
                    {
                        if (!adjList.ContainsKey(word1))
                        {
                            adjList[word1] = new List<string>();
                        }

                        adjList[word1].Add(word2);
                    }
                    else
                    {
                        if (!adjList.ContainsKey(word2))
                        {
                            adjList[word2] = new List<string>();
                        }

                        adjList[word2].Add(word1);
                    }
                }

                bool BFS(string beginWord, string endWord,
                                HashSet<string> wordList)
                {
                    if (!wordList.Contains(endWord))
                    {
                        return false;
                    }

                    if (wordList.Contains(beginWord))
                    {
                        wordList.Remove(beginWord);
                    }

                    HashSet<string> forwardQueue = new HashSet<string>();
                    HashSet<string> backwardQueue = new HashSet<string>();
                    forwardQueue.Add(beginWord);
                    backwardQueue.Add(endWord);
                    bool found = false;
                    int direction = 1;
                    while (forwardQueue.Count != 0)
                    {
                        HashSet<string> visited = new HashSet<string>();
                        if (forwardQueue.Count > backwardQueue.Count)
                        {
                            HashSet<string> temp = forwardQueue;
                            forwardQueue = backwardQueue;
                            backwardQueue = temp;
                            direction ^= 1;
                        }

                        foreach (string currWord in forwardQueue)
                        {
                            List<string> neighbors = findNeighbors(currWord, wordList);
                            foreach (string word in neighbors)
                            {
                                if (backwardQueue.Contains(word))
                                {
                                    found = true;
                                    AddEdge(currWord, word, direction);
                                }
                                else if (!found && wordList.Contains(word) &&
                                           !forwardQueue.Contains(word))
                                {
                                    visited.Add(word);
                                    AddEdge(currWord, word, direction);
                                }
                            }
                        }

                        foreach (string currWord in forwardQueue)
                        {
                            if (wordList.Contains(currWord))
                            {
                                wordList.Remove(currWord);
                            }
                        }

                        if (found)
                        {
                            break;
                        }

                        forwardQueue = visited;
                    }

                    return found;
                }
            }
        }


        /* 943. Find the Shortest Superstring
        https://leetcode.com/problems/find-the-shortest-superstring/description/
         */
        public class ShortestSuperstringSol
        {
            /*
            
Approach 1: Dynamic Programming + Bit Masking

            Complexity Analysis
•	Time Complexity: O(N^2(2^N+W)), where N is the number of words, and W is the maximum length of each word.
•	Space Complexity: O(N(2^N+W)).

            */
            public string DPWithBitMasking(string[] strings)
            {
                int numberOfStrings = strings.Length;

                // Populate overlaps
                int[][] overlaps = new int[numberOfStrings][];
                for (int i = 0; i < numberOfStrings; ++i)
                    overlaps[i] = new int[numberOfStrings];

                for (int i = 0; i < numberOfStrings; ++i)
                    for (int j = 0; j < numberOfStrings; ++j)
                    {
                        if (i != j)
                        {
                            int minLength = Math.Min(strings[i].Length, strings[j].Length);
                            for (int k = minLength; k >= 0; --k)
                            {
                                if (strings[i].EndsWith(strings[j].Substring(0, k)))
                                {
                                    overlaps[i][j] = k;
                                    break;
                                }
                            }
                        }
                    }

                // dp[mask][i] = most overlap with mask, ending with ith element
                int[][] dp = new int[1 << numberOfStrings][];
                int[][] parent = new int[1 << numberOfStrings][];
                for (int mask = 0; mask < (1 << numberOfStrings); ++mask)
                {
                    dp[mask] = new int[numberOfStrings];
                    parent[mask] = new int[numberOfStrings];
                    Array.Fill(parent[mask], -1);

                    for (int bit = 0; bit < numberOfStrings; ++bit)
                    {
                        if (((mask >> bit) & 1) > 0)
                        {
                            // Let's try to find dp[mask][bit].  Previously, we had
                            // a collection of items represented by pmask.
                            int previousMask = mask ^ (1 << bit);
                            if (previousMask == 0) continue;
                            for (int i = 0; i < numberOfStrings; ++i)
                            {
                                if (((previousMask >> i) & 1) > 0)
                                {
                                    // For each bit i in previousMask, calculate the value
                                    // if we ended with word i, then added word 'bit'.
                                    int value = dp[previousMask][i] + overlaps[i][bit];
                                    if (value > dp[mask][bit])
                                    {
                                        dp[mask][bit] = value;
                                        parent[mask][bit] = i;
                                    }
                                }
                            }
                        }
                    }
                }

                // # Answer will have length sum(len(strings[i]) for i) - max(dp[-1])
                // Reconstruct answer, first as a sequence 'perm' representing
                // the indices of each word from left to right.

                int[] permutation = new int[numberOfStrings];
                bool[] seen = new bool[numberOfStrings];
                int currentIndex = 0;
                int currentMask = (1 << numberOfStrings) - 1;

                // p: the last element of permutation (last word written left to right)
                int p = 0;
                for (int j = 0; j < numberOfStrings; ++j)
                {
                    if (dp[(1 << numberOfStrings) - 1][j] > dp[(1 << numberOfStrings) - 1][p])
                        p = j;
                }

                // Follow parents down backwards path that retains maximum overlap
                while (p != -1)
                {
                    permutation[currentIndex++] = p;
                    seen[p] = true;
                    int previousP = parent[currentMask][p];
                    currentMask ^= 1 << p;
                    p = previousP;
                }

                // Reverse permutation
                for (int i = 0; i < currentIndex / 2; ++i)
                {
                    int temp = permutation[i];
                    permutation[i] = permutation[currentIndex - 1 - i];
                    permutation[currentIndex - 1 - i] = temp;
                }

                // Fill in remaining words not yet added
                for (int i = 0; i < numberOfStrings; ++i)
                {
                    if (!seen[i])
                        permutation[currentIndex++] = i;
                }

                // Reconstruct final answer given permutation
                System.Text.StringBuilder result = new System.Text.StringBuilder(strings[permutation[0]]);
                for (int i = 1; i < numberOfStrings; ++i)
                {
                    int overlap = overlaps[permutation[i - 1]][permutation[i]];
                    result.Append(strings[permutation[i]].Substring(overlap));
                }

                return result.ToString();
            }
        }


        /*   358. Rearrange String k Distance Apart
        https://leetcode.com/problems/rearrange-string-k-distance-apart/description/
         */

        public class RearrangeStringKDistanceApartSol
        {
            /* 
            Approach 1: Priority Queue - MaxHeap
Complexity Analysis
Here, N is the length of the string S, and K is the number of unique characters in the string S.
•	Time complexity: O((N+K)logK)
Creating a freq map will take O(K) time, and initializing the heap free will take O(KlogK). The main loop runs N times, and for each character, the operations are of O(K) as the heap size can be K at max. Hence this would need O(NlogK) time. Therefore, the total time complexity is equal to O((N+K)logK), but considering the problem constraints, the value of K can be 26 in the worst case; this can be simplified to O(N) as well.
•	Space complexity: O(K)
The size of map freq, heap free and the queue busy can be, at worst equal to K. Since the space to store the output is generally not considered part of space complexity, the total space complexity equals O(K).

             */
            public string UsingMaxHeapPQ(string inputString, int k)
            {
                Dictionary<char, int> characterFrequency = new Dictionary<char, int>();
                // Store the frequency for each character.
                foreach (char character in inputString.ToCharArray())
                {
                    if (characterFrequency.ContainsKey(character))
                    {
                        characterFrequency[character]++;
                    }
                    else
                    {
                        characterFrequency[character] = 1;
                    }
                }
                //TODO: Check below PQ logic wherther it acts as a MaxHeap based on Char frequency
                PriorityQueue<(int Frequency, char Character), int> availableCharacters =
                    new PriorityQueue<(int, char), int>(Comparer<int>.Create((x, y) => y.CompareTo(x)));

                // Insert the characters with their frequencies in the max heap.
                foreach (var entry in characterFrequency)
                {
                    availableCharacters.Enqueue((entry.Value, entry.Key), entry.Value);
                }

                System.Text.StringBuilder resultString = new System.Text.StringBuilder();
                // This queue stores the characters that cannot be used now.
                Queue<(int Index, char Character)> unavailableCharacters = new Queue<(int, char)>();

                while (resultString.Length != inputString.Length)
                {
                    int currentIndex = resultString.Length;

                    // Insert the character that could be used now into the available heap.
                    if (unavailableCharacters.Count > 0 && (currentIndex - unavailableCharacters.Peek().Index) >= k)
                    {
                        var characterToReinsert = unavailableCharacters.Dequeue();
                        availableCharacters.Enqueue((characterFrequency[characterToReinsert.Character], characterToReinsert.Character), characterFrequency[characterToReinsert.Character]);
                    }

                    // If the available heap is empty, it implies no character can be used at this index.
                    if (availableCharacters.Count == 0)
                    {
                        return "";
                    }

                    char currentCharacter = availableCharacters.Peek().Character;
                    availableCharacters.Dequeue();
                    resultString.Append(currentCharacter);

                    // Insert the used character into unavailable queue with the current index.
                    characterFrequency[currentCharacter]--;
                    if (characterFrequency[currentCharacter] > 0)
                    {
                        unavailableCharacters.Enqueue((currentIndex, currentCharacter));
                    }
                }

                return resultString.ToString();
            }
            /*
             Approach 2: Greedy
             Complexity Analysis
            Here, N is the length of the string S, and K is the number of unique characters in the string S.
            •	Time complexity: O(N)
            Creating map freqs and the hashset mostChars and secondChars can take at-max O(N) time. Iterate over the characters and insert their instances over the different segments; this will again cannot take more than O(N) time as the characters in mostChars and secondChars will be skipped. Hence, the total time complexity is O(N).
            •	Space complexity: O(K)
            The map freqs and the hashset mostChars and secondChars will take O(K) space. The rest of the space in the algorithm is used to store the output, which is not generally considered part of space complexity, and hence the space complexity is equal to O(K).

             */
            public string WithGreedy(string inputString, int k)
            {
                Dictionary<char, int> characterFrequencies = new Dictionary<char, int>();
                int maximumFrequency = 0;
                // Store the frequency, and find the highest frequency.
                foreach (char character in inputString.ToCharArray())
                {
                    if (characterFrequencies.ContainsKey(character))
                    {
                        characterFrequencies[character]++;
                    }
                    else
                    {
                        characterFrequencies[character] = 1;
                    }
                    maximumFrequency = Math.Max(maximumFrequency, characterFrequencies[character]);
                }

                HashSet<char> highestFrequencyChars = new HashSet<char>();
                HashSet<char> secondHighestFrequencyChars = new HashSet<char>();
                // Store all the characters with the highest and second-highest frequency - 1.
                foreach (var pair in characterFrequencies)
                {
                    if (pair.Value == maximumFrequency)
                    {
                        highestFrequencyChars.Add(pair.Key);
                    }
                    else if (pair.Value == maximumFrequency - 1)
                    {
                        secondHighestFrequencyChars.Add(pair.Key);
                    }
                }

                // Create maximumFrequency number of different strings.
                StringBuilder[] stringSegments = new StringBuilder[maximumFrequency];
                // Insert one instance of characters with frequency maxFreq & maxFreq - 1 in each segment.
                for (int index = 0; index < maximumFrequency; index++)
                {
                    stringSegments[index] = new StringBuilder();

                    foreach (char character in highestFrequencyChars)
                    {
                        stringSegments[index].Append(character);
                    }

                    // Skip the last segment as the frequency is only maximumFrequency - 1.
                    if (index < maximumFrequency - 1)
                    {
                        foreach (char character in secondHighestFrequencyChars)
                        {
                            stringSegments[index].Append(character);
                        }
                    }
                }

                int segmentIndex = 0;
                // Iterate over the remaining characters, and for each, distribute the instances over the segments.
                foreach (var pair in characterFrequencies)
                {
                    char character = pair.Key;
                    int frequency = pair.Value;

                    // Skip characters with maximumFrequency or maximumFrequency - 1 
                    // frequency as they have already been inserted.
                    if (highestFrequencyChars.Contains(character) || secondHighestFrequencyChars.Contains(character))
                    {
                        continue;
                    }

                    // Distribute the instances of these characters over the segments in a round-robin manner.
                    for (int count = frequency; count > 0; count--)
                    {
                        stringSegments[segmentIndex].Append(character);
                        segmentIndex = (segmentIndex + 1) % (maximumFrequency - 1);
                    }
                }

                // Each segment except the last should have exactly K elements; else, return "".
                for (int index = 0; index < maximumFrequency - 1; index++)
                {
                    if (stringSegments[index].Length < k)
                    {
                        return "";
                    }
                }

                // Join all the segments and return them.
                return string.Join("", stringSegments.AsEnumerable());
            }

        }


        /* 1153. String Transforms Into Another String
        https://leetcode.com/problems/string-transforms-into-another-string/description/
         */
        public class CanConvertSol
        {
            /* 
            Approach 1: Greedy + Hashing 
            Complexity Analysis
Here N is the length of string str1 or str2 and K is the maximum number of distinct characters in str1 or str2.
•	Time complexity: O(N)
We iterate over string str1. Then, we search or insert a key in a map that is O(1) for each character. Also, adding character to HashSet takes O(1). Hence the time complexity is O(N).
•	Space complexity: O(K)
The maximum possible number of mappings stored in the map is K. Additionally, HashSet will contain at most K characters. Since the maximum value of K is fixed at 26, we could consider the space complexity to be constant for this problem.

            */

            public bool CanConvert(string sourceString, string targetString)
            {
                if (sourceString.Equals(targetString))
                {
                    return true;
                }

                Dictionary<char, char> conversionMappings = new Dictionary<char, char>();
                HashSet<char> uniqueCharactersInTargetString = new HashSet<char>();

                // Make sure that no character in sourceString is mapped to multiple characters in targetString.
                for (int index = 0; index < sourceString.Length; index++)
                {
                    if (!conversionMappings.ContainsKey(sourceString[index]))
                    {
                        conversionMappings[sourceString[index]] = targetString[index];
                        uniqueCharactersInTargetString.Add(targetString[index]);
                    }
                    else if (conversionMappings[sourceString[index]] != targetString[index])
                    {
                        // this letter maps to 2 different characters, so sourceString cannot transform into targetString.
                        return false;
                    }
                }

                // No character in sourceString maps to 2 or more different characters in targetString and there
                // is at least one temporary character that can be used to break any loops.
                if (uniqueCharactersInTargetString.Count < 26)
                {
                    return true;
                }

                // The conversion mapping forms one or more cycles and there are no temporary 
                // characters that we can use to break the loops, so sourceString cannot transform into targetString.
                return false;
            }
        }

        /* 2759. Convert JSON String to Object
        https://leetcode.com/problems/convert-json-string-to-object/description/
         */


        /* 936. Stamping The Sequence
        https://leetcode.com/problems/stamping-the-sequence/description/
         */
        class MovesToStampSol
        {
            /* Approach 1: Work Backwards
            Complexity Analysis
•	Time Complexity: O(N(N−M)), where M,N are the lengths of stamp, target.
•	Space Complexity: O(N(N−M)).

             */
            public int[] UsingWorkBackwards(string stamp, string target)
            {
                int stampLength = stamp.Length, targetLength = target.Length;
                Queue<int> queue = new Queue<int>();
                bool[] isDone = new bool[targetLength];
                Stack<int> resultStack = new Stack<int>();
                List<Node> nodeList = new List<Node>();

                for (int i = 0; i <= targetLength - stampLength; ++i)
                {
                    // For each window [i, i+M), nodeList[i] will contain
                    // info on what needs to change before we can
                    // reverse stamp at this window.

                    HashSet<int> madeSet = new HashSet<int>();
                    HashSet<int> todoSet = new HashSet<int>();
                    for (int j = 0; j < stampLength; ++j)
                    {
                        if (target[i + j] == stamp[j])
                            madeSet.Add(i + j);
                        else
                            todoSet.Add(i + j);
                    }

                    nodeList.Add(new Node(madeSet, todoSet));

                    // If we can reverse stamp at i immediately,
                    // enqueue letters from this window.
                    if (todoSet.Count == 0)
                    {
                        resultStack.Push(i);
                        for (int j = i; j < i + stampLength; ++j)
                        {
                            if (!isDone[j])
                            {
                                queue.Enqueue(j);
                                isDone[j] = true;
                            }
                        }
                    }
                }

                // For each enqueued letter (position),
                while (queue.Count > 0)
                {
                    int currentIndex = queue.Dequeue();

                    // For each window that is potentially affected,
                    // j: start of window
                    for (int j = Math.Max(0, currentIndex - stampLength + 1); j <= Math.Min(targetLength - stampLength, currentIndex); ++j)
                    {
                        if (nodeList[j].Todo.Contains(currentIndex))
                        {  // This window is affected
                            nodeList[j].Todo.Remove(currentIndex);
                            if (nodeList[j].Todo.Count == 0)
                            {
                                resultStack.Push(j);
                                foreach (int m in nodeList[j].Made)
                                {
                                    if (!isDone[m])
                                    {
                                        queue.Enqueue(m);
                                        isDone[m] = true;
                                    }
                                }
                            }
                        }
                    }
                }

                for (int b = 0; b < isDone.Length; b++)
                {
                    if (!isDone[b]) return new int[0];
                }

                int[] resultArray = new int[resultStack.Count];
                int index = 0;
                while (resultStack.Count > 0)
                    resultArray[index++] = resultStack.Pop();

                return resultArray;
            }
            class Node
            {
                public HashSet<int> Made { get; set; }
                public HashSet<int> Todo { get; set; }

                public Node(HashSet<int> made, HashSet<int> todo)
                {
                    Made = made;
                    Todo = todo;
                }
            }
        }



        /* 899. Orderly Queue
        https://leetcode.com/problems/orderly-queue/description/
         */
        class OrderlyQueueSol
        {

            /* Approach 1: Mathematical
            Complexity Analysis
•	Time Complexity: O(N2), where N is the length of s.
o	If k = 1, we need O(N) time to build each new string and O(N) time to check whether it's the lexicographically smallest string among the strings generated so far. In total, there are N such different strings to build and check. Therefore, the time complexity for this case is O(N2).
o	If k > 1, we need to convert our given string to an array of characters (this costs O(N) time), then sort the newly obtained array (sorting takes O(NlogN) time), and build the output string from the sorted array which takes O(N) time.
o	Thus, the worst-case scenario is when k is 1, so the overall time complexity of the solution is O(N2).
•	Space Complexity: O(N).
o	If k = 1, we need the space to store only two strings: the lexicographically smallest string found so far and a newly built string, that will be compared to the lexicographically smallest string. This requires O(N) space.
o	If k > 1, we need O(N) space to store the character array. Other than that, sorting the array requires O(logN) additional space for Java and O(N) additional space for Python.
o	Therefore, the overall space complexity of the solution is O(N).

             */
            public String UsingMaths(String s, int k)
            {
                if (k == 1)
                {
                    String ans = s;
                    for (int i = 0; i < s.Length; ++i)
                    {
                        String temp = s.Substring(i) + s.Substring(0, i);
                        if (temp.CompareTo(ans) < 0)
                        {
                            ans = temp;
                        }
                    }
                    return ans;
                }
                else
                {
                    char[] chars = s.ToCharArray();
                    Array.Sort(chars);
                    return new String(chars);
                }
            }
        }

        /* 839. Similar String Groups________________________________________
        https://leetcode.com/problems/similar-string-groups/description/
         */
        class CountSimilarGroupsSol
        {
            /* Approach 1: Depth First Search
            Complexity Analysis
            Here n is the size of strs and m is length of each word in strs.
            •	Time complexity: O(n^2⋅m).
            o	To iterate over all the pairs of words that can be formed using strs, we need O(n^2) time. We also need O(m) time to determine whether the chosen two words are similar or not, which results in O(n^2⋅m) operations to check all the pairs.
            o	The dfs function visits each node once, which takes O(n) time because there are n nodes in total. We can have up to O(n^2) edges between n nodes (assume every word is similar to every other word). Because we have undirected edges, each edge can only be iterated twice (by nodes at the end), resulting in O(n^2) operations total in the worst-case scenario while visiting all nodes.
            •	Space complexity: O(n^2).
            o	As there can be a maximum of O(n^2) edges, building the adjacency list takes O(n^2) space.
            o	The visit array takes O(n) space.
            o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

             */
            public int DFS(string[] stringArray)
            {
                int arrayLength = stringArray.Length;
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                // Form the required graph from the given strings array.
                for (int outerIndex = 0; outerIndex < arrayLength; outerIndex++)
                {
                    for (int innerIndex = outerIndex + 1; innerIndex < arrayLength; innerIndex++)
                    {
                        if (AreStringsSimilar(stringArray[outerIndex], stringArray[innerIndex]))
                        {
                            if (!adjacencyList.ContainsKey(outerIndex))
                            {
                                adjacencyList[outerIndex] = new List<int>();
                            }
                            adjacencyList[outerIndex].Add(innerIndex);

                            if (!adjacencyList.ContainsKey(innerIndex))
                            {
                                adjacencyList[innerIndex] = new List<int>();
                            }
                            adjacencyList[innerIndex].Add(outerIndex);
                        }
                    }
                }

                bool[] visitedNodes = new bool[arrayLength];
                int connectedComponentsCount = 0;
                // Count the number of connected components.
                for (int index = 0; index < arrayLength; index++)
                {
                    if (!visitedNodes[index])
                    {
                        DepthFirstSearch(index, adjacencyList, visitedNodes);
                        connectedComponentsCount++;
                    }
                }

                return connectedComponentsCount;
            }
            private void DepthFirstSearch(int node, Dictionary<int, List<int>> adjacencyList, bool[] visitedNodes)
            {
                visitedNodes[node] = true;
                if (!adjacencyList.ContainsKey(node))
                {
                    return;
                }
                foreach (int neighbor in adjacencyList[node])
                {
                    if (!visitedNodes[neighbor])
                    {
                        visitedNodes[neighbor] = true;
                        DepthFirstSearch(neighbor, adjacencyList, visitedNodes);
                    }
                }
            }

            private bool AreStringsSimilar(string stringA, string stringB)
            {
                int differingCharactersCount = 0;
                for (int index = 0; index < stringA.Length; index++)
                {
                    if (stringA[index] != stringB[index])
                    {
                        differingCharactersCount++;
                    }
                }
                return differingCharactersCount == 0 || differingCharactersCount == 2;
            }

            /* Approach 2: Breadth First Search
            Complexity Analysis
            Here n is the size of strs and m is length of each word in strs.
            •	Time complexity: O(n^2⋅m).
            o	We need O(n^2) time to iterate over all the pairs of words that can be formed using strs. We further need O(m) time to check whether the chosen two words are similar or not, resulting in O(n^2⋅m) operations to check all the pairs.
            o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can only be pushed once, leading to O(n) operations for n nodes. As discussed above, we can have up to O(^n2) edges between n nodes (assume every word is similar to every other word). Because we have undirected edges, each edge can only be iterated twice (by nodes at the end), resulting in O(n2) operations total in the worst-case scenario while visiting all nodes.
            •	Space complexity: O(n^2).
            o	As there can be a maximum of O(n^2). edges, building the adjacency list takes O(n^2). space in the worst case.
            o	The BFS queue takes O(n) because each node is added at most once.
            o	The visit array takes O(n) space as well. 

             */
            public int BFS(string[] strings)
            {
                int numberOfStrings = strings.Length;
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                // Form the required graph from the given strings array.
                for (int i = 0; i < numberOfStrings; i++)
                {
                    for (int j = i + 1; j < numberOfStrings; j++)
                    {
                        if (IsSimilar(strings[i], strings[j]))
                        {
                            if (!adjacencyList.ContainsKey(i))
                            {
                                adjacencyList[i] = new List<int>();
                            }
                            adjacencyList[i].Add(j);
                            if (!adjacencyList.ContainsKey(j))
                            {
                                adjacencyList[j] = new List<int>();
                            }
                            adjacencyList[j].Add(i);
                        }
                    }
                }

                bool[] visited = new bool[numberOfStrings];
                int connectedComponentsCount = 0;
                // Count the number of connected components.
                for (int i = 0; i < numberOfStrings; i++)
                {
                    if (!visited[i])
                    {
                        Bfs(i, adjacencyList, visited);
                        connectedComponentsCount++;
                    }
                }

                return connectedComponentsCount;
            }
            private void Bfs(int node, Dictionary<int, List<int>> adjacencyList, bool[] visited)
            {
                Queue<int> queue = new Queue<int>();
                queue.Enqueue(node);
                visited[node] = true;
                while (queue.Count > 0)
                {
                    node = queue.Dequeue();
                    if (!adjacencyList.ContainsKey(node))
                    {
                        continue;
                    }
                    foreach (int neighbor in adjacencyList[node])
                    {
                        if (!visited[neighbor])
                        {
                            visited[neighbor] = true;
                            queue.Enqueue(neighbor);
                        }
                    }
                }
            }

            private bool IsSimilar(string a, string b)
            {
                int differenceCount = 0;
                for (int i = 0; i < a.Length; i++)
                {
                    if (a[i] != b[i])
                    {
                        differenceCount++;
                    }
                }
                return differenceCount == 0 || differenceCount == 2;
            }

            /* Approach 3: Union-find
            Complexity Analysis
            Here n is the size of strs and m is length of each word in strs.
            •	Time complexity: O(n^2⋅m).
            o	We need O(n^2) time to iterate over all the pairs of words that can be formed using strs. We further need O(m) time to check whether the chosen two words are similar or not, resulting in O(n^2⋅m) operations to check all the pairs.
            o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1).
            o	Initializing UnionFind takes O(n) time beacuse we are initializing the parent and rank arrays of size n each.
            o	We iterate through every edge and use the find operation to find the component of nodes connected by each edge. It takes O(1) per operation and takes O(e) time for all the e edges. As discussed above, we can have a maximum of O(n^2) edges in between n nodes, so it would take O(n^2) time. If nodes from different components are connected by an edge, we also perform union of the nodes, which takes O(1) time per operation. In the worst-case scenario, it may be called O(n) times to connect all the components to form a connected graph with only one component.
            •	Space complexity: O(n).
            o	We are using the parent and rank arrays, both of which require O(n) space each.

             */
            public int NumSimilarGroups(string[] strs)
            {
                int numberOfStrings = strs.Length;
                UnionFind disjointSetUnion = new UnionFind(numberOfStrings);
                int groupCount = numberOfStrings;
                // Form the required graph from the given strings array.
                for (int i = 0; i < numberOfStrings; i++)
                {
                    for (int j = i + 1; j < numberOfStrings; j++)
                    {
                        if (IsSimilar(strs[i], strs[j]) && disjointSetUnion.Find(i) != disjointSetUnion.Find(j))
                        {
                            groupCount--;
                            disjointSetUnion.UnionSet(i, j);
                        }
                    }
                }

                return groupCount;
            }
            public class UnionFind
            {
                private int[] parent;
                private int[] rank;

                public UnionFind(int size)
                {
                    parent = new int[size];
                    for (int i = 0; i < size; i++)
                        parent[i] = i;
                    rank = new int[size];
                }

                public int Find(int x)
                {
                    if (parent[x] != x)
                        parent[x] = Find(parent[x]);
                    return parent[x];
                }

                public void UnionSet(int x, int y)
                {
                    int xSet = Find(x), ySet = Find(y);
                    if (xSet == ySet)
                    {
                        return;
                    }
                    else if (rank[xSet] < rank[ySet])
                    {
                        parent[xSet] = ySet;
                    }
                    else if (rank[xSet] > rank[ySet])
                    {
                        parent[ySet] = xSet;
                    }
                    else
                    {
                        parent[ySet] = xSet;
                        rank[xSet]++;
                    }
                }
            }

        }


        /* 791. Custom Sort String
        https://leetcode.com/problems/custom-sort-string/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class CustomSortStringSol
        {
            /* Approach 1: Custom Comparator
            Complexity Analysis
Here, we define N as the length of string s, and K as the length of string order.
•	Time Complexity: O(NlogN)
Sorting an array of length N requires O(NlogN) time, and the indices of order have to be retrieved for each distinct letter, which results in an O(NlogN+K) complexity. K is at most 26, the number of unique English letters, so we can simplify the time complexity to O(NlogN).
•	Space Complexity: O(N) or O(log⁡N)
Note that some extra space is used when we sort arrays in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm, which has a space complexity of O(logN) for sorting two arrays. The Java solution also uses an auxiliary array of length N. This is the dominating term for the Java solution.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(log⁡N). This is the main space used by the C++ solution.

             */
            public string WithCustomComparator(string order, string inputString)
            {
                // Create char array for editing
                int length = inputString.Length;
                char[] result = new char[length];
                for (int i = 0; i < length; i++)
                {
                    result[i] = inputString[i];
                }

                // Define the custom comparator
                Array.Sort(result, (c1, c2) =>
                {
                    // The index of the character in order determines the value to be sorted by
                    return order.IndexOf(c1) - order.IndexOf(c2);
                });

                // Return the result
                string resultString = string.Empty;
                foreach (char c in result)
                {
                    resultString += c;
                }
                return resultString;
            }
            /* Approach 2: Frequency Table and Counting
            Complexity Analysis
Here, we define N as the length of string s, and K as the length of string order.
•	Time Complexity: O(N)
It takes O(N) time to populate the frequency table, and all other hashmap operations performed take O(1) time in the average case. Building the result string also takes O(N) time because each letter from s is appended to the result in the custom order, making the overall time complexity O(N).
•	Space Complexity: O(N)
A hash map and a result string are created, which results in an additional space complexity of O(N).

             */
            public string FreqTableAndCounting(string order, string s)
            {
                // Create a frequency table
                Dictionary<char, int> frequencyTable = new Dictionary<char, int>();

                // Initialize frequencies of letters
                // frequencyTable[c] = frequency of char c in s
                int stringLength = s.Length;
                for (int i = 0; i < stringLength; i++)
                {
                    char letter = s[i];
                    if (frequencyTable.ContainsKey(letter))
                    {
                        frequencyTable[letter]++;
                    }
                    else
                    {
                        frequencyTable[letter] = 1;
                    }
                }

                // Iterate order string to append to result
                int orderLength = order.Length;
                StringBuilder resultBuilder = new StringBuilder();
                for (int i = 0; i < orderLength; i++)
                {
                    char letter = order[i];
                    while (frequencyTable.ContainsKey(letter) && frequencyTable[letter] > 0)
                    {
                        resultBuilder.Append(letter);
                        frequencyTable[letter]--;
                    }
                }

                // Iterate through frequencyTable and append remaining letters
                // This is necessary because some letters may not appear in `order`
                foreach (var kvp in frequencyTable)
                {
                    char letter = kvp.Key;
                    int count = kvp.Value;
                    while (count > 0)
                    {
                        resultBuilder.Append(letter);
                        count--;
                    }
                }

                // Return the result
                return resultBuilder.ToString();
            }
        }

        /* 249. Group Shifted Strings
        https://leetcode.com/problems/group-shifted-strings/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class GroupShiftedStringsSol
        {/* 
            Approach 1: Hashing
            Complexity Analysis
Let N be the length of strings and K be the maximum length of a string in strings.
•	Time complexity: O(N∗K)
We iterate over all N strings and for each string, we iterate over all the characters to generate the Hash value, which takes O(K) time. To sum up, the overall time complexity is O(N∗K).
•	Space complexity: O(N∗K)
We need to store all the strings plus their Hash values in mapHashToList. In the worst scenario, when each string in the given list belongs to a different Hash value, the maximum number of strings stored in mapHashToList is 2∗N. Each string takes at most O(K) space. Hence the overall space complexity is O(N∗K).
Note: The time and space complexity for both solutions are same because the getHash() function has the same time and space complexity, O(K).

             */
            // Create a hash value
            private string GetHash(string inputString)
            {
                char[] characterArray = inputString.ToCharArray();
                StringBuilder hashKeyBuilder = new StringBuilder();

                for (int index = 1; index < characterArray.Length; index++)
                {
                    hashKeyBuilder.Append((char)((characterArray[index] - characterArray[index - 1] + 26) % 26 + 'a'));
                }

                return hashKeyBuilder.ToString();
            }

            public IList<IList<string>> GroupStrings(string[] inputStrings)
            {
                Dictionary<string, List<string>> hashToListMap = new Dictionary<string, List<string>>();

                // Create a hash_value (hashKey) for each string and append the string
                // to the list of hash values i.e. hashToListMap["cd"] = ["acf", "gil", "xzc"]
                foreach (string currentString in inputStrings)
                {
                    string hashKey = GetHash(currentString);
                    if (!hashToListMap.ContainsKey(hashKey))
                    {
                        hashToListMap[hashKey] = new List<string>();
                    }
                    hashToListMap[hashKey].Add(currentString);
                }

                // Iterate over the map, and add the values to groups
                List<IList<string>> groupedStrings = new List<IList<string>>();
                foreach (List<string> group in hashToListMap.Values)
                {
                    groupedStrings.Add(group);
                }

                // Return a list of all of the grouped strings
                return groupedStrings;
            }
        }

        /* 394. Decode String
        https://leetcode.com/problems/decode-string/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class DecodeStringSol
        {

            /* Approach 1: Using Stack
            Complexity Analysis
            •	Time Complexity: O(maxK^(countK)⋅n), where maxK is the maximum value of k, countK is the count of nested k values and n is the maximum length of encoded string.
            Example, for s = 20[a10[bc]], maxK is 20, countK is 2 as there are 2 nested k values (20 and 10) . Also, there are 2 encoded strings a and bc with maximum length of encoded string ,n as 2
            The worst case scenario would be when there are multiple nested patterns. Let's assume that all the k values (maxK) are 10 and all encoded string(n) are of size 2.
            For, s = 10[ab10[cd]]10[ef], time complexity would be roughly equivalent to 10∗cd∗10∗ab+10∗2=10^2∗2.
            Hence, for an encoded pattern of form maxK[nmaxK[n]], the time complexity to decode the pattern can be given as, O(maxK^(countK)⋅n).
            •	Space Complexity: O(sum(maxK^(countK)⋅n)), where maxK is the maximum value of k, countK is the count of nested k values and n is the maximum length of encoded string.
            The maximum stack size would be equivalent to the sum of all the decoded strings in the form maxK[nmaxK[n]]

             */
            public string UsingStack(string inputString)
            {
                Stack<char> characterStack = new Stack<char>();
                for (int index = 0; index < inputString.Length; index++)
                {
                    if (inputString[index] == ']')
                    {
                        List<char> decodedString = new List<char>();
                        // get the encoded string
                        while (characterStack.Peek() != '[')
                        {
                            decodedString.Add(characterStack.Pop());
                        }
                        // pop [ from the stack
                        characterStack.Pop();
                        int baseValue = 1;
                        int repeatCount = 0;
                        // get the number k
                        while (characterStack.Count > 0 && char.IsDigit(characterStack.Peek()))
                        {
                            repeatCount = repeatCount + (characterStack.Pop() - '0') * baseValue;
                            baseValue *= 10;
                        }
                        // decode k[decodedString], by pushing decodedString k times into stack
                        while (repeatCount != 0)
                        {
                            for (int j = decodedString.Count - 1; j >= 0; j--)
                            {
                                characterStack.Push(decodedString[j]);
                            }
                            repeatCount--;
                        }
                    }
                    // push the current character to stack
                    else
                    {
                        characterStack.Push(inputString[index]);
                    }
                }
                // get the result from stack
                char[] resultArray = new char[characterStack.Count];
                for (int i = resultArray.Length - 1; i >= 0; i--)
                {
                    resultArray[i] = characterStack.Pop();
                }
                return new string(resultArray);
            }
            /* Approach 2: Using 2 Stack
            Complexity Analysis
Assume, n is the length of the string s.
•	Time Complexity: O(maxK⋅n), where maxK is the maximum value of k and n is the length of a given string s. We traverse a string of size n and iterate k times to decode each pattern of form k[string]. This gives us worst case time complexity as O(maxK⋅n).
•	Space Complexity: O(m+n), where m is the number of letters(a-z) and n is the number of digits(0-9) in string s. In worst case, the maximum size of stringStack and countStack could be m and n respectively.

             */
            String UsingTwoStack(String s)
            {
                Stack<int> countStack = new();
                Stack<StringBuilder> stringStack = new();
                StringBuilder currentString = new StringBuilder();
                int k = 0;
                foreach (char ch in s)
                {
                    if (Char.IsDigit(ch))
                    {
                        k = k * 10 + ch - '0';
                    }
                    else if (ch == '[')
                    {
                        // push the number k to countStack
                        countStack.Push(k);
                        // push the currentString to stringStack
                        stringStack.Push(currentString);
                        // reset currentString and k
                        currentString = new StringBuilder();
                        k = 0;
                    }
                    else if (ch == ']')
                    {
                        StringBuilder decodedString = stringStack.Pop();
                        // decode currentK[currentString] by appending currentString k times
                        for (int currentK = countStack.Pop(); currentK > 0; currentK--)
                        {
                            decodedString.Append(currentString);
                        }
                        currentString = decodedString;
                    }
                    else
                    {
                        currentString.Append(ch);
                    }
                }
                return currentString.ToString();
            }
            /* 
Approach 3: Using Recursion
Complexity Analysis
Assume, n is the length of the string s.
•	Time Complexity: O(maxK⋅n) as in Approach 2
•	Space Complexity: O(n). This is the space used to store the internal call stack used for recursion. As we are recursively decoding each nested pattern, the maximum depth of recursive call stack would not be more than n

 */
            private int index = 0;

            public string UsingRecursion(string s)
            {
                StringBuilder result = new StringBuilder();
                while (index < s.Length && s[index] != ']')
                {
                    if (!char.IsDigit(s[index]))
                        result.Append(s[index++]);
                    else
                    {
                        int k = 0;
                        // build k while next character is a digit
                        while (index < s.Length && char.IsDigit(s[index]))
                            k = k * 10 + s[index++] - '0';
                        // ignore the opening bracket '['    
                        index++;
                        string decodedString = UsingRecursion(s);
                        // ignore the closing bracket ']'
                        index++;
                        // build k[decodedString] and append to the result
                        while (k-- > 0)
                            result.Append(decodedString);
                    }
                }
                return result.ToString();
            }

        }

        /* 1209. Remove All Adjacent Duplicates in String II
        https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        public class RemoveAllAdjacentDuplicatesSol
        {
            /* Approach 1: Brute Force
            Complexity Analysis
•	Time complexity: O(n^2/k), where n is a string length. We scan the string no more than n/k times.
•	Space complexity: O(1). A copy of a string may be created in some languages, however, the algorithm itself only uses the current string.

             */
            public string Naive(string s, int k)
            {
                StringBuilder sb = new StringBuilder(s);
                int length = -1;
                while (length != sb.Length)
                {
                    length = sb.Length;
                    for (int i = 0, count = 1; i < sb.Length; ++i)
                    {
                        if (i == 0 || sb[i] != sb[i - 1])
                        {
                            count = 1;
                        }
                        else if (++count == k)
                        {
                            sb.Remove(i - k + 1, i + 1);
                            break;
                        }
                    }
                }
                return sb.ToString();

            }
            /* Approach 2: Memoise Count
Complexity Analysis
•	Time complexity: O(n), where n is a string length. We process each character in the string once.
•	Space complexity: O(n) to store the count for each character.

             */
            public String UsingMemoiseCount(String s, int k)
            {
                StringBuilder sb = new StringBuilder(s);
                int[] count = new int[sb.Length];
                for (int i = 0; i < sb.Length; ++i)
                {
                    if (i == 0 || sb[i] != sb[i - 1])
                    {
                        count[i] = 1;
                    }
                    else
                    {
                        count[i] = count[i - 1] + 1;
                        if (count[i] == k)
                        {
                            sb.Remove(i - k + 1, i + 1);
                            i = i - k;
                        }
                    }
                }
                return sb.ToString();
            }/* 
Approach 3: Stack
Complexity Analysis
•	Time complexity: O(n), where n is a string length. We process each character in the string once.
•	Space complexity: O(n) for the stack.

 */
            public String UsingStack(String s, int k)
            {
                StringBuilder sb = new StringBuilder(s);
                Stack<int> counts = new Stack<int>();
                for (int i = 0; i < sb.Length; ++i)
                {
                    if (i == 0 || sb[i] != sb[i - 1])
                    {
                        counts.Push(1);
                    }
                    else
                    {
                        int incremented = counts.Pop() + 1;
                        if (incremented == k)
                        {
                            sb.Remove(i - k + 1, k);
                            i = i - k;
                        }
                        else
                        {
                            counts.Push(incremented);
                        }
                    }
                }
                return sb.ToString();

            }

            /* Approach 4: Stack with Reconstruction
            Complexity Analysis
            •	Time complexity: O(n), where n is a string length. We process each character in the string once.
            •	Space complexity: O(n) for the stack.

             */
            class Pair
            {
                public int Count { get; set; }
                public char Character { get; set; }

                public Pair(int count, char character)
                {
                    this.Character = character;
                    this.Count = count;
                }
            }

            public string UsingStackWithReconstruction(string inputString, int k)
            {
                Stack<Pair> counts = new Stack<Pair>();
                for (int i = 0; i < inputString.Length; ++i)
                {
                    if (counts.Count == 0 || inputString[i] != counts.Peek().Character)
                    {
                        counts.Push(new Pair(1, inputString[i]));
                    }
                    else
                    {
                        if (++counts.Peek().Count == k)
                        {
                            counts.Pop();
                        }
                    }
                }
                System.Text.StringBuilder stringBuilder = new System.Text.StringBuilder();
                while (counts.Count > 0)
                {
                    Pair pair = counts.Pop();
                    for (int i = 0; i < pair.Count; i++)
                    {
                        stringBuilder.Append(pair.Character);
                    }
                }
                return stringBuilder.ToString().Reverse().ToString(); ;
            }
            /* Approach 5: Two Pointers
            Complexity Analysis
            •	Time complexity: O(n), where n is a string length. We process each character in the string once.
            •	Space complexity: O(n) for the stack.

             */
            public String UsingTwoPointers(String s, int k)
            {
                Stack<int> counts = new();
                char[] sa = s.ToCharArray();
                int j = 0;
                for (int i = 0; i < s.Length; ++i, ++j)
                {
                    sa[j] = sa[i];
                    if (j == 0 || sa[j] != sa[j - 1])
                    {
                        counts.Push(1);
                    }
                    else
                    {
                        int incremented = counts.Pop() + 1;
                        if (incremented == k)
                        {
                            j = j - k;
                        }
                        else
                        {
                            counts.Push(incremented);
                        }
                    }
                }
                return new String(sa, 0, j);
            }

        }

        /* 
        424. Longest Repeating Character Replacement
        https://leetcode.com/problems/longest-repeating-character-replacement/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class CharacterReplacementSol
        {
            /* Approach 1: Sliding Window + Binary Search
Complexity Analysis
If there are n characters in the given string -
•	Time complexity: O(nlogn). Binary search divides the search space in half in each iteration until one element is left. So from n elements to reach 1 element it takes O(logn) iterations. We go through the full length of the string using a sliding window in every iteration. So it takes O(n) additional time per iteration. So the final time complexity is O(logn)∗O(n)=O(nlogn) .
•	Space complexity: O(m) where m is the number of distinct characters in the string. The core logic of binary search doesn't involve any auxiliary data structure but checking for valid string involves creating a hash map. The number of keys could be as many as the number of distinct characters. For uppercase English alphabets, m=26.

             */
            public int UsingBinarySearchAndSlidingWindow(String s, int k)
            {
                // binary search over the length of substring
                // lo contains the valid value, and hi contains the
                // invalid value
                int lo = 1;
                int hi = s.Length + 1;

                while (lo + 1 < hi)
                {
                    int mid = lo + (hi - lo) / 2;

                    // can we make a valid substring of length `mid`?
                    if (CanMakeValidSubstring(s, mid, k))
                    {
                        // explore the right half
                        lo = mid;
                    }
                    else
                    {
                        // explore the left half
                        hi = mid;
                    }
                }

                // length of the longest substring that satisfies
                // the given condition
                return lo;
            }

            private Boolean CanMakeValidSubstring(
                    String s,
                    int substringLength,
                    int k)
            {
                // take a window of length `substringLength` on the given
                // string, and move it from left to right. If this window
                // satisfies the condition of a valid string, then we return
                // true

                int[] freqMap = new int[26];
                int maxFrequency = 0;
                int start = 0;
                for (int end = 0; end < s.Length; end += 1)
                {
                    freqMap[s[end] - 'A'] += 1;

                    // if the window [start, end] exceeds substringLength
                    // then move the start pointer one step toward right
                    if (end + 1 - start > substringLength)
                    {
                        // before moving the pointer toward right, decrease
                        // the frequency of the corresponding character
                        freqMap[s[start] - 'A'] -= 1;
                        start += 1;
                    }

                    // record the maximum frequency seen so far
                    maxFrequency = Math.Max(maxFrequency, freqMap[s[end] - 'A']);
                    if (substringLength - maxFrequency <= k)
                    {
                        return true;
                    }
                }

                // we didn't a valid substring of the given size
                return false;
            }

            /* Approach 2: Sliding Window (Slow)
            Complexity Analysis
Let n be the number of characters in the string and m be the number of unique characters.
•	Time complexity: O(nm). We iterate over each unique character once, which requires O(k) time. We move a sliding window for each unique character from left to right of the string. As the window moves, each character of the string is visited at most two times. Once when it enters the window and again when it leaves the window. This adds O(n) time complexity for each iteration. So the final time complexity is O(nm). For all uppercase English letters, the maximum value of m would be 26.
•	Space complexity: O(m). We use an auxiliary set to store all unique characters, so the space complexity required here is O(m). Since there are only uppercase English letters in the string, m=26

             */
            public int UsingSlidingWindowSlow(string s, int k)
            {
                HashSet<char> allLetters = new HashSet<char>();

                // collect all unique letters
                for (int i = 0; i < s.Length; i++)
                {
                    allLetters.Add(s[i]);
                }

                int maxLength = 0;
                foreach (char letter in allLetters)
                {
                    int start = 0;
                    int count = 0;
                    // initialize a sliding window for each unique letter
                    for (int end = 0; end < s.Length; end += 1)
                    {
                        if (s[end] == letter)
                        {
                            // if the letter matches, increase the count
                            count += 1;
                        }
                        // bring start forward until the window is valid again
                        while (!IsWindowValid(start, end, count, k))
                        {
                            if (s[start] == letter)
                            {
                                // if the letter matches, decrease the count
                                count -= 1;
                            }
                            start += 1;
                        }
                        // at this point the window is valid, update maxLength
                        maxLength = Math.Max(maxLength, end + 1 - start);
                    }
                }
                return maxLength;
            }

            private bool IsWindowValid(int start, int end, int count, int k)
            {
                return end + 1 - start - count <= k;
            }

            /* Approach 3: Sliding Window (Fast)
Complexity Analysis
If there are n characters in the given string -
•	Time complexity: O(n). In this approach, we access each index of the string at most two times. When it is added to the sliding window, and when it is removed from the sliding window. The sliding window always moves forward. In each step, we update the frequency map, maxFrequency, and check for validity, they are all constant-time operations. To sum up, the time complexity is proportional to the number of characters in the string - O(n).
•	Space complexity: O(m). Similar to the previous approaches, this approach requires an auxiliary frequency map. The maximum number of keys in the map equals the number of unique characters in the string. If there are m unique characters, then the memory required is proportional to m. So the space complexity is O(m). Considering uppercase English letters only, m=26.

             */
            public int UsingSlidingWindowFast(String s, int k)
            {
                int start = 0;
                int[] frequencyMap = new int[26];
                int maxFrequency = 0;
                int longestSubstringLength = 0;

                for (int end = 0; end < s.Length; end += 1)
                {
                    // if 'A' is 0, then what is the relative order
                    // or offset of the current character entering the window
                    // 0 is 'A', 1 is 'B' and so on
                    int currentChar = s[end] - 'A';

                    frequencyMap[currentChar] += 1;

                    // the maximum frequency we have seen in any window yet
                    maxFrequency = Math.Max(maxFrequency, frequencyMap[currentChar]);

                    // move the start pointer towards right if the current
                    // window is invalid
                    Boolean isValid = (end + 1 - start - maxFrequency <= k);
                    if (!isValid)
                    {
                        // offset of the character moving out of the window
                        int outgoingChar = s[start] - 'A';

                        // decrease its frequency
                        frequencyMap[outgoingChar] -= 1;

                        // move the start pointer forward
                        start += 1;
                    }

                    // the window is valid at this point, note down the length
                    // size of the window never decreases
                    longestSubstringLength = end + 1 - start;
                }

                return longestSubstringLength;
            }

        }


        /* 151. Reverse Words in a String
         https://leetcode.com/problems/reverse-words-in-a-string/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
          */

        public class ReverseWordsSol
        {
            /*             Approach 1: Built-in Split + Reverse
            Complexity Analysis
            •	Time complexity: O(N), where N is the number of characters in the input string.
            •	Space complexity: O(N), to store the result of split by spaces.

             */
            public string UsingSplitAndReverse(string s)
            {
                // remove leading and trailing spaces
                s = s.Trim();
                // split by spaces and reverse
                string[] words = s.Split(new char[] { ' ' },
                    StringSplitOptions.RemoveEmptyEntries);
                Array.Reverse(words);
                // join the words with a space
                return String.Join(" ", words);
            }
            /* Approach 2: Reverse the Whole String and Then Reverse Each Word
Complexity Analysis
•	Time complexity: O(N).
•	Space complexity: O(N).

             */
            public string UsingReverseWholeAndEachWord(string s)
            {
                StringBuilder sb = TrimSpaces(s);
                // reverse the whole string
                Reverse(sb, 0, sb.Length - 1);
                // reverse each word
                ReverseEachWord(sb);
                return sb.ToString();
            }

            private StringBuilder TrimSpaces(string s)
            {
                int left = 0, right = s.Length - 1;
                // remove leading spaces
                while (left <= right && s[left] == ' ') ++left;
                // remove trailing spaces
                while (left <= right && s[right] == ' ') --right;
                // reduce multiple spaces to single one
                StringBuilder sb = new StringBuilder();
                while (left <= right)
                {
                    if (s[left] != ' ') sb.Append(s[left]);
                    else if (sb[sb.Length - 1] != ' ') sb.Append(s[left]);
                    ++left;
                }

                return sb;
            }

            private void ReverseEachWord(StringBuilder sb)
            {
                int n = sb.Length;
                int start = 0, end = 0;
                while (start < n)
                {
                    // go to the end of the word
                    while (end < n && sb[end] != ' ') ++end;
                    // reverse the word
                    Reverse(sb, start, end - 1);
                    // move to the next word
                    start = end + 1;
                    ++end;
                }
            }

            private void Reverse(StringBuilder sb, int left, int right)
            {
                while (left < right)
                {
                    char tmp = sb[left];
                    sb[left++] = sb[right];
                    sb[right--] = tmp;
                }
            }

            /* Approach 3: Deque of Words
Complexity Analysis
•	Time complexity: O(N).
•	Space complexity: O(N).

             */
            public string UsingDequeOfWords(string s)
            {
                int left = 0, right = s.Length - 1;
                while (left <= right && s[left] == ' ') ++left;
                while (left <= right && s[right] == ' ') --right;

                LinkedList<string> d = new LinkedList<string>();
                StringBuilder word = new StringBuilder();

                while (left <= right)
                {
                    if ((word.Length != 0) && (s[left] == ' '))
                    {
                        d.AddFirst(word.ToString());
                        word.Clear();
                    }
                    else if (s[left] != ' ')
                    {
                        word.Append(s[left]);
                    }

                    ++left;
                }

                d.AddFirst(word.ToString());

                return string.Join(" ", d);
            }
        }


        /* 767. Reorganize String
        https://leetcode.com/problems/reorganize-string/description/
         */
        class ReorganizeStringSol
        {
            /*             Approach 1: Counting and Priority Queue
            Complexity Analysis
            Let N be the total characters in the string.
            Let k be the total unique characters in the string.
            •	Time complexity: O(N⋅logk). We add one character to the string per iteration, so there are O(N) iterations. In each iteration, we perform a maximum of 3 priority queue operations. Each priority queue operation costs logk. For this problem, k is bounded by 26, so one could argue that the time complexity is actually O(N).
            •	Space complexity: O(k). The counter used to count the number of occurrences will incur a space complexity of O(k). Similarly, the maximum size of the priority queue will also be O(k). Given that k <= 26 in this problem, one could argue the space complexity is O(1).

             */
            public String UsingCoutingAndMaxHeapPQ(String s)
            {
                var charCounts = new int[26];
                foreach (char c in s)
                {
                    charCounts[c - 'a']++;
                }

                // Max heap ordered by character counts
                var maxHeap = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => b[1].CompareTo(a[1])));
                for (int i = 0; i < 26; i++)
                {
                    if (charCounts[i] > 0)
                    {
                        maxHeap.Enqueue(new int[] { i + 'a', charCounts[i] }, new int[] { i + 'a', charCounts[i] });
                    }
                }

                var sb = new StringBuilder();
                while (maxHeap.Count > 0)
                {
                    var first = maxHeap.Dequeue();
                    if (sb.Length == 0 || first[0] != sb[sb.Length - 1])
                    {
                        sb.Append((char)first[0]);
                        if (--first[1] > 0)
                        {
                            maxHeap.Enqueue(first, first);
                        }
                    }
                    else
                    {
                        if (maxHeap.Count == 0)
                        {
                            return "";
                        }

                        var second = maxHeap.Dequeue();
                        sb.Append((char)second[0]);
                        if (--second[1] > 0)
                        {
                            maxHeap.Enqueue(second, second);
                        }

                        maxHeap.Enqueue(first, first);
                    }
                }

                return sb.ToString();
            }
            /* Approach 2: Counting and Odd/Even	
Complexity Analysis
Let N be the total characters in the string.
Let k be the total unique characters in the string.
•	Time complexity: O(N). We will have to iterate over the entire string once to gather the counts of each character. Then, we we place each character in the answer which costs O(N).
•	Space complexity: O(k). The counter used to count the number of occurrences will incur a space complexity of O(k). Again, one could argue that because k <= 26, the space complexity is constant.

             */
            public String UsingCountingAndOddEOrEven(String s)
            {
                var charCounts = new int[26];
                foreach (char c in s)
                {
                    charCounts[c - 'a']++;
                }
                int maxCount = 0, letter = 0;
                for (int i = 0; i < charCounts.Length; i++)
                {
                    if (charCounts[i] > maxCount)
                    {
                        maxCount = charCounts[i];
                        letter = i;
                    }
                }
                if (maxCount > (s.Length + 1) / 2)
                {
                    return "";
                }
                var ans = new char[s.Length];
                int index = 0;

                // Place the most frequent letter
                while (charCounts[letter] != 0)
                {
                    ans[index] = (char)(letter + 'a');
                    index += 2;
                    charCounts[letter]--;
                }

                // Place rest of the letters in any order
                for (int i = 0; i < charCounts.Length; i++)
                {
                    while (charCounts[i] > 0)
                    {
                        if (index >= s.Length)
                        {
                            index = 1;
                        }
                        ans[index] = (char)(i + 'a');
                        index += 2;
                        charCounts[i]--;
                    }
                }

                return ans.ToString();
            }
        }


        /* 1653. Minimum Deletions to Make String Balanced
        https://leetcode.com/problems/minimum-deletions-to-make-string-balanced/description/
         */
        class MinimumDeletionsToMakeStringBalancedSol
        {

            /* Approach 1: Three-Pass Count 
            Complexity Analysis
            Let n be the length of the string s.
            •	Time complexity: O(n)
            The algorithm performs three linear passes over the string.
            •	Space complexity: O(n)
            We use two arrays of size n to store counts, resulting in linear space complexity.	

             */
            public int UsingThreePassCount(String s)
            {
                int n = s.Length;
                int[] countA = new int[n];
                int[] countB = new int[n];
                int bCount = 0;

                // First pass: compute count_b which stores the number of
                // 'b' characters to the left of the current position.
                for (int i = 0; i < n; i++)
                {
                    countB[i] = bCount;
                    if (s[i] == 'b') bCount++;
                }

                int aCount = 0;
                // Second pass: compute count_a which stores the number of
                // 'a' characters to the right of the current position
                for (int i = n - 1; i >= 0; i--)
                {
                    countA[i] = aCount;
                    if (s[i] == 'a') aCount++;
                }

                int minDeletions = n;
                // Third pass: iterate through the string to find the minimum deletions
                for (int i = 0; i < n; i++)
                {
                    minDeletions = Math.Min(minDeletions, countA[i] + countB[i]);
                }

                return minDeletions;
            }

            /* Approach 2: Combined Pass Method	
            Complexity Analysis
Let n be the length of the string s.
•	Time complexity: O(n)
The algorithm performs two linear passes over the string.
•	Space complexity: O(n)
We use one array of size n to store counts, resulting in linear space complexity.

             */
            public int UsingCombinedPass(String s)
            {
                int n = s.Length;
                int[] countA = new int[n];
                int aCount = 0;

                // First pass: compute count_a which stores the number of
                // 'a' characters to the right of the current position
                for (int i = n - 1; i >= 0; i--)
                {
                    countA[i] = aCount;
                    if (s[i] == 'a') aCount++;
                }

                int minDeletions = n;
                int bCount = 0;
                // Second pass: compute minimum deletions on the fly
                for (int i = 0; i < n; i++)
                {
                    minDeletions = Math.Min(countA[i] + bCount, minDeletions);
                    if (s[i] == 'b') bCount++;
                }

                return minDeletions;
            }
            /* Approach 3: Two-Variable Method
            Complexity Analysis
Let n be the length of the string s.
•	Time complexity: O(n)
The algorithm performs a single linear pass over the string.
•	Space complexity: O(1)
We only use constant space auxiliary variables, resulting in constant space complexity.

             */
            public int UsingTwoVariable(String s)
            {
                int n = s.Length;
                int aCount = 0;

                // First pass: count the number of 'a's
                for (int i = 0; i < n; i++)
                {
                    if (s[i] == 'a') aCount++;
                }

                int bCount = 0;
                int minDeletions = n;

                // Second pass: iterate through the string to compute minimum deletions
                for (int i = 0; i < n; i++)
                {
                    if (s[i] == 'a') aCount--;
                    minDeletions = Math.Min(minDeletions, aCount + bCount);
                    if (s[i] == 'b') bCount++;
                }

                return minDeletions;
            }
            /*             Approach 4: Using stack (one pass)
            Complexity Analysis
            Let n be the size of string s.
            •	Time complexity: O(n)
            The algorithm performs a single linear pass over the string, with stack operations (push and pop) taking O(1) time.
            •	Space complexity: O(n)
            The algorithm uses a stack that may grow up to the size of the string.

             */
            public int UsingStackOnePass(String s)
            {
                int n = s.Length;
                Stack<char> charStack = new();
                int deleteCount = 0;

                // Iterate through each character in the string
                for (int i = 0; i < n; i++)
                {
                    // If stack is not empty, top of stack is 'b',
                    // and current char is 'a'
                    if (
                        charStack.Count > 0 &&
                        charStack.Peek() == 'b' &&
                        s[i] == 'a'
                    )
                    {
                        charStack.Pop(); // Remove 'b' from stack
                        deleteCount++; // Increment deletion count
                    }
                    else
                    {
                        charStack.Push(s[i]); // Push current character onto stack
                    }
                }

                return deleteCount;
            }
            /* Approach 5: Using DP (One Pass)
            Complexity Analysis
            Let n be the size of string s.
            •	Time complexity: O(n)
            The algorithm performs a single linear pass over the string with updates to the dp array.
            •	Space complexity: O(n)
            The algorithm uses requires additional space for the dp array.

             */
            public int UsingDPOnePass(String s)
            {
                int n = s.Length;
                int[] dp = new int[n + 1];
                int bCount = 0;

                // dp[i]: The number of deletions required to
                // balance the substring s[0, i)
                for (int i = 0; i < n; i++)
                {
                    if (s[i] == 'b')
                    {
                        dp[i + 1] = dp[i];
                        bCount++;
                    }
                    else
                    {
                        // Two cases: remove 'a' or keep 'a'
                        dp[i + 1] = Math.Min(dp[i] + 1, bCount);
                    }
                }

                return dp[n];
            }
            /*             Approach 6: Optimized DP
Complexity Analysis
Let n be the size of string s.
•	Time complexity: O(n)
The algorithm performs a single linear pass over the string.
•	Space complexity: O(1)
The algorithm uses a constant amount of additional space for min_deletions and b_count.

             */
            public int UsingDPOptimal(String s)
            {
                int n = s.Length;
                int minDeletions = 0;
                int bCount = 0;

                // minDeletions variable represents dp[i]
                for (int i = 0; i < n; i++)
                {
                    if (s[i] == 'b')
                    {
                        bCount++;
                    }
                    else
                    {
                        // Two cases: remove 'a' or keep 'a'
                        minDeletions = Math.Min(minDeletions + 1, bCount);
                    }
                }

                return minDeletions;
            }
        }


        /* 3016. Minimum Number of Pushes to Type Word II
        https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-ii/description/
         */
        public class MinPushesToTypeWordIISol
        {
            /* 
            Approach 1: Greedy Sorting

            Complexity Analysis
            Let n be the length of the string.
            •	Time complexity: O(n)
            Iterating through the word string to count the frequency of each letter takes O(n).
            Sorting the frequency array, which has a fixed size of 26 (for each letter in the alphabet), takes O(1) because the size of the array is constant.
            Iterating through the frequency array to compute the total number of presses is O(1) because the array size is constant.
            Overall, the dominant term is O(n) due to the frequency counting step.
            •	Space complexity: O(1)
            Frequency array and sorting takes O(1) space, as it always requires space for 26 integers.
            Overall, the space complexity is O(1) because the space used does not depend on the input size.
             */
            public int WithGreedySorting(String word)
            {
                // Frequency array to store count of each letter
                int[] frequency = new int[26];

                // Count occurrences of each letter
                foreach (char c in word)
                {
                    frequency[c - 'a']++;
                }

                // Sort frequencies in descending order
                Array.Sort(frequency);
                int[] sortedFrequency = new int[26];
                for (int i = 0; i < 26; i++)
                {
                    sortedFrequency[i] = frequency[25 - i];
                }

                /*
                 Or do like this 
                 Sort frequencies in descending order
                Integer[] sortedFrequency = Arrays.stream(frequency).boxed().toArray(Integer[]::new);
                Arrays.sort(sortedFrequency, (a, b) -> b - a);
                */

                int totalPushes = 0;

                // Calculate total number of presses
                for (int i = 0; i < 26; i++)
                {
                    if (sortedFrequency[i] == 0) break;
                    totalPushes += (i / 8 + 1) * sortedFrequency[i];
                }
                return totalPushes;

            }
            /*             Approach 2: Using Heap
            Complexity Analysis
            Let n be the length of the string.
            •	Time complexity: O(n)
            Iterating through the word string to count the frequency of each letter takes O(n).
            Inserting each frequency into the priority queue and extracting the maximum frequency both operate with a time complexity of O(klogk), where k represents the number of distinct letters. Each of these operations—insertions, and extractions—is logarithmic due to the heap structure of the priority queue. However, since the number of distinct letters is limited to a maximum of 26 (one for each letter in the alphabet), the size of the priority queue remains constant and thus the time complexity effectively becomes O(1) in practice.
            Overall, the dominant term is O(n) due to the frequency counting step.
            •	Space complexity: O(1)
            The frequency map and priority queue take O(26)=O(1) space, as it always requires a fixed space for 26 integers.  
            Overall, the space complexity is O(1) because the space used does not depend on the input size.

             */
            public int UsingMaxHeapPQ(String word)
            {
                // Frequency map to store count of each letter
                Dictionary<char, int> frequencyMap = new();

                // Count occurrences of each letter
                foreach (char c in word)
                {
                    frequencyMap[c] = frequencyMap.GetValueOrDefault(c, 0) + 1;
                }

                // Priority queue to store frequencies in descending order
                PriorityQueue<int, int> frequencyQueue = new PriorityQueue<int, int>(
                    Comparer<int>.Create((a, b) => b - a
                ));
                foreach (int val in frequencyMap.Values)
                {
                    frequencyQueue.Enqueue(val, val);
                }

                int totalPushes = 0;
                int index = 0;

                // Calculate total number of presses
                while (frequencyQueue.Count > 0)
                {
                    totalPushes += (index / 8 + 1) * frequencyQueue.Dequeue();
                    index++;
                }

                return totalPushes;
            }

        }


        /* 2370. Longest Ideal Subsequence
        https://leetcode.com/problems/longest-ideal-subsequence/description/
         */
        class LongestIdealSubseqSol
        {

            /* 
            
Approach 1: Recursive Dynamic Programming (Top Down)

            Complexity Analysis
        Let N be the length of s and L be the number of letters in the English alphabet, which is 26.
        •	Time complexity: O(NL).
        In the main function, we check each possible ending letter of some subsequence, calling dfs() L times. The dfs() function recursively calls itself, and the total number of dfs() calls that run prior to memoizing is bounded by N⋅L, so this step takes O(NL+L), which is essentially O(NL).
        The loop inside the dfs() function makes up to 26 iterations. This loop is executed only if match is true, which is the case if c corresponds to the same ASCII value as the character s[i]. There is only one instance of c that fits this description for each distinct i, so this loop is executed at most once for each character in s. In other words, L transitions are executed only for N total states. Over the course of the whole search process, this loop executes up to O(NL) times.
        Therefore, the total time complexity is O(NL+NL), or O(2NL), which we can simplify to O(NL). Note that L is 26, which is a constant, so we could simplify the time complexity to O(N).
        •	Space complexity: O(NL).
        The additional space complexity is O(NL), since the two-dimensional dp grid needs to be initialized for memoization. L is 26, which is a constant, so we could simplify the time complexity to O(N).
         */
            public int TopDownDPRec(String s, int k)
            {
                int N = s.Length;

                // Initialize all dp values to -1 to indicate non-visited states
                int[][] dp = new int[N][];
                for (int i = 0; i < N; i++)
                {
                    Array.Fill(dp[i], -1);
                }

                // Find the maximum dp[N-1][c] and return the result
                int res = 0;
                for (int c = 0; c < 26; c++)
                {
                    res = Math.Max(res, Dfs(N - 1, c, dp, s, k));
                }
                return res;
            }

            private int Dfs(int i, int c, int[][] dp, String s, int k)
            {
                // Memoized value
                if (dp[i][c] != -1)
                {
                    return dp[i][c];
                }

                // State is not visited yet
                dp[i][c] = 0;
                bool match = c == (s[i] - 'a');
                if (match)
                {
                    dp[i][c] = 1;
                }

                // Non base case handling
                if (i > 0)
                {
                    dp[i][c] = Dfs(i - 1, c, dp, s, k);
                    if (match)
                    {
                        for (int p = 0; p < 26; p++)
                        {
                            if (Math.Abs(c - p) <= k)
                            {
                                dp[i][c] = Math.Max(dp[i][c], Dfs(i - 1, p, dp, s, k) + 1);
                            }
                        }
                    }
                }
                return dp[i][c];
            }

            /* Approach 2: Iterative Dynamic Programming (Bottom Up, Space Optimized)
Complexity Analysis
Let N be the length of s and L be the number of letters in the English alphabet, which is 26.
•	Time complexity: O(NL).
The outer loop iterates through the characters in s, so it runs N times. The inner loop iterates up to L times for each character in s. Therefore, the time complexity is O(NL). Note that L is 26, which is a constant, so we could simplify the time complexity to O(N).
•	Space complexity: O(L)
We use a DP array of size L. L is 26, which is a constant, so we could simplify the time complexity to O(1).

             */
            public int BottomUpDPSpaceOptimal(String s, int k)
            {
                int N = s.Length;
                int[] dp = new int[26];

                int res = 0;
                // Updating dp with the i-th character
                for (int i = 0; i < N; i++)
                {
                    int curr = s[i] - 'a';
                    int best = 0;
                    for (int prev = Math.Max(0, curr - k); prev < Math.Min(26, curr + k + 1); prev++)
                    {
                        best = Math.Max(best, dp[prev]);
                    }

                    // Append s[i] to the previous longest ideal subsequence
                    dp[curr] = best + 1;
                    res = Math.Max(res, dp[curr]);
                }

                return res;
            }
        }

        /* 1347. Minimum Number of Steps to Make Two Strings Anagram
        https://leetcode.com/problems/minimum-number-of-steps-to-make-two-strings-anagram/description/
         */
        class MinStepsToMakeTwoStringsAnagramSol
        {
            /* Approach: HashMap 
            Complexity Analysis
Here, N is the size of the string s and t.
•	Time complexity: O(N)
We are iterating over the indices of string s or t to find the frequencies in the array freq. Then we iterate over the integers from 0 to 26 to find the final answer. Hence, the total time complexity is equal to O(N).
•	Space complexity: O(1)
The only space required is the array count which has the constant size of 26. Therefore, the total space complexity is constant.	

            */
            public int UsingHashTable(String s, String t)
            {
                int[] count = new int[26];
                // Storing the difference of frequencies of characters in t and s.
                for (int i = 0; i < s.Length; i++)
                {
                    count[t[i] - 'a']++;
                    count[s[i] - 'a']--;
                }

                int ans = 0;
                // Adding the difference where string t has more instances than s.
                // Ignoring where t has fewer instances as they are redundant and
                // can be covered by the first case.
                for (int i = 0; i < 26; i++)
                {
                    ans += Math.Max(0, count[i]);
                }

                return ans;
            }
        }

        /* 316. Remove Duplicate Letters
        https://leetcode.com/problems/remove-duplicate-letters/description/
         */
        public class RemoveDuplicateLettersSol
        {


            /* Approach 1: Greedy - Solving Letter by Letter
Complexity Analysis
•	Time complexity : O(N). Each recursive call will take O(N). The number of recursive calls is bounded by a constant (26 letters in the alphabet), so we have O(N)∗C=O(N).
•	Space complexity : O(N). Each time we slice the string we're creating a new one (strings are immutable). The number of slices is bound by a constant, so we have O(N)∗C=O(N).

             */
            public String UsingGreedyLetterByLetter(String s)
            {
                // find pos - the index of the leftmost letter in our solution
                // we create a counter and end the iteration once the suffix doesn't have each unique character
                // pos will be the index of the smallest character we encounter before the iteration ends
                int[] cnt = new int[26];
                int pos = 0;
                for (int i = 0; i < s.Length; i++) cnt[s[i] - 'a']++;
                for (int i = 0; i < s.Length; i++)
                {
                    if (s[i] < s[pos]) pos = i;
                    if (--cnt[s[i] - 'a'] == 0) break;
                }
                // our answer is the leftmost letter plus the recursive call on the remainder of the string
                // note that we have to get rid of further occurrences of s[pos] to ensure that there are no duplicates
                return s.Length == 0 ? "" : s[pos] + UsingGreedyLetterByLetter(s.Substring(pos + 1).Replace("" + s[pos], ""));
            }
            /* Approach 2: Greedy - Solving with Stack
Complexity Analysis
•	Time complexity : O(N). Although there is a loop inside a loop, the time complexity is still O(N). This is because the inner while loop is bounded by the total number of elements added to the stack (each time it fires an element goes). This means that the total amount of time spent in the inner loop is bounded by O(N), giving us a total time complexity of O(N)
•	Space complexity : O(1). At first glance it looks like this is O(N), but that is not true! seen will only contain unique elements, so it's bounded by the number of characters in the alphabet (a constant). You can only add to stack if an element has not been seen, so stack also only consists of unique elements. This means that both stack and seen are bounded by constant, giving us O(1) space complexity.

             */
            public string UsingGreedyWithStack(string inputString)
            {
                Stack<char> characterStack = new Stack<char>();

                // this lets us keep track of what's in our solution in O(1) time
                HashSet<char> charactersSeen = new HashSet<char>();

                // this will let us know if there are any more instances of inputString[i] left in inputString
                Dictionary<char, int> lastOccurrence = new Dictionary<char, int>();
                for (int index = 0; index < inputString.Length; index++)
                {
                    lastOccurrence[inputString[index]] = index;
                }

                for (int index = 0; index < inputString.Length; index++)
                {
                    char currentCharacter = inputString[index];
                    // we can only try to add currentCharacter if it's not already in our solution
                    // this is to maintain only one of each character
                    if (!charactersSeen.Contains(currentCharacter))
                    {
                        // if the last letter in our solution:
                        //     1. exists
                        //     2. is greater than currentCharacter so removing it will make the string smaller
                        //     3. it's not the last occurrence
                        // we remove it from the solution to keep the solution optimal
                        while (characterStack.Count > 0 && currentCharacter < characterStack.Peek() && lastOccurrence[characterStack.Peek()] > index)
                        {
                            charactersSeen.Remove(characterStack.Pop());
                        }
                        charactersSeen.Add(currentCharacter);
                        characterStack.Push(currentCharacter);
                    }
                }
                System.Text.StringBuilder stringBuilder = new System.Text.StringBuilder(characterStack.Count);
                foreach (char character in characterStack)
                {
                    stringBuilder.Append(character);
                }
                return stringBuilder.ToString();
            }
        }





    }









}
