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
        public class stringChain
        {
            public string nextstring;
            public int maxChainLength;

            public stringChain(string nextstring, int maxChainLength)
            {
                this.nextstring = nextstring;
                this.maxChainLength = maxChainLength;
            }
        }

        // O(n * m^2 + nlog(n)) time | O(nm) space - where n is the number of strings
        public static void findLongeststringChain(
            string str, Dictionary<string, stringChain> stringChains
        )
        {
            // Try removing every letter of the current string to see if the
            // remaining strings form a string chain.
            for (int i = 0; i < str.Length; i++)
            {
                string smallerstring = getSmallerstring(str, i);
                if (!stringChains.ContainsKey(smallerstring)) continue;
                tryUpdateLongeststringChain(str, smallerstring, stringChains);
            }
        }

        public static string getSmallerstring(string str, int index)
        {
            return str.Substring(0, index) + str.Substring(index + 1);
        }

        public static void tryUpdateLongeststringChain(
            string currentstring,
            string smallerstring,
            Dictionary<string, stringChain> stringChains
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
            List<string> strings, Dictionary<string, stringChain> stringChains
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
        //https://www.algoexpert.io/questions/interweaving-strings

        // O(2^(n + m)) time | O(n + m) space - where n is the length
        // of the first string and m is the length of the second string
        public static bool InterweavingstringsNaive(string one, string two, string three)
        {
            if (three.Length != one.Length + two.Length)
            {
                return false;
            }

            return areInterwoven(one, two, three, 0, 0);
        }

        public static bool areInterwoven(
          string one, string two, string three, int i, int j
        )
        {
            int k = i + j;
            if (k == three.Length) return true;

            if (i < one.Length && one[i] == three[k])
            {
                if (areInterwoven(one, two, three, i + 1, j)) return true;
            }

            if (j < two.Length && two[j] == three[k])
            {
                return areInterwoven(one, two, three, i, j + 1);
            }

            return false;
        }
        // O(nm) time | O(nm) space - where n is the length of the
        // first string and m is the length of the second string
        public static bool InterweavingstringsOptimal(string one, string two, string three)
        {
            if (three.Length != one.Length + two.Length)
            {
                return false;
            }

            bool?[,] cache = new bool?[one.Length + 1, two.Length + 1];
            return AreInterwoven(one, two, three, 0, 0, cache);
        }

        public static bool AreInterwoven(
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
        //https://www.algoexpert.io/questions/palindrome-
        // O(n^2) time | O(n) space
        public static bool IsPalindromeNaive(string str)
        {
            string reversedstring = "";
            for (int i = str.Length - 1; i >= 0; i--)
            {
                reversedstring += str[i];
            }
            return str.Equals(reversedstring);
        }

        // O(n) time | O(n) space
        public static bool IsPalindrome1(string str)
        {
            StringBuilder reversedstring = new StringBuilder();
            for (int i = str.Length - 1; i >= 0; i--)
            {
                reversedstring.Append(str[i]);
            }
            return str.Equals(reversedstring.ToString());
        }

        // O(n) time | O(n) space
        public static bool IsPalindromeRec(string str)
        {
            return IsPalindromeRec(str, 0);
        }

        public static bool IsPalindromeRec(string str, int i)
        {
            int j = str.Length - 1 - i;
            return i >= j ? true : str[i] == str[j] && IsPalindromeRec(str, i + 1);
        }

        // O(n) time | O(1) space
        public static bool IsPalindromeOptimal(string str)
        {
            int leftIdx = 0;
            int rightIdx = str.Length - 1;
            while (leftIdx < rightIdx)
            {
                if (str[leftIdx] != str[rightIdx])
                {
                    return false;
                }
                leftIdx++;
                rightIdx--;
            }
            return true;
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

        //https://www.algoexpert.io/questions/longest-palindromic-substring
        // O(n^3) time | O(n) space
        public static string LongestPalindromicSubstringNaive(string str)
        {
            string longest = "";
            for (int i = 0; i < str.Length; i++)
            {
                for (int j = i; j < str.Length; j++)
                {
                    string substring = str.Substring(i, j + 1 - i);
                    if (substring.Length > longest.Length && IsPalindrome(substring))
                    {
                        longest = substring;
                    }
                }
            }
            return longest;
        }

        public static bool IsPalindrome(string str)
        {
            int leftIdx = 0;
            int rightIdx = str.Length - 1;
            while (leftIdx < rightIdx)
            {
                if (str[leftIdx] != str[rightIdx])
                {
                    return false;
                }
                leftIdx++;
                rightIdx--;
            }
            return true;
        }

        // O(n^2) time | O(n) space
        public static string LongestPalindromicSubstringOptimal(string str)
        {
            int[] currentLongest = { 0, 1 };
            for (int i = 1; i < str.Length; i++)
            {
                int[] odd = getLongestPalindromeFrom(str, i - 1, i + 1);
                int[] even = getLongestPalindromeFrom(str, i - 1, i);
                int[] longest = odd[1] - odd[0] > even[1] - even[0] ? odd : even;
                currentLongest =
                  currentLongest[1] - currentLongest[0] > longest[1] - longest[0]
                    ? currentLongest
                    : longest;
            }
            return str.Substring(
              currentLongest[0], currentLongest[1] - currentLongest[0]
            );
        }

        public static int[] getLongestPalindromeFrom(
          string str, int leftIdx, int rightIdx
        )
        {
            while (leftIdx >= 0 && rightIdx < str.Length)
            {
                if (str[leftIdx] != str[rightIdx])
                {
                    break;
                }
                leftIdx--;
                rightIdx++;
            }
            return new int[] { leftIdx + 1, rightIdx };
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


        //https://www.algoexpert.io/questions/palindrome-partitioning-min-cuts

        //1.
        // O(n^3) time | O(n^2) space
        public static int PalindromePartitioningMinCuts(string str)
        {
            bool[,] palindromes = new bool[str.Length, str.Length];
            for (int i = 0; i < str.Length; i++)
            {
                for (int j = i; j < str.Length; j++)
                {
                    palindromes[i, j] = IsPalindrome(str.Substring(i, j + 1 - i));
                }
            }
            int[] cuts = new int[str.Length];
            Array.Fill(cuts, Int32.MaxValue);
            for (int i = 0; i < str.Length; i++)
            {
                if (palindromes[0, i])
                {
                    cuts[i] = 0;
                }
                else
                {
                    cuts[i] = cuts[i - 1] + 1;
                    for (int j = 1; j < i; j++)
                    {
                        if (palindromes[j, i] && cuts[j - 1] + 1 < cuts[i])
                        {
                            cuts[i] = cuts[j - 1] + 1;
                        }
                    }
                }
            }
            return cuts[str.Length - 1];
        }
        //2.
        // O(n^2) time | O(n^2) space
        public static int PalindromePartitioningMinCutsOptimal(string str)
        {
            bool[,] palindromes = new bool[str.Length, str.Length];
            for (int i = 0; i < str.Length; i++)
            {
                for (int j = 0; j < str.Length; j++)
                {
                    if (i == j)
                    {
                        palindromes[i, j] = true;
                    }
                    else
                    {
                        palindromes[i, j] = false;
                    }
                }
            }
            for (int length = 2; length < str.Length + 1; length++)
            {
                for (int i = 0; i < str.Length - length + 1; i++)
                {
                    int j = i + length - 1;
                    if (length == 2)
                    {
                        palindromes[i, j] = (str[i] == str[j]);
                    }
                    else
                    {
                        palindromes[i, j] = (str[i] == str[j] && palindromes[i + 1, j - 1]);
                    }
                }
            }
            int[] cuts = new int[str.Length];
            Array.Fill(cuts, Int32.MaxValue);
            for (int i = 0; i < str.Length; i++)
            {
                if (palindromes[0, i])
                {
                    cuts[i] = 0;
                }
                else
                {
                    cuts[i] = cuts[i - 1] + 1;
                    for (int j = 1; j < i; j++)
                    {
                        if (palindromes[j, i] && cuts[j - 1] + 1 < cuts[i])
                        {
                            cuts[i] = cuts[j - 1] + 1;
                        }
                    }
                }
            }
            return cuts[str.Length - 1];
        }
        /*

        5. Longest Palindromic Substring
https://leetcode.com/problems/longest-palindromic-substring/description/	

        */
        public string LongestPalindrome(string s)
        {
            /*
            Approach 1: Check All Substrings

Complexity Analysis
Given n as the length of s,
•	Time complexity: O(n^3)
The two nested for loops iterate O(n2) times. We check one substring of length n, two substrings of length n - 1, three substrings of length n - 2, and so on.
There are n substrings of length 1, but we don't check them all since any substring of length 1 is a palindrome, and we will return immediately.
Therefore, the number of substrings that we check in the worst case is 1 + 2 + 3 + ... + n - 1. This is the partial sum of this series for n - 1, which is equal to 2n⋅(n−1)=O(n2).
In each iteration of the while loop, we perform a palindrome check. The cost of this check is linear with n as well, giving us a time complexity of O(n3).
Note that this time complexity is in the worst case and has a significant constant divisor that is dropped by big O. Due to the optimizations of checking the longer length substrings first and exiting the palindrome check early if we determine that a substring cannot be a palindrome, the practical runtime of this algorithm is not too bad.
•	Space complexity: O(1)
We don't count the answer as part of the space complexity. Thus, all we use are a few integer variables.

            */
            string longestPalindrome = LongestPalindromeNaive(s);
            /*
            Approach 2: Dynamic Programming
Given n as the length of s,
•	Time complexity: O(n^2)
We declare an n * n table dp, which takes O(n2) time. We then populate O(n2) states i, j - each state takes O(1) time to compute.
•	Space complexity: O(n^2)
The table dp takes O(n^2) space.

            
            */
            longestPalindrome = LongestPalindromeNaiveDP(s);
            /*
Approach 3: Expand From Centers
Given n as the length of s,
•	Time complexity: O(n^2)
There are 2n−1=O(n) centers. For each center, we call expand, which costs up to O(n).
Although the time complexity is the same as in the DP approach, the average/practical runtime of the algorithm is much faster. This is because most centers will not produce long palindromes, so most of the O(n) calls to expand will cost far less than n iterations.
The worst case scenario is when every character in the string is the same.
•	Space complexity: O(1)
We don't use any extra space other than a few integers. This is a big improvement on the DP approach.	


            */
            longestPalindrome = LongestPalindromeSpaceOptimal(s);
            /*

Approach 4: Manacher's Algorithm
Given n as the length of s,
•	Time complexity: O(n)
From Wikipedia (the implementation they describe is slightly different from the above code, but it's the same algorithm):
The algorithm runs in linear time. This can be seen by noting that Center strictly increases after each outer loop and the sum Center + Radius is non-decreasing. Moreover, the number of operations in the first inner loop is linear in the increase of the sum Center + Radius while the number of operations in the second inner loop is linear in the increase of Center. Since Center ≤ 2n+1 and Radius ≤ n, the total number of operations in the first and second inner loops is O(n) and the total number of operations in the outer loop, other than those in the inner loops, is also O(n). The overall running time is therefore O(n).
•	Space complexity: O(n)
We use sPrime and palindromeRadii, both of length O(n).

            */
            longestPalindrome = LongestPalindromeOptimal(s);

            return longestPalindrome;
        }

        private string LongestPalindromeOptimal(string s)
        {
            string s_prime = "#";
            foreach (char c in s)
            {
                s_prime += c;
                s_prime += "#";
            }

            int n = s_prime.Length;
            int[] palindromeRadii = new int[n];
            int center = 0;
            int radius = 0;

            for (int i = 0; i < n; i++)
            {
                int mirror = 2 * center - i;

                if (radius > i)
                    palindromeRadii[i] =
                        System.Math.Min(radius - i, palindromeRadii[mirror]);

                while (i + 1 + palindromeRadii[i] < n &&
                       i - 1 - palindromeRadii[i] >= 0 &&
                       s_prime[i + 1 + palindromeRadii[i]] ==
                           s_prime[i - 1 - palindromeRadii[i]])
                    palindromeRadii[i]++;

                if (i + palindromeRadii[i] > radius)
                {
                    center = i;
                    radius = i + palindromeRadii[i];
                }
            }

            int maxLength = 0;
            int centerIndex = 0;
            for (int i = 0; i < n; i++)
            {
                if (palindromeRadii[i] > maxLength)
                {
                    maxLength = palindromeRadii[i];
                    centerIndex = i;
                }
            }

            int startIndex = (centerIndex - maxLength) / 2;
            return s.Substring(startIndex, maxLength);
        }

        private string LongestPalindromeSpaceOptimal(string s)
        {
            int[] ans = new int[] { 0, 0 };

            for (int i = 0; i < s.Length; i++)
            {
                int oddLength = Expand(s, i, i);
                if (oddLength > ans[1] - ans[0] + 1)
                {
                    int dist = oddLength / 2;
                    ans[0] = i - dist;
                    ans[1] = i + dist;
                }

                int evenLength = Expand(s, i, i + 1);
                if (evenLength > ans[1] - ans[0] + 1)
                {
                    int dist = (evenLength / 2) - 1;
                    ans[0] = i - dist;
                    ans[1] = i + dist + 1;
                }
            }

            return s.Substring(ans[0], ans[1] - ans[0] + 1);

            int Expand(string s, int i, int j)
            {
                int left = i;
                int right = j;

                while (left >= 0 && right < s.Length && s[left] == s[right])
                {
                    left--;
                    right++;
                }

                return right - left - 1;
            }
        }

        private string LongestPalindromeNaiveDP(string s)
        {
            int n = s.Length;
            bool[,] dp = new bool[n, n];
            int[] ans = new int[] { 0, 0 };

            for (int i = 0; i < n; i++)
            {
                dp[i, i] = true;
            }

            for (int i = 0; i < n - 1; i++)
            {
                if (s[i] == s[i + 1])
                {
                    dp[i, i + 1] = true;
                    ans = new int[] { i, i + 1 };
                }
            }

            for (int diff = 2; diff < n; diff++)
            {
                for (int i = 0; i < n - diff; i++)
                {
                    int j = i + diff;
                    if (s[i] == s[j] && dp[i + 1, j - 1])
                    {
                        dp[i, j] = true;
                        ans = new int[] { i, j };
                    }
                }
            }

            int start = ans[0];
            int end = ans[1];
            return s.Substring(start, end - start + 1);
        }

        private string LongestPalindromeNaive(string s)
        {
            for (int length = s.Length; length > 0; length--)
            {
                for (int start = 0; start <= s.Length - length; start++)
                {
                    if (Check(start, start + length, s))
                    {
                        return s.Substring(start, length);
                    }
                }
            }

            return "";
            bool Check(int i, int j, string s)
            {
                int left = i;
                int right = j - 1;

                while (left < right)
                {
                    if (s[left] != s[right])
                    {
                        return false;
                    }

                    left++;
                    right--;
                }

                return true;
            }


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
        9. Palindrome Number
        https://leetcode.com/problems/palindrome-number/description/
        */
        public bool IsPalindrome(int x)
        {
            /*
            Approach 1: Revert half of the number

Complexity Analysis
•	Time complexity : O(log n base 10).
We divided the input by 10 for every iteration, so the time complexity is O(log10(n))
•	Space complexity : O(1).

            */

            // Special cases:
            // As discussed above, when x < 0, x is not a palindrome.
            // Also if the last digit of the number is 0, in order to be a
            // palindrome, the first digit of the number also needs to be 0. Only 0
            // satisfy this property.
            if (x < 0 || (x % 10 == 0 && x != 0))
            {
                return false;
            }

            int revertedNumber = 0;
            while (x > revertedNumber)
            {
                revertedNumber = revertedNumber * 10 + x % 10;
                x /= 10;
            }

            // When the length is an odd number, we can get rid of the middle digit
            // by revertedNumber/10 For example when the input is 12321, at the end
            // of the while loop we get x = 12, revertedNumber = 123, since the
            // middle digit doesn't matter in palidrome(it will always equal to
            // itself), we can simply get rid of it.
            return x == revertedNumber || x == revertedNumber / 10;
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
22. Generate Parentheses		
https://leetcode.com/problems/generate-parentheses/editorial/

        */
        public IList<string> GenerateParenthesis(int n)
        {
            /*
   Approach 1: Brute Force         
   Complexity Analysis
•	Time complexity: O(2^2n⋅n)
o	We are generating all possible strings of length 2n. At each character, we have two choices: choosing ( or ), which means there are a total of 22n unique strings.
o	For each string of length 2n, we need to iterate through each character to verify it is a valid combination of parentheses, which takes an average of O(n) time.
•	Space complexity: O(2^2n⋅n)
o	While we don't count the answer as part of the space complexity, for those interested, it is the nth Catalan number, which is asymptotically bounded by nn4n. Thus answer takes O(n4n) space.
Please find the explanation behind this intuition in approach 3!
You can also refer to Catalan number on Wikipedia for more information about Catalan numbers.
o	Before we dequeue the first string of length 2n from queue, it has stored 22n−1 strings of length n−1, which takes O(22n⋅n).
o	To sum up, the overall space complexity is O(22n⋅n).

            */
            IList<string> parenths = GenerateParenthesisNaive(n);

            /*
Approach 2: Backtracking, Keep Candidate Valid
Complexity Analysis
•	Time complexity: O(4^n/root n)
o	We only track the valid prefixes during the backtracking procedure. As explained in the approach 1 time complexity analysis, the total number of valid parentheses strings is O(nn4n).
Please find the explanation behind this intuition in approach 3!
You can also refer to Catalan number on Wikipedia for more information about Catalan numbers.
o	When considering each valid string, it is important to note that we use a mutable instance (StringBuilder in Java, list in Python etc.). As a result, in order to add each instance of a valid string to answer, we must first convert it to a string. This process brings an additional n factor in the time complexity.
•	Space complexity: O(n)
o	The space complexity of a recursive call depends on the maximum depth of the recursive call stack, which is 2n. As each recursive call either adds a left parenthesis or a right parenthesis, and the total number of parentheses is 2n. Therefore, at most O(n) levels of recursion will be created, and each level consumes a constant amount of space.


            
            */
            parenths = GenerateParenthesisBacktrack(n);
            /*
Approach 3: Divide and Conquer (DAC)
 Complexity Analysis
•	Time complexity: O(4^n/root n)
o	We begin by generating all valid parentheses strings of length 2, 4, ..., 2n. As shown in approach 2, 
the time complexity for generating all valid parentheses strings of length 2n is given by the expression O(n4n). Therefore, the total time complexity can be expressed T(n)=i=1∑ni4i which is asymptotically bounded by O(n4n).
•	Space complexity: O(n)
o	We don't count the answer as part of the space complexity, so the space complexity would be the maximum depth of the recursion stack. At any given time, the recursive function call stack would contain at most n function calls.
         
            
            */
            parenths = GenerateParenthesisDAC(n);

            return parenths;


        }
        public IList<string> GenerateParenthesisDAC(int n)
        {
            if (n == 0)
            {
                return new List<string> { "" };
            }

            List<string> answer = new List<string>();
            for (int leftCount = 0; leftCount < n; ++leftCount)
            {
                foreach (string leftString in GenerateParenthesisDAC(leftCount))
                {
                    foreach (string rightString in GenerateParenthesisDAC(n - 1 -
                                                                       leftCount))
                    {
                        answer.Add("(" + leftString + ")" + rightString);
                    }
                }
            }

            return answer;
        }
        public IList<string> GenerateParenthesisBacktrack(int n)
        {
            List<string> answer = new List<string>();
            Backtracking(answer, new StringBuilder(), 0, 0, n);
            return answer;
        }

        private void Backtracking(List<string> answer, StringBuilder curString,
                                  int leftCount, int rightCount, int n)
        {
            if (curString.Length == 2 * n)
            {
                answer.Add(curString.ToString());
                return;
            }

            if (leftCount < n)
            {
                curString.Append("(");
                Backtracking(answer, curString, leftCount + 1, rightCount, n);
                curString.Remove(curString.Length - 1, 1);
            }

            if (leftCount > rightCount)
            {
                curString.Append(")");
                Backtracking(answer, curString, leftCount, rightCount + 1, n);
                curString.Remove(curString.Length - 1, 1);
            }
        }
        private bool IsValid(string pString)
        {
            int leftCount = 0;
            foreach (char p in pString.ToCharArray())
            {
                if (p == '(')
                {
                    leftCount++;
                }
                else
                {
                    leftCount--;
                }

                if (leftCount < 0)
                {
                    return false;
                }
            }

            return leftCount == 0;
        }

        public IList<string> GenerateParenthesisNaive(int n)
        {
            IList<string> answer = new List<string>();
            Queue<string> queue = new Queue<string>();
            queue.Enqueue("");
            while (queue.Count != 0)
            {
                string curString = queue.Dequeue();
                if (curString.Length == 2 * n)
                {
                    if (IsValid(curString))
                    {
                        answer.Add(curString);
                    }

                    continue;
                }

                queue.Enqueue(curString + ")");
                queue.Enqueue(curString + "(");
            }

            return answer;
        }

        /*
20. Valid Parentheses 
https://leetcode.com/problems/valid-parentheses/description/	

Approach1 : Using Stacks
Complexity analysis
•	Time complexity : O(n) because we simply traverse the given string one character at a time and push and pop operations on a stack take O(1) time.
•	Space complexity : O(n) as we push all opening brackets onto the stack and in the worst case, we will end up pushing all the brackets onto the stack. e.g. ((((((((((.

        */
        private Dictionary<char, char> mappings;
        public bool IsParenValid(string s)
        {
            mappings = new Dictionary<char, char> {
            { ')', '(' }, { '}', '{' }, { ']', '[' } };
            var stack = new Stack<char>();
            foreach (var c in s)
            {
                if (mappings.ContainsKey(c))
                {
                    char topElement = stack.Count == 0 ? '#' : stack.Pop();
                    if (topElement != mappings[c])
                    {
                        return false;
                    }
                }
                else
                {
                    stack.Push(c);
                }
            }

            return stack.Count == 0;
        }

        /*
32. Longest Valid Parentheses
https://leetcode.com/problems/longest-valid-parentheses/description/

        */
        public int LongestValidParentheses(string s)
        {
            /*
Approach 1: Brute Force
Complexity Analysis
•	Time complexity: O(n^3). Generating every possible substring from a string of length n requires O(n^2). Checking validity of a string of length n requires O(n).
•	Space complexity: O(n). A stack of depth n will be required for the longest substring.
           
            */
            int maxLen = LongestValidParenthesesNaive(s);
            /*
 Approach 2: Using Dynamic Programming           
 Complexity Analysis
•	Time complexity: O(n). Single traversal of string to fill dp array is done.
•	Space complexity: O(n). dp array of size n is used.
           
            */

            maxLen = LongestValidParenthesesDP(s);
            /*            
 Approach 3: Using Stack           
 Complexity Analysis
•	Time complexity: O(n). n is the length of the given string.
•	Space complexity: O(n). The size of stack can go up to n.

            */
            maxLen = LongestValidParenthesesStack(s);

            /*
Approach 4: Without extra space/Two Pass
Complexity Analysis
•	Time complexity: O(n). Two traversals of the string.
•	Space complexity: O(1). Only two extra variables left and right are needed.

            
            */
            maxLen = LongestValidParenthesesTwoPass(s);

            return maxLen;
        }

        public int LongestValidParenthesesNaive(string s)
        {
            int maxlen = 0;
            for (int i = 0; i < s.Length; i++)
            {
                for (int j = i + 2; j <= s.Length; j += 2)
                {
                    if (IsValid(s.Substring(i, j - i)))
                    {
                        maxlen = Math.Max(maxlen, j - i);
                    }
                }
            }

            return maxlen;
            bool IsValid(string s)
            {
                Stack<char> stack = new Stack<char>();
                for (int i = 0; i < s.Length; i++)
                {
                    if (s[i] == '(')
                    {
                        stack.Push('(');
                    }
                    else if (stack.Count > 0 && stack.Peek() == '(')
                    {
                        stack.Pop();
                    }
                    else
                    {
                        return false;
                    }
                }

                return stack.Count == 0;
            }

        }
        public int LongestValidParenthesesDP(string s)
        {
            int maxans = 0;
            int[] dp = new int[s.Length];
            for (int i = 1; i < s.Length; i++)
            {
                if (s[i] == ')')
                {
                    if (s[i - 1] == '(')
                    {
                        dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                    }
                    else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(')
                    {
                        dp[i] = dp[i - 1] +
                                ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) +
                                2;
                    }

                    maxans = Math.Max(maxans, dp[i]);
                }
            }

            return maxans;
        }
        public int LongestValidParenthesesStack(string s)
        {
            int maxans = 0;
            Stack<int> stack = new Stack<int>();
            stack.Push(-1);
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '(')
                {
                    stack.Push(i);
                }
                else
                {
                    stack.Pop();
                    if (stack.Count == 0)
                    {
                        stack.Push(i);
                    }
                    else
                    {
                        maxans = Math.Max(maxans, i - stack.Peek());
                    }
                }
            }

            return maxans;
        }
        public int LongestValidParenthesesTwoPass(string s)
        {
            int left = 0, right = 0, maxlength = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '(')
                {
                    left++;
                }
                else
                {
                    right++;
                }

                if (left == right)
                {
                    maxlength = Math.Max(maxlength, 2 * right);
                }
                else if (right > left)
                {
                    left = right = 0;
                }
            }

            left = right = 0;
            for (int i = s.Length - 1; i >= 0; i--)
            {
                if (s[i] == '(')
                {
                    left++;
                }
                else
                {
                    right++;
                }

                if (left == right)
                {
                    maxlength = Math.Max(maxlength, 2 * left);
                }
                else if (left > right)
                {
                    left = right = 0;
                }
            }

            return maxlength;
        }

        /*
        301. Remove Invalid Parentheses
https://leetcode.com/problems/remove-invalid-parentheses/description/	
        */

        public IList<string> RemoveInvalidParentheses(string s)
        {
            /*
            Approach 1: Backtracking
Complexity analysis
•	Time Complexity : O(2^N) since in the worst case we will have only left parentheses in the expression and for every bracket we will have two options i.e. whether to remove it or consider it. Considering that the expression has N parentheses, the time complexity will be O(2^N).
•	Space Complexity : O(N) because we are resorting to a recursive solution and for a recursive solution there is always stack space used as internal function states are saved onto a stack during recursion. The maximum depth of recursion decides the stack space used. Since we process one character at a time and the base case for the recursion is when we have processed all of the characters of the expression string, the size of the stack would be O(N). Note that we are not considering the space required to store the valid expressions. We only count the intermediate space here.
            */
            IList<string> validParens = RemoveInvalidParenthesesBacktrack(s);

            /*
 Approach 2: Limited Backtracking!           
Complexity analysis
•	Time Complexity : The optimization that we have performed is simply a better form of pruning. Pruning here is something that will vary from one test case to another. In the worst case, we can have something like ((((((((( and the left_rem = len(S) and in such a case we can discard all of the characters because all are misplaced. So, in the worst case we still have 2 options per parenthesis and that gives us a complexity of O(2^N).
•	Space Complexity : The space complexity remains the same i.e. O(N) as previous solution. We have to go to a maximum recursion depth of N before hitting the base case. Note that we are not considering the space required to store the valid expressions. We only count the intermediate space here.

            */
            validParens = RemoveInvalidParenthesesLimitedBacktrack(s);

            return validParens;

        }
        private HashSet<string> validExpressions = new HashSet<string>();
        private int minimumRemoved;

        private void Reset()
        {
            this.validExpressions.Clear();
            this.minimumRemoved = int.MaxValue;
        }

        private void Recurse(
            string inputString,
            int index,
            int leftCount,
            int rightCount,
            StringBuilder expression,
            int removedCount)
        {
            // If we have reached the end of string.
            if (index == inputString.Length)
            {
                // If the current expression is valid.
                if (leftCount == rightCount)
                {
                    // If the current count of removed parentheses is <= the current minimum count
                    if (removedCount <= this.minimumRemoved)
                    {
                        // Convert StringBuilder to a String. This is an expensive operation.
                        // So we only perform this when needed.
                        string possibleAnswer = expression.ToString();

                        // If the current count beats the overall minimum we have till now
                        if (removedCount < this.minimumRemoved)
                        {
                            this.validExpressions.Clear();
                            this.minimumRemoved = removedCount;
                        }
                        this.validExpressions.Add(possibleAnswer);
                    }
                }
            }
            else
            {
                char currentCharacter = inputString[index];
                int length = expression.Length;

                // If the current character is neither an opening bracket nor a closing one,
                // simply recurse further by adding it to the expression StringBuilder
                if (currentCharacter != '(' && currentCharacter != ')')
                {
                    expression.Append(currentCharacter);
                    this.Recurse(inputString, index + 1, leftCount, rightCount, expression, removedCount);
                    expression.Remove(length, 1);
                }
                else
                {
                    // Recursion where we delete the current character and move forward
                    this.Recurse(inputString, index + 1, leftCount, rightCount, expression, removedCount + 1);
                    expression.Append(currentCharacter);

                    // If it's an opening parenthesis, consider it and recurse
                    if (currentCharacter == '(')
                    {
                        this.Recurse(inputString, index + 1, leftCount + 1, rightCount, expression, removedCount);
                    }
                    else if (rightCount < leftCount)
                    {
                        // For a closing parenthesis, only recurse if right < left
                        this.Recurse(inputString, index + 1, leftCount, rightCount + 1, expression, removedCount);
                    }

                    // Undoing the append operation for other recursions.
                    expression.Remove(length, 1);
                }
            }
        }

        public List<string> RemoveInvalidParenthesesBacktrack(string inputString)
        {
            this.Reset();
            this.Recurse(inputString, 0, 0, 0, new StringBuilder(), 0);
            return new List<string>(this.validExpressions);
        }

        private void Recurse(
     string inputString,
     int currentIndex,
     int leftCount,
     int rightCount,
     int leftRem,
     int rightRem,
     StringBuilder expression)
        {

            // If we reached the end of the string, just check if the resulting expression is
            // valid or not and also if we have removed the total number of left and right
            // parentheses that we should have removed.
            if (currentIndex == inputString.Length)
            {
                if (leftRem == 0 && rightRem == 0)
                {
                    this.validExpressions.Add(expression.ToString());
                }
            }
            else
            {
                char currentCharacter = inputString[currentIndex];
                int expressionLength = expression.Length;

                // The discard case. Note that here we have our pruning condition.
                // We don't recurse if the remaining count for that parenthesis is == 0.
                if ((currentCharacter == '(' && leftRem > 0) || (currentCharacter == ')' && rightRem > 0))
                {
                    this.Recurse(
                        inputString,
                        currentIndex + 1,
                        leftCount,
                        rightCount,
                        leftRem - (currentCharacter == '(' ? 1 : 0),
                        rightRem - (currentCharacter == ')' ? 1 : 0),
                        expression);
                }

                expression.Append(currentCharacter);

                // Simply recurse one step further if the current character is not a parenthesis.
                if (currentCharacter != '(' && currentCharacter != ')')
                {
                    this.Recurse(inputString, currentIndex + 1, leftCount, rightCount, leftRem, rightRem, expression);
                }
                else if (currentCharacter == '(')
                {
                    // Consider an opening bracket.
                    this.Recurse(inputString, currentIndex + 1, leftCount + 1, rightCount, leftRem, rightRem, expression);
                }
                else if (rightCount < leftCount)
                {
                    // Consider a closing bracket.
                    this.Recurse(inputString, currentIndex + 1, leftCount, rightCount + 1, leftRem, rightRem, expression);
                }

                // Delete for backtracking.
                expression.Remove(expressionLength - 1, 1);
            }
        }

        public List<string> RemoveInvalidParenthesesLimitedBacktrack(string inputString)
        {
            int left = 0, right = 0;

            // First, we find out the number of misplaced left and right parentheses.
            for (int i = 0; i < inputString.Length; i++)
            {
                // Simply record the left one.
                if (inputString[i] == '(')
                {
                    left++;
                }
                else if (inputString[i] == ')')
                {
                    // If we don't have a matching left, then this is a misplaced right, record it.
                    right = left == 0 ? right + 1 : right;

                    // Decrement count of left parentheses because we have found a right
                    // which CAN be a matching one for a left.
                    left = left > 0 ? left - 1 : left;
                }
            }

            this.Recurse(inputString, 0, 0, 0, left, right, new StringBuilder());
            return new List<string>(this.validExpressions);
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
            int indexOfFirstOccur = FindIndexOfFirstOccurSW(haystack,needle);
            /*
 Approach 2: Rabin-Karp Algorithm (Single Hash)(RKSH)           
Complexity Analysis
•	Time complexity: O(nm).
In the worst case, hashNeedle might match with the hash value of all haystack substrings. Hence, we then have to iterate character by character in each window. There are n−m+1 such windows of length m. Hence, the time complexity is O(nm).
But in the best case, if no hash value of the haystack substring matches with hashNeedle, then we don't have to iterate character by character in each window. In that case, it will be O(n+m). Computing the hash value of haystack and needle will be O(m) and for traversing all windows, we will have O(n−m) time complexity. And during traversal, we are doing constant number of operations, hence, in that case, it will be O(n−m+2m)=O(n+m).
•	Space complexity: O(1). We are not using any extra space.
There are a handful of variables in the code, namely, hashNeedle, hashHay, windowStart, windowEnd, m, n, MAX_WEIGHT, RADIX, MOD. All of them use constant space, and hence, the space complexity is O(1).
            
            */
            indexOfFirstOccur = FindIndexOfFirstOccurRKSH(haystack,needle);
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
            indexOfFirstOccur = FindIndexOfFirstOccurRKDH(haystack,needle);
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
            indexOfFirstOccur = FindIndexOfFirstOccurKMP(haystack,needle);

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


    }
}