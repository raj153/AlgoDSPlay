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


    }
}