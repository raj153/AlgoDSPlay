using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class PalindromeProbs
    {

        /*     647. Palindromic Substrings
https://leetcode.com/problems/palindromic-substrings/description/
 */

        public class PalindromicSubstringsSol
        {
            /*
Approach 1: Check All Substrings - Naive
Complexity Analysis
•	Time Complexity: O(N^3) for input string of length N.
Since we just need to traverse every substring once, the total time taken is sum of the length of all substrings.
•	Space Complexity: O(1). We don't need to allocate any extra space since we are repeatedly iterating on the input string itself.

            */
            public int Naive(string s)
            {
                int ans = 0;

                for (int start = 0; start < s.Length; ++start)
                    for (int end = start; end < s.Length; ++end)
                        ans += IsPalindrome(s, start, end) ? 1 : 0;

                return ans;
            }
            private bool IsPalindrome(string s, int start, int end)
            {
                while (start < end)
                {
                    if (s[start] != s[end])
                        return false;

                    ++start;
                    --end;
                }

                return true;
            }
            /*
            Approach 2: Dynamic Programming
      Complexity Analysis
•	Time Complexity: O(N^2) for input string of length N. The number of dynamic programming states that need to calculated is the same as the number of substrings i.e. (N/2)=N(N−1)/2. Each state can be calculated in constant time using a previously calculated state. Thus overall time take in the order of O(N^2).
•	Space Complexity: O(N^2) for an input string of length N. We need to allocate extra space to hold all (N/2) dynamic programming states.
      
            */
            public int DP(String s)
            {
                int n = s.Length, ans = 0;

                if (n == 0)
                    return 0;

                bool[][] dp = new bool[n][];

                // Base case: single letter substrings
                for (int i = 0; i < n; ++i, ++ans)
                    dp[i][i] = true;

                // Base case: double letter substrings
                for (int i = 0; i < n - 1; ++i)
                {
                    dp[i][i + 1] = (s[i] == s[i + 1]);
                    ans += (dp[i][i + 1] ? 1 : 0);
                }

                // All other cases: substrings of length 3 to n
                for (int len = 3; len <= n; ++len)
                    for (int i = 0, j = i + len - 1; j < n; ++i, ++j)
                    {
                        dp[i][j] = dp[i + 1][j - 1] && (s[i] == s[j]);
                        ans += (dp[i][j] ? 1 : 0);
                    }

                return ans;
            }
            /*
Approach 3: Expand Around Possible Centers
Complexity Analysis
•	Time Complexity: O(N^2) for input string of length N. The total time taken in this approach is dictated by two variables:
o	The number of possible palindromic centers we process.
o	How much time we spend processing each center.
The number of possible palindromic centers is 2N−1: there are N single character centers and N−1 consecutive character pairs as centers.
Each center can potentially expand to the length of the string, so time spent on each center is linear on average. Thus total time spent is N⋅(2N−1)≃N^2.
•	Space Complexity: O(1). We don't need to allocate any extra space since we are repeatedly iterating on the input string itself.


            */
            public int ExpandAroundPossibleCenters(String s)
            {
                int ans = 0;

                for (int i = 0; i < s.Length; ++i)
                {
                    // odd-length palindromes, single character center
                    ans += CountPalindromesAroundCenter(s, i, i);

                    // even-length palindromes, consecutive characters center
                    ans += CountPalindromesAroundCenter(s, i, i + 1);
                }

                return ans;
            }

            private int CountPalindromesAroundCenter(String ss, int lo, int hi)
            {
                int ans = 0;

                while (lo >= 0 && hi < ss.Length)
                {
                    if (ss[lo] != ss[hi])
                        break;      // the first and last characters don't match!

                    // expand around the center
                    lo--;
                    hi++;

                    ans++;
                }

                return ans;
            }

        }


        /*
                         266. Palindrome Permutation
                 https://leetcode.com/problems/palindrome-permutation/description/
                         */
        public class CanPermutePalindromeSol
        {


            public bool CanPermutePalindrome(string s)
            {
                /*
    Approach #1 Brute Force [Accepted]
    Complexity Analysis
    •	Time complexity : O(n). We iterate constant number of times(128) over the string s of length n, i.e. O(128⋅n)=O(n).
    •	Space complexity : O(1). Constant extra space is used

                */
                bool canPermutePalindrome = CanPermutePalindromeNaive(s);

                /*
    Approach #2 Using HashMap/Dict [Accepted] 
    Complexity Analysis
    •	Time complexity : O(n). We traverse over the given string s with n characters once. We also traverse over the map which can grow up to a size of n in case all characters in s are distinct.
    •	Space complexity : O(1). The map can grow up to a maximum number of all distinct elements. However, the number of distinct characters are bounded, so as the space complexity.

                */

                canPermutePalindrome = CanPermutePalindromeDict(s);
                /*

    Approach #3 Using Array [Accepted]
    Complexity Analysis**
    •	Time complexity : O(n). We traverse once over the string s of length n. Then, we traverse over the map of length 128(constant).
    •	Space complexity : O(1). Constant extra space is used for map of size 128.


                */
                canPermutePalindrome = CanPermutePalindromeArray(s);

                /*
    Approach #4 Single Pass [Accepted]:
    Complexity Analysis
    •	Time complexity : O(n). We traverse over the string s of length n once only.
    •	Space complexity : O(1). A map of constant size(128) is used

                */
                canPermutePalindrome = CanPermutePalindromeSinglePass(s);
                /*
           Approach #5 Using Set [Accepted]:
        Complexity Analysis
        •	Time complexity : O(n). We traverse over the string s of length n once only.
        •	Space complexity : O(1). The set can grow up to a maximum number of all distinct elements. However, the number of distinct characters are bounded, so as the space complexity.

                */
                canPermutePalindrome = CanPermutePalindromeSet(s);

                return canPermutePalindrome;

            }

            public bool CanPermutePalindromeNaive(String s)
            {
                int count = 0;
                for (char character = '0'; character < 128 && count <= 1; character++)
                {
                    int ct = 0;
                    for (int j = 0; j < s.Length; j++)
                    {
                        if (s[j] == character)
                            ct++;
                    }
                    count += ct % 2;
                }
                return count <= 1;
            }
            public bool CanPermutePalindromeDict(string inputString)
            {
                Dictionary<char, int> characterCountMap = new Dictionary<char, int>();
                for (int index = 0; index < inputString.Length; index++)
                {
                    char currentCharacter = inputString[index];
                    if (characterCountMap.ContainsKey(currentCharacter))
                    {
                        characterCountMap[currentCharacter]++;
                    }
                    else
                    {
                        characterCountMap[currentCharacter] = 1;
                    }
                }
                int oddCount = 0;
                foreach (char key in characterCountMap.Keys)
                {
                    oddCount += characterCountMap[key] % 2;
                }
                return oddCount <= 1;
            }
            public bool CanPermutePalindromeArray(String s)
            {
                int[] map = new int[128];
                for (int i = 0; i < s.Length; i++)
                {
                    map[s[i]]++;
                }
                int count = 0;
                for (int key = 0; key < map.Count() && count <= 1; key++)
                {
                    count += map[key] % 2;
                }
                return count <= 1;
            }

            public bool CanPermutePalindromeSinglePass(String s)
            {
                int[] map = new int[128];
                int count = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    map[s[i]]++;
                    if (map[s[i]] % 2 == 0)
                        count--;
                    else
                        count++;
                }
                return count <= 1;
            }

            public bool CanPermutePalindromeSet(String s)
            {
                HashSet<char> set = new HashSet<char>();
                for (int i = 0; i < s.Length; i++)
                {
                    if (!set.Add(s[i]))
                        set.Remove(s[i]);
                }
                return set.Count() <= 1;
            }
        }


        /*
     267. Palindrome Permutation II
https://leetcode.com/problems/palindrome-permutation-ii/description/

     */
        public class GeneratePalindromesSol
        {
            public IList<string> GeneratePalindromes(string s)
            {
                /*
     Approach #1 Brute Force [Time Limit Exceeded]           
    Complexity Analysis
    •	Time complexity : O((n+1)!). A total of n! permutations are possible. For every permutation generated, we need to check if it is a palindrome, each of which requires O(n) time.
    •	Space complexity : O(n). The depth of the recursion tree can go upto n.

                */
                var palindroms = GeneratePalindromesNaive(s);

                /*
    Approach #2 Backtracking 
     Complexity Analysis
    •	Time complexity : O((n/2)+1)!). At most 2n! permutations need to be generated in the worst case. Further, for each permutation generated, string.reverse() function will take n/4 time.
    •	Space complexity : O(n). The depth of recursion tree can go upto n/2 in the worst case
                */
                palindroms = GeneratePalindromesBacktrack(s);

                return palindroms;
            }

            HashSet<string> set = new HashSet<string>();

            public List<String> GeneratePalindromesNaive(String s)
            {
                Permute(s.ToCharArray(), 0);
                return new List<string>(set);
            }

            public bool IsPalindrome(char[] s)
            {
                for (int i = 0; i < s.Length; i++)
                {
                    if (s[i] != s[s.Length - 1 - i]) return false;
                }
                return true;
            }

            public void Swap(char[] s, int i, int j)
            {
                char temp = s[i];
                s[i] = s[j];
                s[j] = temp;
            }

            void Permute(char[] s, int l)
            {
                if (l == s.Length)
                {
                    if (IsPalindrome(s)) set.Add(new string(s));
                }
                else
                {
                    for (int i = l; i < s.Length; i++)
                    {
                        Swap(s, l, i);
                        Permute(s, l + 1);
                        Swap(s, l, i);
                    }
                }
            }

            public List<String> GeneratePalindromesBacktrack(String s)
            {
                int[] map = new int[128];
                char[] st = new char[s.Length / 2];
                if (!CanPermutePalindrome(s, map)) return new List<string>();
                char ch = '0';
                int k = 0;
                for (int i = 0; i < map.Length; i++)
                {
                    if (map[i] % 2 == 1) ch = (char)i;
                    for (int j = 0; j < map[i] / 2; j++)
                    {
                        st[k++] = (char)i;
                    }
                }
                Permute(st, 0, ch);
                return new List<string>(set);
            }
            public bool CanPermutePalindrome(String s, int[] map)
            {
                int count = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    map[s[i]]++;
                    if (map[s[i]] % 2 == 0) count--;
                    else count++;
                }
                return count <= 1;

            }


            void Permute(char[] s, int l, char ch)
            {
                if (l == s.Length)
                {
                    set.Add(
                        new String(s) +
                        (ch == 0 ? "" : ch) +
                        new string(s)
                    );
                }
                else
                {
                    for (int i = l; i < s.Length; i++)
                    {
                        if (s[l] != s[i] || l == i)
                        {
                            Swap(s, l, i);
                            Permute(s, l + 1, ch);
                            Swap(s, l, i);
                        }
                    }
                }
            }
        }

        /*
        //https://www.algoexpert.io/questions/linked-list-
        public class LinkedListPalindromeSol
        {


            //1.
            // O(n) time | O(n) space - where n is the number of nodes in the Linked List
            public bool LinkedListPalindromeNaive(LinkedList head)
            {
                LinkedListInfo isPalindromeResults = isPalindrome(head, head);
                return isPalindromeResults.outerNodesAreEqual;
            }

            public LinkedListInfo isPalindrome(
              LinkedList leftNode, LinkedList rightNode
            )
            {
                if (rightNode == null) return new LinkedListInfo(true, leftNode);

                LinkedListInfo recursiveCallResults =
                  isPalindrome(leftNode, rightNode.Next);
                LinkedList leftNodeToCompare = recursiveCallResults.leftNodeToCompare;
                bool outerNodesAreEqual = recursiveCallResults.outerNodesAreEqual;

                bool recursiveIsEqual =
                  outerNodesAreEqual && (leftNodeToCompare.Value == rightNode.Value);
                LinkedList nextLeftNodeToCompare = leftNodeToCompare.Next;

                return new LinkedListInfo(recursiveIsEqual, nextLeftNodeToCompare);
            }

            //2.
            // O(n) time | O(1) space - where n is the number of nodes in the Linked List
            public bool LinkedListPalindrome(LinkedList head)
            {
                LinkedList slowNode = head;
                LinkedList fastNode = head;

                while (fastNode != null && fastNode.Next != null)
                {
                    slowNode = slowNode.Next;
                    fastNode = fastNode.Next.Next;
                }

                LinkedList reversedSecondHalfNode = ReverseLinkedList(slowNode);
                LinkedList firstHalfNode = head;

                while (reversedSecondHalfNode != null)
                {
                    if (reversedSecondHalfNode.Value != firstHalfNode.Value) return false;
                    reversedSecondHalfNode = reversedSecondHalfNode.Next;
                    firstHalfNode = firstHalfNode.Next;
                }

                return true;
            }

            public static LinkedList ReverseLinkedList(LinkedList head)
            {
                LinkedList previousNode = null;
                LinkedList currentNode = head;
                while (currentNode != null)
                {
                    LinkedList nextNode = currentNode.Next;
                    currentNode.Next = previousNode;
                    previousNode = currentNode;
                    currentNode = nextNode;
                }
                return previousNode;
            }

            public class LinkedListInfo
            {
                public bool outerNodesAreEqual;
                public LinkedList leftNodeToCompare;
                public LinkedListInfo(
                  bool outerNodesAreEqual, LinkedList leftNodeToCompare
                )
                {
                    this.outerNodesAreEqual = outerNodesAreEqual;
                    this.leftNodeToCompare = leftNodeToCompare;
                }
            }
        }

        /*
    125. Valid Palindrome	
    https://leetcode.com/problems/valid-palindrome/description/

    https://www.algoexpert.io/questions/palindrome-
    */
        public class IsPalindromeSol
        {

            /*
            Approach 1: Compare with Reverse
            
            Complexity Analysis
•	Time complexity : O(n), in length n of the string.
We need to iterate thrice through the string:
1.	When we filter out non-alphanumeric characters, and convert the remaining characters to lower-case.
2.	When we reverse the string.
3.	When we compare the original and the reversed strings.
Each iteration runs linear in time (since each character operation completes in constant time). Thus, the effective run-time complexity is linear.
•	Space complexity : O(n), in length n of the string. We need O(n) additional space to stored the filtered string and the reversed string.

            */

            public static bool CompareWithReverse(string str)
            {
                string filteredString = String.Empty;
                foreach (char ch in str)
                {
                    if (Char.IsLetterOrDigit(ch))
                    {
                        filteredString += Char.ToLower(ch);
                    }
                }

                char[] reversedChars = filteredString.ToCharArray();
                Array.Reverse(reversedChars);
                string reversedString = new string(reversedChars);
                return filteredString == reversedString;
            }

            // O(n) time | O(n) space - Using StringBuilder instead String
            public static bool IsPalindrome1(string str)
            {
                StringBuilder reversedstring = new StringBuilder();
                for (int i = str.Length - 1; i >= 0; i--)
                {
                    reversedstring.Append(str[i]);
                }
                return str.Equals(reversedstring.ToString());
            }

            /*
            Approach 2: Two Pointers
            Complexity Analysis
    •	Time complexity : O(n), in length n of the string. We traverse over each character at-most once, until the two pointers meet in the middle, or when we break and return early.
    •	Space complexity : O(1). No extra space required, at all.


            */
            public bool TwoPointers(string s)
            {
                int i = 0;
                int j = s.Length - 1;
                while (i < j)
                {
                    while (i < j && !Char.IsLetterOrDigit(s[i]))
                    {
                        i++;
                    }

                    while (i < j && !Char.IsLetterOrDigit(s[j]))
                    {
                        j--;
                    }

                    if (char.ToLower(s[i]) != char.ToLower(s[j]))
                        return false;
                    i++;
                    j--;
                }

                return true;
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


        }

        /* 
        5. Longest Palindromic Substring
        https://leetcode.com/problems/longest-palindromic-substring/description

        //https://www.algoexpert.io/questions/longest-palindromic-substring
         */
        public class LongestPalindromicSubstringSol
        {

            /*

            Approach 1: Check All Substrings - Naive
            Complexity Analysis
            Given n as the length of s,
            •	Time complexity: O(n^3)
            The two nested for loops iterate O(n^2) times. We check one substring of length n, two substrings of length n - 1, three substrings of length n - 2, and so on.
            There are n substrings of length 1, but we don't check them all since any substring of length 1 is a palindrome, and we will return immediately.
            Therefore, the number of substrings that we check in the worst case is 1 + 2 + 3 + ... + n - 1. This is the partial sum of this series for n - 1, which is equal to (n⋅(n−1))/2=O(n^2).
            In each iteration of the while loop, we perform a palindrome check. The cost of this check is linear with n as well, giving us a time complexity of O(n^3).
            Note that this time complexity is in the worst case and has a significant constant divisor that is dropped by big O. Due to the optimizations of checking the longer length substrings first and exiting the palindrome check early if we determine that a substring cannot be a palindrome, the practical runtime of this algorithm is not too bad.
            •	Space complexity: O(1)
            We don't count the answer as part of the space complexity. Thus, all we use are a few integer variables.

            */
            public static string Naive(string str)
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

            private static bool IsPalindrome(string str)
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
            /*
            Approach 2: Dynamic Programming
            Complexity Analysis
            Given n as the length of s,
            •	Time complexity: O(n^2)
            We declare an n * n table dp, which takes O(n^2) time. We then populate O(n^2) states i, j - each state takes O(1) time to compute.
            •	Space complexity: O(n^2)
            The table dp takes O(n^2) space.
            */
            public string DP(string s)
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
            /*
            Approach 3: Expand From Centers
Complexity Analysis
Given n as the length of s,
•	Time complexity: O(n^2)
There are 2n−1=O(n) centers. For each center, we call expand, which costs up to O(n).
Although the time complexity is the same as in the DP approach, the average/practical runtime of the algorithm is much faster. This is because most centers will not produce long palindromes, so most of the O(n) calls to expand will cost far less than n iterations.
The worst case scenario is when every character in the string is the same.
•	Space complexity: O(1)
We don't use any extra space other than a few integers. This is a big improvement on the DP approach.

            */
            public static string ExpandFromCenter(string str)
            {
                int[] currentLongest = { 0, 1 };
                for (int i = 1; i < str.Length; i++)
                {
                    int[] odd = GetLongestPalindromeFrom(str, i - 1, i + 1);
                    int[] even = GetLongestPalindromeFrom(str, i - 1, i);
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

            public static int[] GetLongestPalindromeFrom(
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

            /*
            
Approach 4: Manacher's Algorithm
Complexity Analysis
Given n as the length of s,
•	Time complexity: O(n)
From Wikipedia (the implementation they describe is slightly different from the above code, but it's the same algorithm):
The algorithm runs in linear time. This can be seen by noting that Center strictly increases after each outer loop and the sum Center + Radius is non-decreasing. Moreover, the number of operations in the first inner loop is linear in the increase of the sum Center + Radius while the number of operations in the second inner loop is linear in the increase of Center. Since Center ≤ 2n+1 and Radius ≤ n, the total number of operations in the first and second inner loops is O(n) and the total number of operations in the outer loop, other than those in the inner loops, is also O(n). The overall running time is therefore O(n).
•	Space complexity: O(n)
We use sPrime and palindromeRadii, both of length O(n).

            */
            public string ManacherAlgo(string s)
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
        }

        /* 214. Shortest Palindrome
            https://leetcode.com/problems/shortest-palindrome/description/
             */
        public class ShortestPalindromeSol
        {
            /*
        Approach 1: Brute Force
        Complexity Analysis
        Let n be the length of the input string s.
        •	Time complexity: O(n^2)
        The reversal of the string s involves traversing the string once, which has a time complexity of O(n).
        In the loop, for each iteration, we check if the substring of length n−i of s matches the substring of length n−i of the reversed string. Each check involves string operations that are linear in the length of the substring being compared. Thus, for each iteration i, the comparison is O(n−i). Since i ranges from 0 to n−1, the total time complexity of the palindrome check part can be expressed as the sum of comparisons of decreasing lengths. This sum is roughly O(n^2).
        Combining these operations, the overall time complexity is O(n^2).
        •	Space complexity: O(n)
        Creating the reversed string involves additional space proportional to the length of the input string, i.e., O(n).
        The substring operations in the for loop do not require additional space proportional to the length of the string but do create new string objects temporarily, which is still O(n) space for each substring.
        Therefore, the overall space complexity is O(n).

            */
            public string Naive(string inputString)
            {
                int stringLength = inputString.Length;
                string reversedString = new string(inputString.Reverse().ToArray());

                // Iterate through the string to find the longest palindromic prefix
                for (int index = 0; index < stringLength; index++)
                {
                    if (inputString.Substring(0, stringLength - index).Equals(reversedString.Substring(index)))
                    {
                        return new string(reversedString.Substring(0, index).ToCharArray()) + inputString;
                    }
                }
                return string.Empty;
            }

            /*
            Approach 2: Two Pointer
            Complexity Analysis
            Let n be the length of the input string.
            •	Time Complexity: O(n^2)
            Each iteration of the shortestPalindrome function operates on a substring of size n. In the worst-case scenario, where the string is not a palindrome and we must continually reduce its size, the function might need to be called up to n/2 times.
            The time complexity T(n) represents the total time taken by the algorithm. At each step, the algorithm processes a substring and then works with a smaller substring by removing two characters. This can be expressed as T(n)=T(n−2)+O(n), where O(n) is the time taken to process the substring of size n.
            Summing up all the steps, we get:
            T(n)=O(n)+O(n−2)+O(n−4)+…+O(1)
            This sum of terms approximates to O(n^2) because it is an arithmetic series where the number of terms grows linearly with n.
            •	Space Complexity: O(n)
            The space complexity is linear, O(n), due to the space needed to store the reversed suffix and other temporary variables.

            */
            public string TwoPointer(string inputString)
            {
                int inputLength = inputString.Length;
                if (inputLength == 0)
                {
                    return inputString;
                }

                // Find the longest palindromic prefix
                int leftIndex = 0;
                for (int rightIndex = inputLength - 1; rightIndex >= 0; rightIndex--)
                {
                    if (inputString[rightIndex] == inputString[leftIndex])
                    {
                        leftIndex++;
                    }
                }

                // If the whole string is a palindrome, return the original string
                if (leftIndex == inputLength)
                {
                    return inputString;
                }

                // Extract the suffix that is not part of the palindromic prefix
                string nonPalindromeSuffix = inputString.Substring(leftIndex);
                StringBuilder reversedSuffix = new StringBuilder(nonPalindromeSuffix.Reverse().ToString()); //TODO: double check this

                // Form the shortest palindrome by prepending the reversed suffix
                return reversedSuffix.ToString() +
                       TwoPointer(inputString.Substring(0, leftIndex)) +
                       nonPalindromeSuffix;
            }
            /*
            Approach 3: KMP (Knuth-Morris-Pratt) Algorithm
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n)
Creating the reversed string requires a pass through the original string, which takes O(n) time.
Concatenating s, #, and reversedString takes O(n) time, as concatenating strings of length n is linear in the length of the strings.
Constructing the prefix table involves iterating over the combined string of length 2n+1. The buildPrefixTable method runs in O(m) time, where m is the length of the combined string. In this case, m=2n+1, so the time complexity is O(n).
Extracting the suffix and reversing it are both O(n) operations.
Combining these, the overall time complexity is O(n).
•	Space complexity: O(n)
The reversedString and combinedString each use O(n) space.
The prefixTable array has a size of 2n+1, which is O(n). Other variables used (such as length and indices) use O(1) space.
Combining these, the overall space complexity is O(n).	

            */
            public string KMPAlgo(string inputString)
            {
                string reversedString = new string(inputString.Reverse().ToArray());
                string combinedString = inputString + "#" + reversedString;
                int[] prefixTable = BuildPrefixTable(combinedString);

                int palindromeLength = prefixTable[combinedString.Length - 1];
                StringBuilder suffix = new StringBuilder(
                    inputString.Substring(palindromeLength).Reverse().ToString()
                );
                return suffix.Append(inputString).ToString();
            }

            private int[] BuildPrefixTable(string inputString)
            {
                int[] prefixTable = new int[inputString.Length];
                int length = 0;
                for (int i = 1; i < inputString.Length; i++)
                {
                    while (length > 0 && inputString[i] != inputString[length])
                    {
                        length = prefixTable[length - 1];
                    }
                    if (inputString[i] == inputString[length])
                    {
                        length++;
                    }
                    prefixTable[i] = length;
                }
                return prefixTable;
            }

            /*
            Approach 4: Rolling Hash Based Algorithm
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n)
The algorithm performs a single pass over the input string to compute rolling hashes and determine the longest palindromic prefix, resulting in O(n) time complexity. This pass involves constant-time operations for each character, including hash updates and power calculations. After this, we perform an additional pass to reverse the suffix, which is also O(n). The total time complexity remains O(n).
•	Space complexity: O(n)
The space complexity is determined by the space used for the reversed suffix and the additional string manipulations. The space required for the forward and reverse hash values, power value, and palindrome end index is constant and does not scale with input size. However, storing the reversed suffix and the final result string both require O(n) space. Thus, the space complexity is O(n).

            */
            public String RollingHashBasedAlgo(String s)
            {
                long hashBase = 29;
                long modValue = (long)1e9 + 7;
                long forwardHash = 0, reverseHash = 0, powerValue = 1;
                int palindromeEndIndex = -1;

                // Calculate rolling hashes and find the longest palindromic prefix
                for (int i = 0; i < s.Length; i++)
                {
                    char currentChar = s[i];

                    // Update forward hash
                    forwardHash = (forwardHash * hashBase + (currentChar - 'a' + 1)) %
                    modValue;

                    // Update reverse hash
                    reverseHash = (reverseHash + (currentChar - 'a' + 1) * powerValue) %
                    modValue;
                    powerValue = (powerValue * hashBase) % modValue;

                    // If forward and reverse hashes match, update palindrome end index
                    if (forwardHash == reverseHash)
                    {
                        palindromeEndIndex = i;
                    }
                }

                // Find the remaining suffix after the longest palindromic prefix
                String suffix = s.Substring(palindromeEndIndex + 1);
                // Reverse the remaining suffix
                StringBuilder reversedSuffix = new StringBuilder(suffix.Reverse().ToString());

                // Prepend the reversed suffix to the original string and return the result
                return reversedSuffix.Append(s).ToString();
            }
            /*
            Approach 5: Manacher's Algorithm
            Complexity Analysis
            Let n be the length of the input string.
            •	Time complexity: O(n)
            The preprocessString method adds boundaries and separators to the input string. This takes linear time, O(n), where n is the length of the input string.
            The core algorithm iterates through the characters of the modified string once. The expansion step and the updates of the center and right boundary each take constant time in the average case for each character. Thus, this step has a time complexity of O(m), where m is the length of the modified string.
            Since the length of the modified string is 2n+1(for separators)+2(for boundaries)=2n+3 , the time complexity of Manacher's algorithm is O(n).
            Constructing the result involves reversing the suffix of the original string and concatenating it with the original string, both of which take linear time, O(n).
            Combining these steps, the total time complexity is O(n).
            •	Space complexity: O(n)
            The space used to store the modified string is proportional to its length, which is 2n+3. Therefore, the space complexity for storing this string is O(n).
            The palindromeRadiusArray is used to store the radius of palindromes for each character in the modified string, which is O(m). Since m is 2n+3, the space complexity for this array is O(n).
            The additional space used for temporary variables, and other operations is constant, O(1).
            Combining these factors, the total space complexity is O(n).	

            */
            public string ManachersAlgo(string inputString)
            {
                // Return early if the string is null or empty
                if (inputString == null || inputString.Length == 0)
                {
                    return inputString;
                }

                // Preprocess the string to handle palindromes uniformly
                string modifiedString = PreprocessString(inputString);
                int[] palindromeRadiusArray = new int[modifiedString.Length];
                int center = 0, rightBoundary = 0;
                int maxPalindromeLength = 0;

                // Iterate through each character in the modified string
                for (int i = 1; i < modifiedString.Length - 1; i++)
                {
                    int mirrorIndex = 2 * center - i;

                    // Use previously computed values to avoid redundant calculations
                    if (rightBoundary > i)
                    {
                        palindromeRadiusArray[i] = Math.Min(
                            rightBoundary - i,
                            palindromeRadiusArray[mirrorIndex]
                        );
                    }

                    // Expand around the current center while characters match
                    while (
                        modifiedString[i + 1 + palindromeRadiusArray[i]] ==
                        modifiedString[i - 1 - palindromeRadiusArray[i]]
                    )
                    {
                        palindromeRadiusArray[i]++;
                    }

                    // Update the center and right boundary if the palindrome extends beyond the current boundary
                    if (i + palindromeRadiusArray[i] > rightBoundary)
                    {
                        center = i;
                        rightBoundary = i + palindromeRadiusArray[i];
                    }

                    // Update the maximum length of palindrome starting at the beginning
                    if (i - palindromeRadiusArray[i] == 1)
                    {
                        maxPalindromeLength = Math.Max(
                            maxPalindromeLength,
                            palindromeRadiusArray[i]
                        );
                    }
                }

                // Construct the shortest palindrome by reversing the suffix and prepending it to the original string
                StringBuilder suffix = new StringBuilder(
                    inputString.Substring(maxPalindromeLength).Reverse().ToString()
                );
                return suffix.Append(inputString).ToString();
            }

            private string PreprocessString(string inputString)
            {
                // Add boundaries and separators to handle palindromes uniformly
                StringBuilder stringBuilder = new StringBuilder("^");
                foreach (char character in inputString)
                {
                    stringBuilder.Append("#").Append(character);
                }
                return stringBuilder.Append("#$").ToString();
            }

        }

        //https://www.algoexpert.io/questions/palindrome-partitioning-min-cuts
        public class PalindromePartitioningMinCutsSol
        {



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
            private static bool IsPalindrome(string str)
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

        }


        /*
             9. Palindrome Number
             https://leetcode.com/problems/palindrome-number/description/
             */
        public class IsNumberPalindromeSol
        {
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
        }


        /* 336. Palindrome Pairs
        https://leetcode.com/problems/palindrome-pairs/description/
         */
        class PalindromePairsSol
        {
            /* Approach 1: Brute force
            Complexity Analysis
Let n be the number of words, and k be the length of the longest word.
•	Time Complexity : O(n^2⋅k).
There are n^2 pairs of words. Then appending 2 words requires time 2k, as does reversing it and then comparing it for equality. The constants are dropped, leaving k. So in total, we get O(n^2⋅k). We can't do better than this with the brute-force approach.
•	Auxiliary Space Complexity : O(n^2+k).
Auxiliary space is where we do not consider the size of the input.
Let's start by working out the size of the output. In the worst case, there'll be n⋅(n−1) pairs of integers in the output list, as each of the n words could pair with any of the other n−1 words. Each pair will add 2 integers to the input list, giving a total of 2⋅n⋅(n−1)=2⋅n^2−2⋅n. Dropping the constant and insignificant terms, we are left with an output size of O(n^2).
Now, how much space do we use to find all the pairs? Each time around the loop, we are combining 2 words and creating an additional (reversed) copy of the combined words. This is 4⋅k, which gives us O(k). We don't need to multiply this by n^2 because we aren't keeping the combined/ reversed words.
In total, this gives us O(n^2+k). It might initially seem like the k should be dropped, as it's less significant than the n^2. This isn't always the case though. If the words were really long, and the list very short, then it's possible for k to be bigger than n^2.
It's possible to optimize this slightly to O(n^2). By using an in-place algorithm to determine whether or not 2 given words form a palindrome, the k would become a 1 and therefore be dropped. Like I said above though, it'd be wasted effort to do so. Especially given that in practice it's likely that k is smaller than n^2 anyway.
•	Space Complexity : O(n⋅k+n^2).
For this, we also need to take into account the size of the input. There are n words, with a length of up to k each. This gives us O(n⋅k).
Like above, we can't assume anything about whether k>n or k<n. Therefore, we don't know whether O(n^2+k) or O(n⋅k) is bigger.

             */
            public List<List<int>> Naive(String[] words)
            {

                List<List<int>> pairs = new List<List<int>>();

                for (int i = 0; i < words.Length; i++)
                {
                    for (int j = 0; j < words.Length; j++)
                    {
                        if (i == j) continue;
                        String combined = (string)words[i].Concat(words[j]);
                        String reversed = combined.Reverse().ToString();
                        if (combined.Equals(reversed))
                        {
                            pairs.Add(new List<int> { i, j });
                        }
                    }
                }

                return pairs;
            }
            /*
            Approach 2: Hashing
            Complexity Analysis
Let n be the number of words, and k be the length of the longest word.
•	Time Complexity : O(k^2⋅n).
Building the hash table takes O(n⋅k) time. Each word takes O(k) time to insert and there are n words.
Then, for each of the n words we are searching for 3 different cases. First is the word's own reverse. This takes O(k) time. Second is words that are a palindrome followed by the reverse of another word. Third is words that are the reverse of another word followed by a palindrome. These second 2 cases have the same cost, so we'll just focus on the first one. We need to find all the prefixes of the given word, that are palindromes. Finding all palindrome prefixes of a word can be done in O(k^2) time, as there are k possible prefixes, and checking each one takes O(k) time. So, for each word we are doing k^2+k^2+k processing, which in big-oh notation is O(k^2). Because are doing this with n words, we get a final result of O(k^2⋅n).
It's worth noting that the previous approach had a cost of O(n^2⋅k). Therefore, this approach isn't better in every case. It is only better where n>k. In the test cases your solution is tested on, this is indeed the case.
•	Space Complexity : O((k+n) ^2).
Like before, there are several components we need to consider. This time however, the space complexity is the same regardless of whether or not we include the input in the calculations. This is because the algorithm immediately creates a hash table the same size as the input.
In the input, there are n words, with a length of up to k each. This gives us O(n⋅k). We are then building a hash table with n keys of size k. The hash table is the same size as the original input, so it too is O(n⋅k).
For each word, we're making a list of all possible pair words that need to be looked up in the hash table. In the worst case, there'll be k words to look up, with lengths of up to k. This means that at each cycle of the loop, we're using up to k2 memory for the lookup list. This could be optimized down to O(k) by only creating one of the words at a time. In practice though, it's unlikely to make much difference due to the way strings are handled under the hood. So, we'll say that we're using an additional O(k^2) memory.
Determining the size of the output is the same as the other approaches. In the worst case, there'll be n⋅(n−1) pairs of integers in the output list, as each of the n words could pair with any of the other n−1 words. Each pair will add 2 integers to the input list, giving a total of 2⋅n⋅(n−1)=2⋅n^2−2⋅n. Dropping the constant and insignificant terms, we are left with an output size of O(n^2).
Putting this all together, we get 2⋅n⋅k+k2+n2=(k+n) ^2, which is O((k+n) ^2).


            */
            public IList<IList<int>> UsingHashing(string[] words)
            {
                Dictionary<string, int> wordSet = new Dictionary<string, int>();
                for (int i = 0; i < words.Length; i++)
                {
                    wordSet[words[i]] = i;
                }

                List<IList<int>> solution = new List<IList<int>>();

                foreach (string word in wordSet.Keys)
                {
                    int currentWordIndex = wordSet[word];
                    string reversedWord = new string(word.ToCharArray().Reverse().ToArray());

                    if (wordSet.ContainsKey(reversedWord) && wordSet[reversedWord] != currentWordIndex)
                    {
                        solution.Add(new List<int> { currentWordIndex, wordSet[reversedWord] });
                    }

                    foreach (string suffix in AllValidSuffixes(word))
                    {
                        string reversedSuffix = new string(suffix.ToCharArray().Reverse().ToArray());
                        if (wordSet.ContainsKey(reversedSuffix))
                        {
                            solution.Add(new List<int> { wordSet[reversedSuffix], currentWordIndex });
                        }
                    }

                    foreach (string prefix in AllValidPrefixes(word))
                    {
                        string reversedPrefix = new string(prefix.ToCharArray().Reverse().ToArray());
                        if (wordSet.ContainsKey(reversedPrefix))
                        {
                            solution.Add(new List<int> { currentWordIndex, wordSet[reversedPrefix] });
                        }
                    }
                }
                return solution;
            }
            private List<string> AllValidPrefixes(string word)
            {
                List<string> validPrefixes = new List<string>();
                for (int i = 0; i < word.Length; i++)
                {
                    if (IsPalindromeBetween(word, i, word.Length - 1))
                    {
                        validPrefixes.Add(word.Substring(0, i));
                    }
                }
                return validPrefixes;
            }

            private List<string> AllValidSuffixes(string word)
            {
                List<string> validSuffixes = new List<string>();
                for (int i = 0; i < word.Length; i++)
                {
                    if (IsPalindromeBetween(word, 0, i))
                    {
                        validSuffixes.Add(word.Substring(i + 1, word.Length - i - 1));
                    }
                }
                return validSuffixes;
            }

            private bool IsPalindromeBetween(string word, int front, int back)
            {
                while (front < back)
                {
                    if (word[front] != word[back]) return false;
                    front++;
                    back--;
                }
                return true;
            }

            /*
            Approach 3: Using a Trie
            Complexity Analysis
Let n be the number of words, and k be the length of the longest word.
•	Time Complexity : O(k^2⋅n).
There were 2 major steps to the algorithm. Firstly, we needed to build the Trie. Secondly, we needed to look up each word in the Trie.
Inserting each word into the Trie takes O(k) time. As well as inserting the word, we also checked at each letter whether or not the remaining part of the word was a palindrome. These checks had a cost of O(k), and with k of them, gave a total cost of O(k^2). With n words to insert, the total cost of building the Trie was therefore O(k^2⋅n).
Checking for each word in the Trie had a similar cost. Each time we encountered a node with a word ending index, we needed to check whether or not the current word we were looking up had a palindrome remaining. In the worst case, we'd have to do this k times at a cost of k for each time. So like before, there is a cost of k^2 for looking up a word, and an overall cost of k^2⋅n for all the checks.
This is the same as for the hash table approach.
•	Space Complexity : O((k+n) ^2).
The Trie is the main space usage. In the worst case, each of the O(n⋅k) letters in the input would be on separate nodes, and each node would have up to n indexes in its list. This gives us a worst case of O(n^2⋅k), which is strictly larger than the input or the output.
Inserting and looking up words only takes k space though, because we're not generating a list of prefixes like we were in approach 2. This is insignificant compared to the size of the Trie itself.
So in total, the size of the Trie has a worst case of O(k⋅n^2). In practice however, it'll use a lot less, as we based this on the worst case. Tries are difficult to analyze in the general case, because their performance is so dependent on the type of data going into them. As n gets really, really, big, the Trie approach will eventually beat the hash table approach on both time and space. For the values of n that we're dealing with in this question though, you'll probably notice that the hash table approach performs better.

            */
            public IList<IList<int>> UsingTrie(string[] words)
            {

                TrieNode trie = new TrieNode();

                // Build the Trie
                for (int wordId = 0; wordId < words.Length; wordId++)
                {
                    string word = words[wordId];
                    string reversedWord = new string(word.ToCharArray().Reverse().ToArray());
                    TrieNode currentTrieLevel = trie;
                    for (int j = 0; j < word.Length; j++)
                    {
                        if (HasPalindromeRemaining(reversedWord, j))
                        {
                            currentTrieLevel.palindromePrefixRemaining.Add(wordId);
                        }
                        char c = reversedWord[j];
                        if (!currentTrieLevel.next.ContainsKey(c))
                        {
                            currentTrieLevel.next[c] = new TrieNode();
                        }
                        currentTrieLevel = currentTrieLevel.next[c];
                    }
                    currentTrieLevel.wordEnding = wordId;
                }

                // Find pairs
                List<IList<int>> pairs = new List<IList<int>>();
                for (int wordId = 0; wordId < words.Length; wordId++)
                {
                    string word = words[wordId];
                    TrieNode currentTrieLevel = trie;
                    for (int j = 0; j < word.Length; j++)
                    {
                        // Check for pairs of case 3.
                        if (currentTrieLevel.wordEnding != -1
                           && HasPalindromeRemaining(word, j))
                        {
                            pairs.Add(new List<int> { wordId, currentTrieLevel.wordEnding });
                        }
                        // Move down to the next trie level.
                        char c = word[j];
                        if (!currentTrieLevel.next.ContainsKey(c))
                        {
                            currentTrieLevel = null;
                            break;
                        }
                        currentTrieLevel = currentTrieLevel.next[c];
                    }
                    if (currentTrieLevel == null) continue;
                    // Check for pairs of case 1. Note the check to prevent non distinct pairs.
                    if (currentTrieLevel.wordEnding != -1 && currentTrieLevel.wordEnding != wordId)
                    {
                        pairs.Add(new List<int> { wordId, currentTrieLevel.wordEnding });
                    }
                    // Check for pairs of case 2.
                    foreach (int other in currentTrieLevel.palindromePrefixRemaining)
                    {
                        pairs.Add(new List<int> { wordId, other });
                    }
                }

                return pairs;
            }
            // Is the given string a palindrome after index i?
            // Tip: Leave this as a method stub in an interview unless you have time
            // or the interviewer tells you to write it. The Trie itself should be
            // the main focus of your time.
            public bool HasPalindromeRemaining(string s, int i)
            {
                int p1 = i;
                int p2 = s.Length - 1;
                while (p1 < p2)
                {
                    if (s[p1] != s[p2]) return false;
                    p1++; p2--;
                }
                return true;
            }
            class TrieNode
            {
                public int wordEnding = -1; // We'll use -1 to mean there's no word ending here.
                public Dictionary<char, TrieNode> next = new Dictionary<char, TrieNode>();
                public List<int> palindromePrefixRemaining = new List<int>();
            }



        }


        /* 2131. Longest Palindrome by Concatenating Two Letter Words
        https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/description/

         */
        class LongestPalindromeByConcatenatingTwoLetterWordsSol
        {
            /*
            Approach 1: A Hash Map Approach
Complexity Analysis
Let N be the number of words in the input array and ∣Σ∣
be the size of the English alphabet (∣Σ∣=26).
•	Time complexity: O(N+min(N,∣Σ∣^2)).
We count the words in O(N) time (assuming one
operation with a hash map takes O(1) time). Calculating the
answer after that takes O(min(N,∣Σ∣^2)) time as we
iterate all hash map elements, and the size of the hash map is
O(min(N,∣Σ∣^2)).
•	Space complexity: O(min(N,∣Σ∣^2)).
There can be up to ∣Σ∣^2 distinct words of two letters
(∣Σ∣ options for the first letter and ∣Σ∣ options
for the second one). Also, the total number of words is N.

            */
            public int UsingHashMap(String[] words)
            {
                Dictionary<String, int> count = new Dictionary<string, int>();
                foreach (String word in words)
                {
                    if (!count.ContainsKey(word))
                        count[word] = 0;
                    count[word] += 1;

                }
                int answer = 0;
                bool central = false;
                foreach (KeyValuePair<string, int> entry in count)
                {
                    String word = entry.Key;
                    int countOfTheWord = entry.Value;
                    // if the word is a palindrome
                    if (word[0] == word[1])
                    {
                        if (countOfTheWord % 2 == 0)
                        {
                            answer += countOfTheWord;
                        }
                        else
                        {
                            answer += countOfTheWord - 1;
                            central = true;
                        }
                        // consider a pair of non-palindrome words such that one is the reverse of another
                    }
                    else if (word[0] < word[1])
                    {
                        String reversedWord = "" + word[1] + word[0];
                        if (count.ContainsKey(reversedWord))
                        {
                            answer += 2 * Math.Min(countOfTheWord, count[reversedWord]);
                        }
                    }
                }
                if (central)
                {
                    answer++;
                }
                return 2 * answer;
            }
            /*
            
Approach 2: A Two-Dimensional Array Approach
Complexity Analysis
Let N be the number of words in the input array.
•	Time complexity: O(N+∣Σ∣^2).
We count the words in O(N) time and then calculate the answer in O(∣Σ∣^2) time.
•	Space complexity: O(∣Σ∣^2).
We are using an auxilary two-dimensional array count of size ∣Σ∣2^.

            */
            public int Using2DArray(String[] words)
            {
                const int alphabetSize = 26;
                int[][] count = new int[alphabetSize][];
                foreach (String word in words)
                {
                    count[word[0] - 'a'][word[1] - 'a']++;
                }
                int answer = 0;
                bool central = false;
                for (int i = 0; i < alphabetSize; i++)
                {
                    if (count[i][i] % 2 == 0)
                    {
                        answer += count[i][i];
                    }
                    else
                    {
                        answer += count[i][i] - 1;
                        central = true;
                    }
                    for (int j = i + 1; j < alphabetSize; j++)
                    {
                        answer += 2 * Math.Min(count[i][j], count[j][i]);
                    }
                }
                if (central)
                {
                    answer++;
                }
                return 2 * answer;
            }
        };


        /* 409. Longest Palindrome
        https://leetcode.com/problems/longest-palindrome/description/
         */
        class LongestPalindromeSol
        {

            /*

            Approach 1: Greedy Way (Hash Map)
            Complexity Analysis
            Let n be the length of the given string s.
            •	Time complexity: O(n)
            The algorithm goes through the characters of s twice: once to count their frequencies and once to construct the palindrome. Since hash table operations like inserting and updating take constant time (O(1)), the time complexity of the algorithm is O(2⋅n), which simplifies to O(n).
            •	Space complexity: O(1)
            The algorithm uses a hash MAP to store the frequency of characters. Given that there can be at most 52 unique characters in s, the space complexity is O(52), which can be simplified to O(1) space.

            */
            public int GreedyWayHashMap(String s)
            {
                // Map to store frequency of occurrence of each character
                Dictionary<char, int> frequencyMap = new Dictionary<char, int>();
                // Count frequencies
                foreach (char c in s)
                {
                    frequencyMap[c] = frequencyMap.GetValueOrDefault(c, 0) + 1;
                }

                int res = 0;
                bool hasOddFrequency = false;
                foreach (int freq in frequencyMap.Values)
                {
                    // Check is the frequency is even
                    if ((freq % 2) == 0)
                    {
                        res += freq;
                    }
                    else
                    {
                        // If the frequency is odd, one occurrence of the
                        // character will remain without a match
                        res += freq - 1;
                        hasOddFrequency = true;
                    }
                }
                // If hasOddFrequency is true, we have at least one unmatched
                // character to make the center of an odd length palindrome.
                if (hasOddFrequency) return res + 1;

                return res;
            }
            /*
            
Approach 2: Greedy Way (Optimized)
Complexity Analysis
Let n be the length of the given string s.
•	Time complexity: O(n)
The algorithm loops over the entire string s only once. Since hash table operations like inserting and updating take constant time (O(1)), the time complexity of the algorithm is O(2⋅n), which simplifies to O(n).
•	Space complexity: O(1)
The only data structure used in our algorithm is a hash table, which stores the frequencies of at most 52 unique characters. Thus, the space complexity of the algorithm is O(52), which can be simplified to O(1).

            */
            public int GreedyWayHashMapOptimal(string s)
            {
                int oddFreqCharsCount = 0;
                Dictionary<char, int> frequencyMap = new Dictionary<char, int>();

                // Loop over the string
                foreach (char c in s.ToCharArray())
                {
                    // Update count of current character
                    if (frequencyMap.ContainsKey(c))
                    {
                        frequencyMap[c]++;
                    }
                    else
                    {
                        frequencyMap[c] = 1;
                    }

                    // If the current freq of the character is odd,
                    // increment oddCount
                    if (frequencyMap[c] % 2 == 1)
                    {
                        oddFreqCharsCount++;
                    }
                    else
                    {
                        oddFreqCharsCount--;
                    }
                }

                // If there are characters with odd frequencies, we are
                // guaranteed to have at least one letter left unmatched,
                // which can make the center of an odd length palindrome.
                if (oddFreqCharsCount > 0)
                {
                    return s.Length - oddFreqCharsCount + 1;
                }
                else
                {
                    return s.Length;
                }
            }

            /*

Approach 3: Greedy Way (Hash Set)
Complexity Analysis
Let n be the length of the given string s.
•	Time complexity: O(n)
The algorithm loops over the entire string only once, which takes O(n) time. All insert, query and delete operations on the set takes constant time, so the time complexity of the algorithm remains O(n).
•	Space complexity: O(1)
The maximum number of unique characters in the string is 52 (considering both uppercase and lowercase English letters). Since 52 is a constant number, the space complexity of the set is O(52), which simplifies to O(1).

            */
            public int GreedyWayHashSet(String s)
            {
                HashSet<char> characterSet = new HashSet<char>();
                int res = 0;

                // Loop over characters in the string
                foreach (char c in s)
                {
                    // If set contains the character, match found
                    if (characterSet.Contains(c))
                    {
                        characterSet.Remove(c);
                        // add the two occurrences to our palindrome
                        res += 2;
                    }
                    else
                    {
                        // add the character to the set
                        characterSet.Add(c);
                    }
                }

                // if any character remains, we have at least one unmatched
                // character to make the center of an odd length palindrome.
                if (characterSet.Count > 0) res++;

                return res;
            }

        }

        /* 2384. Largest Palindromic Number
        https://leetcode.com/problems/largest-palindromic-number/description/
         */
        string LargestPalindromicNumber(string num)
        {
            int[] freq = new int[10];
            //Count all frequencies
            foreach (char v in num)
            {
                freq[v - '0']++;
            }
            StringBuilder even = new StringBuilder();
            int max = -1;
            for (int i = 9; i >= 0; i--)
            {
                //Tax Max of all odd occurences
                if (freq[i] % 2 == 1)
                {
                    max = Math.Max(max, i);
                }
                //Talk half of the occurences of number
                for (int j = 0; j < freq[i] / 2; j++)
                {
                    even.Append(i.ToString());
                }
            }
            //For eg : 444947137,  we would have 744
            //Check for max, if all even occurences are there in number, max will be -1
            string ans = even.ToString() + (max == -1 ? "" : max.ToString()) + new string(even.ToString().Reverse().ToArray());
            //Edge case
            if (ans[0] == '0')
            {
                //If the whole string is 0, then return 0
                if (max == -1) return "0";
                //For case like 00090, max will be 9
                return max.ToString();
            }
            return ans;
        }


        /* 516. Longest Palindromic Subsequence
        https://leetcode.com/problems/longest-palindromic-subsequence/description/
         */
        class LongestPalindromeSubseqSol
        {
            /*
            
Approach 1: Recursive Dynamic Programming
Complexity Analysis
Here, n is the length of s.
•	Time complexity: O(n^2)
o	Initializing the memo array takes O(n^2) time.
o	Since there are O(n^2) states that we need to iterate over, the recursive function is called O(n^2) times.
•	Space complexity: O(n^2)
o	The memo array consumes O(n^2) space.
o	The recursion stack used in the solution can grow to a maximum size of O(n). When we try to form the recursion tree, we see that there are maximum of two branches that can be formed at each level (when s[i]!= s[j]). The recursion stack would only have one call out of the two branches. The height of such a tree will be O(n) because at each level we are decrementing the length of the string under consideration by '1'. As a result, the recursion tree that will be formed will have O(n) height. Hence, the recursion stack will have a maximum of O(n) elements.

            */
            public int DPRec(String s)
            {
                int n = s.Length;
                int[][] memo = new int[n][];
                return lps(s, 0, n - 1, memo);
            }

            private int lps(String s, int i, int j, int[][] memo)
            {
                if (memo[i][j] != 0)
                {
                    return memo[i][j];
                }
                if (i > j)
                {
                    return 0;
                }
                if (i == j)
                {
                    return 1;
                }

                if (s[i] == s[j])
                {
                    memo[i][j] = lps(s, i + 1, j - 1, memo) + 2;
                }
                else
                {
                    memo[i][j] = Math.Max(lps(s, i + 1, j, memo), lps(s, i, j - 1, memo));
                }
                return memo[i][j];
            }

            /*            
Approach 2: Iterative Dynamic Programming
Complexity Analysis
Here, n is the length of s.
•	Time complexity: O(n^2)
o	Initializing the dp array takes O(n^2) time.
o	We fill the dp array which takes O(n^2) time.
•	Space complexity: O(n^2)
o	The dp array consumes O(n^2) space.
            */
            public int DPIterative(String s)
            {
                int[][] dp = new int[s.Length][];

                for (int i = s.Length - 1; i >= 0; i--)
                {
                    dp[i][i] = 1;
                    for (int j = i + 1; j < s.Length; j++)
                    {
                        if (s[i] == s[j])
                        {
                            dp[i][j] = dp[i + 1][j - 1] + 2;
                        }
                        else
                        {
                            dp[i][j] = Math.Max(dp[i + 1][j], dp[i][j - 1]);
                        }
                    }
                }

                return dp[0][s.Length - 1];
            }
            /*
            Approach 3: Dynamic Programming with Space Optimization
Complexity Analysis
Here, n is the length of s.
•	Time complexity: O(n^2)
o	Initializing the dp and dpPrev arrays take O(n) time.
o	To get the answer, we use two loops that take O(n^2) time.
•	Space complexity: O(n)
o	The dp and dpPrev arrays take O(n) space each.

            */
            public int DPSpaceOptimal(String s)
            {
                int n = s.Length;
                int[] dp = new int[n];
                int[] dpPrev = new int[n];

                for (int i = n - 1; i >= 0; --i)
                {
                    dp[i] = 1;
                    for (int j = i + 1; j < n; ++j)
                    {
                        if (s[i] == s[j])
                        {
                            dp[j] = dpPrev[j - 1] + 2;
                        }
                        else
                        {
                            dp[j] = Math.Max(dpPrev[j], dp[j - 1]);
                        }
                    }
                    dpPrev = (int[])dp.Clone();
                }

                return dp[n - 1];
            }
        }

        /* 730. Count Different Palindromic Subsequences
        https://leetcode.com/problems/count-different-palindromic-subsequences/description/
         */
        class CountDiffPalindromicSubsequencesSol
        {
            public int CountPalindromicSubsequences(String s)
            {
                int len = s.Length;
                int[][] dp = new int[len][];

                char[] chs = s.ToCharArray();
                for (int i = 0; i < len; i++)
                {
                    dp[i][i] = 1;   // Consider the test case "a", "b" "c"...
                }

                for (int distance = 1; distance < len; distance++)
                {
                    for (int i = 0; i < len - distance; i++)
                    {
                        int j = i + distance;
                        if (chs[i] == chs[j])
                        {
                            int low = i + 1;
                            int high = j - 1;

                            /* Variable low and high here are used to get rid of the duplicate*/

                            while (low <= high && chs[low] != chs[j])
                            {
                                low++;
                            }
                            while (low <= high && chs[high] != chs[j])
                            {
                                high--;
                            }
                            if (low > high)
                            {
                                // consider the string from i to j is "a...a" "a...a"... where there is no character 'a' inside the leftmost and rightmost 'a'
                                /* eg:  "aba" while i = 0 and j = 2:  dp[1][1] = 1 records the palindrome{"b"}, 
                                  the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"b"}, 
                                  and additional time as {"aba"}. The reason why 2 counted is that we also count {"a", "aa"}. 
                                  So totally dp[i][j] record the palindrome: {"a", "b", "aa", "aba"}. 
                                  */

                                dp[i][j] = dp[i + 1][j - 1] * 2 + 2;
                            }
                            else if (low == high)
                            {
                                // consider the string from i to j is "a...a...a" where there is only one character 'a' inside the leftmost and rightmost 'a'
                                /* eg:  "aaa" while i = 0 and j = 2: the dp[i + 1][j - 1] records the palindrome {"a"}.  
                                  the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"a"}, 
                                  and additional time as {"aaa"}. the reason why 1 counted is that 
                                  we also count {"aa"} that the first 'a' come from index i and the second come from index j. So totally dp[i][j] records {"a", "aa", "aaa"}
                                 */
                                dp[i][j] = dp[i + 1][j - 1] * 2 + 1;
                            }
                            else
                            {
                                // consider the string from i to j is "a...a...a... a" where there are at least two character 'a' close to leftmost and rightmost 'a'
                                /* eg: "aacaa" while i = 0 and j = 4: the dp[i + 1][j - 1] records the palindrome {"a",  "c", "aa", "aca"}. 
                                   the reason why dp[i + 1][j  - 1] * 2 counted is that we count dp[i + 1][j - 1] one time as {"a",  "c", "aa", "aca"}, 
                                   and additional time as {"aaa",  "aca", "aaaa", "aacaa"}.  Now there is duplicate :  {"aca"}, 
                                   which is removed by deduce dp[low + 1][high - 1]. So totally dp[i][j] record {"a",  "c", "aa", "aca", "aaa", "aaaa", "aacaa"}
                                   */
                                dp[i][j] = dp[i + 1][j - 1] * 2 - dp[low + 1][high - 1];
                            }
                        }
                        else
                        {
                            dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1];  //s.charAt(i) != s.charAt(j)
                        }
                        dp[i][j] = dp[i][j] < 0 ? dp[i][j] + 1000000007 : dp[i][j] % 1000000007;
                    }
                }

                return dp[0][len - 1];
            }
        }
        /* 
        2484. Count Palindromic Subsequences
        https://leetcode.com/problems/count-palindromic-subsequences/description/
         */
        public int CountPalindromicSubsequencesSol(String s)
        {
            /*             Complexity
            •	Time complexity: O(n)
            •	Space complexity: O(n)
             */
            int mod = 1000_000_007, n = s.Length, ans = 0;
            int[] cnts = new int[10];
            int[][][] pre = new int[n][][];
            int[][][] suf = new int[n][][];
            for (int i = 0; i < n; i++)
            {
                int c = s[i] - '0';
                if (i > 0)
                    for (int j = 0; j < 10; j++)
                        for (int k = 0; k < 10; k++)
                        {
                            pre[i][j][k] = pre[i - 1][j][k];
                            if (k == c) pre[i][j][k] += cnts[j];
                        }
                cnts[c]++;
            }
            Array.Fill(cnts, 0);
            for (int i = n - 1; i >= 0; i--)
            {
                int c = s[i] - '0';
                if (i < n - 1)
                    for (int j = 0; j < 10; j++)
                        for (int k = 0; k < 10; k++)
                        {
                            suf[i][j][k] = suf[i + 1][j][k];
                            if (k == c) suf[i][j][k] += cnts[j];
                        }
                cnts[c]++;
            }
            for (int i = 2; i < n - 2; i++)
                for (int j = 0; j < 10; j++)
                    for (int k = 0; k < 10; k++)
                        ans = (int)((ans + 1L * pre[i - 1][j][k] * suf[i + 1][j][k]) % mod);
            return ans;
        }

        /* 1930. Unique Length-3 Palindromic Subsequences
        https://leetcode.com/problems/unique-length-3-palindromic-subsequences/description/
         */
        class CountUniquePalindromicSubsequenceOfLength3Sol
        {
            /*
Approach 1: Count Letters In-Between
Complexity Analysis
Given n as the length of s,
•	Time complexity: O(n)
To create letters, we use O(n) time to iterate over s.
Next, we iterate over each letter in letters. Because s only contains lowercase letters of the English alphabet, there will be no more than 26 iterations.
At each iteration, we iterate over s to find i and j, which costs O(n). Next, we iterate between i and j, which could cost O(n) in the worst-case scenario.
Overall, each iteration costs O(n). This gives us a time complexity of O(26n)=O(n)
•	Space complexity: O(1)
letters and between cannot grow beyond a size of 26, since s only contains letters of the English alphabet.

            */
            public int CountLettersInBetween(String s)
            {
                HashSet<char> letters = new HashSet<char>();
                foreach (var c in s)
                {
                    letters.Add(c);
                }

                int ans = 0;
                foreach (char letter in letters)
                {
                    int i = -1;
                    int j = 0;

                    for (int k = 0; k < s.Length; k++)
                    {
                        if (s[k] == letter)
                        {
                            if (i == -1)
                            {
                                i = k;
                            }

                            j = k;
                        }
                    }

                    HashSet<char> between = new HashSet<char>();
                    for (int k = i + 1; k < j; k++)
                    {
                        between.Add(s[k]);
                    }

                    ans += between.Count();
                }

                return ans;
            }
            /*
            
Approach 2: Pre-Compute First and Last Indices
Complexity Analysis
Given n as the length of s,
•	Time complexity: O(n)
First, we calculate first and last by iterating over s, which costs O(n).
Next, we iterate over 26 alphabet positions. At each iteration, we iterate j over some indices, which in the worst-case scenario would cost O(n). Overall, each of the 26 iterations cost O(n), giving us a time complexity of O(26n)=O(n).
•	Space complexity: O(1)
first, last, and between all use constant space since s only contains letters in the English alphabet.	

            */
            public int PrecomputeFirstandLastIndices(String s)
            {
                int[] first = new int[26];
                int[] last = new int[26];
                Array.Fill(first, -1);

                for (int i = 0; i < s.Length; i++)
                {
                    int curr = s[i] - 'a';
                    if (first[curr] == -1)
                    {
                        first[curr] = i;
                    }

                    last[curr] = i;
                }

                int ans = 0;
                for (int i = 0; i < 26; i++)
                {
                    if (first[i] == -1)
                    {
                        continue;
                    }

                    HashSet<char> between = new HashSet<char>();
                    for (int j = first[i] + 1; j < last[i]; j++)
                    {
                        between.Add(s[j]);
                    }

                    ans += between.Count;
                }

                return ans;
            }
        }


        /* 1682. Longest Palindromic Subsequence II
        https://leetcode.com/problems/longest-palindromic-subsequence-ii/description/
         */
        public class LongestPalindromicSubsequenceIISol
        {
            /*
              Approach1: Naive Recursive - TLE
              */
            public int NaiveRec(String s)
            {
                return LongestPalindromeSubseq(0, s.Length - 1, 26, s);
            }

            private int LongestPalindromeSubseq(int i, int j, int prev, String s)
            {
                if (i >= j) return 0;
                if (s[i] - 'a' == prev) return LongestPalindromeSubseq(i + 1, j, prev, s);
                if (s[j] - 'a' == prev) return LongestPalindromeSubseq(i, j - 1, prev, s);

                if (s[i] == s[j])
                {
                    return LongestPalindromeSubseq(i + 1, j - 1, s[i] - 'a', s) + 2;
                }
                else
                {
                    return Math.Max(
                        LongestPalindromeSubseq(i + 1, j, prev, s),
                        LongestPalindromeSubseq(i, j - 1, prev, s)
                    );
                }
            }
            /*
                           Approach2: DP Memoization 
                           */
            public int DPMemo(string s)
            {
                int[][][] memo = new int[s.Length][][];

                for (int r = 0; r < memo.GetLength(0); r++)
                {
                    for (int c = 0; c < memo.GetLength(1); c++)
                    {
                        Array.Fill(memo[r][c], -1);
                    }
                }

                return LongestPalindromeSubseq(0, s.Length - 1, 26, s, memo);
            }

            private int LongestPalindromeSubseq(int i, int j, int prev, string s, int[][][] memo)
            {
                if (i >= j) return 0;
                if (memo[i][j][prev] != -1) return memo[i][j][prev];

                if (s[i] - 'a' == prev) return memo[i][j][prev] = LongestPalindromeSubseq(i + 1, j, prev, s, memo);
                if (s[j] - 'a' == prev) return memo[i][j][prev] = LongestPalindromeSubseq(i, j - 1, prev, s, memo);

                if (s[i] == s[j])
                {
                    memo[i][j][prev] = LongestPalindromeSubseq(i + 1, j - 1, s[i] - 'a', s, memo) + 2;
                }
                else
                {
                    memo[i][j][prev] = Math.Max(
                        LongestPalindromeSubseq(i + 1, j, prev, s, memo),
                        LongestPalindromeSubseq(i, j - 1, prev, s, memo)
                    );
                }

                return memo[i][j][prev];
            }

            /*
            Approach3: 3D Bottom Up  
            */
            public int BottomUp3DArray(String s)
            {
                int[][][] length = new int[s.Length][][];

                for (int i = s.Length - 1; i >= 0; i--)
                {
                    for (int j = 0; j < s.Length; j++)
                    {
                        for (int prev = 0; prev <= 26; prev++)
                        {
                            if (i >= j) continue;
                            if (s[i] - 'a' == prev)
                            {
                                length[i][j][prev] = length[i + 1][j][prev];
                                continue;
                            }
                            if (s[j] - 'a' == prev)
                            {
                                length[i][j][prev] = length[i][j - 1][prev];
                                continue;
                            }

                            if (s[i] == s[j])
                            {
                                length[i][j][prev] = length[i + 1][j - 1][s[i] - 'a'] + 2;
                            }
                            else
                            {
                                length[i][j][prev] = Math.Max(
                                        length[i + 1][j][prev],
                                        length[i][j - 1][prev]
                                );
                            }
                        }
                    }
                }

                return length[0][s.Length - 1][26];
            }
            /*
            4. 2D Bottom Up 
            */
            public int BottomUp2DArray(String s)
            {
                int[][] length = new int[s.Length][];

                for (int i = s.Length - 1; i >= 0; i--)
                {
                    for (int j = 0; j < s.Length; j++)
                    {
                        for (int prev = 26; prev >= 0; prev--)
                        {
                            if (i >= j) continue;
                            if (s[i] - 'a' == prev)
                            {
                                continue;
                            }
                            if (s[j] - 'a' == prev)
                            {
                                length[j][prev] = length[j - 1][prev];
                                continue;
                            }

                            if (s[i] == s[j])
                            {
                                length[j][prev] = length[j - 1][s[i] - 'a'] + 2;
                            }
                            else
                            {
                                length[j][prev] = Math.Max(
                                    length[j][prev],
                                    length[j - 1][prev]
                                );
                            }
                        }
                    }
                }

                return length[s.Length - 1][26];
            }



        }


        /* 1771. Maximize Palindrome Length From Subsequences
        https://leetcode.com/problems/maximize-palindrome-length-from-subsequences/description/
         */
        class MaximizePalindromeLengthFromSubseqSol
        {
            public int DP(String word1, String word2)
            {
                String m = word1 + word2;
                return Lps(m, word1, word2);
            }

            int Lps(String m, String word1, String word2)
            {
                int n = m.Length;

                int[][] dp = new int[n][];
                for (int l = 0; l < n; l++)
                {
                    for (int i = 0; i + l < n; i++)
                    {
                        int j = i + l;
                        if (l == 0)
                        {
                            dp[i][i] = 1;
                            continue;
                        }
                        if (m[i] == m[j])
                        {
                            dp[i][j] = 2 + dp[i + 1][j - 1];

                        }
                        else
                        {
                            dp[i][j] = Math.Max(dp[i + 1][j], dp[i][j - 1]);
                        }
                    }
                }

                int max = 0;
                for (int i = 0; i < word1.Length; i++)
                {
                    for (int j = word2.Length - 1; j >= 0; j--)
                    {
                        if (word1[i] == word2[j])
                        {
                            max = Math.Max(max, dp[i][word1.Length + j]);
                        }
                    }
                }
                return max;

            }
        }

        /* 2002. Maximum Product of the Length of Two Palindromic Subsequences
        https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/description/
         */
        public class MaximumProductOfLengthOfTwoPaliSubseqs
        {
            public int DPWithBitMasking(String s)
            {
                int[] dp = new int[4096];
                int res = 0, mask = (1 << s.Length) - 1;
                for (int m = 1; m <= mask; ++m)
                    dp[m] = PalSize(s, m);
                for (int m1 = mask; m1 > 0; --m1)
                    if (dp[m1] * (s.Length - dp[m1]) > res)
                        for (int m2 = mask ^ m1; m2 > 0; m2 = (m2 - 1) & (mask ^ m1))
                            res = Math.Max(res, dp[m1] * dp[m2]);
                return res;
            }
            private int PalSize(String s, int mask)
            {
                int p1 = 0, p2 = s.Length, res = 0;
                while (p1 <= p2)
                {
                    if ((mask & (1 << p1)) == 0)
                        ++p1;
                    else if ((mask & (1 << p2)) == 0)
                        --p2;
                    else if (s[p1] != s[p2])
                        return 0;
                    else
                        res += 1 + (p1++ != p2-- ? 1 : 0);
                }
                return res;
            }

        }


        /* 2472. Maximum Number of Non-overlapping Palindrome Substrings
        https://leetcode.com/problems/maximum-number-of-non-overlapping-palindrome-substrings/description/
         */
        public class MaxmimumNumberOfNonOverlappingPaliSubstrings
        {
            /*
            1.Find NonOverallpaing Intervals
            Time complexity = O(nk)

            */
            public int MaxPalindromes(string s, int k)
            {
                int n = s.Length, last = int.MinValue, ans = 0;
                List<List<int>> intervals = new List<List<int>>();
                for (int center = 0; center < 2 * n; center++)
                {
                    int left = center / 2;
                    int right = left + center % 2;
                    while (left >= 0 && right < n && s[left] == s[right])
                    {
                        if (right + 1 - left >= k)
                        {
                            intervals.Add(new List<int> { left, right + 1 });
                            break;
                        }
                        left--; right++;
                    }
                }
                foreach (var v in intervals)
                {
                    if (v[0] >= last)
                    {
                        last = v[1];
                        ans++;
                    }
                    else if (v[1] < last)
                        last = v[1];
                }
                return ans;
            }
            /*
            #2.No need to find non overlapping intervals just record the end of the last found palindromic substring.

            Time complexity = O(nk)
            */
            public int MaxPalindromes2(String s, int k)
            {
                int n = s.Length, ans = 0, start = 0;
                for (int center = 0; center < 2 * n; center++)
                {
                    int left = center / 2;
                    int right = left + center % 2;
                    while (left >= start && right < n && s[left] == s[right])
                    {
                        if (right + 1 - left >= k)
                        {
                            ans++;
                            start = right + 1;
                            break;
                        }
                        left--; right++;
                    }
                }
                return ans;
            }

        }

        /*     131. Palindrome Partitioning
        https://leetcode.com/problems/palindrome-partitioning/description/
         */
        public class PalindromePartitioning
        {
            /*
            Approach 1: Backtracking+DFS
Complexity Analysis
•	Time Complexity : O(N⋅2^N), where N is the length of string s. This is the worst-case time complexity when all the possible substrings are palindrome.
Example, if s is aaa, the recursive tree can be illustrated as follows:
Hence, there could be 2^N possible substrings in the worst case. For each substring, it takes O(N) time to generate the substring and determine if it is a palindrome or not. This gives us a time complexity of O(N⋅2^N)
•	Space Complexity: O(N), where N is the length of the string s. This space will be used to store the recursion stack. For s = aaa, the maximum depth of the recursive call stack is 3 which is equivalent to N.
            */
            public IList<IList<string>> WithBacktrackingDFS(string s)
            {
                var ans = new List<IList<string>>();
                Dfs(0, new List<string>(), s, ans);
                return ans;
            }

            private void Dfs(int start, List<string> currentList, string s,
                             List<IList<string>> result)
            {
                if (start >= s.Length)
                    result.Add(new List<string>(currentList));
                else
                {
                    for (int end = start; end < s.Length; end++)
                    {
                        if (IsPalindrome(s, start, end))
                        {
                            currentList.Add(s.Substring(start, end - start + 1));
                            Dfs(end + 1, currentList, s, result);
                            currentList.RemoveAt(currentList.Count - 1);
                        }
                    }
                }
            }

            bool IsPalindrome(string s, int low, int high)
            {
                while (low < high)
                    if (s[low++] != s[high--])
                        return false;
                return true;
            }

            /*
            Approach 2: Backtracking with Dynamic Programming
Complexity Analysis
•	Time Complexity : O(N⋅2^N), where N is the length of the string s. In the worst case, there could be 2^N possible substrings and it will take O(N) to generate each substring using substr as in Approach 1. However, we are eliminating one additional iteration to check if the substring is a palindrome or not.
•	Space Complexity: O(N⋅N), where N is the length of the string s. The recursive call stack would require N space as in Approach 1. Additionally we also use 2 dimensional array dp of size N⋅N .

            */
            public IList<IList<string>> WithBacktrackingDFSDP(string s)
            {
                int len = s.Length;
                bool[,] dp = new bool[len, len];
                IList<IList<string>> result = new List<IList<string>>();
                dfs(result, s, 0, new List<string>(), dp);
                return result;
            }

            void dfs(IList<IList<string>> result, string s, int start,
                     IList<string> currentList, bool[,] dp)
            {
                if (start >= s.Length)
                    result.Add(new List<string>(currentList));
                for (int end = start; end < s.Length; end++)
                {
                    if (s[start] == s[end] &&
                        (end - start <= 2 || dp[start + 1, end - 1]))
                    {
                        dp[start, end] = true;
                        currentList.Add(s.Substring(start, end - start + 1));
                        dfs(result, s, end + 1, currentList, dp);
                        currentList.RemoveAt(currentList.Count - 1);
                    }
                }
            }
        }


        /* 132. Palindrome Partitioning II
        https://leetcode.com/problems/palindrome-partitioning-ii/description/
         */
        public class MinCutPalindromePartitioningIISol
        {
            /*            
Approach 1: Backtracking
Complexity Analysis
•	Time Complexity: O(N⋅2^N), where N is the length of string s.
•	Space Complexity: O(n). The recursive method uses an internal call stack. In this case, if we place a cut after every character in the string (a|a|b), the size of the internal stack would be at most n.

            */
            public int WithBacktracking(string s)
            {
                return FindMinimumCut(s, 0, s.Length - 1, s.Length - 1);
            }

            private int FindMinimumCut(string s, int start, int end, int minimumCut)
            {
                // base condition, no cut needed for an empty substring or palindrome
                // substring.
                if (start == end || IsPalindrome(s, start, end))
                {
                    return 0;
                }

                for (int currentEndIndex = start; currentEndIndex <= end;
                     currentEndIndex++)
                {
                    // find result for substring (start, currentEndIndex) if it is
                    // palindrome
                    if (IsPalindrome(s, start, currentEndIndex))
                    {
                        minimumCut = Math.Min(
                            minimumCut, 1 + FindMinimumCut(s, currentEndIndex + 1, end,
                                                           minimumCut));
                    }
                }

                return minimumCut;
            }

            private bool IsPalindrome(string s, int start, int end)
            {
                while (start < end)
                {
                    if (s[start++] != s[end--])
                    {
                        return false;
                    }
                }

                return true;
            }

            /*
            
Approach 2: Dynamic Programming - Top Down (Recursion, Memoization)
Complexity Analysis
•	Time Complexity: O(N^2⋅N), where N is the length of string s.
In the recursive method findMinimumCut, we are calculating the results for any substring only once. We know that a string size N has N^2 possible substrings. Thus, the worst-case time complexity of the recursive method findMinimumCut is O(N^2).
Additionally, within each recursive call, we are also checking if a substring is palindrome or not. The worst-case time complexity for method isPalindrome is O(N/2).
This gives us total time complexity as, O(N2)⋅O(N/2)=O(N^2⋅N)
•	Space Complexity: O(N^2), as we are using two 2-dimensional arrays memoCuts and memoPalindrome of size N⋅N.
This gives us total space complexity as (N^2+N^2)=N^2.	

            */

            public class TopDownDPRecWithMemo
            {
                private int?[][] memoCuts;
                private bool?[][] memoPalindrome;

                public int MinCut(string s)
                {
                    memoCuts = new int?[s.Length][];
                    memoPalindrome = new bool?[s.Length][];
                    for (int i = 0; i < s.Length; i++)
                    {
                        memoCuts[i] = new int?[s.Length];
                        memoPalindrome[i] = new bool?[s.Length];
                    }

                    return FindMinimumCut(s, 0, s.Length - 1, s.Length - 1).Value;
                }

                private int? FindMinimumCut(string s, int start, int end, int minimumCut)
                {
                    // base case
                    if (start == end || IsPalindrome(s, start, end).Value)
                    {
                        return 0;
                    }

                    // check for results in memoCuts
                    if (memoCuts[start][end].HasValue)
                    {
                        return memoCuts[start][end].Value;
                    }

                    for (int currentEndIndex = start; currentEndIndex <= end;
                         currentEndIndex++)
                    {
                        if (IsPalindrome(s, start, currentEndIndex).Value)
                        {
                            minimumCut = Math.Min(
                                minimumCut,
                                1 + FindMinimumCut(s, currentEndIndex + 1, end, minimumCut)
                                        .Value);
                        }
                    }

                    return memoCuts[start][end] = minimumCut;
                }

                private bool? IsPalindrome(string s, int start, int end)
                {
                    if (start >= end)
                    {
                        return true;
                    }

                    // check for results in memoPalindrome
                    if (memoPalindrome[start][end] != null)
                    {
                        return memoPalindrome[start][end].Value;
                    }

                    return memoPalindrome[start][end] =
                               (s[start] == s[end]) &&
                               IsPalindrome(s, start + 1, end - 1).Value;
                }
            }

            /*
            Approach 3: Dynamic Programming - Top Down (Optimized Space Complexity)
Complexity Analysis
•	Time Complexity: O(N^2⋅N), where N is the length of string s.
The time complexity is the same as in Approach 2.
•	Space Complexity: O(N^2), as we are using one 1-dimensional array memoCuts of size N and one 2-dimensional array memoPalindrome of size N⋅N.
This gives us a total space complexity of (N+N^N)=N^2.

            */
            public class TopDownDPRecSpaceOptimal
            {
                private int?[] memoCuts;
                private bool?[,] memoPalindrome;

                public int MinCut(string s)
                {
                    memoCuts = new int?[s.Length];
                    memoPalindrome = new bool?[s.Length, s.Length];
                    return FindMinimumCut(s, 0, s.Length - 1, s.Length - 1) ?? 0;
                }

                private int? FindMinimumCut(string s, int start, int end, int minimumCut)
                {
                    // base case
                    if (start == end || (IsPalindrome(s, start, end) ?? false))
                    {
                        return 0;
                    }

                    // check for results in memoCuts
                    if (memoCuts[start].HasValue)
                    {
                        return memoCuts[start].Value;
                    }

                    for (int currentEndIndex = start; currentEndIndex <= end;
                         currentEndIndex++)
                    {
                        if (IsPalindrome(s, start, currentEndIndex) ?? false)
                        {
                            minimumCut = Math.Min(
                                minimumCut, 1 + (FindMinimumCut(s, currentEndIndex + 1, end,
                                                                minimumCut) ??
                                                 0));
                        }
                    }

                    return memoCuts[start] = minimumCut;
                }

                private bool? IsPalindrome(string s, int start, int end)
                {
                    if (start >= end)
                    {
                        return true;
                    }

                    // check for results in memoPalindrome
                    if (memoPalindrome[start, end].HasValue)
                    {
                        return memoPalindrome[start, end].Value;
                    }

                    return memoPalindrome[start, end] =
                               (s[start] == s[end]) &&
                               (IsPalindrome(s, start + 1, end - 1) ?? false);
                }
            }
            /*
            Approach 4: Dynamic Programming - Bottom Up (Tabulation)
            Complexity Analysis
            •	Time Complexity: O(N^2), where N is the length of string s.
            We are iterating N⋅N times to build the palindromeDp array and N⋅N times to find the minimum cuts in a nested for-loop. This gives us a total time complexity of O(N⋅N)+O(N⋅N)=O(N⋅N).
            •	Space Complexity: O(N^2), as we are using a 2-dimensional arrays palindromeDp of size N⋅N and a 1-dimensional array cutsDp of size N. Thus, the space complexity can be given by, O(N⋅N)+O(N)=O(N⋅N).

            */
            public class BottomUpDPSol
            {
                public int MinCut(string s)
                {
                    int[] cutsDp = new int[s.Length];
                    bool[,] palindromeDp = new bool[s.Length, s.Length];
                    // build the palindrome cutsDp for all susbtrings
                    BuildPalindromeDp(s, s.Length, ref palindromeDp);
                    for (int end = 0; end < s.Length; end++)
                    {
                        int minimumCut = end;
                        for (int start = 0; start <= end; start++)
                        {
                            if (palindromeDp[start, end])
                            {
                                minimumCut = start == 0 ? 0
                                                        : Math.Min(minimumCut,
                                                                   cutsDp[start - 1] + 1);
                            }
                        }

                        cutsDp[end] = minimumCut;
                    }

                    return cutsDp[s.Length - 1];
                }

                void BuildPalindromeDp(string s, int n, ref bool[,] palindromeDp)
                {
                    for (int end = 0; end < s.Length; end++)
                    {
                        for (int start = 0; start <= end; start++)
                        {
                            if (s[start] == s[end] &&
                                (end - start <= 2 || palindromeDp[start + 1, end - 1]))
                            {
                                palindromeDp[start, end] = true;
                            }
                        }
                    }
                }
            }
            /*
            Approach 5: Optimized Tabulation Approach
Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of string s.
We are iterating N⋅N times only once to find the minimum cuts.
•	Space Complexity: O(N^2), as we are using two 2-dimensional arrays palindromeDp and 1-dimensional array cutsDp of size N⋅N. Thus the space complexity can be given by, O(N⋅N)+O(N)=O(N⋅N) .

            */
            public class BottomUpDPSpaceOptimalSpl
            {
                public int MinCut(string s)
                {
                    int[] cuts = new int[s.Length];
                    bool[][] palindrome = new bool[s.Length][];
                    for (int i = 0; i < s.Length; i++) palindrome[i] = new bool[s.Length];
                    for (int end = 0; end < s.Length; end++)
                    {
                        int minimumCut = end;
                        for (int start = 0; start <= end; start++)
                        {
                            // check if substring (start, end) is palindrome
                            if (s[start] == s[end] &&
                                (end - start <= 2 || palindrome[start + 1][end - 1]))
                            {
                                palindrome[start][end] = true;
                                minimumCut =
                                    start == 0 ? 0
                                               : Math.Min(minimumCut, cuts[start - 1] + 1);
                            }
                        }

                        cuts[end] = minimumCut;
                    }

                    return cuts[s.Length - 1];
                }
            }

            /*
            Approach 6: Expand Around the Center
Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of string s.
The outer loop that fixes the middle index iterates N times. The are 2 inner loops iterates for N/2 times each. This gives us time complexity as, O(N⋅(N/2+N/2))=O(N^2).
•	Space Complexity: O(N), as we are using single 1 dimensional array cutsDp of size N.
            */
            public class ExpandAroundCenterSol
            {
                public int MinCut(string s)
                {
                    int[] cutsDp = new int[s.Length];
                    for (int i = 1; i < s.Length; i++)
                    {
                        cutsDp[i] = i;
                    }

                    for (int mid = 0; mid < s.Length; mid++)
                    {
                        // check for odd length palindrome around mid index
                        FindMinimumCuts(mid, mid, cutsDp, s);
                        // check for even length palindrome around mid index
                        FindMinimumCuts(mid - 1, mid, cutsDp, s);
                    }

                    return cutsDp[s.Length - 1];
                }

                public void FindMinimumCuts(int startIndex, int endIndex, int[] cutsDp,
                                            string s)
                {
                    for (int start = startIndex, end = endIndex;
                         start >= 0 && end < s.Length && s[start] == s[end];
                         start--, end++)
                    {
                        int newCut = start == 0 ? 0 : cutsDp[start - 1] + 1;
                        cutsDp[end] = Math.Min(cutsDp[end], newCut);
                    }
                }
            }

        }


        /* 1278. Palindrome Partitioning III
        https://leetcode.com/problems/palindrome-partitioning-iii/description/
         */
        public class PalindromePartitionIIISol
        {
            /*
            1. Top Down DP with Memo
            */
            public int TopDownDPWithMemo(string s, int k)
            {
                int[][] toPal = new int[s.Length][];
                int[][] dp = new int[k + 1][];
                for (int i = 0; i < s.Length; i++)
                {
                    toPal[i][i] = 0;
                }
                for (int i = s.Length - 1; i >= 0; i--)
                {
                    for (int j = i + 1; j < s.Length; j++)
                    {
                        toPal[i][j] = GetChanges(s, i, j);
                    }
                }
                for (int i = 0; i < s.Length; i++)
                {
                    dp[1][i] = toPal[0][i];
                }
                for (int i = 2; i <= k; i++)
                {
                    for (int end = i - 1; end < s.Length; end++)
                    {
                        int min = s.Length;
                        for (int start = end - 1; start >= i - 2; start--)
                        {
                            min = Math.Min(min, dp[i - 1][start] + toPal[start + 1][end]);
                        }
                        dp[i][end] = min;
                    }
                }
                return dp[k][s.Length - 1];
            }


            private int GetChanges(string s, int start, int end)
            {
                int changes = 0;
                while (start < end)
                {
                    if (s[start++] != s[end--])
                    {
                        changes++;
                    }
                }
                return changes;
            }
        }

        /* 1745. Palindrome Partitioning IV
        https://leetcode.com/problems/palindrome-partitioning-iv/description/
         */
        public class PalindromePartitionIVSol
        {
            /*
            O(N^2)
            */
            public bool CheckPartitioning(string s)
            {
                int N = s.Length;
                char[] A = s.ToCharArray();

                // build dp table
                bool[][] dp = new bool[N][];
                for (int i = N - 1; i >= 0; --i)
                {
                    for (int j = i; j < N; ++j)
                    {
                        if (A[i] == A[j]) dp[i][j] = ((i + 1 <= j - 1) ? dp[i + 1][j - 1] : true);
                        else dp[i][j] = false;
                    }
                }

                // iterate every mid and then check: left, mid and right
                for (int i = 1; i < N - 1; ++i)
                {
                    for (int j = i; j < N - 1; ++j)
                    {
                        if (dp[0][i - 1] && dp[i][j] && dp[j + 1][N - 1]) return true;
                    }
                }

                return false;

            }
        }
        /* 
        2911. Minimum Changes to Make K Semi-palindromes
        https://leetcode.com/problems/minimum-changes-to-make-k-semi-palindromes/description/
         */
        public class MinimumChangesToMakeKSemiPalindromesSol
        {
            /*
            Complexity
There are n^2 pairs of (i,j),
and have logn factors.
For funciton semi,
Time and space are O(n^2logn).
For funciton change,
Time O(n^2logn), space are O(n^2).
For funciton dp,
Time O(nnk), space are O(nk).
Time O(nnlogn + nnk)
Space O(n^2logn)


            */
            public const int maxSubstringNum = 101;
            public const int baseValue = (int)1e9; // 1000 000 000

            public int MinimumChanges(string s, int k)
            {
                int n = s.Length;
                // Define dp[i][j] as the minimum count of letter changes needed to split the suffix of string s starting from s[i] into j valid parts.
                int[,] dp = new int[n, k + 1];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < k + 1; j++)
                        dp[i, j] = baseValue;

                // We have dp[i][j] = min(dp[x + 1][j - 1] + v[i][x]).
                // Here v[i][x] is the minimum number of letter changes to change substring s[i..x] into semi-palindrome. - changeToPal value -
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        int changeToPal = CTP(j, i, ref s);
                        for (int p = 1; p < k + 1; p++)
                        {
                            if (p == 1 && j == 0)
                                dp[i, p] = Math.Min(dp[i, p], changeToPal);
                            else if (p > 0 && j > 0)
                                dp[i, p] = Math.Min(dp[i, p], dp[j - 1, p - 1] + changeToPal);
                        }
                    }
                }

                return dp[n - 1, k];
            }

            // v[i][j] - changeToPal value - can be calculated separately by brute-force. 
            private int CTP(int l, int r, ref string s)
            {
                if (r - l == 1)
                    return s[l] == s[r] ? 0 : 1;

                // Count changes value for each valid substring length - t -
                int ans = baseValue;
                for (int t = 1; t < r - l; t++)
                {
                    if ((r - l + 1) % t != 0) continue;

                    string[] str = new string[maxSubstringNum];
                    Array.Fill(str, "");
                    for (int i = l; i < r + 1; i++)
                        str[(i - l + 1) % t] += s[i];

                    int changes = 0;
                    for (int d = 0; d < maxSubstringNum; d++)
                        if (str[d].Length > 0)
                        {
                            for (int i = 0, j = str[d].Length - 1; i < j + 1; i++, j--)
                                if (str[d][i] != str[d][j])
                                    changes++;
                        }
                        // break if we reached the void tail
                        else break;

                    ans = Math.Min(ans, changes);
                }

                return ans;
            }
        }


        /* 564. Find the Closest Palindrome
        https://leetcode.com/problems/find-the-closest-palindrome/description/
         */
        public class ClosestPalindromeSol
        {
            /*

Approach 1: Find Previous and Next Palindromes
Complexity Analysis
Let n be the number of digits in the input number.
•	Time complexity: O(n)
We perform operations on exactly 5 strings. The palindrome construction for each string takes O(n) time. Therefore, total time complexity is given by O(n).
•	Space complexity: O(n)
We store the 5 possible candidates in the possibilities array. Apart from this, the built-in functions used to make the firstHalf can potentially lead to O(n) space complexity, as they copy the characters into a new String. Therefore, the total space complexity is O(n).

            */
            public string FindPreviousAndNextPalindromes(string n)
            {
                int length = n.Length;
                int index = length % 2 == 0 ? length / 2 - 1 : length / 2;
                long firstHalf = long.Parse(n.Substring(0, index + 1));
                /* 
                Generate possible palindromic candidates:
                1. Create a palindrome by mirroring the first half.
                2. Create a palindrome by mirroring the first half incremented by 1.
                3. Create a palindrome by mirroring the first half decremented by 1.
                4. Handle edge cases by considering palindromes of the form 999... 
                   and 100...001 (smallest and largest n-digit palindromes).
                */
                List<long> possibilities = new List<long>();

                possibilities.Add(HalfToPalindrome(firstHalf, length % 2 == 0));
                possibilities.Add(HalfToPalindrome(firstHalf + 1, length % 2 == 0));
                possibilities.Add(HalfToPalindrome(firstHalf - 1, length % 2 == 0));
                possibilities.Add((long)Math.Pow(10, length - 1) - 1);
                possibilities.Add((long)Math.Pow(10, length) + 1);

                // Find the palindrome with minimum difference, and minimum value.
                long difference = long.MaxValue, result = 0, number = long.Parse(n);
                foreach (long candidate in possibilities)
                {
                    if (candidate == number) continue;
                    if (Math.Abs(candidate - number) < difference)
                    {
                        difference = Math.Abs(candidate - number);
                        result = candidate;
                    }
                    else if (Math.Abs(candidate - number) == difference)
                    {
                        result = Math.Min(result, candidate);
                    }
                }

                return result.ToString();
            }

            private long HalfToPalindrome(long left, bool isEven)
            {
                // Convert the given half to palindrome.
                long result = left;
                if (!isEven) left = left / 10;
                while (left > 0)
                {
                    result = result * 10 + (left % 10);
                    left /= 10;
                }
                return result;
            }
            /*
            Approach 2: Binary Search
         Complexity Analysis
Let m be the input number and n be the number of digits in it.
•	Time complexity: O(n⋅log(m))
We perform two binary search operations on a search space of size m, and in each operation iterate through all the digits. Therefore, the total time complexity is given by O(n⋅log(m)).
•	Space complexity: O(n)
The space complexity is primarily determined by the storage needed for the string representation of the number and the intermediate list or character array used for manipulation. Since these data structures are proportional to the number of digits in O(n), the total space complexity is O(n).
For C++: to_string(num) - Converts the number to a string, which requires space proportional to the number of digits in O(n), i.e., O(n).
For Java: Long.toString(num) - Converts the number to a string, requiring O(n) space.
For Python: ''.join(s_list) - Creates a new string from the list, requiring O(n) space.
   
            */
            public string UsingBinarySearch(string numberString)
            {
                long number = long.Parse(numberString);
                long previous = PreviousPalindrome(number);
                long next = NextPalindrome(number);
                if (Math.Abs(previous - number) <= Math.Abs(next - number))
                {
                    return previous.ToString();
                }
                return next.ToString();
            }
            // Convert to palindrome keeping first half constant.
            private long Convert(long number)
            {
                string numberString = number.ToString();
                int stringLength = numberString.Length;
                int leftIndex = (stringLength - 1) / 2, rightIndex = stringLength / 2;
                char[] numberArray = numberString.ToCharArray();
                while (leftIndex >= 0)
                {
                    numberArray[rightIndex++] = numberArray[leftIndex--];
                }
                return long.Parse(new string(numberArray));
            }

            // Find the previous palindrome, just smaller than number.
            private long PreviousPalindrome(long number)
            {
                long leftBoundary = 0, rightBoundary = number;
                long answer = long.MinValue;
                while (leftBoundary <= rightBoundary)
                {
                    long middle = (rightBoundary - leftBoundary) / 2 + leftBoundary;
                    long palindrome = Convert(middle);
                    if (palindrome < number)
                    {
                        answer = palindrome;
                        leftBoundary = middle + 1;
                    }
                    else
                    {
                        rightBoundary = middle - 1;
                    }
                }
                return answer;
            }

            // Find the next palindrome, just greater than number.
            private long NextPalindrome(long number)
            {
                long leftBoundary = number, rightBoundary = (long)1e18;
                long answer = long.MinValue;
                while (leftBoundary <= rightBoundary)
                {
                    long middle = (rightBoundary - leftBoundary) / 2 + leftBoundary;
                    long palindrome = Convert(middle);
                    if (palindrome > number)
                    {
                        answer = palindrome;
                        rightBoundary = middle - 1;
                    }
                    else
                    {
                        leftBoundary = middle + 1;
                    }
                }
                return answer;
            }
        }


        /* 1842. Next Palindrome Using Same Digits
        https://leetcode.com/problems/next-palindrome-using-same-digits/description/
         */
        public class NextPalindromeUsingSameDigitsSol
        {
            /*
            Time O(N)
            */
            public string NextPalindrome(string number)
            {
                int length = number.Length;
                int[] halfArray = new int[length / 2];
                for (int index = 0; index < halfArray.Length; index++)
                {
                    halfArray[index] = number[index] - '0';
                }
                if (!NextPermutation(halfArray)) return "";

                StringBuilder stringBuilder = new StringBuilder();
                foreach (int item in halfArray)
                {
                    stringBuilder.Append(item);
                }

                if (length % 2 == 0)
                    return stringBuilder.ToString() + ReverseString(stringBuilder.ToString());
                else
                    return stringBuilder.ToString() + number.Substring(length / 2, 1) + ReverseString(stringBuilder.ToString());
            }

            private bool NextPermutation(int[] numbers)
            {
                int lastIndex = numbers.Length - 1, pivotIndex = -1;
                for (int index = lastIndex - 1; index >= 0; index--)
                {
                    if (numbers[index] < numbers[index + 1])
                    {
                        pivotIndex = index;
                        break;
                    }
                }

                if (pivotIndex == -1)
                {
                    ReverseArray(numbers, 0, lastIndex);
                    return false;
                }

                for (int index = lastIndex; index >= 0; index--)
                {
                    if (numbers[index] > numbers[pivotIndex])
                    {
                        Swap(numbers, pivotIndex, index);
                        break;
                    }
                }
                ReverseArray(numbers, pivotIndex + 1, lastIndex);
                return true;
            }

            private void ReverseArray(int[] numbers, int start, int end)
            {
                while (start < end) Swap(numbers, start++, end--);
            }

            private void Swap(int[] numbers, int start, int end)
            {
                int temp = numbers[start];
                numbers[start] = numbers[end];
                numbers[end] = temp;
            }

            private string ReverseString(string str)
            {
                char[] charArray = str.ToCharArray();
                Array.Reverse(charArray);
                return new string(charArray);
            }
        }






    }
}