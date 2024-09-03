using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class Combinatorics
    {
        /*
        46. Permutations
https://leetcode.com/problems/permutations/description/

        */
        public IList<IList<int>> Permute(int[] nums)
        {
            //Approach: Backtracking
            /*
Complexity Analysis
Note: most backtracking problems, including this one, have extremely difficult time complexities to derive. Don't be discouraged if you can't derive it on your own - most of the time, the analysis requires an esoteric understanding of math.
Given n as the length of nums,
•	Time complexity, what you should say in an interview: O(n⋅n!)
•	Space complexity: O(n)
We don't count the answer as part of the space complexity. The extra space we use here is for curr and the recursion call stack. The depth of the call stack is equal to the length of curr, which is limited to n.

            */
            List<IList<int>> ans = new List<IList<int>>();
            Backtrack(new List<int>(), ans, nums);
            return ans;

        }
        void Backtrack(List<int> curr, List<IList<int>> ans, int[] nums)
        {
            if (curr.Count == nums.Length)
            {
                ans.Add(new List<int>(curr));
                return;
            }

            foreach (int num in nums)
            {
                if (!curr.Contains(num))
                {
                    curr.Add(num);
                    Backtrack(curr, ans, nums);
                    curr.RemoveAt(curr.Count - 1);
                }
            }
        }

        /*
        47. Permutations II
        https://leetcode.com/problems/permutations-ii/description/

        Approach 1: Backtracking with Groups of Numbers
        Complexity Analysis
        Let N be the length of the input array.
        Hence, the number of permutations would be at maximum N!, i.e. N⋅(N−1)⋅(N−2)...1, when each number in the array is unique.
        •	Time Complexity: O(∑k=1NP(N,k)) where P(N,k)=(N−k)!N!=N(N−1)...(N−k+1)
        is so-called k-permutations_of_N or partial permutation.
        •	Space Complexity: O(N)
        o	First of all, we build a hash table out of the input numbers. In the worst case where each number is unique, we would need O(N) space for the table.
        o	Since we applied recursion in the algorithm which consumes some extra space in the function call stack, we would need another O(N) space for the recursion.
        o	During the exploration, we keep a candidate of permutation along the way, which takes yet another O(N).

        */
        public IList<IList<int>> PermuteUnique(int[] nums)
        {
            List<IList<int>> results = new List<IList<int>>();
            Dictionary<int, int> counter = new Dictionary<int, int>();
            foreach (int num in nums)
            {
                if (!counter.ContainsKey(num))
                    counter.Add(num, 0);
                counter[num]++;
            }

            List<int> comb = new List<int>();
            this.Backtrack(comb, nums.Length, counter, results);
            return results;
        }

        private void Backtrack(List<int> comb, int N, Dictionary<int, int> counter,
                               List<IList<int>> results)
        {
            if (comb.Count == N)
            {
                results.Add(new List<int>(comb));
                return;
            }

            foreach (var entry in counter.ToList())
            {
                int num = entry.Key;
                int count = entry.Value;
                if (count == 0)
                    continue;
                comb.Add(num);
                counter[num]--;
                this.Backtrack(comb, N, counter, results);
                comb.RemoveAt(comb.Count - 1);
                counter[num]++;
            }
        }

        /*
        31. Next Permutation	
      https://leetcode.com/problems/next-permutation/description/
        */
        public void NextPermutation(int[] nums)
        {
            /*
Approach 1: Brute Force
 Complexity Analysis
•	Time complexity : O(n!). Total possible permutations is n!.
•	Space complexity : O(n). Since an array will be used to store the permutations.

Approach 2: Single Pass Approach (SP)
Complexity Analysis
•	Time complexity : O(n). In worst case, only two scans of the whole array are needed.
•	Space complexity : O(1). No extra space is used. In place replacements are done

            */
            int i = nums.Length - 2;
            while (i >= 0 && nums[i + 1] <= nums[i])
            {
                i--;
            }

            if (i >= 0)
            {
                int j = nums.Length - 1;
                while (nums[j] <= nums[i])
                {
                    j--;
                }

                Swap(nums, i, j);
            }

            Reverse(nums, i + 1);

        }
        private void Reverse(int[] nums, int start)
        {
            int i = start, j = nums.Length - 1;
            while (i < j)
            {
                Swap(nums, i, j);
                i++;
                j--;
            }
        }

        private void Swap(int[] nums, int i, int j)
        {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }



        /*
    77. Combinations
    https://leetcode.com/problems/combinations/description/
        /*
          Approach: Backtracking With Tree(BTT)
          Complexity Analysis
    Note: most backtracking problems, including this one, have extremely difficult time complexities to derive. Don't be discouraged if you can't derive it on your own - most of the time, the analysis requires an esoteric understanding of math. If you are asked this question in an interview, do your best to state an upper bound on the complexity by analyzing the number of nodes in the tree and the work done at each node.
    •	Time complexity: O(n! /(k−1)!⋅(n−k)!)

    •	Space complexity: O(k)
    We don't count the answer as part of the space complexity. The extra space we use here is for curr and the recursion call stack. The depth of the call stack is equal to the length of curr, which is limited to k.

          */
        private int n;
        private int k;

        public IList<IList<int>> Combine(int n, int k)
        {
            this.n = n;
            this.k = k;
            List<List<int>> ans = new List<List<int>>();
            Backtrack(new List<int>(), 1, ans);
            return ans.Cast<IList<int>>().ToList();
        }

        public void Backtrack(List<int> curr, int firstNum, List<List<int>> ans)
        {
            if (curr.Count == k)
            {
                ans.Add(new List<int>(curr));
                return;
            }

            int need = k - curr.Count;
            int remain = n - firstNum + 1;
            int available = remain - need;
            for (int num = firstNum; num <= firstNum + available; num++)
            {
                curr.Add(num);
                Backtrack(curr, num + 1, ans);
                curr.RemoveAt(curr.Count - 1);
            }
        }


        /*
        39. Combination Sum
        https://leetcode.com/problems/combination-sum/description/

        Approach 1: Backtracking
        Complexity Analysis
        Let N be the number of candidates, T be the target value, and M be the minimal value among the candidates.
        •	Time Complexity: O(N^(T/M)+1)
        o	As we illustrated before, the execution of the backtracking is unfolded as a DFS traversal in a n-ary tree.
        The total number of steps during the backtracking would be the number of nodes in the tree.
        o	At each node, it takes a constant time to process, except the leaf nodes which could take a linear time to make a copy of combination. So we can say that the time complexity is linear to the number of nodes of the execution tree.
        o	Here we provide a loose upper bound on the number of nodes.
        o	First of all, the fan-out of each node would be bounded to N, i.e. the total number of candidates.
        o	The maximal depth of the tree, would be MT, where we keep on adding the smallest element to the combination.
        o	As we know, the maximal number of nodes in N-ary tree of MT height would be NMT+1.
        o	Note that, the actual number of nodes in the execution tree would be much smaller than the upper bound, since the fan-out of the nodes are decreasing level by level.
        •	Space Complexity: O(T/M)
        o	We implement the algorithm in recursion, which consumes some additional memory in the function call stack.
        o	The number of recursive calls can pile up to MT, where we keep on adding the smallest element to the combination.
        As a result, the space overhead of the recursion is O(T/M).
        o	In addition, we keep a combination of numbers during the execution, which requires at most O(T/M) space as well.
        o	To sum up, the total space complexity of the algorithm would be O(T/M).
        o	Note that, we did not take into the account the space used to hold the final results for the space complexity.


        */
        public IList<IList<int>> CombinationSum(int[] candidates, int target)
        {
            List<IList<int>> results = new List<IList<int>>();
            this.backtrack(target, new List<int>(), candidates, 0, results);
            return results;
        }

        private void backtrack(int remain, List<int> comb, int[] candidates,
                               int start, List<IList<int>> results)
        {
            if (remain == 0)
            {
                results.Add(new List<int>(comb));
                return;
            }
            else if (remain < 0)
            {
                return;
            }

            for (int i = start; i < candidates.Length; ++i)
            {
                comb.Add(candidates[i]);
                this.backtrack(remain - candidates[i], comb, candidates, i,
                               results);
                comb.RemoveAt(comb.Count - 1);
            }
        }

        /*
        40. Combination Sum II
        https://leetcode.com/problems/combination-sum-ii/description/

        Approach: Backtracking

        Complexity Analysis
        Let N be the number of candidates in the array.
        •	Time complexity: O(2^N)
        In the worst case, our algorithm will exhaust all possible combinations from the input array. Again, in the worst case, let us assume that each number is unique. The number of combinations for an array of size N would be 2^N, i.e. each number is included or excluded in a combination.
        Additionally, it takes O(N) time to build a counter table out of the input array.
        Therefore, the overall time complexity of the algorithm is dominated by the backtracking process, which is O(2^N).

        •	Space complexity: O(N)
        We first create a tempList, which in the worst case will consume O(N) space to keep track of the combinations. In addition, we apply recursion in the algorithm, which will incur additional memory consumption in the function call stack. In the worst case, the stack will pile up to O(N) space.
        To sum up, the overall space complexity of the algorithm is O(N).

        */
        public List<List<int>> CombinationSum2(int[] candidates, int target)
        {
            List<List<int>> resultList = new List<List<int>>();
            Array.Sort(candidates);
            Backtrack(resultList, new List<int>(), candidates, target, 0);
            return resultList;
        }

        private void Backtrack(
            List<List<int>> answer,
            List<int> tempList,
            int[] candidates,
            int totalLeft,
            int index
        )
        {
            if (totalLeft < 0) return;
            else if (totalLeft == 0) // Add to the answer array, if the sum is equal to target.
            {
                answer.Add(new List<int>(tempList));
            }
            else
            {
                for (int i = index; i < candidates.Length && totalLeft >= candidates[i]; i++)
                {
                    if (i > index && candidates[i] == candidates[i - 1]) continue;
                    // Add it to tempList.
                    tempList.Add(candidates[i]);
                    // Check for all possible scenarios.
                    Backtrack(answer, tempList, candidates, totalLeft - candidates[i], i + 1);
                    // Backtrack the tempList.
                    tempList.RemoveAt(tempList.Count - 1);
                }
            }
        }

        protected void Backtrack(
        int remainingSum,
        int combinationSize,
        LinkedList<int> combination,
        int nextStart,
        List<List<int>> results
    )
        {
            if (remainingSum == 0 && combination.Count == combinationSize)
            {
                // Note: it's important to make a deep copy here,
                // Otherwise the combination would be reverted in other branch of backtracking.
                results.Add(new List<int>(combination));
                return;
            }
            else if (remainingSum < 0 || combination.Count == combinationSize)
            {
                // Exceed the scope, no need to explore further.
                return;
            }

            // Iterate through the reduced list of candidates.
            for (int i = nextStart; i < 9; ++i)
            {
                combination.AddLast(i + 1);
                this.Backtrack(remainingSum - i - 1, combinationSize, combination, i + 1, results);
                combination.RemoveLast();
            }
        }

        /*
        216. Combination Sum III
        https://leetcode.com/problems/combination-sum-iii/description/

        Approach 1: Backtracking
        Complexity Analysis
        Let K be the number of digits in a combination.
        •	Time Complexity: O(9!⋅K /(9−K)! )
        o	In a worst scenario, we have to explore all potential combinations to the very end, i.e. the sum n is a large number (n>9∗9). At the first step, we have 9 choices, while at the second step, we have 8 choices, so on and so forth.
        o	The number of exploration we need to make in the worst case would be P(9,K)= 9! /(9−K)! , assuming that K<=9. By the way, K cannot be greater than 9, otherwise we cannot have a combination whose digits are all unique.
        o	Each exploration takes a constant time to process, except the last step where it takes O(K) time to make a copy of combination.
        o	To sum up, the overall time complexity of the algorithm would be 9! /(9−K)! ⋅O(K)=O(9!⋅K /(9−K)!).
        •	Space Complexity: O(K)
        o	During the backtracking, we used a list to keep the current combination, which holds up to K elements, i.e. O(K).
        o	Since we employed recursion in the backtracking, we would need some additional space for the function call stack, which could pile up to K consecutive invocations, i.e. O(K).
        o	Hence, to sum up, the overall space complexity would be O(K).
        o	Note that, we did not take into account the space for the final results in the space complexity.

        */
        public List<List<int>> CombinationSum3(int k, int n)
        {
            List<List<int>> results = new List<List<int>>();
            LinkedList<int> combination = new LinkedList<int>();

            this.Backtrack(n, k, combination, 0, results);
            return results;
        }

        /*
        377. Combination Sum IV
        https://leetcode.com/problems/combination-sum-iv/description/

        */
        public int CombinationSum4(int[] nums, int target)
        {
            /*
            Approach 1: Top-Down Dynamic Programming (TDDP)
Complexity Analysis
Let T be the target value, and N be the number of elements in the input array.
•	Time Complexity: O(T⋅N)
o	Thanks to the memoization technique, for each invocation of combs(remain), it would be evaluated only once, for each unique input value.
In the worst case, we could have T different input values.
o	For each invocation of combs(remain), in the worst case it takes O(N) time for the non-recursive part.
o	To sum up, the overall time complexity of the algorithm is T⋅O(N)=O(T⋅N).
•	Space Complexity: O(T)
o	Due to the recursive function, the algorithm will incur additional memory consumption in the function call stack.
In the worst case, the recursive function can pile up to T times.
As a result, we would need O(T) space for the recursive function.
o	In addition, since we applied the memoization technique where we keep the intermediate results in the cache, we would need addtionally O(T) space.
o	To sum up, the overall space complexity of the algorithm is O(T)+O(T)=O(T).

            */
            int combSum = CombinationSum4TDDP(nums, target);
            /*
        Approach 2: Bottom-Up Dynamic Programming (BUDP)
        Complexity Analysis
Let T be the target value, and N be the number of elements in the input array.
•	Time Complexity: O(T⋅N)
o	We have a nested loop, with the number of iterations as T and N respectively.
o	Hence, the overall time complexity of the algorithm is O(T⋅N).
•	Space Complexity: O(T)
o	We allocate an array dp[i] to hold all the intermediate values, which amounts to O(T) space.

            */
            combSum = CombinationSum4BUDP(nums, target);

            return combSum;
        }

        private Dictionary<int, int> memo;

        public int CombinationSum4TDDP(int[] nums, int target)
        {
            // minor optimization
            // Arrays.sort(nums);
            memo = new Dictionary<int, int>();
            return combs(nums, target);
        }

        private int combs(int[] nums, int remain)
        {
            if (remain == 0)
                return 1;
            if (memo.ContainsKey(remain))
                return memo[remain];

            int result = 0;
            foreach (int num in nums)
            {
                if (remain - num >= 0)
                    result += combs(nums, remain - num);
                // minor optimizaton, early stopping
                // else
                //     break;
            }
            memo.Add(remain, result);
            return result;
        }
        public int CombinationSum4BUDP(int[] nums, int target)
        {
            // minor optimization
            // Arrays.sort(nums);
            int[] dp = new int[target + 1];
            dp[0] = 1;

            for (int combSum = 1; combSum < target + 1; ++combSum)
            {
                foreach (int num in nums)
                {
                    if (combSum - num >= 0)
                        dp[combSum] += dp[combSum - num];
                    // minor optimizaton, early stopping
                    // else
                    //     break;
                }
            }
            return dp[target];
        }

        /*
        266. Palindrome Permutation
https://leetcode.com/problems/palindrome-permutation/description/
        */

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
        /*
        267. Palindrome Permutation II
https://leetcode.com/problems/palindrome-permutation-ii/description/

        */
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

        /*
        60. Permutation Sequence		
        https://leetcode.com/problems/permutation-sequence/description/

        Approach 1: Factorial Number System
        Complexity Analysis
        •	Time complexity: O(N^2), because to delete elements from the list in a loop one has to perform N+(N−1)+...+1=N(N−1)/2 operations.
        •	Space complexity: O(N).

        */
        public string GetPermutation(int n, int k)
        {
            int[] factorials = new int[n];               // Factorial system bases
            List<char> nums = new List<char>() { '1' };  // Numbers
            factorials[0] =
                1;  // Generate factorial system bases 0!, 1!, ..., (n - 1)!
            for (int i = 1; i < n; ++i)
            {
                // Generate nums 1, 2, ..., n
                factorials[i] = factorials[i - 1] * i; 
                nums.Add((char)(i + 1 + '0'));
            }

            // Fit k in the interval 0 ... (n! - 1)
            k--;
            // Compute the factorial representation of k
            StringBuilder result = new StringBuilder();
            for (int i = n - 1; i > -1; --i)
            {
                int idx = k / factorials[i];
                k -= idx * factorials[i];
                result.Append(nums[idx]);
                nums.RemoveAt(idx);
            }

            return result.ToString();
        }



    }


}