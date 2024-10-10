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

        /*
        567. Permutation in String
        https://leetcode.com/problems/permutation-in-string/description/


        */
        public bool CheckPermuteInclusion(string s1, string s2)
        {
            /*
Approach 1: Brute Force
Complexity Analysis
Let n be the length of s1
•	Time complexity: O(n!).
•	Space complexity: O(n2). The depth of the recursion tree is n(n refers to the length of the short string s1). Every node of the recursion tree contains a string of max. length n.

            */

            bool isPermuteIncluded = CheckPermuteInclusionNaive(s1, s2);
            /*
  Approach 2: Using sorting:          
   Complexity Analysis
Let l1 be the length of string s1 and l2 be the length of string s2.
•	Time complexity: O(l1log(l1)+(l2−l1)l1log(l1)).
•	Space complexity: O(l1). t array is used.
         
            */
            isPermuteIncluded = CheckPermuteInclusionSort(s1, s2);

            /*
 Approach 3: Using Hashmap
   Complexity Analysis
Let l1 be the length of string s1 and l2 be the length of string s2.
•	Time complexity: O(l1+26l1(l2−l1)). The hashmap contains at most 26 keys.
•	Space complexity: O(1). Hashmap contains at most 26 key-value pairs.
         
            
            */
            isPermuteIncluded = CheckPermuteInclusionHM(s1, s2);

            /*
      Approach 4: Using Array [Accepted]           
       Complexity Analysis
     Let l1 be the length of string s1 and l2 be the length of string s2.
     •	Time complexity: O(l1+26l1(l2−l1)).
     •	Space complexity: O(1). s1map and s2map of size 26 is used.

                 */
            isPermuteIncluded = CheckPermuteInclusionArray(s1, s2);

            /*
     Approach 5: Sliding Window [Accepted]: (SW)
      Complexity Analysis
     Let l1 be the length of string s1 and l2 be the length of string s2.
     •	Time complexity: O(l1+26∗(l2−l1)).
     •	Space complexity: O(1). Constant space is used.

                 */
            isPermuteIncluded = CheckPermuteInclusionSW(s1, s2);


            /*

Approach 6: Optimized Sliding Window [Accepted]: (SWOptimal)
Complexity Analysis
Let l1 be the length of string s1 and l2 be the length of string s2.
•	Time complexity: O(l1+(l2−l1)).
•	Space complexity: O(1). Constant space is used.


            */
            isPermuteIncluded = CheckPermuteInclusionSWOptimal(s1, s2);

            return isPermuteIncluded;

        }

        private bool isPermutationFound = false;

        public bool CheckPermuteInclusionNaive(string string1, string string2)
        {
            GeneratePermutations(string1, string2, 0);
            return isPermutationFound;
        }

        public string Swap(string inputString, int index0, int index1)
        {
            if (index0 == index1)
                return inputString;

            string substring1 = inputString.Substring(0, index0);
            string substring2 = inputString.Substring(index0 + 1, index1 - index0 - 1);
            string substring3 = inputString.Substring(index1 + 1);

            return substring1 + inputString[index1] + substring2 + inputString[index0] + substring3;
        }

        private void GeneratePermutations(string string1, string string2, int leftIndex)
        {
            if (leftIndex == string1.Length)
            {
                if (string2.IndexOf(string1) >= 0)
                    isPermutationFound = true;
            }
            else
            {
                for (int i = leftIndex; i < string1.Length; i++)
                {
                    string1 = Swap(string1, leftIndex, i);
                    GeneratePermutations(string1, string2, leftIndex + 1);
                    string1 = Swap(string1, leftIndex, i);
                }
            }
        }

        public bool CheckPermuteInclusionSort(string s1, string s2)
        {
            s1 = SortedString(s1);
            for (int i = 0; i <= s2.Length - s1.Length; i++)
            {
                if (s1.Equals(SortedString(s2.Substring(i, i + s1.Length))))
                    return true;
            }
            return false;
        }

        public string SortedString(String s)
        {
            char[] t = s.ToCharArray();
            Array.Sort(t);
            return new string(t);
        }
        public bool CheckPermuteInclusionHM(string s1, string s2)
        {
            if (s1.Length > s2.Length)
                return false;

            Dictionary<char, int> s1Map = new Dictionary<char, int>();

            for (int i = 0; i < s1.Length; i++)
                s1Map[s1[i]] = s1Map.GetValueOrDefault(s1[i], 0) + 1;

            for (int i = 0; i <= s2.Length - s1.Length; i++)
            {
                Dictionary<char, int> s2Map = new Dictionary<char, int>();
                for (int j = 0; j < s1.Length; j++)
                {
                    s2Map[s2[i + j]] = s2Map.GetValueOrDefault(s2[i + j], 0) + 1;
                }
                if (Matches(s1Map, s2Map))
                    return true;
            }
            return false;
        }

        public bool Matches(Dictionary<char, int> s1Map, Dictionary<char, int> s2Map)
        {
            foreach (var key in s1Map.Keys)
            {
                if (s1Map[key] - s2Map.GetValueOrDefault(key, -1) != 0)
                    return false;
            }
            return true;
        }

        public bool CheckPermuteInclusionArray(string firstString, string secondString)
        {
            if (firstString.Length > secondString.Length)
                return false;

            int[] firstStringMap = new int[26];
            for (int index = 0; index < firstString.Length; index++)
                firstStringMap[firstString[index] - 'a']++;

            for (int index = 0; index <= secondString.Length - firstString.Length; index++)
            {
                int[] secondStringMap = new int[26];
                for (int innerIndex = 0; innerIndex < firstString.Length; innerIndex++)
                {
                    secondStringMap[secondString[index + innerIndex] - 'a']++;
                }
                if (Matches(firstStringMap, secondStringMap))
                    return true;
            }
            return false;
        }

        public bool Matches(int[] firstStringMap, int[] secondStringMap)
        {
            for (int index = 0; index < 26; index++)
            {
                if (firstStringMap[index] != secondStringMap[index])
                    return false;
            }
            return true;
        }
        public bool CheckPermuteInclusionSW(string firstString, string secondString)
        {
            if (firstString.Length > secondString.Length)
                return false;

            int[] firstStringMap = new int[26];
            int[] secondStringMap = new int[26];

            for (int index = 0; index < firstString.Length; index++)
            {
                firstStringMap[firstString[index] - 'a']++;
                secondStringMap[secondString[index] - 'a']++;
            }

            for (int index = 0; index < secondString.Length - firstString.Length; index++)
            {
                if (Matches(firstStringMap, secondStringMap))
                    return true;

                secondStringMap[secondString[index + firstString.Length] - 'a']++;
                secondStringMap[secondString[index] - 'a']--;
            }

            return Matches(firstStringMap, secondStringMap);
        }

        public bool CheckPermuteInclusionSWOptimal(string string1, string string2)
        {
            if (string1.Length > string2.Length)
                return false;

            int[] string1Map = new int[26];
            int[] string2Map = new int[26];

            for (int index = 0; index < string1.Length; index++)
            {
                string1Map[string1[index] - 'a']++;
                string2Map[string2[index] - 'a']++;
            }

            int matchingCount = 0;
            for (int index = 0; index < 26; index++)
            {
                if (string1Map[index] == string2Map[index])
                    matchingCount++;
            }

            for (int index = 0; index < string2.Length - string1.Length; index++)
            {
                int rightCharIndex = string2[index + string1.Length] - 'a';
                int leftCharIndex = string2[index] - 'a';

                if (matchingCount == 26)
                    return true;

                string2Map[rightCharIndex]++;
                if (string2Map[rightCharIndex] == string1Map[rightCharIndex])
                {
                    matchingCount++;
                }
                else if (string2Map[rightCharIndex] == string1Map[rightCharIndex] + 1)
                {
                    matchingCount--;
                }

                string2Map[leftCharIndex]--;
                if (string2Map[leftCharIndex] == string1Map[leftCharIndex])
                {
                    matchingCount++;
                }
                else if (string2Map[leftCharIndex] == string1Map[leftCharIndex] - 1)
                {
                    matchingCount--;
                }
            }

            return matchingCount == 26;
        }

        /* 1220. Count Vowels Permutation
        https://leetcode.com/problems/count-vowels-permutation/description/
         */
        class CountVowelPermutationSol
        {
            /* Approach 1: Dynamic Programming (Bottom-up)	
Complexity Analysis
•	Time complexity: O(N) (N equals the input length n). This is because iterating from 1 to n will take O(N) time. The initializations take constant time. Putting them together gives us O(N) time.
•	Space complexity: O(N). This is because we initialized and used five 1D arrays to store the intermediate results.

             */
            public int BottomUpDP(int n)
            {

                long[] aVowelPermutationCount = new long[n];
                long[] eVowelPermutationCount = new long[n];
                long[] iVowelPermutationCount = new long[n];
                long[] oVowelPermutationCount = new long[n];
                long[] uVowelPermutationCount = new long[n];

                aVowelPermutationCount[0] = 1L;
                eVowelPermutationCount[0] = 1L;
                iVowelPermutationCount[0] = 1L;
                oVowelPermutationCount[0] = 1L;
                uVowelPermutationCount[0] = 1L;

                int MOD = 1000000007;


                for (int i = 1; i < n; i++)
                {
                    aVowelPermutationCount[i] = (eVowelPermutationCount[i - 1] + iVowelPermutationCount[i - 1] + uVowelPermutationCount[i - 1]) % MOD;
                    eVowelPermutationCount[i] = (aVowelPermutationCount[i - 1] + iVowelPermutationCount[i - 1]) % MOD;
                    iVowelPermutationCount[i] = (eVowelPermutationCount[i - 1] + oVowelPermutationCount[i - 1]) % MOD;
                    oVowelPermutationCount[i] = iVowelPermutationCount[i - 1] % MOD;
                    uVowelPermutationCount[i] = (iVowelPermutationCount[i - 1] + oVowelPermutationCount[i - 1]) % MOD;
                }

                long result = 0L;

                result = (aVowelPermutationCount[n - 1] + eVowelPermutationCount[n - 1] + iVowelPermutationCount[n - 1] + oVowelPermutationCount[n - 1] + uVowelPermutationCount[n - 1]) % MOD;


                return (int)result;
            }

            /* Approach 2: Dynamic Programming (Bottom-up) with Optimized Space
            Complexity Analysis
            •	Time complexity: O(N) (N equals the input length n). This is because iterating from 1 to n will take O(N) time. The initializations take constant time. Putting them together gives us O(N) time.
            •	Space complexity: O(1). This is because we don't use any additional data structures to store data.

             */
            public int BottomUpDPWithSpaceOptimal(int n)
            {
                long aCount = 1, eCount = 1, iCount = 1, oCount = 1, uCount = 1;
                int MOD = 1000000007;

                for (int i = 1; i < n; i++)
                {
                    long aCountNew = (eCount + iCount + uCount) % MOD;
                    long eCountNew = (aCount + iCount) % MOD;
                    long iCountNew = (eCount + oCount) % MOD;
                    long oCountNew = (iCount) % MOD;
                    long uCountNew = (iCount + oCount) % MOD;
                    aCount = aCountNew;
                    eCount = eCountNew;
                    iCount = iCountNew;
                    oCount = oCountNew;
                    uCount = uCountNew;
                }
                long result = (aCount + eCount + iCount + oCount + uCount) % MOD;
                return (int)result;
            }
            /* 
            Approach 3: Dynamic Programming (Top-down, Recursion)
•	Time complexity: O(N). This is because there are N recursive calls to each vowel. Therefore, the total number of function calls to vowelPermutationCount is 5⋅N, which leads to time complexity of O(N). Initializations will take O(1) time. Putting them together, this solution takes O(N) time.
•	Space complexity: O(N). This is because O(5⋅N) space is required for memoization. Furthermore, the size of the system stack used by recursion calls will be O(N). Putting them together, this solution uses O(N) space.

             */
            private long[][] memo;
            private int MOD = 1000000007;
            public int TopDownDPRec(int n)
            {
                // each row stands for the length of string
                // each column indicates the vowels
                // specifically, a: 0, e: 1, i: 2, o: 3, u: 4
                memo = new long[n][];
                long result = 0;
                for (int i = 0; i < 5; i++)
                {
                    result = (result + VowelPermutationCountRec(n - 1, i)) % MOD;
                }
                return (int)result;
            }

            public long VowelPermutationCountRec(int i, int vowel)
            {
                if (memo[i][vowel] != 0) return memo[i][vowel];
                if (i == 0)
                {
                    memo[i][vowel] = 1;
                }
                else
                {
                    if (vowel == 0)
                    {
                        memo[i][vowel] = (VowelPermutationCountRec(i - 1, 1) + VowelPermutationCountRec(i - 1, 2) + VowelPermutationCountRec(i - 1, 4)) % MOD;
                    }
                    else if (vowel == 1)
                    {
                        memo[i][vowel] = (VowelPermutationCountRec(i - 1, 0) + VowelPermutationCountRec(i - 1, 2)) % MOD;
                    }
                    else if (vowel == 2)
                    {
                        memo[i][vowel] = (VowelPermutationCountRec(i - 1, 1) + VowelPermutationCountRec(i - 1, 3)) % MOD;
                    }
                    else if (vowel == 3)
                    {
                        memo[i][vowel] = VowelPermutationCountRec(i - 1, 2);
                    }
                    else if (vowel == 4)
                    {
                        memo[i][vowel] = (VowelPermutationCountRec(i - 1, 2) + VowelPermutationCountRec(i - 1, 3)) % MOD;
                    }
                }
                return memo[i][vowel];
            }

        }





















    }


}