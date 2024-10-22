namespace AlgoDSPlay;

public class BitManipProbs
{

    /* 
    2997. Minimum Number of Operations to Make Array XOR Equal to K
    https://leetcode.com/problems/minimum-number-of-operations-to-make-array-xor-equal-to-k/description/
     */
    class MinOperationsToMakeArrayXOREqualToKSol
    {
        /*         Approach: Bit Manipulation
        Complexity Analysis
        Here, N is the number of integers in the array nums.
        •	Time complexity: O(N)
        The XOR operation takes O(1), and we iterate over the N elements in the array nums. The STL function to count the number of set bits takes O(logV) where V is the value of finalXor. Since the values in the array are less than or equal to 106<220, the value of O(logV) will be ~20. Hence, the total time complexity is equal to O(N).
        •	Space complexity: O(1)
        The only space required is the variable finalXor so the space complexity is constant.

         */
        public int UsingBitManip(int[] nums, int k)
        {
            int finalXor = 0;
            foreach (int n in nums)
            {
                finalXor = finalXor ^ n;
            }

            return int.PopCount(finalXor ^ k);
        }
    }

    /* 1310. XOR Queries of a Subarray
    https://leetcode.com/problems/xor-queries-of-a-subarray/description/
     */

    class XorQueriesOfSubarraySol
    {
        /* 
        Approach 1: Iterative Approach
Complexity Analysis
Let n be the number of elements in arr and q be the number of queries.
•	Time Complexity: O(q⋅n)
For each query, we iterate through the range [left, right] in the arr to compute the XOR. Given that q is the number of queries and each query can potentially cover up to n elements, the worst-case time complexity is O(q⋅n). This can be quite slow if both q and n are large.
•	Space Complexity: O(1)
The space complexity is constant because we are using only a few extra variables for calculations and storing results in the output array. The space required does not grow with the input size, except for the result storage, which is proportional to the number of queries. Since the result storage is a requirement of the problem statement, we will not count it towards the space complexity.

         */
        public int[] UsingIterative(int[] arr, int[][] queries)
        {
            int[] result = new int[queries.Length];
            // Process each query
            for (int i = 0; i < queries.Length; i++)
            {
                int xorSum = 0;
                // Calculate XOR for the range [q[0], q[1]]
                for (int j = queries[i][0]; j <= queries[i][1]; j++)
                {
                    xorSum ^= arr[j];
                }
                result[i] = xorSum;
            }
            return result;
        }
        /* Approach 2: Prefix XOR Array 
        Complexity Analysis
Let n be the number of elements in arr and q be the number of queries.
•	Time Complexity: O(n+q)
We first compute the prefix XOR array in O(n) time. Each query is then resolved in constant time O(1) using the prefix XOR array. Thus, the total time complexity is O(n+q).
•	Space Complexity: O(n)
The space complexity is O(n) due to the additional prefix XOR array of size n+1.
        */
        public int[] UsingPrefixXORArray(int[] arr, int[][] queries)
        {
            int n = arr.Length;
            int[] prefixXOR = new int[n + 1];

            // Build prefix XOR array
            for (int i = 0; i < n; i++)
            {
                prefixXOR[i + 1] = prefixXOR[i] ^ arr[i];
            }

            int[] result = new int[queries.Length];
            // Process each query using prefix XOR
            for (int i = 0; i < queries.Length; i++)
            {
                result[i] = prefixXOR[queries[i][1] + 1] ^ prefixXOR[queries[i][0]];
            }
            return result;
        }
        /* Approach 3: In place Prefix XOR 
Complexity Analysis
Let n be the number of elements in arr and q be the number of queries.
•	Time Complexity: O(n+q)
The time complexity is the same as the prefix XOR array approach. We first convert the arr into an in-place prefix XOR array in O(n) time. Each query is then resolved in constant time O(1), leading to an overall time complexity of O(n+q).
•	Space Complexity: O(1)
The space complexity is constant because the in-place prefix XOR modification does not require extra space beyond what is needed to store the results.

        */
        public int[] UsingInplacePrefixXOR(int[] arr, int[][] queries)
        {
            List<int> result = new List<int>();

            // Step 1: Convert arr into an in-place prefix XOR array
            for (int index = 1; index < arr.Length; ++index)
            {
                arr[index] ^= arr[index - 1];
            }

            // Step 2: Resolve each query using the prefix XOR array
            foreach (int[] query in queries)
            {
                if (query[0] > 0)
                {
                    result.Add(arr[query[0] - 1] ^ arr[query[1]]);
                }
                else
                {
                    result.Add(arr[query[1]]);
                }
            }

            return result.ToArray();
        }




    }



    /* 371. Sum of Two Integers
    https://leetcode.com/problems/sum-of-two-integers/description/
     */

    class GetSumSol
    {

        /* Approach 1: Bit Manipulation: Easy and Language-Independent
        Complexity Analysis
•	Time complexity: O(1) because each integer contains 32 bits.
•	Space complexity: O(1) because we don't use any additional data structures.

         */
        public int UsingBitManip(int a, int b)
        {
            int x = Math.Abs(a), y = Math.Abs(b);
            // ensure that abs(a) >= abs(b)
            if (x < y) return UsingBitManip(b, a);

            // abs(a) >= abs(b) --> 
            // a determines the sign
            int sign = a > 0 ? 1 : -1;

            if (a * b >= 0)
            {
                // sum of two positive integers x + y
                // where x > y
                while (y != 0)
                {
                    int answer = x ^ y;
                    int carry = (x & y) << 1;
                    x = answer;
                    y = carry;
                }
            }
            else
            {
                // difference of two positive integers x - y
                // where x > y
                while (y != 0)
                {
                    int answer = x ^ y;
                    int borrow = ((~x) & y) << 1;
                    x = answer;
                    y = borrow;
                }
            }
            return x * sign;
        }
        /* Approach 2: Bit Manipulation: Short Language-Specific Solution
Complexity Analysis
•	Time complexity: O(1).
•	Space complexity: O(1).

         */
        public int UsingBitManip2(int a, int b)
        {
            while (b != 0)
            {
                int answer = a ^ b;
                int carry = (a & b) << 1;
                a = answer;
                b = carry;
            }

            return a;
        }


    }

    /* 898. Bitwise ORs of Subarrays
    https://leetcode.com/problems/bitwise-ors-of-subarrays/description/
    https://algo.monster/liteproblems/898
     */

    class SubarrayBitwiseORsSol
    {
        /* Time and Space Complexity
        The given code aims to find the number of distinct subarray bitwise ORs. To do this, it iterates over the given array and computes the OR of elements from the current element to all previous elements by keeping a record of the previous OR in prev and the current progression in curr.
        Time Complexity
        The time complexity of this algorithm mainly depends on the number of iterations within the double-loop structure.
        •	The outer loop runs exactly n times where n is the number of elements in arr.
        •	The inner loop runs up to i+1 times in the worst case (when curr never equals prev early).
        However, due to the properties of the bitwise OR operation, repetitions are likely to occur much earlier, resulting in earlier breaks from the inner loop. Specifically, the sequence of ORs will eventually stablize into a set of values that does not grow with each additional OR operation. The actual number of unique elements in these OR sequences across all iterations is bounded by a factor much smaller than n^2.
        While it's difficult to put a precise bound on this without specifics about the input distribution, let's denote the average unique sequence length as k (which is considerably smaller than n due to the saturation of OR operations). Therefore, the total number of operations is approximately O(n*k).
        However, it is important to note that k is not guaranteed to be a constant and its relation with n can depend heavily on the input, implying that in the worst case the time complexity could tend towards O(n^2), but in practical scenarios, it is expected to perform significantly better.
        Space Complexity
        The space complexity is due to the set s that is used to store the unique subarray OR results.
        •	In the worst case, each subarray OR could be unique, which means the set could grow to the size of the sum of all subarray counts. As with the time complexity argument, this won't actually occur due to the saturation of bitwise ORs.
        Let m represent the maximum possible unique OR values which can be much less than the total subarray count of roughly n*(n+1)/2. Therefore, the space complexity can be approximated as O(m).
        In conclusion, the time complexity of the code is approximately O(n*k) (with k being influenced by the input nature and much smaller than n) and space complexity is around O(m) for storing the unique OR results set, where m represents the maximum number of unique OR values across all subarrays.
         */
        public int SubarrayBitwiseORs(int[] arr)
        {
            // We use a set to store unique values of bitwise ORs for all subarrays
            HashSet<int> uniqueBitwiseORs = new();

            // We iterate through each element in the array
            for (int i = 0; i < arr.Length; ++i)
            {
                // 'aggregate' will hold the cumulative bitwise OR value up to the current element
                int aggregate = 0;

                // We iterate from the current element down to the start of the array
                for (int j = i; j >= 0; --j)
                {
                    // We calculate the bitwise OR from the current element to the 'jth' element
                    aggregate |= arr[j];

                    // Add the current subarray's bitwise OR to the set
                    uniqueBitwiseORs.Add(aggregate);

                    /* If the current aggregate value is the same as the previous
                       aggregate value, all future aggregates will also be the same
                       due to the properties of bitwise OR, so we break out early. */
                    if (aggregate == (aggregate | arr[i]))
                    {
                        break;
                    }
                }
            }

            // Return the number of unique bitwise ORs found
            return uniqueBitwiseORs.Count;
        }
    }


    /* 1734. Decode XORed Permutation
    https://leetcode.com/problems/decode-xored-permutation/description/
    https://algo.monster/liteproblems/1734
     */

    class DecodeXORedPermutationSol
    {
        /*Time and Space Complexity
Time Complexity
The time complexity of the given algorithm involves iterating over the encoded list and then iterating over a range of numbers from 1 to n to compute the XOR of all elements and the original permutation's elements. Here's the breakdown:
1.	The first for loop runs from 0 to n-1 with a step of 2, resulting in approximately n/2 iterations.
2.	The second for loop runs from 1 to n, inclusive, resulting in n iterations.
3.	The last for loop reverses the encoded array while XORing each element with the next element of the perm list, resulting in n-1 iterations.
Since n, n/2, and n-1 are all linearly proportional to the length of the encoded list, the overall time complexity is O(n).
Space Complexity
The space complexity is determined by:
1.	Variables a and b, which are constant space and thus O(1).
2.	The perm list that stores the result, with a length equal to n, and running n iterations for decoding the permutation.
Since no additional space is used that grows with the input size apart from the perm list, the space complexity is O(n) due to the output data structure.
  */
        public int[] Decode(int[] encoded)
        {
            // Calculate the size of the original permutation array
            int n = encoded.Length + 1;

            // Initialize 'xorEven' to perform XOR on even-indexed elements
            int xorEven = 0;

            // Initialize 'xorAll' to store the XOR of all numbers from 1 to n
            int xorAll = 0;

            // XOR even-indexed elements in the encoded array
            for (int i = 0; i < n - 1; i += 2)
            {
                xorEven ^= encoded[i];
            }

            // XOR all numbers from 1 to n to find the XOR of the entire permutation
            for (int i = 1; i <= n; ++i)
            {
                xorAll ^= i;
            }

            // Initialize the permutation array to be returned
            int[] permutation = new int[n];

            // Find the last element of the permutation by XORing 'xorEven' with 'xorAll', because
            // the XOR of all elements except the last one has been accounted for in 'xorEven'
            permutation[n - 1] = xorEven ^ xorAll;

            // Work backwards to fill in the rest of the permutation array by using the property
            // that encoded[i] = permutation[i] XOR permutation[i + 1]
            for (int i = n - 2; i >= 0; --i)
            {
                permutation[i] = encoded[i] ^ permutation[i + 1];
            }

            // Return the decoded permutation array
            return permutation;
        }
    }











}
