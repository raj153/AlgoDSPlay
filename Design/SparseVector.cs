using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1570. Dot Product of Two Sparse Vectors
    https://leetcode.com/problems/dot-product-of-two-sparse-vectors/description/

    */

    public class SparseVector
    {
        //Approach 1: Non-efficient Array Approach
        /*
        Let n be the length of the input array.
        Time complexity: O(n) for both constructing the sparse vector and calculating the dot product.
        Space complexity: O(1) for constructing the sparse vector as we simply save a reference to the input array and O(1) for calculating the dot product.

        */
        public class SparseVectorNaive
        {
            private int[] array;

            SparseVectorNaive(int[] nums)
            {
                array = nums;
            }

            public int DotProduct(SparseVectorNaive vec)
            {
                int result = 0;

                for (int i = 0; i < array.Length; i++)
                {
                    result += array[i] * vec.array[i];
                }
                return result;
            }

        }

        //Approach 2: Hash Table
        /*
        Let n be the length of the input array and L be the number of non-zero elements.
        Time complexity: O(n) for creating the Hash Map; O(L) for calculating the dot product.
        Space complexity: O(L) for creating the Hash Map, as we only store elements that are non-zero. O(1) for calculating the dot product.
        */
        public class SparseVectorHT
        {
            // Map the index to value for all non-zero values in the vector
            private Dictionary<int, int> mapping;

            SparseVectorHT(int[] nums)
            {
                mapping = new Dictionary<int, int>();
                for (int i = 0; i < nums.Length; i++)
                {
                    if (nums[i] != 0)
                    {
                        mapping.Add(i, nums[1]);
                    }
                }
            }

            public int DotProduct(SparseVectorHT vec)
            {
                int result = 0;

                // iterate through each non-zero element in this sparse vector
                // update the dot product if the corresponding index has a non-zero value in the other vector
                foreach (int i in this.mapping.Keys)
                {
                    if (vec.mapping.ContainsKey(i))
                    {
                        result += this.mapping[i] * vec.mapping[i];
                    }
                }
                return result;
            }

        }

        //Approach 3: Index-Value Pairs
        /*
        Let n be the length of the input array and L and L2 be the number of non-zero elements for the two vectors.
        Time complexity: O(n) for creating the <index, value> pair for non-zero values; O(L+L2) for calculating the dot product.
        Space complexity: O(L) for creating the <index, value> pairs for non-zero values. O(1) for calculating the dot product.

        */
        public class SparseVectorIVPairs
        {
            private List<int[]> pairs;

            SparseVectorIVPairs(int[] nums)
            {
                pairs = new List<int[]>();
                for (int i = 0; i < nums.Length; i++)
                {
                    if (nums[i] != 0)
                    {
                        pairs.Add(new int[] { i, nums[1] });
                    }
                }
            }

            // Return the dotProduct of two sparse vectors
            public int DotProduct(SparseVectorIVPairs vec)
            {
                int result = 0, p = 0, q = 0;
                while (p < pairs.Count && q < vec.pairs.Count)
                {
                    if (pairs[p][0] == vec.pairs[q][0])
                    {
                        result += pairs[p][1] * vec.pairs[q][1];
                        p++;
                        q++;
                    }
                    else if (pairs[p][0] > vec.pairs[q][0])
                    {
                        q++;
                    }
                    else
                    {
                        p++;
                    }
                }


                return result;
            }

        }

    }
    // Your SparseVector object will be instantiated and called as such:
    // SparseVector v1 = new SparseVector(nums1);
    // SparseVector v2 = new SparseVector(nums2);
    // int ans = v1.DotProduct(v2);
}